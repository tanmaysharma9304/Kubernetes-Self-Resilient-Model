"""
╔══════════════════════════════════════════════════════════════╗
║       RESILIENCE BRAIN — FULLY INTEGRATED ML VERSION        ║
║                                                              ║
║  Changes from original code:                                 ║
║  • StandardScaler added (normalizes features before ML)      ║
║  • cpu_x_memory interaction feature added                    ║
║  • Model only retrains when 50+ new rows have been added     ║
║  • predict_anomaly() is now a clean reusable function        ║
║  • scaler.pkl saved/loaded alongside model.pkl               ║
║  • All original DB, tracker, and API logic kept intact       ║
╚══════════════════════════════════════════════════════════════╝
"""

import time
import os
from datetime import datetime, timezone
import uuid
import csv

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import requests

from prometheus_collector import collect_system_metrics
from recovery_tracker import RecoveryTracker
from db import get_conn


# ======================================================
# CONFIG
# ======================================================
BASELINE_READY_PODS = 34
DATA_FILE            = "dataset.csv"
MODEL_FILE           = "model.pkl"
SCALER_FILE          = "scaler.pkl"   # ← NEW: scaler is saved separately
SLEEP_INTERVAL       = 30
MIN_ROWS_TO_TRAIN    = 50             # don't train until we have enough data
RETRAIN_EVERY_N_ROWS = 50            # retrain each time 50 new rows are added


# ======================================================
# CREATE EXPERIMENT (ONCE)
# ======================================================
current_experiment_id = uuid.uuid4()
print(f"🧪 Experiment registered: {current_experiment_id}")

conn = get_conn()
cur  = conn.cursor()

cur.execute("""
INSERT INTO chaos_experiment (
    experiment_id,
    experiment_type,
    service,
    namespace,
    target,
    blast_radius,
    start_time
) VALUES (?, ?, ?, ?, ?, ?, ?)
""", (
    str(current_experiment_id),
    "observational_resilience_run",
    "cartservice",
    "default",
    "deployment/cartservice",
    "single-service",
    datetime.now(timezone.utc)
))

conn.commit()
cur.close()
conn.close()


# ======================================================
# RECOVERY TRACKER
# ======================================================
tracker = RecoveryTracker(
    experiment_id=str(current_experiment_id),
    baseline_ready_pods=BASELINE_READY_PODS
)


# ======================================================
# FAILURE CLASSIFICATION + FIX SUGGESTER
# ======================================================

# Threshold config — tweak these values to tune sensitivity
CPU_CRITICAL    = 0.90   # CPU > 90%  → critical
CPU_HIGH        = 0.75   # CPU > 75%  → high
MEM_CRITICAL    = 0.90   # Memory > 90% → critical
MEM_HIGH        = 0.75   # Memory > 75% → high
RESTART_CRASH   = 5      # Restarts ≥ 5 → crash loop
RESTART_WARNING = 2      # Restarts ≥ 2 → warning


def classify_failure(rows):
    """
    Rule-based classifier — runs AFTER Isolation Forest flags an anomaly.
    Checks CPU, memory, and restart thresholds to identify failure type.

    Priority order (most severe first):
      1. Both CPU + Memory critical            → resource_exhaustion
      2. Memory ≥ 90% (critical)               → memory_collapse_critical
      3. Memory ≥ 75% + restarts               → memory_collapse
      4. CPU ≥ 90% (critical)                  → cpu_saturation_critical
      5. CPU ≥ 75%                             → cpu_saturation
      6. Restarts ≥ 5                          → crash_loop
      7. Restarts ≥ 2                          → crash_loop_warning
      8. Anything else flagged by ML           → transient_anomaly

    Returns: (failure_type, avg_cpu, avg_mem, total_restarts)
    """
    avg_cpu  = sum(r[0] for r in rows) / len(rows)
    avg_mem  = sum(r[1] for r in rows) / len(rows)
    restarts = sum(r[2] for r in rows)

    if avg_cpu >= CPU_CRITICAL and avg_mem >= MEM_CRITICAL:
        failure = "resource_exhaustion"

    elif avg_mem >= MEM_CRITICAL:
        failure = "memory_collapse_critical"

    elif avg_mem >= MEM_HIGH and restarts >= RESTART_WARNING:
        failure = "memory_collapse"

    elif avg_cpu >= CPU_CRITICAL:
        failure = "cpu_saturation_critical"

    elif avg_cpu >= CPU_HIGH:
        failure = "cpu_saturation"

    elif restarts >= RESTART_CRASH:
        failure = "crash_loop"

    elif restarts >= RESTART_WARNING:
        failure = "crash_loop_warning"

    else:
        failure = "transient_anomaly"

    return failure, avg_cpu, avg_mem, restarts


# Fix suggestions per failure type
# Each entry: (short_fix, full human-readable explanation)
FIX_SUGGESTIONS = {
    "resource_exhaustion": (
        "scale_up_resources",
        "🔴 CRITICAL: Both CPU ({cpu:.0%}) and Memory ({mem:.0%}) are maxed out.\n"
        "   → Immediately increase both CPU and memory limits in your deployment YAML.\n"
        "   → Consider horizontal scaling (add more pod replicas).\n"
        "   → Check for runaway processes or memory leaks."
    ),
    "memory_collapse_critical": (
        "increase_memory_limit_urgent",
        "🔴 CRITICAL: Memory usage is {mem:.0%} — container is about to OOMKill.\n"
        "   → Urgently increase memory limit: e.g. set resources.limits.memory to 2x current value.\n"
        "   → Add a memory request so Kubernetes schedules on a node with enough RAM.\n"
        "   → Profile the app for memory leaks (check heap dumps / GC logs)."
    ),
    "memory_collapse": (
        "increase_memory_limit",
        "🟠 HIGH: Memory at {mem:.0%} with {restarts} restart(s) — heading toward OOMKill.\n"
        "   → Increase memory limit in deployment YAML (e.g. from 512Mi to 1Gi).\n"
        "   → Set memory request = 70–80%% of limit to avoid over-scheduling.\n"
        "   → Monitor with: kubectl top pod -n default"
    ),
    "cpu_saturation_critical": (
        "increase_cpu_limit_urgent",
        "🔴 CRITICAL: CPU usage is {cpu:.0%} — service is severely throttled.\n"
        "   → Immediately increase CPU limit in deployment YAML.\n"
        "   → Add more replicas to spread the load (kubectl scale deployment).\n"
        "   → Check for infinite loops or blocking synchronous calls in code."
    ),
    "cpu_saturation": (
        "increase_cpu_limit",
        "🟠 HIGH: CPU usage is {cpu:.0%} — requests are being throttled.\n"
        "   → Increase CPU limit (e.g. from 500m to 1000m in resources.limits.cpu).\n"
        "   → Consider a Horizontal Pod Autoscaler (HPA) to auto-scale on CPU.\n"
        "   → Review recent deployments — a new release may have introduced a hot loop."
    ),
    "crash_loop": (
        "add_probes_and_investigate_logs",
        "🔴 CRITICAL: {restarts} pod restarts detected — container is crash-looping.\n"
        "   → Check logs immediately: kubectl logs <pod> --previous\n"
        "   → Add/fix liveness probe so Kubernetes can detect and restart unhealthy pods.\n"
        "   → Add readiness probe to stop traffic reaching unready pods.\n"
        "   → Check exit codes: OOMKilled (137) = memory, Error (1) = app crash."
    ),
    "crash_loop_warning": (
        "add_probes",
        "🟡 WARNING: {restarts} restart(s) detected — early crash-loop signs.\n"
        "   → Add liveness + readiness probes to your deployment spec.\n"
        "   → Check recent config changes or secret/env-var issues.\n"
        "   → Watch pod events: kubectl describe pod <pod-name>"
    ),
    "transient_anomaly": (
        None,
        "🟡 INFO: ML model flagged unusual metrics but no clear threshold was breached.\n"
        "   → Monitor for the next 2–3 intervals to see if it persists.\n"
        "   → Could be a brief traffic spike or GC pause — no immediate action needed."
    ),
}


def recommend_fix(failure_class, avg_cpu=0.0, avg_mem=0.0, restarts=0):
    """
    Returns a formatted fix recommendation string for the given failure type.
    Injects actual metric values into the message for context.
    """
    entry = FIX_SUGGESTIONS.get(failure_class, FIX_SUGGESTIONS["transient_anomaly"])
    short_fix, explanation = entry

    # Fill in actual metric values in the message
    message = explanation.format(
        cpu=avg_cpu,
        mem=avg_mem,
        restarts=int(restarts)
    )
    return short_fix, message


# ======================================================
# DATASET HELPERS
# ======================================================
def append_dataset(rows):
    """
    Appends new metric rows to dataset.csv.
    Each row: [cpu, memory, restarts]
    Also stores the derived cpu_x_memory feature.
    """
    new_file = not os.path.exists(DATA_FILE)
    with open(DATA_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            # ← cpu_x_memory column added vs original
            writer.writerow(["cpu", "memory", "restarts", "cpu_x_memory"])
        for r in rows:
            cpu, mem, rst = r[0], r[1], r[2]
            writer.writerow([cpu, mem, rst, cpu * mem])


def count_dataset_rows():
    """Returns number of data rows in dataset.csv (excluding header)."""
    if not os.path.exists(DATA_FILE):
        return 0
    with open(DATA_FILE, "r") as f:
        return sum(1 for _ in f) - 1  # subtract header


# ======================================================
# ML — TRAIN MODEL
# ======================================================
def train_model():
    """
    Trains an Isolation Forest on the current dataset.

    Key improvements vs original:
      • StandardScaler normalizes features so no one feature dominates
      • cpu_x_memory interaction feature included
      • Scaler saved to disk so it can be used for prediction later

    Returns: (model, scaler) or (None, None) if not enough data
    """
    if not os.path.exists(DATA_FILE):
        return None, None

    df = pd.read_csv(DATA_FILE)

    if len(df) < MIN_ROWS_TO_TRAIN:
        print(f"   ⏳ Not enough data yet ({len(df)}/{MIN_ROWS_TO_TRAIN} rows)")
        return None, None

    feature_cols = ["cpu", "memory", "restarts", "cpu_x_memory"]
    X = df[feature_cols]

    # ── Normalize features ──────────────────────────────────────────
    # Without this, "restarts" (0–8) could overpower "cpu" (0.0–1.0).
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Train Isolation Forest ───────────────────────────────────────
    # contamination=0.05 → model expects ~5% of data to be anomalies.
    # Tune this up if you see too many false alarms, down for more sensitivity.
    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42
    )
    model.fit(X_scaled)

    # ── Save both model and scaler ───────────────────────────────────
    joblib.dump(model,  MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    print(f"   ✅ Model trained on {len(df)} rows and saved.")
    return model, scaler


# ======================================================
# ML — DETECT ANOMALY
# ======================================================
def detect_anomaly(model, scaler, rows):
    """
    Runs the trained model on fresh metric rows.

    Parameters:
        model  : trained IsolationForest
        scaler : fitted StandardScaler (MUST be the same one used at training)
        rows   : list of [cpu, memory, restarts]

    Returns:
        True if any row is flagged as an anomaly, False otherwise
    """
    records = []
    for r in rows:
        cpu, mem, rst = r[0], r[1], r[2]
        records.append([cpu, mem, rst, cpu * mem])   # ← add interaction feature

    df      = pd.DataFrame(records, columns=["cpu", "memory", "restarts", "cpu_x_memory"])
    scaled  = scaler.transform(df)                   # ← scale using saved scaler
    preds   = model.predict(scaled)

    return (-1 in preds)  # -1 means anomaly in sklearn


# ======================================================
# LOAD EXISTING MODEL (if available from previous run)
# ======================================================
model  = None
scaler = None

if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
    model  = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    print("📂 Loaded existing model and scaler from disk.")

rows_at_last_train = count_dataset_rows()


# ======================================================
# MAIN LOOP
# ======================================================
print("🧠 Resilience Brain started")

while True:
    try:
        # ------------------------------------------------
        # Collect metrics
        # ------------------------------------------------
        metrics   = collect_system_metrics()
        timestamp = datetime.now(timezone.utc)

        rows = [[
            metrics["cpu_usage"],
            metrics["memory_usage"],
            metrics["pod_restarts"]
        ]]

        print("📊 Metrics:", metrics)

        # ------------------------------------------------
        # Store system metrics in DB
        # ------------------------------------------------
        conn = get_conn()
        cur  = conn.cursor()

        cur.execute("""
        INSERT INTO system_metrics (
            experiment_id,
            timestamp,
            cpu_usage,
            memory_usage,
            ready_pods,
            total_pods,
            pod_restarts
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            str(current_experiment_id),
            timestamp,
            metrics["cpu_usage"],
            metrics["memory_usage"],
            metrics["ready_pods"],
            metrics["total_pods"],
            metrics["pod_restarts"]
        ))

        conn.commit()
        cur.close()
        conn.close()

        # ------------------------------------------------
        # Append to dataset
        # ------------------------------------------------
        append_dataset(rows)

        # ------------------------------------------------
        # Retrain model periodically
        # (every RETRAIN_EVERY_N_ROWS new data points)
        # ------------------------------------------------
        current_row_count = count_dataset_rows()
        rows_since_train  = current_row_count - rows_at_last_train

        if rows_since_train >= RETRAIN_EVERY_N_ROWS or model is None:
            print(f"🔄 Retraining model ({current_row_count} total rows)...")
            model, scaler = train_model()
            rows_at_last_train = current_row_count

        # ------------------------------------------------
        # Anomaly detection
        # ------------------------------------------------
        is_anomaly = (
            model is not None
            and scaler is not None
            and detect_anomaly(model, scaler, rows)   # ← scaler now passed in
        )

        # ------------------------------------------------
        # ANOMALY DETECTED (only triggers once per event)
        # ------------------------------------------------
        if is_anomaly and not tracker.active:
            failure_class, avg_cpu, avg_mem, restarts = classify_failure(rows)
            fix_hint, fix_message = recommend_fix(failure_class, avg_cpu, avg_mem, restarts)

            print(f"\n⚠️  ANOMALY DETECTED → {failure_class}")
            print(f"   CPU: {avg_cpu:.0%}  |  Memory: {avg_mem:.0%}  |  Restarts: {int(restarts)}")
            print(f"\n💡 Suggested Fix:\n{fix_message}\n")

            # Optional API call
            try:
                requests.post(
                    "http://127.0.0.1:8000/anomalies/",
                    params={
                        "experiment_id": str(current_experiment_id),
                        "service":       "cartservice",
                        "failure_class": failure_class,
                        "fix_hint":      fix_hint or "none"
                    },
                    timeout=2
                )
            except Exception:
                pass

            tracker.start_anomaly(
                timestamp,
                metrics["ready_pods"],
                metrics["pod_restarts"]
            )

        # ------------------------------------------------
        # Recovery observation
        # ------------------------------------------------
        recovery_result = tracker.observe(
            timestamp,
            metrics["ready_pods"],
            metrics["pod_restarts"]
        )

        # ------------------------------------------------
        # On recovery complete
        # ------------------------------------------------
        if recovery_result:
            print("✅ Recovery Metrics:", recovery_result)

            conn = get_conn()
            cur  = conn.cursor()

            cur.execute("""
            UPDATE chaos_experiment
            SET end_time = %s
            WHERE experiment_id = %s
            """, (
                timestamp,
                str(current_experiment_id)
            ))

            conn.commit()
            cur.close()
            conn.close()

            print("📊 Experiment Result:", {
                "experiment_id":    str(current_experiment_id),
                "mttr":             recovery_result["mttr_seconds"],
                "availability_drop": recovery_result["availability_drop"]
            })

        print("📸 Snapshot collected:", timestamp)
        time.sleep(SLEEP_INTERVAL)

    except Exception as e:
        print("❌ Error:", e)
        time.sleep(10)
