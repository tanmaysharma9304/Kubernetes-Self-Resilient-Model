"""
prometheus_collector.py — Stable Version

Collects cluster-level metrics from Prometheus installed via kube-prometheus-stack.

Metrics collected:
- Node CPU usage
- Node memory usage
- Ready pods
- Total pods
- Pod restarts

Works with:
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090
"""

import requests

PROMETHEUS_URL = "http://localhost:9090"
TIMEOUT = 5


def query_prometheus(promql: str) -> float:
    """
    Executes a PromQL query and returns the first value.
    """

    try:
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": promql},
            timeout=TIMEOUT
        )

        response.raise_for_status()

        data = response.json()
        results = data.get("data", {}).get("result", [])

        if not results:
            return 0.0

        return float(results[0]["value"][1])

    except Exception as e:
        print("Prometheus query failed:", e)
        return 0.0


def collect_system_metrics() -> dict:
    """
    Collect metrics from Kubernetes cluster via Prometheus.
    """

    # CPU usage (cluster average)
    cpu_usage = query_prometheus("""
    1 - avg(rate(node_cpu_seconds_total{mode="idle"}[2m]))
    """)

    # Memory usage
    memory_usage = query_prometheus("""
    1 - (
        sum(node_memory_MemAvailable_bytes)
        /
        sum(node_memory_MemTotal_bytes)
    )
    """)

    # Ready pods
    ready_pods = int(query_prometheus("""
    sum(kube_pod_status_ready{condition="true"})
    """))

    # Total pods
    total_pods = int(query_prometheus("""
    count(kube_pod_info)
    """))

    # Pod restart count
    restarts = int(query_prometheus("""
    sum(kube_pod_container_status_restarts_total)
    """))

    # Safety clamp
    cpu_usage = max(0.0, min(1.0, cpu_usage))
    memory_usage = max(0.0, min(1.0, memory_usage))

    return {
        "cpu_usage": round(cpu_usage, 4),
        "memory_usage": round(memory_usage, 4),
        "ready_pods": ready_pods,
        "total_pods": total_pods,
        "pod_restarts": restarts
    }


if __name__ == "__main__":

    print("Connecting to Prometheus:", PROMETHEUS_URL)
    print()

    metrics = collect_system_metrics()

    print("Collected Metrics")
    print("------------------")

    print("CPU Usage:", f"{metrics['cpu_usage']:.1%}")
    print("Memory Usage:", f"{metrics['memory_usage']:.1%}")
    print("Ready Pods:", metrics["ready_pods"])
    print("Total Pods:", metrics["total_pods"])
    print("Restarts:", metrics["pod_restarts"])

    if metrics["cpu_usage"] == 0.0 and metrics["memory_usage"] == 0.0:
        print()
        print("WARNING: All metrics zero.")
        print("Make sure Prometheus port-forward is running:")
        print("kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090")
    else:
        print()
        print("Prometheus metrics collected successfully.")