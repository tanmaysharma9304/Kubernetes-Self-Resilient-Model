import time


class RecoveryTracker:
    """
    Tracks anomaly start and recovery events in the cluster.

    Calculates:
    - MTTR (Mean Time To Recovery)
    - Availability drop
    """

    def __init__(self, experiment_id: str, baseline_ready_pods: int):

        self.experiment_id = experiment_id
        self.baseline_ready_pods = baseline_ready_pods

        self.active = False
        self.anomaly_start_time = None
        self.min_ready_pods = baseline_ready_pods
        self.restart_start = 0

    def start_anomaly(self, timestamp, ready_pods, restarts):
        """
        Called when anomaly is first detected.
        """

        self.active = True
        self.anomaly_start_time = timestamp
        self.min_ready_pods = ready_pods
        self.restart_start = restarts

        print("\n🚨 Anomaly event started")
        print("Start Time:", timestamp)
        print("Ready Pods:", ready_pods)
        print("Restarts:", restarts)

    def observe(self, timestamp, ready_pods, restarts):
        """
        Monitor cluster during anomaly and detect recovery.
        """

        if not self.active:
            return None

        # Track worst availability
        if ready_pods < self.min_ready_pods:
            self.min_ready_pods = ready_pods

        # Recovery condition:
        # system returned to baseline ready pods
        if ready_pods >= self.baseline_ready_pods:

            mttr = (timestamp - self.anomaly_start_time).total_seconds()

            availability_drop = (
                (self.baseline_ready_pods - self.min_ready_pods)
                / self.baseline_ready_pods
            )

            result = {
                "mttr_seconds": round(mttr, 2),
                "availability_drop": round(availability_drop, 4),
                "restart_increase": restarts - self.restart_start
            }

            print("\n✅ Recovery detected")
            print("MTTR:", result["mttr_seconds"], "seconds")
            print("Availability drop:", result["availability_drop"])

            self.reset()

            return result

        return None

    def reset(self):
        """
        Reset tracker after recovery.
        """

        self.active = False
        self.anomaly_start_time = None
        self.min_ready_pods = self.baseline_ready_pods
        self.restart_start = 0