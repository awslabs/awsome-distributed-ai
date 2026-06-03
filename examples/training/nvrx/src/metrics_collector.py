"""
Metrics Collector
Collects and reports MTBF and Goodput metrics
"""

import os
import time
import json
import logging
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Collects training metrics for MTBF and Goodput calculation
    """

    def __init__(self, rank, world_size, output_dir="/checkpoints"):
        self.rank = rank
        self.world_size = world_size
        self.is_main = rank == 0
        self.output_dir = output_dir

        self.training_start_time = None
        self.training_end_time = None
        self.total_training_time = 0

        self.failure_events = []
        self.recovery_events = []
        self.total_downtime = 0

        self.steps_completed = 0
        self.total_samples_processed = 0
        self.throughput_history = []

        self.checkpoint_save_times = []
        self.checkpoint_load_times = []
        self.checkpoint_sizes = []
        self.checkpoint_performance = None

        # Run completion tracking
        self.max_steps = None
        self.termination_reason = None  # "max_steps_reached" or "time_limit"

    def start_training(self):
        """Mark training start"""
        self.training_start_time = time.time()
        if self.is_main:
            logger.info("Metrics: Training started")

    def end_training(self, success=True, error=None):
        """Mark training end"""
        self.training_end_time = time.time()
        self.total_training_time = self.training_end_time - self.training_start_time

        if self.is_main:
            status = "SUCCESS" if success else "FAILED"
            logger.info(f"Metrics: Training ended - {status}")
            if error:
                logger.info(f"Metrics: Error - {error}")

    def log_step(self, step, loss, elapsed_time):
        """Log training step"""
        self.steps_completed = step

        throughput = step / elapsed_time if elapsed_time > 0 else 0
        self.throughput_history.append(
            {
                "step": step,
                "timestamp": time.time(),
                "throughput": throughput,
                "loss": loss,
            }
        )

    def log_failure(self, failure_type, timestamp=None):
        """Log a failure event"""
        if timestamp is None:
            timestamp = time.time()

        self.failure_events.append(
            {"timestamp": timestamp, "type": failure_type, "step": self.steps_completed}
        )

        if self.is_main:
            logger.warning(f"Metrics: Failure logged - {failure_type}")

    def log_recovery(self, recovery_time, timestamp=None):
        """Log a recovery event"""
        if timestamp is None:
            timestamp = time.time()

        self.recovery_events.append(
            {
                "timestamp": timestamp,
                "recovery_time": recovery_time,
                "step": self.steps_completed,
            }
        )

        self.total_downtime += recovery_time

        if self.is_main:
            logger.info(f"Metrics: Recovery logged - {recovery_time:.2f}s")

    def log_checkpoint_save(self, duration):
        """Log checkpoint save time"""
        self.checkpoint_save_times.append(duration)

    def log_checkpoint_load(self, duration):
        """Log checkpoint load time"""
        self.checkpoint_load_times.append(duration)

    def log_checkpoint_size(self, size_mb):
        """Log checkpoint size in MB"""
        self.checkpoint_sizes.append(size_mb)

    def set_checkpoint_performance(
        self,
        mode,
        checkpoint_count,
        checkpoint_interval,
        total_checkpoint_time,
        checkpoint_times,
        checkpoint_sizes,
        total_wall_time,
        total_steps,
    ):
        """Store computed checkpoint performance metrics for JSON output."""
        avg_ckpt_time = (
            total_checkpoint_time / checkpoint_count if checkpoint_count else 0
        )
        overhead_pct = (
            (total_checkpoint_time / total_wall_time) * 100 if total_wall_time else 0
        )
        efficiency_pct = 100 - overhead_pct
        avg_size = (
            sum(checkpoint_sizes) / len(checkpoint_sizes) if checkpoint_sizes else 0
        )
        bandwidth = avg_size / avg_ckpt_time if avg_ckpt_time > 0 else 0

        self.checkpoint_performance = {
            "mode": mode,
            "checkpoint_count": checkpoint_count,
            "checkpoint_interval": checkpoint_interval,
            "total_checkpoint_time_seconds": round(total_checkpoint_time, 3),
            "avg_checkpoint_time_seconds": round(avg_ckpt_time, 3),
            "min_checkpoint_time_seconds": round(min(checkpoint_times), 3)
            if checkpoint_times
            else 0,
            "max_checkpoint_time_seconds": round(max(checkpoint_times), 3)
            if checkpoint_times
            else 0,
            "checkpoint_times": [round(t, 3) for t in checkpoint_times],
            "avg_checkpoint_size_mb": round(avg_size, 1),
            "checkpoint_sizes_mb": [round(s, 1) for s in checkpoint_sizes],
            "write_bandwidth_mb_s": round(bandwidth, 1),
            "checkpoint_overhead_pct": round(overhead_pct, 2),
            "training_efficiency_pct": round(efficiency_pct, 2),
            "total_wall_time_seconds": round(total_wall_time, 1),
            "total_steps": total_steps,
        }

    def calculate_mtbf(self):
        """
        Calculate Mean Time Between Failures
        MTBF = Total Uptime / Number of Failures
        """
        if len(self.failure_events) == 0:
            return float("inf")

        total_uptime = self.total_training_time - self.total_downtime
        mtbf = total_uptime / len(self.failure_events)

        return mtbf

    def calculate_goodput(self):
        """
        Calculate Goodput
        Goodput = Useful Work / Total Time
        """
        if self.total_training_time == 0:
            return 0.0

        useful_time = self.total_training_time - self.total_downtime
        goodput = useful_time / self.total_training_time

        return goodput * 100

    def calculate_average_throughput(self):
        """Calculate average throughput"""
        if not self.throughput_history:
            return 0.0

        throughputs = [t["throughput"] for t in self.throughput_history]
        return sum(throughputs) / len(throughputs)

    def set_run_completion(self, max_steps, termination_reason):
        """Record how the run terminated for fair cross-experiment comparison.

        Args:
            max_steps: The target step count (--max_steps).
            termination_reason: "max_steps_reached" or "time_limit".
        """
        self.max_steps = max_steps
        self.termination_reason = termination_reason

    def get_summary(self):
        """Get metrics summary"""
        summary = {
            "training_duration_seconds": self.total_training_time,
            "steps_completed": self.steps_completed,
            "max_steps": self.max_steps,
            "termination_reason": self.termination_reason,
            "num_failures": len(self.failure_events),
            "num_recoveries": len(self.recovery_events),
            "total_downtime_seconds": self.total_downtime,
            "mtbf_seconds": self.calculate_mtbf(),
            "goodput_percentage": self.calculate_goodput(),
            "average_throughput": self.calculate_average_throughput(),
            "failure_events": self.failure_events,
            "recovery_events": self.recovery_events,
            "avg_checkpoint_save_time": sum(self.checkpoint_save_times)
            / len(self.checkpoint_save_times)
            if self.checkpoint_save_times
            else 0,
            "avg_checkpoint_load_time": sum(self.checkpoint_load_times)
            / len(self.checkpoint_load_times)
            if self.checkpoint_load_times
            else 0,
        }

        # Include detailed checkpoint performance if set by the training script
        if self.checkpoint_performance:
            summary["checkpoint_performance"] = self.checkpoint_performance

        return summary

    def print_summary(self):
        """Print metrics summary"""
        if not self.is_main:
            return

        summary = self.get_summary()

        print("\n" + "=" * 80)
        print("TRAINING METRICS SUMMARY")
        print("=" * 80)
        print(
            f"Total Training Time: {summary['training_duration_seconds']:.2f} seconds ({summary['training_duration_seconds'] / 60:.2f} minutes)"
        )
        # Run completion status
        if summary.get("max_steps") is not None:
            reason = summary.get("termination_reason", "unknown")
            reason_label = (
                "completed" if reason == "max_steps_reached" else "time-limited"
            )
            print(
                f"Steps: {summary['steps_completed']}/{summary['max_steps']} ({reason_label})"
            )
            print(f"Termination: {reason}")
        else:
            print(f"Steps Completed: {summary['steps_completed']}")
        print(f"Number of Failures: {summary['num_failures']}")
        print(f"Number of Recoveries: {summary['num_recoveries']}")
        print(
            f"Total Downtime: {summary['total_downtime_seconds']:.2f} seconds ({summary['total_downtime_seconds'] / 60:.2f} minutes)"
        )
        print("")
        print("RESILIENCY METRICS:")
        print(
            f"  MTBF (Mean Time Between Failures): {summary['mtbf_seconds']:.2f} seconds ({summary['mtbf_seconds'] / 60:.2f} minutes)"
        )
        print(f"  Goodput: {summary['goodput_percentage']:.2f}%")
        print(f"  Average Throughput: {summary['average_throughput']:.2f} steps/sec")
        print("")
        print("CHECKPOINT METRICS:")
        print(
            f"  Avg Checkpoint Save Time: {summary['avg_checkpoint_save_time']:.2f} seconds"
        )
        print(
            f"  Avg Checkpoint Load Time: {summary['avg_checkpoint_load_time']:.2f} seconds"
        )
        print("=" * 80)

        try:
            metrics_path = os.path.join(self.output_dir, "metrics_summary.json")
            with open(metrics_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"Metrics saved to {metrics_path}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
