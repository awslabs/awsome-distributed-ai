import importlib.util
from pathlib import Path
import sys
import unittest


module_path = Path(__file__).with_name("summarize_fair_results.py")
spec = importlib.util.spec_from_file_location("summarize_fair_results", module_path)
assert spec and spec.loader
summary_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = summary_module
spec.loader.exec_module(summary_module)


def fake_result(arm, world_size, run_index, dtype, latency_ms):
    digest = {
        "uccl": "a",
        "deepep-v1-nvshmem": "b",
        "deepep-v2-gin-gda": "c",
    }[arm]
    return {
        "arm": arm,
        "world_size_ranks": world_size,
        "run_index_dimensionless": run_index,
        "dispatch_dtype": dtype,
        "route_hash_sha256": f"route-{world_size}",
        "input_hash_sha256": f"input-{world_size}",
        "correctness": {"status": "PASS"},
        "runtime": {"image_reference": f"example.invalid/{arm}@sha256:{digest * 64}"},
        "latency_ms": {"median": latency_ms},
        "aggregate_input_tokens_per_second": 1000 / latency_ms,
        "effective_logical_gigabytes_per_second_per_rank": 10 / latency_ms,
        "effective_scaleout_logical_gigabytes_per_second_per_rank": 5 / latency_ms,
        "timing_boundary": "common boundary",
        "logical_payload_definition": "common payload",
    }


class SummarizeFairResultsTest(unittest.TestCase):
    def setUp(self):
        arm_latency = {
            "uccl": 1.0,
            "deepep-v1-nvshmem": 1.2,
            "deepep-v2-gin-gda": 0.8,
        }
        self.results = [
            fake_result(
                arm,
                world,
                run,
                dtype,
                arm_latency[arm] * (1 + (run - 2) * 0.01),
            )
            for arm in summary_module.ARMS
            for world in summary_module.WORLD_SIZES
            for run in range(1, 4)
            for dtype in summary_module.DTYPES
        ]

    def test_valid_matrix_and_paired_delta(self):
        summary_module.validate(self.results, 3)
        summary = summary_module.summarize(self.results, 3)
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(len(summary["cells"]), 4)
        comparison = summary["cells"][0]["comparisons"][
            "deepep-v2-gin-gda_vs_uccl"
        ]
        self.assertAlmostEqual(
            comparison["median_paired_latency_reduction_percent"], 20.0
        )
        self.assertTrue(comparison["winner_supported"])

    def test_missing_start_is_rejected(self):
        with self.assertRaises(ValueError):
            summary_module.validate(self.results[:-1], 3)

    def test_route_mismatch_is_rejected(self):
        self.results[0]["route_hash_sha256"] = "different"
        with self.assertRaises(ValueError):
            summary_module.validate(self.results, 3)


if __name__ == "__main__":
    unittest.main()
