import importlib.util
from pathlib import Path
import sys
import tempfile
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
    logical_bytes = 22_249_472 if dtype == "fp8" else 29_360_128
    scaleout_bytes = logical_bytes * (world_size - 8) // world_size
    tolerance = 9e-4 if dtype == "fp8" else 1e-5
    global_input_tokens = 128 * world_size
    result = {
        "benchmark": "common-boundary-dispatch-combine",
        "arm": arm,
        "world_size_ranks": world_size,
        "nodes": world_size // 8,
        "gpus_per_node": 8,
        "run_index_dimensionless": run_index,
        "dispatch_dtype": dtype,
        "tokens_per_rank": 128,
        "global_input_tokens": global_input_tokens,
        "hidden_dimensions": 7168,
        "experts": 256,
        "top_k_dimensionless": 8,
        "warmup_iterations": 20,
        "measured_iterations": 100,
        "route_hash_sha256": ("d" if world_size == 16 else "e") * 64,
        "input_hash_sha256": ("f" if world_size == 16 else "0") * 64,
        "global_valid_expert_selections": global_input_tokens * 8,
        "avg_logical_payload_bytes_per_rank": logical_bytes,
        "avg_scaleout_logical_payload_bytes_per_rank": scaleout_bytes,
        "correctness": {
            "status": "PASS",
            "tolerance_dimensionless": tolerance,
            "normalized_diff_dimensionless": 0.0,
        },
        "runtime": {
            "image_reference": f"example.invalid/{arm}@sha256:{digest * 64}",
            "gpu": "NVIDIA B200",
            "torch_version": "2.13.0+cu130",
            "cuda_version": "13.0",
            "nccl_version": [2, 29, 7],
        },
        "latency_ms": {"median": latency_ms},
        "aggregate_input_tokens_per_second": global_input_tokens / (latency_ms / 1e3),
        "effective_logical_gigabytes_per_second_per_rank": logical_bytes
        / (latency_ms / 1e3)
        / 1e9,
        "effective_scaleout_logical_gigabytes_per_second_per_rank": scaleout_bytes
        / (latency_ms / 1e3)
        / 1e9,
        "timing_boundary": summary_module.TIMING_BOUNDARY,
        "logical_payload_definition": summary_module.LOGICAL_PAYLOAD_DEFINITION,
    }
    return result


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
        comparison = summary["cells"][0]["comparisons"]["deepep-v2-gin-gda_vs_uccl"]
        self.assertAlmostEqual(
            comparison["median_paired_latency_reduction_percent"], 20.0
        )
        self.assertTrue(comparison["direction_supported"])

    def test_missing_start_is_rejected(self):
        with self.assertRaises(ValueError):
            summary_module.validate(self.results[:-1], 3)

    def test_route_mismatch_is_rejected(self):
        self.results[0]["route_hash_sha256"] = "different"
        with self.assertRaises(ValueError):
            summary_module.validate(self.results, 3)

    def test_runtime_mismatch_is_rejected(self):
        self.results[0]["runtime"]["nccl_version"] = [9, 9, 9]
        with self.assertRaises(ValueError):
            summary_module.validate(self.results, 3)

    def test_logical_payload_mismatch_is_rejected(self):
        self.results[0]["avg_logical_payload_bytes_per_rank"] += 1
        with self.assertRaises(ValueError):
            summary_module.validate(self.results, 3)

    def test_derived_metric_mismatch_is_rejected(self):
        self.results[0]["aggregate_input_tokens_per_second"] += 1
        with self.assertRaises(ValueError):
            summary_module.validate(self.results, 3)

    def test_provenance_image_mismatch_is_rejected(self):
        provenance = {
            "campaign_id": "test-campaign",
            "created_at_utc": "2026-08-24T00:00:00Z",
            "region": "ap-south-1",
            "cluster": "test-cluster",
            "git_commit": "a" * 40,
            "images": {
                arm: next(
                    result["runtime"]["image_reference"]
                    for result in self.results
                    if result["arm"] == arm
                )
                for arm in summary_module.ARMS
            },
            "comparison": {
                "tokens_per_rank": 128,
                "hidden_dimensions": 7168,
                "experts": 256,
                "top_k_dimensionless": 8,
                "warmup_iterations": 20,
                "measured_iterations": 100,
                "independent_starts": 3,
            },
        }
        provenance["images"]["uccl"] = "example.invalid/changed@sha256:" + "9" * 64
        with self.assertRaises(ValueError):
            summary_module.validate_provenance(provenance, self.results, 3)

    def test_load_results_accepts_native_diagnostic_after_json(self):
        result = fake_result("deepep-v2-gin-gda", 32, 1, "fp8", 0.9)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rank-zero.log"
            path.write_text(
                "ADAI_FAIR_RESULT "
                + summary_module.json.dumps(result)
                + "Elastic buffer uses 3 channels per SM\n"
            )
            self.assertEqual(summary_module.load_results(Path(directory)), [result])


if __name__ == "__main__":
    unittest.main()
