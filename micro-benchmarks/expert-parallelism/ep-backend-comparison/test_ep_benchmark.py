import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


MODULE = None
if torch is not None:
    module_path = Path(__file__).with_name("ep_benchmark.py")
    spec = importlib.util.spec_from_file_location("ep_benchmark", module_path)
    assert spec and spec.loader
    MODULE = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = MODULE
    spec.loader.exec_module(MODULE)


@unittest.skipIf(torch is None, "PyTorch is required for benchmark helper tests")
class EpBenchmarkTest(unittest.TestCase):
    def test_route_is_balanced_and_unique(self):
        routes = [
            MODULE.make_route(rank, 128, 256, 8, 20260824, torch.device("cpu"))
            for rank in range(32)
        ]
        route = torch.cat(routes)
        histogram = torch.bincount(route.flatten(), minlength=256)
        self.assertEqual(histogram.min().item(), 128)
        self.assertEqual(histogram.max().item(), 128)
        for row in route:
            self.assertEqual(torch.unique(row).numel(), 8)

    def test_common_payload_formula(self):
        route = MODULE.make_route(0, 128, 256, 8, 20260824, torch.device("cpu"))
        bf16_all, bf16_remote, selections = MODULE.logical_payload_bytes_per_rank(
            route, 0, 16, 8, 7168, "bf16", 256
        )
        fp8_all, fp8_remote, _ = MODULE.logical_payload_bytes_per_rank(
            route, 0, 16, 8, 7168, "fp8", 256
        )
        self.assertEqual(selections, 1024)
        self.assertEqual(bf16_all, 1024 * (7168 * 2 + 7168 * 2))
        self.assertEqual(fp8_all, 1024 * (7168 + 56 * 4 + 7168 * 2))
        self.assertGreater(bf16_remote, 0)
        self.assertGreater(fp8_remote, 0)
        self.assertLess(bf16_remote, bf16_all)
        self.assertLess(fp8_remote, fp8_all)

    def test_profile_shapes_are_fixed(self):
        self.assertEqual(MODULE.WORKLOAD_PROFILES["decode"].tokens_per_rank, 128)
        self.assertEqual(MODULE.WORKLOAD_PROFILES["decode"].api_mode, "low-latency")
        self.assertEqual(MODULE.WORKLOAD_PROFILES["prefill"].tokens_per_rank, 4096)
        self.assertEqual(MODULE.WORKLOAD_PROFILES["prefill"].api_mode, "normal")

    def test_prefill_payload_scales_with_tokens(self):
        decode_route = MODULE.make_route(0, 128, 256, 8, 20260824, torch.device("cpu"))
        prefill_route = MODULE.make_route(
            0, 4096, 256, 8, 20260824, torch.device("cpu")
        )
        decode_bytes, _, _ = MODULE.logical_payload_bytes_per_rank(
            decode_route, 0, 16, 8, 7168, "bf16", 256
        )
        prefill_bytes, _, _ = MODULE.logical_payload_bytes_per_rank(
            prefill_route, 0, 16, 8, 7168, "bf16", 256
        )
        self.assertEqual(prefill_bytes, decode_bytes * 32)

    def test_percentile_interpolates(self):
        self.assertEqual(MODULE.percentile([1.0, 2.0, 3.0], 0.5), 2.0)
        self.assertAlmostEqual(MODULE.percentile([1.0, 2.0], 0.95), 1.95)

    def test_deepep_v2_build_lib_requires_one_extension_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "build" / "lib.linux-x86_64-cpython-312" / "deep_ep"
            package.mkdir(parents=True)
            (package / "_C.cpython-312-x86_64-linux-gnu.so").touch()
            self.assertEqual(MODULE.deepep_v2_build_lib(root), package.parent)

            second = root / "build" / "lib.second" / "deep_ep"
            second.mkdir(parents=True)
            (second / "_C.so").touch()
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                MODULE.deepep_v2_build_lib(root)

    def test_received_fp8_accepts_noncontiguous_scales(self):
        adapter = object.__new__(MODULE.BackendAdapter)
        adapter.arm = "uccl"
        adapter.profile = MODULE.WORKLOAD_PROFILES["decode"]
        adapter.hidden = 256
        observed = {}

        def cast_back(fp8, scales):
            observed["fp8_shape"] = tuple(fp8.shape)
            observed["scales_shape"] = tuple(scales.shape)
            return torch.zeros((fp8.shape[0], 256), dtype=torch.bfloat16)

        adapter._cast_back = cast_back
        fp8 = torch.zeros((2, 3, 256), dtype=torch.float8_e4m3fn)
        scales = torch.arange(12, dtype=torch.float32).reshape(2, 6).t()
        self.assertFalse(scales.is_contiguous())

        received = adapter.received_as_bf16((fp8, scales), "fp8")

        self.assertEqual(observed["fp8_shape"], (6, 256))
        self.assertEqual(observed["scales_shape"], (6, 2))
        self.assertEqual(tuple(received.shape), (2, 3, 256))

    def test_identity_expert_output_applies_local_gates(self):
        adapter = object.__new__(MODULE.BackendAdapter)
        adapter.arm = "uccl"
        adapter.hidden = 2
        adapter.profile = MODULE.WORKLOAD_PROFILES["prefill"]
        state = MODULE.DispatchState(
            recv_x=torch.tensor([[2.0, 4.0]], dtype=torch.bfloat16),
            recv_topk_idx=torch.tensor([[0, -1, 1]]),
            recv_topk_weights=torch.tensor([[0.25, 0.5, 0.125]]),
            handle=None,
        )
        output = adapter.identity_expert_output(state, "bf16")
        torch.testing.assert_close(
            output,
            torch.tensor([[0.75, 1.5]], dtype=torch.bfloat16),
        )


if __name__ == "__main__":
    unittest.main()
