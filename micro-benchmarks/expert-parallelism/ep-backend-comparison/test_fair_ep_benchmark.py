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
    module_path = Path(__file__).with_name("fair_ep_benchmark.py")
    spec = importlib.util.spec_from_file_location("fair_ep_benchmark", module_path)
    assert spec and spec.loader
    MODULE = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = MODULE
    spec.loader.exec_module(MODULE)


@unittest.skipIf(torch is None, "PyTorch is required for benchmark helper tests")
class FairEpBenchmarkTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
