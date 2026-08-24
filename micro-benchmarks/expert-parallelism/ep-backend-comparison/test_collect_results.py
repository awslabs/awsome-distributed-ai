#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Unit tests for the EP comparison log collector."""

import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("collect_results.py")
SPEC = importlib.util.spec_from_file_location("collect_results", MODULE_PATH)
COLLECT_RESULTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECT_RESULTS)


class ParseDeepEPV2Test(unittest.TestCase):
    def test_parses_rank_zero_selected_cases(self):
        log = """
> Testing with do_handle_copy=1, use_fp8_dispatch=1, num_bias=0 ...
   * EP:   1/16 | dispatch: 34 GB/s (SO), 201 GB/s (SU), 866.000 us, 1 bytes
   * EP:   0/16 | dispatch: 35 GB/s (SO), 203 GB/s (SU), 864.688 us, 2 bytes
   @ EP:   0/16 | combine: 36 GB/s (SO), 208 GB/s (SU), 1614.000 us, 3 bytes
   - EP:   0/16 | expanded dispatch: 99 GB/s (SO), 999 GB/s (SU), 1.000 us, 4 bytes
> Testing with do_handle_copy=1, use_fp8_dispatch=0, num_bias=0 ...
   * EP:   0/16 | dispatch: 39 GB/s (SO), 225 GB/s (SU), 1501.000 us, 5 bytes
   @ EP:   0/16 | combine: 36 GB/s (SO), 208 GB/s (SU), 1615.000 us, 6 bytes
"""
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as handle:
            handle.write(log)
            handle.flush()
            result = COLLECT_RESULTS.parse_deepep_v2(handle.name)

        self.assertEqual(result[("FP8", "dispatch")], (35.0, 203.0, 864.688))
        self.assertEqual(result[("FP8", "combine")], (36.0, 208.0, 1614.0))
        self.assertEqual(result[("BF16", "dispatch")], (39.0, 225.0, 1501.0))
        self.assertEqual(result[("BF16", "combine")], (36.0, 208.0, 1615.0))
        self.assertEqual(len(result), 4)

    def test_retains_result_without_dtype_marker(self):
        log = "* EP: 0/32 | dispatch: 10 GB/s (SO), 19 GB/s (SU), 275.659 us, 1 bytes\n"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as handle:
            handle.write(log)
            handle.flush()
            result = COLLECT_RESULTS.parse_deepep_v2(handle.name)

        self.assertEqual(
            result[("unspecified", "dispatch")],
            (10.0, 19.0, 275.659),
        )


if __name__ == "__main__":
    unittest.main()
