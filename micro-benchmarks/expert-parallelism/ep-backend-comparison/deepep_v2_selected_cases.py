#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Run the matched FP8/BF16 DeepEP V2 direct-EP cases.

This wrapper targets the pinned synthetic DeepEP V2 revision documented in
RESULTS-b200.md. It limits the upstream elastic benchmark to the 2 dispatch
dtypes used by the comparison while retaining correctness checks.
"""

import argparse
import sys
from argparse import Namespace
from pathlib import Path

import torch


def _load_upstream():
    candidates = (
        Path("/opt/amazon/deepep-v2/tests/elastic"),
        Path("/opt/amazon/deepep/tests/elastic"),
    )
    for candidate in candidates:
        if (candidate / "test_ep.py").is_file():
            sys.path.insert(0, str(candidate))
            import test_ep  # pylint: disable=import-outside-toplevel

            return test_ep
    locations = ", ".join(str(path) for path in candidates)
    raise RuntimeError(f"DeepEP V2 test_ep.py not found under: {locations}")


UPSTREAM = _load_upstream()


def selected_modes():
    """Yield FP8 and BF16 dispatch with the same remaining mode controls."""
    # handle copy, expert alignment, FP8 dispatch, bias count, previous event,
    # async compute stream, allocate on communication stream
    yield (1, 128, 1, 0, 0, 0, 0)
    yield (1, 128, 0, 0, 0, 0, 0)


UPSTREAM.enumerate_ep_modes = selected_modes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--num-tokens", type=int, required=True)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    selected = Namespace(
        num_processes=args.num_processes,
        num_sms=0,
        num_qps=0,
        num_allocated_qps=0,
        num_gpu_timeout_secs=180,
        num_cpu_timeout_secs=180,
        sl_idx=0,
        num_tokens=args.num_tokens,
        hidden=args.hidden,
        num_topk=args.num_topk,
        num_experts=args.num_experts,
        do_cpu_sync=1,
        allow_hybrid_mode=1,
        allow_multiple_reduction=1,
        prefer_overlap_with_compute=0,
        deterministic=False,
        seed=args.seed,
        skip_check=False,
        skip_perf_test=False,
        do_pressure_test=False,
        pressure_iterations=0,
        reuse_elastic_buffer=False,
        test_first_only=False,
        unbalanced_ratio=1.0,
        precise_unbalanced_ratio=False,
        masked_ratio=0.0,
        dump_profile_traces="",
        ignore_local_traffic=True,
    )
    torch.multiprocessing.spawn(
        UPSTREAM.test_loop,
        args=(args.num_processes, selected),
        nprocs=args.num_processes,
    )


if __name__ == "__main__":
    main()
