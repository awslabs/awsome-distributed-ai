#!/usr/bin/env python3
"""Run one DeepEP v2 ElasticBuffer dispatch/combine correctness case."""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace

import torch


def selected_mode():
    # handle copy, expert alignment, BF16 dispatch, bias count, previous event,
    # async compute stream, allocate on communication stream
    yield (1, 128, 0, 0, 0, 0, 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", type=int, required=True)
    parser.add_argument("--num-processes", type=int, default=8)
    args = parser.parse_args()
    ranks = args.domains * args.num_processes

    for source in (
        "/opt/amazon/deepep-v2/tests/elastic",
        "/opt/amazon/deepep-v2-stock/tests/elastic",
    ):
        if source not in sys.path:
            sys.path.insert(0, source)
    import test_ep as upstream  # noqa: PLC0415

    upstream.enumerate_ep_modes = selected_mode
    selected = Namespace(
        num_processes=args.num_processes,
        num_sms=8,
        num_qps=0,
        num_allocated_qps=0,
        num_gpu_timeout_secs=300,
        num_cpu_timeout_secs=300,
        sl_idx=0,
        num_tokens=16,
        hidden=1024,
        num_topk=8,
        num_experts=ranks,
        do_cpu_sync=1,
        allow_hybrid_mode=1,
        allow_multiple_reduction=1,
        prefer_overlap_with_compute=0,
        deterministic=False,
        seed=20260823,
        skip_check=False,
        skip_perf_test=True,
        do_pressure_test=False,
        pressure_iterations=0,
        reuse_elastic_buffer=False,
        test_first_only=True,
        unbalanced_ratio=1.0,
        precise_unbalanced_ratio=False,
        masked_ratio=0.0,
        dump_profile_traces="",
        ignore_local_traffic=True,
    )
    torch.multiprocessing.spawn(
        upstream.test_loop,
        args=(args.num_processes, selected),
        nprocs=args.num_processes,
    )
    print(
        f"ADAI_PR5_CORRECTNESS_PASS domains={args.domains}_domains "
        f"ranks={ranks}_ranks dispatches=1_dispatch combines=1_combine",
        flush=True,
    )


if __name__ == "__main__":
    main()
