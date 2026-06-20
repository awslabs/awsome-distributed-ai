#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Collate EP-backend comparison logs into a single markdown table.

Parses the launcher logs from the three benchmarks run on the same world size:

  * NVSHMEM (DeepEP)  -- dispatch/combine "bottleneck bandwidth" in GB/s
  * UCCL    (UCCL-EP) -- dispatch/combine "bottleneck bandwidth" in GB/s
  * NCCL    (nccl-tests alltoall_perf) -- peak busbw in GB/s (transport reference)

The EP benchmarks print dispatch/combine bandwidth lines; NCCL prints the
standard nccl-tests size-sweep table. Both formats vary slightly across versions,
so the regexes below are intentionally permissive. If a value comes back N/A,
print the raw log and adjust the patterns.

Usage:
  python3 collect_results.py \
      --nvshmem-internode nvshmem_internode.log \
      --nvshmem-lowlat   nvshmem_lowlat.log \
      --uccl-internode   uccl_internode.log \
      --uccl-lowlat      uccl_lowlat.log \
      --nccl             nccl_alltoall.log \
      > RESULTS_table.md
"""
import argparse
import re
import sys

# EP benchmarks: capture the largest "dispatch ... <N> GB/s" / "combine ... <N> GB/s"
# value in the log (the reported bottleneck / best bandwidth).
_BW = r"([0-9]+(?:\.[0-9]+)?)\s*GB/s"
DISPATCH_RE = re.compile(r"dispatch[^\n]*?" + _BW, re.IGNORECASE)
COMBINE_RE = re.compile(r"combine[^\n]*?" + _BW, re.IGNORECASE)
# nccl-tests alltoall_perf data rows look like:
#   size count type redop root  time algbw busbw #wrong  time algbw busbw #wrong
# (the second time/algbw/busbw/#wrong group is the in-place result). busbw is
# column index 7 (out-of-place) and 11 (in-place).
NCCL_ROW_RE = re.compile(r"^\s*\d+\s+\d+\s+\w+.*$")
NCCL_BUSBW_COLS = (7, 11)


def _max_bw(text, regex):
    vals = [float(m) for m in regex.findall(text)]
    return max(vals) if vals else None


def parse_ep(path):
    """Return (dispatch_GBps, combine_GBps) bottleneck bandwidths."""
    if not path:
        return None, None
    with open(path) as f:
        text = f.read()
    return _max_bw(text, DISPATCH_RE), _max_bw(text, COMBINE_RE)


def parse_nccl(path):
    """Return peak busbw (GB/s) across the alltoall_perf size sweep."""
    if not path:
        return None
    best = None
    with open(path) as f:
        for line in f:
            if line.lstrip().startswith("#") or not NCCL_ROW_RE.match(line):
                continue
            cols = line.split()
            for idx in NCCL_BUSBW_COLS:
                if idx < len(cols):
                    try:
                        v = float(cols[idx])
                    except ValueError:
                        continue
                    best = v if best is None else max(best, v)
    return best


def fmt(v):
    return f"{v:.1f}" if isinstance(v, float) else "N/A"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nvshmem-internode")
    p.add_argument("--nvshmem-lowlat")
    p.add_argument("--uccl-internode")
    p.add_argument("--uccl-lowlat")
    p.add_argument("--nccl", help="NCCL alltoall_perf log (transport reference)")
    args = p.parse_args()

    nv_i_d, nv_i_c = parse_ep(args.nvshmem_internode)
    nv_l_d, nv_l_c = parse_ep(args.nvshmem_lowlat)
    uc_i_d, uc_i_c = parse_ep(args.uccl_internode)
    uc_l_d, uc_l_c = parse_ep(args.uccl_lowlat)
    nccl_busbw = parse_nccl(args.nccl)

    out = sys.stdout
    out.write("| Backend | Mode | Dispatch (GB/s) | Combine (GB/s) |\n")
    out.write("|---|---|---:|---:|\n")
    out.write(f"| NVSHMEM (DeepEP) | internode | {fmt(nv_i_d)} | {fmt(nv_i_c)} |\n")
    out.write(f"| NVSHMEM (DeepEP) | low-latency | {fmt(nv_l_d)} | {fmt(nv_l_c)} |\n")
    out.write(f"| UCCL (UCCL-EP) | internode | {fmt(uc_i_d)} | {fmt(uc_i_c)} |\n")
    out.write(f"| UCCL (UCCL-EP) | low-latency | {fmt(uc_l_d)} | {fmt(uc_l_c)} |\n")
    out.write("\n")
    out.write("| Reference | Metric | GB/s |\n")
    out.write("|---|---|---:|\n")
    out.write(f"| NCCL all-to-all (transport ceiling) | peak busbw | {fmt(nccl_busbw)} |\n")


if __name__ == "__main__":
    main()
