#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Collate EP-backend comparison logs into a single markdown table.

Parses the launcher logs from the three benchmarks run at the same world size:

  * NVSHMEM (DeepEP)  -- dispatch/combine bandwidth
  * UCCL    (UCCL-EP) -- dispatch/combine bandwidth
  * NCCL    (nccl-tests alltoall_perf) -- busbw (transport reference)

Output formats parsed (DeepEP and UCCL print identically -- UCCL's bench is
derived from DeepEP's tests):

  internode (test_internode.py):
    [tuning] Best dispatch (BF16): ... BW: <RDMA> GB/s (RDMA), <NVL> GB/s (NVL)
    [tuning] Best combine: ...      BW: <RDMA> GB/s (RDMA), <NVL> GB/s (NVL)
    -> we report the RDMA leg (the cross-node bottleneck). Reporting the NVL leg
       here would be wrong: it is the intra-node ~hundreds-GB/s number.

  low-latency (test_low_latency.py):
    [rank N] Dispatch bandwidth: <BW> GB/s, avg_t=... | Combine bandwidth: <BW> GB/s, avg_t=...
    -> single bandwidth per dispatch/combine (no RDMA/NVL split).

  NCCL alltoall_perf: the standard size-sweep table; busbw is column 7
    (out-of-place) / 11 (in-place). We report busbw at the row whose size is
    closest to the EP per-rank dispatch payload (num_tokens * hidden * 2 bytes,
    default ~56 MiB) AND the asymptotic peak, because the peak overstates the
    transport ceiling relative to EP's smaller messages.

Both EP formats vary slightly across versions; if a value comes back N/A, print
the raw log and adjust the regexes.

Usage:
  python3 collect_results.py \
      --nvshmem-internode nvshmem_internode.log \
      --nvshmem-lowlat   nvshmem_lowlat.log \
      --uccl-internode   uccl_internode.log \
      --uccl-lowlat      uccl_lowlat.log \
      --nccl             nccl_alltoall.log \
      --nccl-target-bytes 58720256
"""
import argparse
import re
import sys

_BW = r"([0-9]+(?:\.[0-9]+)?)"
# internode: pull the RDMA leg from the "Best dispatch/combine" summary lines.
BEST_DISPATCH_RDMA = re.compile(r"Best dispatch[^\n]*?" + _BW + r"\s*GB/s\s*\(RDMA\)", re.I)
BEST_COMBINE_RDMA = re.compile(r"Best combine[^\n]*?" + _BW + r"\s*GB/s\s*\(RDMA\)", re.I)
# low-latency: "Dispatch bandwidth: X GB/s" / "Combine bandwidth: Y GB/s".
# Case-SENSITIVE on purpose: the aggregate line "Dispatch + combine bandwidth: Z"
# uses a lowercase "combine" and must NOT match the per-leg "Combine bandwidth:".
LL_DISPATCH = re.compile(r"Dispatch bandwidth:\s*" + _BW + r"\s*GB/s")
LL_COMBINE = re.compile(r"Combine bandwidth:\s*" + _BW + r"\s*GB/s")
# nccl-tests data row: size count type redop root  time algbw busbw #wrong ...
NCCL_ROW_RE = re.compile(r"^\s*\d+\s+\d+\s+\w+")
NCCL_BUSBW_COLS = (7, 11)


def _last(text, regex):
    m = regex.findall(text)
    return float(m[-1]) if m else None


def _max(text, regex):
    m = regex.findall(text)
    return max(float(x) for x in m) if m else None


def parse_internode(path):
    """(dispatch_rdma, combine_rdma) GB/s from the Best-config summary lines."""
    if not path:
        return None, None
    with open(path) as f:
        text = f.read()
    # The "Best" line is printed once per config after tuning; take the last.
    return _last(text, BEST_DISPATCH_RDMA), _last(text, BEST_COMBINE_RDMA)


def parse_lowlat(path):
    """(dispatch, combine) GB/s. Per-rank lines; take the max across ranks."""
    if not path:
        return None, None
    with open(path) as f:
        text = f.read()
    return _max(text, LL_DISPATCH), _max(text, LL_COMBINE)


def parse_nccl(path, target_bytes):
    """Return (busbw_at_target, size_at_target, busbw_peak) GB/s."""
    if not path:
        return None, None, None
    peak = None
    best_at = None       # (abs_size_delta, size, busbw)
    with open(path) as f:
        for line in f:
            if line.lstrip().startswith("#") or not NCCL_ROW_RE.match(line):
                continue
            cols = line.split()
            try:
                size = int(cols[0])
            except ValueError:
                continue
            for idx in NCCL_BUSBW_COLS:
                if idx >= len(cols):
                    continue
                try:
                    bw = float(cols[idx])
                except ValueError:
                    continue
                peak = bw if peak is None else max(peak, bw)
                delta = abs(size - target_bytes)
                if best_at is None or delta < best_at[0]:
                    best_at = (delta, size, bw)
    if best_at is None:
        return None, None, peak
    return best_at[2], best_at[1], peak


def fmt(v):
    return f"{v:.1f}" if isinstance(v, float) else "N/A"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nvshmem-internode")
    p.add_argument("--nvshmem-lowlat")
    p.add_argument("--uccl-internode")
    p.add_argument("--uccl-lowlat")
    p.add_argument("--nccl", help="NCCL alltoall_perf log (transport reference)")
    p.add_argument("--nccl-target-bytes", type=int, default=4096 * 7168 * 2,
                   help="EP per-rank dispatch payload to read busbw at (default num_tokens*hidden*2)")
    args = p.parse_args()

    nv_i_d, nv_i_c = parse_internode(args.nvshmem_internode)
    nv_l_d, nv_l_c = parse_lowlat(args.nvshmem_lowlat)
    uc_i_d, uc_i_c = parse_internode(args.uccl_internode)
    uc_l_d, uc_l_c = parse_lowlat(args.uccl_lowlat)
    nccl_at, nccl_size, nccl_peak = parse_nccl(args.nccl, args.nccl_target_bytes)

    out = sys.stdout
    out.write("| Backend | Mode | Dispatch (GB/s) | Combine (GB/s) |\n")
    out.write("|---|---|---:|---:|\n")
    out.write(f"| NVSHMEM (DeepEP) | internode (RDMA) | {fmt(nv_i_d)} | {fmt(nv_i_c)} |\n")
    out.write(f"| NVSHMEM (DeepEP) | low-latency | {fmt(nv_l_d)} | {fmt(nv_l_c)} |\n")
    out.write(f"| UCCL (UCCL-EP) | internode (RDMA) | {fmt(uc_i_d)} | {fmt(uc_i_c)} |\n")
    out.write(f"| UCCL (UCCL-EP) | low-latency | {fmt(uc_l_d)} | {fmt(uc_l_c)} |\n")
    out.write("\n")
    sz_mib = f"{nccl_size / 2**20:.0f} MiB" if isinstance(nccl_size, int) else "N/A"
    out.write("| Reference (NCCL all-to-all, transport ceiling) | Metric | GB/s |\n")
    out.write("|---|---|---:|\n")
    out.write(f"| busbw at EP payload (~{sz_mib}) | matched-size | {fmt(nccl_at)} |\n")
    out.write(f"| busbw peak (asymptotic, overstates ceiling) | peak | {fmt(nccl_peak)} |\n")


if __name__ == "__main__":
    main()
