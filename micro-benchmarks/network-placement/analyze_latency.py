# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Summarize a 1.efa-latency-matrix.sh result directory.

For each message size: the within-host floor per pod, the mean of each pair's kept rounds grouped by how many
network-node layers the pair shares, and whether every pair under one layer-3 node was faster than every pair that
differs at layer 3. Writes summary.txt and pair-means-<size>.csv into the result directory.

Usage: python3 analyze_latency.py <result dir>
"""
import csv
import os
import statistics as st
import sys

d = sys.argv[1]
floor = [r for r in csv.DictReader(open(os.path.join(d, "floor.csv"))) if r["usec_per_xfer"] not in ("NA", "")]
rows = [r for r in csv.DictReader(open(os.path.join(d, "latency.csv")))
        if r["discarded"] == "no" and r["usec_per_xfer"] not in ("NA", "")]
out = []


def say(s=""):
    out.append(s)
    print(s)


for size in sorted({r["size_bytes"] for r in rows}, key=int):
    per_pod = {}
    for r in floor:
        if r["size_bytes"] == size:
            per_pod.setdefault(r["pod"], []).append(float(r["usec_per_xfer"]))
    pairs = {}
    for r in rows:
        if r["size_bytes"] == size:
            pairs.setdefault((r["pod_a"], r["pod_b"], int(r["shared_layers"])), []).append(float(r["usec_per_xfer"]))
    say(f"{size}-byte messages, one-way latency (us)")
    if per_pod:
        fm = {k: st.mean(v) for k, v in per_pod.items()}
        say(f"  within one host (floor): {len(fm)} pods, mean {st.mean(fm.values()):.2f} "
            f"({min(fm.values()):.2f} to {max(fm.values()):.2f} by pod)")
    by_depth = {}
    for (a, b, depth), v in pairs.items():
        by_depth.setdefault(depth, []).append(st.mean(v))
    names = {3: "same layer-3 node", 2: "different layer-3 node, same layer 2",
             1: "different layer-2 node, same layer 1", 0: "different layer-1 node"}
    for depth in sorted(by_depth, reverse=True):
        v = by_depth[depth]
        say(f"  {names.get(depth, depth)}: {len(v)} pairs, mean {st.mean(v):.2f} ({min(v):.2f} to {max(v):.2f})")
    if 3 in by_depth and 2 in by_depth:
        gap = min(by_depth[2]) - max(by_depth[3])
        say(f"  every same-layer-3 pair faster than every other pair: {gap > 0} (gap between bands {gap:+.2f} us); "
            f"crossing layer 3 adds {st.mean(by_depth[2]) - st.mean(by_depth[3]):.2f} us "
            f"({100 * (st.mean(by_depth[2]) / st.mean(by_depth[3]) - 1):.0f}%)")
    say()
    with open(os.path.join(d, f"pair-means-{size}.csv"), "w") as fh:
        fh.write("pod_a,pod_b,shared_layers,mean_usec,rounds\n")
        for (a, b, depth), v in sorted(pairs.items()):
            fh.write(f"{a},{b},{depth},{st.mean(v):.3f},{len(v)}\n")
open(os.path.join(d, "summary.txt"), "w").write("\n".join(out) + "\n")
