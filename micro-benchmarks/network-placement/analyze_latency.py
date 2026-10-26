# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Summarize a 1.efa-latency-matrix.sh result directory.

For each message size: the within-host floor per pod, the mean of each
pair's kept rounds grouped by how many network-node layers the pair shares,
and whether every pair under one layer-3 node was faster than every pair
that differs at layer 3. Writes summary.txt and pair-means-<size>.csv into
the result directory.

Usage: python3 analyze_latency.py <result dir>
"""

import csv
import os
import statistics as st
import sys

NAMES = {
    3: "same layer-3 node",
    2: "different layer-3 node, same layer 2",
    1: "different layer-2 node, same layer 1",
    0: "different layer-1 node",
}


def read_rows(path):
    """Rows of a CSV file as dictionaries."""
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def measured(rows):
    """Rows that hold a latency value."""
    return [r for r in rows if r["usec_per_xfer"] not in ("NA", "")]


def floor_lines(floor, size):
    """Summary line for the within-host floor at one message size."""
    per_pod = {}
    for r in floor:
        if r["size_bytes"] == size:
            per_pod.setdefault(r["pod"], []).append(float(r["usec_per_xfer"]))
    if not per_pod:
        return []
    means = [st.mean(v) for v in per_pod.values()]
    return [
        f"  within one host (floor): {len(means)} pods, "
        f"mean {st.mean(means):.2f} "
        f"({min(means):.2f} to {max(means):.2f} by pod)"
    ]


def depth_lines(by_depth):
    """Summary lines per number of shared layers, and the layer-3 check."""
    lines = []
    for depth in sorted(by_depth, reverse=True):
        v = by_depth[depth]
        lines.append(
            f"  {NAMES.get(depth, depth)}: {len(v)} pairs, "
            f"mean {st.mean(v):.2f} ({min(v):.2f} to {max(v):.2f})"
        )
    if 3 in by_depth and 2 in by_depth:
        gap = min(by_depth[2]) - max(by_depth[3])
        near, far = st.mean(by_depth[3]), st.mean(by_depth[2])
        lines.append(
            "  every same-layer-3 pair faster than every other pair: "
            f"{gap > 0} (gap between bands {gap:+.2f} us); "
            f"crossing layer 3 adds {far - near:.2f} us "
            f"({100 * (far / near - 1):.0f}%)"
        )
    return lines


def write_pair_means(d, size, pairs):
    """Write each pair's mean latency at one message size."""
    path = os.path.join(d, f"pair-means-{size}.csv")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("pod_a,pod_b,shared_layers,mean_usec,rounds\n")
        for (a, b, depth), v in sorted(pairs.items()):
            fh.write(f"{a},{b},{depth},{st.mean(v):.3f},{len(v)}\n")


def summarize(d):
    """Return the summary lines for result directory d."""
    floor = measured(read_rows(os.path.join(d, "floor.csv")))
    rows = [
        r
        for r in measured(read_rows(os.path.join(d, "latency.csv")))
        if r["discarded"] == "no"
    ]
    out = []
    for size in sorted({r["size_bytes"] for r in rows}, key=int):
        pairs = {}
        for r in rows:
            if r["size_bytes"] == size:
                key = (r["pod_a"], r["pod_b"], int(r["shared_layers"]))
                pairs.setdefault(key, []).append(float(r["usec_per_xfer"]))
        by_depth = {}
        for (_a, _b, depth), v in pairs.items():
            by_depth.setdefault(depth, []).append(st.mean(v))
        out.append(f"{size}-byte messages, one-way latency (us)")
        out += floor_lines(floor, size) + depth_lines(by_depth) + [""]
        write_pair_means(d, size, pairs)
    return out


def main():
    """Print the summary and write summary.txt."""
    d = sys.argv[1]
    out = summarize(d)
    print("\n".join(out))
    with open(os.path.join(d, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
