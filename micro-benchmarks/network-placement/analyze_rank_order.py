# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Summarize a 2.rank-order.sh result directory.

For each collective and message size: the median time and bus bandwidth of
each node order over rounds, and the interleaved/sorted time ratio within
each round (both orders run back to back in every round), as its median and
range. A ratio above 1 means the interleaved order, with more ring hops
between layer-3 nodes, was slower. Writes summary.csv into the result
directory.

Usage: python3 analyze_rank_order.py <result dir>
"""

import csv
import os
import statistics as st
import sys

HEADER = (
    "coll,bytes,sorted_us,interleaved_us,ratio_median,ratio_min,ratio_max,"
    "sorted_busbw_GBps,interleaved_busbw_GBps,rounds"
)


def median_of(by_round, index):
    """Median over rounds of element `index` of each round's tuple.

    Index 0 is the time in microseconds, index 1 the bus bandwidth.
    """
    return st.median(v[index] for v in by_round.values())


def read_runs(d):
    """(collective, bytes) -> order -> round -> (time_us, busbw_GBps)."""
    by = {}
    path = os.path.join(d, "coll.csv")
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            rnd, order = r["tag"].split("-", 1)
            key = (r["coll"], int(r["bytes"]))
            by.setdefault(key, {}).setdefault(order, {})[rnd] = (
                float(r["time_us"]),
                float(r["busbw_GBps"]),
            )
    return by


def summarize(d):
    """Return the summary CSV lines for result directory d."""
    lines = [HEADER]
    for (coll, nbytes), orders in sorted(read_runs(d).items()):
        if "sorted" not in orders or "interleaved" not in orders:
            continue
        srt, ilv = orders["sorted"], orders["interleaved"]
        rounds = sorted(set(srt) & set(ilv))
        if not rounds:
            continue
        ratios = [ilv[k][0] / srt[k][0] for k in rounds]
        lines.append(
            f"{coll},{nbytes},{median_of(srt, 0):.2f},"
            f"{median_of(ilv, 0):.2f},{st.median(ratios):.4f},"
            f"{min(ratios):.4f},{max(ratios):.4f},"
            f"{median_of(srt, 1):.2f},{median_of(ilv, 1):.2f},{len(rounds)}"
        )
    return lines


def main():
    """Print the summary and write summary.csv."""
    d = sys.argv[1]
    lines = summarize(d)
    with open(os.path.join(d, "summary.csv"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
