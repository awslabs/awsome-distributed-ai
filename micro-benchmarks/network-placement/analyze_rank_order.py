# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Summarize a 2.rank-order.sh result directory.

For each collective and message size: the median time and bus bandwidth of each node order over rounds, and the
interleaved/sorted time ratio within each round (both orders run back to back in every round), as its median and
range. A ratio above 1 means the interleaved order, with more ring hops between layer-3 nodes, was slower.
Writes summary.csv into the result directory.

Usage: python3 analyze_rank_order.py <result dir>
"""
import csv
import os
import statistics as st
import sys

d = sys.argv[1]
by = {}
for r in csv.DictReader(open(os.path.join(d, "coll.csv"))):
    rnd, order = r["tag"].split("-", 1)
    by.setdefault((r["coll"], int(r["bytes"])), {}).setdefault(order, {})[rnd] = (float(r["time_us"]), float(r["busbw_GBps"]))
lines = ["coll,bytes,sorted_us,interleaved_us,ratio_median,ratio_min,ratio_max,sorted_busbw_GBps,interleaved_busbw_GBps,rounds"]
for (coll, nbytes), o in sorted(by.items()):
    if "sorted" not in o or "interleaved" not in o:
        continue
    rounds = sorted(set(o["sorted"]) & set(o["interleaved"]))
    if not rounds:
        continue
    ratios = [o["interleaved"][k][0] / o["sorted"][k][0] for k in rounds]
    med = lambda order, i: st.median(v[i] for v in o[order].values())
    lines.append(f"{coll},{nbytes},{med('sorted', 0):.2f},{med('interleaved', 0):.2f},{st.median(ratios):.4f},"
                 f"{min(ratios):.4f},{max(ratios):.4f},{med('sorted', 1):.2f},{med('interleaved', 1):.2f},{len(rounds)}")
open(os.path.join(d, "summary.csv"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
