# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Summarize a 3.benchmark-layouts.sh result directory.

Per layout and concurrency: the median over rounds of time per output
token (TPOT), output throughput and time to first token for the near and
the far second instance, and the far/near TPOT ratio within each round
(both placements run back to back in every round), as its median and
range. Each two-instance layout is also compared with TP 8 on one
instance. Writes summary.csv into the result directory.

Usage: python3 4.analyze.py <result dir>
"""

import csv
import os
import statistics as st
import sys

HEADER = (
    "layout,concurrency,rounds,near_tpot_ms,far_tpot_ms,far_over_near,"
    "far_over_near_min,far_over_near_max,near_tps,far_tps,near_ttft_ms,"
    "far_ttft_ms,one_instance_tpot_ms,one_instance_tps,near_tps_over_one"
)


def read_rows(d):
    """Benchmark rows that completed, keyed by (layout, conc, arm, round)."""
    rows = {}
    path = os.path.join(d, "serving.csv")
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["tpot_p50_ms"] in ("", "NA"):
                continue
            key = (r["layout"], int(r["concurrency"]), r["arm"], r["round"])
            rows[key] = r
    return rows


def med(rows, field):
    """Median of one field over a list of rows."""
    return st.median(float(r[field]) for r in rows)


def layout_line(layout, conc, near, far, one):
    """One summary CSV line for a two-instance layout at one concurrency."""
    rounds = sorted(set(near) & set(far))
    ratios = [
        float(far[k]["tpot_p50_ms"]) / float(near[k]["tpot_p50_ms"])
        for k in rounds
    ]
    n, f = list(near.values()), list(far.values())
    cells = [
        layout,
        conc,
        len(rounds),
        f"{med(n, 'tpot_p50_ms'):.2f}",
        f"{med(f, 'tpot_p50_ms'):.2f}",
        f"{st.median(ratios):.4f}",
        f"{min(ratios):.4f}",
        f"{max(ratios):.4f}",
        f"{med(n, 'output_tps'):.1f}",
        f"{med(f, 'output_tps'):.1f}",
        f"{med(n, 'ttft_p50_ms'):.0f}",
        f"{med(f, 'ttft_p50_ms'):.0f}",
    ]
    if one:
        one_tps = float(one["output_tps"])
        cells += [
            f"{float(one['tpot_p50_ms']):.2f}",
            f"{one_tps:.1f}",
            f"{med(n, 'output_tps') / one_tps:.3f}",
        ]
    else:
        cells += ["", "", ""]
    return ",".join(str(c) for c in cells)


def summarize(d):
    """Return the summary CSV lines for result directory d."""
    rows = read_rows(d)
    one = {k[1]: v for k, v in rows.items() if k[0] == "tp8"}
    lines = [HEADER]
    for layout, conc in sorted({(k[0], k[1]) for k in rows if k[0] != "tp8"}):
        by_arm = {"near": {}, "far": {}}
        for (lay, c, arm, rnd), r in rows.items():
            if lay == layout and c == conc and arm in by_arm:
                by_arm[arm][rnd] = r
        if set(by_arm["near"]) & set(by_arm["far"]):
            lines.append(
                layout_line(
                    layout, conc, by_arm["near"], by_arm["far"], one.get(conc)
                )
            )
    for conc, r in sorted(one.items()):
        lines.append(
            f"tp8-one-instance,{conc},1,{float(r['tpot_p50_ms']):.2f},,,,,"
            f"{float(r['output_tps']):.1f},,{float(r['ttft_p50_ms']):.0f},,,,"
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
