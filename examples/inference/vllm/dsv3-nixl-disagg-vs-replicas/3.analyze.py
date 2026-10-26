# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Summarize a 2.benchmark.sh result directory.

Per workload and concurrency: each arm's mean over rounds of output
tokens/s, median time to first token (TTFT), median and 99th-percentile
time per output token (TPOT) and 99th-percentile gap between tokens (ITL),
and the disaggregated/replicas output ratio per round as its median and
range. Writes summary.csv into the result directory.

Usage: python3 3.analyze.py <result dir>
"""

import csv
import os
import statistics as st
import sys

HEADER = (
    "workload,concurrency,rounds,disagg_tps,replicas_tps,ratio_median,"
    "ratio_min,ratio_max,disagg_ttft_ms,replicas_ttft_ms,disagg_tpot_ms,"
    "replicas_tpot_ms,disagg_tpot_p99_ms,replicas_tpot_p99_ms,"
    "disagg_itl_p99_ms,replicas_itl_p99_ms"
)


def read_rows(d):
    """Completed rows keyed by (workload, concurrency, arm) and round."""
    by = {}
    path = os.path.join(d, "serving.csv")
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["output_tps"] in ("", "NA"):
                continue
            key = (int(r["workload"]), int(r["concurrency"]), r["arm"])
            by.setdefault(key, {})[r["round"]] = r
    return by


def mean_of(rows, rounds, field):
    """Mean of one field over the given rounds."""
    return st.mean(float(rows[k][field]) for k in rounds)


def summarize(d):
    """Return the summary CSV lines for result directory d."""
    by = read_rows(d)
    lines = [HEADER]
    for workload, conc in sorted({(k[0], k[1]) for k in by}):
        a = by.get((workload, conc, "disagg"), {})
        b = by.get((workload, conc, "replicas"), {})
        rounds = sorted(set(a) & set(b))
        if not rounds:
            continue
        ratios = [
            float(a[k]["output_tps"]) / float(b[k]["output_tps"])
            for k in rounds
        ]
        cells = [workload, conc, len(rounds)]
        cells += [f"{mean_of(x, rounds, 'output_tps'):.0f}" for x in (a, b)]
        cells += [
            f"{st.median(ratios):.3f}",
            f"{min(ratios):.3f}",
            f"{max(ratios):.3f}",
        ]
        for field, fmt in (
            ("ttft_p50_ms", ".0f"),
            ("tpot_p50_ms", ".1f"),
            ("tpot_p99_ms", ".1f"),
            ("itl_p99_ms", ".1f"),
        ):
            cells += [f"{mean_of(x, rounds, field):{fmt}}" for x in (a, b)]
        lines.append(",".join(str(c) for c in cells))
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
