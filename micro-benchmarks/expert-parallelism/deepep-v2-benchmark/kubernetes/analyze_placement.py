# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Compare DeepEP V2 dispatch and combine times between placements.

Reads logs of test_ep.py runs (kubectl logs of each MPIJob's launcher pod,
or any log with the test's per-rank lines), one file per run, and names
each run after its file without a round suffix (near-r1.log -> near), so
rounds of the same placement are averaged. Lines look like
  * EP:   3/16 | dispatch: 81 GB/s (SO), 263 GB/s (SU), 107 us, ... bytes
For each placement and kind (dispatch, combine, and their cached, expanded
or reduced variants) it prints the mean over runs of the mean over ranks of
the time, and, when near and far runs are both given, the far/near ratio.

Usage: python3 analyze_placement.py near-r1.log far-r1.log [intra.log ...]
"""

import os
import re
import statistics as st
import sys

LINE = re.compile(
    r"[*#@+-] EP:\s+(\d+)/(\d+) \| (expanded |cached |reduced )?"
    r"(dispatch|combine): ([\d.]+) GB/s \(SO\), ([\d.]+) GB/s \(SU\), "
    r"([\d.]+) us, (\d+) bytes"
)


def parse(path):
    """Kind -> list of per-rank times in us (first line per rank)."""
    seen = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for m in LINE.finditer(fh.read()):
            rank, variant, op, us = (
                m.group(1),
                m.group(3),
                m.group(4),
                m.group(7),
            )
            kind = (variant.strip() + "-" if variant else "") + op
            seen.setdefault((kind, int(rank)), float(us))
    by_kind = {}
    for (kind, _rank), us in seen.items():
        by_kind.setdefault(kind, []).append(us)
    return by_kind


def collect(paths):
    """Placement -> kind -> list of (mean time over ranks, ranks), per run."""
    runs = {}
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        placement = re.sub(r"-r\d+$", "", name)
        for kind, times in parse(path).items():
            entry = (st.mean(times), len(times))
            runs.setdefault(placement, {}).setdefault(kind, []).append(entry)
    return runs


def main():
    """Print per-placement means and the far/near ratio."""
    runs = collect(sys.argv[1:])
    print("placement,kind,runs,ranks,mean_time_us")
    for placement, kinds in sorted(runs.items()):
        for kind, entries in sorted(kinds.items()):
            mean_us = st.mean(e[0] for e in entries)
            ranks = min(e[1] for e in entries)
            print(f"{placement},{kind},{len(entries)},{ranks},{mean_us:.1f}")
    if "near" in runs and "far" in runs:
        print("\nkind,near_us,far_us,far_over_near")
        for kind in sorted(set(runs["near"]) & set(runs["far"])):
            near = st.mean(e[0] for e in runs["near"][kind])
            far = st.mean(e[0] for e in runs["far"][kind])
            print(f"{kind},{near:.1f},{far:.1f},{far / near:.4f}")


if __name__ == "__main__":
    main()
