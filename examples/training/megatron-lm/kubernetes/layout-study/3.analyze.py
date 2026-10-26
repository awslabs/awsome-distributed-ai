# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Summarise 2.run-layouts.sh: per-arm medians and what each layout costs.

Reads <result dir>/training.csv and leaves out runs whose transport wasn't
proven or that logged no step times. For each arm (layout, placement,
sequence length) it prints the median over rounds of the step time and the
throughput. Then, as throughput changes (negative is slower):
  tp16 vs tp8-dp2        tensor parallelism across instances instead of
                         data parallelism
  tp8-pp2 vs tp8-dp2     pipeline parallelism across instances instead of
                         data parallelism
  tp16 vs tp8-half       the crossing itself: the same per-GPU work on one
                         instance
  tp8-split vs tp8       one TP 8 group split across two instances
  far vs near            placement, for every layout run on both
and the share of tp16's extra step time over tp8-dp2 that is the crossing.

Usage: python3 3.analyze.py <result dir>
"""

import csv
import os
import statistics as st
import sys


def load(path):
    """(layout, placement, seq_len) -> (median step ms, median tok/s, n)."""
    runs = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["step_ms"] == "NA":
                continue
            key = (row["layout"], row["placement"], int(row["seq_len"]))
            value = (float(row["step_ms"]), float(row["tokens_per_sec"]))
            runs.setdefault(key, []).append(value)
    return {
        key: (
            st.median(v[0] for v in vals),
            st.median(v[1] for v in vals),
            len(vals),
        )
        for key, vals in runs.items()
    }


def comparisons(arms):
    """(name, arm, baseline) pairs for every comparison the data allows."""
    pairs = []
    for seq in sorted({key[2] for key in arms}):
        for pl in ("near", "far"):
            pairs += [
                (
                    f"tp16 vs tp8-dp2 ({pl})",
                    ("tp16", pl, seq),
                    ("tp8-dp2", pl, seq),
                ),
                (
                    f"tp8-pp2 vs tp8-dp2 ({pl})",
                    ("tp8-pp2", pl, seq),
                    ("tp8-dp2", pl, seq),
                ),
                (
                    f"tp16 ({pl}) vs tp8-half",
                    ("tp16", pl, seq),
                    ("tp8-half", "one", seq),
                ),
                (
                    f"tp8-split ({pl}) vs tp8",
                    ("tp8-split", pl, seq),
                    ("tp8", "one", seq),
                ),
            ]
        for layout in sorted({key[0] for key in arms}):
            pairs.append(
                (
                    f"{layout} far vs near",
                    (layout, "far", seq),
                    (layout, "near", seq),
                )
            )
    return [(name, a, b) for name, a, b in pairs if a in arms and b in arms]


def print_arms(arms):
    """Per-arm medians over rounds."""
    print("layout,placement,seq_len,rounds,step_ms,tokens_per_sec")
    for key in sorted(arms, key=lambda k: (k[2], k[0], k[1])):
        step, tps, n = arms[key]
        print(f"{key[0]},{key[1]},{key[2]},{n},{step:.1f},{tps:.0f}")


def print_comparisons(arms):
    """Throughput of each arm against its baseline."""
    print(
        "\ncomparison,seq_len,tokens_per_sec,baseline_tokens_per_sec,"
        "throughput_change"
    )
    for name, a, b in comparisons(arms):
        change = arms[a][1] / arms[b][1] - 1
        print(f"{name},{a[2]},{arms[a][1]:.0f},{arms[b][1]:.0f},{change:+.1%}")


def print_crossing_share(arms):
    """How much of tp16's extra step time over tp8-dp2 is the crossing."""
    print(
        "\nplacement,seq_len,tp16_extra_step_ms_over_tp8-dp2,"
        "crossing_ms,crossing_share"
    )
    for (layout, pl, seq), (step, _tps, _n) in sorted(arms.items()):
        dp2, half = ("tp8-dp2", pl, seq), ("tp8-half", "one", seq)
        if layout != "tp16" or dp2 not in arms or half not in arms:
            continue
        extra, crossing = step - arms[dp2][0], step - arms[half][0]
        share = f"{crossing / extra:.0%}" if extra > 0 else "NA"
        print(f"{pl},{seq},{extra:.1f},{crossing:.1f},{share}")


def main():
    """Print the per-arm table, the comparisons and tp16's crossing share."""
    arms = load(os.path.join(sys.argv[1], "training.csv"))
    print_arms(arms)
    print_comparisons(arms)
    print_crossing_share(arms)


if __name__ == "__main__":
    main()
