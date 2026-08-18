#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# =============================================================================
# Difficulty Filter — drop already-solved code problems before mixing
#
# WHY
# ---
# GRPO computes an advantage per prompt GROUP (n_responses_per_prompt=4):
#     advantage_i = (r_i - mean(r)) / std(r)
# A group whose 4 rewards are IDENTICAL has std=0, advantage 0, and therefore
# contributes NO GRADIENT. Problems the base model already solves every time are
# dead weight: they consume rollout tokens and sandbox capacity and teach nothing.
#
# Measured on the r=128 run, from the [REWARD] diagnostics:
#     apps           87.5% of samples scored exactly 1.0  -> ~58.6% dead groups
#     codecontests   70.0%                                -> ~24.0%
#     codeforces     57.5%                                -> ~11.0%
#     taco           52.5%                                -> ~ 8.0%
#
# apps is the most saturated set AND the one val is dominated by (66.7% of code
# val rows). Its `introductory` tier is the saturated part. taco's `EASY` tier is
# the analogous slice. Dropping those concentrates the code gradient on problems
# the model can still learn from, WITHOUT adding data volume (code is already 69%
# of training and 163 steps of it produced ~+0.01).
#
# WHAT THIS DOES NOT DO
# It does not touch validation. The val split must stay byte-identical so the
# measured base@32768 (weighted CODE 0.7231) remains valid and the new run is a
# clean single-variable comparison against the unfiltered control run. Pin val
# explicitly when mixing:
#     mix_datasets.py --val-files /fsx/data/verl/data/mixed-code-math/val.parquet:1.0
#
# DIFFICULTY LABELS PRESENT IN THE DATA (verified 2026-08-04 on FSx)
#     apps          introductory 54.9%  interview 37.6%  competition  7.5%
#     taco          EASY 35.0%  UNKNOWN_DIFFICULTY 19.7%  MEDIUM 12.8%
#                   HARD 12.4%  MEDIUM_HARD 10.8%  VERY_HARD 9.3%
#     codecontests  no labels at all (100% empty) -> unfilterable, passes through
#
# Rows with a missing/empty/unknown difficulty are KEPT. Only an explicit match
# against --drop is removed, so an unlabeled dataset is never silently gutted.
#
# USAGE (runs where the parquets live — the fsx-utils pod has pyarrow)
#   kubectl cp data/filter_difficulty.py fsx-utils:/tmp/filter_difficulty.py
#   kubectl exec fsx-utils -- python3 /tmp/filter_difficulty.py \
#       --input  /fsx/data/verl/data/apps/train.parquet \
#       --output /fsx/data/verl/data/apps-hard/train.parquet \
#       --drop introductory
#   kubectl exec fsx-utils -- python3 /tmp/filter_difficulty.py \
#       --input  /fsx/data/verl/data/taco/train.parquet \
#       --output /fsx/data/verl/data/taco-hard/train.parquet \
#       --drop EASY
#
# PREFERRED MODE — filter the ALREADY-MIXED train file
# ----------------------------------------------------
# Filtering the per-source parquets then re-mixing requires reproducing the
# original mix ratios exactly, and those cannot be reverse-engineered: the live
# mix holds MORE taco rows (25,837) than the source train file (25,443), so a
# train+val recombination or a non-obvious ratio was used. Guessing risks changing
# TWO things at once (composition AND difficulty), which would ruin the experiment.
#
# Instead, filter the existing mix in place with --drop-by-source. The result is
# provably the control's training set MINUS exactly the dropped rows, so the only
# variable is difficulty:
#
#   kubectl exec fsx-utils -- python3 /tmp/filter_difficulty.py \
#       --input  /fsx/data/verl/data/mixed-code-math/train.parquet \
#       --output /fsx/data/verl/data/mixed-code-math-hard/train.parquet \
#       --drop-by-source apps:introductory taco:EASY
#
# Then point training at the new train.parquet and REUSE the untouched val file.
#
# --dry-run reports the counts without writing anything.
# =============================================================================

from __future__ import annotations

import argparse
import collections
import os
import sys

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    print("ERROR: pyarrow is required (available in the fsx-utils pod)", file=sys.stderr)
    raise

# Rows are streamed in batches rather than filtered as one table. Competitive
# programming ground_truth blobs are enormous (verl truncates 100k+ line test
# inputs in its logs for this reason), so a whole-table `take()` overflows
# Arrow's 32-bit string offsets:
#     pyarrow.lib.ArrowInvalid: offset overflow while concatenating arrays
# Batching keeps every intermediate array small and writes incrementally.
_BATCH_ROWS = 512


def extract_difficulty(entry) -> str:
    """Pull `difficulty` out of an extra_info value, tolerating shape changes.

    prepare_data.py writes extra_info as a struct with a `difficulty` key, but be
    defensive: a dict is the expected case, anything else yields "" (= keep).
    """
    if isinstance(entry, dict):
        return str(entry.get("difficulty", "") or "")
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Drop rows whose extra_info.difficulty matches --drop (case-insensitive)."
    )
    ap.add_argument("--input", required=True, help="source parquet")
    ap.add_argument("--output", help="destination parquet (omit with --dry-run)")
    ap.add_argument(
        "--drop",
        nargs="+",
        default=[],
        metavar="LABEL",
        help='difficulty labels to remove, e.g. --drop introductory  /  --drop EASY',
    )
    ap.add_argument(
        "--drop-by-source",
        nargs="+",
        default=[],
        metavar="SOURCE:LABEL",
        help="drop only where data_source matches too, e.g. apps:introductory taco:EASY. "
        "Use this on an already-mixed train.parquet to guarantee a single-variable change.",
    )
    ap.add_argument("--dry-run", action="store_true", help="report counts, write nothing")
    args = ap.parse_args()

    if not args.dry_run and not args.output:
        ap.error("--output is required unless --dry-run")

    drop = {d.strip().lower() for d in args.drop if d.strip()}
    by_source: dict[str, set[str]] = {}
    for spec in args.drop_by_source:
        if ":" not in spec:
            ap.error(f"--drop-by-source expects SOURCE:LABEL, got {spec!r}")
        src, _, lab = spec.partition(":")
        by_source.setdefault(src.strip(), set()).add(lab.strip().lower())
    if not drop and not by_source:
        ap.error("give --drop and/or --drop-by-source (nothing to do otherwise)")

    print(f"Reading {args.input}")
    pf = pq.ParquetFile(args.input)
    schema = pf.schema_arrow
    if "extra_info" not in schema.names:
        print(f"ERROR: no extra_info column; columns are {schema.names}", file=sys.stderr)
        return 2
    has_src = "data_source" in schema.names
    if by_source and not has_src:
        print("ERROR: --drop-by-source needs a data_source column", file=sys.stderr)
        return 2

    def is_dropped(diff: str, src: str) -> bool:
        d = diff.lower()
        if d and d in drop:
            return True
        return d in by_source.get(src, ())

    pair_before: collections.Counter = collections.Counter()
    pair_after: collections.Counter = collections.Counter()
    total = kept = 0
    writer = None
    if not args.dry_run:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        writer = pq.ParquetWriter(args.output, schema)

    try:
        for batch in pf.iter_batches(batch_size=_BATCH_ROWS):
            tbl = pa.Table.from_batches([batch], schema=schema)
            n = tbl.num_rows
            total += n
            diffs = [extract_difficulty(e) for e in tbl.column("extra_info").to_pylist()]
            srcs = ([str(x) for x in tbl.column("data_source").to_pylist()]
                    if has_src else [""] * n)
            mask = []
            for i in range(n):
                dropped = is_dropped(diffs[i], srcs[i])
                key = (srcs[i] or "(none)", diffs[i] or "(empty)")
                pair_before[key] += 1
                mask.append(not dropped)
                if not dropped:
                    pair_after[key] += 1
            keep_n = sum(mask)
            kept += keep_n
            if writer is not None and keep_n:
                writer.write_table(tbl.filter(pa.array(mask, type=pa.bool_())))
    finally:
        if writer is not None:
            writer.close()

    removed = total - kept
    print(f"\n  composition BEFORE ({total} rows):")
    for (src, label), n in sorted(pair_before.items(), key=lambda x: -x[1]):
        mark = "  <-- DROP" if (label.lower() in drop or label.lower() in by_source.get(src, ())) else ""
        print(f"    {src:24s} {label:22s} {n:7d}  {n / max(total, 1) * 100:5.1f}%{mark}")

    if removed == 0:
        bs_repr = {k: sorted(v) for k, v in by_source.items()}
        print(f"\n  !! nothing matched (drop={sorted(drop)} by_source={bs_repr}) "
              f"— check the label spelling")
    print(f"\n  dropping {removed} rows ({removed / max(total, 1) * 100:.1f}%), keeping {kept}")

    if kept == 0:
        print("ERROR: filter would remove every row; refusing", file=sys.stderr)
        if not args.dry_run and os.path.exists(args.output):
            os.remove(args.output)
        return 2

    if args.dry_run:
        print("\n  --dry-run: nothing written")
        return 0

    # Verify by streaming the written file back, so a corrupt write cannot
    # silently poison a training run.
    check = pq.ParquetFile(args.output)
    if not check.schema_arrow.equals(schema):
        print("ERROR: written schema drifted from source — mixing would break", file=sys.stderr)
        return 2
    rows_back = 0
    leaked: collections.Counter = collections.Counter()
    for batch in check.iter_batches(batch_size=_BATCH_ROWS):
        tbl = pa.Table.from_batches([batch], schema=check.schema_arrow)
        rows_back += tbl.num_rows
        diffs = [extract_difficulty(e) for e in tbl.column("extra_info").to_pylist()]
        srcs = ([str(x) for x in tbl.column("data_source").to_pylist()]
                if has_src else [""] * tbl.num_rows)
        for i in range(tbl.num_rows):
            if is_dropped(diffs[i], srcs[i]):
                leaked[(srcs[i], diffs[i])] += 1

    print(f"\n  wrote {args.output} ({rows_back} rows)")
    print("  composition AFTER:")
    for (s_, l_), n in sorted(pair_after.items(), key=lambda x: -x[1]):
        print(f"    {s_:24s} {l_:22s} {n:7d}  {n / max(rows_back, 1) * 100:5.1f}%")
    if leaked or rows_back != kept:
        print(f"ERROR: verification failed (leaked={dict(leaked)}, "
              f"rows={rows_back} expected={kept})", file=sys.stderr)
        return 2
    print("  verified: no dropped labels remain, row count and schema match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
