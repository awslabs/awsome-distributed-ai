#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# =============================================================================
# build_val_holdout.py — append never-trained MATH rows to the val split
#
# WHY
# The weighted MATH val aggregate has n=515, giving it a 95% margin of +-0.060 --
# the same size as the effect it is used to measure (+0.0661 at step 100, 2.14
# sigma). That makes every MATH claim marginal by construction. Enlarging the
# split is the cheapest resolution win in the project and costs ZERO GPU training
# time.
#
#   MATH val 515 -> 2515 rows  =>  margin +-0.060 -> +-0.027 (2.2x)
#   single-run 2-sigma success threshold  +0.12 -> +0.054
#
# THREE PROPERTIES THIS SCRIPT MUST GUARANTEE, IN ORDER OF IMPORTANCE
#
# 1. THE ORIGINAL 8117 ROWS ARE COPIED BYTE-FOR-BYTE, LABELS UNCHANGED.
#    The legacy aggregates must stay computable so every prior number remains
#    comparable. CODE val is therefore completely untouched: same 7601 rows, same
#    per-source counts, same weighted-CODE definition. Only MATH gains rows.
#
# 2. THE NEW ROWS WERE NEVER TRAINED ON.
#    Enforced by prompt-hash set difference against BOTH mixed-code-math
#    train.parquet and val.parquet -- not by replaying the sampler's RNG.
#    mix_datasets.load_and_sample() draws from a single np.random.default_rng(42)
#    shared across every dataset in the --datasets list, so reproducing the exact
#    index set requires knowing the original argument order. Hashing does not.
#
# 3. THE NEW ROWS ARE SCORED BY THE SAME CODE PATH AS THEIR LEGACY TWINS.
#    They carry a `_h2` suffix so verl reports them under separate
#    `val-core/<data_source>/acc/mean@1` keys (which is the whole point -- it is
#    what keeps the legacy 515-row aggregate computable). A suffixed label is not
#    in custom_reward_fn._KNOWN_DATA_SOURCES, so WITHOUT the suffix-strip added
#    alongside this script it would take the `fallback-math` path and be scored as
#    numina_olympiads instead of as its own source. Proven identical by
#    scripts/test_reward_routing.py -- do not deploy this data without that fix.
#
# STRATIFICATION: MATCH, DO NOT BALANCE.
# Quotas reproduce the legacy MATH source proportions rather than equalising
# them. Matching makes the extended aggregate an estimate of the SAME population
# as the legacy one, so the two are comparable and poolable. Balancing would
# silently redefine what "MATH" means and make the comparison meaningless.
#
# TRAP: SCHEMA MISMATCH.
# eurus/train.parquet has extra_info = struct<index, split> (2 fields).
# mixed-code-math/val.parquet has extra_info = struct<...> (15 fields, carrying
# the code-task metadata). Appending without aligning the struct produces either
# a write error or, worse, a silently different schema that verl may not read.
# Every new row's extra_info is rebuilt field-by-field against the val schema,
# with absent fields set to typed nulls.
#
# TRAP: DO NOT CONCATENATE.
# table.take() / concat on this data dies with
#   pyarrow.lib.ArrowInvalid: offset overflow while concatenating arrays
# because competitive-programming ground_truth blobs exceed Arrow's 32-bit string
# offsets. Everything here streams: iter_batches -> ParquetWriter.
#
# USAGE (runs inside the fsx-utils pod, which has pyarrow):
#   kubectl cp data/build_val_holdout.py fsx-utils:/tmp/build_val_holdout.py
#   kubectl exec fsx-utils -- python3 /tmp/build_val_holdout.py \
#       --n-holdout 2000 --verify-only false
# =============================================================================

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq

MIXED_DIR = "/fsx/data/verl/data/mixed-code-math"
SOURCE_POOL = "/fsx/data/verl/data/eurus/train.parquet"
OUT_DIR = "/fsx/data/verl/data/mixed-code-math-valplus"

HOLDOUT_SUFFIX = "_h2"

# Legacy MATH per-source counts in mixed-code-math/val.parquet, verified by
# direct count. numina_amc_aime is EXCLUDED from
# the stratification: it has n=1 in the legacy split and is dropped by the
# aggregate's --min-n 10 threshold, so adding to it would change what the legacy
# MATH aggregate spans instead of just sharpening it.
LEGACY_MATH_COUNTS = {
    "numina_cn_k12": 216,
    "numina_synthetic_math": 173,
    "numina_olympiads": 76,
    "numina_synthetic_amc": 37,
    "numina_aops_forum": 13,
}
LEGACY_VAL_TOTAL = 8117
LEGACY_ALL_COUNTS = {
    "apps": 5067,
    "codecontests": 1272,
    "taco": 1199,
    "codeforces": 63,
    "numina_cn_k12": 216,
    "numina_synthetic_math": 173,
    "numina_olympiads": 76,
    "numina_synthetic_amc": 37,
    "numina_aops_forum": 13,
    "numina_amc_aime": 1,
}


def prompt_hash(prompt_obj) -> str:
    """Stable content hash of a chat-format prompt (list of {role, content})."""
    return hashlib.sha1(
        json.dumps(prompt_obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def collect_hashes(path: str, batch_size: int = 512) -> set[str]:
    seen: set[str] = set()
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=["prompt"]):
        for p in batch.column("prompt").to_pylist():
            seen.add(prompt_hash(p))
    return seen


def quotas(n_total: int) -> dict[str, int]:
    """Stratify n_total across MATH sources in the legacy proportions."""
    legacy_sum = sum(LEGACY_MATH_COUNTS.values())
    q = {s: round(n_total * c / legacy_sum) for s, c in LEGACY_MATH_COUNTS.items()}
    # Fix rounding drift on the largest source so the total is exact.
    drift = n_total - sum(q.values())
    if drift:
        biggest = max(q, key=lambda s: q[s])
        q[biggest] += drift
    return q


def align_extra_info(src_struct: pa.Array, target_type: pa.DataType) -> pa.Array:
    """Rebuild an extra_info struct against the target schema, nulls for absent fields."""
    present = {f.name for f in src_struct.type}
    arrays, fields = [], []
    for field in target_type:
        if field.name in present:
            arrays.append(src_struct.field(field.name).cast(field.type))
        else:
            arrays.append(pa.nulls(len(src_struct), type=field.type))
        fields.append(field)
    return pa.StructArray.from_arrays(arrays, fields=fields)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-holdout", type=int, default=2000)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    legacy_val = f"{MIXED_DIR}/val.parquet"
    legacy_train = f"{MIXED_DIR}/train.parquet"
    out_path = f"{args.out_dir}/val.parquet"

    val_pf = pq.ParquetFile(legacy_val)
    val_schema = val_pf.schema_arrow
    extra_type = val_schema.field("extra_info").type

    print("=" * 72)
    print("build_val_holdout")
    print("=" * 72)
    print(f"legacy val   : {legacy_val}  ({val_pf.metadata.num_rows} rows)")
    print(f"source pool  : {SOURCE_POOL}")
    print(f"output       : {out_path}")
    q = quotas(args.n_holdout)
    print(f"quotas (match legacy proportions, total {sum(q.values())}):")
    for s, n in q.items():
        print(f"    {s:<26} {n:>5}   (legacy {LEGACY_MATH_COUNTS[s]})")

    # ---- 2. provable disjointness -------------------------------------------
    print("\n[1/4] hashing prompts already seen by the control run ...")
    exclude = collect_hashes(legacy_train, args.batch_size)
    n_train = len(exclude)
    exclude |= collect_hashes(legacy_val, args.batch_size)
    print(f"      train hashes {n_train}, +val -> {len(exclude)} total exclude set")

    # ---- select holdout rows -------------------------------------------------
    print("\n[2/4] streaming source pool, selecting never-seen math rows ...")
    want = dict(q)
    picked_batches: list[pa.RecordBatch] = []
    picked_hashes: set[str] = set()
    n_scanned = n_dup_excluded = n_dup_internal = 0
    pool = pq.ParquetFile(SOURCE_POOL)
    for batch in pool.iter_batches(batch_size=args.batch_size):
        if not any(v > 0 for v in want.values()):
            break
        srcs = batch.column("data_source").to_pylist()
        prompts = batch.column("prompt").to_pylist()
        keep_idx = []
        for i, (src, prm) in enumerate(zip(srcs, prompts, strict=True)):
            n_scanned += 1
            if want.get(src, 0) <= 0:
                continue
            h = prompt_hash(prm)
            if h in exclude:
                n_dup_excluded += 1
                continue
            if h in picked_hashes:
                n_dup_internal += 1
                continue
            keep_idx.append(i)
            picked_hashes.add(h)
            want[src] -= 1
        if not keep_idx:
            continue
        mask = pa.array([i in set(keep_idx) for i in range(batch.num_rows)])
        sub = batch.filter(mask)
        # relabel data_source, align extra_info to the val schema
        new_src = pa.array([f"{s}{HOLDOUT_SUFFIX}" for s in sub.column("data_source").to_pylist()])
        cols = {
            "data_source": new_src,
            "prompt": sub.column("prompt"),
            "ability": sub.column("ability"),
            "reward_model": sub.column("reward_model"),
            "extra_info": align_extra_info(sub.column("extra_info"), extra_type),
        }
        picked_batches.append(
            pa.RecordBatch.from_arrays([cols[f.name] for f in val_schema], schema=val_schema)
        )

    n_picked = sum(b.num_rows for b in picked_batches)
    print(f"      scanned {n_scanned} rows")
    print(f"      excluded {n_dup_excluded} already in control train/val, {n_dup_internal} internal dups")
    print(f"      selected {n_picked} rows")
    unmet = {s: v for s, v in want.items() if v > 0}
    if unmet:
        print(f"      ERROR: quota not met for {unmet}", file=sys.stderr)
        return 1

    # ---- write: originals byte-identical first, then holdout ------------------
    print(f"\n[3/4] writing {out_path} (streaming, no concatenation) ...")
    n_orig = 0
    writer = pq.ParquetWriter(out_path, val_schema)
    try:
        for batch in val_pf.iter_batches(batch_size=args.batch_size):
            writer.write_batch(batch)
            n_orig += batch.num_rows
        for batch in picked_batches:
            writer.write_batch(batch)
    finally:
        writer.close()
    print(f"      wrote {n_orig} original + {n_picked} holdout = {n_orig + n_picked}")

    # ---- 4. verification gates ----------------------------------------------
    print("\n[4/4] verification gates")
    ok = True
    out_pf = pq.ParquetFile(out_path)
    counts: Counter[str] = Counter()
    out_hashes_new: set[str] = set()
    for batch in out_pf.iter_batches(batch_size=args.batch_size, columns=["data_source", "prompt"]):
        srcs = batch.column("data_source").to_pylist()
        prompts = batch.column("prompt").to_pylist()
        counts.update(srcs)
        for s, p in zip(srcs, prompts, strict=True):
            if s.endswith(HOLDOUT_SUFFIX):
                out_hashes_new.add(prompt_hash(p))

    def gate(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and cond
        print(f"      {'PASS' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")

    gate(
        "schema identical to legacy val",
        out_pf.schema_arrow.equals(val_schema),
    )
    gate(
        f"total rows == {LEGACY_VAL_TOTAL} + {n_picked}",
        out_pf.metadata.num_rows == LEGACY_VAL_TOTAL + n_picked,
        f"got {out_pf.metadata.num_rows}",
    )
    for src, exp in LEGACY_ALL_COUNTS.items():
        gate(f"legacy count unchanged: {src} == {exp}", counts.get(src, 0) == exp, f"got {counts.get(src, 0)}")
    for src, exp in q.items():
        h = f"{src}{HOLDOUT_SUFFIX}"
        gate(f"holdout count: {h} == {exp}", counts.get(h, 0) == exp, f"got {counts.get(h, 0)}")
    gate(
        "zero holdout overlap with control train/val",
        len(out_hashes_new & exclude) == 0,
        f"{len(out_hashes_new & exclude)} overlapping",
    )
    gate(
        "holdout rows are internally unique",
        len(out_hashes_new) == n_picked,
        f"{len(out_hashes_new)} unique of {n_picked}",
    )

    code_rows = sum(counts.get(s, 0) for s in ("apps", "codecontests", "taco", "codeforces"))
    gate("CODE val untouched at 7601 rows", code_rows == 7601, f"got {code_rows}")

    legacy_math = sum(counts.get(s, 0) for s in LEGACY_MATH_COUNTS)
    ext_math = legacy_math + sum(counts.get(f"{s}{HOLDOUT_SUFFIX}", 0) for s in LEGACY_MATH_COUNTS)
    # Margins: the authoritative figure comes from the sample-weighted aggregate
    # variance over the observed per-source accuracies (computed by a local
    # rescoring script, not shipped). Reported here only as the relative
    # improvement, which is what this script controls and which is independent
    # of p: margin scales as 1/sqrt(n).
    print(f"\n      MATH legacy   n={legacy_math}")
    print(f"      MATH extended n={ext_math}   ({(ext_math / legacy_math) ** 0.5:.2f}x tighter margin)")
    print(f"      published legacy margin +-0.060  =>  extended ~ +-{0.060 * (legacy_math / ext_math) ** 0.5:.4f}")
    print("      (approximate: exact figure requires sample-weighted rescoring of a run log)")

    print("\n" + ("ALL GATES PASSED" if ok else "GATES FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
