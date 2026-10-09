#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# =============================================================================
# Dataset Mixing Utility
# Combines multiple prepared parquet datasets with configurable ratios
# Outputs a single train.parquet and val.parquet for verl training
# =============================================================================
"""
Usage:
    python3 data/mix_datasets.py \
        --datasets /fsx/data/verl/data/eurus/train.parquet:1.0 \
                   /fsx/data/verl/data/apps/train.parquet:0.5 \
                   /fsx/data/verl/data/taco/train.parquet:0.3 \
        --output /fsx/data/verl/data/mixed \
        --val-ratio 0.1 \
        --seed 42

Each --datasets entry is a parquet file path followed by a colon and a sampling
ratio. A ratio of 1.0 means use all samples; 0.5 means use 50% of samples.

If --val-files are provided, they are combined the same way for the val set.
Otherwise, if --val-ratio > 0, a fraction of the combined train data is split
off as validation.
"""

import argparse
import os
import sys

import datasets
import numpy as np


def parse_dataset_spec(spec):
    """Parse a 'path:ratio' string into (path, ratio)."""
    if ":" in spec:
        # Split on the last colon to handle paths with colons (rare but possible)
        last_colon = spec.rfind(":")
        path = spec[:last_colon]
        try:
            ratio = float(spec[last_colon + 1 :])
        except ValueError:
            # If ratio isn't a valid float, treat the whole thing as a path
            path = spec
            ratio = 1.0
    else:
        path = spec
        ratio = 1.0
    return path, ratio


def load_and_sample(path, ratio, rng):
    """Load a parquet file and sample according to ratio."""
    print(f"  Loading {path} (ratio={ratio})...")
    ds = datasets.load_dataset("parquet", data_files=path, split="train")
    total = len(ds)

    if ratio < 1.0:
        n_samples = max(1, int(total * ratio))
        indices = rng.choice(total, size=n_samples, replace=False)
        indices.sort()
        ds = ds.select(indices.tolist())
        print(f"    Sampled {len(ds)}/{total} rows")
    else:
        print(f"    Using all {total} rows")

    return ds


def validate_dataset(ds, path):
    """Validate that a dataset has the required verl columns."""
    required = ["prompt"]
    recommended = ["data_source", "reward_model"]

    for col in required:
        if col not in ds.column_names:
            print(f"  ERROR: {path} is missing required column '{col}'")
            sys.exit(1)

    for col in recommended:
        if col not in ds.column_names:
            print(f"  WARNING: {path} is missing recommended column '{col}'")

    # Check that prompt is in chat message format
    sample = ds[0]["prompt"]
    if not isinstance(sample, list) or not (
        len(sample) > 0 and isinstance(sample[0], dict)
    ):
        print(
            f"  WARNING: {path} prompt is not in chat message format (expected list of dicts)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Combine multiple verl-compatible parquet datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Parquet files with optional ratio: /path/to/train.parquet:0.5",
    )
    parser.add_argument(
        "--val-files",
        nargs="*",
        default=None,
        help="Separate val parquet files with optional ratio (same format as --datasets)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for mixed train.parquet and val.parquet",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of combined data to use as validation if --val-files not provided (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Maximum total samples in combined dataset (-1 for unlimited)",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.output, exist_ok=True)

    # Load and sample train datasets
    print("Loading training datasets...")
    train_parts = []
    for spec in args.datasets:
        path, ratio = parse_dataset_spec(spec)
        ds = load_and_sample(path, ratio, rng)
        validate_dataset(ds, path)
        train_parts.append(ds)

    # Concatenate
    print("\nConcatenating datasets...")
    combined = datasets.concatenate_datasets(train_parts)
    print(f"  Combined: {len(combined)} rows")

    # Shuffle
    combined = combined.shuffle(seed=args.seed)

    # Apply max_samples limit
    if args.max_samples > 0 and len(combined) > args.max_samples:
        combined = combined.select(range(args.max_samples))
        print(f"  Truncated to {args.max_samples} rows")

    # Handle validation set
    if args.val_files:
        # Load separate val files
        print("\nLoading validation datasets...")
        val_parts = []
        for spec in args.val_files:
            path, ratio = parse_dataset_spec(spec)
            ds = load_and_sample(path, ratio, rng)
            validate_dataset(ds, path)
            val_parts.append(ds)
        val_data = datasets.concatenate_datasets(val_parts)
        val_data = val_data.shuffle(seed=args.seed)
        train_data = combined
    elif args.val_ratio > 0:
        # Split from combined
        val_size = max(1, int(len(combined) * args.val_ratio))
        train_size = len(combined) - val_size
        train_data = combined.select(range(train_size))
        val_data = combined.select(range(train_size, len(combined)))
    else:
        train_data = combined
        val_data = None

    # Save
    train_file = os.path.join(args.output, "train.parquet")
    train_data.to_parquet(train_file)
    print(f"\nSaved {len(train_data)} train samples to {train_file}")

    if val_data is not None and len(val_data) > 0:
        val_file = os.path.join(args.output, "val.parquet")
        val_data.to_parquet(val_file)
        print(f"Saved {len(val_data)} val samples to {val_file}")

    # Summary
    print("\n" + "=" * 50)
    print("Dataset mixing complete!")
    print("=" * 50)

    # Count data sources
    sources = {}
    for i in range(len(train_data)):
        src = train_data[i].get("data_source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    print("\nData source distribution (train):")
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(train_data)
        print(f"  {src}: {count} ({pct:.1f}%)")

    print("\nTo use this data, set in env_vars:")
    print(f'  export TRAIN_FILE="{train_file}"')
    if val_data is not None:
        print(f'  export VAL_FILE="{os.path.join(args.output, "val.parquet")}"')


if __name__ == "__main__":
    main()
