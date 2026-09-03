#!/usr/bin/env python3
"""
Pre-download a subset of the training dataset to local storage.

Downloads N samples from the HuggingFace Hub and saves them as an Arrow
dataset on the local/shared filesystem. This eliminates HuggingFace API
calls during training, preventing 429 rate-limiting errors when
ft_launcher or K8s restarts spawn many workers simultaneously.

Usage:
    python prepare_dataset.py --output_path /checkpoints/c4_subset
    python prepare_dataset.py --output_path /checkpoints/c4_subset --num_samples 100000
    python prepare_dataset.py --dataset_name allenai/c4 --subset en --num_samples 50000

The saved dataset can be used in training with:
    --dataset_path /checkpoints/c4_subset

Without --dataset_path, training scripts fall back to HF streaming
(the original behavior).
"""

import argparse
import time
from datasets import load_dataset, Dataset


def main():
    parser = argparse.ArgumentParser(
        description="Pre-download dataset subset to local storage"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="allenai/c4",
        help="HuggingFace dataset name (default: allenai/c4)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="en",
        help="Dataset subset/config (default: en)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split (default: train)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100000,
        help="Number of samples to download (default: 100000)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the dataset (e.g., /checkpoints/c4_subset)",
    )
    args = parser.parse_args()

    print(
        f"Downloading {args.num_samples} samples from {args.dataset_name}/{args.subset}..."
    )
    start = time.time()

    # Stream from HF and collect N samples
    ds_stream = load_dataset(
        args.dataset_name,
        args.subset,
        split=args.split,
        streaming=True,
        trust_remote_code=True,
    )

    samples = []
    for i, sample in enumerate(ds_stream):
        if i >= args.num_samples:
            break
        samples.append(sample)
        if (i + 1) % 10000 == 0:
            print(f"  Downloaded {i + 1}/{args.num_samples} samples...")

    elapsed = time.time() - start
    print(f"Downloaded {len(samples)} samples in {elapsed:.1f}s")

    # Save as Arrow dataset
    print(f"Saving to {args.output_path}...")
    dataset = Dataset.from_list(samples)
    dataset.save_to_disk(args.output_path)

    # Report size
    import os

    total_size = sum(
        os.path.getsize(os.path.join(dirpath, f))
        for dirpath, _, filenames in os.walk(args.output_path)
        for f in filenames
    )
    print(
        f"Saved {len(samples)} samples to {args.output_path} ({total_size / 1e6:.1f} MB)"
    )
    print("Done! Use --dataset_path in training scripts to load this dataset.")


if __name__ == "__main__":
    main()
