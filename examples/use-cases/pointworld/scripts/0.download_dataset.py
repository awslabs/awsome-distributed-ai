#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Download the PointWorld DROID and BEHAVIOR dataset packages from Hugging Face.

PointWorld distributes its generated datasets as packaged archives on the Hugging
Face Hub:

    DROID    : https://huggingface.co/datasets/nvidia/PointWorld-DROID
    BEHAVIOR : https://huggingface.co/datasets/nvidia/PointWorld-BEHAVIOR

Each dataset repo ships a ``recover_dataset_from_parts.sh`` helper that
reassembles the multi-part archive into the restored dataset root. This script
only performs the *download* step into the shared filesystem; you then run the
upstream recovery + integrity-check + WDS-conversion steps (see step 1,
``1.convert_wds.py``, and the README "Data Pipeline" section).

The DROID flow package is split into independent shards, so you can download a
subset for development / smoke tests, or the full package for full-dataset
pre-training and evaluation.

Examples
--------
Full DROID + BEHAVIOR download to FSx:

    python 0.download_dataset.py \
        --output_dir /fsx/$USER/pointworld/downloads \
        --datasets droid behavior

Subset of DROID flow shards only (development):

    python 0.download_dataset.py \
        --output_dir /fsx/$USER/pointworld/downloads \
        --datasets droid \
        --allow_patterns "droid/flow/shard-0000*.tar" "droid/confidence/*"
"""

import argparse
import sys

DATASET_REPOS = {
    "droid": "nvidia/PointWorld-DROID",
    "behavior": "nvidia/PointWorld-BEHAVIOR",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Destination directory on the shared filesystem (e.g. /fsx/$USER/pointworld/downloads).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_REPOS.keys()),
        default=["droid", "behavior"],
        help="Which dataset packages to download (default: both).",
    )
    parser.add_argument(
        "--allow_patterns",
        nargs="*",
        default=None,
        help="Optional glob patterns to download a subset (e.g. a few DROID flow shards).",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional dataset repo revision (branch, tag, or commit) to pin.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Parallel download workers.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit(
            "huggingface_hub is required. Install it with:\n"
            "    pip install huggingface_hub==0.26.2"
        )

    for name in args.datasets:
        repo_id = DATASET_REPOS[name]
        local_dir = f"{args.output_dir.rstrip('/')}/{name}"
        print(f"[0.download_dataset] Downloading {repo_id} -> {local_dir}", flush=True)
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            allow_patterns=args.allow_patterns,
            revision=args.revision,
            max_workers=args.max_workers,
        )
        print(
            f"[0.download_dataset] Done: {repo_id}\n"
            f"  Next: run the upstream 'recover_dataset_from_parts.sh' inside\n"
            f"  {local_dir} to restore the dataset, then convert to WebDataset\n"
            f"  shards with 1.convert_wds.py.",
            flush=True,
        )


if __name__ == "__main__":
    main()
