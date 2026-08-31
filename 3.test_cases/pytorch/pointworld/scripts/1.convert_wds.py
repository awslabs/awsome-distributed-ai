#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Convert restored PointWorld H5 datasets into WebDataset (WDS) shards.

The PointWorld ``main`` (train/eval) branch consumes local WebDataset shards, not
the packaged Hugging Face archives directly. After downloading (0.download_dataset.py)
and restoring a dataset with the upstream ``recover_dataset_from_parts.sh``, the
conversion tooling lives on the PointWorld ``data`` branch. The end-to-end order is:

    data_integrity_check.py  ->  make_wds_manifest.py  ->  convert_wds.py

This wrapper runs all three from a checkout of the PointWorld ``data`` branch so
the WDS layout exactly matches what training/eval expect:

    <output_dir>/{train,test}/*.tar
    <output_dir>/metadata_rank0.json     (written by convert_wds on completion)

NOTE: ``convert_wds.py`` requires the ``--manifest`` produced by
``make_wds_manifest.py`` and only writes the ``metadata_rank*.json`` index when it
finishes processing the selected clips. Use ``--max_clips`` to bound the run for
development / smoke tests so it completes (and writes the metadata) quickly.

For DROID filtered metrics, remember to also copy the released expert-confidence
artifact into the generated WDS test split:

    droid/confidence/expert_confidence-seed=42.h5
        -> <output_dir>/test/expert_confidence-seed=42.h5

Example
-------
    python 1.convert_wds.py \
        --pointworld_data_branch /fsx/$USER/pointworld/PointWorld-data \
        --input_dir /fsx/$USER/pointworld/restored/behavior/behavior/flows \
        --domain behavior \
        --output_dir /fsx/$USER/pointworld/dataset/behavior/wds \
        --max_clips 300
"""

import argparse
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--pointworld_data_branch",
        required=True,
        help="Path to a local checkout of the PointWorld repo on the 'data' branch "
        "(contains data_integrity_check.py, make_wds_manifest.py, convert_wds.py).",
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory of restored .h5/.hdf5 files (e.g. <restored>/behavior/flows "
        "or <restored>/droid/flows-fs-optimized).",
    )
    parser.add_argument(
        "--domain",
        required=True,
        choices=["droid", "behavior"],
        help="Which domain is being converted.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="WDS output dir. Training expects <LOCAL_DATASET_DIR>/<domain>/wds.",
    )
    parser.add_argument(
        "--test_percentage",
        type=float,
        default=0.1,
        help="Fraction of selected clips placed in the test split.",
    )
    parser.add_argument(
        "--max_clips",
        type=int,
        default=-1,
        help="Cap total clips processed (-1 = all). Bound this for dev/smoke runs so "
        "convert_wds finishes and writes metadata_rank*.json.",
    )
    parser.add_argument(
        "--maxsize",
        type=float,
        default=1e9,
        help="Max WDS shard size in bytes.",
    )
    parser.add_argument(
        "--skip_integrity_check",
        action="store_true",
        help="Skip data_integrity_check.py (not recommended for full-scale runs).",
    )
    return parser.parse_args()


def run(cmd, cwd):
    print(f"[1.convert_wds] $ {' '.join(str(c) for c in cmd)}  (cwd={cwd})", flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def main():
    args = parse_args()
    branch = os.path.abspath(args.pointworld_data_branch)
    integrity = os.path.join(branch, "data_integrity_check.py")
    make_manifest = os.path.join(branch, "make_wds_manifest.py")
    convert = os.path.join(branch, "convert_wds.py")

    for script in (make_manifest, convert):
        if not os.path.isfile(script):
            sys.exit(
                f"{os.path.basename(script)} not found at {script}.\n"
                "Check out the PointWorld 'data' branch first:\n"
                "    git clone --branch data https://github.com/NVlabs/PointWorld.git "
                f"{branch}"
            )

    os.makedirs(args.output_dir, exist_ok=True)
    integrity_file = os.path.join(args.output_dir, "integrity_check.json")
    manifest_file = os.path.join(args.output_dir, "wds_manifest.json")

    if not args.skip_integrity_check and os.path.isfile(integrity):
        run(
            [sys.executable, integrity, "--input_dir", args.input_dir, "--domain", args.domain,
             "--output_file", integrity_file, "--fastmode"],
            cwd=branch,
        )

    run(
        [sys.executable, make_manifest, "--input_dir", args.input_dir, "--domain", args.domain,
         "--output_manifest", manifest_file, "--integrity_check_file", integrity_file,
         "--test_percentage", args.test_percentage, "--max_clips", args.max_clips],
        cwd=branch,
    )

    run(
        [sys.executable, convert, "--input_dir", args.input_dir, "--output_dir", args.output_dir,
         "--domain", args.domain, "--integrity_check_file", integrity_file,
         "--manifest", manifest_file, "--max_clips", args.max_clips, "--maxsize", args.maxsize],
        cwd=branch,
    )

    print(
        f"[1.convert_wds] WDS shards + metadata written under {args.output_dir}\n"
        f"  Training consumes this via --data_dirs={args.output_dir} "
        f"(or LOCAL_DATASET_DIR=<parent-of-domain>).",
        flush=True,
    )


if __name__ == "__main__":
    main()
