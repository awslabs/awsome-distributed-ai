#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Convert restored PointWorld H5 datasets into WebDataset (WDS) shards.

The PointWorld ``main`` (train/eval) branch consumes local WebDataset shards, not
the packaged Hugging Face archives directly. After downloading (0.download_dataset.py)
and restoring a dataset with the upstream ``recover_dataset_from_parts.sh``, the
conversion + integrity-check tooling lives on the PointWorld ``data`` branch
(``data_integrity_check.py`` and ``convert_wds.py``).

This wrapper is a thin, documented entrypoint that invokes the upstream
``convert_wds.py`` from a checkout of the PointWorld ``data`` branch so the WDS
layout exactly matches what training/eval expect:

    <output_root>/droid/wds/{train,test}/...
    <output_root>/behavior/wds/{train,test}/...

For DROID filtered metrics, remember to copy the released expert-confidence
artifact into the generated WDS test split:

    droid/confidence/expert_confidence-seed=42.h5
        -> <output_root>/droid/wds/test/expert_confidence-seed=42.h5

Example
-------
    python 1.convert_wds.py \
        --pointworld_data_branch /fsx/$USER/pointworld/PointWorld-data \
        --restored_root /fsx/$USER/pointworld/downloads/droid/pointworld_droid_restored \
        --domain droid \
        --output_root /dataset
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
        "(contains data_integrity_check.py and convert_wds.py).",
    )
    parser.add_argument(
        "--restored_root",
        required=True,
        help="Path to the restored dataset root produced by recover_dataset_from_parts.sh.",
    )
    parser.add_argument(
        "--domain",
        required=True,
        choices=["droid", "behavior"],
        help="Which domain is being converted.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="WDS output root. Training expects <output_root>/<domain>/wds (LOCAL_DATASET_DIR).",
    )
    parser.add_argument(
        "--skip_integrity_check",
        action="store_true",
        help="Skip data_integrity_check.py (not recommended for full-scale runs).",
    )
    parser.add_argument(
        "--extra_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional args passed verbatim to the upstream convert_wds.py.",
    )
    return parser.parse_args()


def run(cmd, cwd):
    print(f"[1.convert_wds] $ {' '.join(cmd)}  (cwd={cwd})", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    args = parse_args()
    branch_dir = os.path.abspath(args.pointworld_data_branch)
    convert_script = os.path.join(branch_dir, "convert_wds.py")
    integrity_script = os.path.join(branch_dir, "data_integrity_check.py")

    if not os.path.isfile(convert_script):
        sys.exit(
            f"convert_wds.py not found at {convert_script}.\n"
            "Check out the PointWorld 'data' branch first:\n"
            "    git clone https://github.com/NVlabs/PointWorld.git "
            f"{branch_dir} && cd {branch_dir} && git checkout data"
        )

    output_wds = os.path.join(args.output_root.rstrip("/"), args.domain, "wds")
    os.makedirs(output_wds, exist_ok=True)

    if not args.skip_integrity_check and os.path.isfile(integrity_script):
        run(
            [sys.executable, integrity_script, "--restored_root", args.restored_root, "--domain", args.domain],
            cwd=branch_dir,
        )

    run(
        [
            sys.executable,
            convert_script,
            "--restored_root",
            args.restored_root,
            "--domain",
            args.domain,
            "--output_root",
            output_wds,
            *args.extra_args,
        ],
        cwd=branch_dir,
    )

    print(
        f"[1.convert_wds] WDS shards written under {output_wds}\n"
        f"  Training consumes this via --data_dirs={output_wds} "
        f"(or LOCAL_DATASET_DIR={args.output_root}).",
        flush=True,
    )


if __name__ == "__main__":
    main()
