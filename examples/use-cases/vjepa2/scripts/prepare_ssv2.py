#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Prepare Something-Something v2 (SSv2) dataset CSV for V-JEPA 2 training.

V-JEPA 2's VideoDataset expects a space-delimited CSV with format:
    <video_path> <label_index>

Usage:
    python prepare_ssv2.py \
        --video_dir /fsx/datasets/ssv2/videos \
        --labels_json /fsx/datasets/ssv2/labels.json \
        --train_json /fsx/datasets/ssv2/train.json \
        --output_csv /fsx/datasets/ssv2/ssv2_train_paths.csv
"""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Prepare SSv2 CSV for V-JEPA 2")
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Directory containing SSv2 video files (e.g., .webm)",
    )
    parser.add_argument(
        "--labels_json",
        type=str,
        required=True,
        help="Path to SSv2 labels.json (template -> label_id mapping)",
    )
    parser.add_argument(
        "--train_json",
        type=str,
        required=True,
        help="Path to SSv2 train.json (video_id -> template mapping)",
    )
    parser.add_argument(
        "--output_csv", type=str, required=True, help="Output CSV file path"
    )
    args = parser.parse_args()

    # Load label mapping: template_string -> label_index
    with open(args.labels_json, "r") as f:
        labels = json.load(f)

    # Load training split: list of {id, template} dicts
    with open(args.train_json, "r") as f:
        train_data = json.load(f)

    video_dir = Path(args.video_dir)
    found = 0
    missing = 0

    with open(args.output_csv, "w") as out:
        for entry in train_data:
            video_id = entry["id"]
            template = entry["template"].replace("[", "").replace("]", "")

            if template not in labels:
                missing += 1
                continue

            label_idx = labels[template]

            # SSv2 videos can be .webm or .mp4
            video_path = None
            for ext in [".webm", ".mp4"]:
                candidate = video_dir / f"{video_id}{ext}"
                if candidate.exists():
                    video_path = candidate
                    break

            if video_path is None:
                missing += 1
                continue

            out.write(f"{video_path} {label_idx}\n")
            found += 1

    print(f"Wrote {found} entries to {args.output_csv}")
    if missing > 0:
        print(f"Skipped {missing} entries (missing video file or unknown label)")


if __name__ == "__main__":
    main()
