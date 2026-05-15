#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Generate a synthetic video dataset for V-JEPA 2 benchmarking.

Creates short random-content video files and a CSV file compatible
with V-JEPA 2's VideoDataset format.

Usage:
    python generate_synthetic_dataset.py \
        --output_dir /fsx/<your_username>/vjepa2/datasets/synthetic_50k \
        --num_videos 50000 \
        --workers 64

Note: We recommend generating at least 50,000 videos for reliable benchmark
results. With fewer videos (e.g. 5,000), the data loader must frequently
re-initialize workers between epochs, which inflates iteration times and
masks the true GPU throughput.
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def generate_video(args_tuple):
    """Generate a single synthetic video. Accepts a tuple for pool.map()."""
    output_path, num_frames, width, height, fps, seed = args_tuple
    if os.path.exists(output_path):
        return True

    try:
        import cv2

        rng = np.random.RandomState(seed)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for _ in range(num_frames):
            frame = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
            out.write(frame)
        out.release()
        return True
    except ImportError:
        pass

    # Fallback: use ffmpeg
    duration = num_frames / fps
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={duration}:size={width}x{height}:rate={fps}",
        "-vf",
        f"drawtext=text='frame %{{n}}':x=10:y=10:fontsize=20:fontcolor=white",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "ultrafast",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error for {output_path}: {result.stderr}", file=sys.stderr)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic video dataset for V-JEPA 2"
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--num_videos",
        type=int,
        default=50000,
        help="Number of synthetic videos to generate (50k recommended for benchmarks)",
    )
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument(
        "--num_classes",
        type=int,
        default=174,
        help="Number of label classes (174 matches SSv2)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (set to CPU count for max speed)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    # Build work items
    work = []
    for i in range(args.num_videos):
        video_path = str(video_dir / f"video_{i:06d}.mp4")
        work.append((video_path, args.num_frames, args.width, args.height, args.fps, i))

    print(f"Generating {args.num_videos} videos with {args.workers} workers...")
    t0 = time.time()
    success = 0
    fail = 0

    if args.workers <= 1:
        for i, item in enumerate(work):
            ok = generate_video(item)
            success += ok
            fail += not ok
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{args.num_videos} videos...")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(generate_video, item): i for i, item in enumerate(work)
            }
            for future in as_completed(futures):
                ok = future.result()
                success += ok
                fail += not ok
                done = success + fail
                if done % 2000 == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    print(f"  {done}/{args.num_videos} videos ({rate:.0f}/s)...")

    elapsed = time.time() - t0
    print(
        f"\nGenerated {success} videos in {elapsed:.1f}s ({success / elapsed:.0f}/s), {fail} failed"
    )

    # Write CSV (sequential, fast)
    csv_path = output_dir / "synthetic_train_paths.csv"
    with open(csv_path, "w") as csv_file:
        for i in range(args.num_videos):
            video_path = video_dir / f"video_{i:06d}.mp4"
            if video_path.exists():
                label = i % args.num_classes
                csv_file.write(f"{video_path} {label}\n")

    print(f"CSV: {csv_path}")
    print(f"Videos: {video_dir}")


if __name__ == "__main__":
    main()
