#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Test that decord is working correctly inside the container.

Usage:
    python test_decord.py --video_path /path/to/sample.webm

If no video path is provided, creates a synthetic test video with ffmpeg.
"""

import argparse
import os
import subprocess
import sys
import tempfile


def create_synthetic_video(path, num_frames=32, width=256, height=256, fps=4):
    """Create a small synthetic video using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={num_frames / fps}:size={width}x{height}:rate={fps}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(
        f"Created synthetic video: {path} ({num_frames} frames, {width}x{height}, {fps} fps)"
    )


def test_decord(video_path):
    """Test decord video loading."""
    print(f"\n=== Testing decord with: {video_path} ===\n")

    # 1. Import test
    print("[1/4] Importing decord...")
    from decord import VideoReader, cpu

    print("  OK: decord imported successfully")

    # 2. Load video
    print(f"[2/4] Loading video...")
    vr = VideoReader(video_path, num_threads=-1, ctx=cpu(0))
    num_frames = len(vr)
    avg_fps = vr.get_avg_fps()
    print(f"  OK: {num_frames} frames, avg fps: {avg_fps:.1f}")

    # 3. Read frames
    print("[3/4] Reading frame batch...")
    import numpy as np

    indices = np.linspace(0, num_frames - 1, min(16, num_frames), dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    print(f"  OK: batch shape = {frames.shape}, dtype = {frames.dtype}")

    # 4. Verify frame content
    print("[4/4] Verifying frame content...")
    assert frames.ndim == 4, f"Expected 4D tensor, got {frames.ndim}D"
    assert frames.shape[-1] == 3, f"Expected 3 channels (RGB), got {frames.shape[-1]}"
    assert frames.max() > 0, "All frames are black (max=0)"
    print(
        f"  OK: frames are valid (min={frames.min()}, max={frames.max()}, mean={frames.mean():.1f})"
    )

    print("\n=== All decord tests passed ===\n")


def main():
    parser = argparse.ArgumentParser(description="Test decord video loading")
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Path to a video file to test. If not provided, a synthetic video is created.",
    )
    args = parser.parse_args()

    if args.video_path and os.path.exists(args.video_path):
        test_decord(args.video_path)
    else:
        # Create a temporary synthetic video
        with tempfile.TemporaryDirectory() as tmpdir:
            synth_path = os.path.join(tmpdir, "test_video.mp4")
            create_synthetic_video(synth_path)
            test_decord(synth_path)


if __name__ == "__main__":
    main()
