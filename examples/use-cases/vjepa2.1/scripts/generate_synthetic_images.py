#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Generate a synthetic image dataset for V-JEPA 2.1 image co-training.

V-JEPA 2.1 co-trains on images and videos simultaneously. Image ranks use
VideoDataset with dataset_fpcs=[1] and load .jpg/.png files via
torchvision.io.read_image(). This script generates random-content JPEG
images and a CSV file compatible with that format.

Usage:
    python generate_synthetic_images.py \
        --output_dir /fsx/<your_username>/vjepa2.1/datasets/synthetic_images_50k \
        --num_images 50000 \
        --workers 64

Note: We recommend generating at least 50,000 images for reliable benchmark
results. With fewer images (e.g. 5,000), the data loader must frequently
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


def generate_image(args_tuple):
    """Generate a single synthetic JPEG image. Accepts a tuple for pool.map()."""
    output_path, width, height, seed = args_tuple
    if os.path.exists(output_path):
        return True

    try:
        from PIL import Image
    except ImportError:
        # Fallback: use raw numpy + PPM -> JPEG via ffmpeg
        rng = np.random.RandomState(seed)
        pixels = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
        ppm_path = output_path.replace(".jpg", ".ppm")
        with open(ppm_path, "wb") as f:
            f.write(f"P6\n{width} {height}\n255\n".encode())
            f.write(pixels.tobytes())
        result = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", ppm_path, output_path],
            capture_output=True,
            text=True,
        )
        if os.path.exists(ppm_path):
            os.remove(ppm_path)
        return result.returncode == 0

    rng = np.random.RandomState(seed)
    pixels = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(pixels, "RGB")
    img.save(output_path, "JPEG", quality=85)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic image dataset for V-JEPA 2.1"
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--num_images",
        type=int,
        default=50000,
        help="Number of synthetic images to generate (50k recommended for benchmarks)",
    )
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument(
        "--num_classes",
        type=int,
        default=1000,
        help="Number of label classes (1000 matches ImageNet)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (set to CPU count for max speed)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    # Build work items
    work = []
    for i in range(args.num_images):
        image_path = str(image_dir / f"img_{i:06d}.jpg")
        work.append((image_path, args.width, args.height, i))

    print(f"Generating {args.num_images} images with {args.workers} workers...")
    t0 = time.time()
    success = 0
    fail = 0

    if args.workers <= 1:
        for i, item in enumerate(work):
            ok = generate_image(item)
            success += ok
            fail += not ok
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{args.num_images} images...")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(generate_image, item): i for i, item in enumerate(work)
            }
            for future in as_completed(futures):
                ok = future.result()
                success += ok
                fail += not ok
                done = success + fail
                if done % 5000 == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    print(f"  {done}/{args.num_images} images ({rate:.0f}/s)...")

    elapsed = time.time() - t0
    print(
        f"\nGenerated {success} images in {elapsed:.1f}s ({success / elapsed:.0f}/s), {fail} failed"
    )

    # Write CSV (sequential, fast)
    csv_path = output_dir / "synthetic_image_paths.csv"
    with open(csv_path, "w") as csv_file:
        for i in range(args.num_images):
            image_path = image_dir / f"img_{i:06d}.jpg"
            if image_path.exists():
                label = i % args.num_classes
                csv_file.write(f"{image_path} {label}\n")

    print(f"CSV: {csv_path}")
    print(f"Images: {image_dir}")


if __name__ == "__main__":
    main()
