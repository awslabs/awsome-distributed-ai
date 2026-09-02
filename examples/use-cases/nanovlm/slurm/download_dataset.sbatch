#!/bin/bash
#SBATCH --job-name=download_dataset
#SBATCH --output=logs/download_%A.out
#SBATCH --error=logs/download_%A.err
#SBATCH --nodes=2
#SBATCH --cpus-per-task=48
#SBATCH --partition=p5en

HF_TOKEN=$HF_TOKEN

mkdir -p logs
mkdir -p $DATASET_DIR


python3 << 'EOF'
from datasets import load_dataset
import os
import time

local_path = os.environ["DATASET_DIR"]

os.makedirs(local_path, exist_ok=True)

configs = ["clevr", "vqav2", "docvqa"]
total_start = time.time()

for i, config in enumerate(configs):
    print(f"Downloading {i+1}/{len(configs)}: {config}")
    start_time = time.time()
    try:
        dataset = load_dataset("HuggingFaceM4/the_cauldron", config)
        dataset.save_to_disk(f"{local_path}/{config}")
        elapsed = time.time() - start_time
        print(f"✓ Saved {config} in {elapsed:.1f}s")
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ Failed {config} after {elapsed:.1f}s: {e}")

total_elapsed = time.time() - total_start
print(f"Total time: {total_elapsed:.1f}s")
EOF
