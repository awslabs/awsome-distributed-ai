#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Download allenai/c4 and convert to Megatron indexed format (.bin + .idx).

Fully parallelized pipeline:
  Phase 1: Download c4/en parquet shards in parallel (requires HF_TOKEN)
  Phase 2: Tokenize in parallel from local Arrow cache using all CPUs
  Phase 3: Merge into Megatron MMapIndexedDataset format

Requirements:
    pip install numpy transformers datasets
    export HF_TOKEN=<your huggingface token>

Usage:
    python preprocess.py --output-prefix /fsx/ubuntu/qwen3-8b-pretraining/datasets/c4_qwen3_8b \
                         --num-tokens 1000000000 --workers 96
"""
import argparse
import os
import struct
import sys
import time
from multiprocessing import Pool, cpu_count

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# Megatron MMapIndexedDataset constants
_HDR_MAGIC = b"MMIDIDX\x00\x00"
_DTYPE = np.int32
_DTYPE_CODE = 4


def parse_args():
    p = argparse.ArgumentParser(description="Download allenai/c4 and convert to Megatron indexed format")
    p.add_argument("--output-prefix", default="/fsx/ubuntu/qwen3-8b-pretraining/datasets/c4_qwen3_8b",
                   help="Output path prefix (creates <prefix>.bin and <prefix>.idx)")
    p.add_argument("--tokenizer", default="Qwen/Qwen3-8B", help="HuggingFace tokenizer name")
    p.add_argument("--num-tokens", type=int, default=1_000_000_000, help="Target number of tokens")
    p.add_argument("--workers", type=int, default=min(96, cpu_count()), help="Parallel workers")
    p.add_argument("--cache-dir", default="/fsx/ubuntu/qwen3-8b-pretraining/cache/c4",
                   help="HuggingFace datasets cache directory")
    return p.parse_args()


def check_hf_token():
    """Verify HF_TOKEN is set. Exit with error if not."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN environment variable is not set.", file=sys.stderr)
        print("       Set it with: export HF_TOKEN=<your huggingface token>", file=sys.stderr)
        print("       Get a token at: https://huggingface.co/settings/tokens", file=sys.stderr)
        sys.exit(1)
    return token


def write_idx_file(idx_path, sizes, doc_idx):
    """Write a Megatron .idx file."""
    with open(idx_path, "wb") as f:
        f.write(_HDR_MAGIC)
        f.write(struct.pack("<Q", 1))  # version
        f.write(struct.pack("<B", _DTYPE_CODE))
        f.write(struct.pack("<Q", len(sizes)))
        f.write(struct.pack("<Q", len(doc_idx)))
        sizes_arr = np.array(sizes, dtype=np.int32)
        sizes_arr.tofile(f)
        pointers = np.zeros(len(sizes), dtype=np.int64)
        if len(sizes) > 0:
            pointers[1:] = np.cumsum(sizes_arr[:-1].astype(np.int64)) * _DTYPE().itemsize
        pointers.tofile(f)
        np.array(doc_idx, dtype=np.int64).tofile(f)


def tokenize_chunk(args_tuple):
    """Tokenize a chunk of documents. Returns (tmp_bin_path, sizes, doc_idx, total_tokens)."""
    chunk_id, texts, tokenizer_name, output_dir = args_tuple

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    tmp_bin = os.path.join(output_dir, f"_tmp_chunk_{chunk_id:05d}.bin")
    sizes = []
    doc_idx = [0]
    total_tokens = 0

    with open(tmp_bin, "wb") as out:
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if not tokens:
                continue
            arr = np.array(tokens, dtype=_DTYPE)
            out.write(arr.tobytes())
            sizes.append(len(tokens))
            doc_idx.append(len(sizes))
            total_tokens += len(tokens)

    return tmp_bin, sizes, doc_idx, total_tokens


def main():
    args = parse_args()
    token = check_hf_token()
    output_dir = os.path.dirname(args.output_prefix)
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== allenai/c4 -> Megatron indexed format ===")
    print(f"  Tokenizer: {args.tokenizer}")
    print(f"  Target: {args.num_tokens / 1e9:.1f}B tokens")
    print(f"  Workers: {args.workers}")
    print(f"  Output: {args.output_prefix}.{{bin,idx}}")
    t0 = time.time()

    # Phase 1: Download in parallel using datasets library
    # ~200 tokens/doc average for c4, with 20% buffer
    est_docs = int(args.num_tokens / 200 * 1.2)
    print(f"\nPhase 1: Download c4/en (~{est_docs / 1e6:.0f}M docs) with {args.workers} workers...")
    t1 = time.time()

    dataset = load_dataset(
        "allenai/c4", "en",
        split=f"train[:{est_docs}]",
        cache_dir=args.cache_dir,
        num_proc=args.workers,
        token=token,
    )
    print(f"  Downloaded {len(dataset):,} docs in {time.time() - t1:.0f}s")

    # Phase 2: Tokenize in parallel
    print(f"\nPhase 2: Tokenize with {args.workers} workers...")
    t2 = time.time()

    # Split dataset into chunks for parallel processing
    texts = dataset["text"]
    chunk_size = max(1, len(texts) // args.workers)
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]

    worker_args = [
        (i, chunk, args.tokenizer, output_dir)
        for i, chunk in enumerate(chunks)
    ]

    with Pool(args.workers) as pool:
        results = pool.map(tokenize_chunk, worker_args)
    print(f"  Tokenization done in {time.time() - t2:.0f}s")

    # Phase 3: Merge into single .bin + .idx
    print("\nPhase 3: Merge...")
    bin_path = args.output_prefix + ".bin"
    idx_path = args.output_prefix + ".idx"
    all_sizes = []
    all_doc_idx = [0]
    total_tokens = 0

    with open(bin_path, "wb") as out:
        for tmp_bin, sizes, doc_idx, n_tokens in results:
            if total_tokens >= args.num_tokens:
                os.remove(tmp_bin)
                continue
            with open(tmp_bin, "rb") as f:
                while chunk := f.read(128 * 1024 * 1024):
                    out.write(chunk)
            os.remove(tmp_bin)
            offset = len(all_sizes)
            all_sizes.extend(sizes)
            all_doc_idx.extend(idx + offset for idx in doc_idx[1:])
            total_tokens += n_tokens

    write_idx_file(idx_path, all_sizes, all_doc_idx)

    elapsed = time.time() - t0
    print(f"\n=== Done! ===")
    print(f"  {total_tokens / 1e9:.2f}B tokens, {len(all_sizes):,} documents")
    print(f"  Total time: {elapsed / 60:.1f} min")
    print(f"  {bin_path} ({os.path.getsize(bin_path) / 1e9:.1f} GB)")
    print(f"  {idx_path} ({os.path.getsize(idx_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
