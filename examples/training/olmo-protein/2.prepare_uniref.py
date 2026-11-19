"""Download UniRef50 and tokenize into OLMo memmap .npy format.

Produces:
  data/protein/train/part-{NN}-00000.npy   (uint16, flat token IDs)
  data/protein/val/part-00-00000.npy       (uint16, flat token IDs)

Each .npy file is a flat array of uint16 token IDs. Protein sequences are
concatenated with EOS (token 0) between them, matching OLMo's MemMapDataset
expectations: no special framing, just a flat stream of tokens chunked into
max_sequence_length blocks at training time.

Usage:
    python scripts/prepare_protein_data.py [--max-sequences N] [--val-fraction 0.005]

Set --max-sequences for a quick test run (e.g. --max-sequences 100000).
"""

import argparse
import gzip
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

UNIREF50_URL = "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz"
DATA_DIR = Path("data/protein")
RAW_FILE = DATA_DIR / "uniref50.fasta.gz"

CANONICAL = "ACDEFGHIKLMNPQRSTVWY"
EXTENDED = "XBZJUO"
ALL_RESIDUES = CANONICAL + EXTENDED

EOS_ID = 0
PAD_ID = 1
RESIDUE_TO_ID = {}
for i, aa in enumerate(ALL_RESIDUES):
    RESIDUE_TO_ID[aa] = i + 2  # ids 2..27
UNK_ID = RESIDUE_TO_ID["X"]  # fallback for unexpected characters

TOKENS_PER_SHARD = 50_000_000  # ~100 MB per shard at uint16


def download_uniref50():
    if RAW_FILE.exists():
        size_gb = RAW_FILE.stat().st_size / 1e9
        print(f"uniref50.fasta.gz already exists ({size_gb:.1f} GB), skipping download")
        return
    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {UNIREF50_URL} ...")
    subprocess.run(["wget", "-q", "--show-progress", "-O", str(RAW_FILE), UNIREF50_URL], check=True)
    print(f"downloaded {RAW_FILE.stat().st_size / 1e9:.1f} GB")


def parse_fasta_gz(path, max_sequences=None):
    """Yield protein sequences from a gzipped FASTA file."""
    count = 0
    seq_parts = []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if seq_parts:
                    yield "".join(seq_parts)
                    count += 1
                    if max_sequences and count >= max_sequences:
                        return
                seq_parts = []
            else:
                seq_parts.append(line.strip().upper())
    if seq_parts:
        yield "".join(seq_parts)


def tokenize_sequence(seq):
    """Convert amino acid string to list of uint16 token IDs, with trailing EOS."""
    ids = [RESIDUE_TO_ID.get(ch, UNK_ID) for ch in seq]
    ids.append(EOS_ID)
    return ids


def write_shard(tokens, path):
    arr = np.array(tokens, dtype=np.uint16)
    arr.tofile(str(path))
    print(f"  wrote {path} ({len(tokens):,} tokens, {path.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-sequences", type=int, default=None,
                        help="Limit number of sequences (for testing)")
    parser.add_argument("--val-fraction", type=float, default=0.005,
                        help="Fraction of sequences for validation")
    args = parser.parse_args()

    download_uniref50()

    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    print("tokenizing sequences...")
    train_tokens = []
    val_tokens = []
    shard_idx = 0
    total_seqs = 0
    total_train_tokens = 0
    total_val_tokens = 0

    rng = np.random.RandomState(42)

    for seq in parse_fasta_gz(RAW_FILE, max_sequences=args.max_sequences):
        ids = tokenize_sequence(seq)
        total_seqs += 1

        if rng.random() < args.val_fraction:
            val_tokens.extend(ids)
            total_val_tokens += len(ids)
        else:
            train_tokens.extend(ids)
            total_train_tokens += len(ids)

        if len(train_tokens) >= TOKENS_PER_SHARD:
            path = train_dir / f"part-{shard_idx:02d}-00000.npy"
            write_shard(train_tokens[:TOKENS_PER_SHARD], path)
            train_tokens = train_tokens[TOKENS_PER_SHARD:]
            shard_idx += 1

        if total_seqs % 1_000_000 == 0:
            print(f"  processed {total_seqs:,} sequences...")

    # Write remaining train tokens
    if train_tokens:
        path = train_dir / f"part-{shard_idx:02d}-00000.npy"
        write_shard(train_tokens, path)

    # Write val shard
    if val_tokens:
        path = val_dir / "part-00-00000.npy"
        write_shard(val_tokens, path)

    print(f"\ndone: {total_seqs:,} sequences")
    print(f"  train: {total_train_tokens:,} tokens in {shard_idx + 1} shards")
    print(f"  val:   {total_val_tokens:,} tokens in 1 shard")


if __name__ == "__main__":
    main()
