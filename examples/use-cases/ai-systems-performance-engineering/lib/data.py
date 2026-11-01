"""Identical deterministic token records in individual files and packed binary shards."""
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class Tokens(Dataset):
    def __init__(self, root, layout, cpu_rounds=0):
        self.root = Path(root)
        self.meta = json.loads((self.root / 'manifest.json').read_text())
        self.layout = layout
        self.cpu_rounds = cpu_rounds

    def __len__(self):
        return self.meta['records']

    @functools.lru_cache(maxsize=2)
    def shard(self, index):
        return np.memmap(self.root / 'shards' / f'{index:06d}.bin', mode='r', dtype='<i4')

    def __getitem__(self, index):
        length = self.meta['sequence_tokens'] + 1
        if self.layout == 'small':
            path = self.root / 'small' / f'{index:08d}.bin'
            # Real metadata work, deliberately repeated only on the small-file path.
            for _ in range(self.meta['metadata_checks_per_record']):
                os.stat(path)
            data = np.fromfile(path, dtype='<i4')
        else:
            shard, offset = divmod(index, self.meta['records_per_shard'])
            data = self.shard(shard)[offset * length:(offset + 1) * length]
        # Deterministic CPU preprocessing is identical across all configurations.
        # The rounds parameter is a calibration knob, not a simulated wait.
        digest = data.tobytes()
        for _ in range(self.cpu_rounds):
            digest = hashlib.sha256(digest).digest()
        return torch.from_numpy(data.astype(np.int64, copy=True))


def prepare(root, records, sequence, per_shard, vocab, metadata_checks):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError(f'dataset directory must be empty: {root}')
    (root / 'small').mkdir()
    (root / 'shards').mkdir()
    rng = np.random.default_rng(347)
    digest = hashlib.sha256()
    for start in range(0, records, per_shard):
        shard = rng.integers(0, vocab, size=(min(per_shard, records-start), sequence+1), dtype=np.int32).astype('<i4')
        raw = shard.tobytes(); digest.update(raw)
        (root / 'shards' / f'{start//per_shard:06d}.bin').write_bytes(raw)
        for offset, row in enumerate(shard):
            (root / 'small' / f'{start+offset:08d}.bin').write_bytes(row.tobytes())
    metadata = dict(records=records, sequence_tokens=sequence, records_per_shard=per_shard,
                    vocabulary_tokens=vocab, metadata_checks_per_record=metadata_checks,
                    sha256=digest.hexdigest(), seed_dimensionless=347)
    (root / 'manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata))


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('root')
    p.add_argument('--records', type=int, default=32768)
    p.add_argument('--sequence', type=int, default=512)
    p.add_argument('--per-shard', type=int, default=2048)
    p.add_argument('--vocab', type=int, default=151936)
    p.add_argument('--metadata-checks', type=int, default=4)
    a=p.parse_args()
    if min(a.records,a.sequence,a.per_shard,a.vocab) < 1 or a.metadata_checks < 0:
        p.error('dimensions must be positive; metadata checks must be nonnegative')
    prepare(a.root,a.records,a.sequence,a.per_shard,a.vocab,a.metadata_checks)
