"""Pretokenized documents with stable update membership and padding masks."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from llm_metrics import accumulation_steps


def write_documents(root, documents, tokenizer, *, records, sequence_length, provenance,
                    minimum_document_tokens=0):
    """Prepare once. Documents never share an attention context."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError('prepared data directory must be empty')
    if records < 1 or sequence_length < 2:
        raise ValueError('positive records and at least two positions required')
    if not 0 <= minimum_document_tokens <= sequence_length:
        raise ValueError('minimum document tokens must fit sequence length')
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    digest = hashlib.sha256()
    lengths = []
    with (root / 'tokens.bin').open('wb') as output, (root / 'sources.jsonl').open('w') as sources:
        for document in documents:
            ids = tokenizer.encode(document['text'], add_special_tokens=False)
            if not ids:
                continue
            if len(ids) < minimum_document_tokens:
                continue
            original_tokens = len(ids)
            ids = (ids + [tokenizer.eos_token_id])[:sequence_length]
            if len(ids) < 2:
                continue
            row = np.full(sequence_length, pad, dtype='<i4')
            row[:len(ids)] = ids
            raw = row.tobytes()
            digest.update(raw)
            output.write(raw)
            lengths.append(len(ids))
            sources.write(json.dumps({'record': len(lengths) - 1,
                                      'original_tokens': original_tokens,
                                      'text_sha256': hashlib.sha256(document['text'].encode()).hexdigest(),
                                      'url': document.get('url'),
                                      'id': document.get('id')}) + '\n')
            if len(lengths) == records:
                break
    if len(lengths) != records:
        raise ValueError('source exhausted before the requested record count')
    raw_lengths = np.asarray(lengths, dtype='<i4').tobytes()
    (root / 'lengths.bin').write_bytes(raw_lengths)
    digest.update(raw_lengths)
    manifest = dict(provenance, records=records, sequence_length=sequence_length,
                    minimum_document_tokens=minimum_document_tokens,
                    selection='first eligible documents in pinned source order',
                    sha256=digest.hexdigest(), pad_token_id=pad,
                    eos_token_id=tokenizer.eos_token_id, packing='none',
                    positions='zero-based per document',
                    truncation='right; EOS only when original document end fits',
                    useful_tokens=sum(length - 1 for length in lengths))
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


class Documents(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        shape = (self.manifest['records'], self.manifest['sequence_length'])
        if (self.root / 'tokens.bin').stat().st_size != shape[0] * shape[1] * 4:
            raise ValueError('token file size does not match manifest')
        if (self.root / 'lengths.bin').stat().st_size != shape[0] * 4:
            raise ValueError('length file size does not match manifest')
        digest = hashlib.sha256()
        for name in ('tokens.bin', 'lengths.bin'):
            with (self.root / name).open('rb') as source:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    digest.update(block)
        if digest.hexdigest() != self.manifest['sha256']:
            raise ValueError('prepared data fingerprint mismatch')
        self.tokens = np.memmap(self.root / 'tokens.bin', dtype='<i4', mode='r', shape=shape)
        self.lengths = np.memmap(self.root / 'lengths.bin', dtype='<i4', mode='r')
        if np.any(self.lengths < 2) or np.any(self.lengths > shape[1]):
            raise ValueError('invalid document lengths')

    def __len__(self):
        return self.manifest['records']

    def __getitem__(self, index):
        ids = torch.from_numpy(self.tokens[index].astype(np.int64, copy=True))
        mask = torch.arange(ids.numel()) < int(self.lengths[index])
        labels = ids.clone()
        labels[~mask] = -100
        return {'input_ids': ids, 'attention_mask': mask.long(), 'labels': labels,
                'position_ids': torch.arange(ids.numel())}

    def update_tokens(self, update, global_batch):
        start = update * global_batch
        end = start + global_batch
        if end > len(self):
            raise ValueError('not enough documents for fixed update membership')
        return int((self.lengths[start:end] - 1).sum())


class UpdateBatches(Sampler):
    """The same rank sees the same samples when microbatch/accumulation changes."""

    def __init__(self, *, start_update, stop_update, global_batch, microbatch, rank, dp_size):
        self.accumulation = accumulation_steps(global_batch, microbatch, dp_size)
        if not 0 <= rank < dp_size or not 0 <= start_update < stop_update:
            raise ValueError('invalid rank or update interval')
        self.start = start_update
        self.stop = stop_update
        self.global_batch = global_batch
        self.microbatch = microbatch
        self.rank = rank
        self.dp_size = dp_size

    def __iter__(self):
        for update in range(self.start, self.stop):
            indices = list(range(update * self.global_batch + self.rank,
                                 (update + 1) * self.global_batch, self.dp_size))
            for offset in range(0, len(indices), self.microbatch):
                yield indices[offset:offset + self.microbatch]

    def __len__(self):
        return (self.stop - self.start) * self.accumulation


def trim_padding(batch):
    """Remove only all-padding suffix columns of a right-padded CPU batch.

    Document membership, positions and shifted loss labels remain unchanged.
    This changes executed tensor shapes, not the useful-token denominator.
    """
    width = max(2, int(batch['attention_mask'].sum(dim=1).max()))
    return {key: value[:, :width].contiguous() for key, value in batch.items()}
