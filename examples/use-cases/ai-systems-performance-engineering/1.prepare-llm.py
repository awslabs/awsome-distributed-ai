#!/usr/bin/env python3
"""Stage a finite, attributed real-text dataset with pinned public artifacts."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'lib'))
from llm_data import write_documents


def main():
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).parent / 'configs/llm.json')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--records', type=int, default=160)
    parser.add_argument('--minimum-document-tokens', type=int, default=4096)
    parser.add_argument('--download-weights', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_id'], revision=cfg['tokenizer_revision'],
                                              token=False, trust_remote_code=False)
    documents = load_dataset(cfg['dataset_id'], name=cfg['dataset_subset'],
                             revision=cfg['dataset_revision'], split=cfg['dataset_split'],
                             streaming=True, token=False)
    provenance = {key: cfg[key] for key in ('model_id', 'tokenizer_revision', 'dataset_id',
                                           'dataset_revision', 'dataset_subset',
                                           'dataset_split', 'dataset_license')}
    provenance['attribution'] = 'FineWeb-Edu, Hugging Face, 2024; derived from Common Crawl'
    provenance['license_url'] = 'https://opendatacommons.org/licenses/by/1-0/'
    provenance['source_terms_url'] = 'https://commoncrawl.org/terms-of-use'
    result = write_documents(args.output / 'tokens', documents, tokenizer,
                             records=args.records, sequence_length=cfg['sequence_length'],
                             provenance=provenance, minimum_document_tokens=args.minimum_document_tokens)
    if args.download_weights:
        snapshot_download(cfg['model_id'], revision=cfg['model_revision'], token=False,
                          local_dir=args.output / 'model',
                          allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja',
                                          'LICENSE', 'README.md'])
        (args.output / 'model' / 'aim347-pin.json').write_text(json.dumps({
            'model_id': cfg['model_id'], 'model_revision': cfg['model_revision']}, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
