#!/usr/bin/env python3
"""Combine model bytes, observed decode intervals, and measured GPU bandwidth."""
import argparse
import json
from pathlib import Path
from lib.serving_metrics import compute_mbu
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('serving', type=Path)
p.add_argument('--model-bytes', type=Path, required=True)
p.add_argument('--bandwidth-map', type=Path, required=True, help='JSON map from endpoint URL to its GPU measurement paths, relative to this map')
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
paths = json.loads(a.bandwidth_map.read_text())
samples = {endpoint: [json.loads((a.bandwidth_map.parent/path).read_text()) for path in files] for endpoint, files in paths.items()}
result = compute_mbu(json.loads(a.serving.read_text()), json.loads(a.model_bytes.read_text()), samples)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({key: result[key] for key in ['mbu_weights_only_ratio', 'modeled_weight_read_bytes', 'measured_decode_capacity_bytes', 'output_tokens_per_second', 'median_ttft_seconds', 'mean_server_itl_seconds']}, indent=2))
