#!/usr/bin/env python3
"""Record the weights-only numerator from the staged served checkpoint."""
import argparse
import json
from pathlib import Path
from lib.serving_metrics import model_bytes
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('model', type=Path)
p.add_argument('--tensor-parallel-size', type=int, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
result = model_bytes(a.model, a.tensor_parallel_size)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({key: result[key] for key in ['weights_bytes_per_decode_step', 'checkpoint_weight_bytes', 'tensor_parallel_size', 'scope']}))
