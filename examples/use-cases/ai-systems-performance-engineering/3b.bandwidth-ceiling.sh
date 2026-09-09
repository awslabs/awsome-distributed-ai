#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/lib/common.sh"
submit bandwidth
python3 - "$LAB_DIR/results/$RUN_ID" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
nodes = json.loads((root/'bandwidth-resources.json').read_text())['nodes']
mapping = {}
for node in nodes:
    rank = node['slurm_node_rank']
    if rank is None:
        raise ValueError('Slurm node rank is required to map measured GPUs to serving replicas')
    paths = [f'dram-node-{rank}-gpu-{gpu}.json' for gpu in range(node['gpus'])]
    samples = [json.loads((root/path).read_text()) for path in paths]
    mapping[f"http://{node['host']}:8000"] = paths
    print(json.dumps(dict(host=node['host'], gpu_count=node['gpus'], measured_dram_bytes_per_second=sum(s['median_bytes_per_second'] for s in samples))))
(root/'bandwidth-map.json').write_text(json.dumps(mapping, indent=2)+'\n')
PY
