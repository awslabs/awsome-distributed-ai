"""Require a homogeneous GPU allocation before constructing distributed ranks."""
import argparse
import json
from pathlib import Path


def plan(rows, nodes):
    if len(rows) != nodes or len({r['host'] for r in rows}) != nodes:
        raise ValueError('Expected exactly one inventory per allocated node')
    counts = {r['gpus'] for r in rows}
    if len(counts) != 1 or next(iter(counts)) < 1:
        raise ValueError('torchrun requires a homogeneous positive GPU count per node')
    gpus = next(iter(counts))
    return dict(nnodes=nodes, gpus_per_node=gpus, gpu_count=nodes*gpus, nodes=rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('inventory', type=Path)
    p.add_argument('--nodes', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = plan([json.loads(line) for line in a.inventory.read_text().splitlines() if line.strip()], a.nodes)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['nnodes'], result['gpus_per_node'], result['gpu_count'])
