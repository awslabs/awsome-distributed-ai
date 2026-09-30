"""Collect small per-node ceiling measurements into the batch node's results."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
code = """import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
print(json.dumps({p.name: p.read_text() for pattern in ['gemm-node-*.json', 'dram-node-*.json', 'gpu-*.txt'] for p in root.glob(pattern)}))
"""
for node in subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']], text=True).split():
    raw = subprocess.check_output(['srun', '--nodes=1', '--ntasks=1', '--ntasks-per-node=1',
                                   '--mpi=none', '--cpu-bind=none', '--nodelist=' + node,
                                   'python3', '-c', code, str(root)], text=True)
    for name, value in json.loads(raw).items():
        if Path(name).name != name:
            raise ValueError('Invalid measurement filename')
        target = root / name
        if target.exists() and target.read_text() != value:
            raise ValueError(f'Conflicting node measurement: {name}')
        target.write_text(value)
