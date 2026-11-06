#!/usr/bin/env bash
set -euo pipefail
hostname
printf 'SLURM_JOB_ID=%s SLURM_GPUS_ON_NODE=%s SLURM_CPUS_ON_NODE=%s CUDA_VISIBLE_DEVICES=%s FI_EFA_IFACE=%s\n' "$SLURM_JOB_ID" "${SLURM_GPUS_ON_NODE:-}" "${SLURM_CPUS_ON_NODE:-}" "${CUDA_VISIBLE_DEVICES:-}" "${FI_EFA_IFACE:-}"
nvidia-smi -L
nvidia-smi --query-gpu=index,uuid,memory.total,driver_version --format=csv
nproc
cat /proc/self/cgroup
python3 - <<'PY'
from pathlib import Path
p=Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().split('::',1)[1].strip().lstrip('/')
while p.is_relative_to('/sys/fs/cgroup'):
 for name in ['cpuset.cpus.effective','memory.max','cpu.max']:
  f=p/name
  if f.exists(): print(str(f),f.read_text().strip())
 if p==Path('/sys/fs/cgroup'): break
 p=p.parent
PY

if [[ ${AIM_ENFORCE_STANDIN:-0} == 1 ]]; then
    python3 - <<'PY_CHECK'
import os, pathlib, subprocess
assert int(os.environ['SLURM_GPUS_ON_NODE']) == 4
assert int(subprocess.check_output(['nproc'])) <= 96
assert len([line for line in subprocess.check_output(['nvidia-smi','-L'],text=True).splitlines() if line.startswith('GPU ')]) == 4
path = pathlib.Path('/sys/fs/cgroup') / pathlib.Path('/proc/self/cgroup').read_text().split('::',1)[1].strip().lstrip('/')
limits=[]
while path.is_relative_to('/sys/fs/cgroup'):
    f=path/'memory.max'
    if f.exists() and f.read_text().strip() != 'max': limits.append(int(f.read_text()))
    if path == pathlib.Path('/sys/fs/cgroup'): break
    path=path.parent
assert limits and min(limits) <= 384 * 1024**3, limits
print('PASS: 4 accessible GPUs, at most 96 CPU cores, and at most 384 GiB memory')
PY_CHECK
fi
