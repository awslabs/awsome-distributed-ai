#!/usr/bin/env bash
set -euo pipefail
if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    count=$SLURM_GPUS_ON_NODE
    source_name=SLURM_GPUS_ON_NODE
else
    inventory=$(nvidia-smi -L)
    count=$(awk '/^GPU [0-9]+:/ {n++} END {print n+0}' <<< "$inventory")
    source_name=nvidia-smi
fi
[[ "$count" =~ ^[1-9][0-9]*$ ]] || { echo 'No positive allocated GPU count' >&2; exit 2; }
source "$(dirname -- "${BASH_SOURCE[0]}")/instance-type.sh"
instance_type=$(detect_instance_type)
efas=$(find /sys/class/infiniband -mindepth 1 -maxdepth 1 -name 'rdmap*' | wc -l)
cores=$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)
python3 - "$count" "$source_name" "$cores" "${SLURM_CPUS_ON_NODE:-}" "$instance_type" "$efas" <<'PY'
import json, os, socket, sys
gpus, source, cores, slurm_cpus, instance_type, efas = sys.argv[1:]
print(json.dumps(dict(host=socket.gethostname(), gpus=int(gpus), gpu_count_source=source,
                     instance_type=instance_type, efa_devices=int(efas), available_processors=int(cores), slurm_cpus_on_node=slurm_cpus,
                     slurm_node_rank=int(os.environ['SLURM_PROCID']) if 'SLURM_PROCID' in os.environ else None)))
PY
