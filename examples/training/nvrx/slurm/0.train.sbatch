#!/bin/bash
#SBATCH --job-name=nvrx
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --time=00:10:00
# Submit with ordinary sbatch; script arguments are passed to train_gpt2.py.
set -euo pipefail
: "${NVRX_RUN_DIR:?Set a new shared run directory}" "${NVRX_IMAGE:?Set the immutable Pyxis image path}"
: "${NVRX_MOUNTS:?Set Pyxis shared filesystem mounts}"

first_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
first_node=${first_node%%$'\n'*}
node_info=$(scontrol show node "$first_node" -o)
if [[ $node_info =~ NodeAddr=([^[:space:]]+) ]]; then
    export NVRX_RDZV_HOST=${BASH_REMATCH[1]}
else
    printf 'Cannot resolve Slurm NodeAddr for %s\n' "$first_node" >&2
    exit 1
fi
export NVRX_RDZV_PORT=${NVRX_RDZV_PORT:-29500}
export NVRX_GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:?}
export NVRX_MAX_RESTARTS=${NVRX_MAX_RESTARTS:-2}
export NVRX_SECTION_TIMEOUTS=${NVRX_SECTION_TIMEOUTS:-setup:180,step:60,checkpointing:120}
export NVRX_LOCAL_TMPDIR=${NVRX_LOCAL_TMPDIR:-/opt/build-tmp}
export NVRX_LAUNCHER=${NVRX_LAUNCHER:-ft_launcher}
container_env=NVRX_RUN_DIR,NVRX_RDZV_HOST,NVRX_RDZV_PORT,NVRX_GPUS_PER_NODE,NVRX_MAX_RESTARTS,NVRX_SECTION_TIMEOUTS,NVRX_LOCAL_TMPDIR,NVRX_LAUNCHER
# Pyxis requires explicitly listed variables to exist, even for normal runs.
if [[ -v NVRX_INJECT_GPU_FAILURE ]]; then
    export NVRX_INJECT_GPU_FAILURE
    container_env+=,NVRX_INJECT_GPU_FAILURE
fi
# One launcher per node, not per GPU. Pyxis must preserve the image's PATH and
# LD_LIBRARY_PATH; pass only the job settings that the launcher needs explicitly.
srun --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
    --container-image="$NVRX_IMAGE" --container-mounts="$NVRX_MOUNTS" \
    --container-workdir=/app \
    --container-env="$container_env" \
    bash /app/slurm/1.launch.sh "$@"
