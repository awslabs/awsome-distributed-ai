#!/usr/bin/env bash
# Run one node launcher per Slurm node; replicas split only assigned GPUs.
set -euo pipefail
lab=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${SLURM_JOB_ID:?}" "${SLURM_JOB_NUM_NODES:?}" "${SLURM_JOB_CPUS_PER_NODE:?}"
cpus_per_node=$(PYTHONPATH="$lab/lib" python3 -c 'import os; from serve_llm import allocation_cpus; print(allocation_cpus(os.environ["SLURM_JOB_CPUS_PER_NODE"], int(os.environ["SLURM_JOB_NUM_NODES"])))')
: "${VLLM_IMAGE:?}" "${LLM_DATA_DIR:?}" "${LLM_RESULTS_DIR:?}"
: "${GPUS_PER_NODE:?}" "${SERVING_RUN:?}" "${NCCL_SOCKET_IFNAME:?}"
[[ "$SERVING_RUN" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'invalid serving run name' >&2; exit 2; }
export SERVING_RUN GPUS_PER_NODE
# Container-side variables intentionally expand in the child shell.
# shellcheck disable=SC2016
exec srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
    --cpus-per-task="$cpus_per_node" --mpi=none --cpu-bind=none \
    --container-image="$VLLM_IMAGE" \
    --container-mounts="$lab:/opt/aim347,$LLM_DATA_DIR:/data,$LLM_RESULTS_DIR:/results" \
    --container-workdir=/opt/aim347 --no-container-remap-root \
    --container-env=NCCL_SOCKET_IFNAME,SERVING_RUN,GPUS_PER_NODE \
    bash -c 'exec python3 lib/serve_llm.py --gpus-per-node="$GPUS_PER_NODE" --output="/results/$SERVING_RUN/node-$SLURM_PROCID" "$@"' bash "$@"
