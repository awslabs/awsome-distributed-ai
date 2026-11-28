#!/usr/bin/env bash
set -euo pipefail
export LD_PRELOAD=/opt/nccl/build/lib/libnccl.so
exec torchrun --nnodes="${NNODES:?}" --nproc-per-node="${GPUS_PER_NODE:?}" \
    --node-rank="${SLURM_PROCID:?}" --master-addr="${MASTER_ADDR:?}" --master-port="${MASTER_PORT:?}" \
    lib/train_llm.py "$@"
