#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Entrypoint for the QLoRA training container.
#
# When the Kubeflow Training Operator creates a PyTorchJob with
# nprocPerNode > 1, it sets PET_NPROC_PER_NODE (and related PET_*
# env vars) but does NOT invoke torchrun itself.  This entrypoint
# bridges the gap: it reads the PET env vars and launches the
# training script via torchrun for multi-GPU, or runs it directly
# for single-GPU / local dev.

set -euo pipefail

NPROC="${PET_NPROC_PER_NODE:-1}"
NNODES="${PET_NNODES:-1}"
NODE_RANK="${PET_NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [ "$NPROC" -gt 1 ] || [ "$NNODES" -gt 1 ]; then
    echo "Launching distributed training: torchrun"
    echo "  nproc_per_node=$NPROC  nnodes=$NNODES  node_rank=$NODE_RANK"
    echo "  master=$MASTER_ADDR:$MASTER_PORT"
    exec torchrun \
        --nproc_per_node="$NPROC" \
        --nnodes="$NNODES" \
        --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        "$@"
else
    echo "Single-process mode (no PET_NPROC_PER_NODE or == 1)"
    exec python -u "$@"
fi
