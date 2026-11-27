#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --job-name=olmo-protein
#SBATCH --exclusive
#SBATCH --wait-all-nodes=1
#SBATCH --nodes=1
#SBATCH --output=logs/%x_%j.out

# OLMo-core protein pretraining — containerized Slurm launch (HyperPod or
# ParallelCluster with enroot/pyxis). Runs OLMo-core's own LLM training
# example with the protein overrides from ../protein_overrides.sh.
#
# Env (export before sbatch, or edit here):
#   ARCH            moe | dense                     (default moe)
#   DATA_ROOT       shards from 1.prepare_uniref.py (default /fsx/olmo-protein/data/protein)
#   SAVE_ROOT       checkpoint root                 (default /fsx/olmo-protein/checkpoints)
#   RUN_NAME        run name                        (default protein-v1)
#   IMAGE           enroot squashfs of the 0.Dockerfile image
#                   (default ./olmo-protein-core.sqsh; see README for `enroot import`)
#   NPROC_PER_NODE  GPUs per node (default 4; submit with matching --gres if your
#                   cluster requires it, e.g. sbatch --gres=gpu:8 for p5/p4d)
#   RECIPE_DIR      directory containing protein_overrides.sh (default: parent of
#                   the submission directory's slurm/ — Slurm runs a spooled COPY
#                   of this script, so $0 cannot locate the recipe)

set -euo pipefail

export LOGLEVEL=INFO
# Jobs are --exclusive (whole node), so let enroot expose all host GPUs to the
# container. If your cluster allocates GPUs via Slurm gres, you can instead
# `export NVIDIA_VISIBLE_DEVICES=void` and submit with --gres so only the
# allocated devices are injected.

ARCH=${ARCH:-moe}
SAVE_ROOT=${SAVE_ROOT:-/fsx/olmo-protein/checkpoints}
RUN_NAME=${RUN_NAME:-protein-v1}
IMAGE=${IMAGE:-${PWD}/olmo-protein-core.sqsh}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
OLMO_CORE=/opt/OLMo-core   # path inside the container

# Model + data overrides (defines COMMON, MOE, DENSE; reads DATA_ROOT).
# Sourced on the host so README single-node runs and this script share one file.
# $SLURM_SUBMIT_DIR, not $0: Slurm executes a spooled copy of this script, so
# paths relative to $0 point into /var/spool. Submit from the slurm/ directory
# or set RECIPE_DIR explicitly.
RECIPE_DIR=${RECIPE_DIR:-${SLURM_SUBMIT_DIR}/..}
source "$RECIPE_DIR/protein_overrides.sh"

case "$ARCH" in
  moe)   ARCH_ARGS=("${MOE[@]}") ;;
  dense) ARCH_ARGS=("${DENSE[@]}") ;;
  *) echo "ARCH must be moe or dense, got '$ARCH'" >&2; exit 1 ;;
esac

# Mount sources must exist before the container starts — enroot does not
# create them and fails the job with "failed to mount ... No such file".
mkdir -p "$SAVE_ROOT"

declare -a CONTAINER_ARGS=(
    --container-image "$IMAGE"
    --container-mounts "$DATA_ROOT:$DATA_ROOT,$SAVE_ROOT:$SAVE_ROOT"
)

declare -a TORCHRUN_ARGS=(
    --nproc_per_node="$NPROC_PER_NODE"
    --nnodes=$SLURM_JOB_NUM_NODES
    --rdzv_id=$SLURM_JOB_ID
    --rdzv_backend=c10d
    --rdzv_endpoint=$(hostname)
)

AUTO_RESUME=""
if [ -d "/opt/sagemaker_cluster" ]; then
    echo "Detected Hyperpod cluster.. enabling --auto-resume=1"
    AUTO_RESUME="--auto-resume=1"
fi

# Each arm gets its own save folder — pointing both arms at one folder makes
# resume load the other arm's checkpoint and fail on state_dict keys.
srun ${AUTO_RESUME} -l "${CONTAINER_ARGS[@]}" torchrun \
    "${TORCHRUN_ARGS[@]}" \
    "$OLMO_CORE/src/examples/llm/train.py" "$RUN_NAME-$ARCH" \
    --save-folder="$SAVE_ROOT/$RUN_NAME-$ARCH" \
    "${ARCH_ARGS[@]}" "${COMMON[@]}"
