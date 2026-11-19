#!/bin/bash
#SBATCH --job-name=olmo-protein
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --output=%x-%j.log
#SBATCH --exclusive

# OLMo-core protein pretraining — Slurm (HyperPod or ParallelCluster).
# MoE vs dense twin is a CLI flag on the same script: --arch moe|dense
#
# Required env (export before sbatch, or edit here):
#   DATA_ROOT   dir holding tokenized shards from 2.prepare_uniref.py
#               (default /fsx/olmo-protein/data/protein)
#   SAVE_ROOT   checkpoint root (default /fsx/olmo-protein/checkpoints)
#   RUN_NAME    run name (default protein-v1)
#   ARCH        moe | dense (default moe)
# Optional:
#   GPUS_PER_NODE  default 4 (set 8 on p5/p4d)

set -euxo pipefail

export DATA_ROOT=${DATA_ROOT:-/fsx/olmo-protein/data/protein}
export SAVE_ROOT=${SAVE_ROOT:-/fsx/olmo-protein/checkpoints}
RUN_NAME=${RUN_NAME:-protein-v1}
ARCH=${ARCH:-moe}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}

cd /opt/olmo-protein

torchrun \
  --nproc_per_node=$GPUS_PER_NODE \
  --nnodes=$SLURM_JOB_NUM_NODES \
  --node_rank=$SLURM_NODEID \
  --master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1) \
  --master_port=29500 \
  3.train.py "$RUN_NAME" --arch "$ARCH"
