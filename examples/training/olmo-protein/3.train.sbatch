#!/bin/bash
#SBATCH --job-name=olmo-protein
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --output=%x-%j.log
#SBATCH --exclusive

# OLMo/OLMoE protein pretraining — Slurm (HyperPod or ParallelCluster).
# MoE vs dense is purely a config swap: pass a different $CONFIG.
#
# Required env (export before sbatch, or edit here):
#   DATA_HOME   shared-FS root holding config/, data/, checkpoints/  (default /fsx/olmo-protein)
#   CONFIG      training config YAML            (default $DATA_HOME/config/OLMoE-protein.yml)
#   RUN_NAME    checkpoint/metrics run name     (default protein-moe-v1)
# Optional:
#   MLFLOW_TRACKING_URI   e.g. a SageMaker managed MLflow ARN — enables metric mirroring
#   GPUS_PER_NODE         default 4 (set 8 on p5/p4d)

set -euxo pipefail

DATA_HOME=${DATA_HOME:-/fsx/olmo-protein}
CONFIG=${CONFIG:-$DATA_HOME/config/OLMoE-protein.yml}
RUN_NAME=${RUN_NAME:-protein-moe-v1}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}

mkdir -p "$DATA_HOME/checkpoints" "$DATA_HOME/logs"
cd /opt/OLMo

# Lessons encoded from a real two-dataset pretraining campaign:
#  --save_interval_unsharded: resume MUST use unsharded checkpoints — sharded
#    layout is tied to GPU count, and sharded resume OOMs on this branch.
#  step-based max_duration in the config (not epochs): a cross-instance resume
#  undercounts epochs and can exit rc=0 before training is done.
torchrun \
  --nproc_per_node=$GPUS_PER_NODE \
  --nnodes=$SLURM_JOB_NUM_NODES \
  --node_rank=$SLURM_NODEID \
  --master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1) \
  --master_port=29500 \
  scripts/train.py \
  "$CONFIG" \
  --run_name="$RUN_NAME" \
  --save_folder="$DATA_HOME/checkpoints/$RUN_NAME" \
  --device_train_microbatch_size=4 \
  --model.max_sequence_length=2048 \
  --activation_checkpointing=fine_grained \
  --save_interval=200 \
  --save_interval_unsharded=1000 \
  --save_overwrite
