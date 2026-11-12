#!/usr/bin/env bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=00:05:00
set -euo pipefail
: "${LAB_DIR:?Export the shared companion asset directory before submission}"
# Expand LAB_DIR on each compute node inside its allocation.
# shellcheck disable=SC2016
srun --cpu-bind=none bash -c '
set -euo pipefail
hostname
docker version
docker compose version
nvidia-ctk --version
docker info --format "{{json .Runtimes}}"
command -v lctl
nvidia-smi -L
bash "$LAB_DIR/lib/resources.sh"
'
