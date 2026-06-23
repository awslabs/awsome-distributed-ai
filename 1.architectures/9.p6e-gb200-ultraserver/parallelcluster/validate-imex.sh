#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Validate the P6e-GB200 IMEX / NVLink domain on Slurm. Submit with: sbatch validate-imex.sh
#SBATCH --job-name=gb200-imex-validate
#SBATCH --nodes=18
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --output %x_%j.out
#SBATCH --error  %x_%j.err

set -euo pipefail

echo "Checking IMEX domain across $SLURM_NNODES nodes..."
srun -N "$SLURM_NNODES" --ntasks-per-node=1 bash -c '
  if nvidia-imex-ctl -N | grep -q "Domain State: UP"; then
    echo "OK  $(hostname): Domain State: UP"
  else
    echo "BAD $(hostname): IMEX domain NOT up"; exit 1
  fi
'
echo "All nodes report IMEX Domain State: UP. NVLink domain is formed."
