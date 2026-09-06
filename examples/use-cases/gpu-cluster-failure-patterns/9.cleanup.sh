#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Run before leaving the dedicated allocation. Retains measured result files.
set -euo pipefail
: "${SLURM_JOB_ID:?Run before exiting salloc}"
# Expand the Slurm job ID on each compute node, not on the login node.
# shellcheck disable=SC2016
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --mpi=none bash -c '
set -euo pipefail
for label in "aim344_$SLURM_JOB_ID" "aim344_torch_$SLURM_JOB_ID"; do
    # Pyxis supports either user or job container scope.
    for name in "pyxis_$label" "pyxis_${SLURM_JOB_ID}_$label"; do
        if enroot list | grep -Fxq "$name"; then
            enroot remove -f "$name"
        fi
    done
done
'
