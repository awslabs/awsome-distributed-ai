#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# One coordinator invocation of the named suite's Check 5 per allocation.
set -euo pipefail
LAB_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$LAB_DIR/common.sh"
set_allocation_cpus
: "${SLURM_JOB_ID:?Run from the coordinator inside a fresh dedicated two-node allocation}"
[[ ${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-0}} == 2 ]]
: "${NCCL_ENROOT_IMAGE:?Load lab.env with the native-kernel staged image}"
suite=${AIM344_SUITE_ENTRY:-/opt/aim344-healthcheck/validation/gpu-cluster-healthcheck/gpu-healthcheck.sh}
out=${RESULTS_DIR:-$LAB_DIR/results/$SLURM_JOB_ID}/healthcheck-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$out"
export AIM344_COUNTER_DIR=$out AIM344_COUNTER_SCRIPT=$LAB_DIR/efa-counters.py
unset SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_name
unset SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_env
srun -N2 -n2 --ntasks-per-node=1 --cpus-per-task="${AIM344_CPUS_PER_NODE:?}" --mpi=none --cpu-bind=none \
 bash -c 'mkdir -p "$AIM344_COUNTER_DIR"; python3 "$AIM344_COUNTER_SCRIPT" snapshot "$AIM344_COUNTER_DIR/before"'
rc=0
run_check5 "$suite" "$out" 2>&1 | tee "$out/command.log" || rc=$?
srun -N2 -n2 --ntasks-per-node=1 --cpus-per-task="$AIM344_CPUS_PER_NODE" --mpi=none --cpu-bind=none \
 python3 "$AIM344_COUNTER_SCRIPT" delta "$out/before" | tee "$out/efa-deltas.jsonl"
printf 'Check 5 exit_code=%s (dimensionless); results=%s\n' "$rc" "$out"
exit "$rc"
