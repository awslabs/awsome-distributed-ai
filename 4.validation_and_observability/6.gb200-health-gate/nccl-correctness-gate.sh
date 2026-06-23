#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Mandatory NCCL correctness gate for P6e-GB200 NVL72 domains.
#
# Runs all_reduce_perf with the correctness check on (-c 1) across the whole domain and
# FAILS if any data row reports a nonzero #wrong. This catches fabric-level faults
# (miscabled NVSwitch port pair, degraded NVLS bind) that pass single-asset diagnostics
# but silently corrupt cross-asset-group collectives. busbw is reported but advisory.
#
# Run under mpirun/srun (the launcher provides the rank environment). Reads nccl-tests
# stdout on stdin, or runs the test itself if given no piped input.
#
# Usage:
#   srun ... all_reduce_perf -b 8 -e 16G -f 2 -g 1 -c 1 -n 100 | ./nccl-correctness-gate.sh
#   # or as a wrapper:
#   NCCL_TESTS_PATH=/opt/nccl-tests/build ./nccl-correctness-gate.sh --run
set -euo pipefail

: "${NCCL_TESTS_PATH:=/opt/nccl-tests/build}"

if [[ "${1:-}" == "--run" ]]; then
  # Self-contained run (expects to be launched under mpirun/srun already).
  input="$("$NCCL_TESTS_PATH/all_reduce_perf" -b 8 -e 16G -f 2 -g 1 -c 1 -n 100)"
else
  input="$(cat)"
fi

echo "$input"

# nccl-tests data rows look like:
#   size  count  type  redop  root  time  algbw  busbw  #wrong   (out-of-place, then in-place)
# The last numeric field on a data row is #wrong. Validate every data row.
worst="$(printf '%s\n' "$input" | awk '
  /^[[:space:]]*[0-9]+[[:space:]]+[0-9]+/ {
    w = $NF
    if (w ~ /^[0-9]+$/ && w+0 > max) max = w+0
  }
  END { print (max=="" ? 0 : max) }
')"

echo "------------------------------------------------------------"
if [[ "$worst" -gt 0 ]]; then
  echo "FAIL: NCCL correctness gate -- max #wrong = $worst (expected 0)."
  echo "      A nonzero #wrong on GB200 indicates silent cross-asset-group NVLink"
  echo "      corruption. Quarantine the domain, re-vet NVSwitch cabling / NVLS bind,"
  echo "      and do NOT run production workloads on it."
  exit 1
fi
echo "PASS: NCCL correctness gate -- #wrong = 0 across the domain."
