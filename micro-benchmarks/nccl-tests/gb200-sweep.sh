#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Two-tier NCCL collective sweep for P6e-GB200. Run INSIDE the container, under mpirun
# (the launcher provides the rank environment). Sweeps four collectives across message
# sizes for two scenarios:
#
#   intra  : <=72 GPUs, one NVLink domain. Compares NCCL_NVLS_ENABLE=1 vs 0 to quantify
#            what NVSwitch NVLS (NVLink SHARP) buys when the collective never leaves NVLink.
#   cross  : 144 GPUs, two domains over EFA. Compares the AWS tuner default vs explicit
#            NCCL_ALGO in {NVLSTree, Ring, Tree} to show the EFA-leg ceiling without SHARP.
#
# Usage (inside the launcher container):
#   SCENARIO=intra ./gb200-sweep.sh        # default
#   SCENARIO=cross ./gb200-sweep.sh
#
# This is the measurement that distinguishes "NVLS preserved" from "SHARP lost" on AWS
# GB200 -- a number AWS publishes nowhere. Results print the nccl-tests table; feed the
# busbw column to nccl_to_csv.py for plotting.

set -euo pipefail

: "${SCENARIO:=intra}"
: "${NCCL_TESTS_PATH:=/opt/nccl-tests/build}"
: "${SIZE_MIN:=8}"
: "${SIZE_MAX:=16G}"
: "${ITERS:=100}"

COLLECTIVES=(all_reduce_perf all_gather_perf reduce_scatter_perf alltoall_perf)

run_one() {  # $1=collective binary  $2=label  (extra NCCL env exported by caller)
  echo "=================================================================="
  echo "[$SCENARIO] $2 :: $1  (NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-unset} NCCL_ALGO=${NCCL_ALGO:-tuner})"
  echo "=================================================================="
  "$NCCL_TESTS_PATH/$1" -b "$SIZE_MIN" -e "$SIZE_MAX" -f 2 -g 1 -c 1 -n "$ITERS"
}

case "$SCENARIO" in
  intra)
    for c in "${COLLECTIVES[@]}"; do
      NCCL_NVLS_ENABLE=1 unset NCCL_ALGO 2>/dev/null || true
      export NCCL_NVLS_ENABLE=1; unset NCCL_ALGO || true;  run_one "$c" "NVLS on"
      export NCCL_NVLS_ENABLE=0;                           run_one "$c" "NVLS off"
    done
    ;;
  cross)
    for c in "${COLLECTIVES[@]}"; do
      export NCCL_NVLS_ENABLE=1
      unset NCCL_ALGO || true;          run_one "$c" "tuner default"
      export NCCL_ALGO=NVLSTree;        run_one "$c" "NVLSTree (NVLS intra + Tree over EFA)"
      export NCCL_ALGO=Ring;            run_one "$c" "Ring"
      export NCCL_ALGO=Tree;            run_one "$c" "Tree"
      unset NCCL_ALGO || true
    done
    ;;
  *)
    echo "Unknown SCENARIO='$SCENARIO' (expected: intra | cross)" >&2; exit 2;;
esac
