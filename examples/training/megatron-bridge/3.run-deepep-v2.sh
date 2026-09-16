#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Explicit Kimi-K2 dev-image launcher. RENDER_ONLY=1 performs no cluster actions.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ARM="${1:-deepepv2}"
case "$ARM" in
  alltoall|deepepv2) ;;
  *) echo 'Expected alltoall or deepepv2' >&2; exit 2 ;;
esac
if [ "${STORAGE:-pvc}" != pvc ]; then
  echo "This dev validation requires FSx PVC storage; hostPath is not permitted" >&2
  exit 2
fi
export STORAGE=pvc
export MODEL=kimi-k2 EP_BACKEND=deepepv2
export BENCH_PY="${BENCH_PY:-/opt/benchmark/bench_kimi_k2_pretrain.py}"
export MOE_A2A_OVERLAP="${MOE_A2A_OVERLAP:-off}"
if [ "$MOE_A2A_OVERLAP" != off ]; then
  echo 'This dev comparison requires MOE_A2A_OVERLAP=off (upstream should_free_input limitation)' >&2
  exit 2
fi
export GDRCOPY_DEV=on
export TORCHRUN_LOGS=on
# NCCL 2.31 symmetric GIN collectives require signals the EFA plugin lacks.
# DeepEP keeps its own GIN type 5 path; no transport backend is changed.
export NCCL_SYM_GIN_KERNELS_ENABLE="${NCCL_SYM_GIN_KERNELS_ENABLE:-0}"
export ARM_LABEL="${ARM_LABEL:-dev-${ARM}}"
exec bash "$HERE/run-ab-rawpods.sh" "$ARM" "${2:-32}"
