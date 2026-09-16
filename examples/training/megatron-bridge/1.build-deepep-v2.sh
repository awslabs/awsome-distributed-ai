#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Local build only. Does not log in, push, or create cloud resources.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
exec docker build --progress=plain \
  -f "$HERE/Dockerfile.deepep-v2" \
  --build-arg MAX_JOBS="${MAX_JOBS:-4}" \
  --build-arg TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0;10.0;10.3}" \
  -t "${IMAGE:-megatron-bridge:dev-bb5dfd0-pr5153-init-deepepv2-874779c}" "$@" "$ROOT"
