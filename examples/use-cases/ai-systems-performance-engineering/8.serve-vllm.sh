#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/lib/common.sh"
: "${VLLM_IMAGE:?Set VLLM_IMAGE}"; export VLLM_IMAGE
submit serving
