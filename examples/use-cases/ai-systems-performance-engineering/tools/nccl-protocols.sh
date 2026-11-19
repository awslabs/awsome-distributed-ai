#!/usr/bin/env bash
# Optional: all_reduce_perf over EFA with OFI_NCCL_PROTOCOL=RDMA, 1 MiB to 64 MiB, on
# the session allocation, once with NCCL's own protocol choice and once with
# NCCL_PROTO=Simple. Usage: bash tools/nccl-protocols.sh   (see tools/README.md)
set -euo pipefail
lab=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
[[ $# == 0 ]] || { echo 'Usage: bash tools/nccl-protocols.sh' >&2; exit 2; }
# shellcheck source=tools/common.sh
source "$lab/tools/common.sh"
aim347_require_allocation
exec python3 "$lab/lib/workshop.py" nccl-protocols
