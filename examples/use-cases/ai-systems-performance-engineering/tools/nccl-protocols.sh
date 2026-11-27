#!/usr/bin/env bash
# Optional: all_reduce_perf over EFA with OFI_NCCL_PROTOCOL=RDMA, 1 MiB to 64 MiB, on both
# nodes, once with NCCL's own protocol choice and once with NCCL_PROTO=Simple. salloc holds
# the nodes for the two runs. Usage: bash tools/nccl-protocols.sh   (see tools/README.md)
set -euo pipefail
lab=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
[[ $# == 0 ]] || { echo 'Usage: bash tools/nccl-protocols.sh' >&2; exit 2; }
# shellcheck source=tools/common.sh
source "$lab/tools/common.sh"
exec salloc --job-name=aim347-nccl "${two_nodes[@]}" python3 "$lab/lib/workshop.py" nccl-protocols
