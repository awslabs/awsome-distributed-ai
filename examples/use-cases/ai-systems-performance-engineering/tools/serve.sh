#!/usr/bin/env bash
# Optional: serve the model with vLLM, one server per GPU on both nodes, and measure its
# decode rate per GPU against the memory-read ceiling. salloc holds the nodes while the
# tool runs and frees them when it ends. Usage: bash tools/serve.sh   (see tools/README.md)
set -euo pipefail
lab=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
[[ $# == 0 ]] || { echo 'Usage: bash tools/serve.sh' >&2; exit 2; }
# shellcheck source=tools/common.sh
source "$lab/tools/common.sh"
exec salloc --job-name=aim347-serve "${two_nodes[@]}" python3 "$lab/lib/serving_decode.py" run
