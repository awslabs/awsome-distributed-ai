#!/usr/bin/env bash
# Optional: serve the model with vLLM, one server per GPU of the session allocation, and
# measure its decode rate per GPU against the memory-read ceiling. The servers stop when
# the tool ends. Usage: bash tools/serve.sh   (see tools/README.md)
set -euo pipefail
lab=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
[[ $# == 0 ]] || { echo 'Usage: bash tools/serve.sh' >&2; exit 2; }
# shellcheck source=tools/common.sh
source "$lab/tools/common.sh"
aim347_require_allocation
exec python3 "$lab/lib/serving_decode.py" run
