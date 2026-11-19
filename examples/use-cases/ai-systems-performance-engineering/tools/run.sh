#!/usr/bin/env bash
# Run one stage of the training job on the session allocation, or follow it.
# Usage: bash tools/run.sh <subcommand>   (see tools/README.md)
set -euo pipefail
lab=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=tools/common.sh
source "$lab/tools/common.sh"
exec python3 "$lab/lib/workshop.py" run "$@"
