#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/lib/common.sh"
# Run on the appropriate host; only this lab's resources are addressed.
exec "$LAB_DIR/0.deploy-observability.sh" "down-${1:?Specify login or compute}"
