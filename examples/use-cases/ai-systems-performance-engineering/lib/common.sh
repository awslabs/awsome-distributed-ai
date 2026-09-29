#!/usr/bin/env bash
set -euo pipefail
LAB_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export LAB_DIR
# Caller supplies the generated Smol environment; never source a local .env.
