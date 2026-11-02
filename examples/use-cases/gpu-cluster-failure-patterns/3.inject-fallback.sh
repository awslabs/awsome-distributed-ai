#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
set -euo pipefail
[[ $# -eq 0 ]] || { echo "Usage: bash ${0##*/} (it takes no arguments)" >&2; exit 2; }
LAB_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "$LAB_DIR/common.sh"
prepare_slurm
# Restore on success, failure, or interruption. The following recovery script
# repeats the healthy sweep and is also the manual recovery after a lost shell.
trap 'node_command bash /opt/aim344/plugin-state.sh restore' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
node_command bash /opt/aim344/plugin-state.sh hide
run_sweep fallback
