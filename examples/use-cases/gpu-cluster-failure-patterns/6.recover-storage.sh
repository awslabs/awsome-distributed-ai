#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
set -euo pipefail
# The cause-told form of the storage exercise. The only option is the rank that
# writes the checkpoint; anything else is refused before the allocation is used.
if [[ $# -ne 0 ]] && ! [[ $# -eq 2 && $1 == --writer-rank && $2 =~ ^[0-9]{1,4}$ ]]; then
    echo "Usage: bash ${0##*/} [--writer-rank N]" >&2
    exit 2
fi
LAB_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "$LAB_DIR/common.sh"
prepare_slurm
prepare_torch
node_command python3 /opt/aim344/recover-checkpoint.py
run_torch storage-recovered storage --resume "$@"
