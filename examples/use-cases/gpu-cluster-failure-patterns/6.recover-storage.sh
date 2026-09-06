#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
set -euo pipefail
LAB_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "$LAB_DIR/common.sh"
prepare_slurm
prepare_torch
node_command python3 /opt/aim344/recover-checkpoint.py
run_torch storage-recovered storage --resume
