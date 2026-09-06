#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
set -euo pipefail
LAB_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "$LAB_DIR/common.sh"
prepare_slurm
export FI_EFA_USE_HUGE_PAGE=0
run_torch dataloader-spawn dataloader --start-method spawn
