#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Combined OnInitComplete entrypoint for SageMaker HyperPod Slurm clusters.
#
# This script orchestrates multiple standalone extensions. Enable or disable
# each feature by setting the flags below. Each feature's directory must be
# uploaded to S3 alongside this script.
#
# Expected S3 layout:
#   s3://<bucket>/<prefix>/
#   |-- run_extensions.sh          (this file -- OnInitComplete target)
#   |-- detect-node/               (node type detection utility)
#   |-- add-users/                 (user management scripts + config)
#   |-- observability/             (observability scripts + config.json)
#
# Prerequisites:
#   - AMI-based configuration (Slurm, Docker pre-installed)
#   - Each enabled feature's config file must be populated (not placeholders)
#
# Usage:
#   Specify this script as the OnInitComplete target in your cluster config.

set -ex

# ===========================================================================
# Feature toggles -- set to "true" or "false"
# ===========================================================================
ENABLE_ADD_USERS="true"
ENABLE_OBSERVABILITY="false"

# ===========================================================================
# Logging -- matches AMI on_create.sh pattern
# ===========================================================================
LOG_FILE="/var/log/provision/provisioning.log"
mkdir -p /var/log/provision
touch "$LOG_FILE"

logger() {
    echo "$@" | tee -a "$LOG_FILE"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ===========================================================================
# Run a feature extension
# ===========================================================================
run_feature() {
    local name="$1"
    local script="$2"

    if [ ! -f "$script" ]; then
        logger "[warning] $name script not found at $script, skipping"
        return 0
    fi

    logger "[start] $name"
    if ! bash "$script" >> "$LOG_FILE" 2>&1; then
        logger "[error] $name failed, waiting 60 seconds before exit, to make sure logs are uploaded"
        sync
        sleep 60
        logger "[stop] run_extensions.sh with error"
        exit 1
    fi
    logger "[stop] $name"
}

# ===========================================================================
# Main
# ===========================================================================
logger "[start] run_extensions.sh"

# Always run node detection first -- other extensions depend on nodeinfo.json
run_feature "detect-node" "$SCRIPT_DIR/detect-node/detect_node.sh"

if [ "$ENABLE_ADD_USERS" = "true" ]; then
    run_feature "add-users" "$SCRIPT_DIR/add-users/add_users.sh"
fi

if [ "$ENABLE_OBSERVABILITY" = "true" ]; then
    run_feature "observability" "$SCRIPT_DIR/observability/setup_observability.sh"
fi

logger "[stop] run_extensions.sh"
