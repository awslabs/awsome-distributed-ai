#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Registers users in Slurm accounting so they can submit jobs.
# Runs only on the controller node. Waits for slurmdbd to be ready.
#
# Usage: setup_slurm_accounts.sh <users_file> <slurm_account>

set -e

USERS_FILE="$1"
SLURM_ACCOUNT="${2:-root}"

if [[ -z "$USERS_FILE" ]]; then
    echo "Usage: setup_slurm_accounts.sh <users_file> [slurm_account]"
    exit 1
fi

wait_for_slurmdbd() {
    local max_attempts=30
    local attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if systemctl is-active --quiet slurmdbd; then
            echo "[INFO] slurmdbd is active"
            sleep 3
            return 0
        fi
        echo "[INFO] Waiting for slurmdbd... (attempt $((attempt+1))/$max_attempts)"
        sleep 2
        attempt=$((attempt + 1))
    done
    echo "[ERROR] slurmdbd failed to start within timeout"
    return 1
}

echo "[INFO] Setting up Slurm accounting associations"

wait_for_slurmdbd

sacctmgr -i add account "$SLURM_ACCOUNT" Description="$SLURM_ACCOUNT account" 2>/dev/null || true
echo "[INFO] Ensured Slurm account '$SLURM_ACCOUNT' exists"

while IFS="," read -r username uid; do
    username=$(echo "$username" | xargs)
    [[ -z "$username" ]] && continue

    if ! id -u "$username" >/dev/null 2>&1; then
        echo "[WARN] User $username does not exist on this node. Skipping."
        continue
    fi

    sacctmgr -i add user "$username" account="$SLURM_ACCOUNT" 2>/dev/null || true
    echo "[INFO] Added $username to Slurm account '$SLURM_ACCOUNT'"
done < "$USERS_FILE"

echo "[INFO] Slurm accounting setup complete."
