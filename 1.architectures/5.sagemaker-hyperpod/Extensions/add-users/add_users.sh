#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Standalone script to add users to SageMaker HyperPod Slurm clusters.
#
# Designed for use as an OnInitComplete script with AMI-based configuration.
# Creates POSIX users, sets up home directories on the shared filesystem,
# generates SSH keypairs for passwordless inter-node access, and registers
# users in Slurm accounting.
#
# Supports three config sources (checked in this order):
#   1. shared_users.txt |-- legacy CSV format (username,uid,home) for backward
#      compatibility with the base lifecycle scripts. If present (even if empty),
#      shared_users.yaml is ignored.
#   2. shared_users.yaml |-- YAML format with simple user list or groups with
#      per-group Slurm accounts and filesystem mounts.
#   3. If neither file exists, the script exits with an error.
#
# Prerequisites:
#   - /opt/ml/config/nodeinfo.json must exist (run detect-node/detect_node.sh first)
#   - shared_users.txt or shared_users.yaml must be configured with user definitions
#   - A shared filesystem (FSx for Lustre or OpenZFS) is recommended but not required.
#     Without one, home directories and SSH keys are local to each node.
#
# Usage:
#   As OnInitComplete:
#     Upload this directory to S3 and specify add_users.sh as OnInitComplete.
#
#   Manual execution on a running cluster:
#     sudo bash add_users.sh

set -e

LOG_FILE="/var/log/provision/add_users.log"
mkdir -p /var/log/provision
touch "$LOG_FILE"

logger() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

logger_err() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE" >&2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Read node type from nodeinfo.json (written by detect-node/detect_node.sh).
# Steps 1-3 run on all nodes. Only step 4 (Slurm accounting) is controller-only.
# ---------------------------------------------------------------------------
NODEINFO_FILE="/opt/ml/config/nodeinfo.json"
if [[ -f "$NODEINFO_FILE" ]]; then
    NODE_TYPE=$(python3 -c "import json; print(json.load(open('$NODEINFO_FILE'))['node_type'])")
    logger "Node type from nodeinfo.json: $NODE_TYPE"
else
    logger "WARNING: $NODEINFO_FILE not found. Run detect-node/detect_node.sh first."
    logger "Falling back: assuming not controller (Slurm accounting will be skipped)."
    NODE_TYPE="unknown"
fi

# ---------------------------------------------------------------------------
# Detect shared filesystem mount
# ---------------------------------------------------------------------------
detect_shared_filesystem() {
    local override="$1"

    if [[ -n "$override" ]]; then
        if mountpoint -q "$override" 2>/dev/null; then
            echo "$override"
            return 0
        else
            logger_err "WARNING: Configured mount $override is not a mountpoint. Auto-detecting..."
        fi
    fi

    # OpenZFS at /home takes priority (matches base LCS behavior)
    if mountpoint -q "/home" 2>/dev/null && grep -qsE '\s/home\s+nfs4?\s' /proc/mounts; then
        echo "/home"
        return 0
    fi

    # Fallback: FSx for Lustre at /fsx
    if mountpoint -q "/fsx" 2>/dev/null; then
        echo "/fsx"
        return 0
    fi

    return 1
}

# ---------------------------------------------------------------------------
# Determine user source: shared_users.txt (legacy) or shared_users.yaml
#
# Priority: shared_users.txt takes precedence over shared_users.yaml for
# backward compatibility with the base lifecycle scripts. If shared_users.txt
# exists (even if empty), shared_users.yaml is ignored.
# ---------------------------------------------------------------------------
SHARED_USERS_FILE="$SCRIPT_DIR/shared_users.txt"
CONFIG_FILE="$SCRIPT_DIR/shared_users.yaml"
WORK_DIR=$(mktemp -d /tmp/add_users.XXXXXX)
USE_LEGACY=false

if [[ -f "$SHARED_USERS_FILE" ]]; then
    USE_LEGACY=true
    logger "Found shared_users.txt |-- using legacy format (shared_users.yaml ignored if present)."

    if [[ ! -s "$SHARED_USERS_FILE" ]]; then
        logger "shared_users.txt is empty. No users to configure."
        rm -rf "$WORK_DIR"
        exit 0
    fi

    # Convert legacy CSV (username,uid,home) to our internal format (username,uid)
    # and build a single-group manifest with default settings
    USERS_CSV="$WORK_DIR/users_default.csv"
    while IFS="," read -r username uid home; do
        username=$(echo "$username" | xargs)
        uid=$(echo "$uid" | xargs)
        [[ -z "$username" ]] || [[ -z "$uid" ]] && continue
        echo "$username,$uid" >> "$USERS_CSV"
    done < "$SHARED_USERS_FILE"

    if [[ ! -s "$USERS_CSV" ]]; then
        logger "No valid users in shared_users.txt. Nothing to do."
        rm -rf "$WORK_DIR"
        exit 0
    fi

    USER_COUNT=$(wc -l < "$USERS_CSV")
    python3 -c "
import json, os
manifest = [{
    'name': 'default',
    'slurm_account': 'root',
    'shared_filesystem_mount': '',
    'users_file': '$USERS_CSV',
    'user_count': $USER_COUNT
}]
with open(os.path.join('$WORK_DIR', 'manifest.json'), 'w') as f:
    json.dump(manifest, f)
"
    logger "Parsed $USER_COUNT user(s) from shared_users.txt."

elif [[ -f "$CONFIG_FILE" ]]; then
    logger "Using shared_users.yaml for user definitions."

    python3 - "$CONFIG_FILE" "$WORK_DIR" << 'PYEOF'
import yaml, json, os, sys

config_file = sys.argv[1]
work_dir = sys.argv[2]

with open(config_file) as f:
    config = yaml.safe_load(f)

global_slurm_account = config.get('slurm_account', 'root')
global_fs_mount = config.get('shared_filesystem_mount', '')

groups = config.get('groups', [])
users = config.get('users', [])

if groups and users:
    print("ERROR: shared_users.yaml has both 'groups' and 'users' at top level. Use one or the other.")
    sys.exit(1)

if not groups and not users:
    print("WARNING: No users or groups defined in shared_users.yaml.")
    sys.exit(0)

if users and not groups:
    groups = [{
        'name': 'default',
        'slurm_account': global_slurm_account,
        'shared_filesystem_mount': global_fs_mount,
        'users': users
    }]
else:
    for g in groups:
        g.setdefault('slurm_account', global_slurm_account)
        g.setdefault('shared_filesystem_mount', global_fs_mount)

manifest = []
for g in groups:
    name = g['name']
    group_users = g.get('users', [])
    if not group_users:
        continue

    users_file = os.path.join(work_dir, f'users_{name}.csv')
    with open(users_file, 'w') as uf:
        for u in group_users:
            uf.write(f"{u['username']},{u['uid']}\n")

    manifest.append({
        'name': name,
        'slurm_account': g['slurm_account'],
        'shared_filesystem_mount': g['shared_filesystem_mount'],
        'users_file': users_file,
        'user_count': len(group_users)
    })

with open(os.path.join(work_dir, 'manifest.json'), 'w') as mf:
    json.dump(manifest, mf)

total = sum(g['user_count'] for g in manifest)
print(f"Parsed {total} user(s) in {len(manifest)} group(s).")
PYEOF

else
    logger "ERROR: Neither shared_users.txt nor shared_users.yaml found in $SCRIPT_DIR"
    rm -rf "$WORK_DIR"
    exit 1
fi

MANIFEST="$WORK_DIR/manifest.json"
if [[ ! -f "$MANIFEST" ]]; then
    logger "No users to configure. Exiting."
    rm -rf "$WORK_DIR"
    exit 0
fi

# ---------------------------------------------------------------------------
# Process each group
# ---------------------------------------------------------------------------
GROUP_COUNT=$(python3 -c "import json; print(len(json.load(open('$MANIFEST'))))")
logger "Processing $GROUP_COUNT group(s)..."

for (( i=0; i<GROUP_COUNT; i++ )); do
    eval "$(python3 -c "
import json
m = json.load(open('$MANIFEST'))[$i]
print(f\"GROUP_NAME='{m['name']}'\")
print(f\"USERS_FILE='{m['users_file']}'\")
print(f\"SLURM_ACCOUNT='{m['slurm_account']}'\")
print(f\"FS_OVERRIDE='{m['shared_filesystem_mount']}'\")
print(f\"USER_COUNT={m['user_count']}\")
")"

    logger "--- Group: $GROUP_NAME ($USER_COUNT users) ---"

    SHARED_FS_MOUNT=$(detect_shared_filesystem "$FS_OVERRIDE" || true)
    if [[ -z "$SHARED_FS_MOUNT" ]]; then
        SHARED_FS_MOUNT="/home"
        logger "WARNING: No shared filesystem detected for group $GROUP_NAME."
        logger "WARNING: Using local /home for home directories. SSH keys will be per-node only."
        logger "WARNING: Passwordless cross-node SSH will NOT work without a shared filesystem."
    else
        logger "Shared filesystem for $GROUP_NAME: $SHARED_FS_MOUNT"
    fi

    export LUSTRE_MOUNT=""
    if [[ "$SHARED_FS_MOUNT" != "/fsx" ]] && mountpoint -q "/fsx" 2>/dev/null; then
        LUSTRE_MOUNT="/fsx"
    fi

    logger "  Step 1: Creating POSIX users"
    bash "$SCRIPT_DIR/create_posix_users.sh" "$USERS_FILE" "$SHARED_FS_MOUNT" 2>&1 | tee -a "$LOG_FILE"

    logger "  Step 2: Setting up home directories on $SHARED_FS_MOUNT"
    bash "$SCRIPT_DIR/setup_home_dirs.sh" "$USERS_FILE" "$SHARED_FS_MOUNT" 2>&1 | tee -a "$LOG_FILE"

    logger "  Step 3: Setting up SSH keys"
    bash "$SCRIPT_DIR/setup_ssh_keys.sh" "$USERS_FILE" "$SHARED_FS_MOUNT" 2>&1 | tee -a "$LOG_FILE"

    if [[ "$NODE_TYPE" == "controller" ]]; then
        logger "  Step 4: Setting up Slurm accounting (account: $SLURM_ACCOUNT)"
        bash "$SCRIPT_DIR/setup_slurm_accounts.sh" "$USERS_FILE" "$SLURM_ACCOUNT" 2>&1 | tee -a "$LOG_FILE"
    else
        logger "  Step 4: Skipping Slurm accounting (not controller)"
    fi
done

rm -rf "$WORK_DIR"
logger "Add users complete."
