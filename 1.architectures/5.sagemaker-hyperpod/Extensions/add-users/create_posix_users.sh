#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Creates POSIX users from the provided users list file.
# Must run on all nodes so UIDs are consistent cluster-wide.
#
# Usage: create_posix_users.sh <users_file> <shared_fs_mount>

set -e

USERS_FILE="$1"
SHARED_FS_MOUNT="$2"

if [[ -z "$USERS_FILE" ]] || [[ -z "$SHARED_FS_MOUNT" ]]; then
    echo "Usage: create_posix_users.sh <users_file> <shared_fs_mount>"
    exit 1
fi

create_user() {
    local username="$1"
    local uid="$2"
    local home_dir="${SHARED_FS_MOUNT}/${username}"

    if id -u "$username" >/dev/null 2>&1; then
        echo "[INFO] User $username already exists. Skipping."
        return 0
    fi

    if getent passwd "$uid" >/dev/null 2>&1; then
        echo "[WARN] UID $uid is already in use. Skipping user $username."
        return 0
    fi

    if useradd -m "$username" --uid "$uid" -d "$home_dir" --shell /bin/bash; then
        echo "[INFO] Created user $username (uid=$uid, home=$home_dir)"
    else
        echo "[ERROR] Failed to create user $username (uid=$uid)"
        return 1
    fi
}

while IFS="," read -r username uid; do
    username=$(echo "$username" | xargs)
    uid=$(echo "$uid" | xargs)
    [[ -z "$username" ]] || [[ -z "$uid" ]] && continue
    create_user "$username" "$uid"
done < "$USERS_FILE"

echo "[INFO] POSIX user creation complete."
