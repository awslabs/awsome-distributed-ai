#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Sets up user home directories on the shared filesystem.
# When OpenZFS is the home filesystem and Lustre is also available,
# a /fsx/<username> directory is created for data access.
#
# Usage: setup_home_dirs.sh <users_file> <shared_fs_mount>

set -e

USERS_FILE="$1"
SHARED_FS_MOUNT="$2"
LUSTRE_MOUNT="${LUSTRE_MOUNT:-}"

if [[ -z "$USERS_FILE" ]] || [[ -z "$SHARED_FS_MOUNT" ]]; then
    echo "Usage: setup_home_dirs.sh <users_file> <shared_fs_mount>"
    exit 1
fi

setup_home() {
    local username="$1"
    local target_home="${SHARED_FS_MOUNT}/${username}"

    if ! id -u "$username" >/dev/null 2>&1; then
        echo "[WARN] User $username does not exist. Skipping home setup."
        return 0
    fi

    if [[ ! -d "$target_home" ]]; then
        if sudo -u "$username" mkdir -p "$target_home" 2>/dev/null; then
            sudo -u "$username" chmod 750 "$target_home" 2>/dev/null || true
            echo "[INFO] Created home directory $target_home (as $username)"
        elif mkdir -p "$target_home" 2>/dev/null; then
            chown "$username:$username" "$target_home"
            chmod 750 "$target_home"
            echo "[INFO] Created home directory $target_home (as root)"
        else
            echo "[WARN] Failed to create $target_home. Skipping."
            return 0
        fi
    else
        chown "$username:$username" "$target_home" 2>/dev/null || true
        echo "[INFO] Home directory $target_home already exists."
    fi

    local current_home
    current_home=$(getent passwd "$username" | cut -d: -f6)

    if [[ "$current_home" == "$target_home" ]]; then
        echo "[INFO] Home for $username already set to $target_home"
    else
        if [[ -d "$current_home" ]] && [[ "$current_home" != "$target_home" ]]; then
            echo "[INFO] Copying contents from $current_home to $target_home"
            rsync -a "$current_home/" "$target_home/" 2>/dev/null || true
            chown -R "$username:$username" "$target_home" 2>/dev/null || true
        fi
        usermod -d "$target_home" "$username"
        echo "[INFO] Updated home for $username to $target_home"
    fi

    if [[ -n "$LUSTRE_MOUNT" ]]; then
        local lustre_dir="${LUSTRE_MOUNT}/${username}"
        if [[ ! -d "$lustre_dir" ]]; then
            mkdir -p "$lustre_dir"
            chown "$username:$username" "$lustre_dir"
            echo "[INFO] Created Lustre data directory $lustre_dir"
        fi
    fi
}

while IFS="," read -r username uid; do
    username=$(echo "$username" | xargs)
    [[ -z "$username" ]] && continue
    setup_home "$username"
done < "$USERS_FILE"

echo "[INFO] Home directory setup complete."
