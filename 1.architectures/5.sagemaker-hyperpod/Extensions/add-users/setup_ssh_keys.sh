#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Generates SSH keypairs on the shared filesystem for passwordless inter-node SSH.
# First node to run creates the keypair; subsequent nodes find it already there.
#
# Usage: setup_ssh_keys.sh <users_file> <shared_fs_mount>

set -e

USERS_FILE="$1"
SHARED_FS_MOUNT="$2"

if [[ -z "$USERS_FILE" ]] || [[ -z "$SHARED_FS_MOUNT" ]]; then
    echo "Usage: setup_ssh_keys.sh <users_file> <shared_fs_mount>"
    exit 1
fi

setup_ssh() {
    local username="$1"
    local ssh_dir="${SHARED_FS_MOUNT}/${username}/.ssh"

    if ! id -u "$username" >/dev/null 2>&1; then
        echo "[WARN] User $username does not exist. Skipping SSH setup."
        return 0
    fi

    mkdir -p "$ssh_dir"

    if [[ ! -f "$ssh_dir/id_rsa" ]]; then
        echo "[INFO] Generating SSH keypair for $username on shared filesystem"
        ssh-keygen -t rsa -b 4096 -q -f "$ssh_dir/id_rsa" -N "" 2>/dev/null || true
        cat "$ssh_dir/id_rsa.pub" >> "$ssh_dir/authorized_keys"
    else
        echo "[INFO] SSH keypair for $username already exists on shared filesystem"
        if [[ -f "$ssh_dir/id_rsa.pub" ]]; then
            if ! grep -qF "$(cat "$ssh_dir/id_rsa.pub")" "$ssh_dir/authorized_keys" 2>/dev/null; then
                cat "$ssh_dir/id_rsa.pub" >> "$ssh_dir/authorized_keys"
            fi
        fi
    fi

    chmod 700 "$ssh_dir"
    chmod 600 "$ssh_dir/id_rsa" 2>/dev/null || true
    chmod 644 "$ssh_dir/id_rsa.pub" 2>/dev/null || true
    touch "$ssh_dir/authorized_keys"
    chmod 600 "$ssh_dir/authorized_keys"
    chown -R "$username:$username" "$ssh_dir"

    local user_home
    user_home=$(getent passwd "$username" | cut -d: -f6)

    if [[ "$user_home" == "${SHARED_FS_MOUNT}/${username}" ]]; then
        local local_home="/home/$username"
        if [[ -d "$local_home" ]] && [[ "$local_home" != "$user_home" ]]; then
            if [[ -L "$local_home/.ssh" ]]; then
                echo "[INFO] Symlink $local_home/.ssh already exists"
            elif [[ -d "$local_home/.ssh" ]]; then
                rm -rf "$local_home/.ssh"
                ln -s "$ssh_dir" "$local_home/.ssh"
                chown -h "$username:$username" "$local_home/.ssh"
                echo "[INFO] Replaced $local_home/.ssh with symlink to $ssh_dir"
            else
                ln -s "$ssh_dir" "$local_home/.ssh"
                chown -h "$username:$username" "$local_home/.ssh"
                echo "[INFO] Created symlink $local_home/.ssh -> $ssh_dir"
            fi
        fi
    fi

    echo "[INFO] SSH setup complete for $username"
}

while IFS="," read -r username uid; do
    username=$(echo "$username" | xargs)
    [[ -z "$username" ]] && continue
    setup_ssh "$username"
done < "$USERS_FILE"

echo "[INFO] SSH key setup complete."
