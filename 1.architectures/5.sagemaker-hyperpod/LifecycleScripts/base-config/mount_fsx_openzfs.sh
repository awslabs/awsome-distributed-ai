#!/bin/bash

# must be run as sudo

set -x
set -e

# FSx OpenZFS Endpoints and versions
FSX_OPENZFS_DNS_NAME="$1"
OPENZFS_MOUNT_POINT="$2"
NFS_VERSION=4.2

# Retry settings
MAX_ATTEMPTS=5
INITIAL_BACKOFF=1

# Function for exponential backoff
retry_with_backoff() {
    local max_attempts=$1
    local initial_backoff=$2
    local cmd="${@:3}"
    local attempt=1
    local backoff=$initial_backoff

    while [ $attempt -le $max_attempts ]; do
        if eval "$cmd"; then
            return 0
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            echo "Failed after $attempt attempts"
            return 1
        fi
        
        echo "Attempt $attempt failed. Retrying in $backoff seconds..."
        sleep $backoff
        
        # Exponential backoff with jitter
        backoff=$(( backoff * 2 + (RANDOM % 3) ))
        attempt=$((attempt + 1))
    done
}

# Function for error handling
handle_error()
{
    local exit_code=$?
    echo "Error occurred in command: $BASH_COMMAND"
    echo "Exit code: $exit_code"
    exit $exit_code
}

trap handle_error ERR

# DEBUG: Verify parameters are set
verify_parameters()
{
    if [ -z "$FSX_OPENZFS_DNS_NAME" ] || [ -z "$OPENZFS_MOUNT_POINT" ]; then
        echo "Usage: $0 <fsx_dns_name> <mount_point>"
        exit 1
    fi
}

# Install NFS Client based on OS
install_nfs_client()
{
    if [ -f /etc/lsb-release ]; then
        # Ubuntu
        retry_with_backoff $MAX_ATTEMPTS $INITIAL_BACKOFF "ansible localhost -b -m ansible.builtin.apt -a 'name=nfs-common state=present update_cache=yes'"
    elif [ -f /etc/redhat-release ]; then
        # CentOS/RHEL
        retry_with_backoff $MAX_ATTEMPTS $INITIAL_BACKOFF "ansible localhost -b -m ansible.builtin.yum -a 'name=nfs-utils state=present'"
    fi
}

# Mount the FSx OpenZFS file system
mount_fs()
{
    local max_attempts=5
    local attempt=1
    local delay=5
    local test_file="$OPENZFS_MOUNT_POINT/test_file_$(hostname)"

    echo "[INFO] Ensuring $OPENZFS_MOUNT_POINT directory exists..."
    ansible localhost -b -m ansible.builtin.file -a "path=$OPENZFS_MOUNT_POINT state=directory" || true

    echo "[INFO] Mounting FSx OpenZFS on $OPENZFS_MOUNT_POINT..."
    echo "[INFO] Using test file: $test_file"

    while (( attempt <= max_attempts )); do
        echo "============================"
        echo "[INFO] Attempt $attempt of $max_attempts"
        echo "============================"

        echo "[STEP] Mounting FSx OpenZFS..."
        if ! ansible localhost -b -m ansible.posix.mount -a \
            "path=$OPENZFS_MOUNT_POINT src=$FSX_OPENZFS_DNS_NAME:/fsx fstype=nfs opts=nfsvers=$NFS_VERSION,_netdev,nconnect=16,x-systemd.automount,x-systemd.requires=network-online.target dump=0 passno=0 state=mounted"; then
            echo "[WARN] Mount command failed — retrying in $delay seconds"
            sleep "$delay"; ((attempt++)); continue
        fi

        echo "[STEP] Verifying mountpoint..."
        if ! ansible localhost -b -m ansible.builtin.command -a "mountpoint $OPENZFS_MOUNT_POINT"; then
            echo "[WARN] Mountpoint verification failed — retrying in $delay seconds"
            sleep "$delay"; ((attempt++)); continue
        fi

        echo "[STEP] Triggering automount..."
        ls -la "$OPENZFS_MOUNT_POINT" >/dev/null 2>&1 || true

        echo "[STEP] Testing file access (touch)..."
        if ! ansible localhost -b -m ansible.builtin.file -a "path=$test_file state=touch"; then
            echo "[WARN] Touch failed — retrying in $delay seconds"
            sleep "$delay"; ((attempt++)); continue
        fi

        echo "[STEP] Testing file access (delete)..."
        if ! ansible localhost -b -m ansible.builtin.file -a "path=$test_file state=absent"; then
            echo "[WARN] Delete failed — retrying in $delay seconds"
            sleep "$delay"; ((attempt++)); continue
        fi

        echo "[SUCCESS] FSx OpenZFS mount succeeded on attempt $attempt"
        return 0
    done

    echo "[ERROR] FSx OpenZFS mount failed after $max_attempts attempts"
    return 1
}

# Restart systemd daemon to ensure mount units are properly loaded
restart_daemon()
{
    ansible localhost -b -m ansible.builtin.systemd -a "daemon_reload=yes"
    ansible localhost -b -m ansible.builtin.systemd -a "name=remote-fs.target state=restarted"
    echo "Check status of OpenZFS automount..."
    systemctl list-units | grep -i automount || true
}

main() 
{
    echo "Mount_fsx_openzfs called with fsx_openzfs_dns_name: $FSX_OPENZFS_DNS_NAME"
    echo "Using openzfs_mount_point: $OPENZFS_MOUNT_POINT"
    verify_parameters
    install_nfs_client
    mount_fs || exit 1
    restart_daemon
    echo "FSx OpenZFS mounted successfully to $OPENZFS_MOUNT_POINT"
}

main "$@"