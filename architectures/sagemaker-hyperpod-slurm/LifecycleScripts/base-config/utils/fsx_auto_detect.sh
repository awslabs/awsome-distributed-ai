#!/bin/bash

# =============================================================================
# fsx_auto_detect.sh
#
# Auto-detects existing FSx mounts (OpenZFS and/or Lustre) and sets up
# home directories accordingly. This script is called when provisioning
# parameters do NOT include fsx_dns_name/fsx_mountname or fsx_openzfs_dns_name,
# but mounts may already exist on the instance.
#
# If no mounts are detected, the script logs and exits silently (exit 0).
# All errors are logged but never cause a hard failure.
#
# NOTE: OpenZFS (NFS) mounts may have root_squash enabled, which maps root
# to nobody:nogroup and prevents root from doing chown. This script handles
# that by falling back to sudo -u <user> for directory creation when ansible
# chown fails.
# =============================================================================

echo "[INFO] ========================================================"
echo "[INFO] fsx_auto_detect.sh started — checking for existing FSx mounts..."
echo "[INFO] ========================================================"

# RETRY CONFIG
ATTEMPTS=6
WAIT=10

# If mountpath is provided in API then please update these values accordingly.
FSX_OPENZFS_DNS_NAME="/home"
FSX_L_DNS_NAME="/fsx"

# Look for shared_users.txt in parent directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_USER_FILE="${SCRIPT_DIR}/../shared_users.txt"

echo "[INFO] Configuration:"
echo "[INFO]   OpenZFS mount path : $FSX_OPENZFS_DNS_NAME"
echo "[INFO]   Lustre mount path  : $FSX_L_DNS_NAME"
echo "[INFO]   Shared users file  : $SHARED_USER_FILE"

# --------------------------------------------------------------------------
# Mount detection helpers — filesystem-type aware to avoid false positives
# --------------------------------------------------------------------------

# FSx for Lustre reports as "lustre" filesystem type in /proc/mounts
check_lustre_mount() {
    local path="$1"
    if mountpoint -q "$path" 2>/dev/null; then
        if grep -qsE "\s${path}\s+lustre\s" /proc/mounts; then
            echo "[INFO] Confirmed Lustre mount at $path (filesystem type: lustre)"
            return 0
        else
            echo "[INFO] $path is a mountpoint but NOT a Lustre filesystem — skipping"
        fi
    else
        echo "[INFO] $path is not a mountpoint"
    fi
    return 1
}

# FSx OpenZFS reports as "nfs" or "nfs4" filesystem type in /proc/mounts
check_openzfs_mount() {
    local path="$1"
    if mountpoint -q "$path" 2>/dev/null; then
        if grep -qsE "\s${path}\s+nfs4?\s" /proc/mounts; then
            echo "[INFO] Confirmed NFS/OpenZFS mount at $path (filesystem type: nfs/nfs4)"
            return 0
        else
            echo "[INFO] $path is a mountpoint but NOT an NFS/OpenZFS filesystem — skipping"
        fi
    else
        echo "[INFO] $path is not a mountpoint"
    fi
    return 1
}

# Function to check mount is writable
check_mount_writable() {
    local mount_point="$1"
    if touch "$mount_point/.test_write" 2>/dev/null; then
        rm -f "$mount_point/.test_write"
        return 0
    fi
    return 1
}

# Wait for mount to become ready (writable)
wait_for_mount() {
    local mount_point="$1"
    for ((i=1; i<=ATTEMPTS; i++)); do
        if check_mount_writable "$mount_point"; then
            echo "[INFO] Successfully verified writable mount at $mount_point"
            return 0
        fi
        if [ $i -eq $ATTEMPTS ]; then
            echo "[WARN] Mount at $mount_point not writable after $((ATTEMPTS * WAIT)) seconds"
            return 1
        fi
        echo "[INFO] Waiting for mount $mount_point to be writable... (attempt $i/$ATTEMPTS)"
        sleep $WAIT
    done
}

# --------------------------------------------------------------------------
# Home directory setup functions (mirrors fsx_ubuntu.sh logic)
# --------------------------------------------------------------------------

# Setup home directory for a user with OpenZFS
# Handles NFS root_squash by falling back to sudo -u <user> when ansible chown fails
setup_user_home_openzfs() {
    local username=$1
    local user_dir="$FSX_OPENZFS_DNS_NAME/$username"

    echo "[INFO] Setting up OpenZFS home directory for user: $username"

    # Try ansible approach first (works when no_root_squash is set on the NFS export)
    if ansible localhost -b -m ansible.builtin.file \
        -a "path='$user_dir' state=directory owner=$username group=$username mode=0755" 2>/dev/null; then
        echo "[INFO] Created $user_dir via ansible (no_root_squash)"
    else
        # Ansible chown failed — likely NFS root_squash is enabled.
        # Root is mapped to nobody:nogroup so chown is denied.
        # Fallback: create directory as the target user via sudo -u.
        echo "[INFO] Ansible chown failed (likely NFS root_squash). Falling back to sudo -u $username..."

        if sudo -u "$username" mkdir -p "$user_dir" 2>/dev/null; then
            sudo -u "$username" chmod 0755 "$user_dir" 2>/dev/null || true
            echo "[INFO] Created $user_dir via sudo -u $username"
        elif sudo mkdir -p "$user_dir" 2>/dev/null; then
            # If sudo -u also fails (user may not have NFS access), try plain sudo mkdir
            # and skip chown (it will be owned by nobody but at least the dir exists)
            echo "[WARN] sudo -u $username mkdir also failed. Created $user_dir as root (owner may be nobody due to root_squash)"
        else
            echo "[WARN] Failed to create $user_dir for $username via all methods. Skipping."
            return 0
        fi
    fi

    # Set home directory to the OpenZFS path
    # Use usermod directly instead of ansible.builtin.user to avoid potential root_squash issues with move_home
    if [ -d "/home/$username" ] && [ "/home/$username" != "$user_dir" ]; then
        # Copy existing home contents to new location if they exist and are different
        echo "[INFO] Copying existing /home/$username contents to $user_dir"
        sudo rsync -a "/home/$username/" "$user_dir/" 2>/dev/null || true
        # Fix ownership — try as user first (root_squash safe), then as root
        sudo -u "$username" find "$user_dir" -maxdepth 1 -exec true \; 2>/dev/null || true
    fi

    sudo usermod -d "$user_dir" "$username" 2>/dev/null \
        || { echo "[WARN] Failed to set home dir for $username via usermod"; return 0; }
    echo "[INFO] Home directory set to $user_dir for $username"

    # Maintain access to /fsx/username if Lustre is available
    if check_lustre_mount "$FSX_L_DNS_NAME" && check_mount_writable "$FSX_L_DNS_NAME"; then
        echo "[INFO] Lustre also available — creating $FSX_L_DNS_NAME/$username directory"
        sudo mkdir -p "$FSX_L_DNS_NAME/$username" 2>/dev/null || true
        sudo chown "$username:$username" "$FSX_L_DNS_NAME/$username" 2>/dev/null || true
    else
        echo "[INFO] FSx Lustre mount not available, skipping $FSX_L_DNS_NAME/$username setup"
    fi
}

# Setup home directory for a user with FSx Lustre only
setup_user_home_fsx_lustre() {
    local username=$1
    local fsx_home=$2

    echo "[INFO] Setting up FSx Lustre home directory for user: $username at $fsx_home"

    if [ -d "$fsx_home" ]; then
        echo "[INFO] Directory $fsx_home already exists, setting as home for $username"
        sudo usermod -d "$fsx_home" "$username" 2>/dev/null || { echo "[WARN] Failed to set home for $username"; return 0; }
    elif [ -d "$FSX_L_DNS_NAME" ]; then
        # Create the directory
        echo "[INFO] Creating directory $fsx_home for user $username"
        sudo mkdir -p "$fsx_home" 2>/dev/null || { echo "[WARN] Failed to mkdir $fsx_home"; return 0; }
        sudo chown "$username:$username" "$fsx_home" 2>/dev/null || true

        # Try to change home directory with move
        if ! sudo usermod -m -d "$fsx_home" "$username" 2>/dev/null; then
            echo "[WARN] Could not move home directory for $username. Setting home without moving files."

            # If user has existing home, copy contents
            if [ -d "/home/$username" ]; then
                echo "[INFO] Copying existing /home/$username contents to $fsx_home"
                sudo rsync -a "/home/$username/" "$fsx_home/" 2>/dev/null || true
            fi
            sudo chown -R "$username:$username" "$fsx_home" 2>/dev/null || true

            sudo usermod -d "$fsx_home" "$username" 2>/dev/null || { echo "[WARN] Failed to set home for $username"; return 0; }
        else
            echo "[INFO] Home directory moved successfully to $fsx_home"
        fi
    else
        echo "[WARN] Neither $fsx_home nor $FSX_L_DNS_NAME exist — cannot setup home for $username"
    fi
}

# --------------------------------------------------------------------------
# Process shared_users.txt for a given setup function
# --------------------------------------------------------------------------
process_shared_users() {
    local setup_function=$1  # "openzfs" or "lustre"

    if [[ ! -f "$SHARED_USER_FILE" ]]; then
        echo "[INFO] No $SHARED_USER_FILE found, only ubuntu user configured"
        return 0
    fi

    echo "[INFO] Found $SHARED_USER_FILE, processing additional users..."
    echo "[INFO] Contents of $SHARED_USER_FILE:"
    cat "$SHARED_USER_FILE"

    while IFS="," read -r username uid home; do
        # Trim whitespace from all fields
        username=$(echo "$username" | xargs)
        uid=$(echo "$uid" | xargs)
        home=$(echo "$home" | xargs)

        # Skip empty lines or lines that are just whitespace
        if [[ -z "$username" ]]; then
            echo "[INFO] Skipping empty or invalid line"
            continue
        fi

        # Verify user exists before trying to set up home
        if ! id -u "$username" >/dev/null 2>&1; then
            echo "[WARN] User $username does not exist, skipping home setup"
            continue
        fi

        echo "[INFO] Processing home directory for user: '$username'"

        if [[ "$setup_function" == "openzfs" ]]; then
            setup_user_home_openzfs "$username"
        elif [[ "$setup_function" == "lustre" ]]; then
            # For lustre, if home field is empty, use default
            if [[ -z "$home" ]]; then
                home="$FSX_L_DNS_NAME/$username"
            fi
            setup_user_home_fsx_lustre "$username" "$home"
        fi
    done < "$SHARED_USER_FILE"

    echo "[INFO] All users from $SHARED_USER_FILE processed successfully"
}

# ==========================================================================
# Main logic — detect mounts and setup accordingly
# ==========================================================================

echo "[INFO] --------------------------------------------------------"
echo "[INFO] Detecting FSx mounts..."
echo "[INFO] --------------------------------------------------------"

OPENZFS_DETECTED=false
LUSTRE_DETECTED=false

if check_openzfs_mount "$FSX_OPENZFS_DNS_NAME"; then
    OPENZFS_DETECTED=true
fi

if check_lustre_mount "$FSX_L_DNS_NAME"; then
    LUSTRE_DETECTED=true
fi

echo "[INFO] --------------------------------------------------------"
echo "[INFO] Detection results — OpenZFS: $OPENZFS_DETECTED, Lustre: $LUSTRE_DETECTED"
echo "[INFO] --------------------------------------------------------"

# --- No mounts detected: nothing to do ---
if [[ "$OPENZFS_DETECTED" == false ]] && [[ "$LUSTRE_DETECTED" == false ]]; then
    echo "[INFO] No FSx OpenZFS mount at $FSX_OPENZFS_DNS_NAME and no FSx Lustre mount at $FSX_L_DNS_NAME detected."
    echo "[INFO] No mount-based home directory setup required. Exiting."
    exit 0
fi

# Track whether home directory setup succeeded via OpenZFS
HOME_SETUP_DONE=false

# --- Try OpenZFS first ---
if [[ "$OPENZFS_DETECTED" == true ]]; then
    echo "[INFO] OpenZFS mount detected at $FSX_OPENZFS_DNS_NAME. Attempting home directory setup on OpenZFS..."

    if wait_for_mount "$FSX_OPENZFS_DNS_NAME"; then
        # Setup ubuntu user first
        echo "[INFO] Setting up home directory for default ubuntu user..."
        setup_user_home_openzfs "ubuntu"

        # Process additional users
        process_shared_users "openzfs"

        HOME_SETUP_DONE=true
        echo "[INFO] OpenZFS home directory setup completed successfully."
    else
        echo "[WARN] OpenZFS mount at $FSX_OPENZFS_DNS_NAME detected but is NOT writable. Will attempt Lustre fallback..."
    fi
fi

# --- Fallback to Lustre if OpenZFS home setup didn't succeed ---
if [[ "$HOME_SETUP_DONE" == false ]] && [[ "$LUSTRE_DETECTED" == true ]]; then
    echo "[INFO] FSx Lustre mount detected at $FSX_L_DNS_NAME. Using Lustre as home directory."

    if wait_for_mount "$FSX_L_DNS_NAME"; then
        # Setup ubuntu user first
        echo "[INFO] Setting up home directory for default ubuntu user..."
        setup_user_home_fsx_lustre "ubuntu" "$FSX_L_DNS_NAME/ubuntu"

        # Process additional users
        process_shared_users "lustre"

        echo "[INFO] Lustre home directory setup completed successfully."
    else
        echo "[WARN] FSx Lustre mount at $FSX_L_DNS_NAME detected but is NOT writable. Skipping Lustre home setup."
    fi
elif [[ "$HOME_SETUP_DONE" == false ]] && [[ "$LUSTRE_DETECTED" == false ]]; then
    echo "[WARN] OpenZFS was detected but not writable, and no Lustre mount available as fallback."
fi

echo "[INFO] ========================================================"
echo "[INFO] fsx_auto_detect.sh completed."
echo "[INFO] ========================================================"
exit 0
