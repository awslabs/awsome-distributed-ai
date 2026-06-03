#!/bin/bash

# NOTE: We intentionally do NOT use 'set -e' because this script uses
# background jobs for parallel SSH, and set -e behaves unpredictably
# with subshells/background processes. Errors are handled explicitly.

# ============================================================================
# create-users.sh — Create one or more users across all nodes in a SageMaker
# HyperPod Slurm cluster (controller, login nodes, and compute nodes).
#
# This script can be run from ANY node (controller, login, or compute).
# It auto-detects the current node and SSHes to all other nodes.
# Compute nodes are configured in parallel for large clusters (500+ nodes).
#
# Modes (priority order):
#   1. CLI mode:        sudo ./create-users.sh user1 user2 user3
#                       Bypasses file/interactive, auto-assigns UIDs.
#   2. File mode (.txt): If "shared_users.txt" exists, reads users from it,
#                       validates, and creates only new users on all nodes.
#                       Also offers to add additional users interactively.
#   3. File mode (.yaml): If "shared_users.yaml" exists, parses user definitions
#                       (supports simple user list or groups format).
#   4. Interactive mode: If no file found, prompts for usernames/UIDs.
#
# After creating users, appends to "shared_users.txt" and optionally
# uploads to the lifecycle S3 bucket.
#
# Usage:
#   sudo ./create-users.sh                  # file or interactive mode
#   sudo ./create-users.sh user1 user2      # CLI mode (auto-assign UIDs)
#
# Prerequisites:
#   - jq installed on the current node
#   - SSH access from current node to all other nodes
#   - /opt/ml/config/resource_config.json present
#   - /opt/ml/config/provisioning_parameters.json present
# ============================================================================

RESOURCE_CONFIG="/opt/ml/config/resource_config.json"
PROVISIONING_PARAMS="/opt/ml/config/provisioning_parameters.json"
SHARED_USERS_FILE="shared_users.txt"
MAX_PARALLEL=64

# --- Validate prerequisites ---
for config_file in "$RESOURCE_CONFIG" "$PROVISIONING_PARAMS"; do
    if [[ ! -f "$config_file" ]]; then
        echo "[ERROR] $config_file not found. Are you running this on a HyperPod node?"
        exit 1
    fi
done

if ! command -v jq &>/dev/null; then
    echo "[ERROR] jq is required but not installed."
    exit 1
fi

# --- Detect SSH user ---
if [[ -n "$SUDO_USER" ]]; then
    SSH_USER="$SUDO_USER"
elif [[ "$(id -u)" -ne 0 ]]; then
    SSH_USER="$(whoami)"
else
    SSH_USER="ubuntu"
    echo "[WARN] Could not detect original user, defaulting to '$SSH_USER' for SSH"
fi

# --- Helpers ---

ip_to_hostname() { echo "ip-${1//./-}"; }

get_group_hostnames() {
    local ips
    ips=$(jq -r --arg name "$1" \
        '.InstanceGroups[] | select(.Name == $name) | .Instances // [] | .[].CustomerIpAddress' \
        "$RESOURCE_CONFIG" 2>/dev/null)
    while IFS= read -r ip; do
        [[ -n "$ip" ]] && ip_to_hostname "$ip"
    done <<< "$ips"
}

create_users_on_remote() {
    local node="$1" node_type="$2" make_sudoer="$3"
    shift 3
    local users_data=("$@") remote_cmd=""

    for entry in "${users_data[@]}"; do
        IFS=',' read -r user uid home <<< "$entry"
        local uid_part=""
        [[ -n "$uid" ]] && uid_part="-u $uid"
        local cmd="(id $user &>/dev/null && echo '${user}:EXISTS'"
        cmd+=" || (sudo useradd -M $uid_part $user -d $home --shell /bin/bash"
        cmd+=" && (getent group docker &>/dev/null && sudo usermod -aG docker $user && echo '${user}:DOCKER' || true)"
        [[ "$make_sudoer" == "y" ]] && cmd+=" && (sudo usermod -aG sudo $user 2>/dev/null || sudo usermod -aG wheel $user 2>/dev/null || true) && echo '${user}:SUDOER'"
        cmd+=" && echo '${user}:CREATED' || echo '${user}:FAILED'))"
        [[ -n "$remote_cmd" ]] && remote_cmd+=" ; "
        remote_cmd+="$cmd"
    done

    local output
    output=$(sudo -u "$SSH_USER" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$node" "$remote_cmd" 2>&1) || true

    echo "  [$node_type] $node:"
    for entry in "${users_data[@]}"; do
        IFS=',' read -r user uid home <<< "$entry"
        if [[ "$output" == *"${user}:EXISTS"* ]]; then
            echo "    ⚠ '$user' already exists (skipped)"
        elif [[ "$output" == *"${user}:CREATED"* ]]; then
            echo "    ✓ '$user' created (UID: $uid)"
            [[ "$output" == *"${user}:DOCKER"* ]] && echo "      Added to docker group"
            [[ "$output" == *"${user}:SUDOER"* ]] && echo "      Added to sudoer group"
        elif [[ "$output" == *"${user}:FAILED"* ]]; then
            echo "    ✗ '$user' failed"
        else
            echo "    ✗ '$user' — unknown status (SSH may have failed)"
        fi
    done
    return 0
}

# ============================================================================
# CLUSTER CONFIGURATION (auto-detect, no prompts)
# ============================================================================

CONTROLLER_GROUP=$(jq -r '.controller_group // empty' "$PROVISIONING_PARAMS")
LOGIN_GROUP=$(jq -r '.login_group // empty' "$PROVISIONING_PARAMS")
WORKER_GROUPS=$(jq -r '.worker_groups[]?.instance_group_name // empty' "$PROVISIONING_PARAMS")
CURRENT_HOST=$(hostname)

echo "========================================"
echo " Cluster Configuration"
echo "========================================"
echo "  Current node:      $CURRENT_HOST"
echo "  SSH user:          $SSH_USER"
echo "  Controller group:  ${CONTROLLER_GROUP:-not found}"
echo "  Login group:       ${LOGIN_GROUP:-not configured}"
echo "  Worker group(s):   $(echo "${WORKER_GROUPS:-not found}" | tr '\n' ', ' | sed 's/, *$//')"

# Fetch fresh resource_config.json from controller if not running on controller
if [[ -n "$CONTROLLER_GROUP" ]]; then
    CONTROLLER_IP=$(jq -r --arg name "$CONTROLLER_GROUP" \
        '.InstanceGroups[] | select(.Name == $name) | .Instances // [] | .[0].CustomerIpAddress' \
        "$RESOURCE_CONFIG" 2>/dev/null)
    CURRENT_IP=$(hostname -I 2>/dev/null | awk '{print $1}')

    if [[ -n "$CONTROLLER_IP" && "$CURRENT_IP" != "$CONTROLLER_IP" ]]; then
        CONTROLLER_HOSTNAME=$(ip_to_hostname "$CONTROLLER_IP")
        FRESH_RC="/tmp/resource_config_fresh.json"
        if sudo -u "$SSH_USER" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
            "$CONTROLLER_HOSTNAME" "cat /opt/ml/config/resource_config.json" > "$FRESH_RC" 2>/dev/null; then
            RESOURCE_CONFIG="$FRESH_RC"
            echo "  ✓ Using fresh resource_config.json from controller ($CONTROLLER_HOSTNAME)"
        else
            echo "  ⚠ Could not fetch fresh resource_config from controller — using local copy"
        fi
    fi
fi

# Build node lists
CONTROLLER_NODES=()
[[ -n "$CONTROLLER_GROUP" ]] && read -ra CONTROLLER_NODES <<< "$(get_group_hostnames "$CONTROLLER_GROUP" | tr '\n' ' ')"

LOGIN_NODES=()
[[ -n "$LOGIN_GROUP" ]] && read -ra LOGIN_NODES <<< "$(get_group_hostnames "$LOGIN_GROUP" | tr '\n' ' ')"

COMPUTE_NODES=()
if [[ -n "$WORKER_GROUPS" ]]; then
    while IFS= read -r wg; do
        [[ -z "$wg" ]] && continue
        read -ra wg_hosts <<< "$(get_group_hostnames "$wg" | tr '\n' ' ')"
        COMPUTE_NODES+=("${wg_hosts[@]}")
    done <<< "$WORKER_GROUPS"
fi

# Detect role
CURRENT_ROLE="unknown"
for n in "${CONTROLLER_NODES[@]}"; do [[ "$n" == "$CURRENT_HOST" ]] && CURRENT_ROLE="controller" && break; done
[[ "$CURRENT_ROLE" == "unknown" ]] && for n in "${LOGIN_NODES[@]}"; do [[ "$n" == "$CURRENT_HOST" ]] && CURRENT_ROLE="login" && break; done
[[ "$CURRENT_ROLE" == "unknown" ]] && for n in "${COMPUTE_NODES[@]}"; do [[ "$n" == "$CURRENT_HOST" ]] && CURRENT_ROLE="compute" && break; done

echo "  Current node role: $CURRENT_ROLE"

# Detect OpenZFS
USE_OPENZFS=false
if mountpoint -q /home 2>/dev/null; then
    USE_OPENZFS=true
    echo "  OpenZFS:          detected at /home"
else
    echo "  OpenZFS:          not detected"
fi

echo "  Nodes:            ${#CONTROLLER_NODES[@]} controller, ${#LOGIN_NODES[@]} login, ${#COMPUTE_NODES[@]} compute"
echo ""

# ===================================================================
# Step 1: Determine mode + gather user data
# ===================================================================
echo "========================================"
echo " Step 1: User Configuration"
echo "========================================"

declare -a USERS_DATA=()        # "user,uid,/fsx/user" entries to create locally
declare -a USERS_TO_PROPAGATE=() # All users to ensure exist on remote nodes (new + existing locally)
MAKE_SUDOER="n"

if [[ $# -gt 0 ]]; then
    # --- CLI MODE — usernames passed as arguments ---
    echo "  Users from CLI args: $*"
    USERS=("$@")

    # Validate
    for user in "${USERS[@]}"; do
        [[ -z "$user" ]] && continue
        if id "$user" &>/dev/null; then
            echo "    ⚠ $user already exists — will propagate to remote"
            uid=$(id -u "$user")
            USERS_TO_PROPAGATE+=("${user},${uid},/fsx/${user}")
        else
            echo "    ✓ $user — will be created (auto-assign UID)"
            USERS_DATA+=("${user},,/fsx/${user}")
            USERS_TO_PROPAGATE+=("${user},,/fsx/${user}")
        fi
    done

    read -p "  Make these user(s) sudoer(s)? (y/N): " MAKE_SUDOER
    MAKE_SUDOER=${MAKE_SUDOER:-n}
    MAKE_SUDOER=$(echo "$MAKE_SUDOER" | tr '[:upper:]' '[:lower:]')

elif [[ -f "$SHARED_USERS_FILE" && -s "$SHARED_USERS_FILE" ]]; then
    # --- FILE MODE ---
    echo "  Found $SHARED_USERS_FILE — validating entries..."
    echo ""

    new_count=0
    existing_count=0
    conflict_count=0
    while IFS="," read -r username uid_str home; do
        [[ -z "$username" || "$username" == \#* ]] && continue
        username=$(echo "$username" | tr -d ' ')
        uid_str=$(echo "$uid_str" | tr -d ' ')
        home=$(echo "$home" | tr -d ' ')

        if id "$username" &>/dev/null; then
            echo "    ⚠ $username (UID: $uid_str) — exists locally, will verify on remote nodes"
            USERS_TO_PROPAGATE+=("${username},${uid_str},${home}")
            ((existing_count++))
        elif getent passwd "$uid_str" &>/dev/null; then
            existing=$(getent passwd "$uid_str" | cut -d: -f1)
            echo "    ✗ $username (UID: $uid_str) — UID conflict with '$existing', will skip"
            ((conflict_count++))
        else
            echo "    ✓ $username (UID: $uid_str) — new, will be created"
            USERS_DATA+=("${username},${uid_str},${home}")
            USERS_TO_PROPAGATE+=("${username},${uid_str},${home}")
            ((new_count++))
        fi
    done < "$SHARED_USERS_FILE"

    echo ""
    echo "  New: $new_count | Existing (will propagate): $existing_count | Conflicts: $conflict_count"

    if [[ $new_count -eq 0 && $existing_count -eq 0 ]]; then
        echo "  No users to create or propagate. Exiting."
        exit 0
    fi

    if [[ $new_count -gt 0 ]]; then
        read -p "  Create $new_count new user(s) on current node and propagate to all remote nodes? [Y/n]: " confirm
        confirm=${confirm:-Y}
        [[ ! "$confirm" =~ ^[Yy] ]] && { USERS_DATA=(); echo "  Skipping new users from file."; }
    else
        echo "  No new users to create locally. Existing user(s) will be verified on remote nodes."
    fi

    read -p "  Make user(s) sudoer(s)? (y/N): " MAKE_SUDOER
    MAKE_SUDOER=${MAKE_SUDOER:-n}
    MAKE_SUDOER=$(echo "$MAKE_SUDOER" | tr '[:upper:]' '[:lower:]')

    # Checking if more users needs to be added which are not part of the shared_users.txt, this will enter interactive mode
    echo ""
    read -p "  Add additional users not in the file? [y/N]: " ADD_MORE
    ADD_MORE=$(echo "${ADD_MORE:-n}" | tr '[:upper:]' '[:lower:]')
    if [[ "$ADD_MORE" == "y" ]]; then
        read -p "  Enter username(s), comma-separated: " EXTRA_INPUT
        if [[ -n "$EXTRA_INPUT" ]]; then
            IFS=',' read -ra EXTRA_USERS <<< "$EXTRA_INPUT"
            EXTRA_USERS=("${EXTRA_USERS[@]// /}")

            echo ""
            read -p "  Specify UIDs for these users? (Enter for auto-assign, or comma-separated): " EXTRA_UID_INPUT
            EXTRA_UIDS=()
            if [[ -n "$EXTRA_UID_INPUT" ]]; then
                IFS=',' read -ra EXTRA_UIDS <<< "$EXTRA_UID_INPUT"
                EXTRA_UIDS=("${EXTRA_UIDS[@]// /}")
                if [[ ${#EXTRA_UIDS[@]} -ne ${#EXTRA_USERS[@]} ]]; then
                    echo "  [WARN] UID count mismatch — will auto-assign UIDs for extra users"
                    EXTRA_UIDS=()
                fi
            fi

            for i in "${!EXTRA_USERS[@]}"; do
                user="${EXTRA_USERS[$i]}"
                [[ -z "$user" ]] && continue
                if id "$user" &>/dev/null; then
                    echo "    ⚠ $user already exists locally — will propagate to remote"
                    uid=$(id -u "$user")
                    USERS_TO_PROPAGATE+=("${user},${uid},/fsx/${user}")
                elif [[ ${#EXTRA_UIDS[@]} -gt 0 ]] && getent passwd "${EXTRA_UIDS[$i]}" &>/dev/null; then
                    echo "    ✗ UID ${EXTRA_UIDS[$i]} conflict — skipping $user"
                else
                    echo "    ✓ $user — will be created"
                    uid_flag=""
                    [[ ${#EXTRA_UIDS[@]} -gt 0 ]] && uid_flag="${EXTRA_UIDS[$i]}"
                    # Store with uid_flag or empty (auto-assign handled in Step 2)
                    USERS_DATA+=("${user},${uid_flag},/fsx/${user}")
                    USERS_TO_PROPAGATE+=("${user},${uid_flag},/fsx/${user}")
                fi
            done
        fi
    fi

elif [[ -f "shared_users.yaml" ]]; then
    # --- YAML FILE MODE ---
    echo "  Found shared_users.yaml — parsing with Python..."

    if ! python3 -c "import yaml" &>/dev/null; then
        echo "[ERROR] PyYAML not available. Install with: pip3 install pyyaml"
        exit 1
    fi

    # Extract username,uid pairs from YAML (supports both simple 'users:' and 'groups:' format)
    yaml_output=$(python3 -c "
import yaml, sys
try:
    with open('shared_users.yaml') as f:
        config = yaml.safe_load(f)
    if not config:
        sys.exit(0)
    users = config.get('users', [])
    for g in config.get('groups', []):
        users.extend(g.get('users', []))
    for u in users:
        print(f\"{u['username']},{u['uid']},/fsx/{u['username']}\")
except Exception as e:
    print(f'ERROR:{e}', file=sys.stderr)
    sys.exit(1)
" 2>&1)

    if [[ $? -ne 0 ]]; then
        echo "[ERROR] Failed to parse shared_users.yaml: $yaml_output"
        exit 1
    fi

    if [[ -z "$yaml_output" ]]; then
        echo "  No users defined in shared_users.yaml."
        exit 0
    fi

    new_count=0
    existing_count=0
    while IFS="," read -r username uid_str home; do
        [[ -z "$username" ]] && continue
        if id "$username" &>/dev/null; then
            echo "    ⚠ $username (UID: $uid_str) — exists locally, will verify on remote nodes"
            USERS_TO_PROPAGATE+=("${username},${uid_str},${home}")
            ((existing_count++))
        elif getent passwd "$uid_str" &>/dev/null; then
            existing=$(getent passwd "$uid_str" | cut -d: -f1)
            echo "    ✗ $username (UID: $uid_str) — UID conflict with '$existing', will skip"
        else
            echo "    ✓ $username (UID: $uid_str) — new, will be created"
            USERS_DATA+=("${username},${uid_str},${home}")
            USERS_TO_PROPAGATE+=("${username},${uid_str},${home}")
            ((new_count++))
        fi
    done <<< "$yaml_output"

    echo ""
    echo "  New: $new_count | Existing (will propagate): $existing_count"

    if [[ $new_count -eq 0 && $existing_count -eq 0 ]]; then
        echo "  No users to create or propagate. Exiting."
        exit 0
    fi

    if [[ $new_count -gt 0 ]]; then
        read -p "  Create $new_count new user(s) on current node and propagate to all remote nodes? [Y/n]: " confirm
        confirm=${confirm:-Y}
        [[ ! "$confirm" =~ ^[Yy] ]] && { USERS_DATA=(); echo "  Skipping new users from YAML."; }
    fi

    read -p "  Make user(s) sudoer(s)? (y/N): " MAKE_SUDOER
    MAKE_SUDOER=${MAKE_SUDOER:-n}
    MAKE_SUDOER=$(echo "$MAKE_SUDOER" | tr '[:upper:]' '[:lower:]')

else
    # --- INTERACTIVE MODE ---
    if [[ -f "$SHARED_USERS_FILE" ]]; then
        echo "  $SHARED_USERS_FILE is empty."
    else
        echo "  No $SHARED_USERS_FILE or shared_users.yaml found."
    fi
    echo "  Entering interactive mode..."
    echo ""

    read -p "  Enter username(s), comma-separated (e.g. 'sean' or 'sean,alice,bob'): " USER_INPUT
    if [[ -z "$USER_INPUT" ]]; then
        echo "[ERROR] Username(s) cannot be empty."
        exit 1
    fi

    IFS=',' read -ra USERS <<< "$USER_INPUT"
    USERS=("${USERS[@]// /}")

    # Validate usernames
    for user in "${USERS[@]}"; do
        [[ -z "$user" ]] && echo "[ERROR] Empty username detected." && exit 1
        id "$user" &>/dev/null && echo "[ERROR] User '$user' already exists." && exit 1
    done

    # Optional UIDs
    echo ""
    read -p "  Specify UIDs? (Enter for auto-assign, or comma-separated UIDs): " UID_INPUT
    USER_UIDS=()
    if [[ -n "$UID_INPUT" ]]; then
        IFS=',' read -ra USER_UIDS <<< "$UID_INPUT"
        USER_UIDS=("${USER_UIDS[@]// /}")
        [[ ${#USER_UIDS[@]} -ne ${#USERS[@]} ]] && echo "[ERROR] UID count doesn't match username count." && exit 1
        for i in "${!USER_UIDS[@]}"; do
            uid="${USER_UIDS[$i]}"
            [[ ! "$uid" =~ ^[0-9]+$ ]] && echo "[ERROR] Invalid UID '$uid'." && exit 1
            if getent passwd "$uid" &>/dev/null; then
                existing=$(getent passwd "$uid" | cut -d: -f1)
                echo "[ERROR] UID $uid already in use by '$existing'."
                exit 1
            fi
        done
    fi

    read -p "  Make these user(s) sudoer(s)? (y/N): " MAKE_SUDOER
    MAKE_SUDOER=${MAKE_SUDOER:-n}
    MAKE_SUDOER=$(echo "$MAKE_SUDOER" | tr '[:upper:]' '[:lower:]')
fi

# ===================================================================
# Step 2: Create users locally on current node
# ===================================================================
echo ""
echo "========================================"
echo " Step 2: Create users on current node"
echo "========================================"

# Track created users for summary
declare -a CREATED_USERS_DATA=()

if [[ ${#USERS_DATA[@]} -gt 0 ]]; then
    # FILE MODE — create from validated USERS_DATA
    for entry in "${USERS_DATA[@]}"; do
        IFS=',' read -r user uid_str fsx_dir <<< "$entry"
        if [[ -n "$uid_str" ]]; then uid_flag="--uid $uid_str"; else uid_flag=""; fi

        if [[ "$USE_OPENZFS" == true ]]; then
            home_dir="/home/$user"
            useradd "$user" -m -d "$home_dir" $uid_flag --shell /bin/bash 2>/dev/null
        else
            home_dir="$fsx_dir"
            useradd "$user" -m -d "$home_dir" $uid_flag --shell /bin/bash 2>/dev/null
        fi

        # Verify user was actually created before proceeding
        if ! id -u "$user" &>/dev/null; then
            echo "  ✗ Failed to create user '$user' — skipping"
            continue
        fi

        uid=$(id -u "$user")
        [[ "$USE_OPENZFS" == true ]] && { mkdir -p "$fsx_dir"; chown "$user":"$user" "$fsx_dir"; }
        CREATED_USERS_DATA+=("${user},${uid},${fsx_dir}")
        echo "  ✓ User '$user' created with UID $uid (home: $home_dir)"

        getent group docker &>/dev/null && usermod -aG docker "$user" && echo "    Added to docker group"
        [[ "$MAKE_SUDOER" == "y" ]] && { usermod -aG sudo "$user" 2>/dev/null || usermod -aG wheel "$user" 2>/dev/null || true; echo "    Added to sudoer group"; }
    done
elif [[ ${USERS+x} && ${#USERS[@]} -gt 0 ]]; then
    # INTERACTIVE/CLI MODE — create from USERS array (only new users)
    for i in "${!USERS[@]}"; do
        user="${USERS[$i]}"
        # Skip users that already exist locally (CLI mode propagation-only users)
        if id "$user" &>/dev/null; then
            echo "  ⚠ User '$user' already exists locally (will propagate to remote)"
            continue
        fi
        fsx_dir="/fsx/$user"
        uid_flag=""
        [[ ${#USER_UIDS[@]} -gt 0 ]] && uid_flag="--uid ${USER_UIDS[$i]}"

        if [[ "$USE_OPENZFS" == true ]]; then
            home_dir="/home/$user"
            useradd "$user" -m -d "$home_dir" $uid_flag --shell /bin/bash 2>/dev/null
        else
            home_dir="$fsx_dir"
            useradd "$user" -m -d "$home_dir" $uid_flag --shell /bin/bash 2>/dev/null
        fi

        # Verify user was actually created before proceeding
        if ! id -u "$user" &>/dev/null; then
            echo "  ✗ Failed to create user '$user' — skipping"
            continue
        fi

        uid=$(id -u "$user")
        [[ "$USE_OPENZFS" == true ]] && { mkdir -p "$fsx_dir"; chown "$user":"$user" "$fsx_dir"; }
        CREATED_USERS_DATA+=("${user},${uid},${fsx_dir}")
        echo "  ✓ User '$user' created with UID $uid (home: $home_dir)"

        getent group docker &>/dev/null && usermod -aG docker "$user" && echo "    Added to docker group"
        [[ "$MAKE_SUDOER" == "y" ]] && { usermod -aG sudo "$user" 2>/dev/null || usermod -aG wheel "$user" 2>/dev/null || true; echo "    Added to sudoer group"; }
    done
fi

if [[ ${#CREATED_USERS_DATA[@]} -eq 0 && ${#USERS_TO_PROPAGATE[@]} -eq 0 ]]; then
    echo "  No users were created or need propagation. Exiting."
    exit 0
fi

if [[ ${#CREATED_USERS_DATA[@]} -eq 0 && ${#USERS_TO_PROPAGATE[@]} -gt 0 ]]; then
    echo "  No new users created locally. Will propagate existing users to remote nodes."
fi

# Backfill empty UIDs in USERS_TO_PROPAGATE with actual UIDs from local creation
# This ensures remote nodes get the same UID assigned locally (Making sure UID stays consistent across cluster)
for i in "${!USERS_TO_PROPAGATE[@]}"; do
    IFS=',' read -r p_user p_uid p_home <<< "${USERS_TO_PROPAGATE[$i]}"
    if [[ -z "$p_uid" ]]; then
        for entry in "${CREATED_USERS_DATA[@]}"; do
            IFS=',' read -r c_user c_uid c_home <<< "$entry"
            if [[ "$c_user" == "$p_user" ]]; then
                USERS_TO_PROPAGATE[$i]="${p_user},${c_uid},${p_home}"
                break
            fi
        done
    fi
done

# ===================================================================
# Step 3: Setup SSH keypairs
# ===================================================================
echo ""
echo "========================================"
echo " Step 3: Setup SSH keypairs"
echo "========================================"
for entry in "${CREATED_USERS_DATA[@]}"; do
    IFS=',' read -r user uid fsx_dir <<< "$entry"
    ssh_dir="/fsx/$user/.ssh"

    # Ensure .ssh directory exists
    mkdir -p "$ssh_dir" 2>/dev/null || true
    chown "$user":"$user" "$ssh_dir" 2>/dev/null || true
    chmod 700 "$ssh_dir" 2>/dev/null || true

    # Generate keypair (skip if already exists)
    if [[ -f "$ssh_dir/id_rsa" ]]; then
        echo "  ⚠ SSH keypair already exists for $user (skipped)"
    else
        sudo -u "$user" ssh-keygen -t rsa -q -f "$ssh_dir/id_rsa" -N "" 2>/dev/null || true

        if [[ -f "$ssh_dir/id_rsa.pub" ]]; then
            sudo -u "$user" cp "$ssh_dir/id_rsa.pub" "$ssh_dir/authorized_keys" 2>/dev/null || true
            chmod 600 "$ssh_dir/authorized_keys" 2>/dev/null || true
            chown "$user":"$user" "$ssh_dir/authorized_keys" 2>/dev/null || true
            echo "  ✓ SSH keypair created for $user"
        else
            echo "  ⚠ SSH keypair generation failed for $user (home dir may not exist on shared storage)"
        fi
    fi
done

# ===================================================================
# Step 4: Create users on remote nodes
# ===================================================================

# Build remote node lists (exclude current node)
REMOTE_CONTROLLER=(); for n in "${CONTROLLER_NODES[@]}"; do [[ "$n" != "$CURRENT_HOST" ]] && REMOTE_CONTROLLER+=("$n"); done
REMOTE_LOGIN=();      for n in "${LOGIN_NODES[@]}";      do [[ "$n" != "$CURRENT_HOST" ]] && REMOTE_LOGIN+=("$n");      done
REMOTE_COMPUTE=();    for n in "${COMPUTE_NODES[@]}";    do [[ "$n" != "$CURRENT_HOST" ]] && REMOTE_COMPUTE+=("$n");    done

echo ""
echo "========================================"
echo " Step 4: Create users on remote nodes"
echo "========================================"

# Determine which users to propagate to remote nodes
# In file mode: USERS_TO_PROPAGATE includes both new + existing-locally users
# In interactive mode: only newly created users (CREATED_USERS_DATA)
if [[ ${#USERS_TO_PROPAGATE[@]} -gt 0 ]]; then
    REMOTE_USERS=("${USERS_TO_PROPAGATE[@]}")
else
    REMOTE_USERS=("${CREATED_USERS_DATA[@]}")
fi

# Controller (sequential)
if [[ ${#REMOTE_CONTROLLER[@]} -gt 0 ]]; then
    for node in "${REMOTE_CONTROLLER[@]}"; do
        create_users_on_remote "$node" "controller" "$MAKE_SUDOER" "${REMOTE_USERS[@]}"
    done
fi

# Login (sequential)
if [[ ${#REMOTE_LOGIN[@]} -gt 0 ]]; then
    for node in "${REMOTE_LOGIN[@]}"; do
        create_users_on_remote "$node" "login" "$MAKE_SUDOER" "${REMOTE_USERS[@]}"
    done
fi

# Compute (parallel)
if [[ ${#REMOTE_COMPUTE[@]} -gt 0 ]]; then
    echo "  Compute nodes (${#REMOTE_COMPUTE[@]}, parallel max=$MAX_PARALLEL):"
    running=0
    for node in "${REMOTE_COMPUTE[@]}"; do
        create_users_on_remote "$node" "compute" "$MAKE_SUDOER" "${REMOTE_USERS[@]}" &
        ((running++)) || true
        if ((running >= MAX_PARALLEL)); then
            wait -n 2>/dev/null || true
            ((running--)) || true
        fi
    done
    wait || true
fi

TOTAL_REMOTE=$(( ${#REMOTE_CONTROLLER[@]} + ${#REMOTE_LOGIN[@]} + ${#REMOTE_COMPUTE[@]} ))
[[ $TOTAL_REMOTE -eq 0 ]] && echo "  No remote nodes to configure."

# ===================================================================
# Step 5: Slurm accounting (add users to Slurm on controller)
# ===================================================================
echo ""
echo "========================================"
echo " Step 5: Slurm accounting"
echo "========================================"

CONTROLLER_HOST="${CONTROLLER_NODES[0]:-}"
SLURM_ACCT_SUCCESS=false

if [[ -n "$CONTROLLER_HOST" ]]; then
    # Build sacctmgr commands for all new users
    slurm_cmd="sacctmgr -i add account root Description='Root Account' 2>/dev/null || true"
    for entry in "${CREATED_USERS_DATA[@]}"; do
        IFS=',' read -r user uid home <<< "$entry"
        slurm_cmd+="; sacctmgr -i add user $user account=root 2>/dev/null || true"
    done

    if [[ "$CURRENT_HOST" == "$CONTROLLER_HOST" ]]; then
        echo "  Running sacctmgr locally (this is the controller)..."
        eval "$slurm_cmd" && SLURM_ACCT_SUCCESS=true
    else
        echo "  Running sacctmgr on controller ($CONTROLLER_HOST) via SSH..."
        sudo -u "$SSH_USER" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$CONTROLLER_HOST" "$slurm_cmd" 2>&1 && SLURM_ACCT_SUCCESS=true
    fi

    if [[ "$SLURM_ACCT_SUCCESS" == true ]]; then
        echo "  ✓ Added ${#CREATED_USERS_DATA[@]} user(s) to Slurm accounting (account: root)"
    else
        echo "  ✗ Slurm accounting failed — users may need to be added manually:"
        echo "    sacctmgr -i add user <username> account=root"
    fi
else
    echo "  ⚠ No controller node found — skipping Slurm accounting"
fi

# ===================================================================
# Step 6: Update shared_users.txt + optional S3 upload
# ===================================================================
echo ""
echo "========================================"
echo " Step 6: Update shared_users.txt"
echo "========================================"

appended_count=0
for entry in "${CREATED_USERS_DATA[@]}"; do
    IFS=',' read -r user uid home <<< "$entry"
    # Check for duplicate by username or UID before appending
    if grep -q "^${user}," "$SHARED_USERS_FILE" 2>/dev/null; then
        echo "  ⚠ $user already in file — skipping"
    elif grep -q ",${uid}," "$SHARED_USERS_FILE" 2>/dev/null; then
        echo "  ⚠ UID $uid already in file — skipping"
    else
        echo "$entry" >> "$SHARED_USERS_FILE"
        echo "  + $entry"
        ((appended_count++)) || true
    fi
done
echo "  ✓ Appended $appended_count user(s) to $SHARED_USERS_FILE"
echo ""
read -p "  Upload shared_users.txt to S3? Enter bucket URI (or Enter to skip): " S3_BUCKET_URI
S3_BUCKET_URI="${S3_BUCKET_URI%/}"
S3_UPLOAD_SUCCESS=false

if [[ -n "$S3_BUCKET_URI" ]]; then
    echo "  Uploading..."
    if sudo -u "$SSH_USER" aws s3 cp "$SHARED_USERS_FILE" "${S3_BUCKET_URI}/shared_users.txt" 2>&1; then
        echo "  ✓ Uploaded to ${S3_BUCKET_URI}/shared_users.txt"
        S3_UPLOAD_SUCCESS=true
    else
        echo "  ✗ Upload failed. Manual command:"
        echo "    aws s3 cp $SHARED_USERS_FILE ${S3_BUCKET_URI}/shared_users.txt"
    fi
fi

# ===================================================================
# Step 7: Summary
# ===================================================================
echo ""
echo "========================================"
echo " ✓ ${#CREATED_USERS_DATA[@]} user(s) created!"
echo "========================================"
echo ""
echo "  Users created:"
for entry in "${CREATED_USERS_DATA[@]}"; do
    IFS=',' read -r user uid home <<< "$entry"
    echo "    $user (UID: $uid, Home: $home)"
done
echo ""
echo "  Sudoer: $([[ "$MAKE_SUDOER" == "y" ]] && echo "Yes" || echo "No")"
echo ""
echo "  Nodes configured:"
echo "    Current:    $CURRENT_HOST ($CURRENT_ROLE) ✓"
[[ ${#REMOTE_CONTROLLER[@]} -gt 0 ]] && echo "    Controller: ${REMOTE_CONTROLLER[*]}"
[[ ${#REMOTE_LOGIN[@]} -gt 0 ]]      && echo "    Login:      ${REMOTE_LOGIN[*]}"
[[ ${#REMOTE_COMPUTE[@]} -gt 0 ]]    && echo "    Compute:    ${#REMOTE_COMPUTE[@]} node(s)"
echo ""
echo "  Slurm acct:   $([[ "$SLURM_ACCT_SUCCESS" == true ]] && echo "✓" || echo "⚠ manual setup needed")"
echo "  Shared users: $SHARED_USERS_FILE ✓"
[[ "$S3_UPLOAD_SUCCESS" == true ]] && echo "  S3 upload:    ${S3_BUCKET_URI}/shared_users.txt ✓"
[[ "$S3_UPLOAD_SUCCESS" != true && -z "$S3_BUCKET_URI" ]] && echo "  S3 upload:    skipped — run: aws s3 cp $SHARED_USERS_FILE s3://<bucket>/shared_users.txt"

if [[ "$MAKE_SUDOER" == "y" ]]; then
    echo ""
    echo "  NOTE: For passwordless sudo, run on each node:"
    echo "    sudo visudo → change %sudo ALL=(ALL:ALL) ALL to %sudo ALL=(ALL:ALL) NOPASSWD: ALL"
fi
echo ""
echo "========================================"
