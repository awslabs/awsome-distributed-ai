#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

declare -a HELP=(
    "[-h|--help]"
    "[-c|--controller-group]"
    "[-u|--user]"
    "[-k|--key SSH_KEY_PATH]"
    "[-r|--region]"
    "[-p|--profile]"
    "[-d|--dry-run]"
    "CLUSTER_NAME"
)

cluster_name=""
node_group="controller-machine"
ssh_user="ubuntu"
ssh_key=""
declare -a aws_cli_args=()
DRY_RUN=0

parse_args() {
    local key
    while [[ $# -gt 0 ]]; do
        key="$1"
        case $key in
        -h|--help)
            echo "Access a HyperPod Slurm controller via ssh-over-ssm."
            echo "Usage: $(basename ${BASH_SOURCE[0]}) ${HELP[@]}"
            echo ""
            echo "Options:"
            echo "  -h, --help              Show this help message"
            echo "  -c, --controller-group  Specify the controller group name (default: controller-machine)"
            echo "  -u, --user              Specify the SSH user (default: ubuntu)"
            echo "  -k, --key               Specify path to SSH public key (default: auto-detect)"
            echo "  -r, --region            Specify AWS region"
            echo "  -p, --profile           Specify AWS profile"
            echo "  -d, --dry-run           Show the SSM command without executing"
            echo ""
            echo "Examples:"
            echo "  $(basename ${BASH_SOURCE[0]}) ml-cluster"
            echo "  $(basename ${BASH_SOURCE[0]}) -c login-group ml-cluster"
            echo "  $(basename ${BASH_SOURCE[0]}) -u user1 ml-cluster"
            echo "  $(basename ${BASH_SOURCE[0]}) -u user2 -r us-west-2 ml-cluster"
            echo "  $(basename ${BASH_SOURCE[0]}) -k ~/.ssh/id_ed25519.pub ml-cluster"
            echo ""
            echo "Note: For non-ubuntu users, ensure your IAM user has the SSMSessionRunAs tag"
            echo "      set to the desired OS username for passwordless login."
            exit 0
            ;;
        -c|--controller-group)
            node_group="$2"
            shift 2
            ;;
        -u|--user)
            ssh_user="$2"
            shift 2
            ;;
        -k|--key)
            if [[ -z "${2:-}" || "$2" == -* ]]; then
                echo "Error: -k/--key requires an argument" ; exit 1
            fi
            ssh_key="$2"
            shift 2
            ;;
        -r|--region)
            aws_cli_args+=(--region "$2")
            shift 2
            ;;
        -p|--profile)
            aws_cli_args+=(--profile "$2")
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=1
            shift
            ;;
        *)
            [[ "$cluster_name" == "" ]] \
                && cluster_name="$key" \
                || { echo "Must define one cluster name only" ; exit -1 ;  }
            shift
            ;;
        esac
    done

    [[ "$cluster_name" == "" ]] && { echo "Must define a cluster name" ; exit -1 ;  }
}

# Function to check if cluster config exists in ~/.ssh/config
check_ssh_config() {
    local ssh_host="${cluster_name}"
    
    # If user is not ubuntu, append username to host for unique config entry
    if [[ "$ssh_user" != "ubuntu" ]]; then
        ssh_host="${cluster_name}-${ssh_user}"
    fi
    
    if grep -wq "Host ${ssh_host}$" ~/.ssh/config; then
        echo -e "${BLUE}1. Detected ${GREEN}${ssh_host}${BLUE} in ${GREEN}~/.ssh/config${BLUE}. Skipping adding...${NC}"
    else
        echo -e "${BLUE}Would you like to add ${GREEN}${ssh_host}${BLUE} to ~/.ssh/config (yes/no)?${NC}"
        read -p "> " ADD_CONFIG

        if [[ $ADD_CONFIG == "yes" ]]; then
            if [ ! -f ~/.ssh/config ]; then
                mkdir -p ~/.ssh
                touch ~/.ssh/config
            fi
            # Derive private key path from the public key (strip .pub if present)
            local identity_file="${ssh_key%.pub}"
            echo -e "${GREEN}✅ adding ${ssh_host} to ~/.ssh/config:${NC}"
            cat <<EOL >> ~/.ssh/config 
Host ${ssh_host}
    User ${ssh_user}
    IdentityFile ${identity_file}
    ProxyCommand sh -c "aws ssm start-session ${aws_cli_args[@]} --target sagemaker-cluster:${cluster_id}_${node_group}-${instance_id} --document-name AWS-StartSSHSession --parameters 'portNumber=%p'"
EOL
        else
            echo -e "${GREEN}❌ skipping adding ${ssh_host} to ~/.ssh/config${NC}"
        fi      
    fi
    
    # Store the ssh_host for later use
    SSH_HOST="${ssh_host}"
}

escape_spaces() {
    local input="$1"
    echo "${input// /\\ }"
}

# Function to check if a file contains a valid SSH public key
is_public_key() {
    local file="$1"
    [[ ! -f "$file" ]] && return 1
    local first_line
    first_line=$(head -1 "$file")
    if [[ "$first_line" == ssh-rsa\ * ]] || \
       [[ "$first_line" == ssh-ed25519\ * ]] || \
       [[ "$first_line" == ecdsa-sha2-* ]] || \
       [[ "$first_line" == ssh-dss\ * ]]; then
        return 0
    fi
    return 1
}

# Function to detect or validate the SSH public key
detect_ssh_key() {
    # If user specified a key via -k, validate it
    if [[ -n "$ssh_key" ]]; then
        # Expand leading tilde that may not have been shell-expanded (e.g. quoted argument)
        if [[ "$ssh_key" == "~/"* ]]; then
            ssh_key="${HOME}/${ssh_key#\~/}"
        fi
        if [[ ! -f "$ssh_key" ]]; then
            echo "Error: Specified SSH key file not found: ${ssh_key}"
            exit 1
        fi
        if ! is_public_key "$ssh_key"; then
            echo "Error: ${ssh_key} does not appear to be a valid SSH public key."
            exit 1
        fi
        return
    fi

    # Auto-detect: check .pub files first, then files without .pub (content-validated)
    local key_names=("id_ed25519" "id_ecdsa" "id_rsa" "id_dsa")

    # First pass: check for .pub files (content-validated)
    for key_name in "${key_names[@]}"; do
        local candidate="${HOME}/.ssh/${key_name}.pub"
        if is_public_key "$candidate"; then
            ssh_key="$candidate"
            return
        fi
    done

    # Second pass: check files without .pub extension (content-validated)
    for key_name in "${key_names[@]}"; do
        local candidate="${HOME}/.ssh/${key_name}"
        if is_public_key "$candidate"; then
            ssh_key="$candidate"
            return
        fi
    done

    # No key found
    echo "Error: No SSH public key found in ~/.ssh/"
    echo ""
    echo "Generate one with:  ssh-keygen -t <type>"
    echo "  Types: ed25519, ecdsa, rsa, dsa"
    echo "Or specify one with: $(basename ${BASH_SOURCE[0]}) -k /path/to/key.pub CLUSTER_NAME"
    exit 1
}

# Function to add the user's SSH public key to the cluster
add_keypair_to_cluster() {
    PUBLIC_KEY=$(cat "${ssh_key}")
    
    # Determine the authorized_keys path based on user and filesystem
    # Check if OpenZFS is mounted (home directory would be /home/username)
    local auth_keys_path="/fsx/${ssh_user}/.ssh/authorized_keys"
    
    # Try to detect if user home is on OpenZFS
    # Note: </dev/null detaches stdin so the SSM session plugin does not attach to the terminal TTY,
    # which ensures output is captured reliably in a subshell.
    local home_check=$(aws ssm start-session "${aws_cli_args[@]}" --target sagemaker-cluster:${cluster_id}_${node_group}-${instance_id} --document-name AmazonEKS-ExecuteNonInteractiveCommand --parameters command="getent passwd ${ssh_user} | cut -d: -f6" </dev/null 2>/dev/null || echo "")
    
    if echo "$home_check" | grep -q "^/home/${ssh_user}"; then
        # User home is on OpenZFS, but .ssh is symlinked to /fsx
        auth_keys_path="/fsx/${ssh_user}/.ssh/authorized_keys"
    fi

    # Check whether our key is already present, by counting matches REMOTELY and
    # reading back a single integer (see the long note on the verify step below
    # for why we do not cat the whole file back and grep locally -- the SSM
    # non-interactive read truncates large output). Match on the base64 key body
    # (field 2, no spaces) so the command carries no spaces/quotes/&&/||.
    local existing_blob existing_count="" existing_tries=0
    existing_blob=$(awk '{print $2}' <<<"$PUBLIC_KEY")
    while [[ $existing_tries -lt 5 ]]; do
        existing_tries=$((existing_tries+1))
        local existing_out
        existing_out=$(aws ssm start-session "${aws_cli_args[@]}" --target sagemaker-cluster:${cluster_id}_${node_group}-${instance_id} --document-name AmazonEKS-ExecuteNonInteractiveCommand --parameters command="grep -cF ${existing_blob} ${auth_keys_path}" </dev/null 2>/dev/null)
        existing_count=$(printf '%s\n' "$existing_out" | grep -Eo '^[0-9]+$' | head -1)
        [[ -n "$existing_count" ]] && break
        sleep 2
    done

    if [[ -z "$existing_count" ]]; then
        echo -e "${YELLOW}Warning: Could not read authorized_keys from the cluster (SSM command returned no readable result).${NC}"
        echo -e "${YELLOW}The cluster node may not be connected to SSM. Skipping key upload.${NC}"
        echo -e "${YELLOW}You may need to manually add your public key to ${auth_keys_path} on the cluster.${NC}"
        return 1
    fi

    if [[ "$existing_count" -ge 1 ]]; then
        echo -e "${BLUE}2. Detected SSH public key ${GREEN}${ssh_key}${BLUE} for user ${GREEN}${ssh_user}${BLUE} on the cluster. Skipping adding...${NC}"
        return
    else
        echo -e "${BLUE}2. Do you want to add your SSH public key ${GREEN}${ssh_key}${BLUE} to user ${GREEN}${ssh_user}${BLUE} on the cluster (yes/no)?${NC}" 
        read -p "> " ADD_KEYPAIR
        if [[ $ADD_KEYPAIR == "yes" ]]; then
            echo "Adding ... ${PUBLIC_KEY}"
            command="sed -i \$a$(escape_spaces "$PUBLIC_KEY") ${auth_keys_path}"
            aws ssm start-session "${aws_cli_args[@]}" --target sagemaker-cluster:${cluster_id}_${node_group}-${instance_id}  --document-name AmazonEKS-ExecuteNonInteractiveCommand  --parameters command="$command" </dev/null >/dev/null 2>/dev/null

            # Verify by counting matches REMOTELY and returning just the number.
            #
            # Do NOT cat the whole authorized_keys back and grep locally: the
            # AmazonEKS-ExecuteNonInteractiveCommand output over start-session is
            # unreliable for large / multi-line payloads -- it intermittently comes
            # back as banner-only or truncated mid-line, which made the old
            # `cat | grep -Fq "$PUBLIC_KEY"` check report a false "key not found"
            # even though the key was written correctly.
            #
            # Instead run `grep -cF <blob>` on the node and read back a single
            # integer. We match on the base64 key body (field 2, no spaces) so the
            # command carries no spaces, quotes, && or || -- all of which either
            # break the AWS CLI --parameters shorthand parser or are not honored by
            # the document. The blob is unique per key, so the count is authoritative.
            local key_blob
            key_blob=$(awk '{print $2}' <<<"$PUBLIC_KEY")
            local count="" tries=0
            while [[ $tries -lt 5 ]]; do
                tries=$((tries+1))
                local out
                out=$(aws ssm start-session "${aws_cli_args[@]}" --target sagemaker-cluster:${cluster_id}_${node_group}-${instance_id} --document-name AmazonEKS-ExecuteNonInteractiveCommand --parameters command="grep -cF ${key_blob} ${auth_keys_path}" </dev/null 2>/dev/null)
                # Pull the first standalone integer out of the (banner-wrapped) output.
                count=$(printf '%s\n' "$out" | grep -Eo '^[0-9]+$' | head -1)
                [[ -n "$count" ]] && break
                sleep 2
            done

            if [[ -n "$count" && "$count" -ge 1 ]]; then
                echo "✅ Your SSH public key ${ssh_key} has been added to user ${ssh_user} on the cluster."
            elif [[ -z "$count" ]]; then
                # Never got a numeric answer back -- inconclusive, NOT a confirmed failure.
                echo -e "${YELLOW}ℹ️  Added your SSH public key, but could not auto-confirm it (the cluster did not return a readable result after ${tries} tries).${NC}"
                echo -e "${YELLOW}    This is usually harmless. If '${GREEN}ssh ${SSH_HOST}${YELLOW}' works, you are connected.${NC}"
            else
                # Got a definitive 0 back -- the key really is not in authorized_keys.
                echo -e "${RED}Error: Failed to add SSH public key to the cluster. The key was not found after writing.${NC}"
                echo -e "${YELLOW}You may need to manually add your public key to ${auth_keys_path} on the cluster.${NC}"
            fi
        else
            echo "❌ Skipping adding SSH public key to the cluster."
        fi
    fi
}

parse_args $@

#===Style Definitions===
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print a yellow header
print_header() {
    echo -e "\n${BLUE}=================================================${NC}"
    echo -e "\n${YELLOW}==== $1 ====${NC}\n"
    echo -e "\n${BLUE}=================================================${NC}"

}


print_header "🚀 HyperPod Cluster Easy SSH Script! 🚀"

detect_ssh_key

cluster_id=$(aws sagemaker describe-cluster "${aws_cli_args[@]}" --cluster-name $cluster_name | jq '.ClusterArn' | awk -F/ '{gsub(/"/, "", $NF); print $NF}')
instance_id=$(aws sagemaker list-cluster-nodes "${aws_cli_args[@]}" --cluster-name $cluster_name --instance-group-name-contains ${node_group} | jq '.ClusterNodeSummaries[0].InstanceId' | tr -d '"')

# Exit immediately if cluster or instance ID is not found.
if [[ -z "$cluster_id" || -z "$instance_id" ]]; then
    echo "Error: Cluster or instance not found for the specified cluster name (${cluster_name}). Exiting."
    exit 1
fi

# print_header
echo -e "Cluster id: ${GREEN}${cluster_id}${NC}"
echo -e "Instance id: ${GREEN}${instance_id}${NC}"
echo -e "Node Group: ${GREEN}${node_group}${NC}"
echo -e "SSH User: ${GREEN}${ssh_user}${NC}"
echo -e "SSH Key: ${GREEN}${ssh_key}${NC}"

check_ssh_config
add_keypair_to_cluster

echo -e "\nNow you can run:\n"
echo -e "$ ${GREEN}ssh ${SSH_HOST}${NC}"

[[ DRY_RUN -eq 1 ]] && echo -e  "\n${GREEN}aws ssm start-session "${aws_cli_args[@]}" --target sagemaker-cluster:${cluster_id}_${node_group}-${instance_id}${NC}\n" && exit 0

# Determine which SSM document to use based on the user
if [[ "$ssh_user" == "ubuntu" ]]; then
    # Start session as Ubuntu if the SSM-SessionManagerRunShellAsUbuntu document exists
    if aws ssm describe-document "${aws_cli_args[@]}" --name SSM-SessionManagerRunShellAsUbuntu > /dev/null 2>&1; then
        aws ssm start-session "${aws_cli_args[@]}" --target sagemaker-cluster:${cluster_id}_${node_group}-${instance_id} --document SSM-SessionManagerRunShellAsUbuntu
    else
        aws ssm start-session "${aws_cli_args[@]}" --target sagemaker-cluster:${cluster_id}_${node_group}-${instance_id}
    fi
else
    # For non-ubuntu users, check if they have an IAM user with SSMSessionRunAs tag
    echo -e "${BLUE}Connecting as user: ${GREEN}${ssh_user}${NC}"
    echo -e "${YELLOW}Note: Make sure the IAM user has the SSMSessionRunAs tag set to '${ssh_user}' for passwordless login${NC}"
    echo -e "${YELLOW}See: https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-getting-started-enable-ssh-connections.html${NC}"
    
    # Use standard SSM session - the SSMSessionRunAs tag on the IAM user will determine which OS user to use
    aws ssm start-session "${aws_cli_args[@]}" --target sagemaker-cluster:${cluster_id}_${node_group}-${instance_id}
fi
