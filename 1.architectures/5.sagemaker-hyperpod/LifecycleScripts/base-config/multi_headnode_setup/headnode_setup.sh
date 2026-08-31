# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#!/bin/bash

: <<'SUMMARY'
Script: headnode_setup.sh

Purpose:
This script performs the setup and configuration of a Slurm head node in a multi-head node
cluster environment.

Key Functions:
- should_wait_to_start: It sets up an internal lock to prevent race condition on multiple controller nodes trying to run this script at the same time.
- get_region_from_arn: Extracts the AWS region from an ARN.
- extract_value:  Helper function to extract a specific value from a JSON string.
- get_cluster_information: Retrieves cluster configuration details.
- create_shared_cluster_directory: Sets up a shared directory for the cluster.
- validate_symlink:  Validates if a symbolic link exists and creates it if not.
- create_etc_and_spool_symlink: create etc and spool folder symbolic link
- create_symlink: Moves configuration files to a shared location, create symbolic link.
- create_support_folders:  Creates support folders for Slurm.
- get_slurm_database_secret: Retrieves the Slurm database secret from AWS Secrets Manager.
- add_accounting_to_slurm_config:  Adds accounting database information to the Slurm configuration.
- setup_remote_slurmdbd_database_config: Create the slurmdbd configuration file with remote database endpoint and secret.
- setup_remote_slurmdbd_database_privilege:  Sets up the remote slurmdbd database privilege.
- setup_sns_notification: Sets up Amazon SNS notifications (if configured) to receive notification when headnode failover.
- restart_slurm_services: Restarts Slurm services after configuration.
- stop_slurm_services: Stops Slurm services before configuration.
- get_ip_address: Retrieves the IP address of the current instance.

Usage:
This script is typically executed as part of the AWS HyperPod mutil headnode cluster setup process.
It should be run with appropriate permissions to modify system configurations and
access AWS resources.

Notes:
- Requires AWS CLI and jq to be installed and configured, which it's already included.
- Assumes certain AWS resources (like Secrets Manager, SNS) are properly set up.
- Modifies system configurations, so use with caution in production environments.

For detailed information on each function, refer to the individual function comments
within the script.

SUMMARY

set -ex
set -o pipefail  # trace ERR through pipes
set -o errtrace  # trace ERR through 'time command' and other functions
set -o nounset   ## set -u : exit the script if you try to use an uninitialised variable

# Constants
readonly RESOURCE_CONFIG="resource_config.json"
readonly PROVISIONING_PARAMS="provisioning_parameters.json"

main() {
    local current_host_ip
    local ml_config_dir="/opt/ml/config"
    current_host_ip=$(get_ip_address)
    echo "Starting the multiple head node migration script for instance with ip $current_host_ip..."
    local cluster_info
    cluster_info=$(get_cluster_information "${ml_config_dir}")
    IFS=' ' read -ra cluster_info_array <<< "$cluster_info"
    local cluster_name="${cluster_info_array[0]}"
    local slurm_database_secret_arn="${cluster_info_array[1]}"
    local slurm_database_endpoint="${cluster_info_array[2]}"
    local slurm_shared_directory="${cluster_info_array[3]}"
    local slurm_database_user="${cluster_info_array[4]}"
    local slurm_sns_arn="${cluster_info_array[5]:-}"
    echo "Cluster name: $cluster_name"
    echo "Slurm database secret ARN: $slurm_database_secret_arn"
    echo "Slurm database endpoint: $slurm_database_endpoint"
    echo "Slurm shared directory: $slurm_shared_directory"
    echo "Slurm database user: $slurm_database_user"
    echo "Slurm SNS ARN: $slurm_sns_arn"
    should_wait_to_start "$ml_config_dir" "$current_host_ip" "$slurm_shared_directory" "$cluster_name" $((3 * 60 / 5))
    local destination_folder
    destination_folder=$(create_shared_cluster_directory "$slurm_shared_directory" "$cluster_name")
    stop_slurm_services
    create_support_folders "$destination_folder" "slurm:slurm"
    add_accounting_to_slurm_config "/opt/slurm/etc/accounting.conf" "$slurm_database_user"
    local slurm_database_region
    slurm_database_region=$(get_region_from_arn "$slurm_database_endpoint")
    set +x
    local slurm_database_secret
    slurm_database_secret=$(get_slurm_database_secret "$slurm_database_secret_arn" "$slurm_database_region")
    setup_remote_slurmdbd_database_config "$slurm_database_secret" "$slurm_database_endpoint" "$slurm_database_user" "/opt/slurm/etc/slurmdbd.conf" "slurm:slurm"
    setup_remote_slurmdbd_database_privilege "$slurm_database_secret" "$slurm_database_endpoint" "$slurm_database_user"
    set -x
    update_slurmdbd_service
    # Setup SNS Notification
    if [[ -n "$slurm_sns_arn" ]]; then
        local slurm_sns_region
        slurm_sns_region=$(get_region_from_arn "$slurm_sns_arn")
        setup_sns_notification "$slurm_sns_arn" "$slurm_sns_region" "$destination_folder" "/opt/slurm/etc/slurm.conf" "slurm:slurm"
    else
        echo "Skipping SNS topic creation: SNS ARN: '$slurm_sns_arn' Reason: SNS ARN is not set, is null."
    fi
    create_etc_and_spool_symlink "$destination_folder" "/opt/slurm/etc" "/var/spool/slurmctld"
    restart_slurm_services
    return 0
}

########################################
# The function sets up an internal lock to prevent race condition on multiple controller nodes trying to run
#  this script at the same time. It always let the -1 suffix controller node to execute the script first to create the
#  shared folder and resources.
# Arguments:
#   $1: current host ip
#   $2: slurm shared directory
#   $3: cluster name
# Returns:
#   0 if folder exists, 1 otherwise
########################################
should_wait_to_start() {
    local resource_config_path="${1}/${RESOURCE_CONFIG}"
    local provisioning_params_path="${1}/${PROVISIONING_PARAMS}"
    # use jq to find the controller
    controller_group=$(jq -r '.controller_group' "$provisioning_params_path")
    index_one_node_ip=$(jq -r --arg group "${controller_group}" '
      .InstanceGroups[] |
      select(.Name == $group) |
      .Instances[] |
      select(.InstanceName == ($group + "-1")) |
      .CustomerIpAddress
    ' "$resource_config_path")
    if [ "$index_one_node_ip" == "$2" ]; then
      echo "This is the index 1 controller node. Proceeding with the script immediately."
    else
        local root_dir="${3}/aws/hyperpod/${4}"
        echo "This is not index 1 controller node. Will wait for shared folder created before"
        local max_attempts="$5"  # Check every 5 seconds
        local attempts=0
        while [ ! -d "$root_dir" ]; do
            if (( attempts >= max_attempts )); then
                echo "Timeout: ${root_dir} was not created within $(( max_attempts * 5 )) seconds."
                return 1
            fi
            echo "Waiting for ${root_dir} to be created... ($(( (max_attempts - attempts) * 5 )) seconds remaining)"
            sleep 5
            ((attempts+=1))
        done
        echo "${root_dir} has been created."
        # once started, it should take a few seconds to complete, so give it 15 seconds as buffer
        sleep 15
    fi
}

#######################################
# Get the resource region from arn
# Arguments:
#   arn - resource arn
# Returns:
#   region if exists, 1 otherwise
#######################################
get_region_from_arn() {
    local arn="$1"
    local region

    if [[ "$arn" == arn:* ]]; then
        # Handle ARN
        region=$(echo "$arn" | cut -d':' -f4)
    elif [[ "$arn" == *.rds.amazonaws.com ]]; then
        # Handle RDS endpoint
        region=$(echo "$arn" | sed -E 's/.*\.([a-z]{2}-[a-z]+-[0-9])\.rds\.amazonaws\.com/\1/')
    else
        echo "Unable to extract region from input: $arn" >&2
        return 1
    fi

    # Validate if it looks like a valid AWS region
    if [[ "$region" =~ ^[a-z]{2}-[a-z]+-[0-9]$ ]]; then
        echo "$region"
        return 0
    else
        echo "Extracted value doesn't appear to be a valid region: $region" >&2
        return 1
    fi
}

#######################################
# Helper function extra required file from given json.
# Outputs:
#   if value not found: "" else value
#######################################
extract_value() {
    local value
    value=$(sed -n "s/.*\"$1\": *\"\([^\"]*\)\".*/\1/p" "$2")
    if [ -z "$value" ]; then
        echo ""
    else
        echo "$value"
    fi
}

#######################################
# Retrieves cluster information from resource_config.json and provisioning_parameters.json.
# Arguments:
#   $1 - directory of resource_config.json and provisioning_parameters.json
# Outputs:
#   Prints an array of cluster information
# Returns:
#   0 on success, 1 on failure
#######################################
get_cluster_information() {
    # Get the slurm configuration name from the resource_config.json file
    # create resource config path
    local resource_config_path="${1}/${RESOURCE_CONFIG}"
    local provisioning_params_path="${1}/${PROVISIONING_PARAMS}"
    local cluster_name;
    local slurm_database_secret_arn;
    local slurm_database_endpoint;
    local slurm_shared_directory;
    local slurm_sns_arn;

    cluster_name=$(extract_value "ClusterName" "$resource_config_path")
    slurm_database_secret_arn=$(extract_value "slurm_database_secret_arn" "$provisioning_params_path")
    slurm_database_endpoint=$(extract_value "slurm_database_endpoint" "$provisioning_params_path")
    slurm_shared_directory=$(extract_value "slurm_shared_directory" "$provisioning_params_path")
    slurm_sns_arn=$(extract_value "slurm_sns_arn" "$provisioning_params_path")
    slurm_database_user=$(extract_value "slurm_database_user" "$provisioning_params_path")

    # if slurm_database_user empty set it to admin
    if [ -z "$slurm_database_user"  ]; then
        slurm_database_user="admin"
    fi

    # Check if any required fields are missing
    if [[ -z "$cluster_name" ||
          -z "$slurm_database_secret_arn" ||
          -z "$slurm_database_endpoint" ||
          -z "$slurm_shared_directory" ]]; then
        echo "Error: One or more required fields are missing" >&2
        return 1
    fi

    # create an array to store then
    local cluster_info
    cluster_info=(
        "$cluster_name"
        "$slurm_database_secret_arn"
        "$slurm_database_endpoint"
        "$slurm_shared_directory"
        "$slurm_database_user"
        "$slurm_sns_arn"
    )
    echo "${cluster_info[@]}"
}

#######################################
# Creates a shared cluster directory at the given shared volume.
# Arguments:
#   root_volume_dir - root shared volume path
#   cluster_name - name of the cluster
# Outputs:
#   Prints the path of the created cluster directory
# Returns:
#   0 on success, 1 on failure
#######################################
create_shared_cluster_directory() {
    # Set the target slurm directory
    local root_volume_dir="$1"
    local cluster_name="$2"
    local root_dir="${root_volume_dir}/aws/hyperpod"
    local cluster_dir="$root_dir/$cluster_name"

    # create a folder for the cluster
    if [ -d "$cluster_dir" ]; then
        echo -n "$cluster_dir"
        return 0
    fi

    if ! mkdir -p "$cluster_dir"; then
        echo "Error: Failed to create cluster folder" >&2
        return 1
    fi

    echo -n "$cluster_dir"
    return 0
}

#######################################
# Validate the folder is a symlink to the designated folder
# Arguments:
#   targeted_folder_path - the shared folder that will designated folder will be linked to
#   original_folder_path - the symlink folder
# Returns:
#   linked - already linked to the correct path
#   incorrectly_linked - already linked to an incorrect path
#   not_linked - not yet linked
#######################################
validate_symlink() {
    local targeted_folder_path=$1
    local original_folder_path=$2

    if [ -L "$original_folder_path" ]; then
        local linked_path
        linked_path=$(realpath "$original_folder_path")
        if [ "$linked_path" = "$(realpath "$targeted_folder_path")" ]; then
            echo -n "linked"
        else
            echo -n "incorrectly_linked"
        fi
    else
      echo -n "not_linked"
    fi
}

#######################################
# Create symlink for the given folder
# Arguments:
#   targeted_base_path - the shared based path
#   original_folder_path - the shared folder that will designated folder will be linked to
#   targeted_folder_path - the symlink folder
# Returns:
#   0 on success, 1 on failure
#######################################
create_etc_and_spool_symlink() {
    local targeted_base_path=$1
    local original_etc_path=$2
    local original_spool_path=$3
    # etc
    files_and_folders=$(find "$original_etc_path" -mindepth 1 -maxdepth 1 ! -name "slurmdbd.conf" ! -name "slurmdbd.conf.template")
    # create symlink to etc
    for file_or_folder in $files_and_folders; do
        # skip .bak files
        if [[ "$file_or_folder" == *.bak ]]; then
            echo "Skipping .bak file: $file_or_folder"
            continue
        fi
        if create_symlink "$file_or_folder" "$targeted_base_path"; then
            echo "Successfully processed directory: $file_or_folder"
        else
            echo "Failed to process directory: $file_or_folder"
            return 1
        fi
    done
    # spool
    if create_symlink "$original_spool_path" "$targeted_base_path"; then
        echo "Successfully processed directory: $original_spool_path"
    else
        echo "Failed to process directory: $original_spool_path"
        return 1
    fi
    return 0
}

#######################################
# Migrates the Slurm folder to a designated location, create symlink, and keep the origianl file/folder with .bak
# should handle cases:
#  initial setup:
#     original_folder - exists,
#     targeted_folder - exists.
#       need to remove source, and map destination
#  original_folder - exists and already a symbolic link to targeted_folder
#     should return early with 0
#  original_folder - exists and already a symbolic link to something else
#     should return early with 1
#  original_folder - targeted_folder doesn't exists (handled by previous function)
#     when two instances writing run this function only primary controller will execute the copy command
# Arguments:
#   original_folder_path - the shared folder that will designated folder will be linked to
#   targeted_bash_path - the symlink folder
# Returns:
#   0 on success, 1 on failure or failure because of already symlinked to other dir
#######################################
create_symlink() {
    local original_path=$1
    local targeted_base_path=$2
    local targeted_path="$targeted_base_path$original_path"
    local has_symlink
    has_symlink=$(validate_symlink "$targeted_path" "$original_path")

    if [ "$has_symlink" = "linked" ]; then
        echo "Source folder $original_path is already a symbolic link to $targeted_path."
        return 0
    elif [ "$has_symlink" = "incorrectly_linked" ]; then
        echo "Error: Source folder $original_path is already a symbolic link to a different location." >&2
        return 1
    fi

    # folder doesn't exists, creating and copying to it
    # -a, will copy files and dirs recursively preserver file attributes and symbolic links
    # -n not overwrite any existing files at the destination
    # Check if the original path is a directory or a file
    if [ -d "$original_path" ]; then
        # It's a directory
        # if exists skip
        if [ ! -d "$targeted_path" ]; then
            mkdir -p "$(dirname "$targeted_path")"
            if ! cp -an "$original_path/." "$targeted_path/"; then
                echo "Error: Failed to copy contents of directory $original_path to $targeted_path." >&2
                return 1
            fi
        else
            echo "Directory $targeted_path already exists. Skipping copy."
        fi
    elif [ -f "$original_path" ]; then
        # It's a file
        mkdir -p "$(dirname "$targeted_path")"
        copy_file_helper "$original_path" "$targeted_path"
    else
        echo "Error: $original_path is neither a file nor a directory." >&2
        return 1
    fi
    echo "Successfully copied files from $original_path to $targeted_path"
    # change the original folder as backup
    if ! mv "$original_path" "${original_path}.bak"; then
        echo "Error: Failed to move $original_path to ${original_path}.bak." >&2
        return 1
    fi

    # create link
    if ! ln -s "$targeted_path" "$original_path"; then
        echo "Error: Failed to create symlink for $targeted_path." >&2
        return 1
    fi
}

#######################################
# Create symlink for the given folder
# Arguments:
#   targeted_base_path - the shared based path
#   original_folder_path - the shared folder that will designated folder will be linked to
#   targeted_folder_path - the symlink folder
# Returns:
#   0 on success, 1 on failure
#######################################
copy_file_helper() {
    local original_path=$1
    local targeted_path=$2
    if [ -f "$targeted_path" ]; then
        echo "File $targeted_path already exists. Skipping copy."
        return 0
    fi
    if ! cp -an "$original_path" "$targeted_path"; then
        echo "Error: Failed to copy file $original_path to $targeted_path." >&2
        return 1
    fi
}

#######################################
# Creates additional folders required for multi headnode Slurm setup: such as scripts folder to store slurm
#  related program scripts and shared log files
# Arguments:
#   destination_folder - base folder where new directories should be created
#   owner - owner of the new directories
# Returns:
#   0 on success, 1 on failure
#######################################
create_support_folders() {
    local destination_folder="$1"
    local paths=(
        "$destination_folder/opt/slurm/etc/scripts"
        "$destination_folder/var/log/slurm"
    )
    for path in "${paths[@]}"; do
        if [ ! -d "$path" ]; then
            mkdir -p "$path" && chmod 750 "$path" && chown "$2" "$path"
        fi
    done
    return 0
}

#######################################
# Logic to obtain the rds database secret
# Arguments:
#   secret_arn - secret arn
#   region - region of the secret
# Returns:
#   0 on success, 1 on failure
#######################################
get_slurm_database_secret() {
    local secret_arn="$1"
    local region="$2"

    if [ -z "$secret_arn" ] || [ -z "$region" ]; then
        echo "Error: Secret name and region must be provided" >&2
        return 1
    fi
    local secret_value
    secret_value=$(aws secretsmanager \
        --region "$region" get-secret-value \
        --secret-id "$secret_arn" \
        --query 'SecretString' \
        --output text | grep -o '"password":"[^"]*"' | sed 's/"password":"//;s/"//')
    echo "$secret_value"
    return 0
}

#######################################
# Logic to create the slurmdbd accounting db configureation file
# Arguments:
#   $1 - path to the config file
#######################################
add_accounting_to_slurm_config() {
    # if file not created, create
    if [ ! -f "$1" ]; then
        touch "$1"
    fi
    cat << EOF > "$1"
# ACCOUNTING
JobAcctGatherType=jobacct_gather/linux
JobAcctGatherFrequency=30
AccountingStorageType=accounting_storage/slurmdbd
AccountingStorageHost=localhost
AccountingStorageUser=${2}
AccountingStoragePort=6819
EOF
}

#######################################
# Create slurmdbd configuration file
#
# Arguments:
#   slurm_database_secret - The name of the secret in AWS Secrets Manager
#   slurm_database_endpoint - The AWS region where the secret is stored
#   conf_path - path to the config file
#   ownership - owner of the config file
# Returns:
#   0 on success, non-zero on failure
#######################################
setup_remote_slurmdbd_database_config() {
    local slurm_database_secret="$1"
    local slurm_database_endpoint="$2"
    local slurm_database_user="$3"
    local conf_path="$4"
    local ownership="$5"
    echo "Updating $conf_path with StorageHost: $slurm_database_endpoint"
    SLURM_DB_USER=${slurm_database_user} SLURM_DB_PASSWORD=${slurm_database_secret} envsubst < "$conf_path.template" > "$conf_path"
    # setup slurmdbd conf and create a backup
    if ! sed -i.bak \
    -e "s|^StorageHost=.*|StorageHost=${slurm_database_endpoint}|" \
    "$conf_path"; then
        echo "Error: Failed to update SlurmDBD configuration file ${conf_path}." >&2
        return 1
    fi
    chown "$ownership" "$conf_path"
    chmod 600 "$conf_path"
    echo "Successfully updated $conf_path with StorageHost and StoragePass."
    return 0
}

#######################################
# Setup remote slurmdbd database user credential and privilege
#
# Arguments:
#   slurm_database_secret - The name of the secret in AWS Secrets Manager
#   slurm_database_endpoint - The AWS region where the secret is stored
# Returns:
#   0 on success, non-zero on failure
#######################################
setup_remote_slurmdbd_database_privilege() {
    local slurm_database_secret="$1"
    local slurm_database_endpoint="$2"
    local slurm_database_user="$3"

    local temp_dir
    temp_dir=$(mktemp -d)
    local temp_db_config="$temp_dir/db.config"
    local temp_grant_sql="$temp_dir/grant.mysql"

    # Create db_config template
    cat > "$temp_db_config" << EOF
[client]
user=${slurm_database_user}
password="${slurm_database_secret}"

[mysql]
no-auto-rehash
host=${slurm_database_endpoint}
port=3306

EOF

    # Create grant_sql template
    cat > "$temp_grant_sql" << EOF
GRANT ALL ON \`%\`.* TO ${slurm_database_user}@\`%\`;
FLUSH PRIVILEGES;
EOF
    # Execute MySQL commands
    if mysql --defaults-extra-file="$temp_db_config" < "$temp_grant_sql"; then
        echo "Remote Slurm database setup completed successfully."
    else
        echo "Error: Failed to setup remote Slurm database."
        rm -rf "$temp_dir"
        return 1
    fi

    # Clean up temporary files
    rm -rf "$temp_dir"
}


#######################################
# Sets up sns notification for headnode failover alarm
#
# Arguments:
#   slurm_sns_arn - The address of the sns arn
#   slurm_sns_region - The AWS region where the sns arn is stored
#   designated_folder - the shared based path
#   slurm_conf_path - instance slurm config path
#   ownership - file ownership permission
# Returns:
#   0 on success, non-zero on failure
#######################################
setup_sns_notification() {
    local slurm_sns_arn="$1"
    local slurm_sns_region="$2"
    local designated_folder="$3"
    local slurm_conf_path="$4"
    local ownership="$5"

    # Check if the config file exists
    if [[ ! -f "$slurm_conf_path" ]]; then
        echo "Error: Slurm config file $slurm_conf_path does not exist." >&2
        return 1
    fi

    # SlurmctldPrimaryOnProg and SlurmctldPrimaryOffProg already exists skip
    if grep -q -e "SlurmctldPrimaryOnProg" -e "SlurmctldPrimaryOffProg" "$slurm_conf_path"; then
        echo "SNS notification already setup"
        return 0
    fi

    script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    # Call the main function and capture the output
    local output_string
    output_string=$(source "$script_dir/headnode_notification.sh" "$slurm_sns_arn" "$slurm_sns_region" "$designated_folder" "$ownership")
    set -- $output_string
    # Add or update the variables in the config file
    sed -i.bak'' \
        -e '/^SlurmctldPrimaryOnProg=/d' \
        -e '/^SlurmctldPrimaryOffProg=/d' \
        -e '/^# Slurmctld settings/a\
SlurmctldPrimaryOnProg='"$1"' \
SlurmctldPrimaryOffProg='"$2"'' \
        "$slurm_conf_path"

    echo "SNS notification setup completed successfully"
    return 0
}

#######################################
# Function to update slurmdbd.service
#######################################
update_slurmdbd_service() {
    # update slurmdbd.service
  cat >/etc/systemd/system/slurmdbd.service <<EOF
[Unit]
Description=Slurm DBD accounting daemon
After=network-online.target munge.service
Before=slurmctld.service
Wants=network-online.target
Requires=munge.service
ConditionPathExists=/opt/slurm/etc/slurmdbd.conf

[Service]
Type=simple
EnvironmentFile=-/opt/slurm/etc/default/slurmdb
ExecStart=/opt/slurm/sbin/slurmdbd -D -s \$SLURMDBD_OPTIONS
ExecReload=/bin/kill -HUP \$MAINPID
PIDFile=/var/run/slurmdbd.pid
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload
    return 0
}


#######################################
# Function to stop Slurm services
#######################################
stop_slurm_services() {
    # stop slurm services
    systemctl stop slurmctld
    systemctl stop slurmdbd
    return 0
}

#######################################
# Function to restart Slurm services
#######################################
restart_slurm_services() {
    # restart slurm services
    systemctl restart slurmctld
    systemctl restart slurmdbd
    return 0
}

#######################################
# Uses ip route get 1 to determine the route to the internet (1 is a valid IP address representing the internet).
# Uses awk to print the 7th field, which is typically the source IP address. [1]
# Returns:
#   If the ip command fails or doesn't return an IP, it defaults to "127.0.0.1".
#######################################
get_ip_address() {
    IP=$(ip route get 1 | awk '{print $7;exit}')
    echo "${IP:-127.0.0.1}"
}

#######################################
# Main logic to handle different function calls for individual function execution
#######################################
if [[ $# -eq 0 ]]; then
    # If no arguments are provided, run the main function
    main
else
    # If an argument is provided, try to run it as a function
    if [[ $(type -t "$1") == function ]]; then
        "$1" "${@:2}"
    else
        echo "Error: Function $1 not found" >&2
        exit 1
    fi
fi