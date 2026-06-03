# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#!/bin/bash

: <<'SUMMARY'
Script: headnode_notification.sh

Purpose:
This script sets up SNS (Simple Notification Service) notification scripts for Slurm
controller failover in a multi-head node cluster environment. It creates two scripts:
one for when the primary controller comes online, and another for when it goes offline.

Key Functions:
1. main: The entry point of the script. It processes arguments and calls create_script
   for both "ON" and "OFF" scenarios.

2. create_script: Generates a notification script for a given scenario (ON/OFF).
   It creates a bash script that logs controller status changes and sends SNS
   notifications.

Arguments:
1. SNS Topic ARN
2. SNS Region
3. Slurm Cluster Path  /<shared volume dir>/aws/hyperpod/<cluster-name>
4. File Ownership - it's used for executor to setup file ownership

Output:
- Creates two scripts:
  1. slurm_controller_on.sh: Notifies when the slurm controller comes online.
  2. slurm_controller_off.sh: Notifies when the slurm controller goes offline.
- Returns the paths of the created scripts.

Usage:

To use this script, run the following command:
./headnode_notification.sh <sns_topic_arn> <sns_region> <slurm_cluster_path>

Or execute the main script headnode_setup.sh as a LifeCycle Script upon cluster creation which it will also
execute this headnode_notification.sh

Notes:
- Requires AWS CLI to be installed and configured for sending SNS notifications.
- The created scripts log events to status and notification log files.
- Ensures proper permissions (executable) for the created scripts.
- The scripts are owned by the 'slurm' user and group.

Important:
This script is crucial for maintaining awareness of the Slurm controller's status
in a high-availability setup. Ensure that the SNS topic is properly configured and
that the necessary permissions are in place for sending notifications.

SUMMARY

main() {
    local sns_topic_arn=$1
    local sns_region=$2
    local slurm_cluster_path=$3
    local ownership=$4
    local on_script_path="${slurm_cluster_path}/opt/slurm/etc/scripts/slurm_controller_on.sh"
    local off_script_path="${slurm_cluster_path}/opt/slurm/etc/scripts/slurm_controller_off.sh"
    # Create the scripts
    create_script "$on_script_path" "$sns_topic_arn" "$sns_region" "ON", "Primary" "$slurm_cluster_path" "$ownership"
    create_script "$off_script_path" "$sns_topic_arn" "$sns_region" "OFF" "Backup" "$slurm_cluster_path" "$ownership"
    # return the two path in an array 
    local paths=("$on_script_path" "$off_script_path")
    echo "${paths[@]}"
}

#######################################
# Sets up sns notification script for headnode failover alarm
#
# Arguments:
#   script_path: Where the path should be located
#   sns_topic_arn: sns topic arn
#   sns_region: sns topic region
#   status: ON or OFF enum for the controller script
#   is_primary: is it a primary controller
#   slurm_cluster_path: path to slurm cluster
#   ownership: file ownership
# Outputs:
#   Returns array of script paths
# Returns:
#   0 on success, non-zero on failure
#######################################
create_script() {
    local script_path="$1"
    local sns_topic_arn="$2"
    local sns_region="$3"
    local status="$4"
    local is_primary="$5"
    local slurm_cluster_path="$6"
    local ownership="$7"

    cat << EOF > "$script_path"
#!/bin/bash

set -e
set -x
set -o pipefail  # trace ERR through pipes
set -o errtrace  # trace ERR through 'time command' and other functions
set -o nounset   ## set -u : exit the script if you try to use an uninitialised variable

# Log file paths
STATUS_LOG="${slurm_cluster_path}/var/log/slurm/controller_status.log"
NOTIFICATION_LOG="${slurm_cluster_path}/var/log/slurm/notification_attempts.log"

# Event details
TIMESTAMP=\$(date -u +'%Y-%m-%dT%H:%M:%SZ')
EVENT="Slurm Controller is $status"
NODE_TYPE="$is_primary Node"
hostname=\$(ip route get 1 | sed -n 's/.*src \([^ ]*\).*/\1/p')

# Log the event
echo "\$TIMESTAMP: \$EVENT (\$hostname, \$NODE_TYPE)" >> "\$STATUS_LOG"

# Attempt to send notification via SNS
if command -v aws &> /dev/null; then
    RESPONSE=\$(aws sns publish \
        --topic-arn "$sns_topic_arn" \
        --message "\$EVENT (\$hostname, \$NODE_TYPE)" \
        --subject "Slurm Controller Status Change" \
        --region "$sns_region" 2>&1)

    if [ \$? -eq 0 ]; then
        echo "\$TIMESTAMP: Notification sent successfully to SNS" >> "\$NOTIFICATION_LOG"
    else
        echo "\$TIMESTAMP: Failed to send notification to SNS. Error: \$RESPONSE" >> "\$NOTIFICATION_LOG"
    fi
else
    echo "\$TIMESTAMP: AWS CLI not found. Unable to send notification." >> "\$NOTIFICATION_LOG"
fi
EOF
    chmod 755 "$script_path"
    chown "$ownership" "$script_path"
}

main "$@"

