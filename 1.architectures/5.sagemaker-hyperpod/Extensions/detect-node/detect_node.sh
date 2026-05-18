#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Detects node type and writes node-specific information to nodeinfo.json.
#
# This script reads the HyperPod platform config files at /opt/ml/config/
# (provisioning_parameters.json and resource_config.json) to determine the
# node type (controller, compute, or login) and extract node-specific details.
#
# Output: /opt/ml/config/nodeinfo.json
#
# Usage:
#   Run once at the start of OnInitComplete (before other extensions):
#     sudo bash detect_node.sh
#
#   Other scripts can then read the output:
#     NODE_TYPE=$(python3 -c "import json; print(json.load(open('/opt/ml/config/nodeinfo.json'))['node_type'])")

set -e

CONFIG_DIR="/opt/ml/config"
PP_FILE="$CONFIG_DIR/provisioning_parameters.json"
RC_FILE="$CONFIG_DIR/resource_config.json"
OUTPUT_FILE="$CONFIG_DIR/nodeinfo.json"

LOG_FILE="/var/log/provision/detect_node.log"
mkdir -p /var/log/provision
touch "$LOG_FILE"

logger() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# ---------------------------------------------------------------------------
# Validate platform config files exist
# ---------------------------------------------------------------------------
if [[ ! -f "$PP_FILE" ]]; then
    logger "ERROR: $PP_FILE not found. This script must run on a HyperPod cluster node."
    exit 1
fi

if [[ ! -f "$RC_FILE" ]]; then
    logger "ERROR: $RC_FILE not found. This script must run on a HyperPod cluster node."
    exit 1
fi

# ---------------------------------------------------------------------------
# Detect node type and write nodeinfo.json using Python
# ---------------------------------------------------------------------------
python3 - "$PP_FILE" "$RC_FILE" "$OUTPUT_FILE" << 'PYEOF'
import json
import socket
import sys
import time

pp_file = sys.argv[1]
rc_file = sys.argv[2]
output_file = sys.argv[3]

def get_ip_address():
    """Get this node's IP address using the UDP socket trick (no network call)."""
    max_retries = 7
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('10.254.254.254', 1))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                return '127.0.0.1'

# Read platform config files
with open(pp_file) as f:
    pp = json.load(f)

with open(rc_file) as f:
    rc = json.load(f)

# Get this node's IP
my_ip = get_ip_address()

# Find this node in resource_config
my_group = None
my_instance = None
for group in rc.get('InstanceGroups', []):
    for inst in group.get('Instances', []):
        if inst.get('CustomerIpAddress') == my_ip:
            my_group = group
            my_instance = inst
            break
    if my_group:
        break

if not my_group or not my_instance:
    print(f"ERROR: Could not find instance with IP {my_ip} in resource_config.json", file=sys.stderr)
    sys.exit(1)

# Determine node type by comparing group name to provisioning_parameters
group_name = my_group.get('Name', '')
controller_group = pp.get('controller_group', '')
login_group = pp.get('login_group', '') or ''

if group_name == controller_group:
    node_type = 'controller'
elif group_name == login_group:
    node_type = 'login'
else:
    node_type = 'compute'

# Extract cluster info
cluster_config = rc.get('ClusterConfig', {})

# Build nodeinfo
nodeinfo = {
    "node_type": node_type,
    "instance_group_name": group_name,
    "instance_name": my_instance.get('InstanceName', ''),
    "instance_id": my_instance.get('InstanceId', ''),
    "instance_type": my_group.get('InstanceType', ''),
    "ip_address": my_ip,
    "cluster_name": cluster_config.get('ClusterName', ''),
    "cluster_arn": cluster_config.get('ClusterArn', ''),
}

# Write output
with open(output_file, 'w') as f:
    json.dump(nodeinfo, f, indent=2)

print(f"Node type: {node_type} (group: {group_name}, ip: {my_ip}, instance: {my_instance.get('InstanceId', '')})")
PYEOF

EXIT_CODE=$?
if [[ $EXIT_CODE -ne 0 ]]; then
    logger "ERROR: Node detection failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

# Log the result
NODE_TYPE=$(python3 -c "import json; print(json.load(open('$OUTPUT_FILE'))['node_type'])")
logger "Node info written to $OUTPUT_FILE (node_type=$NODE_TYPE)"
