#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Standalone observability extension script for SageMaker HyperPod Slurm clusters.
#
# Designed for use as an OnInitComplete script with AMI-based configuration.
# Can also be used with OnCreate-based clusters by calling it after Slurm is started.
#
# This script auto-detects the node type (controller, compute, login) by checking
# which Slurm daemon is running, reads configuration from config.json, and installs
# the appropriate metric exporters and OpenTelemetry collector.
#
# Prerequisites:
#   - Docker must be installed and running (included in AMI-based configuration)
#   - Slurm daemons must be started (included in AMI-based configuration)
#   - An Amazon Managed Service for Prometheus (AMP) workspace must exist
#   - The cluster execution role must have permissions to remote-write to AMP
#
# Usage:
#   As OnInitComplete:
#     Upload this directory to S3 and specify setup_observability.sh as OnInitComplete.
#
#   Manual execution on a running cluster:
#     sudo bash setup_observability.sh

set -ex

LOG_FILE="/var/log/provision/setup_observability.log"
mkdir -p /var/log/provision
touch "$LOG_FILE"

logger() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Like logger but writes to stderr + log file. Use inside functions that
# return values via stdout (e.g. detect_node_type).
logger_err() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE" >&2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Wait for scontrol to return node data (controller only)
# ---------------------------------------------------------------------------
wait_for_scontrol() {
    local timeout=120
    local interval=5
    local elapsed=0

    logger "Waiting for scontrol to report registered nodes..."
    while [ $elapsed -lt $timeout ]; do
        if scontrol show nodes 2>/dev/null | grep -q "NodeName"; then
            logger "Slurm nodes are registered. Proceeding."
            return 0
        fi
        logger "No nodes registered yet. Retrying in ${interval}s... (${elapsed}s/${timeout}s)"
        sleep $interval
        elapsed=$((elapsed + interval))
    done

    logger "WARNING: scontrol did not report nodes within ${timeout}s. Proceeding anyway."
    return 0
}

# ---------------------------------------------------------------------------
# Detect node type
#
# Primary: read from /opt/ml/config/nodeinfo.json (written by detect-node
# utility in run_extensions.sh before this script runs).
#
# Fallback: read resource_config.json + provisioning_parameters.json
# directly (same algorithm as detect-node, for standalone use without
# run_extensions.sh).
#
# This approach is deterministic, instant, and has zero dependency on
# Slurm services being running -- which matters during scale-up when
# slurmd may not be started yet.
# ---------------------------------------------------------------------------
detect_node_type() {
    local nodeinfo="/opt/ml/config/nodeinfo.json"

    # Primary: nodeinfo.json (written by detect-node utility)
    if [ -f "$nodeinfo" ]; then
        local node_type
        node_type=$(python3 -c "import json; print(json.load(open('$nodeinfo'))['node_type'])" 2>/dev/null)
        if [ -n "$node_type" ]; then
            logger_err "Node type from nodeinfo.json: $node_type"
            echo "$node_type"
            return 0
        fi
    fi

    # Fallback: detect from platform config files directly
    logger_err "nodeinfo.json not found, detecting from platform config files..."
    local pp_file="/opt/ml/config/provisioning_parameters.json"
    local rc_file="/opt/ml/config/resource_config.json"

    if [ ! -f "$rc_file" ]; then
        logger_err "ERROR: $rc_file not found. Cannot detect node type."
        exit 1
    fi

    local node_type
    node_type=$(python3 -c "
import json, socket, sys, time

rc_file = '$rc_file'
pp_file = '$pp_file'

# Get this node's IP
def get_ip():
    for attempt in range(5):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('10.254.254.254', 1))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            time.sleep(2)
    return '127.0.0.1'

my_ip = get_ip()

with open(rc_file) as f:
    rc = json.load(f)

# Find this node's instance group
group_name = ''
for g in rc.get('InstanceGroups', []):
    for inst in g.get('Instances', []):
        if inst.get('CustomerIpAddress') == my_ip:
            group_name = g.get('Name', '')
            break
    if group_name:
        break

if not group_name:
    print('compute', end='')
    sys.exit(0)

# Compare to provisioning_parameters if available
try:
    with open(pp_file) as f:
        pp = json.load(f)
    controller_group = pp.get('controller_group', '')
    login_group = pp.get('login_group', '') or ''
    if group_name == controller_group:
        print('controller', end='')
    elif group_name == login_group:
        print('login', end='')
    else:
        print('compute', end='')
except FileNotFoundError:
    # No provisioning_parameters.json (API-driven config)
    # Fall back to group name heuristics
    lower = group_name.lower()
    if 'controller' in lower or 'head' in lower:
        print('controller', end='')
    elif 'login' in lower:
        print('login', end='')
    else:
        print('compute', end='')
" 2>/dev/null)

    if [ -z "$node_type" ]; then
        logger_err "WARNING: Node type detection failed. Defaulting to compute."
        node_type="compute"
    fi

    logger_err "Detected node type: $node_type (from platform config)"
    echo "$node_type"
}

# ---------------------------------------------------------------------------
# Read config.json
# ---------------------------------------------------------------------------
CONFIG_FILE="$SCRIPT_DIR/config.json"
if [[ ! -f "$CONFIG_FILE" ]]; then
    logger "ERROR: config.json not found at $CONFIG_FILE"
    exit 1
fi

PROMETHEUS_REMOTE_WRITE_URL=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['prometheus_remote_write_url'])")
ADVANCED=$(python3 -c "import json; print('1' if json.load(open('$CONFIG_FILE')).get('advanced_metrics', False) else '0')")
NCCL_METRICS=$(python3 -c "import json; print('1' if json.load(open('$CONFIG_FILE')).get('nccl_metrics_enabled', False) else '0')")
NCCL_DUMP_INTERVAL=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('nccl_metrics_dump_interval_seconds', 30))")
NCCL_PLUGIN_PATH=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('nccl_profiler_plugin_path', '/opt/nccl-inspector/libnccl-profiler-inspector.so'))")

# Validate the AMP URL has been configured
if echo "$PROMETHEUS_REMOTE_WRITE_URL" | grep -q '<workspace-id>'; then
    logger "ERROR: prometheus_remote_write_url in config.json still contains placeholder values."
    logger "Update config.json with your Amazon Managed Prometheus workspace URL before running."
    exit 1
fi

# ---------------------------------------------------------------------------
# Detect node type
# ---------------------------------------------------------------------------
NODE_TYPE=$(detect_node_type)
logger "Detected node type: $NODE_TYPE"

# ---------------------------------------------------------------------------
# Ensure Docker is available
# ---------------------------------------------------------------------------
if ! command -v docker &>/dev/null; then
    logger "ERROR: Docker is not installed. AMI-based configuration should have installed it."
    exit 1
fi

if ! systemctl is-active --quiet docker 2>/dev/null; then
    logger "Docker service is not running. Attempting to start..."
    systemctl start docker
fi

# ---------------------------------------------------------------------------
# Wait for scontrol on controller before installing exporters
# ---------------------------------------------------------------------------
if [[ "$NODE_TYPE" == "controller" ]]; then
    wait_for_scontrol
fi

# ---------------------------------------------------------------------------
# Run the observability installer
# ---------------------------------------------------------------------------
logger "Installing observability stack for node type: $NODE_TYPE"

CMD=(
    python3 -u "$SCRIPT_DIR/install_observability.py"
    --node-type "$NODE_TYPE"
    --prometheus-remote-write-url "$PROMETHEUS_REMOTE_WRITE_URL"
)

if [[ "$ADVANCED" == "1" ]]; then
    CMD+=(--advanced)
fi

if [[ "$NCCL_METRICS" == "1" ]]; then
    CMD+=(--nccl-metrics)
fi

set +e
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
INSTALL_EXIT_CODE=${PIPESTATUS[0]}
set -e

if [[ $INSTALL_EXIT_CODE -ne 0 ]]; then
    logger "ERROR: Observability installation failed with exit code $INSTALL_EXIT_CODE"
    exit $INSTALL_EXIT_CODE
fi

# ---------------------------------------------------------------------------
# Configure NCCL Inspector task prolog (compute nodes only, if enabled)
# ---------------------------------------------------------------------------
if [[ "$NCCL_METRICS" == "1" ]] && [[ "$NODE_TYPE" == "compute" || "$NODE_TYPE" == "controller" ]]; then
    logger "Configuring NCCL Inspector task prolog..."
    DUMP_INTERVAL_MICROSECONDS=$((NCCL_DUMP_INTERVAL * 1000000))

    cat > /opt/slurm/etc/task_prolog.sh << EOF
#!/bin/bash
if [ ! -f ${NCCL_PLUGIN_PATH} ]; then
  echo "[WARN] NCCL Inspector plugin not found at ${NCCL_PLUGIN_PATH}, skipping NCCL metrics" >&2
  exit 0
fi
echo "export NCCL_PROFILER_PLUGIN=${NCCL_PLUGIN_PATH}"
echo "export NCCL_INSPECTOR_ENABLE=1"
echo "export NCCL_INSPECTOR_PROM_DUMP=1"
echo "export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=${DUMP_INTERVAL_MICROSECONDS}"
echo "export NCCL_INSPECTOR_DUMP_DIR=/var/lib/node_exporter/nccl_inspector/"
EOF
    chmod +x /opt/slurm/etc/task_prolog.sh

    if [[ "$NODE_TYPE" == "controller" ]]; then
        if ! grep -q "^TaskProlog=" /opt/slurm/etc/slurm.conf 2>/dev/null; then
            # Ensure a newline before appending to avoid merging with the last line
            sed -i -e '$a\' /opt/slurm/etc/slurm.conf
            echo "TaskProlog=/opt/slurm/etc/task_prolog.sh" >> /opt/slurm/etc/slurm.conf
            systemctl restart slurmctld
            # Push updated config to compute nodes so slurmd picks up TaskProlog
            sleep 2
            scontrol reconfigure || logger "WARNING: scontrol reconfigure failed -- compute nodes may need manual reconfigure"
            logger "Added TaskProlog to slurm.conf, restarted slurmctld, and reconfigured cluster"
        fi
    fi
fi

logger "Observability setup complete for $NODE_TYPE node."
