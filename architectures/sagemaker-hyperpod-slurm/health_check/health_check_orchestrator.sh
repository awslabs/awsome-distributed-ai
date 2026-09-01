#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Health-check orchestrator for Slurm clusters.
# Runs on a single management node as the Main slurm batch job.
# Dispatches one worker job per target node to execute health-check test,
# waits for completion, then processes the results,
# updates Slurm node features, and applies any necessary remediation.
#
# Test script stdout contract (one line per node):
#   HEALTH_CHECK_RESULT:<hostname>:<Passed|Failed|Skipped>:<none|reboot|replace>:<reason>

#SBATCH --job-name=health_check_main_job
#SBATCH --nodes=1

set -euo pipefail

# --- Constants ---
readonly HEALTH_CHECK_FEATURE_PREFIX="HealthCheck"
readonly HEALTH_CHECK_ALL_VALUES="${HEALTH_CHECK_FEATURE_PREFIX}:InProgress,${HEALTH_CHECK_FEATURE_PREFIX}:Passed,${HEALTH_CHECK_FEATURE_PREFIX}:Failed,${HEALTH_CHECK_FEATURE_PREFIX}:Skipped"
readonly VALID_STATUSES="Passed Failed Skipped"
readonly VALID_REMEDIATIONS="none reboot replace"
readonly RESOURCE_CONFIG="/opt/ml/config/resource_config.json"
readonly TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
readonly WORKER_TIMEOUT_BUFFER_MINUTES=10

# Exit codes
readonly EXIT_GENERAL_ERROR=1       # Resource errors, runtime errors, general failures
readonly EXIT_INVALID_ARGS=2        # Invalid arguments / usage errors
readonly EXIT_MISSING_DEPENDENCY=127 # Missing required dependency (e.g. jq)

# --- Usage ---
usage() {
    cat << 'EOF'
Usage: sbatch [--time=<minutes>] health_check_orchestrator.sh [OPTIONS]

Target (mutually exclusive — exactly one required):
  --target-nodes <node1,node2,...>  Comma-separated list or Slurm nodelist expression like ip-10-1-4-240,ip-10-1-114-79
  --target-partition <name>         Partition name (resolves all nodes in partition)
  --instance-group <name>           Instance group name (from resource_config.json)

Required:
  --test-script <absolute_path>  Absolute path to health-check script to run on each node
  --output-dir <absolute_path>   Absolute path to directory for all logs and results

Optional:
  --test-script-args <json>      JSON object of parameters passed to the test script
  --remediate <true|false>       Whether to apply remediation (default: true)
  --help                         Display this help message

Timeout control:
  Use sbatch --time=<minutes> to set the overall wall-clock time limit.
  Worker sbatch jobs inherit the timeout (minus a small buffer) as their --time.
  The test script (e.g. dcgm.sh) always uses its own built-in defaults
  (e.g. DCGM level-based timeouts: L2=30m, L3=60m, L4=180m) regardless of --time.
  If --time is not specified, workers run without a Slurm time limit.

Exit codes:
  0     Success — health check completed (Passed, Failed, or Skipped)
  1     General error — resource resolution, runtime, or unexpected failure
  2     Invalid arguments — wrong or missing command-line options
  127   Missing dependency — required tool (e.g. jq) not installed

Examples:
  sbatch --time=120 --output=/fsx/ubuntu/dhc/management_%j.log health_check_orchestrator.sh --target-nodes node1,node2 --test-script /fsx/ubuntu/dhc/dcgm.sh --output-dir /fsx/ubuntu/dhc
  sbatch --output=/fsx/ubuntu/dhc/management_%j.log health_check_orchestrator.sh --target-partition ml.g5.xlarge --test-script /fsx/ubuntu/dhc/dcgm.sh --output-dir /fsx/ubuntu/dhc --test-script-args '{"level": 2}'
  sbatch --output=/fsx/ubuntu/dhc/management_%j.log health_check_orchestrator.sh --instance-group worker-group-1 --test-script /fsx/ubuntu/dhc/dcgm.sh --output-dir /fsx/ubuntu/dhc
EOF
    exit "${1:-$EXIT_INVALID_ARGS}"
}

# --- Helpers ---
die() {
    local rc=$EXIT_GENERAL_ERROR
    if [[ "$1" =~ ^[0-9]+$ ]]; then rc="$1"; shift; fi
    echo "ERROR: $*" >&2; echo "ERROR: $*"; exit "$rc"
}
warn() { echo "WARNING: $*" >&2; }

expand_nodelist() {
    local input="$1"
    local output
    if output=$(scontrol show hostname "$input" 2>&1); then
        echo "$output"
        return 0
    fi

    # scontrol failed — bracket notation requires scontrol to expand
    if [[ "$input" == *"["* ]]; then
        die "scontrol failed to expand nodelist '${input}' (bracket notation requires scontrol)"
    fi

    warn "scontrol show hostname failed for '${input}' — falling back to comma split"
    echo "$input" | tr ',' '\n'
}

# --- Slurm feature management ---
#
# Feature values set on each target node:
#   InProgress — health check is currently running
#   Passed     — test reported Passed (or no result when worker succeeded)
#   Failed     — test reported Failed (actual GPU health failure)
#   Skipped    — test execution Failed (inconclusive; no remediation applied)
#
# After a health check completes, submit jobs only on healthy nodes:
#   sbatch -C "HealthCheck:Passed" my_training_job.sh
#   srun -C "HealthCheck:Passed" -N 4 my_training_job.sh
#
update_node_feature() {
    local node="$1" new_status="$2"
    local desired_active="${HEALTH_CHECK_FEATURE_PREFIX}:${new_status}"

    local node_info
    if ! node_info=$(scontrol show node "$node" 2>&1); then
        warn "scontrol show node failed for $node: $node_info"
        return 1
    fi
    local avail_features active_features
    avail_features=$(echo "$node_info" | grep -oP 'AvailableFeatures=\K[^ ]*' || echo "")
    active_features=$(echo "$node_info" | grep -oP 'ActiveFeatures=\K[^ ]*' || echo "")

    local existing_avail_features
    existing_avail_features=$(echo "$avail_features" | sed "s/${HEALTH_CHECK_FEATURE_PREFIX}:[^,]*//g" | sed 's/,,*/,/g' | sed 's/^,//;s/,$//')
    local new_avail_features
    if [[ -n "$existing_avail_features" && "$existing_avail_features" != "(null)" ]]; then
        new_avail_features="${existing_avail_features},${HEALTH_CHECK_ALL_VALUES}"
    else
        new_avail_features="${HEALTH_CHECK_ALL_VALUES}"
    fi

    local new_active_features
    if [[ "$active_features" == *"${HEALTH_CHECK_FEATURE_PREFIX}:"* ]]; then
        new_active_features=$(echo "$active_features" | sed "s/${HEALTH_CHECK_FEATURE_PREFIX}:[^,]*/${desired_active}/g")
    elif [[ -n "$active_features" && "$active_features" != "(null)" ]]; then
        new_active_features="${active_features},${desired_active}"
    else
        new_active_features="${desired_active}"
    fi

    scontrol update NodeName="$node" AvailableFeatures="$new_avail_features" ActiveFeatures="$new_active_features" 2>&1 \
        || { echo "ERROR: Feature update failed for $node" >&2; return 1; }
}

update_nodes_feature() {
    local nodes="$1" status="$2"
    local desired_active="${HEALTH_CHECK_FEATURE_PREFIX}:${status}"
    local expanded fail=0 count=0

    expanded=$(expand_nodelist "$nodes")
    local node_list=()
    for node in $expanded; do
        [[ -z "$node" ]] && continue
        node_list+=("$node")
    done
    count=${#node_list[@]}
    if (( count == 0 )); then
        echo "Set ${status} on 0 node(s)"
        return 0
    fi

    # --- Batch query: single scontrol show for all nodes ---
    local csv_nodes
    csv_nodes=$(IFS=,; echo "${node_list[*]}")
    local bulk_info
    bulk_info=$(scontrol show node "$csv_nodes" 2>&1) || {
        warn "Batch scontrol show failed — falling back to per-node updates"
        for node in "${node_list[@]}"; do
            update_node_feature "$node" "$status" || fail=$((fail + 1))
        done
        echo "Set ${status} on ${count} node(s)${fail:+, ${fail} failed} (per-node fallback)"
        return 0
    }

    # --- Parse per-node blocks and group by (AvailableFeatures, ActiveFeatures) ---
    # scontrol separates node blocks with blank lines.
    declare -A group_nodes  # key="avail|active" -> value="node1,node2,..."
    local cur_node="" cur_avail="" cur_active=""

    _flush_node() {
        if [[ -z "$cur_node" ]]; then return; fi
        local key="${cur_avail}|${cur_active}"
        if [[ -n "${group_nodes[$key]:-}" ]]; then
            group_nodes["$key"]="${group_nodes[$key]},${cur_node}"
        else
            group_nodes["$key"]="$cur_node"
        fi
    }

    while IFS= read -r line; do
        if [[ "$line" =~ NodeName=([^ ]+) ]]; then
            _flush_node
            cur_node="${BASH_REMATCH[1]}"
            cur_avail=""
            cur_active=""
        fi
        if [[ "$line" =~ AvailableFeatures=([^ ]*) ]]; then
            cur_avail="${BASH_REMATCH[1]}"
        fi
        if [[ "$line" =~ ActiveFeatures=([^ ]*) ]]; then
            cur_active="${BASH_REMATCH[1]}"
        fi
    done <<< "$bulk_info"
    _flush_node
    unset -f _flush_node

    # --- Batch update per group ---
    local group_key group_csv_nodes avail_features active_features
    for group_key in "${!group_nodes[@]}"; do
        group_csv_nodes="${group_nodes[$group_key]}"
        avail_features="${group_key%%|*}"
        active_features="${group_key#*|}"

        # Compute new AvailableFeatures: strip old health-check values, add all states
        local existing_avail
        existing_avail=$(echo "$avail_features" | sed "s/${HEALTH_CHECK_FEATURE_PREFIX}:[^,]*//g" \
            | sed 's/,,*/,/g' | sed 's/^,//;s/,$//')
        local new_avail_features
        if [[ -n "$existing_avail" && "$existing_avail" != "(null)" ]]; then
            new_avail_features="${existing_avail},${HEALTH_CHECK_ALL_VALUES}"
        else
            new_avail_features="${HEALTH_CHECK_ALL_VALUES}"
        fi

        # Compute new ActiveFeatures: replace or append health-check value
        local new_active_features
        if [[ "$active_features" == *"${HEALTH_CHECK_FEATURE_PREFIX}:"* ]]; then
            new_active_features=$(echo "$active_features" | sed "s/${HEALTH_CHECK_FEATURE_PREFIX}:[^,]*/${desired_active}/g")
        elif [[ -n "$active_features" && "$active_features" != "(null)" ]]; then
            new_active_features="${active_features},${desired_active}"
        else
            new_active_features="${desired_active}"
        fi

        scontrol update NodeName="$group_csv_nodes" AvailableFeatures="$new_avail_features" ActiveFeatures="$new_active_features" 2>&1 \
            || { warn "Feature batch update failed for: $group_csv_nodes"; fail=$((fail + ${#group_csv_nodes//[^,]/} + 1)); }
    done

    echo "Set ${status} on ${count} node(s) in ${#group_nodes[@]} batch(es)${fail:+, ${fail} failed}"
}

# --- Node resolution ---
transform_ip() { echo "ip-$(echo "$1" | tr '.' '-')"; }

get_instance_group_nodes() {
    local group_name="$1"
    if [[ ! -f "$RESOURCE_CONFIG" ]]; then die $EXIT_GENERAL_ERROR "Config not found: $RESOURCE_CONFIG"; fi
    if ! command -v jq &>/dev/null; then die $EXIT_MISSING_DEPENDENCY "jq is required but not installed"; fi

    local ips
    ips=$(jq -r --arg name "$group_name" \
        '.InstanceGroups[] | select(.Name == $name) | .Instances[].CustomerIpAddress' \
        "$RESOURCE_CONFIG")
    if [[ -z "$ips" ]]; then die "No instances found for instance group: $group_name"; fi

    local result=""
    while IFS= read -r ip; do
        if [[ -z "$ip" ]]; then continue; fi
        result="${result:+${result},}$(transform_ip "$ip")"
    done <<< "$ips"
    echo "$result"
}

get_partition_nodes() {
    local partition_info
    if ! partition_info=$(scontrol show partition "$1"); then
        die "scontrol show partition failed for '$1'"
    fi
    local nodes
    nodes=$(echo "$partition_info" | grep -oP '(?<![A-Za-z])Nodes=\K[^ ]*' || true)
    if [[ -z "$nodes" ]]; then die "Could not resolve nodes for partition: $1"; fi
    echo "$nodes"
}

# --- Remediation ---
apply_remediation() {
    local node="$1" action="$2" remediate_enabled="$3"
    case "$action" in
        replace|reboot)
            local reason="Action:${action^}"
            if [[ "$remediate_enabled" == "true" ]]; then
                echo "Setting $node FAIL — Reason: $reason"
                scontrol update NodeName="$node" State=FAIL Reason="$reason" \
                    || warn "Failed to set $node FAIL for $action"
            else
                echo "Node $node needs ${action^^} — remediation skipped (--remediate false)"
            fi ;;
        none) echo "Node $node passed — no remediation needed" ;;
    esac
}

# --- Result validation ---
validate_health_result() {
    local hostname="$1" status="$2" remediation="$3"
    if [[ -z "$hostname" ]]; then warn "HEALTH_CHECK_RESULT has empty hostname — skipping"; return 1; fi

    local valid=false
    for s in $VALID_STATUSES; do
        if [[ "$status" == "$s" ]]; then valid=true; break; fi
    done
    if [[ "$valid" != "true" ]]; then warn "HEALTH_CHECK_RESULT for $hostname: invalid status '$status'"; return 1; fi

    valid=false
    for r in $VALID_REMEDIATIONS; do
        if [[ "$remediation" == "$r" ]]; then valid=true; break; fi
    done
    if [[ "$valid" != "true" ]]; then warn "HEALTH_CHECK_RESULT for $hostname: invalid remediation '$remediation'"; return 1; fi
    return 0
}

# --- Argument parsing ---
require_value() {
    if [[ -z "${2:-}" || "$2" == -* ]]; then
        die $EXIT_INVALID_ARGS "$1 requires a non-empty value (got '${2:-}')"
    fi
}

RESOURCE="" TARGET_VALUE="" TEST_SCRIPT="" TEST_SCRIPT_ARGS=""
OUTPUT_DIR="" REMEDIATE="true"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target-nodes)
            if [[ -n "$RESOURCE" ]]; then die $EXIT_INVALID_ARGS "Only one of --target-nodes, --target-partition, --instance-group can be provided"; fi
            require_value "$1" "${2:-}"; RESOURCE="nodelist"; TARGET_VALUE="$2"; shift 2 ;;
        --target-partition)
            if [[ -n "$RESOURCE" ]]; then die $EXIT_INVALID_ARGS "Only one of --target-nodes, --target-partition, --instance-group can be provided"; fi
            require_value "$1" "${2:-}"; RESOURCE="partition"; TARGET_VALUE="$2"; shift 2 ;;
        --instance-group)
            if [[ -n "$RESOURCE" ]]; then die $EXIT_INVALID_ARGS "Only one of --target-nodes, --target-partition, --instance-group can be provided"; fi
            require_value "$1" "${2:-}"; RESOURCE="instance-group"; TARGET_VALUE="$2"; shift 2 ;;
        --test-script)   require_value "$1" "${2:-}"; TEST_SCRIPT="$2"; shift 2 ;;
        --test-script-args)
            if [[ -z "${2:-}" ]]; then die $EXIT_INVALID_ARGS "$1 requires a value"; fi
            TEST_SCRIPT_ARGS="$2"; shift 2 ;;
        --output-dir)    require_value "$1" "${2:-}"; OUTPUT_DIR="$2"; shift 2 ;;
        --remediate)
            require_value "$1" "${2:-}"
            if [[ "$2" != "true" && "$2" != "false" ]]; then die $EXIT_INVALID_ARGS "--remediate must be 'true' or 'false', got: '$2'"; fi
            REMEDIATE="$2"; shift 2 ;;
        --help) usage 0 ;;
        *) die $EXIT_INVALID_ARGS "Unknown option: $1" ;;
    esac
done

# --- Validate required arguments ---
if [[ -z "$RESOURCE" ]]; then            die $EXIT_INVALID_ARGS "A target option (--target-nodes, --target-partition, or --instance-group) is required"; fi
if [[ -z "$TEST_SCRIPT" ]]; then         die $EXIT_INVALID_ARGS "--test-script is required"; fi
if [[ -z "$OUTPUT_DIR" ]]; then          die $EXIT_INVALID_ARGS "--output-dir is required"; fi
if [[ "${OUTPUT_DIR:0:1}" != "/" ]]; then die $EXIT_INVALID_ARGS "--output-dir must be an absolute path (got: '$OUTPUT_DIR')"; fi
if [[ "$OUTPUT_DIR" == *..* ]]; then     die $EXIT_INVALID_ARGS "--output-dir must not contain '..'"; fi
if [[ "${TEST_SCRIPT:0:1}" != "/" ]]; then die $EXIT_INVALID_ARGS "--test-script must be an absolute path (got: '$TEST_SCRIPT')"; fi
if [[ "$TEST_SCRIPT" == *..* ]]; then     die $EXIT_INVALID_ARGS "--test-script must not contain '..'"; fi
if [[ ! -f "$TEST_SCRIPT" ]]; then       die $EXIT_GENERAL_ERROR "Test script not found: $TEST_SCRIPT"; fi
if [[ ! -x "$TEST_SCRIPT" ]]; then       warn "Test script is not executable: $TEST_SCRIPT (will run via 'bash')"; fi
mkdir -p "$OUTPUT_DIR"           || die "Cannot create output directory: $OUTPUT_DIR"

if [[ -n "$TEST_SCRIPT_ARGS" ]]; then
    if ! command -v jq &>/dev/null; then die $EXIT_MISSING_DEPENDENCY "jq is required to validate --test-script-args JSON"; fi
    echo "$TEST_SCRIPT_ARGS" | jq empty 2>/dev/null \
        || die $EXIT_INVALID_ARGS "--test-script-args must be valid JSON (got: '$TEST_SCRIPT_ARGS')"
    json_type=$(echo "$TEST_SCRIPT_ARGS" | jq -r 'type')
    if [[ "$json_type" != "object" ]]; then die $EXIT_INVALID_ARGS "--test-script-args must be a JSON object, got: $json_type"; fi
fi

# --- Parse management job time limit from Slurm ---
# If user specified sbatch --time=<minutes>, we read it here and derive worker timeout.
# If not specified, TimeLimit will be "UNLIMITED" and workers/test scripts use their own defaults.
parse_slurm_timelimit() {
    local tl="$1"
    # Slurm TimeLimit formats: "UNLIMITED", "MM", "MM:SS", "HH:MM:SS", "D-HH:MM:SS"
    if [[ "$tl" == "UNLIMITED" ]]; then echo ""; return; fi
    local days=0 hours=0 minutes=0 seconds=0
    if [[ "$tl" =~ ^([0-9]+)-(.+)$ ]]; then
        days="${BASH_REMATCH[1]}"
        tl="${BASH_REMATCH[2]}"
    fi
    IFS=':' read -ra parts <<< "$tl"
    case ${#parts[@]} in
        1) minutes="${parts[0]}" ;;
        2) minutes="${parts[0]}"; seconds="${parts[1]}" ;;
        3) hours="${parts[0]}"; minutes="${parts[1]}"; seconds="${parts[2]}" ;;
    esac
    echo $(( days * 24 * 60 + hours * 60 + minutes + (seconds > 0 ? 1 : 0) ))
}

MGMT_TIMELIMIT=""
WORKER_TIMEOUT_MINUTES=""
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    raw_timelimit=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -oP 'TimeLimit=\K[^ ]*' || true)
    if [[ -n "$raw_timelimit" ]]; then
        MGMT_TIMELIMIT=$(parse_slurm_timelimit "$raw_timelimit")
    fi
fi

if [[ -n "$MGMT_TIMELIMIT" ]]; then
    WORKER_TIMEOUT_MINUTES=$(( MGMT_TIMELIMIT - WORKER_TIMEOUT_BUFFER_MINUTES ))
    if (( WORKER_TIMEOUT_MINUTES <= 0 )); then
        die $EXIT_INVALID_ARGS "Management job time limit (${MGMT_TIMELIMIT}m) is too short; must be > ${WORKER_TIMEOUT_BUFFER_MINUTES}m buffer"
    fi
    echo "Management job time limit: ${MGMT_TIMELIMIT} minutes (from sbatch --time)"
    echo "Worker job timeout: ${WORKER_TIMEOUT_MINUTES} minutes (management ${MGMT_TIMELIMIT} - ${WORKER_TIMEOUT_BUFFER_MINUTES} buffer)"
else
    echo "No sbatch --time specified — workers will run without a time limit; test scripts use built-in defaults"
fi


# --- Resolve target nodes ---
TARGET_NODES="" TARGET_PARTITION=""
case "$RESOURCE" in
    nodelist)        TARGET_NODES="$TARGET_VALUE" ;;
    partition)       TARGET_PARTITION="$TARGET_VALUE"; TARGET_NODES=$(get_partition_nodes "$TARGET_VALUE") ;;
    instance-group)  TARGET_NODES=$(get_instance_group_nodes "$TARGET_VALUE") ;;
esac
if [[ -z "$TARGET_NODES" ]]; then die "Resolved node list is empty"; fi

# Exclude management node from targets to avoid deadlock
MGMT_HOST="$(hostname)"
expanded_targets=$(expand_nodelist "$TARGET_NODES")
filtered_targets="" mgmt_excluded=false
for node in $expanded_targets; do
    if [[ -z "$node" ]]; then continue; fi
    if [[ "$node" == "$MGMT_HOST" ]]; then
        mgmt_excluded=true
    else
        filtered_targets="${filtered_targets:+${filtered_targets},}${node}"
    fi
done
if [[ "$mgmt_excluded" == "true" ]]; then
    warn "Management node $MGMT_HOST is in target list — excluding to avoid deadlock"
    TARGET_NODES="$filtered_targets"
fi
if [[ -z "$TARGET_NODES" ]]; then die "Target node list is empty after excluding management node ($MGMT_HOST)"; fi

NODE_COUNT=$(expand_nodelist "$TARGET_NODES" | wc -l)

echo "=========================================="
echo "Health Check Management Node — Job $SLURM_JOB_ID"
echo "=========================================="
echo "Started at: $(date)"
echo "Timestamp: $TIMESTAMP"
echo "Management host: $(hostname)"
echo "Resource: $RESOURCE ($TARGET_VALUE)"
echo "Target nodes: $TARGET_NODES ($NODE_COUNT node(s))"
echo "Test script: $TEST_SCRIPT"
if [[ -n "$TEST_SCRIPT_ARGS" ]]; then echo "Test script args: $TEST_SCRIPT_ARGS"; fi
echo "Remediation enabled: $REMEDIATE"
echo "Output directory: $OUTPUT_DIR"
if [[ -n "$WORKER_TIMEOUT_MINUTES" ]]; then
    echo "Worker job timeout: ${WORKER_TIMEOUT_MINUTES} minutes"
else
    echo "Worker job timeout: not set (test script defaults)"
fi
echo ""

# --- Set InProgress on target nodes ---
update_nodes_feature "$TARGET_NODES" "InProgress"

# --- Submit one sbatch job per target node ---
# Each job runs the test script directly on a single node — no srun indirection.
# All jobs use --wait and are backgrounded so they run in parallel; we wait for all.
echo "Submitting per-node worker jobs for $NODE_COUNT node(s)..."

export HC_TEST_SCRIPT="${TEST_SCRIPT}"
export HC_TEST_PARAMS="${TEST_SCRIPT_ARGS}"
export HC_RESULTS_DIR="${OUTPUT_DIR}"
export HC_TIMESTAMP="${TIMESTAMP}"
export HC_FEATURE_PREFIX="${HEALTH_CHECK_FEATURE_PREFIX}"

declare -A WORKER_PIDS WORKER_RC
declare -A node_status node_remediation node_reason
expanded_for_submit=$(expand_nodelist "$TARGET_NODES")

# --- Pre-flight: validate node availability ---
# Nodes that are DOWN, DRAIN, or missing from Slurm are marked Skipped immediately.
# Only nodes in a schedulable state (IDLE, ALLOCATED, etc.) will have jobs submitted.
echo "Pre-flight node validation..."
submit_count=0
for node in $expanded_for_submit; do
    [[ -z "$node" ]] && continue

    node_info=$(scontrol show node "$node" 2>&1) || {
        warn "Node $node not found in Slurm — skipping"
        WORKER_RC["$node"]=1
        node_status["$node"]="Skipped"
        node_remediation["$node"]="none"
        node_reason["$node"]="node not found in Slurm"
        continue
    }
    node_state=$(echo "$node_info" | grep -oP 'State=\K[^ ]*' || true)

    if [[ "$node_state" == *"DOWN"* || "$node_state" == *"DRAIN"* || "$node_state" == *"FAIL"* || "$node_state" == *"NOT_RESPONDING"* ]]; then
        warn "Node $node is unavailable ($node_state) — skipping"
        update_node_feature "$node" "Skipped" 2>/dev/null || true
        WORKER_RC["$node"]=1
        node_status["$node"]="Skipped"
        node_remediation["$node"]="none"
        node_reason["$node"]="node is unavailable ($node_state)"
        continue
    fi

    submit_count=$((submit_count + 1))
done
echo "Pre-flight complete: $submit_count node(s) schedulable"

# --- Submit sbatch per schedulable node ---
for node in $expanded_for_submit; do
    [[ -z "$node" ]] && continue
    # Skip nodes already marked during pre-flight (DOWN, missing)
    [[ -n "${node_status[$node]:-}" ]] && continue

    worker_log="${OUTPUT_DIR}/worker_${node}_${TIMESTAMP}.log"

    SBATCH_ARGS=(
        --wait --job-name="hc_${node}" --nodes=1 --exclusive
        --output="$worker_log"
        --export=ALL --nodelist="$node"
    )
    if [[ -n "$WORKER_TIMEOUT_MINUTES" ]]; then SBATCH_ARGS+=(--time="$WORKER_TIMEOUT_MINUTES"); fi
    if [[ -n "$TARGET_PARTITION" ]]; then SBATCH_ARGS+=(--partition="$TARGET_PARTITION"); fi

    # Background each sbatch --wait so all nodes run in parallel
    sbatch "${SBATCH_ARGS[@]}" "$TEST_SCRIPT" &
    WORKER_PIDS["$node"]=$!
    echo "  Submitted job for $node (pid ${WORKER_PIDS[$node]})"
done

# --- Wait for all per-node jobs to complete ---
if [[ $submit_count -gt 0 ]]; then
    echo "Waiting for ${submit_count} worker job(s) to finish..."
    any_worker_failed=false
    for node in $expanded_for_submit; do
        [[ -z "$node" ]] && continue
        pid="${WORKER_PIDS[$node]:-}"
        [[ -z "$pid" ]] && continue
        wait_rc=0
        wait "$pid" || wait_rc=$?
        WORKER_RC["$node"]=$wait_rc
        if [[ $wait_rc -ne 0 ]]; then
            warn "Worker job for $node exited with code $wait_rc"
            any_worker_failed=true
        fi
    done
    echo "All worker jobs finished."
else
    echo "No schedulable nodes — skipping sbatch submission."
fi

# --- Process results ---
echo ""
echo "=== Processing results ==="

# Read HEALTH_CHECK_RESULT lines from each per-node log file.
# Each node's worker job writes its own log: worker_<node>_<timestamp>.log
expanded=$(expand_nodelist "$TARGET_NODES")
overall="Passed"
passed="" need_reboot="" need_replace="" need_error=""

for node in $expanded; do
    [[ -z "$node" ]] && continue
    # Skip nodes already resolved during pre-flight (DOWN, missing from Slurm)
    [[ -n "${node_status[$node]:-}" ]] && continue
    worker_log="${OUTPUT_DIR}/worker_${node}_${TIMESTAMP}.log"
    node_rc="${WORKER_RC[$node]:-0}"

    if [[ ! -s "$worker_log" ]]; then
        warn "No output file for $node ($worker_log)"
    fi

    # Parse HEALTH_CHECK_RESULT line from this node's log
    while IFS= read -r line; do
        if [[ "$line" != HEALTH_CHECK_RESULT:* ]]; then continue; fi
        _tag="${line%%:*}"; _rest="${line#*:}"
        hostname="${_rest%%:*}"; _rest="${_rest#*:}"
        status="${_rest%%:*}"; _rest="${_rest#*:}"
        remediation="${_rest%%:*}"; reason="${_rest#*:}"
        # If there was no 5th field, reason equals remediation; clear it
        if [[ "$reason" == "$remediation" ]]; then reason=""; fi
        remediation="${remediation%%[[:space:]]}"
        reason="${reason%%[[:space:]]}"
        if validate_health_result "$hostname" "$status" "$remediation"; then
            node_status["$hostname"]="$status"
            node_remediation["$hostname"]="$remediation"
            node_reason["$hostname"]="$reason"
        else
            node_status["$hostname"]="Skipped"
            node_remediation["$hostname"]="none"
            node_reason["$hostname"]="invalid HEALTH_CHECK_RESULT line"
        fi
    done < <(grep "^HEALTH_CHECK_RESULT:" "$worker_log" 2>/dev/null || true)
done

# Collect nodes by final status for batch feature update
batch_passed="" batch_failed="" batch_skipped=""

for node in $expanded; do
    if [[ -z "$node" ]]; then continue; fi
    status="${node_status[$node]:-}"
    remediation="${node_remediation[$node]:-}"
    reason="${node_reason[$node]:-}"
    node_rc="${WORKER_RC[$node]:-0}"

    if [[ -z "$status" ]]; then
        if [[ $node_rc -ne 0 ]]; then
            warn "No result for $node — worker exited with code $node_rc, treating as Skipped"
            status="Skipped"; remediation="none"; reason="no result (worker exit code $node_rc — possible timeout or crash)"
        else
            warn "No result for $node — treating as Passed"
            status="Passed"; remediation="none"; reason=""
        fi
        node_status["$node"]="$status"
        node_remediation["$node"]="$remediation"
        node_reason["$node"]="$reason"
    fi

    if [[ "$status" == "Skipped" ]]; then
        warn "Node $node: Skipped (inconclusive) — marking Skipped, no remediation${reason:+ (reason: $reason)}"
        batch_skipped="${batch_skipped:+${batch_skipped},}${node}"
        need_error="${need_error} ${node}"
        if [[ "$overall" != "Failed" ]]; then overall="Skipped"; fi
    elif [[ "$status" == "Passed" ]]; then
        apply_remediation "$node" "$remediation" "$REMEDIATE"
        batch_passed="${batch_passed:+${batch_passed},}${node}"
        passed="${passed} ${node}"
    else
        apply_remediation "$node" "$remediation" "$REMEDIATE"
        overall="Failed"
        batch_failed="${batch_failed:+${batch_failed},}${node}"
        case "$remediation" in
            replace) need_replace="${need_replace} ${node}" ;;
            reboot)  need_reboot="${need_reboot} ${node}" ;;
        esac
    fi
done

# Batch update Slurm features by status group
if [[ -n "$batch_passed" ]]; then  update_nodes_feature "$batch_passed" "Passed"; fi
if [[ -n "$batch_failed" ]]; then  update_nodes_feature "$batch_failed" "Failed"; fi
if [[ -n "$batch_skipped" ]]; then update_nodes_feature "$batch_skipped" "Skipped"; fi

SUMMARY_LOG="${OUTPUT_DIR}/health_check_summary_${SLURM_JOB_ID}_${TIMESTAMP}.log"
{
    echo ""
    echo "=========================================="
    echo "||          Health Check Summary         ||"
    echo "=========================================="
    echo "Overall Status: $overall"
    echo "Remediation applied: $REMEDIATE"
    echo "=========================================="
    if [[ -n "$passed" ]]; then       echo "Nodes PASSED:$passed"; fi
    if [[ -n "$need_error" ]]; then
        echo "Nodes SKIPPED:$need_error"
        for node in $need_error; do
            [[ -z "$node" ]] && continue
            local_reason="${node_reason[$node]:-}"
            local_status="${node_status[$node]:-}"
            if [[ -n "$local_reason" ]]; then echo "  $node ($local_status): $local_reason"; fi
        done
    fi
    if [[ -n "$need_reboot" ]]; then
        echo "Nodes requiring REBOOT:$need_reboot"
        for node in $need_reboot; do
            [[ -z "$node" ]] && continue
            local_reason="${node_reason[$node]:-}"
            local_status="${node_status[$node]:-}"
            if [[ -n "$local_reason" ]]; then echo "  $node ($local_status): $local_reason"; fi
        done
    fi
    if [[ -n "$need_replace" ]]; then
        echo "Nodes requiring REPLACE:$need_replace"
        for node in $need_replace; do
            [[ -z "$node" ]] && continue
            local_reason="${node_reason[$node]:-}"
            local_status="${node_status[$node]:-}"
            if [[ -n "$local_reason" ]]; then echo "  $node ($local_status): $local_reason"; fi
        done
    fi
    echo "=========================================="
    echo "Output directory: $OUTPUT_DIR"
    echo "Per-node worker logs:"
    for node in $expanded; do
        [[ -z "$node" ]] && continue
        echo "  $node: ${OUTPUT_DIR}/worker_${node}_${TIMESTAMP}.log (exit code ${WORKER_RC[$node]:-N/A})"
    done
    echo "=========================================="
} > "$SUMMARY_LOG"

echo "Summary written to: $SUMMARY_LOG"

exit 0
