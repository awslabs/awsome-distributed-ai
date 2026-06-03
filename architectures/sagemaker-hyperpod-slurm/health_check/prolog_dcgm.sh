#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Slurm Prolog: DCGM GPU health check before job execution.
# Pass → job proceeds. Fail → behavior controlled by FAILURE_ACTION.
# Recent passes are cached to skip redundant checks.

set -euo pipefail

# --- Configuration (edit directly) ---
readonly DCGM_LEVEL=2                        # DCGM diagnostic level (2-4)
readonly CACHE_TTL_HOURS=1                   # If a cached result exists and is less than CACHE_TTL_HOURS hours old, skip the prolog check. Set to 0 to disable caching entirely.
readonly UPDATE_FEATURES=true                # Update Slurm node features
readonly FAILURE_ACTION="none"               # Action on DCGM failure: "none" = mark Failed, job proceeds; "drain" = exit 1, Slurm drains node & requeues job; "remediate" = set node State=FAIL for reboot/replace via HyperPod
readonly PROLOG_BASE_DIR="${HC_PROLOG_BASE_DIR:-/fsx/health_check_prolog}"
readonly LOG_DIR="${PROLOG_BASE_DIR}/logs"
readonly CACHE_DIR="${PROLOG_BASE_DIR}/cache"
readonly FEATURE_PREFIX="HealthCheck"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly DCGM_SCRIPT="${SCRIPT_DIR}/dcgm.sh"
readonly NODE_HOSTNAME="$(hostname)"
readonly CACHE_FILE="${CACHE_DIR}/${NODE_HOSTNAME}.last_pass"
readonly CACHE_TTL_SECONDS=$(( CACHE_TTL_HOURS * 3600 ))

# --- Helpers ---
log() { echo "[prolog_dcgm][${NODE_HOSTNAME}] $*"; }
die() { log "ERROR: $*"; exit 1; }

# --- Slurm feature update ---
update_node_feature() {
    local new_status="$1"
    local desired_active="${FEATURE_PREFIX}:${new_status}"
    local all_values="${FEATURE_PREFIX}:InProgress,${FEATURE_PREFIX}:Passed,${FEATURE_PREFIX}:Failed,${FEATURE_PREFIX}:Skipped"

    local node_info
    if ! node_info=$(scontrol show node "$NODE_HOSTNAME"); then
        log "WARNING: scontrol show node failed for $NODE_HOSTNAME — skipping feature update"
        return 1
    fi

    local avail active
    avail=$(echo "$node_info" | grep -oP 'AvailableFeatures=\K[^ ]*' || echo "")
    active=$(echo "$node_info" | grep -oP 'ActiveFeatures=\K[^ ]*' || echo "")

    # AvailableFeatures: strip old health-check values, add all states
    local base_avail
    base_avail=$(echo "$avail" | sed "s/${FEATURE_PREFIX}:[^,]*//g" \
        | sed 's/,,*/,/g' | sed 's/^,//;s/,$//')
    local new_avail
    if [[ -n "$base_avail" && "$base_avail" != "(null)" ]]; then
        new_avail="${base_avail},${all_values}"
    else
        new_avail="${all_values}"
    fi

    # ActiveFeatures: replace health-check value with desired one
    local new_active
    if [[ "$active" == *"${FEATURE_PREFIX}:"* ]]; then
        new_active=$(echo "$active" | sed "s/${FEATURE_PREFIX}:[^,]*/${desired_active}/g")
    elif [[ -n "$active" && "$active" != "(null)" ]]; then
        new_active="${active},${desired_active}"
    else
        new_active="${desired_active}"
    fi

    scontrol update NodeName="$NODE_HOSTNAME" AvailableFeatures="$new_avail" 2>&1 \
        || log "WARNING: AvailableFeatures update failed"
    scontrol update NodeName="$NODE_HOSTNAME" ActiveFeatures="$new_active" 2>&1 \
        || log "WARNING: ActiveFeatures update failed"
}

maybe_update_feature() {
    if [[ "${UPDATE_FEATURES}" == "true" ]]; then update_node_feature "$1"; fi
}

# --- Cache ---
check_cache() {
    if (( CACHE_TTL_SECONDS == 0 )); then return 1; fi
    if [[ ! -f "$CACHE_FILE" ]]; then return 1; fi
    local last_pass now
    last_pass=$(cat "$CACHE_FILE" 2>/dev/null || echo "0")
    now=$(date +%s)
    if (( now - last_pass < CACHE_TTL_SECONDS )); then return 0; else return 1; fi
}

update_cache() {
    mkdir -p "$CACHE_DIR"
    date +%s > "$CACHE_FILE"
}

invalidate_cache() { rm -f "$CACHE_FILE"; }

# --- Main ---
main() {
    # Skip prolog DCGM check for health-check jobs to avoid circular blocking:
    # the orchestrator (health_check_main_job) and per-node workers (hc_*) ARE the health check.
    local job_name=""
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        job_name=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
            | grep -oP 'JobName=\K[^ ]*' || true)
    fi
    if [[ "$job_name" == "health_check_main_job" || "$job_name" == hc_* ]]; then
        log "Skipping prolog DCGM check for health-check job ${SLURM_JOB_ID:-unknown} (${job_name})"
        exit 0
    fi

    [[ "$DCGM_LEVEL" =~ ^[2-4]$ ]] || die "DCGM_LEVEL must be 2, 3, or 4 (got: '$DCGM_LEVEL')"
    [[ -f "$DCGM_SCRIPT" ]]        || die "dcgm.sh not found at: $DCGM_SCRIPT"
    mkdir -p "$LOG_DIR"

    local job_id="${SLURM_JOB_ID:-unknown}"

    if check_cache; then
        log "Cache hit — node passed DCGM L${DCGM_LEVEL} within last ${CACHE_TTL_HOURS}h, skipping (job ${job_id})"
        exit 0
    fi

    log "Running DCGM L${DCGM_LEVEL} check before job ${job_id}"
    maybe_update_feature "InProgress"

    local log_file="${LOG_DIR}/${NODE_HOSTNAME}_prolog_${job_id}.log"
    export HC_RESULTS_DIR="$LOG_DIR"
    export HC_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    # dcgmi warns when CUDA_VISIBLE_DEVICES is set; health check needs all GPUs anyway
    unset CUDA_VISIBLE_DEVICES

    local output="" dcgm_rc=0
    export HC_TEST_PARAMS='{"level":'"$DCGM_LEVEL"'}'
    output=$(bash "$DCGM_SCRIPT" 2>"$log_file") || dcgm_rc=$?

    local result_line
    result_line=$(echo "$output" | grep "^HEALTH_CHECK_RESULT:" || true)

    if [[ -z "$result_line" ]]; then
        log "No HEALTH_CHECK_RESULT from dcgm.sh (rc=${dcgm_rc}) — treating as Skipped (inconclusive)"
        maybe_update_feature "Skipped"
        invalidate_cache
        exit 0
    fi

    local _tag hostname status remediation reason
    # Parse colon-delimited fields; reason is everything after the 4th colon (may contain colons)
    _tag="${result_line%%:*}"; _rest="${result_line#*:}"
    hostname="${_rest%%:*}"; _rest="${_rest#*:}"
    status="${_rest%%:*}"; _rest="${_rest#*:}"
    remediation="${_rest%%:*}"; reason="${_rest#*:}"
    if [[ "$reason" == "$remediation" ]]; then reason=""; fi
    remediation="${remediation%%[[:space:]]}"
    reason="${reason%%[[:space:]]}"
    log "Result: status=${status} remediation=${remediation}${reason:+ reason=${reason}}"

    if [[ "$status" == "Passed" ]]; then
        log "DCGM PASSED — allowing job ${job_id}${reason:+ (reason: $reason)}"
        maybe_update_feature "Passed"
        update_cache
        exit 0
    elif [[ "$status" == "Skipped" ]]; then
        log "DCGM inconclusive — marking Skipped, allowing job ${job_id}${reason:+ (reason: $reason)}"
        maybe_update_feature "Skipped"
        invalidate_cache
        exit 0
    else
        maybe_update_feature "Failed"
        invalidate_cache
        case "${FAILURE_ACTION}" in
            none)
                log "DCGM FAILED (remediation=${remediation}${reason:+, reason=${reason}}) — marked Failed, job ${job_id} proceeds (FAILURE_ACTION=none)"
                exit 0
                ;;
            drain)
                log "DCGM FAILED (remediation=${remediation}${reason:+, reason=${reason}}) — draining node, requeuing job ${job_id} (FAILURE_ACTION=drain)"
                exit 1
                ;;
            remediate)
                if [[ "$remediation" == "reboot" || "$remediation" == "replace" ]]; then
                    local fail_reason="Action:${remediation^}"
                    log "DCGM FAILED — setting node State=FAIL Reason=${fail_reason} for HyperPod ${remediation} (FAILURE_ACTION=remediate)"
                    scontrol update NodeName="$NODE_HOSTNAME" State=FAIL Reason="$fail_reason" \
                        || log "WARNING: scontrol update State=FAIL failed for $NODE_HOSTNAME"
                else
                    log "DCGM FAILED (remediation=${remediation}${reason:+, reason=${reason}}) — no hardware remediation needed, job ${job_id} proceeds (FAILURE_ACTION=remediate)"
                fi
                exit 0
                ;;
            *)
                log "ERROR: Invalid FAILURE_ACTION '${FAILURE_ACTION}' — must be none, drain, or remediate. Allowing job ${job_id} to proceed."
                exit 0
                ;;
        esac
    fi
}

main "$@"
