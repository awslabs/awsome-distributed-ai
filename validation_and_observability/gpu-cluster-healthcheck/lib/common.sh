#!/usr/bin/env bash
# common.sh -- Shared utilities for GPU health check suite
# Provides logging, instance detection, profile loading, and result formatting.

set -euo pipefail

# ─── Color codes (disabled when stdout is not a terminal) ────────────────────
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    RED='' GREEN='' YELLOW='' BLUE='' BOLD='' NC=''
fi

# ─── Globals ─────────────────────────────────────────────────────────────────
_COMMON_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${_COMMON_LIB_DIR}/.." && pwd)"
INSTANCE_PROFILES_FILE="${BASE_DIR}/instance-profiles.conf"
RESULTS_DIR="${RESULTS_DIR:-/tmp/gpu-healthcheck-${SLURM_JOB_ID:-$(date +%s)}}"
VERBOSE="${VERBOSE:-0}"
DRY_RUN="${DRY_RUN:-0}"
JSON_OUTPUT="${JSON_OUTPUT:-0}"

# Instance profile variables (populated by load_instance_profile)
INSTANCE_TYPE=""
EXPECTED_GPU_COUNT=""
EXPECTED_EFA_COUNT=""
NVLINK_EXPECTED=""
EFA_PROVIDER=""

# ─── Logging ─────────────────────────────────────────────────────────────────

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

log_verbose() {
    if [[ "${VERBOSE}" == "1" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $*" >&2
    fi
}

# ─── Instance Detection ─────────────────────────────────────────────────────

detect_instance_type() {
    # Try IMDSv2 first, fall back to IMDSv1, then ec2-metadata CLI
    local token
    token=$(curl -s --connect-timeout 2 -X PUT "http://169.254.169.254/latest/api/token" \
        -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" 2>/dev/null || true)

    if [[ -n "${token}" ]]; then
        INSTANCE_TYPE=$(curl -s --connect-timeout 2 -H "X-aws-ec2-metadata-token: ${token}" \
            "http://169.254.169.254/latest/meta-data/instance-type" 2>/dev/null || true)
    fi

    if [[ -z "${INSTANCE_TYPE}" ]]; then
        INSTANCE_TYPE=$(curl -s --connect-timeout 2 "http://169.254.169.254/latest/meta-data/instance-type" 2>/dev/null || true)
    fi

    if [[ -z "${INSTANCE_TYPE}" ]]; then
        INSTANCE_TYPE=$(ec2-metadata --instance-type 2>/dev/null | awk '{print $2}' || true)
    fi

    if [[ -z "${INSTANCE_TYPE}" ]]; then
        log_error "Unable to detect instance type via IMDS or ec2-metadata"
        return 1
    fi

    log_verbose "Detected instance type: ${INSTANCE_TYPE}"
    echo "${INSTANCE_TYPE}"
}

# ─── Instance Profile Loading ───────────────────────────────────────────────

load_instance_profile() {
    if [[ -z "${INSTANCE_TYPE}" ]]; then
        detect_instance_type > /dev/null
    fi

    if [[ ! -f "${INSTANCE_PROFILES_FILE}" ]]; then
        log_error "Instance profiles file not found: ${INSTANCE_PROFILES_FILE}"
        return 1
    fi

    local profile_line
    # Use awk for exact literal match (instance types contain dots which
    # are regex wildcards in grep).
    profile_line=$(awk -F'|' -v inst="${INSTANCE_TYPE}" '$1==inst {print; exit}' \
        "${INSTANCE_PROFILES_FILE}")

    if [[ -z "${profile_line}" ]]; then
        log_warn "No profile found for instance type: ${INSTANCE_TYPE}"
        log_warn "Using defaults -- results may be unreliable"
        EXPECTED_GPU_COUNT=0
        EXPECTED_EFA_COUNT=0
        NVLINK_EXPECTED="false"
        EFA_PROVIDER="efa"
        return 0
    fi

    IFS='|' read -r _ EXPECTED_GPU_COUNT EXPECTED_EFA_COUNT NVLINK_EXPECTED EFA_PROVIDER <<< "${profile_line}"
    log_verbose "Profile loaded: GPUs=${EXPECTED_GPU_COUNT}, EFA=${EXPECTED_EFA_COUNT}, NVLink=${NVLINK_EXPECTED}, Provider=${EFA_PROVIDER}"
}

# ─── Results Directory ───────────────────────────────────────────────────────

ensure_results_dir() {
    mkdir -p "${RESULTS_DIR}"
    log_verbose "Results directory: ${RESULTS_DIR}"
}

# ─── Result Formatting ───────────────────────────────────────────────────────

log_result() {
    local check_name="$1"
    local status="$2"   # PASS, FAIL, WARN, SKIP
    local details="${3:-}"
    local severity="${4:-}"

    # Build JSON via python3 so all string values are properly escaped
    # (details may contain quotes, newlines, backslashes from command output).
    local json_result
    json_result="$(
        CHECK_NAME="${check_name}" \
        STATUS="${status}" \
        DETAILS="${details}" \
        SEVERITY="${severity}" \
        INSTANCE_TYPE="${INSTANCE_TYPE}" \
        python3 -c '
import json, os, socket
from datetime import datetime, timezone
print(json.dumps({
    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "hostname": socket.gethostname(),
    "instance_type": os.environ.get("INSTANCE_TYPE", ""),
    "check": os.environ["CHECK_NAME"],
    "status": os.environ["STATUS"],
    "details": os.environ.get("DETAILS", ""),
    "severity": os.environ.get("SEVERITY", ""),
}))
'
    )"

    if [[ "${JSON_OUTPUT}" == "1" ]]; then
        echo "${json_result}"
    fi

    # Write to results file if results dir exists
    if [[ -d "${RESULTS_DIR}" ]]; then
        local safe_name
        safe_name="$(printf '%s' "${check_name}" | tr -cs 'A-Za-z0-9._-' '-')"
        # Atomic write: write to tmp then rename to avoid partial files
        local tmpfile="${RESULTS_DIR}/.check-${safe_name}.json.tmp"
        echo "${json_result}" > "${tmpfile}"
        mv -f "${tmpfile}" "${RESULTS_DIR}/check-${safe_name}.json"
    fi
}

check_pass() {
    local check_name="$1"
    local details="${2:-}"
    echo -e "${GREEN}[PASS]${NC} ${BOLD}${check_name}${NC}: ${details}"
    log_result "${check_name}" "PASS" "${details}"
}

check_fail() {
    local check_name="$1"
    local details="${2:-}"
    local severity="${3:-ISOLATE}"
    echo -e "${RED}[FAIL]${NC} ${BOLD}${check_name}${NC}: ${details} (severity: ${severity})"
    log_result "${check_name}" "FAIL" "${details}" "${severity}"
}

check_warn() {
    local check_name="$1"
    local details="${2:-}"
    echo -e "${YELLOW}[WARN]${NC} ${BOLD}${check_name}${NC}: ${details}"
    log_result "${check_name}" "WARN" "${details}" "MONITOR"
}

check_skip() {
    local check_name="$1"
    local details="${2:-}"
    echo -e "${BLUE}[SKIP]${NC} ${BOLD}${check_name}${NC}: ${details}"
    log_result "${check_name}" "SKIP" "${details}"
}

# ─── Dry Run Helper ─────────────────────────────────────────────────────────

run_cmd() {
    # Execute a command, or print it in dry-run mode
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo -e "${YELLOW}[DRY-RUN]${NC} $*" >&2
        return 0
    fi
    "$@"
}

# ─── Timeout Helper ─────────────────────────────────────────────────────────

run_with_timeout() {
    local timeout_secs="$1"
    shift
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo -e "${YELLOW}[DRY-RUN]${NC} timeout ${timeout_secs}s: $*" >&2
        return 0
    fi
    timeout "${timeout_secs}" "$@"
}

# ─── Environment Detection ─────────────────────────────────────────────────
systemctl_available() {
    command -v systemctl &>/dev/null && systemctl --version &>/dev/null 2>&1
}

# ─── Pre-flight: source instance profile on load ────────────────────────────

preflight_checks() {
    # Verify critical dependencies and log version info for diagnostics.
    # Returns non-zero if a hard dependency (python3) is missing.

    # python3 is a hard dependency: result formatting, DCGM parsing, and
    # aggregation all require it.
    if ! command -v python3 &>/dev/null; then
        log_error "python3 not found on PATH -- required for result formatting and DCGM parsing"
        return 1
    fi
    local py_version
    py_version=$(python3 --version 2>&1 || true)
    log_verbose "Python: ${py_version}"

    # nvidia-smi and driver version (soft dependency -- may not be present in dry-run)
    if command -v nvidia-smi &>/dev/null; then
        local driver_version
        driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
        if [[ -n "${driver_version}" ]]; then
            log_info "NVIDIA driver version: ${driver_version}"
        fi
    else
        log_warn "nvidia-smi not found on PATH"
    fi

    # DCGM version (optional -- only needed for checks 1 and 4)
    if command -v dcgmi &>/dev/null; then
        local dcgm_version
        dcgm_version=$(dcgmi --version 2>/dev/null | grep -i "version" | head -1 || true)
        if [[ -n "${dcgm_version}" ]]; then
            log_info "DCGM: ${dcgm_version}"
        fi
    else
        log_verbose "dcgmi not found on PATH (optional -- needed for DCGM checks)"
    fi

    return 0
}

init_check() {
    local check_name="$1"
    ensure_results_dir
    log_info "Running check: ${check_name}"

    # Load instance profile if not already loaded
    if [[ -z "${EXPECTED_GPU_COUNT}" ]]; then
        load_instance_profile || true
    fi
}
