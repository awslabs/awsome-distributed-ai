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

# Path of the result file a check writes its record to.
result_file_for() {
    local check_name="$1"
    local safe_name
    safe_name="$(printf '%s' "${check_name}" | tr -cs 'A-Za-z0-9._-' '-')"
    echo "${RESULTS_DIR}/check-${safe_name}.json"
}

log_result() {
    local check_name="$1"
    local status="$2"   # PASS, FAIL, WARN, SKIP
    local details="${3:-}"
    local severity="${4:-}"

    local result_file=""
    if [[ -d "${RESULTS_DIR}" ]]; then
        result_file="$(result_file_for "${check_name}")"
    fi

    # Build JSON via python3 so all string values are properly escaped (details
    # may contain quotes, newlines, backslashes from command output), and merge
    # it into any record this check has already written.
    local json_result
    json_result="$(
        CHECK_NAME="${check_name}" \
        STATUS="${status}" \
        DETAILS="${details}" \
        SEVERITY="${severity}" \
        INSTANCE_TYPE="${INSTANCE_TYPE}" \
        RESULT_FILE="${result_file}" \
        python3 -c '
import json, os, socket
from datetime import datetime, timezone

# One check emits several results -- an Xid FAIL, then a benign
# persistence-mode WARN -- and they all land in a single file per check.
# kubernetes/determine-severity.py, lib/aggregate-results.py and
# slurm/sbatch-quarantine-workflow.sh read only that file to decide whether a
# node is drained or replaced, so a later benign result must never overwrite
# an earlier fault. Keep the most severe status and severity seen instead.
STATUS_RANK = {"FAIL": 4, "WARN": 3, "PASS": 2, "SKIP": 1}
SEVERITY_RANK = {"ISOLATE": 4, "REBOOT": 3, "RESET": 2, "MONITOR": 1}

record = {
    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "hostname": socket.gethostname(),
    "instance_type": os.environ.get("INSTANCE_TYPE", ""),
    "check": os.environ["CHECK_NAME"],
    "status": os.environ["STATUS"],
    "details": os.environ.get("DETAILS", ""),
    "severity": os.environ.get("SEVERITY", ""),
}

# stdout stays a per-emission event stream; only the file is cumulative.
print(json.dumps(record))

result_file = os.environ.get("RESULT_FILE", "")
if not result_file:
    raise SystemExit(0)

previous = {}
try:
    with open(result_file) as fh:
        loaded = json.load(fh)
    if isinstance(loaded, dict):
        previous = loaded
except (OSError, ValueError):
    previous = {}

# Start from the previous record so the richer fields written directly by
# parse-dcgm-results.py (per-GPU "tests", "overall_*") survive.
merged = dict(previous)
merged.update(record)

if STATUS_RANK.get(previous.get("status", ""), 0) > STATUS_RANK.get(record["status"], 0):
    merged["status"] = previous["status"]
if SEVERITY_RANK.get(previous.get("severity", ""), 0) > SEVERITY_RANK.get(record["severity"], 0):
    merged["severity"] = previous["severity"]

# parse-dcgm-results.py seeds overall_status/overall_severity mirroring its
# status/severity. Once the merged status/severity are chosen, drop those
# mirror fields so the record cannot report a passing overall_* beside a
# failing status that a consumer keying on overall_* would misread.
merged.pop("overall_status", None)
merged.pop("overall_severity", None)

# Keep the evidence from every emission. Split on the accumulator delimiter so
# a distinct finding is not dropped just for being a substring of the running
# text, and each finding is recorded at most once. Details use "; " internally,
# so separate accumulated findings with " | ".
prev_details = previous.get("details", "") or ""
new_details = record["details"]
seen = [seg for seg in prev_details.split(" | ") if seg]
if new_details and new_details not in seen:
    seen.append(new_details)
merged["details"] = " | ".join(seen)

# Atomic write: write to tmp then rename to avoid partial files.
tmp_file = os.path.join(
    os.path.dirname(result_file), "." + os.path.basename(result_file) + ".tmp"
)
with open(tmp_file, "w") as fh:
    fh.write(json.dumps(merged) + "\n")
os.replace(tmp_file, result_file)
'
    )"

    if [[ "${JSON_OUTPUT}" == "1" ]]; then
        echo "${json_result}"
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

    # Results accumulate within one check execution (see log_result), so drop
    # any record left behind by an earlier run reusing this results directory.
    rm -f "$(result_file_for "${check_name}")"

    # Load instance profile if not already loaded
    if [[ -z "${EXPECTED_GPU_COUNT}" ]]; then
        load_instance_profile || true
    fi
}
