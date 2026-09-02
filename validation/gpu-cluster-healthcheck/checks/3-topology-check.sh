#!/usr/bin/env bash
# Check 3: GPU/NVLink/PCIe Topology Validation
# Validates GPU-to-GPU connectivity matrix, NVLink status,
# and PCIe switch groupings against expected topology.
# Runtime: ~10 seconds

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

CHECK_NAME="3-topology-check"

run_check() {
    init_check "${CHECK_NAME}"

    local failures=0

    # Step 1: Capture GPU topology matrix
    log_info "Capturing GPU topology matrix"
    local topo_output=""
    if [[ "${DRY_RUN}" != "1" ]]; then
        topo_output=$(nvidia-smi topo -m 2>&1) || {
            check_fail "${CHECK_NAME}" "nvidia-smi topo -m failed" "RESET"
            return 1
        }
    else
        echo -e "${YELLOW}[DRY-RUN]${NC} nvidia-smi topo -m" >&2
        check_pass "${CHECK_NAME}" "Dry-run: topology check skipped"
        return 0
    fi

    log_verbose "Topology matrix:\\n${topo_output}"

    # Save raw output
    echo "${topo_output}" > "${RESULTS_DIR}/topo-matrix.txt"

    # Step 2: Check for disconnected GPUs (off-diagonal "X" in the matrix)
    # The diagonal always shows "X" (GPU-to-self), so we skip it per row.
    local x_entries=0
    local row_idx=0
    while IFS= read -r row; do
        # Split the row fields (first field is "GPUn", rest are connectivity entries)
        local fields
        read -ra fields <<< "${row}"
        local col_idx=0
        for field in "${fields[@]:1}"; do
            if [[ "${field}" == "X" && "${col_idx}" -ne "${row_idx}" ]]; then
                x_entries=$((x_entries + 1))
            fi
            col_idx=$((col_idx + 1))
        done
        row_idx=$((row_idx + 1))
    done < <(echo "${topo_output}" | grep "^GPU")

    if [[ "${x_entries}" -gt 0 ]]; then
        check_fail "${CHECK_NAME}" \
            "Found ${x_entries} disconnected GPU link(s) in topology matrix (off-diagonal)" "ISOLATE"
        failures=$((failures + 1))
    fi

    # Step 3: Validate NVLink presence if expected
    if [[ "${NVLINK_EXPECTED}" == "true" ]]; then
        log_info "Validating NVLink connectivity"
        local nvlink_count
        nvlink_count=$(echo "${topo_output}" | grep -c "NV[0-9]" || true)

        if [[ "${nvlink_count}" -eq 0 ]]; then
            check_fail "${CHECK_NAME}" \
                "NVLink expected but no NV* connections found in topology" "ISOLATE"
            failures=$((failures + 1))
        else
            log_verbose "NVLink connections found: ${nvlink_count} entries"
        fi

        # Check NVLink status per GPU
        local nvlink_status
        nvlink_status=$(nvidia-smi nvlink --status 2>/dev/null || true)
        if [[ -n "${nvlink_status}" ]]; then
            local inactive_links
            inactive_links=$(echo "${nvlink_status}" | grep -ci "inactive" || true)
            if [[ "${inactive_links}" -gt 0 ]]; then
                check_warn "${CHECK_NAME}" \
                    "Found ${inactive_links} inactive NVLink(s)"
            fi
        fi

        # Step 3a: Check Fabric Manager status
        log_info "Checking nvidia-fabricmanager service status"
        if systemctl_available; then
            if systemctl list-unit-files nvidia-fabricmanager.service 2>/dev/null | grep -q nvidia-fabricmanager; then
                local fm_status
                fm_status=$(systemctl is-active nvidia-fabricmanager 2>/dev/null || true)
                if [[ "${fm_status}" != "active" ]]; then
                    check_fail "${CHECK_NAME}" \
                        "nvidia-fabricmanager service is not running (status: ${fm_status})" "RESET"
                    failures=$((failures + 1))
                else
                    log_verbose "nvidia-fabricmanager is active"
                fi
            else
                log_verbose "nvidia-fabricmanager service not found -- skipping"
            fi
        elif pgrep -f "nv-fabricmanager" > /dev/null 2>&1; then
            log_verbose "nvidia-fabricmanager is running (detected via pgrep)"
        else
            check_fail "${CHECK_NAME}" \
                "nvidia-fabricmanager process not detected" "RESET"
            failures=$((failures + 1))
        fi

        # Step 3b: Query NVLink error counters
        log_info "Querying NVLink error counters"
        local nvlink_error_output
        if nvlink_error_output=$(nvidia-smi nvlink -e 2>/dev/null); then
            echo "${nvlink_error_output}" > "${RESULTS_DIR}/nvlink-errors.txt"

            # Parse for non-zero error counts across Replay, Recovery, and CRC fields
            local replay_errors=0 recovery_errors=0 crc_errors=0
            replay_errors=$(echo "${nvlink_error_output}" \
                | grep -iE "Replay Errors" | awk -F: '{sum += $NF} END {print sum+0}' || echo 0)
            recovery_errors=$(echo "${nvlink_error_output}" \
                | grep -iE "Recovery Errors" | awk -F: '{sum += $NF} END {print sum+0}' || echo 0)
            crc_errors=$(echo "${nvlink_error_output}" \
                | grep -iE "CRC Error" | awk -F: '{sum += $NF} END {print sum+0}' || echo 0)

            if [[ "${replay_errors}" -gt 0 || "${recovery_errors}" -gt 0 || "${crc_errors}" -gt 0 ]]; then
                check_warn "${CHECK_NAME}" \
                    "NVLink errors detected: ${replay_errors} replay, ${recovery_errors} recovery, ${crc_errors} CRC errors across links"
            else
                log_verbose "No NVLink errors detected"
            fi
        else
            log_warn "nvidia-smi nvlink -e failed -- NVLink error counters not available"
        fi
    fi

    # Step 4: Validate PCIe switch groupings
    log_info "Checking PCIe switch groupings"
    local pcie_groups
    pcie_groups=$(echo "${topo_output}" | grep "^GPU" | head -1 || true)
    # Verify all GPUs are visible in topology
    local topo_gpu_count
    topo_gpu_count=$(echo "${topo_output}" | grep -c "^GPU" || true)

    if [[ -n "${EXPECTED_GPU_COUNT}" && "${EXPECTED_GPU_COUNT}" -gt 0 ]]; then
        if [[ "${topo_gpu_count}" -ne "${EXPECTED_GPU_COUNT}" ]]; then
            check_fail "${CHECK_NAME}" \
                "Topology GPU count mismatch: expected=${EXPECTED_GPU_COUNT}, topo=${topo_gpu_count}" \
                "ISOLATE"
            failures=$((failures + 1))
        fi
    fi

    # Step 5: B200-specific P2P check
    if [[ "${INSTANCE_TYPE}" == *"b200"* ]]; then
        log_info "Running B200-specific P2P read/write/native check"
        local p2p_output
        p2p_output=$(nvidia-smi topo -p2p rwn 2>&1 || true)
        echo "${p2p_output}" > "${RESULTS_DIR}/topo-p2p.txt"

        local p2p_failures
        p2p_failures=$(echo "${p2p_output}" | grep -ci "not supported" || true)
        if [[ "${p2p_failures}" -gt 0 ]]; then
            check_warn "${CHECK_NAME}" \
                "B200 P2P: ${p2p_failures} unsupported P2P path(s) detected"
        fi
    fi

    # Final result
    if [[ ${failures} -gt 0 ]]; then
        return 1
    fi

    check_pass "${CHECK_NAME}" \
        "Topology OK: ${topo_gpu_count} GPUs, connectivity validated"
    return 0
}

# ─── Entry point ─────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

run_check
