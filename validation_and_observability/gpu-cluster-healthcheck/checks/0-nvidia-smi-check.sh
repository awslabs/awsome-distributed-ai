#!/usr/bin/env bash
# Check 0: nvidia-smi Validation
# Verifies nvidia-smi is functional, GPU count matches expectations,
# and no recent Xid/SXid errors exist in kernel log.
# Runtime: ~5 seconds

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

CHECK_NAME="0-nvidia-smi"

# ─── Tunables ─────────────────────────────────────────────────────────────────
KERNEL_LOG_LINES="${KERNEL_LOG_LINES:-4000}"
NVIDIA_LOG_TAIL="${NVIDIA_LOG_TAIL:-200}"
STRICT_PSHC="${STRICT_PSHC:-0}"
NVLINK5_DEFAULT="${NVLINK5_DEFAULT:-MONITOR}"
NVLINK_DEFAULT="${NVLINK_DEFAULT:-RESET}"

# Validate NVLink tunables
for _var in NVLINK5_DEFAULT NVLINK_DEFAULT; do
    case "${!_var}" in
        MONITOR|RESET|REBOOT) ;;
        *) log_warn "Invalid ${_var}='${!_var}', falling back to MONITOR"
           printf -v "${_var}" '%s' "MONITOR" ;;
    esac
done
unset _var

# ─── Helper functions ────────────────────────────────────────────────────────

severity_rank() {
    case "${1:-}" in
        PASS)     echo 0 ;;
        MONITOR)  echo 1 ;;
        RESET)    echo 2 ;;
        REBOOT)   echo 3 ;;
        ISOLATE)  echo 4 ;;
        *)        echo 0 ;;
    esac
}

bump_severity() {
    local current="$1"
    local candidate="$2"
    local cr; cr=$(severity_rank "${current}")
    local ca; ca=$(severity_rank "${candidate}")
    if [[ "${ca}" -gt "${cr}" ]]; then
        echo "${candidate}"
    else
        echo "${current}"
    fi
}

# ─── Main check ──────────────────────────────────────────────────────────────

run_check() {
    init_check "${CHECK_NAME}"
    local check_exit_code=0

    # Step 1: Verify nvidia-smi executes successfully
    log_info "Verifying nvidia-smi availability"
    local smi_output
    if ! smi_output=$(run_cmd nvidia-smi --query-gpu=gpu_name,gpu_bus_id,memory.total \
        --format=csv,noheader 2>&1); then
        check_fail "${CHECK_NAME}" "nvidia-smi failed to execute" "ISOLATE"
        return 1
    fi

    # Step 2: Count detected GPUs
    local detected_gpus
    detected_gpus=$(echo "${smi_output}" | wc -l | tr -d ' ')
    log_verbose "Detected ${detected_gpus} GPU(s)"

    if [[ -n "${EXPECTED_GPU_COUNT}" && "${EXPECTED_GPU_COUNT}" -gt 0 ]]; then
        if [[ "${detected_gpus}" -ne "${EXPECTED_GPU_COUNT}" ]]; then
            check_fail "${CHECK_NAME}" \
                "GPU count mismatch: expected=${EXPECTED_GPU_COUNT}, detected=${detected_gpus}" \
                "ISOLATE"
            return 1
        fi
        log_verbose "GPU count matches expected: ${detected_gpus}"
    fi

    # Step 3: Read kernel log once for Xid and SXid scanning
    log_info "Checking kernel log for Xid/SXid errors"
    local kernel_log=""
    if [[ "${DRY_RUN}" != "1" ]]; then
        # Prefer journalctl; fall back to dmesg
        if command -v journalctl &>/dev/null; then
            kernel_log=$(journalctl -k -o short-iso --no-pager -n "${KERNEL_LOG_LINES}" 2>/dev/null || true)
        fi
        if [[ -z "${kernel_log}" ]]; then
            kernel_log=$(dmesg --time-format iso 2>/dev/null | tail -n "${KERNEL_LOG_LINES}" || true)
        fi
    else
        echo -e "${YELLOW}[DRY-RUN]${NC} journalctl -k / dmesg (Xid + SXid scan)" >&2
    fi

    # Extract Xid and SXid lines from the same buffer
    local xid_errors=""
    local sxid_errors=""
    if [[ -n "${kernel_log}" ]]; then
        xid_errors=$(echo "${kernel_log}" | grep -i "NVRM.*Xid" | tail -n "${NVIDIA_LOG_TAIL}" || true)
        sxid_errors=$(echo "${kernel_log}" | grep -i "SXid" | tail -n "${NVIDIA_LOG_TAIL}" || true)
    fi

    # Save raw errors for post-mortem (only if errors exist)
    if [[ -n "${xid_errors}" ]]; then
        echo "${xid_errors}" > "${RESULTS_DIR}/xid-errors.log"
    fi
    if [[ -n "${sxid_errors}" ]]; then
        echo "${sxid_errors}" > "${RESULTS_DIR}/sxid-errors.log"
    fi

    # GPU detection (for classification overrides)
    local is_a100=false
    local mig_enabled=false
    local uvm_in_use=false

    if echo "${smi_output}" | grep -qi "A100"; then
        is_a100=true
    fi

    if [[ "${DRY_RUN}" != "1" ]]; then
        local mig_query=""
        mig_query=$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader 2>/dev/null || true)
        if echo "${mig_query}" | grep -qi "enabled"; then
            mig_enabled=true
        fi

        # Check if UVM is in use (false-safe if tools unavailable)
        if command -v fuser &>/dev/null && fuser -s /dev/nvidia-uvm 2>/dev/null; then
            uvm_in_use=true
        elif command -v lsof &>/dev/null && lsof /dev/nvidia-uvm &>/dev/null; then
            uvm_in_use=true
        fi
    fi

    # Step 3a: Xid classification
    if [[ -n "${xid_errors}" ]]; then
        # Extract unique Xid codes using sed (no GNU grep -P dependency).
        # The NVIDIA kernel log format is: NVRM: Xid (PCI:0000:xx:00): <code>, ...
        # We skip past the closing parenthesis to avoid matching the PCI bus number.
        local xid_codes
        xid_codes=$(echo "${xid_errors}" | sed -nE 's/.*Xid[^)]*\):[[:space:]]*([0-9]+).*/\1/p' | sort -un || true)

        local highest_severity="MONITOR"
        local severity_details=""

        for code in ${xid_codes}; do
            local code_severity="MONITOR"
            local code_group="UNKNOWN"
            # Classification aligned with NVIDIA XID Errors r590 catalog
            # "Resolution Bucket (Immediate Action)" column
            case "${code}" in
                # ISOLATE: mechanical / hardware check required
                54)
                    code_severity="ISOLATE"; code_group="CHECK_MECHANICALS" ;;
                # REBOOT: bare-metal restart required
                79)
                    code_severity="REBOOT"; code_group="RESTART_BM" ;;
                # REBOOT: VM restart required
                151)
                    code_severity="REBOOT"; code_group="RESTART_VM" ;;
                # REBOOT: boot reattempt or enable ECC
                168)
                    code_severity="REBOOT"; code_group="BOOT_REATTEMPT_OR_ENABLE_ECC" ;;
                # RESET: workflow-driven (Xid 154 parsing may escalate)
                48)
                    code_severity="RESET"; code_group="WORKFLOW_XID_48" ;;
                # RESET: DRAM retirement failure (A100 override → REBOOT)
                64)
                    code_severity="RESET"; code_group="DRAM_RETIREMENT_FAILURE"
                    if [[ "${is_a100}" == true ]]; then
                        code_severity="REBOOT"
                    fi
                    ;;
                # RESET: GPU reset required (A100+MIG_off override → REBOOT)
                95)
                    code_severity="RESET"; code_group="RESET_GPU"
                    if [[ "${is_a100}" == true && "${mig_enabled}" == false ]]; then
                        code_severity="REBOOT"
                    fi
                    ;;
                # RESET: GPU reset required
                109|110|119|120|136|140|143|155|156|158)
                    code_severity="RESET"; code_group="RESET_GPU" ;;
                # NVLink error (configurable default)
                74)
                    code_severity="${NVLINK_DEFAULT}"; code_group="WORKFLOW_NVLINK_ERR" ;;
                # NVLink5 / Blackwell codes (configurable default)
                144|145|146|147|148|149|150)
                    code_severity="${NVLINK5_DEFAULT}"; code_group="WORKFLOW_NVLINK5_ERR" ;;
                # RESET if UVM in use, else MONITOR
                159)
                    if [[ "${uvm_in_use}" == true ]]; then
                        code_severity="RESET"
                    else
                        code_severity="MONITOR"
                    fi
                    code_group="CHECK_UVM" ;;
                # PSHC informational
                162|163)
                    code_severity="MONITOR"; code_group="PSHC_INFO" ;;
                # PSHC low lifetime
                164)
                    code_severity="MONITOR"; code_group="PSHC_LOW_LIFETIME" ;;
                # PSHC zero lifetime (STRICT_PSHC=1 → ISOLATE)
                165)
                    if [[ "${STRICT_PSHC}" == "1" ]]; then
                        code_severity="ISOLATE"
                    else
                        code_severity="MONITOR"
                    fi
                    code_group="PSHC_ZERO_LIFETIME" ;;
                # MONITOR: application restart recommended
                13|31|94|126)
                    code_severity="MONITOR"; code_group="RESTART_APP" ;;
                # MONITOR: informational / ignore class
                43|63|92|121)
                    code_severity="MONITOR"; code_group="IGNORE" ;;
                # MONITOR: Xid 154 is parsed separately for derived action
                154)
                    code_severity="MONITOR"; code_group="XID_154_INFO" ;;
                # MONITOR: contact support / informational
                157)
                    code_severity="MONITOR"; code_group="CONTACT_SUPPORT" ;;
                # MONITOR: unused/deprecated codes
                81|125)
                    code_severity="MONITOR"; code_group="CONTACT_SUPPORT" ;;
                *) code_severity="MONITOR"; code_group="UNKNOWN" ;;
            esac

            if [[ -n "${severity_details}" ]]; then
                severity_details+=", "
            fi
            severity_details+="Xid ${code} (${code_severity}/${code_group})"

            highest_severity=$(bump_severity "${highest_severity}" "${code_severity}")
        done

        # Xid 154 derived action parsing: authoritative recovery action text
        if echo "${xid_codes}" | grep -qw '154'; then
            local xid154_lines
            xid154_lines=$(echo "${xid_errors}" | grep -i 'Xid.*154' || true)
            local xid154_severity="MONITOR"

            if echo "${xid154_lines}" | grep -qi "Node Reboot Required"; then
                xid154_severity="REBOOT"
            elif echo "${xid154_lines}" | grep -qi "GPU Reset Required"; then
                xid154_severity="RESET"
            elif echo "${xid154_lines}" | grep -qi "Drain and Reset"; then
                xid154_severity="RESET"
            elif echo "${xid154_lines}" | grep -qi "Drain P2P"; then
                xid154_severity="RESET"
            fi
            # "(None)" or unrecognized → no escalation (stays MONITOR)

            if [[ "${xid154_severity}" != "MONITOR" ]]; then
                severity_details+=", Xid 154 derived action -> ${xid154_severity}"
                highest_severity=$(bump_severity "${highest_severity}" "${xid154_severity}")
            fi
        fi

        local xid_count
        xid_count=$(echo "${xid_errors}" | wc -l | tr -d ' ')

        if [[ "${highest_severity}" == "ISOLATE" || "${highest_severity}" == "REBOOT" || "${highest_severity}" == "RESET" ]]; then
            check_fail "${CHECK_NAME}" \
                "Found ${xid_count} Xid message(s): ${severity_details}" "${highest_severity}"
            check_exit_code=1
        else
            check_warn "${CHECK_NAME}" "Found ${xid_count} Xid message(s): ${severity_details}"
        fi
        log_verbose "Recent Xid errors:\\n${xid_errors}"
    fi

    # Step 3b: SXid (NVSwitch) errors from the same kernel_log buffer
    if [[ -n "${sxid_errors}" ]]; then
        local sxid_count
        sxid_count=$(echo "${sxid_errors}" | wc -l | tr -d ' ')
        local sxid_msg="Found ${sxid_count} SXid (NVSwitch) error(s) in kernel log"

        # SXid alongside Xid 74 suggests NVSwitch root cause
        if [[ -n "${xid_errors}" ]] && echo "${xid_errors}" | sed -nE 's/.*Xid[^)]*\):[[:space:]]*([0-9]+).*/\1/p' | grep -qw '74'; then
            sxid_msg+="; SXid with Xid 74 indicates likely NVSwitch root cause (recommend RESET)"
        fi

        check_warn "${CHECK_NAME}" "${sxid_msg}"
        log_verbose "Recent SXid errors:\\n${sxid_errors}"
    fi

    # Step 4: Check GPU persistence mode
    local persist_mode
    if persist_mode=$(run_cmd nvidia-smi --query-gpu=persistence_mode \
        --format=csv,noheader 2>/dev/null); then
        local disabled_count
        disabled_count=$(echo "${persist_mode}" | grep -ci "disabled" || true)
        if [[ "${disabled_count}" -gt 0 ]]; then
            check_warn "${CHECK_NAME}" \
                "Persistence mode disabled on ${disabled_count} GPU(s) -- recommend enabling"
        fi
    fi

    # Step 5: Capture GPU UUIDs (informational)
    log_info "Capturing GPU UUIDs"
    local uuid_output=""
    if uuid_output=$(run_cmd nvidia-smi --query-gpu=index,serial,uuid \
        --format=csv,noheader 2>/dev/null); then
        if [[ -n "${uuid_output}" ]]; then
            echo "${uuid_output}" > "${RESULTS_DIR}/gpu-uuids.csv"
            local uuid_count
            uuid_count=$(echo "${uuid_output}" | wc -l | tr -d ' ')
            log_info "Captured UUIDs for ${uuid_count} GPU(s)"
        fi
    else
        log_verbose "nvidia-smi UUID query failed -- skipping UUID capture"
    fi

    # Step 6: GPU ECC and retired pages check
    # Detects uncorrectable memory errors and page retirements that indicate
    # degrading GPU memory health. DBE (double-bit errors) are unrecoverable.
    local MAX_RETIRED_PAGES_SBE="${MAX_RETIRED_PAGES_SBE:-60}"

    log_info "Checking GPU ECC errors and retired pages"
    local ecc_output=""
    if ecc_output=$(run_cmd nvidia-smi \
        --query-gpu=index,ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total,retired_pages.pending,retired_pages.sbe,retired_pages.dbe \
        --format=csv,noheader 2>/dev/null); then

        if [[ -n "${ecc_output}" ]]; then
            echo "${ecc_output}" > "${RESULTS_DIR}/ecc-status.csv"

            local ecc_severity="PASS"
            local ecc_details=""

            while IFS=', ' read -r gpu_idx uncorr_vol uncorr_agg retire_pending retire_sbe retire_dbe; do
                # Skip GPUs that don't support ECC (fields report "N/A")
                if [[ "${uncorr_vol}" == "N/A" || "${uncorr_vol}" == "[N/A]" ]]; then
                    continue
                fi

                # Strip any whitespace from parsed fields
                uncorr_vol="${uncorr_vol// /}"
                uncorr_agg="${uncorr_agg// /}"
                retire_pending="${retire_pending// /}"
                retire_sbe="${retire_sbe// /}"
                retire_dbe="${retire_dbe// /}"

                # DBE (double-bit errors) > 0 → ISOLATE (unrecoverable memory errors)
                if [[ "${retire_dbe}" =~ ^[0-9]+$ && "${retire_dbe}" -gt 0 ]]; then
                    ecc_severity=$(bump_severity "${ecc_severity}" "ISOLATE")
                    ecc_details+="GPU ${gpu_idx}: ${retire_dbe} double-bit retired page(s); "
                fi

                # Retired pages pending → REBOOT (pending retirement needs reboot to take effect)
                if [[ "${retire_pending}" == "Yes" ]]; then
                    ecc_severity=$(bump_severity "${ecc_severity}" "REBOOT")
                    ecc_details+="GPU ${gpu_idx}: retired pages pending reboot; "
                fi

                # Uncorrected volatile errors > 0 → RESET
                if [[ "${uncorr_vol}" =~ ^[0-9]+$ && "${uncorr_vol}" -gt 0 ]]; then
                    ecc_severity=$(bump_severity "${ecc_severity}" "RESET")
                    ecc_details+="GPU ${gpu_idx}: ${uncorr_vol} uncorrected volatile ECC error(s); "
                fi

                # SBE retired pages approaching limit (NVIDIA limit is 64) → MONITOR
                if [[ "${retire_sbe}" =~ ^[0-9]+$ && "${retire_sbe}" -gt "${MAX_RETIRED_PAGES_SBE}" ]]; then
                    ecc_severity=$(bump_severity "${ecc_severity}" "MONITOR")
                    ecc_details+="GPU ${gpu_idx}: ${retire_sbe} SBE retired pages (threshold: ${MAX_RETIRED_PAGES_SBE}); "
                fi
            done <<< "${ecc_output}"

            if [[ "${ecc_severity}" != "PASS" ]]; then
                if [[ "${ecc_severity}" == "MONITOR" ]]; then
                    check_warn "${CHECK_NAME}" "ECC/retired pages: ${ecc_details}"
                else
                    check_fail "${CHECK_NAME}" "ECC/retired pages: ${ecc_details}" "${ecc_severity}"
                    check_exit_code=1
                fi
            else
                log_verbose "ECC and retired pages: all GPUs healthy"
            fi
        fi
    else
        log_verbose "nvidia-smi ECC query not supported -- skipping ECC check"
    fi

    # Emit final result (only PASS if no prior failures)
    if [[ "${check_exit_code}" -eq 0 ]]; then
        check_pass "${CHECK_NAME}" \
            "nvidia-smi OK, ${detected_gpus} GPU(s) detected"
    fi
    return "${check_exit_code}"
}

# ─── Entry point ─────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

run_check
