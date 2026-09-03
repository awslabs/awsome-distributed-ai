#!/usr/bin/env bash
# Check 2: EFA Enumeration
# Validates EFA PCI device count, RDMA devices, libfabric provider,
# and /dev/infiniband device nodes against instance profile expectations.
# Runtime: ~3 seconds

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

CHECK_NAME="2-efa-enumeration"

run_check() {
    init_check "${CHECK_NAME}"

    local failures=0

    # Step 1: Count EFA PCI devices
    log_info "Enumerating EFA PCI devices"
    local efa_pci_count=0
    if [[ "${DRY_RUN}" != "1" ]]; then
        efa_pci_count=$(lspci | grep -ci "EFA" || true)
    else
        echo -e "${YELLOW}[DRY-RUN]${NC} lspci | grep -ci EFA" >&2
    fi
    log_verbose "EFA PCI devices found: ${efa_pci_count}"

    if [[ -n "${EXPECTED_EFA_COUNT}" && "${EXPECTED_EFA_COUNT}" -gt 0 ]]; then
        if [[ "${efa_pci_count}" -ne "${EXPECTED_EFA_COUNT}" ]]; then
            # P5en instances have known late EFA initialization -- retry
            if [[ "${DRY_RUN}" != "1" && "${INSTANCE_TYPE}" == p5en* ]]; then
                log_info "P5en detected -- retrying EFA count (known late initialization)"
                local retry
                for retry in 1 2 3; do
                    sleep 5
                    efa_pci_count=$(lspci | grep -ci "EFA" || true)
                    log_verbose "Retry ${retry}: EFA PCI devices found: ${efa_pci_count}"
                    if [[ "${efa_pci_count}" -eq "${EXPECTED_EFA_COUNT}" ]]; then
                        break
                    fi
                done
            fi

            if [[ "${efa_pci_count}" -ne "${EXPECTED_EFA_COUNT}" ]]; then
                check_fail "${CHECK_NAME}" \
                    "EFA PCI device count mismatch: expected=${EXPECTED_EFA_COUNT}, detected=${efa_pci_count}" \
                    "ISOLATE"
                failures=$((failures + 1))
            else
                log_verbose "EFA PCI device count matches expected: ${efa_pci_count}"
            fi
        else
            log_verbose "EFA PCI device count matches expected: ${efa_pci_count}"
        fi
    fi

    # Step 2: List RDMA devices
    log_info "Listing RDMA devices"
    local rdma_count=0
    local rdma_output=""
    if [[ "${DRY_RUN}" != "1" ]]; then
        if command -v ibv_devices > /dev/null 2>&1; then
            rdma_output=$(ibv_devices 2>/dev/null || true)
            # ibv_devices output: 2 header lines then one device per line.
            # Skip the header and count non-empty lines for a deterministic count.
            rdma_count=$(echo "${rdma_output}" | tail -n +3 | grep -c '[^[:space:]]' || true)
        else
            log_warn "ibv_devices not found -- RDMA tools may not be installed"
        fi
        # Cross-check against /sys/class/infiniband/ entries
        if [[ -d /sys/class/infiniband ]]; then
            local sysfs_count
            sysfs_count=$(ls -1 /sys/class/infiniband/ 2>/dev/null | wc -l || true)
            if [[ "${rdma_count}" -gt 0 && "${sysfs_count}" -gt 0 && "${rdma_count}" -ne "${sysfs_count}" ]]; then
                log_warn "RDMA device count mismatch: ibv_devices=${rdma_count}, /sys/class/infiniband=${sysfs_count}"
            fi
        fi
    else
        echo -e "${YELLOW}[DRY-RUN]${NC} ibv_devices" >&2
    fi
    log_verbose "RDMA devices found: ${rdma_count}"

    # Step 3: Check libfabric EFA provider
    log_info "Checking libfabric EFA provider"
    if [[ "${DRY_RUN}" != "1" ]]; then
        if command -v fi_info > /dev/null 2>&1; then
            local fi_output
            fi_output=$(fi_info -p efa 2>&1 || true)
            if echo "${fi_output}" | grep -qi "provider: efa" ; then
                log_verbose "EFA provider confirmed via fi_info"
            else
                check_warn "${CHECK_NAME}" \
                    "fi_info did not confirm EFA provider -- check libfabric installation"
            fi
        else
            log_warn "fi_info not found -- libfabric may not be installed"
        fi
    else
        echo -e "${YELLOW}[DRY-RUN]${NC} fi_info -p efa" >&2
    fi

    # Step 4: Validate /dev/infiniband device nodes
    log_info "Checking /dev/infiniband device nodes"
    local uverbs_count=0
    if [[ "${DRY_RUN}" != "1" ]]; then
        if [[ -d /dev/infiniband ]]; then
            uverbs_count=$(ls /dev/infiniband/uverbs* 2>/dev/null | wc -l || true)
        else
            log_warn "/dev/infiniband directory not found"
        fi
    else
        echo -e "${YELLOW}[DRY-RUN]${NC} ls /dev/infiniband/uverbs*" >&2
    fi
    log_verbose "uverbs device nodes found: ${uverbs_count}"

    # Step 5: Validate required kernel modules
    log_info "Checking EFA kernel modules"
    if [[ "${DRY_RUN}" != "1" ]]; then
        local required_modules=("efa" "ib_uverbs" "ib_core")
        for mod in "${required_modules[@]}"; do
            if ! lsmod | grep -qw "${mod}"; then
                check_warn "${CHECK_NAME}" "EFA kernel module ${mod} not loaded"
            fi
        done

        # Optional: check gdrdrv if GDRCopy is installed
        if [[ -d /opt/gdrcopy ]]; then
            if ! lsmod | grep -qw "gdrdrv"; then
                check_warn "${CHECK_NAME}" \
                    "gdrdrv module not loaded -- GPUDirect RDMA may fall back to slower paths"
            fi
        fi
    else
        echo -e "${YELLOW}[DRY-RUN]${NC} lsmod | grep -qw {efa,ib_uverbs,ib_core}" >&2
    fi

    # Step 6: GDRCopy validation
    if [[ -x /opt/gdrcopy/bin/sanity ]]; then
        log_info "Running GDRCopy sanity check"
        if [[ "${DRY_RUN}" != "1" ]]; then
            if run_with_timeout 30 /opt/gdrcopy/bin/sanity -v > /dev/null 2>&1; then
                log_verbose "GDRCopy sanity check passed"
            else
                check_warn "${CHECK_NAME}" \
                    "GDRCopy sanity check failed -- GPUDirect RDMA may not function"
            fi
        else
            echo -e "${YELLOW}[DRY-RUN]${NC} /opt/gdrcopy/bin/sanity -v" >&2
        fi
    fi

    # Step 7: Memory locking limits
    log_info "Checking memory lock limits"
    if [[ "${DRY_RUN}" != "1" ]]; then
        local memlock
        memlock=$(ulimit -l 2>/dev/null || echo "0")
        if [[ "${memlock}" != "unlimited" ]]; then
            if [[ "${memlock}" -lt 16777216 ]]; then
                check_warn "${CHECK_NAME}" \
                    "Memory lock limit ${memlock} KB is below 16 GiB -- EFA performance may be degraded"
            else
                log_verbose "Memory lock limit OK"
            fi
        else
            log_verbose "Memory lock limit OK"
        fi
    else
        echo -e "${YELLOW}[DRY-RUN]${NC} ulimit -l" >&2
    fi

    # Final result
    if [[ ${failures} -gt 0 ]]; then
        return 1
    fi

    check_pass "${CHECK_NAME}" \
        "EFA OK: ${efa_pci_count} PCI devices, ${rdma_count} RDMA devices, ${uverbs_count} uverbs nodes"
    return 0
}

# ─── Entry point ─────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

run_check
