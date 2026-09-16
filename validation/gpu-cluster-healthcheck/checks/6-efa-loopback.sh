#!/usr/bin/env bash
# Check 6: EFA Loopback Connectivity Test
# Iterates over all EFA libfabric domains and runs a per-device self-loopback
# connectivity test using AWS's canonical fi_pingpong invocation.
# Runtime: ~5-15 minutes depending on EFA device count

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

CHECK_NAME="6-efa-loopback"
EFA_TEST_TIMEOUT="${EFA_TEST_TIMEOUT:-180}"  # Per-device timeout
EFA_INSTALLER_TEST="${EFA_INSTALLER_TEST:-/opt/amazon/efa/test/efa_test.sh}"

# Run a self-loopback fi_pingpong against a single libfabric EFA domain.
# Mirrors the canonical invocation from AWS's /opt/amazon/efa/test/efa_test.sh:
#   - -e rdm: endpoint type FI_EP_RDM
#   - -p efa: libfabric efa provider
#   - FI_EFA_ENABLE_SHM_TRANSFER=0: force the real EFA hardware path; otherwise
#     libfabric routes same-host traffic through SHM and the test does not
#     exercise EFA at all.
#   - FI_EFA_IFACE=<kernel-device>: pin libfabric to the specific EFA domain.
#   - explicit -B server_port / -B client_port -P server_port: avoid port
#     collisions when called per-device in a loop.
# Returns 0 on success, non-zero on failure. Writes server+client logs to stdout
# on failure for triage.
run_pingpong_for_domain() (
    local domain="$1"
    local iface="${domain%-rdm}"
    local server_port client_port
    server_port=$(shuf -n 1 -i 49152-57342)
    client_port=$(shuf -n 1 -i 57343-65535)

    local server_log client_log
    server_log="${RESULTS_DIR}/efa-${domain}-server.log"
    client_log="${RESULTS_DIR}/efa-${domain}-client.log"

    FI_LOG_LEVEL=warn FI_EFA_ENABLE_SHM_TRANSFER=0 FI_EFA_IFACE="${iface}" \
        fi_pingpong -e rdm -p efa -B "${server_port}" > "${server_log}" 2>&1 &
    local server_pid=$!
    trap 'kill "${server_pid}" 2>/dev/null || true' EXIT
    sleep 3

    if ! kill -0 "${server_pid}" 2>/dev/null; then
        wait "${server_pid}" 2>/dev/null || true
        log_warn "Domain ${domain}: server failed to start"
        cat "${server_log}" >&2
        return 1
    fi

    local ret=0
    FI_LOG_LEVEL=warn FI_EFA_ENABLE_SHM_TRANSFER=0 FI_EFA_IFACE="${iface}" \
        timeout "${EFA_TEST_TIMEOUT}" \
        fi_pingpong -e rdm -p efa -B "${client_port}" -P "${server_port}" localhost \
        > "${client_log}" 2>&1 || ret=$?

    kill -9 "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true

    if [[ "${ret}" -ne 0 ]]; then
        log_warn "Domain ${domain}: fi_pingpong client exit ${ret}"
        echo "--- server log (${domain}) ---" >&2
        cat "${server_log}" >&2
        echo "--- client log (${domain}) ---" >&2
        cat "${client_log}" >&2
    fi

    return "${ret}"
)

run_check() {
    init_check "${CHECK_NAME}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo -e "${YELLOW}[DRY-RUN]${NC} fi_pingpong -e rdm -p efa for each libfabric EFA domain" >&2
        check_pass "${CHECK_NAME}" "Dry-run: EFA loopback tests skipped"
        return 0
    fi

    if ! command -v fi_pingpong > /dev/null 2>&1; then
        log_warn "fi_pingpong not found on PATH; EFA installer may not be present"
        if [[ -x /opt/amazon/efa/bin/fi_pingpong ]]; then
            export PATH="/opt/amazon/efa/bin:${PATH}"
        else
            check_skip "${CHECK_NAME}" "fi_pingpong not available (install the AWS EFA installer)"
            return 0
        fi
    fi
    if ! command -v fi_info > /dev/null 2>&1; then
        if [[ -x /opt/amazon/efa/bin/fi_info ]]; then
            export PATH="/opt/amazon/efa/bin:${PATH}"
        else
            check_fail "${CHECK_NAME}" "fi_info not found -- libfabric not installed" "RESET"
            return 1
        fi
    fi

    # fi_info yields libfabric domain names. FI_EFA_IFACE selects the kernel
    # RDMA name, so run_pingpong_for_domain removes the '-rdm' suffix.
    local domains
    domains=$(fi_info -p efa -t FI_EP_RDM 2>/dev/null \
        | awk '/^[[:space:]]*domain:/{print $2}' \
        | sort -u)

    if [[ -z "${domains}" ]]; then
        check_fail "${CHECK_NAME}" "No EFA libfabric domains found (fi_info -p efa -t FI_EP_RDM returned empty)" "ISOLATE"
        return 1
    fi

    local device_count
    device_count=$(echo "${domains}" | wc -l | tr -d ' ')
    if [[ "${EXPECTED_EFA_COUNT:-0}" -gt 0 && "${device_count}" -ne "${EXPECTED_EFA_COUNT}" ]]; then
        check_fail "${CHECK_NAME}" "EFA domain count mismatch: expected=${EXPECTED_EFA_COUNT}, detected=${device_count}" "ISOLATE"
        return 1
    fi
    log_info "Testing ${device_count} EFA domain(s)"

    local failures=0
    local stats_warning=0
    local results_json="["

    while IFS= read -r domain; do
        [[ -z "${domain}" ]] && continue
        log_info "Testing domain: ${domain}"

        # Retain attribution evidence for every physical EFA during each test.
        grep -H . /sys/class/infiniband/*/ports/1/hw_counters/{send_bytes,recv_bytes} \
            > "${RESULTS_DIR}/efa-${domain}-before.txt" 2>/dev/null || true
        local test_exit=0
        run_pingpong_for_domain "${domain}" || test_exit=$?

        grep -H . /sys/class/infiniband/*/ports/1/hw_counters/{send_bytes,recv_bytes} \
            > "${RESULTS_DIR}/efa-${domain}-after.txt" 2>/dev/null || true

        if [[ "${test_exit}" -ne 0 ]]; then
            failures=$((failures + 1))
        fi

        if [[ "${results_json}" != "[" ]]; then
            results_json+=","
        fi
        results_json+=$(cat <<ENDJSON

    {
      "domain": "${domain}",
      "status": "$([ ${test_exit} -eq 0 ] && echo 'PASS' || echo 'FAIL')",
      "exit_code": ${test_exit}
    }
ENDJSON
)
    done <<< "${domains}"

    results_json+=$'\n]'

    echo "${results_json}" > "${RESULTS_DIR}/efa-loopback-results.json"

    log_info "Collecting EFA statistics"
    if command -v rdma > /dev/null 2>&1; then
        local efa_stats
        if efa_stats=$(rdma -p statistic show 2>/dev/null); then
            echo "${efa_stats}" > "${RESULTS_DIR}/efa-statistics.txt"

            local rx_drops
            rx_drops=$(echo "${efa_stats}" | grep -oP 'rx_drops\s+\K[0-9]+' | awk '{sum+=$1} END {print sum+0}') || rx_drops=0
            local retrans_timeouts
            retrans_timeouts=$(echo "${efa_stats}" | grep -oP 'retrans_timeout_events\s+\K[0-9]+' | awk '{sum+=$1} END {print sum+0}') || retrans_timeouts=0

            if [[ "${rx_drops}" -gt 0 ]]; then
                log_warn "EFA rx_drops detected (${rx_drops}) -- possible network issues"
                stats_warning=1
            fi
            if [[ "${retrans_timeouts}" -gt 0 ]]; then
                log_warn "EFA retransmission timeouts detected (${retrans_timeouts})"
                stats_warning=1
            fi
            if [[ "${rx_drops}" -eq 0 && "${retrans_timeouts}" -eq 0 ]]; then
                log_verbose "EFA statistics clean -- no drops or retransmissions"
            fi
        else
            log_verbose "rdma statistic show failed -- EFA statistics skipped"
        fi
    else
        log_verbose "rdma tool not found -- EFA statistics skipped"
    fi

    if [[ ${failures} -gt 0 ]]; then
        check_fail "${CHECK_NAME}" \
            "${failures}/${device_count} EFA domain(s) failed loopback test" "RESET"
        return 1
    fi

    if [[ ${stats_warning} -eq 1 ]]; then
        check_warn "${CHECK_NAME}" "EFA loopback completed for ${device_count} domain(s); cumulative EFA statistics contain drops or retransmission timeouts (see logs)"
    else
        check_pass "${CHECK_NAME}" "EFA loopback OK: ${device_count} domain(s) tested"
    fi
    return 0
}

# ─── Entry point ─────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

run_check
