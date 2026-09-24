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
#   - FI_EFA_IFACE=<kernel device name>: pin libfabric to the specific EFA
#     device. NOTE: this takes the kernel/ibv device name (e.g. "rdmap80s0"),
#     not the libfabric domain name (e.g. "rdmap80s0-rdm") -- the "-rdm" suffix
#     must be stripped from the domain string discovered below.
#     (FI_EFA_DEVICE_NAME is NOT a real libfabric env var -- it does not
#     appear in `fi_info -e`'s FI_EFA_* list and is silently ignored, which
#     previously made every "per-device" iteration below run on whichever
#     device libfabric picks by default. Confirmed against a running
#     libfabric.so.1 2.4.0amzn3.0: 0 string occurrences of FI_EFA_DEVICE_NAME,
#     vs. FI_EFA_IFACE present and honored.)
#   - explicit -B server_port / -B client_port -P server_port: avoid port
#     collisions when called per-device in a loop.
# Returns 0 on success, non-zero on failure. Writes server+client logs to stdout
# on failure for triage.
run_pingpong_for_domain() {
    local domain="$1"
    # FI_EFA_IFACE takes the kernel/ibv device name, not the libfabric domain
    # name -- strip the "-rdm" suffix (see comment above run_pingpong_for_domain
    # invocation site / the header comment block for why).
    local iface="${domain%-rdm}"
    local server_port client_port
    server_port=$(shuf -n 1 -i 49152-57342)
    client_port=$(shuf -n 1 -i 57343-65535)

    local server_log client_log
    server_log=$(mktemp)
    client_log=$(mktemp)

    FI_LOG_LEVEL=warn FI_EFA_ENABLE_SHM_TRANSFER=0 FI_EFA_IFACE="${iface}" \
        fi_pingpong -e rdm -p efa -B "${server_port}" > "${server_log}" 2>&1 &
    local server_pid=$!
    sleep 3

    if ! kill -0 "${server_pid}" 2>/dev/null; then
        wait "${server_pid}" 2>/dev/null || true
        log_warn "Domain ${domain}: server failed to start"
        cat "${server_log}" >&2
        rm -f "${server_log}" "${client_log}"
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

    rm -f "${server_log}" "${client_log}"
    return "${ret}"
}

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

    # Discover EFA libfabric DOMAINS, not kernel ibv device names. The two
    # naming spaces differ: ibv_devices returns names like 'rdmap86s0', but
    # libfabric domains (as reported by `fi_info -p efa -t FI_EP_RDM`) are
    # named like 'rdmap86s0-rdm'. FI_EFA_IFACE (used to pin below) takes the
    # kernel/ibv name, so the '-rdm' suffix gets stripped per-device.
    # Enumerating via fi_info gets us the correct domain names and also
    # naturally excludes back-side Ethernet NICs that show up under
    # ibv_devices but are not EFA endpoints.
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
    log_info "Testing ${device_count} EFA domain(s)"

    # Regression guard: a bogus, definitely-nonexistent device name MUST fail.
    # This is the exact failure signature of the FI_EFA_DEVICE_NAME defect
    # (a per-device pinning env var that libfabric silently ignores, so every
    # "per-device" iteration -- including one given a nonexistent name -- ran
    # on whichever device libfabric picked by default and reported PASS). If
    # this negative control ever passes, per-device pinning is broken again
    # and the results below cannot be trusted as per-device -- fail loudly
    # instead of reporting a green per-device sweep that isn't one.
    log_verbose "Running negative-control check: bogus device name must fail loopback"
    local negctrl_exit=0
    run_pingpong_for_domain "definitely-nonexistent-device-rdm" > /dev/null 2>&1 || negctrl_exit=$?
    if [[ "${negctrl_exit}" -eq 0 ]]; then
        check_fail "${CHECK_NAME}" \
            "Negative control failed: a nonexistent device name (definitely-nonexistent-device-rdm) returned PASS. Per-device pinning is not working -- results below cannot be trusted as per-device. (This is the FI_EFA_DEVICE_NAME silent-no-op failure mode; see fix commit.)" \
            "RESET"
        return 1
    fi
    log_verbose "Negative control OK: bogus device name correctly failed (exit ${negctrl_exit})"

    local failures=0
    local results_json="["

    while IFS= read -r domain; do
        [[ -z "${domain}" ]] && continue
        log_info "Testing domain: ${domain}"

        local test_exit=0
        run_pingpong_for_domain "${domain}" || test_exit=$?

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
                check_warn "${CHECK_NAME}" "EFA rx_drops detected (${rx_drops}) -- possible network issues"
            fi
            if [[ "${retrans_timeouts}" -gt 0 ]]; then
                check_warn "${CHECK_NAME}" "EFA retransmission timeouts detected (${retrans_timeouts})"
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

    check_pass "${CHECK_NAME}" \
        "EFA loopback OK: ${device_count} domain(s) tested"
    return 0
}

# ─── Entry point ─────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

run_check
