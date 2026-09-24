#!/usr/bin/env bash
# Check 5: Multi-Node NCCL all_reduce Performance Test
# Runs all_reduce_perf from the NCCL tests container across allocated nodes.
# Validates bus bandwidth and verifies EFA provider selection.
# Runtime: ~10-20 minutes
# Requires: minimum 2 nodes, Pyxis/Enroot or standalone MPI

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

CHECK_NAME="5-nccl-allreduce"
# NCCL_CONTAINER is consumed by `srun --container-image=...` (Pyxis/Enroot).
# Enroot separates the registry from the image path with '#'. Callers may
# supply a staged .sqsh, including a native SM120 build for G7.
NCCL_CONTAINER="${NCCL_CONTAINER:-docker://public.ecr.aws#hpc-cloud/nccl-tests:cuda13.0.2-efa1.48.0-ofiv1.19.0-ncclv2.30.4-1-testsv2.18.3}"
# NCCL_TESTS_BIN overrides the executable in either the container or host.
NCCL_MPI="${NCCL_MPI:-pmix}"
NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"  # 30-minute timeout
NCCL_ISOLATION_TESTS="${NCCL_ISOLATION_TESTS:-0}"
NCCL_ISOLATION_TIMEOUT="${NCCL_ISOLATION_TIMEOUT:-600}"  # 10-minute timeout per isolation sub-test

# Minimum expected bus bandwidth (GB/s) per instance type.
# These are conservative defaults; override with NCCL_MIN_BUS_BW env var
# if you have a well-characterized baseline for your cluster.
get_min_bus_bw() {
    # Environment override takes precedence over per-instance defaults
    if [[ -n "${NCCL_MIN_BUS_BW:-}" ]]; then
        echo "${NCCL_MIN_BUS_BW}"
        return
    fi
    case "$1" in
        p4d.24xlarge)    echo 300 ;;
        p5.48xlarge)     echo 800 ;;
        p5e.48xlarge)    echo 800 ;;
        p5en.48xlarge)   echo 800 ;;
        p6-b200.48xlarge) echo 900 ;;
        *)               echo 0 ;;
    esac
}

run_check() {
    init_check "${CHECK_NAME}"

    # Determine number of nodes
    local num_nodes=1
    if [[ -n "${SLURM_JOB_NUM_NODES:-}" ]]; then
        num_nodes="${SLURM_JOB_NUM_NODES}"
    elif [[ -n "${SLURM_NNODES:-}" ]]; then
        num_nodes="${SLURM_NNODES}"
    elif [[ -n "${HEALTHCHECK_NUM_NODES:-}" ]]; then
        num_nodes="${HEALTHCHECK_NUM_NODES}"
    fi

    if [[ "${num_nodes}" -lt 2 ]]; then
        check_skip "${CHECK_NAME}" \
            "NCCL all_reduce requires minimum 2 nodes (allocated: ${num_nodes})"
        return 0
    fi

    local gpus_per_node="${EXPECTED_GPU_COUNT:-0}"
    if [[ ! "${gpus_per_node}" =~ ^[1-9][0-9]*$ ]]; then
        check_fail "${CHECK_NAME}" \
            "No positive GPU count for ${INSTANCE_TYPE}; add a verified instance profile before NCCL validation" "RESET"
        return 1
    fi

    local use_container=0
    local nccl_binary="${NCCL_TESTS_BIN:-all_reduce_perf}"
    # Consume the full help output: grep -q can SIGPIPE srun under pipefail.
    if srun --help 2>&1 | grep "container-image" > /dev/null; then
        use_container=1
        # The suite's NCCL image installs this binary outside PATH.
        nccl_binary="${NCCL_TESTS_BIN:-/opt/nccl-tests/build/all_reduce_perf}"
    fi

    # Override inherited SLURM_NTASKS: one process uses all GPUs on each node.
    local -a launch_args=(--nodes="${num_nodes}" --ntasks="${num_nodes}"
        --ntasks-per-node=1 --mpi="${NCCL_MPI}" --cpu-bind=none)

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo -e "${YELLOW}[DRY-RUN]${NC} srun ${launch_args[*]} (container=${use_container}, image=${NCCL_CONTAINER})" >&2
        echo -e "${YELLOW}[DRY-RUN]${NC}   ${nccl_binary} -b 8 -e 128M -f 2 -g ${gpus_per_node}" >&2
        check_pass "${CHECK_NAME}" "Dry-run: NCCL all_reduce skipped"
        return 0
    fi

    # Compute bandwidth threshold early (used by both isolation sub-tests and main test)
    local min_expected
    min_expected=$(get_min_bus_bw "${INSTANCE_TYPE}")

    # ── Optional NCCL isolation sub-tests ────────────────────────────────────
    if [[ "${NCCL_ISOLATION_TESTS}" == "1" ]]; then
        log_info "Running NCCL isolation sub-tests (NCCL_ISOLATION_TESTS=1)"

        # NVLink-only thresholds (GB/s)
        local nvlink_only_threshold=0
        case "${INSTANCE_TYPE}" in
            p4d.24xlarge)      nvlink_only_threshold=200 ;;
            p5.48xlarge)       nvlink_only_threshold=500 ;;
            p5e.48xlarge)      nvlink_only_threshold=500 ;;
            p5en.48xlarge)     nvlink_only_threshold=500 ;;
            p6-b200.48xlarge)  nvlink_only_threshold=600 ;;
        esac

        # One MPI process per node; a unique color per process keeps NCCL node-local.
        # nccl-tests v2.18.3 checks SPLIT_MASK before SPLIT and splits MPI_COMM_WORLD.
        if [[ "${NVLINK_EXPECTED}" == "true" ]]; then
            log_info "Running NVLink-only isolation test"
            local nvlink_test_output=""
            local nvlink_test_exit=0

            if [[ "${use_container}" == "1" ]]; then
                nvlink_test_output=$(timeout "${NCCL_ISOLATION_TIMEOUT}" \
                    srun "${launch_args[@]}" \
                         --container-image="${NCCL_CONTAINER}" \
                         env NCCL_TESTS_SPLIT_MASK=0xffffffff NCCL_P2P_LEVEL=NVL NCCL_NET=Socket \
                         "${nccl_binary}" -g "${gpus_per_node}" -b 256M -e 256M \
                    2>&1) || nvlink_test_exit=$?
            elif command -v "${nccl_binary}" > /dev/null 2>&1; then
                nvlink_test_output=$(timeout "${NCCL_ISOLATION_TIMEOUT}" \
                    srun "${launch_args[@]}" \
                         env NCCL_TESTS_SPLIT_MASK=0xffffffff NCCL_P2P_LEVEL=NVL NCCL_NET=Socket \
                         "${nccl_binary}" -g "${gpus_per_node}" -b 256M -e 256M \
                    2>&1) || nvlink_test_exit=$?
            fi

            if [[ ${nvlink_test_exit} -eq 124 ]]; then
                log_warn "NVLink-only isolation test timed out after ${NCCL_ISOLATION_TIMEOUT}s -- proceeding to full test"
            elif [[ ${nvlink_test_exit} -ne 0 ]]; then
                log_warn "NVLink-only isolation test failed (exit ${nvlink_test_exit}) -- proceeding to full test"
            fi

            if [[ -n "${nvlink_test_output}" ]]; then
                echo "${nvlink_test_output}" > "${RESULTS_DIR}/nccl-nvlink-only.txt"
                local nvlink_busbw
                nvlink_busbw=$(echo "${nvlink_test_output}" | grep -E "^\s+[0-9]" | awk '{print $(NF-1)}' \
                    | sort -n | tail -1 || echo "0")

                if [[ "${nvlink_only_threshold}" -gt 0 ]]; then
                    local nvlink_bw_int
                    nvlink_bw_int=$(echo "${nvlink_busbw}" | awk '{printf "%d", $1}')
                    if [[ "${nvlink_bw_int}" -lt "${nvlink_only_threshold}" ]]; then
                        check_warn "${CHECK_NAME}" \
                            "NVLink-only bandwidth ${nvlink_busbw} GB/s below expected ${nvlink_only_threshold} GB/s"
                    else
                        log_verbose "NVLink-only bandwidth ${nvlink_busbw} GB/s OK (threshold: ${nvlink_only_threshold} GB/s)"
                    fi
                fi
            fi
        else
            log_info "Skipping NVLink-only isolation test: ${INSTANCE_TYPE} profile has no NVLink"
        fi

        # EFA-only test (only when >= 2 nodes)
        if [[ "${num_nodes}" -ge 2 ]]; then
            log_info "Running EFA-only isolation test"
            local efa_test_output=""
            local efa_test_exit=0

            if [[ "${use_container}" == "1" ]]; then
                efa_test_output=$(NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_NET='AWS Libfabric' \
                    timeout "${NCCL_ISOLATION_TIMEOUT}" \
                    srun "${launch_args[@]}" \
                         --container-image="${NCCL_CONTAINER}" \
                         "${nccl_binary}" -g "${gpus_per_node}" -b 256M -e 256M \
                    2>&1) || efa_test_exit=$?
            elif command -v "${nccl_binary}" > /dev/null 2>&1; then
                efa_test_output=$(NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_NET='AWS Libfabric' \
                    timeout "${NCCL_ISOLATION_TIMEOUT}" \
                    srun "${launch_args[@]}" \
                         "${nccl_binary}" -g "${gpus_per_node}" -b 256M -e 256M \
                    2>&1) || efa_test_exit=$?
            fi

            if [[ ${efa_test_exit} -eq 124 ]]; then
                log_warn "EFA-only isolation test timed out after ${NCCL_ISOLATION_TIMEOUT}s -- proceeding to full test"
            elif [[ ${efa_test_exit} -ne 0 ]]; then
                log_warn "EFA-only isolation test failed (exit ${efa_test_exit}) -- proceeding to full test"
            fi

            if [[ -n "${efa_test_output}" ]]; then
                echo "${efa_test_output}" > "${RESULTS_DIR}/nccl-efa-only.txt"
                local efa_busbw
                efa_busbw=$(echo "${efa_test_output}" | grep -E "^\s+[0-9]" | awk '{print $(NF-1)}' \
                    | sort -n | tail -1 || echo "0")

                if [[ "${min_expected}" -gt 0 ]]; then
                    local efa_bw_int
                    efa_bw_int=$(echo "${efa_busbw}" | awk '{printf "%d", $1}')
                    if [[ "${efa_bw_int}" -lt "${min_expected}" ]]; then
                        check_warn "${CHECK_NAME}" \
                            "EFA-only bandwidth ${efa_busbw} GB/s below expected ${min_expected} GB/s"
                    else
                        log_verbose "EFA-only bandwidth ${efa_busbw} GB/s OK (threshold: ${min_expected} GB/s)"
                    fi
                fi
            fi
        fi
    fi

    # Set NCCL/EFA environment variables
    export FI_PROVIDER="${EFA_PROVIDER:-efa}"
    export FI_EFA_USE_DEVICE_RDMA=1
    export NCCL_NET_GDR_LEVEL=2
    export NCCL_DEBUG=INFO

    log_info "Running NCCL all_reduce_perf across ${num_nodes} nodes (${gpus_per_node} GPUs/node)"

    local nccl_output
    local nccl_exit=0

    # Try Pyxis/Enroot first, fall back to direct execution
    if [[ "${use_container}" == "1" ]]; then
        log_info "Using Pyxis/Enroot container runtime"
        nccl_output=$(run_with_timeout "${NCCL_TIMEOUT}" \
            srun "${launch_args[@]}" \
                 --container-image="${NCCL_CONTAINER}" \
                 "${nccl_binary}" -b 8 -e 128M -f 2 -g "${gpus_per_node}" \
            2>&1) || nccl_exit=$?
    elif command -v "${nccl_binary}" > /dev/null 2>&1; then
        log_info "Using locally installed NCCL tests"
        nccl_output=$(run_with_timeout "${NCCL_TIMEOUT}" \
            srun "${launch_args[@]}" \
                 "${nccl_binary}" -b 8 -e 128M -f 2 -g "${gpus_per_node}" \
            2>&1) || nccl_exit=$?
    else
        check_fail "${CHECK_NAME}" \
            "Neither Pyxis container runtime nor local all_reduce_perf found" "RESET"
        return 1
    fi

    # Save raw output
    echo "${nccl_output}" > "${RESULTS_DIR}/nccl-allreduce-raw.txt"

    if [[ ${nccl_exit} -eq 124 ]]; then
        check_fail "${CHECK_NAME}" \
            "NCCL all_reduce timed out after ${NCCL_TIMEOUT}s" "RESET"
        return 1
    fi

    if [[ ${nccl_exit} -ne 0 ]]; then
        # Check for out-of-bound errors
        if echo "${nccl_output}" | grep -qi "out of bound\|NCCL WARN\|unhandled system error"; then
            check_fail "${CHECK_NAME}" \
                "NCCL all_reduce failed with errors (exit ${nccl_exit})" "ISOLATE"
        else
            check_fail "${CHECK_NAME}" \
                "NCCL all_reduce failed (exit ${nccl_exit})" "RESET"
        fi
        return 1
    fi

    # Verify EFA provider was selected
    if ! grep -Eqi "(Selected Provider is efa|Using network EFA)([^[:alnum:]_-]|$)" <<< "${nccl_output}"; then
        check_warn "${CHECK_NAME}" \
            "EFA provider not confirmed in NCCL output -- performance may be degraded"
    fi

    # Validate the default nccl-tests all_reduce table before accepting exit 0.
    # Columns: size count type redop root time algbw busbw #wrong time algbw busbw #wrong.
    # Missing/disabled correctness, malformed rows, or missing completion are not PASS.
    local max_busbw parse_exit=0
    max_busbw=$(awk '
        function number(value) {
            # Lexical validity alone permits overflow (e.g. 1e9999).
            # Formatting a converted finite, nonnegative value starts with a digit;
            # awk infinity/NaN spellings do not. Keep nccl-tests scientific notation.
            return value ~ /^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$/ &&
                sprintf("%.17g", value + 0) ~ /^[0-9]/
        }
        function integer(value) {
            return value ~ /^[0-9]+$/ && number(value)
        }
        # nccl-tests writeResultHeader/writeResultFooter delimit the table.
        # Do not infer row membership from fields that may themselves be damaged.
        $1 == "#" && $2 == "size" {
            if (NF != 14 || $3 != "count" || $4 != "type" || $5 != "redop" ||
                $6 != "root" || $7 != "time" || $8 != "algbw" || $9 != "busbw" ||
                $10 != "#wrong" || $11 != "time" || $12 != "algbw" ||
                $13 != "busbw" || $14 != "#wrong") invalid = 1
            headers++; in_table = 1
            next
        }
        /Out of bounds values/ {
            if (!in_table || NF != 8 || $1 != "#" || $2 != "Out" || $3 != "of" ||
                $4 != "bounds" || $5 != "values" || $6 != ":" ||
                !integer($7) || $8 != "OK") invalid = 1
            if (integer($7) && $7 != 0) wrong = 1
            summaries++; in_table = 0
            next
        }
        # Units, comments and blank lines are not results. Only anchored NCCL
        # diagnostic prefixes are exempt; a result containing INFO is still a row.
        /^[[:space:]]*#/ || NF == 0 { next }
        /^[[:space:]]*([^[:space:]]+:[0-9]+:[0-9]+[[:space:]]+\[[0-9]+\][[:space:]]+)?NCCL[[:space:]]+(INFO|WARN|TRACE|VERSION)([[:space:]]|$)/ { next }
        in_table {
            if (NF != 13 || !integer($1) || !integer($2) ||
                $3 != "float" || $4 != "sum" || $5 != "-1") {
                invalid = 1; next
            }
            for (i = 6; i <= 12; i++) {
                if (i != 9 && !number($i)) invalid = 1
            }
            if (!integer($9) || !integer($13)) invalid = 1
            else if ($9 != 0 || $13 != 0) wrong = 1
            rows++
            if ($12 > max_bw) max_bw = $12
        }
        END {
            if (wrong) exit 2
            if (invalid || in_table || !headers || !rows || !summaries) exit 1
            print max_bw + 0
        }
    ' "${RESULTS_DIR}/nccl-allreduce-raw.txt") || parse_exit=$?
    if [[ ${parse_exit} -ne 0 ]]; then
        local parse_severity="RESET"
        [[ ${parse_exit} -eq 2 ]] && parse_severity="ISOLATE"
        check_fail "${CHECK_NAME}" \
            "NCCL correctness validation failed: missing, malformed, or nonzero #wrong / Out of bounds results (see raw output)" "${parse_severity}"
        return 1
    fi

    log_info "Maximum bus bandwidth: ${max_busbw} GB/s"

    # Compare against minimum threshold
    if [[ "${min_expected}" -gt 0 ]]; then
        local busbw_int
        busbw_int=$(echo "${max_busbw}" | awk '{printf "%d", $1}')
        if [[ "${busbw_int}" -lt "${min_expected}" ]]; then
            # Bandwidth below threshold is advisory (MONITOR) unless the operator
            # has set NCCL_MIN_BUS_BW to a well-characterized baseline.
            check_warn "${CHECK_NAME}" \
                "Bus bandwidth ${max_busbw} GB/s below minimum ${min_expected} GB/s for ${INSTANCE_TYPE}"
        fi
    fi

    check_pass "${CHECK_NAME}" \
        "NCCL all_reduce OK: ${num_nodes} nodes, max busbw=${max_busbw} GB/s"
    return 0
}

# ─── Entry point ─────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

run_check
