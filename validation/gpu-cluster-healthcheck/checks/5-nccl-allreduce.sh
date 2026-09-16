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
# Enroot separates the registry from the image path with '#'. This default
# pins the base image version; callers may supply a digest or staged .sqsh.
# G7 requires a staged build with native SM120 kernels; see the suite README.
NCCL_CONTAINER="${NCCL_CONTAINER:-docker://public.ecr.aws#hpc-cloud/nccl-tests:cuda13.0.2-efa1.48.0-ofiv1.19.0-ncclv2.30.4-1-testsv2.18.3}"
NCCL_TESTS_BIN="${NCCL_TESTS_BIN:-/opt/nccl-tests/build/all_reduce_perf}"
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
        check_fail "${CHECK_NAME}" "No positive GPU count for ${INSTANCE_TYPE}; add a verified instance profile before NCCL validation" "RESET"
        return 1
    fi

    # salloc exports the job CPU layout, but not SLURM_CPUS_ON_NODE.
    # A homogeneous allocation gives each one-process-per-node launch its CPUs.
    local cpus_per_node="${SLURM_CPUS_ON_NODE:-1}"
    if [[ -n "${SLURM_JOB_CPUS_PER_NODE:-}" ]]; then
        if [[ "${SLURM_JOB_CPUS_PER_NODE}" =~ ^([1-9][0-9]*)(\(x[1-9][0-9]*\))?$ ]]; then
            cpus_per_node="${BASH_REMATCH[1]}"
        else
            check_fail "${CHECK_NAME}" "NCCL check requires a homogeneous CPU allocation; got ${SLURM_JOB_CPUS_PER_NODE}" "RESET"
            return 1
        fi
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo -e "${YELLOW}[DRY-RUN]${NC} srun --nodes=${num_nodes} --ntasks=${num_nodes} --ntasks-per-node=1 --cpus-per-task=${cpus_per_node} --mpi=${NCCL_MPI} --cpu-bind=none --container-image=${NCCL_CONTAINER} \\" >&2
        echo -e "${YELLOW}[DRY-RUN]${NC}   ${NCCL_TESTS_BIN} -b 8 -e 128M -f 2 -c 1 -g ${gpus_per_node}" >&2
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

        # NVLink-only test (always runs, single-node per-process)
        log_info "Running NVLink-only isolation test"
        local nvlink_test_output=""
        local nvlink_test_exit=0

        if srun --help 2>&1 | grep -q "container-image"; then
            nvlink_test_output=$(NCCL_P2P_LEVEL=NVL NCCL_NET=Socket \
                timeout "${NCCL_ISOLATION_TIMEOUT}" \
                srun --nodes="${num_nodes}" --ntasks="${num_nodes}" --ntasks-per-node=1 --cpus-per-task="${cpus_per_node}" --mpi="${NCCL_MPI}" --cpu-bind=none \
                     --container-image="${NCCL_CONTAINER}" \
                     "${NCCL_TESTS_BIN}" -g "${gpus_per_node}" -b 256M -e 256M \
                2>&1) || nvlink_test_exit=$?
        elif command -v "${NCCL_TESTS_BIN}" > /dev/null 2>&1; then
            nvlink_test_output=$(NCCL_P2P_LEVEL=NVL NCCL_NET=Socket \
                timeout "${NCCL_ISOLATION_TIMEOUT}" \
                srun --nodes="${num_nodes}" --ntasks="${num_nodes}" --ntasks-per-node=1 --cpus-per-task="${cpus_per_node}" --mpi="${NCCL_MPI}" --cpu-bind=none \
                     "${NCCL_TESTS_BIN}" -g "${gpus_per_node}" -b 256M -e 256M \
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

        # EFA-only test (only when >= 2 nodes)
        if [[ "${num_nodes}" -ge 2 ]]; then
            log_info "Running EFA-only isolation test"
            local efa_test_output=""
            local efa_test_exit=0

            if srun --help 2>&1 | grep -q "container-image"; then
                efa_test_output=$(NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_NET='AWS Libfabric' \
                    timeout "${NCCL_ISOLATION_TIMEOUT}" \
                    srun --nodes="${num_nodes}" --ntasks="${num_nodes}" --ntasks-per-node=1 --cpus-per-task="${cpus_per_node}" --mpi="${NCCL_MPI}" --cpu-bind=none \
                         --container-image="${NCCL_CONTAINER}" \
                         "${NCCL_TESTS_BIN}" -g "${gpus_per_node}" -b 256M -e 256M \
                    2>&1) || efa_test_exit=$?
            elif command -v "${NCCL_TESTS_BIN}" > /dev/null 2>&1; then
                efa_test_output=$(NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_NET='AWS Libfabric' \
                    timeout "${NCCL_ISOLATION_TIMEOUT}" \
                    srun --nodes="${num_nodes}" --ntasks="${num_nodes}" --ntasks-per-node=1 --cpus-per-task="${cpus_per_node}" --mpi="${NCCL_MPI}" --cpu-bind=none \
                         "${NCCL_TESTS_BIN}" -g "${gpus_per_node}" -b 256M -e 256M \
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
    if srun --help 2>&1 | grep -q "container-image"; then
        log_info "Using Pyxis/Enroot container runtime"
        nccl_output=$(run_with_timeout "${NCCL_TIMEOUT}" \
            srun --nodes="${num_nodes}" --ntasks="${num_nodes}" --ntasks-per-node=1 --cpus-per-task="${cpus_per_node}" --mpi="${NCCL_MPI}" --cpu-bind=none \
                 --container-image="${NCCL_CONTAINER}" \
                 "${NCCL_TESTS_BIN}" -b 8 -e 128M -f 2 -c 1 -g "${gpus_per_node}" \
            2>&1) || nccl_exit=$?
    elif command -v "${NCCL_TESTS_BIN}" > /dev/null 2>&1; then
        log_info "Using locally installed NCCL tests"
        nccl_output=$(run_with_timeout "${NCCL_TIMEOUT}" \
            srun --nodes="${num_nodes}" --ntasks="${num_nodes}" --ntasks-per-node=1 --cpus-per-task="${cpus_per_node}" --mpi="${NCCL_MPI}" --cpu-bind=none \
                 "${NCCL_TESTS_BIN}" -b 8 -e 128M -f 2 -c 1 -g "${gpus_per_node}" \
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
        if grep -qi "out of bound\|NCCL WARN\|unhandled system error" "${RESULTS_DIR}/nccl-allreduce-raw.txt"; then
            check_fail "${CHECK_NAME}" \
                "NCCL all_reduce failed with errors (exit ${nccl_exit})" "ISOLATE"
        else
            check_fail "${CHECK_NAME}" \
                "NCCL all_reduce failed (exit ${nccl_exit})" "RESET"
        fi
        return 1
    fi

    # Correctness columns must be present for both out-of-place and in-place
    # reductions. Empty output or a disabled check is not a successful test.
    if ! awk '
        BEGIN {expected=8}
        $1 ~ /^[0-9]+$/ {
            if (NF != 13 || $1 != expected) bad++
            if ($9 !~ /^[0-9]+$/ || $13 !~ /^[0-9]+$/ || $9 != 0 || $13 != 0) bad++
            expected *= 2
        }
        END {exit !(expected == 268435456 && bad == 0)}' "${RESULTS_DIR}/nccl-allreduce-raw.txt"; then
        check_fail "${CHECK_NAME}" "NCCL sweep incomplete or correctness results missing, disabled, or nonzero" "ISOLATE"
        return 1
    fi

    if ! grep -qi "Selected Provider is efa\|Using network EFA" "${RESULTS_DIR}/nccl-allreduce-raw.txt"; then
        check_fail "${CHECK_NAME}" "EFA provider not confirmed in NCCL output" "RESET"
        return 1
    fi

    local max_busbw
    max_busbw=$(echo "${nccl_output}" | awk '$1 ~ /^[0-9]+$/ && NF == 13 {if ($8 > m) m=$8; if ($12 > m) m=$12} END {print m+0}')
    log_info "Maximum bus bandwidth: ${max_busbw} GB/s"
    if awk -v bw="${max_busbw}" -v limit="${min_expected}" 'BEGIN {exit !(limit > 0 && bw < limit)}'; then
        check_warn "${CHECK_NAME}" "Bus bandwidth ${max_busbw} GB/s below minimum ${min_expected} GB/s for ${INSTANCE_TYPE}"
        return 0
    fi
    check_pass "${CHECK_NAME}" "NCCL all_reduce OK: ${num_nodes} nodes, max busbw=${max_busbw} GB/s, zero incorrect results"

    return 0
}

# ─── Entry point ─────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

run_check
