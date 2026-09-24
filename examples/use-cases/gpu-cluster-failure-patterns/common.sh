#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Sourced by numbered entry points; not an attendee command.
# These are allocation-level launchers, not MPI ranks. lab.env (or an enclosing
# step) can export PMIX_MCA_gds without any live PMIx server. Enroot 3.5's
# 50-slurm-pmi.sh treats ANY exported PMIX_ name as a server when MPI type is
# absent, even for srun --mpi=none. Drop inherited rank/server state; retain the
# GDS preference as a shell-only value until the actual Check 5 MPI launch.
for aim344_pmi_var in $(compgen -e PMI_; compgen -e PMIX_); do
    if [[ $aim344_pmi_var == PMIX_MCA_gds ]]; then
        export -n PMIX_MCA_gds
    else
        unset "$aim344_pmi_var"
    fi
done
unset aim344_pmi_var SLURM_MPI_TYPE

# Check 5 has a separate, job-local cold-start budget. Pyxis names are scoped by
# UID and job, never borrowed from another participant or an earlier allocation.
# Keep SPANK options in this subshell: host counters/storage must not inherit them.
run_check5() (
    local suite=$1 out=$2
    export NCCL_CONTAINER=${NCCL_CONTAINER:-$NCCL_ENROOT_IMAGE}
    export NCCL_TESTS_BIN=${NCCL_TESTS_BIN:-/opt/nccl-tests/build/all_reduce_perf}
    export NCCL_MPI=${NCCL_MPI:-pmix} NCCL_TIMEOUT=${NCCL_TIMEOUT:-180}
    export NCCL_ISOLATION_TESTS=${NCCL_ISOLATION_TESTS:-1}
    export NCCL_ISOLATION_TIMEOUT=${NCCL_ISOLATION_TIMEOUT:-120}
    export OMPI_MCA_pml=${OMPI_MCA_pml:-ob1} OMPI_MCA_btl=${OMPI_MCA_btl:-tcp,self}
    export OMPI_MCA_btl_tcp_if_exclude=${OMPI_MCA_btl_tcp_if_exclude:-lo,docker0,veth_def_agent}
    export OFI_NCCL_PROTOCOL=${OFI_NCCL_PROTOCOL:-RDMA}
    export FI_EFA_USE_HUGE_PAGE=${FI_EFA_USE_HUGE_PAGE:-0} FI_PROVIDER=${FI_PROVIDER:-efa}
    export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
    unset OMPI_MCA_btl_tcp_if_include NCCL_TESTS_SPLIT NCCL_TESTS_SPLIT_MASK
    unset NCCL_NET NCCL_NET_PLUGIN NCCL_IB_DISABLE FI_EFA_IFACE FI_EFA_DEVICE_NAME
    export SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_name="aim344_check5_${SLURM_JOB_ID:?}"
    export SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_env=OMPI_MCA_pml,OMPI_MCA_btl,OMPI_MCA_btl_tcp_if_include,OMPI_MCA_btl_tcp_if_exclude,NCCL_SOCKET_IFNAME,FI_PROVIDER,FI_EFA_USE_HUGE_PAGE,OFI_NCCL_PROTOCOL,PMIX_MCA_gds
    # Initialize both roots before NCCL's bounded collective/isolation timers.
    # This timeout is configurable, not a claim that cold extraction was qualified.
    local start=$SECONDS rc=0
    SLURM_MPI_TYPE=none timeout --signal=TERM --kill-after=30s "${AIM344_CONTAINER_PREP_TIMEOUT:-600}s" \
        srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --mpi=none --cpu-bind=none \
        --container-image="$NCCL_CONTAINER" --container-writable true || rc=$?
    printf 'Check 5 container preparation: elapsed=%s s exit_code=%s\n' "$((SECONDS-start))" "$rc"
    ((rc == 0)) || return "$rc"
    # Match the suite's --mpi option and enable the real step's PMIx IPC hook.
    # Keep both settings inside run_check5; later storage is not an MPI step.
    export SLURM_MPI_TYPE=$NCCL_MPI PMIX_MCA_gds=${PMIX_MCA_gds:-hash}
    bash "$suite" --check 5 --verbose --timeout "${AIM344_CHECK5_TIMEOUT:-360}" --results-dir "$out"
)

# shellcheck source=pins.env
source "$LAB_DIR/pins.env"

set_allocation_cpus() {
    local layout=${SLURM_JOB_CPUS_PER_NODE:-${SLURM_CPUS_ON_NODE:-}}
    [[ $layout =~ ^([1-9][0-9]*)(\(x[1-9][0-9]*\))?$ ]] || {
        echo "Expected a homogeneous Slurm CPU allocation; got: $layout" >&2; return 1;
    }
    export AIM344_CPUS_PER_NODE=${BASH_REMATCH[1]}
}

prepare_slurm() {
    set_allocation_cpus
    : "${SLURM_JOB_ID:?Run inside the dedicated two-node salloc allocation in README.md}"
    [[ ${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-0}} == 2 ]] || {
        echo 'This exercise requires exactly 2 allocated nodes.' >&2; exit 1;
    }
    local counts
    counts=$(srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task="${AIM344_CPUS_PER_NODE:?}" --mpi=none --cpu-bind=none \
        bash -c 'if [[ -n ${SLURM_GPUS_ON_NODE:-} ]]; then printf "%s\n" "$SLURM_GPUS_ON_NODE"; else nvidia-smi -L | awk '\''/^GPU [0-9]+:/ {n++} END {print n+0}'\''; fi')
    mapfile -t gpu_counts <<< "$counts"
    [[ ${#gpu_counts[@]} == 2 && ${gpu_counts[0]} =~ ^[1-9][0-9]*$ && ${gpu_counts[0]} == "${gpu_counts[1]}" ]] || {
        echo "Expected the same positive allocated GPU count on both nodes; got: $counts" >&2; exit 1;
    }
    export GPUS_PER_NODE=${gpu_counts[0]}
    printf 'allocation: 2 nodes, %s GPUs per node, %s GPU ranks\n' "$GPUS_PER_NODE" "$((2 * GPUS_PER_NODE))"
    [[ $LAB_DIR != *[:,]* && $LAB_DIR != *' '* ]] || {
        echo 'Use a shared lab path without spaces, commas, or colons.' >&2; exit 1;
    }
    RESULTS_DIR=${RESULTS_DIR:-$LAB_DIR/results/$SLURM_JOB_ID}
    mkdir -p -- "$RESULTS_DIR"
    srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --mpi=none --cpu-bind=none mkdir -p -- "$RESULTS_DIR"
    RESULTS_DIR=$(cd -- "$RESULTS_DIR" && pwd)
    [[ $RESULTS_DIR != *[:,]* && $RESULTS_DIR != *' '* ]] || exit 1
    CONTAINER_NAME=aim344_$SLURM_JOB_ID
    CONTAINER_ARGS=(--container-image="$NCCL_ENROOT_IMAGE"
        --container-name="$CONTAINER_NAME" --container-writable
        --container-env=FI_EFA_IFACE,AIM344_EFA_IFACE,AIM344_SOCKET_IFNAME --no-container-mount-home
        --container-mounts="$LAB_DIR:/opt/aim344:ro,$RESULTS_DIR:/results")
    # This initializes one private writable root filesystem on each node.
    node_command true
}

node_command() {
    SLURM_MPI_TYPE=none srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task="${AIM344_CPUS_PER_NODE:?}" --mpi=none --cpu-bind=none \
        "${CONTAINER_ARGS[@]}" --container-remap-root "$@"
}

run_sweep() {
    local phase=$1 run_id rc=0
    run_id=$phase-$(date -u +%Y%m%dT%H%M%SZ)
    node_command bash /opt/aim344/inventory.sh | tee "$RESULTS_DIR/$run_id.inventory.log"
    node_command python3 /opt/aim344/efa-counters.py snapshot "/results/$run_id.before"
    local start=$SECONDS
    # The optional sweep uses one MPI rank per GPU within the same CPU budget.
    SLURM_MPI_TYPE=pmix timeout --signal=TERM --kill-after=30s 600s \
        srun --nodes=2 --ntasks="$((2 * GPUS_PER_NODE))" --ntasks-per-node="$GPUS_PER_NODE" --cpus-per-task="$((AIM344_CPUS_PER_NODE / GPUS_PER_NODE))" --mpi=pmix --cpu-bind=none \
        "${CONTAINER_ARGS[@]}" --no-container-remap-root bash /opt/aim344/sweep-rank.sh \
        2>&1 | tee "$RESULTS_DIR/$run_id.nccl.log" || rc=$?
    printf 'sweep_elapsed=%s s; launcher_exit_code=%s (dimensionless)\n' "$((SECONDS-start))" "$rc" \
        | tee "$RESULTS_DIR/$run_id.status.log"
    node_command python3 /opt/aim344/efa-counters.py delta "/results/$run_id.before" \
        | tee "$RESULTS_DIR/$run_id.counters.log"
    ((rc == 0)) || return "$rc"
    python3 "$LAB_DIR/summarize.py" "$RESULTS_DIR/$run_id.nccl.log"
}

prepare_torch() {
    : "${TORCH_IMAGE:?Set TORCH_IMAGE to the absolute path of the built aim344.sqsh image}"
    : "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to the isolated per-attendee quota path or private mount}"
    [[ $TORCH_IMAGE == /* && $CHECKPOINT_DIR == /* && $CHECKPOINT_DIR != *[:,]* && $CHECKPOINT_DIR != *' '* ]] || exit 1
    CONTAINER_ARGS=(--container-image="$TORCH_IMAGE"
        --container-name="aim344_torch_$SLURM_JOB_ID" --container-writable
        --container-env=FI_EFA_IFACE,AIM344_EFA_IFACE,AIM344_SOCKET_IFNAME --no-container-mount-home
        --container-mounts="$LAB_DIR:/opt/aim344:ro,$RESULTS_DIR:/results,$CHECKPOINT_DIR:/checkpoints")
    local first_node
    first_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    MASTER_ADDR=$(scontrol show node "$first_node" -o | tr ' ' '\n' | sed -n 's/^NodeAddr=//p')
    [[ -n $MASTER_ADDR ]] || { echo 'Slurm did not return the first node address' >&2; exit 1; }
    export MASTER_ADDR
}

run_torch() {
    local phase=$1 run_id rc=0
    shift
    prepare_torch
    run_id=$phase-$(date -u +%Y%m%dT%H%M%SZ)
    # Device faults can leave GPU cleanup in D-state beyond the watchdog.
    # Keep tee attached through Slurm's bounded allocation walltime, rather
    # than SIGKILL the srun client and close evidence at 210 seconds.
    local -a deadline=(timeout --signal=TERM --kill-after=30s 180s)
    local -a evidence=()
    if [[ $phase == device ]]; then
        deadline=()
        # Existing workload output, also at a fixed job-bound coordinator path.
        # Run this launcher as the participant on the unaffected coordinator.
        mkdir -p -- "$HOME/aim344-results"
        evidence=("$HOME/aim344-results/device-$SLURM_JOB_ID.log")
    fi
    SLURM_MPI_TYPE=none "${deadline[@]}" \
        srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task="${AIM344_CPUS_PER_NODE:?}" --mpi=none --cpu-bind=none \
        "${CONTAINER_ARGS[@]}" bash /opt/aim344/torch-node.sh "$@" \
        2>&1 | tee "$RESULTS_DIR/$run_id.log" "${evidence[@]}" || rc=$?
    printf 'launcher_exit_code=%s (dimensionless)\n' "$rc" | tee "$RESULTS_DIR/$run_id.status.log"
    return "$rc"
}
