#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Sourced by numbered entry points; not an attendee command.
# shellcheck source=pins.env
source "$LAB_DIR/pins.env"

prepare_slurm() {
    : "${SLURM_JOB_ID:?Run inside the dedicated two-node salloc allocation in README.md}"
    [[ ${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-0}} == 2 ]] || {
        echo 'This exercise requires exactly 2 allocated nodes.' >&2; exit 1;
    }
    local counts
    counts=$(srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --mpi=none --cpu-bind=none \
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
    RESULTS_DIR=$(cd -- "$RESULTS_DIR" && pwd)
    [[ $RESULTS_DIR != *[:,]* && $RESULTS_DIR != *' '* ]] || exit 1
    CONTAINER_NAME=aim344_$SLURM_JOB_ID
    CONTAINER_ARGS=(--container-image="$NCCL_ENROOT_IMAGE"
        --container-name="$CONTAINER_NAME" --container-writable
        --no-container-mount-home
        --container-mounts="$LAB_DIR:/opt/aim344:ro,$RESULTS_DIR:/results")
    # This initializes one private writable root filesystem on each node.
    node_command true
}

node_command() {
    srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --mpi=none --cpu-bind=none \
        "${CONTAINER_ARGS[@]}" --container-remap-root "$@"
}

run_sweep() {
    local phase=$1 run_id rc=0
    run_id=$phase-$(date -u +%Y%m%dT%H%M%SZ)
    node_command bash /opt/aim344/inventory.sh | tee "$RESULTS_DIR/$run_id.inventory.log"
    node_command python3 /opt/aim344/efa-counters.py snapshot "/results/$run_id.before"
    local start=$SECONDS
    # The absolute binary path and registry separator replace stock check 5.
    timeout --signal=TERM --kill-after=30s 600s \
        srun --nodes=2 --ntasks="$((2 * GPUS_PER_NODE))" --ntasks-per-node="$GPUS_PER_NODE" --mpi=pmix --cpu-bind=none \
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
        --no-container-mount-home
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
    timeout --signal=TERM --kill-after=30s 180s \
        srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --mpi=none --cpu-bind=none \
        "${CONTAINER_ARGS[@]}" bash /opt/aim344/torch-node.sh "$@" \
        2>&1 | tee "$RESULTS_DIR/$run_id.log" || rc=$?
    printf 'launcher_exit_code=%s (dimensionless)\n' "$rc" | tee "$RESULTS_DIR/$run_id.status.log"
    return "$rc"
}
