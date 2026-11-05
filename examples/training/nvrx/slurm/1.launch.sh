#!/bin/bash
# Runs inside Pyxis once per node. NVRx owns the finite worker restart loop.
set -euo pipefail
export TMPDIR=${NVRX_LOCAL_TMPDIR:?}
((${#TMPDIR} <= 65)) || { printf 'Use a short node-local NVRX_LOCAL_TMPDIR (<=65 characters)\n' >&2; exit 1; }
mkdir -p "$TMPDIR"
# Only the launcher and monitor notifications differ in the torchrun control.
launcher=${NVRX_LAUNCHER:-ft_launcher}
ft_args=()
train_args=()
case "$launcher" in
    ft_launcher)
        ft_args=(--ft-rank-section-timeouts="$NVRX_SECTION_TIMEOUTS"
                 --ft-rank-out-of-section-timeout=60
                 --ft-checkpoint-iteration-file="$NVRX_RUN_DIR/latest_checkpointed_iteration.txt") ;;
    torchrun) train_args=(--no-nvrx) ;;
    *) printf 'NVRX_LAUNCHER must be ft_launcher or torchrun\n' >&2; exit 1 ;;
esac
printf '%s launcher=%s node=%s SLURM_PROCID=%s SLURM_NODEID=%s\n' \
    "$(date -u +%FT%TZ)" "$launcher" "$(hostname)" "$SLURM_PROCID" "$SLURM_NODEID"
# Logs live beside, not inside, the initially nonexistent training directory.
exec "$launcher" --rdzv-backend=c10d \
    --nnodes="$SLURM_JOB_NUM_NODES" --nproc-per-node="$NVRX_GPUS_PER_NODE" \
    --rdzv-id="$SLURM_JOB_ID" --rdzv-endpoint="$NVRX_RDZV_HOST:$NVRX_RDZV_PORT" \
    --rdzv-conf=join_timeout=120,store_connect_wait_seconds=60 \
    --max-restarts="$NVRX_MAX_RESTARTS" \
    "${ft_args[@]}" \
    --log-dir="${NVRX_RUN_DIR}-logs/$SLURM_JOB_ID/node-$SLURM_NODEID" --tee=3 \
    /app/src/train_gpt2.py --run-dir="$NVRX_RUN_DIR" "${train_args[@]}" "$@"
