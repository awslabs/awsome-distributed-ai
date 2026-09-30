#!/usr/bin/env bash
set -euo pipefail
LAB_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export LAB_DIR

# shellcheck disable=SC1091
if [[ -f "$LAB_DIR/.env" ]]; then set -a; source "$LAB_DIR/.env"; set +a; fi
: "${RUN_ID:=aim347-$(date -u +%Y%m%dT%H%M%SZ)}"
: "${INSTANCE_TYPE:=unknown}"
: "${STEPS:=100}" "${WARMUP:=10}" "${MICROBATCH:=1}" "${CPU_ROUNDS:=2000}"
: "${PUSHGATEWAY_URL:=}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Use an alphanumeric RUN_ID with hyphens or underscores' >&2; exit 2; }
export RUN_ID INSTANCE_TYPE STEPS WARMUP MICROBATCH CPU_ROUNDS PUSHGATEWAY_URL
submit() {
    local action=$1
    : "${COMPUTE_NODES:?Set the assigned Slurm compute hostnames}"
    : "${PARTITION:?Set PARTITION in .env}" "${LAB_IMAGE:?Set LAB_IMAGE to the absolute squashfs path}"
    : "${DATA_DIR:?Set DATA_DIR to an absolute directory staged on both assigned nodes}"
    [[ "$LAB_DIR" != *[[:space:],:]* && "$DATA_DIR" != *[[:space:],:]* ]] || { echo 'Mount paths cannot contain whitespace, commas or colons' >&2; exit 2; }
    export PARTITION LAB_IMAGE DATA_DIR
    mkdir -p "$LAB_DIR/results/$RUN_ID"
    local budget=(--ntasks-per-node=1 --exclusive)
    if [[ -n ${LAB_GPUS_PER_NODE:-} ]]; then
        [[ $LAB_GPUS_PER_NODE =~ ^[1-9][0-9]*$ && ${LAB_CPUS_PER_TASK:-} =~ ^[1-9][0-9]*$ && ${LAB_MEMORY_PER_NODE:-} =~ ^(0|[1-9][0-9]*[GM])$ ]] || {
            echo 'Set positive LAB_GPUS_PER_NODE, LAB_CPUS_PER_TASK and LAB_MEMORY_PER_NODE (G or M; 0 selects all schedulable memory)' >&2; exit 2;
        }
        budget=(--gres="gpu:$LAB_GPUS_PER_NODE" --ntasks-per-node="$LAB_GPUS_PER_NODE"
                --cpus-per-task="$LAB_CPUS_PER_TASK" --mem="$LAB_MEMORY_PER_NODE")
    fi
    if [[ ${LAB_EXCLUSIVE:-0} == 1 ]]; then budget+=(--exclusive); fi
    sbatch --wait --parsable --partition="$PARTITION" --job-name="aim347-$action" \
      --nodelist="$COMPUTE_NODES" --nodes=2 "${budget[@]}" --time=00:20:00 \
      --output="$LAB_DIR/results/$RUN_ID/$action-%j.log" --export=ALL \
      "$LAB_DIR/lib/job.sh" "$action"
}
