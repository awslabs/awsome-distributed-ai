#!/usr/bin/env bash
set -euo pipefail
LAB_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export LAB_DIR

# shellcheck disable=SC1091
if [[ -f "$LAB_DIR/.env" ]]; then set -a; source "$LAB_DIR/.env"; set +a; fi
: "${RUN_ID:=aim347-$(date -u +%Y%m%dT%H%M%SZ)}"
: "${INSTANCE_TYPE:=g7e.12xlarge}"
: "${STEPS:=100}" "${WARMUP:=10}" "${MICROBATCH:=1}" "${CPU_ROUNDS:=2000}"
: "${PUSHGATEWAY_URL:=}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Use an alphanumeric RUN_ID with hyphens or underscores' >&2; exit 2; }
export RUN_ID INSTANCE_TYPE STEPS WARMUP MICROBATCH CPU_ROUNDS PUSHGATEWAY_URL
submit() {
    local action=$1
    : "${COMPUTE_NODES:?Set the assigned Slurm compute hostnames}"
    : "${PARTITION:?Set PARTITION in .env}" "${LAB_IMAGE:?Set LAB_IMAGE to the absolute squashfs path}"
    : "${DATA_DIR:?Set DATA_DIR to an absolute directory on same-AZ FSx}"
    [[ "$LAB_DIR" != *[[:space:],:]* && "$DATA_DIR" != *[[:space:],:]* ]] || { echo 'Mount paths cannot contain whitespace, commas or colons' >&2; exit 2; }
    export PARTITION LAB_IMAGE DATA_DIR
    mkdir -p "$LAB_DIR/results/$RUN_ID"
    sbatch --wait --parsable --partition="$PARTITION" --job-name="aim347-$action" \
      --nodelist="$COMPUTE_NODES" --nodes=2 --ntasks-per-node=1 --exclusive --time=00:20:00 \
      --output="$LAB_DIR/results/$RUN_ID/$action-%j.log" --export=ALL \
      "$LAB_DIR/lib/job.sh" "$action"
}
