#!/usr/bin/env bash
# Hold the session's two GPU nodes and measure their GEMM and memory-read ceilings.
set -euo pipefail
lab=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
[[ $# == 0 ]] || { echo 'Usage: bash tools/hold-and-measure.sh' >&2; exit 2; }
# shellcheck source=tools/common.sh
source "$lab/tools/common.sh"
workshop=(python3 "$lab/lib/workshop.py")
state=$HOME/.aim347
mkdir -p "$state"

job_id=$( (squeue --me -h --name=aim347-session -o '%i %T' || true) | awk '$2 == "RUNNING" {print $1; exit}')
if [[ -n $job_id ]]; then
    echo "Reusing session allocation $job_id"
else
    # --no-shell: the allocation outlives this shell and every later one, until release.sh.
    # --time=UNLIMITED: the participant owns both nodes, so the allocation has no time limit. Without
    # --time, Slurm would apply the partition's DefaultTime, which a PCS queue can set to a finite
    # value. The PCS partitions recorded so far have no DefaultTime and MaxTime=UNLIMITED.
    if ! granted=$(salloc --no-shell --job-name=aim347-session --partition="$PARTITION" --nodes=2 \
            --ntasks-per-node=1 --cpus-per-task="$AIM347_CPUS_PER_NODE" --exclusive --gres=gpu:"$GPUS_PER_NODE" \
            --time=UNLIMITED --immediate=120 2>&1) || ! job_id=$("${workshop[@]}" granted <<<"$granted"); then
        printf '%s\n' "$granted"
        echo "The allocation was not granted. The partition $PARTITION shows:"
        squeue -p "$PARTITION"
        exit 3
    fi
fi
"${workshop[@]}" write-allocation --job-id "$job_id" --cpus-per-node "$AIM347_CPUS_PER_NODE" \
    --output "$state/allocation.env"
# shellcheck disable=SC1091
source "$state/allocation.env"

ceilings=$LLM_RESULTS_DIR/ceilings
mkdir -p "$ceilings"
rm -f "$ceilings"/gemm-*.json "$ceilings"/dram-*.json "$ceilings"/ceilings.json
echo "Measuring the GPU ceilings on all $((2 * GPUS_PER_NODE)) GPUs..."
# shellcheck disable=SC2016
measure='set -u
out=$1 instance=$2 gpus=$3 host=${SLURMD_NODENAME:-$(hostname)} status=0 pids=()
for ((g = 0; g < gpus; g++)); do
    (CUDA_VISIBLE_DEVICES=$g INSTANCE_TYPE=$instance aim347-gemm 8192 30 > "$out/gemm-$host-gpu$g.json.tmp" &&
     CUDA_VISIBLE_DEVICES=$g INSTANCE_TYPE=$instance aim347-dram 0 > "$out/dram-$host-gpu$g.json.tmp" &&
     mv "$out/gemm-$host-gpu$g.json.tmp" "$out/gemm-$host-gpu$g.json" &&
     mv "$out/dram-$host-gpu$g.json.tmp" "$out/dram-$host-gpu$g.json") &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"'
# The named container stays unpacked for the session's later runs.
if ! srun --nodes=2 --ntasks-per-node=1 --cpus-per-task="$AIM347_CPUS_PER_NODE" \
        --container-image="$LAB_IMAGE" --container-name="$LLM_CONTAINER_NAME" \
        --container-mounts="$LLM_RESULTS_DIR:$LLM_RESULTS_DIR" --no-container-remap-root \
        bash -c "$measure" bash "$ceilings" "$AIM347_INSTANCE_TYPE" "$GPUS_PER_NODE" \
    || ! "${workshop[@]}" ceilings --dir "$ceilings" --gpus "$((2 * GPUS_PER_NODE))"; then
    echo 'The measurement failed. The allocation is still held; run bash tools/hold-and-measure.sh again to measure again.'
    exit 4
fi
