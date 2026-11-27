#!/usr/bin/env bash
# Measure the GEMM and memory-read ceilings of every GPU on both nodes.
set -euo pipefail
lab=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
[[ $# == 0 ]] || { echo 'Usage: bash tools/measure-ceilings.sh' >&2; exit 2; }
# shellcheck source=tools/common.sh
source "$lab/tools/common.sh"

ceilings=$LLM_RESULTS_DIR/ceilings
mkdir -p "$ceilings"
rm -f "$ceilings"/gemm-*.json "$ceilings"/dram-*.json "$ceilings"/ceilings.json
echo "Measuring the GPU ceilings on all $((2 * GPUS_PER_NODE)) GPUs..."
# shellcheck disable=SC2016
measure='set -u
out=$1 instance=$2 gpus=$3 host=${SLURMD_NODENAME:-$(hostname)} status=0 pids=()
for ((g = 0; g < gpus; g++)); do
    (CUDA_VISIBLE_DEVICES=$g INSTANCE_TYPE=$instance aim347-gemm 8192 30 > "$out/gemm-$host-gpu$g.json" &&
     CUDA_VISIBLE_DEVICES=$g INSTANCE_TYPE=$instance aim347-dram 0 > "$out/dram-$host-gpu$g.json") &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"'
srun "${two_nodes[@]}" --container-image="$LAB_IMAGE" \
    --container-mounts="$LLM_RESULTS_DIR:$LLM_RESULTS_DIR" --no-container-remap-root \
    bash -c "$measure" bash "$ceilings" "$AIM347_INSTANCE_TYPE" "$GPUS_PER_NODE"
python3 "$lab/lib/workshop.py" ceilings --dir "$ceilings" --gpus "$((2 * GPUS_PER_NODE))"
