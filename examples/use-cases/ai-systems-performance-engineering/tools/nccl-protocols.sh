#!/usr/bin/env bash
# Optional: all_reduce_perf over EFA with OFI_NCCL_PROTOCOL=RDMA, 1 MiB to 64 MiB, on both
# nodes, once with NCCL's own protocol choice and once with NCCL_PROTO=Simple, then the two
# tables side by side. Usage: bash tools/nccl-protocols.sh   (see tools/README.md)
set -euo pipefail
lab=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
[[ $# == 0 ]] || { echo 'Usage: bash tools/nccl-protocols.sh' >&2; exit 2; }
# shellcheck source=tools/common.sh
source "$lab/tools/common.sh"
if [[ -z ${SLURM_JOB_ID:-} ]]; then
    # salloc holds both nodes for the two runs and frees them when this script ends.
    exec salloc --job-name=aim347-nccl "${two_nodes[@]}" bash "$lab/tools/nccl-protocols.sh"
fi
out=$LLM_RESULTS_DIR/nccl-protocols/$(date -u +%Y%m%dT%H%M%SZ)
ranks=$((2 * GPUS_PER_NODE))
echo "[aim347] all_reduce_perf, 1 MiB to 64 MiB, on $ranks GPUs over EFA with OFI_NCCL_PROTOCOL=RDMA"
echo "[aim347] results $out"
number=0
for run in nccl-choice simple; do
    number=$((number + 1))
    if [[ $run == simple ]]; then label='NCCL_PROTO=Simple'; else label='NCCL_PROTO not set'; fi
    mkdir -p "$out/$run"
    echo "[aim347] run $number of 2: $label"
    # One MPI rank per GPU on both nodes, in the training image.
    code=0
    srun --nodes=2 --ntasks="$ranks" --ntasks-per-node="$GPUS_PER_NODE" \
        --cpus-per-task=$((AIM347_CPUS_PER_NODE / GPUS_PER_NODE)) --mpi=pmix --cpu-bind=none --time=10 \
        --container-image="$LAB_IMAGE" --container-name="$LLM_CONTAINER_NAME" \
        --container-mounts="$lab:/opt/aim347,$LLM_RESULTS_DIR:$LLM_RESULTS_DIR" \
        --container-workdir=/opt/aim347 --no-container-remap-root --container-env=NCCL_SOCKET_IFNAME \
        bash lib/nccl-protocols-rank.sh "$run" "$out/$run" > "$out/$run/all_reduce_perf.txt" 2>&1 || code=$?
    if ((code)); then
        echo "[aim347] run $number ($label) failed with exit status $code; last 40 lines of $out/$run/all_reduce_perf.txt:"
        tail -n 40 "$out/$run/all_reduce_perf.txt"
        exit 4
    fi
done
python3 "$lab/lib/workshop.py" nccl-protocols-table "$out"
