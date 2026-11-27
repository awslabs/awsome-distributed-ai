#!/usr/bin/env bash
# Inside a Slurm allocation: one vLLM server per GPU on every node, as one srun step.
# The server of GPU g of a node listens on port 8100 + g and writes its log to
# $LLM_RESULTS_DIR/$SERVING_RUN/<node>-gpu<g>.log. The servers end with the step.
set -euo pipefail
: "${SLURM_JOB_NUM_NODES:?Run inside a Slurm allocation}" "${AIM347_CPUS_PER_NODE:?}" "${GPUS_PER_NODE:?}"
: "${VLLM_IMAGE:?}" "${LLM_DATA_DIR:?}" "${LLM_RESULTS_DIR:?}" "${SERVING_RUN:?}"
export SERVING_RUN GPUS_PER_NODE
# The container's shell expands the variables of the script below.
# shellcheck disable=SC2016
exec srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
    --cpus-per-task="$AIM347_CPUS_PER_NODE" --mpi=none --cpu-bind=none --kill-on-bad-exit=1 \
    --container-image="$VLLM_IMAGE" --container-mounts="$LLM_DATA_DIR:/data,$LLM_RESULTS_DIR:/results" \
    --no-container-remap-root --container-env=SERVING_RUN,GPUS_PER_NODE \
    bash -c '
    node=${SLURMD_NODENAME:-$(hostname)}
    for ((gpu = 0; gpu < GPUS_PER_NODE; gpu++)); do
        # --port is the API. VLLM_PORT is where a server starts looking for its internal ports.
        CUDA_VISIBLE_DEVICES=$gpu VLLM_PORT=$((20000 + 1000 * gpu)) vllm serve /data/model \
            --served-model-name aim347 --dtype bfloat16 --max-model-len 4096 --max-num-seqs 256 \
            --max-num-batched-tokens 8192 --gpu-memory-utilization 0.85 --no-enable-prefix-caching \
            --generation-config vllm --host 0.0.0.0 --port $((8100 + gpu)) \
            > "/results/$SERVING_RUN/$node-gpu$gpu.log" 2>&1 &
    done
    # A server that exits ends the step, and with it the other servers.
    wait -n
    exit 1'
