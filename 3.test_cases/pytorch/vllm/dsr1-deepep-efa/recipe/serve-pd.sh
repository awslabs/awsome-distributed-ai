#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Bring up a 2P2D (2-node prefill + 2-node decode) DeepSeek-R1 deployment on vLLM,
# with the MoE all-to-all backend selectable so DeepEP-on-EFA and vLLM's ordinary
# all-to-all can be compared on identical hardware and fabric.
#
# WHY RAY: vLLM TP=16 across 2 nodes needs a Ray cluster per role. vLLM's own
# --nnodes/--node-rank (the sglang-style launch) fails with "collective_rpc should
# not be called on follower node". So each role is a 2-node Ray cluster, and
# `vllm serve --tensor-parallel-size 16 --distributed-executor-backend ray` runs
# ONCE on that role's head; Ray places the 16 workers across both nodes.
#
# Four stages, in order:
#
#   # 1. on EACH of the 4 nodes — start the long-lived Ray container:
#   recipe/serve-pd.sh raystart prefill head       # on PREFILL_HEAD_IP
#   recipe/serve-pd.sh raystart prefill worker     # on PREFILL_WORKER_IP
#   recipe/serve-pd.sh raystart decode  head       # on DECODE_HEAD_IP
#   recipe/serve-pd.sh raystart decode  worker     # on DECODE_WORKER_IP
#
#   # 2. on each ROLE HEAD only — launch the engine:
#   recipe/serve-pd.sh serve prefill               # on PREFILL_HEAD_IP
#   recipe/serve-pd.sh serve decode                # on DECODE_HEAD_IP
#
#   # 3. on ROUTER_HOST — front both roles:
#   recipe/serve-pd.sh router
#
#   # 4. teardown (run on every node; also required BETWEEN backends, see below):
#   recipe/serve-pd.sh stop prefill
#
# TEARING DOWN BETWEEN CONFIGS IS MANDATORY: restarting a vLLM engine by killing
# it inside a live Ray cluster leaks the GPU placement group, and the next
# `vllm serve` then cannot get 16 GPUs. Always `stop` both roles and redo
# `raystart` when switching MOE_BACKEND.

set -euo pipefail

STAGE=${1:?stage: raystart | serve | router | stop}

: "${IMAGE_URI:?source setup/env_vars first}"

MOE_BACKEND=${MOE_BACKEND:-deepep}
TP_SIZE=${TP_SIZE:-16}
RAY_PORT=${RAY_PORT:-6379}
ROUTER_PORT=${ROUTER_PORT:-8000}
NUM_GPU_PER_NODE=${NUM_GPU_PER_NODE:-8}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-deepseek-ai/DeepSeek-R1}

# Per-role settings. Prefill produces KV, decode consumes it.
role_config() {
    case "$1" in
        prefill)
            HEAD_IP=${PREFILL_HEAD_IP:?set PREFILL_HEAD_IP}
            PORT=${PREFILL_PORT:-8001}
            NIXL_PORT=${PREFILL_NIXL_PORT:-5557}
            KV_ROLE=kv_producer
            # DeepEP high-throughput (normal) kernels suit the large prefill batches.
            A2A_BACKEND=deepep_high_throughput
            # Prefill sees dynamic shapes, so CUDA graphs do not pay off.
            GRAPH_ARGS=(--enforce-eager)
            NAME=r1-vllm-prefill ;;
        decode)
            HEAD_IP=${DECODE_HEAD_IP:?set DECODE_HEAD_IP}
            PORT=${DECODE_PORT:-8002}
            NIXL_PORT=${DECODE_NIXL_PORT:-5558}
            KV_ROLE=kv_consumer
            A2A_BACKEND=deepep_low_latency
            # Decode shapes are fixed, so capture decode-only CUDA graphs.
            GRAPH_ARGS=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}')
            NAME=r1-vllm-decode ;;
        *) echo "ERROR: role must be prefill or decode" >&2; exit 1 ;;
    esac
}

case "$STAGE" in

raystart)
    ROLE=${2:?need role: prefill | decode}
    KIND=${3:?need head | worker}
    role_config "$ROLE"
    : "${IFACE:?source setup/env_vars first}"
    : "${MODEL_DIR:?source setup/env_vars first}"
    CACHE_DIR=${CACHE_DIR:-/opt/dlami/nvme/cache}
    mkdir -p "$CACHE_DIR"

    # CUDA VMM is left ENABLED (unlike the normal-kernel path in
    # run-kernel-test.sh): the decode role uses the DeepEP low-latency kernels,
    # which require it. See README.
    docker rm -f "$NAME" 2>/dev/null || true
    set -x
    docker run -d --name "$NAME" \
        --gpus all --network host --ipc host \
        --device /dev/infiniband --device /dev/gdrdrv \
        --ulimit memlock=-1 --shm-size 32g \
        -v "${MODEL_DIR}:/model:ro" \
        -v "${CACHE_DIR}:/root/.cache" \
        -e HF_TOKEN="${HF_TOKEN:-}" \
        -e NCCL_SOCKET_IFNAME="$IFACE" -e GLOO_SOCKET_IFNAME="$IFACE" \
        -e FI_PROVIDER=efa -e FI_EFA_USE_DEVICE_RDMA=1 \
        -e NVSHMEM_REMOTE_TRANSPORT=libfabric \
        -e NVSHMEM_LIBFABRIC_PROVIDER=efa \
        -e NVSHMEM_NETDEVS_POLICY=EXTERNAL_SHARING_PCIE_SWITCH_NIC_EXCLUSIVE \
        -e NVSHMEM_BOOTSTRAP=UID \
        -e LD_PRELOAD=/opt/nvshmem/install/lib/libnvshmem_host.so.3.7.0 \
        -e VLLM_NIXL_SIDE_CHANNEL_HOST="$HEAD_IP" \
        -e VLLM_NIXL_SIDE_CHANNEL_PORT="$NIXL_PORT" \
        -e VLLM_USE_FLASHINFER_MOE_FP8=1 \
        --entrypoint bash "$IMAGE_URI" -c 'sleep infinity'
    set +x
    sleep 8

    if [[ "$KIND" == "head" ]]; then
        docker exec "$NAME" ray start --head --port="$RAY_PORT" \
            --num-gpus="$NUM_GPU_PER_NODE" --node-ip-address="$HEAD_IP"
    else
        docker exec "$NAME" ray start --address="${HEAD_IP}:${RAY_PORT}" \
            --num-gpus="$NUM_GPU_PER_NODE"
    fi
    echo "Ray ${KIND} up for role=${ROLE} (container ${NAME})."
    echo "Check the cluster has 16 GPUs before serving:  docker exec ${NAME} ray status"
    ;;

serve)
    ROLE=${2:?need role: prefill | decode}
    role_config "$ROLE"

    case "$MOE_BACKEND" in
        deepep)
            A2A_ENV=(-e VLLM_ALL2ALL_BACKEND="$A2A_BACKEND" -e VLLM_USE_DEEP_GEMM=1) ;;
        baseline)
            A2A_ENV=() ;;
        *) echo "ERROR: MOE_BACKEND must be deepep or baseline" >&2; exit 1 ;;
    esac

    KV_CONFIG="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"${KV_ROLE}\",\"kv_load_failure_policy\":\"fail\"}"

    # The launch command is written to a file INSIDE the container and then run,
    # rather than passed through nested shells: the JSON in --kv-transfer-config
    # and --compilation-config otherwise loses its quotes and vLLM rejects it with
    # "Invalid JSON: key must be a string".
    #
    # `docker exec -e` sets the a2a env on this exec only, so switching MOE_BACKEND
    # does not require recreating the Ray container.
    docker exec "${A2A_ENV[@]}" -i "$NAME" bash -s <<LAUNCH
set -euo pipefail
cat > /run-vllm.sh <<'INNER'
#!/bin/bash
exec vllm serve /model \\
    --served-model-name ${SERVED_MODEL_NAME} \\
    --trust-remote-code \\
    --port ${PORT} \\
    --kv-transfer-config '${KV_CONFIG}' \\
    --tensor-parallel-size ${TP_SIZE} \\
    --enable-expert-parallel \\
    --distributed-executor-backend ray \\
    ${GRAPH_ARGS[*]}
INNER
chmod +x /run-vllm.sh
nohup /run-vllm.sh > /serve.log 2>&1 &
echo "serve launched, pid \$!"
LAUNCH

    echo
    echo "vllm serve (${ROLE}, MOE_BACKEND=${MOE_BACKEND}) launched on the Ray head."
    echo "Follow startup:  docker exec ${NAME} tail -f /serve.log"
    echo "R1 takes several minutes to load; wait for 'Application startup complete'."
    echo "Health-check:    curl -s localhost:${PORT}/health && echo OK"
    ;;

router)
    : "${PREFILL_HEAD_IP:?set PREFILL_HEAD_IP}"
    : "${DECODE_HEAD_IP:?set DECODE_HEAD_IP}"
    PREFILL_PORT=${PREFILL_PORT:-8001}
    DECODE_PORT=${DECODE_PORT:-8002}

    docker rm -f r1-vllm-router 2>/dev/null || true
    set -x
    docker run -d --name r1-vllm-router --network host \
        --entrypoint vllm-router "$IMAGE_URI" \
        --policy round_robin --vllm-pd-disaggregation \
        --prefill "http://${PREFILL_HEAD_IP}:${PREFILL_PORT}" \
        --decode  "http://${DECODE_HEAD_IP}:${DECODE_PORT}" \
        --host 0.0.0.0 --port "$ROUTER_PORT"
    set +x
    echo
    echo "Router on :${ROUTER_PORT}. Both engines must already be healthy."
    echo "Smoke test:"
    echo "  curl -s localhost:${ROUTER_PORT}/v1/completions -H 'Content-Type: application/json' \\"
    echo "    -d '{\"model\":\"${SERVED_MODEL_NAME}\",\"prompt\":\"Hello\",\"max_tokens\":16}'"
    ;;

stop)
    ROLE=${2:-}
    if [[ -n "$ROLE" ]]; then role_config "$ROLE"; NAMES=("$NAME"); else NAMES=(r1-vllm-prefill r1-vllm-decode); fi
    for n in "${NAMES[@]}" r1-vllm-router; do
        docker rm -f "$n" 2>/dev/null && echo "removed $n" || true
    done
    echo "Removing the container also tears down its Ray node, releasing the GPU"
    echo "placement group. Re-run 'raystart' before serving again."
    ;;

*)
    echo "ERROR: stage must be raystart, serve, router or stop" >&2; exit 1 ;;
esac
