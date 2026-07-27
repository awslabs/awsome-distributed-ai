#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Serve DeepSeek-R1 across 2 nodes with SGLang, selecting the MoE all-to-all
# backend so DeepEP-on-EFA, the ordinary all-to-all, and pure TP can be compared
# on identical hardware and fabric.
#
# Run on BOTH nodes with the same MOE_BACKEND, varying NODE_RANK.
# The model must already be in $HF_CACHE_DIR on both nodes.
#
# Usage:
#   source setup/env_vars
#   recipe/serve.sh <NODE_RANK>            # backend from $MOE_BACKEND
#   MOE_BACKEND=baseline recipe/serve.sh 0 # or override per invocation
#
# Backends:
#   deepep   = DeepEP-on-EFA dispatch/combine (--moe-a2a-backend deepep)
#   baseline = SGLang's ordinary fused-MoE all-to-all over NCCL, still EP
#   tp       = no expert parallelism at all; MoE weights TP-sharded, the only
#              cross-GPU traffic is the TP all-reduce (no dispatch/combine)

set -euo pipefail

NODE_RANK=${1:?need NODE_RANK (0 or 1)}

: "${IMAGE_URI:?source setup/env_vars first}"
: "${NODE_0_IP:?source setup/env_vars first}"
: "${MODEL:?source setup/env_vars first}"
: "${HF_CACHE_DIR:?source setup/env_vars first}"
: "${IFACE:?source setup/env_vars first}"
MOE_BACKEND=${MOE_BACKEND:-deepep}
NUM_NODES=${NUM_NODES:-2}
TP_SIZE=${TP_SIZE:-16}
EP_SIZE=${EP_SIZE:-16}
SERVE_PORT=${SERVE_PORT:-30000}
DIST_PORT=${DIST_PORT:-29500}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.85}

case "$MOE_BACKEND" in
    tp)
        # No --ep-size => no expert parallelism.
        PARALLEL=(--tp-size "$TP_SIZE")
        A2A=(--moe-a2a-backend none)
        NAME=r1-tp ;;
    deepep)
        PARALLEL=(--tp-size "$TP_SIZE" --ep-size "$EP_SIZE")
        # "auto" lets SGLang pick normal kernels for prefill-heavy batches and
        # low-latency kernels for decode.
        A2A=(--moe-a2a-backend deepep --deepep-mode auto)
        NAME=r1-deepep ;;
    baseline)
        PARALLEL=(--tp-size "$TP_SIZE" --ep-size "$EP_SIZE")
        A2A=(--moe-a2a-backend none)
        NAME=r1-baseline ;;
    *)
        echo "ERROR: MOE_BACKEND must be deepep, baseline or tp" >&2; exit 1 ;;
esac

mkdir -p "${HF_CACHE_DIR}" "${HF_CACHE_DIR}/../dg-cache" "${HF_CACHE_DIR}/../sgl-cache"

# CUDA VMM is left ENABLED here (unlike the normal-kernel path in
# run-kernel-test.sh): with --deepep-mode auto the server uses the low-latency
# kernels for decode, and those require VMM. See README.
docker rm -f "$NAME" 2>/dev/null || true
set -x
docker run -d --name "$NAME" \
    --gpus all --network host --ipc host \
    --device /dev/infiniband --device /dev/gdrdrv \
    --ulimit memlock=-1 --shm-size 32g \
    -v "${HF_CACHE_DIR}:/hf" -e HF_HOME=/hf \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -v "${HF_CACHE_DIR}/../dg-cache:/root/.cache/deep_gemm" \
    -v "${HF_CACHE_DIR}/../sgl-cache:/root/.cache/sglang" \
    -e NCCL_SOCKET_IFNAME="$IFACE" -e GLOO_SOCKET_IFNAME="$IFACE" \
    -e NCCL_NET_PLUGIN=ofi \
    -e FI_PROVIDER=efa -e FI_EFA_USE_DEVICE_RDMA=1 \
    -e NVSHMEM_REMOTE_TRANSPORT=libfabric \
    -e NVSHMEM_LIBFABRIC_PROVIDER=efa \
    -e NVSHMEM_NETDEVS_POLICY=EXTERNAL_SHARING_PCIE_SWITCH_NIC_EXCLUSIVE \
    -e NVSHMEM_BOOTSTRAP=UID \
    -e LD_PRELOAD=/opt/nvshmem/install/lib/libnvshmem_host.so.3.7.0 \
    --entrypoint bash "$IMAGE_URI" -c "
        python3 -m sglang.launch_server \
            --model-path ${MODEL} --trust-remote-code \
            ${PARALLEL[*]} ${A2A[*]} \
            --nnodes ${NUM_NODES} --node-rank ${NODE_RANK} \
            --dist-init-addr ${NODE_0_IP}:${DIST_PORT} \
            --host 0.0.0.0 --port ${SERVE_PORT} \
            --mem-fraction-static ${MEM_FRACTION_STATIC} 2>&1
    "
set +x

echo
echo "Launched ${NAME} (backend=${MOE_BACKEND} rank=${NODE_RANK})."
echo "Follow startup:  docker logs -f ${NAME}"
echo "R1 takes several minutes to load; wait for 'The server is fired up'."
if [[ "$NODE_RANK" == "0" ]]; then
    echo "Then health-check:  curl -s localhost:${SERVE_PORT}/health && echo OK"
fi
