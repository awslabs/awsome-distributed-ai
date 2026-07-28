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
LL_ENV=()
NUM_NODES=${NUM_NODES:-2}
TP_SIZE=${TP_SIZE:-16}
EP_SIZE=${EP_SIZE:-16}
SERVE_PORT=${SERVE_PORT:-30000}
DIST_PORT=${DIST_PORT:-29500}
# Left empty here on purpose: the default depends on DEEPEP_MODE and is filled
# in after the case block below.
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-}

case "$MOE_BACKEND" in
    tp)
        # No --ep-size => no expert parallelism.
        PARALLEL=(--tp-size "$TP_SIZE")
        A2A=(--moe-a2a-backend none)
        NAME=r1-tp ;;
    deepep)
        PARALLEL=(--tp-size "$TP_SIZE" --ep-size "$EP_SIZE")
        # DEEPEP_MODE selects the dispatch/combine kernels:
        #   normal      = high-throughput; pin it when benchmarking prefill
        #   low_latency = decode-optimised; pin it when benchmarking decode, but
        #                 read the token-cap note below first -- on a COLOCATED
        #                 server it needs three more settings or it crashes
        #   auto        = SGLang picks per batch (default; what a deployment runs)
        #
        # LOW_LATENCY TOKEN CAP. The low-latency dispatch caps tokens per rank per
        # call at SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK, default 128. In a
        # colocated server the same ranks also serve prefill, where a chunk is
        # chunked_prefill_size tokens (8192 by default) -- far over the cap. The
        # server starts, answers /health, then every rank dies on the first real
        # request with:
        #   deep_ep.cpp:1553 'x.size(0) <= num_max_dispatch_tokens_per_rank'
        # (measured on 2 x p5en, 2026-07-28: 8/8 ranks down on request one).
        #
        # So pinning low_latency here also requires FOUR settings, which this
        # script applies for you. LL_MAX_TOKENS is the cap (default 512):
        #   1. SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=$LL_MAX_TOKENS
        #      SGLang asserts <= 1024: DeepEP internode low-latency dispatch uses
        #      FINISHED_SUM_TAG=1024 and needs the per-rank-pair count below it.
        #   2. --chunked-prefill-size $LL_MAX_TOKENS, so prefill chunks fit the cap.
        #   3. NVSHMEM_QP_DEPTH >= (LL_MAX_TOKENS + 1) * 2, asserted in
        #      deep_ep/buffer.py:601. Its default is 1024, which only covers a cap
        #      of 511 -- so raising the cap without this fails the same way, on the
        #      first request, after /health has already gone green.
        #   4. A lower --mem-fraction-static (0.75 instead of 0.85). The
        #      low-latency RDMA buffers scale with the cap: at 1024 a single
        #      low_latency_dispatch allocation is 1.75 GiB and scheduler init dies
        #      with torch.OutOfMemoryError on 141GB H200 at 0.85.
        #
        # Constraints 1 and 3 squeeze from opposite sides and 4 prices the result,
        # so the default cap here is 512, not the 1024 maximum: 512 still covers a
        # concurrency-512 decode batch and leaves the buffers room. Override with
        # LL_MAX_TOKENS / MEM_FRACTION_STATIC if you have the headroom.
        #
        # Step 2 is a real workload change: cap-sized prefill chunks instead of
        # 8192. So DEEPEP_MODE=low_latency is for measuring DECODE ONLY; prefill
        # numbers taken under it are not comparable to anything.
        #
        # serve-pd.sh needs none of this: there the decode role never prefills.
        A2A=(--moe-a2a-backend deepep --deepep-mode "${DEEPEP_MODE:-auto}")
        if [[ "${DEEPEP_MODE:-auto}" == "low_latency" ]]; then
            LL_MAX_TOKENS=${LL_MAX_TOKENS:-512}
            MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.75}
            A2A+=(--chunked-prefill-size "$LL_MAX_TOKENS")
            LL_ENV=(-e "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=$LL_MAX_TOKENS"
                    -e "NVSHMEM_QP_DEPTH=$(( (LL_MAX_TOKENS + 1) * 2 ))")
        fi
        NAME=r1-deepep ;;
    baseline)
        PARALLEL=(--tp-size "$TP_SIZE" --ep-size "$EP_SIZE")
        A2A=(--moe-a2a-backend none)
        NAME=r1-baseline ;;
    *)
        echo "ERROR: MOE_BACKEND must be deepep, baseline or tp" >&2; exit 1 ;;
esac

# Everything except low_latency (which set 0.75 above) gets the usual 0.85.
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.85}

mkdir -p "${HF_CACHE_DIR}" "${HF_CACHE_DIR}/../dg-cache" "${HF_CACHE_DIR}/../sgl-cache"

# CUDA VMM has to follow the kernel regime, so it is derived from DEEPEP_MODE
# rather than hardcoded:
#   low_latency / auto -> VMM ENABLED. The low-latency kernels need it, or the
#                         RDMA-buffer cudaMemset fails ("invalid argument",
#                         deep_ep.cpp:371). "auto" can reach them at any time.
#   normal             -> NVSHMEM_DISABLE_CUDA_VMM=1, or NVSHMEM topology /
#                         transport-map init fails in-container.
# Getting this wrong is a startup failure, not a slow server. See README.
VMM_ENV=()
if [[ "$MOE_BACKEND" == "deepep" && "${DEEPEP_MODE:-auto}" == "normal" ]]; then
    VMM_ENV=(-e NVSHMEM_DISABLE_CUDA_VMM=1)
fi

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
    "${VMM_ENV[@]}" "${LL_ENV[@]}" \
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
