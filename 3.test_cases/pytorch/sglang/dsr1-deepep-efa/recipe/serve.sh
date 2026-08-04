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
#
# To benchmark UCCL-EP instead of DeepEP, point IMAGE_URI at an SGLang image built
# with UCCL's ep/deep_ep_wrapper (which presents UCCL under DeepEP's Python API, so
# --moe-a2a-backend deepep drives it unchanged) and set UCCL=1. That image is
# ../Dockerfile.uccl. UCCL needs --privileged, a pinned --deepep-mode, and
# a lower --mem-fraction-static; all three are applied below when UCCL=1.
#
# --disable-radix-cache is passed unconditionally, and it is not a tuning choice.
# recipe/benchmark.sh pins --seed 42, so every point that shares an input length
# replays the same prompts: with the cache on, the conc-1 point populates it and
# the conc-8/32 points that follow read their prefixes back instead of prefilling.
# Measured on one live server, 4096-in conc-8: 27.2k input tok/s with the cache
# flushed vs 334k on an immediate re-run of the identical point -- a 12x
# artefact, and the hit rate is not stable run to run, so it does not even cancel
# out across a comparison. Nothing in the sweep output flags it. If you want to
# measure cache-hit behaviour, that is a separate experiment with varied prompts.

set -euo pipefail

NODE_RANK=${1:?need NODE_RANK (0 or 1)}

: "${IMAGE_URI:?source setup/env_vars first}"
: "${NODE_0_IP:?source setup/env_vars first}"
: "${MODEL:?source setup/env_vars first}"
: "${HF_CACHE_DIR:?source setup/env_vars first}"
: "${IFACE:?source setup/env_vars first}"
MOE_BACKEND=${MOE_BACKEND:-deepep}
UCCL=${UCCL:-0}
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
        #   low_latency = decode-optimised; pin it when benchmarking decode
        #   auto        = SGLang picks per batch (default; what a deployment runs)
        #
        # low_latency on a COLOCATED server needs four coupled settings, which the
        # block below applies from LL_MAX_TOKENS (default 512). The low-latency
        # dispatch caps tokens per rank per call, and here the same ranks also serve
        # prefill, whose chunks are far larger:
        #   1. SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK = the cap. Must be
        #      <= 1024; DeepEP's internode low-latency dispatch uses
        #      FINISHED_SUM_TAG=1024 and needs the per-rank-pair count below it.
        #   2. --chunked-prefill-size = the cap, so prefill chunks fit it.
        #   3. NVSHMEM_QP_DEPTH >= (cap + 1) * 2 (deep_ep/buffer.py:601). Its
        #      default 1024 only covers a cap of 511.
        #   4. --mem-fraction-static 0.75: the low-latency RDMA buffers scale with
        #      the cap, 1.75 GiB per allocation at cap 1024.
        # All four fail on the FIRST REQUEST, after /health already returns 200, so
        # a green health check proves nothing. 512/0.75 is what fits 141GB H200;
        # override via LL_MAX_TOKENS / MEM_FRACTION_STATIC if you have headroom.
        #
        # Setting 2 changes the workload (cap-sized prefill chunks instead of 8192),
        # so DEEPEP_MODE=low_latency measures DECODE ONLY -- prefill numbers taken
        # under it are not comparable to anything. serve-pd.sh needs none of this:
        # there the decode role never prefills.
        #
        # UCCL cannot run auto at all: its prefill path segfaults at startup. So
        # DEEPEP_MODE is required, not defaulted, when UCCL=1 -- either default
        # would be wrong for one of the two stages.
        if [[ "$UCCL" == "1" && -z "${DEEPEP_MODE:-}" ]]; then
            echo "ERROR: UCCL=1 requires DEEPEP_MODE=normal (prefill) or low_latency (decode)." >&2
            echo "       --deepep-mode auto segfaults UCCL's prefill path at startup." >&2
            exit 1
        fi
        A2A=(--moe-a2a-backend deepep --deepep-mode "${DEEPEP_MODE:-auto}")
        if [[ "${DEEPEP_MODE:-auto}" == "low_latency" ]]; then
            LL_MAX_TOKENS=${LL_MAX_TOKENS:-512}
            MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.75}
            A2A+=(--chunked-prefill-size "$LL_MAX_TOKENS")
            LL_ENV=(-e "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=$LL_MAX_TOKENS")
            # Setting 3 is NVSHMEM's, so it is DeepEP-only. Keeping settings 1-2
            # for UCCL as well is deliberate: the two backends then serve the
            # same workload and their numbers stay comparable.
            if [[ "$UCCL" != "1" ]]; then
                LL_ENV+=(-e "NVSHMEM_QP_DEPTH=$(( (LL_MAX_TOKENS + 1) * 2 ))")
            fi
        fi
        NAME=r1-deepep ;;
    baseline)
        PARALLEL=(--tp-size "$TP_SIZE" --ep-size "$EP_SIZE")
        A2A=(--moe-a2a-backend none)
        NAME=r1-baseline ;;
    *)
        echo "ERROR: MOE_BACKEND must be deepep, baseline or tp" >&2; exit 1 ;;
esac

# Everything except low_latency (which set 0.75 above) gets the usual 0.85. Both
# backends get the same value here on purpose, so their tables stay comparable;
# 0.75 has been verified to hold UCCL's larger low_latency RDMA buffers too.
# (serve-pd.sh drops the UCCL decode role to 0.70 -- there the low_latency cap
# chain does not apply, so nothing else lowers it.)
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
if [[ "$UCCL" != "1" && "$MOE_BACKEND" == "deepep" && "${DEEPEP_MODE:-auto}" == "normal" ]]; then
    VMM_ENV=(-e NVSHMEM_DISABLE_CUDA_VMM=1)
fi

# NVSHMEM is DeepEP's transport only. UCCL talks to EFA directly and its image
# does not contain NVSHMEM at all, so the LD_PRELOAD would name a missing file;
# baseline/tp never load it either. Set it only where it is used.
NVSHMEM_ENV=()
if [[ "$UCCL" == "1" ]]; then
    NVSHMEM_ENV=(-e MOONCAKE_PROTOCOL=efa -e MC_FORCE_AUTO_DISCOVERY=1)
elif [[ "$MOE_BACKEND" == "deepep" ]]; then
    NVSHMEM_ENV=(
        -e NVSHMEM_REMOTE_TRANSPORT=libfabric
        -e NVSHMEM_LIBFABRIC_PROVIDER=efa
        -e NVSHMEM_NETDEVS_POLICY=EXTERNAL_SHARING_PCIE_SWITCH_NIC_EXCLUSIVE
        -e NVSHMEM_BOOTSTRAP=UID
        -e LD_PRELOAD=/opt/nvshmem/install/lib/libnvshmem_host.so.3.7.0
    )
fi

docker rm -f "$NAME" 2>/dev/null || true

# Exported so the `-e HF_TOKEN` below can forward it by NAME. Passing
# -e HF_TOKEN="$HF_TOKEN" instead would expand the secret into the xtrace output
# that `set -x` writes to the terminal, and from there into any CI log, `script`
# capture or shared tmux pane. The `:-` keeps the previous default-empty semantics.
export HF_TOKEN="${HF_TOKEN:-}"

set -x
# --privileged for every backend, not just UCCL. UCCL needs it to register GPU
# memory for RDMA through the dma-buf/ibverbs path, and Mooncake needs it to query
# the NUMA node of a registered buffer -- under Docker's default capability set
# that query fails with "Operation not permitted", registration silently proceeds
# anyway, and the first KV transfer then dies with "Memory region not registered by
# any active EFA device(s)". See recipe/serve-pd.sh.
docker run -d --name "$NAME" \
    --gpus all --network host --ipc host --privileged \
    --device /dev/infiniband --device /dev/gdrdrv \
    --ulimit memlock=-1 --shm-size 32g \
    -v "${HF_CACHE_DIR}:/hf" -e HF_HOME=/hf \
    -e HF_TOKEN \
    -v "${HF_CACHE_DIR}/../dg-cache:/root/.cache/deep_gemm" \
    -v "${HF_CACHE_DIR}/../sgl-cache:/root/.cache/sglang" \
    -e NCCL_SOCKET_IFNAME="$IFACE" -e GLOO_SOCKET_IFNAME="$IFACE" \
    -e NCCL_NET_PLUGIN=ofi \
    -e FI_PROVIDER=efa -e FI_EFA_USE_DEVICE_RDMA=1 \
    "${NVSHMEM_ENV[@]}" "${VMM_ENV[@]}" "${LL_ENV[@]}" \
    --entrypoint bash "$IMAGE_URI" -c "
        python3 -m sglang.launch_server \
            --model-path ${MODEL} --trust-remote-code \
            ${PARALLEL[*]} ${A2A[*]} \
            --nnodes ${NUM_NODES} --node-rank ${NODE_RANK} \
            --dist-init-addr ${NODE_0_IP}:${DIST_PORT} \
            --host 0.0.0.0 --port ${SERVE_PORT} \
            --disable-radix-cache \
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
