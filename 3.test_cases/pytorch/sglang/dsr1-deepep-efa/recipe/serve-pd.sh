#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Serve DeepSeek-R1 PD-disaggregated (2 prefill nodes + 2 decode nodes) with
# SGLang, selecting the MoE all-to-all backend so DeepEP-on-EFA, the ordinary
# all-to-all, and pure TP can be compared on identical hardware and fabric.
#
# This is the 4-node PD counterpart of recipe/serve.sh (which is colocated on 2
# nodes). Prefill and decode are separate TP16/EP16 server groups; the KV cache
# moves prefill -> decode over EFA RDMA via Mooncake, and a router in front
# presents one OpenAI-compatible endpoint.
#
#   prefill rank 0,1  ──KV over EFA──▶  decode rank 0,1
#            └────────── sglang_router --pd-disaggregation ──────────┘
#
# Usage (source setup/env_vars first, PD block filled in):
#   recipe/serve-pd.sh precompile <RANK>          # on each prefill node; see below
#   recipe/serve-pd.sh serve prefill <RANK>       # on each prefill node
#   recipe/serve-pd.sh serve decode  <RANK>       # on each decode node
#   recipe/serve-pd.sh router                     # on the router host
#   recipe/serve-pd.sh stop                       # on every node
#
# Backends (MOE_BACKEND, same meaning as recipe/serve.sh):
#   deepep   = DeepEP-on-EFA dispatch/combine (--moe-a2a-backend deepep)
#   baseline = SGLang's ordinary fused-MoE all-to-all over NCCL, still EP
#   tp       = no expert parallelism at all; MoE weights TP-sharded
#
# DP_ATTENTION=1 additionally runs attention data-parallel across all 16 ranks.
# It is orthogonal to MOE_BACKEND and it reverses the decode ranking — see
# benchmarks/README.md. It also needs two things this script handles:
#   - `precompile` must be run first on BOTH prefill nodes for deepep, or
#     DeepEP's dispatch warmup times out while DeepGEMM JIT-compiles the
#     num_groups=16 grouped-GEMM shapes.
#   - the decode role must pin --cuda-graph-bs, or all 16 DP ranks capture the
#     full batch-size list and OOM ("scheduler died").
#
# To benchmark UCCL-EP instead of DeepEP, point IMAGE_URI at an SGLang image
# built with UCCL's ep/deep_ep_wrapper (which presents UCCL under DeepEP's Python
# API, so --moe-a2a-backend deepep drives it unchanged) and set UCCL=1. That
# image is ../Dockerfile.uccl. UCCL needs a pinned
# --deepep-mode, --privileged, and a lower decode --mem-fraction-static; all
# three are applied below when UCCL=1.

set -euo pipefail

CMD=${1:?need COMMAND: serve | precompile | router | stop}

: "${IMAGE_URI:?source setup/env_vars first}"
MOE_BACKEND=${MOE_BACKEND:-deepep}
DP_ATTENTION=${DP_ATTENTION:-0}
UCCL=${UCCL:-0}
TP_SIZE=${TP_SIZE:-16}
EP_SIZE=${EP_SIZE:-16}
DP_SIZE=${DP_SIZE:-16}
NUM_NODES=${NUM_NODES:-2}                       # nodes PER role
PREFILL_PORT=${PREFILL_PORT:-30000}
DECODE_PORT=${DECODE_PORT:-30001}
BOOTSTRAP_PORT=${BOOTSTRAP_PORT:-8998}
ROUTER_PORT=${ROUTER_PORT:-8000}
PREFILL_DIST_PORT=${PREFILL_DIST_PORT:-29500}
DECODE_DIST_PORT=${DECODE_DIST_PORT:-29501}

NAME_PREFIX=r1-pd

stop_all() {
    for n in "${NAME_PREFIX}-prefill" "${NAME_PREFIX}-decode" \
             "${NAME_PREFIX}-precompile" "${NAME_PREFIX}-router"; do
        docker rm -f "$n" 2>/dev/null || true
    done
    echo "Stopped any ${NAME_PREFIX}-* container on this host."
}

# --- router -----------------------------------------------------------------
# sglang_router in --pd-disaggregation mode. Each --prefill takes the server URL
# AND its bootstrap port (that is how the decode side learns where to pull KV
# from); each --decode takes only the URL.
#
# ONE URL per role, node rank 0 only -- NOT one per node. A TP16 group spanning
# two nodes runs the HTTP frontend on rank 0 alone; rank 1 is a worker with no
# API. Registering rank 1 as a second worker looks fine (its /health returns 200,
# because that only proves the process is alive) and then fails every request the
# round-robin policy sends to it with "Decode server returned error status
# status=404", which surfaces as "Warmup failed ... Error: Not Found" from
# bench_serving rather than as a routing error.
run_router() {
    : "${PREFILL_NODE_0_IP:?source setup/env_vars first}"
    : "${DECODE_NODE_0_IP:?source setup/env_vars first}"

    docker rm -f "${NAME_PREFIX}-router" 2>/dev/null || true
    set -x
    docker run -d --name "${NAME_PREFIX}-router" \
        --network host --privileged \
        --entrypoint python3 "$IMAGE_URI" \
        -m sglang_router.launch_router \
            --pd-disaggregation \
            --prefill "http://${PREFILL_NODE_0_IP}:${PREFILL_PORT}" "${BOOTSTRAP_PORT}" \
            --decode  "http://${DECODE_NODE_0_IP}:${DECODE_PORT}" \
            --policy round_robin \
            --host 0.0.0.0 --port "${ROUTER_PORT}"
    set +x
    echo
    echo "Router on :${ROUTER_PORT}. Health: curl -s localhost:${ROUTER_PORT}/health && echo OK"
    echo "Start it only after all four servers report 'The server is fired up'."
}

case "$CMD" in
    stop)   stop_all; exit 0 ;;
    router) run_router; exit 0 ;;
    serve|precompile) ;;
    *) echo "ERROR: COMMAND must be serve, precompile, router or stop" >&2; exit 1 ;;
esac

# --- common requirements for serve / precompile ------------------------------
: "${MODEL:?source setup/env_vars first}"
: "${HF_CACHE_DIR:?source setup/env_vars first}"
: "${IFACE:?source setup/env_vars first}"
: "${PREFILL_NODE_0_IP:?source setup/env_vars first}"
: "${DECODE_NODE_0_IP:?source setup/env_vars first}"

mkdir -p "${HF_CACHE_DIR}" "${HF_CACHE_DIR}/../dg-cache" "${HF_CACHE_DIR}/../sgl-cache"

# MOONCAKE_PROTOCOL=efa is NOT optional. Without it Mooncake silently installs
# its TCP transport ("Installing TCP transport" in the server log) and moves the
# KV cache over sockets: correct output, passes a smoke test, then deadlocks with
# "KVTransferError ... session is not alive" under sustained high-concurrency
# prefill. Nothing else in the stack reports the fallback.
# MC_FORCE_AUTO_DISCOVERY=1 lets Mooncake enumerate the EFA devices itself.
COMMON_ENV=(
    -e HF_HOME=/hf
    -e HF_TOKEN="${HF_TOKEN:-}"
    -e NCCL_SOCKET_IFNAME="$IFACE" -e GLOO_SOCKET_IFNAME="$IFACE"
    # Short name, not an absolute path: NCCL templates it into
    # libnccl-net-ofi.so. Omitting it silently falls back to TCP. See Dockerfile.
    -e NCCL_NET_PLUGIN=ofi
    -e FI_PROVIDER=efa -e FI_EFA_USE_DEVICE_RDMA=1
    -e MOONCAKE_PROTOCOL=efa
    -e MC_FORCE_AUTO_DISCOVERY=1
    # FI_HMEM must be exactly "cuda" -- not unset, and never a list containing
    # "system". Necessary but, on this image, not sufficient; see the tail of
    # this comment and the open blocker in benchmarks/README.md.
    #
    # Mooncake's EFA context calls setenv("FI_HMEM", "system", 0) before it opens
    # a domain. overwrite=0, so a value we set here wins; leaving it unset lets
    # "system" through. libfabric 2.4's EFA provider then restricts the domain's
    # HMEM ifaces to host memory, and every
    # fi_mr_regattr(iface=FI_HMEM_CUDA) -- which is how the GPU-resident KV cache
    # is registered -- fails with ENOSYS ("Operation not supported"):
    #
    #   efa_context.cpp:402] fi_mr_regattr failed for GPU memory 0x... (device N)
    #   efa_transport.cpp:460] Failed to register memory region chunk 0
    #
    # Startup then continues as if healthy -- SGLang's own PD warmup passes,
    # because its 4-token prompt does not need a registered GPU buffer -- and the
    # first real request fails as "Decode instance could be dead" on prefill and
    # "Failed to get kvcache from prefill instance" on decode. Neither role is
    # dead; the KV buffer was never registered.
    #
    # Measured directly against this image (fi_mr_regattr with Mooncake's exact
    # hints, 64 MiB of cudaMalloc'd memory, all 32 EFA domains):
    #   FI_HMEM=cuda           -> 0  (OK)
    #   FI_HMEM unset          -> 0  (OK, but Mooncake sets "system" itself)
    #   FI_HMEM=system         -> -38 ENOSYS
    #   FI_HMEM=system,cuda    -> -38 ENOSYS
    #   FI_HMEM=cuda,system    -> -38 ENOSYS
    # So the plain "cuda" value is the only safe setting; any list including
    # "system" disables CUDA registration even though CUDA is named.
    #
    # FI_EFA_USE_DATA_PATH_DIRECT changes nothing here -- both the "efa" and
    # "efa-direct" fabrics are enumerated regardless, and Mooncake only ever
    # opens "efa".
    #
    # Caveat: with FI_HMEM=cuda confirmed live in the container, registration on
    # this image STILL fails -- but with EOPNOTSUPP rather than the ENOSYS above,
    # so that is a second, separate defect inside Mooncake's EFA transport. The
    # same fi_mr_regattr call succeeds from a standalone C program in the same
    # container while the failing server runs, on every device and at the exact
    # failing length, so it is not the provider or the fabric.
    -e FI_HMEM=cuda
)
# Forwarded only when the caller sets it, so the default launch is unchanged.
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False keeps the torch caching
# allocator off CUDA VMM-backed segments. Worth having as a lever because VMM
# memory (cuMemCreate + cuMemMap) fails fi_mr_regattr with EFAULT on this image
# at the same length cudaMalloc registers cleanly -- so if the KV pool is
# VMM-backed, that is a candidate mechanism for the registration failure above.
if [[ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" ]]; then
    COMMON_ENV+=(-e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}")
fi
# NVSHMEM is DeepEP's transport, and the UCCL image does not contain it -- the
# LD_PRELOAD would resolve to a missing file. UCCL talks to EFA directly.
if [[ "$UCCL" != "1" ]]; then
    COMMON_ENV+=(
        -e NVSHMEM_REMOTE_TRANSPORT=libfabric
        -e NVSHMEM_LIBFABRIC_PROVIDER=efa
        -e NVSHMEM_NETDEVS_POLICY=EXTERNAL_SHARING_PCIE_SWITCH_NIC_EXCLUSIVE
        -e NVSHMEM_BOOTSTRAP=UID
        -e LD_PRELOAD=/opt/nvshmem/install/lib/libnvshmem_host.so.3.7.0
    )
fi
COMMON_MOUNTS=(
    -v "${HF_CACHE_DIR}:/hf"
    -v "${HF_CACHE_DIR}/../dg-cache:/root/.cache/deep_gemm"
    -v "${HF_CACHE_DIR}/../sgl-cache:/root/.cache/sglang"
)
# --privileged is required for the KV path, on BOTH roles and for every backend.
# Mooncake's memory_location.cpp asks the kernel which NUMA node each registered
# buffer sits on, and that query is not permitted under Docker's default
# capability set. The failure is not a permission error at startup: registration
# logs "Failed to get NUMA node ... Operation not permitted", carries on, and then
# the first KV transfer fails with "Memory region not registered by any active EFA
# device(s)" on the prefill side and "it might be dead" on the decode side -- as if
# the peer had crashed. Neither role is dead; the buffer was never registered
# against a device. Confirmed with MOE_BACKEND=tp, which loads neither DeepEP nor
# NVSHMEM, so this is Mooncake's registration path alone.
# (UCCL needs the same privilege for a different reason: it registers GPU memory
# for RDMA through the dma-buf/ibverbs path.)
DOCKER_FLAGS=(
    --gpus all --network host --ipc host --privileged
    --device /dev/infiniband --device /dev/gdrdrv
    --ulimit memlock=-1 --shm-size 32g
)

# --- precompile -------------------------------------------------------------
# Warms the mounted DeepGEMM JIT cache. Required once per prefill host for
# DeepEP + DP-attention; harmless otherwise. Must run at the full 16-rank EP
# shape (hence on both nodes), because a single-node run never compiles the
# num_groups=16 grouped-GEMM kernels that DP+EP uses.
if [[ "$CMD" == "precompile" ]]; then
    RANK=${2:?need NODE_RANK (0 or 1)}
    DP_FLAGS=()
    if [[ "$DP_ATTENTION" == "1" ]]; then
        DP_FLAGS=(--dp-size "$DP_SIZE" --enable-dp-attention --moe-dense-tp-size 1)
    fi
    docker rm -f "${NAME_PREFIX}-precompile" 2>/dev/null || true
    set -x
    docker run --rm --name "${NAME_PREFIX}-precompile" \
        "${DOCKER_FLAGS[@]}" "${COMMON_MOUNTS[@]}" "${COMMON_ENV[@]}" \
        --entrypoint bash "$IMAGE_URI" -c "
            python3 -m sglang.compile_deep_gemm \
                --model-path ${MODEL} --trust-remote-code \
                --tp-size ${TP_SIZE} --ep-size ${EP_SIZE} \
                --moe-a2a-backend deepep --deepep-mode normal \
                ${DP_FLAGS[*]} \
                --nnodes ${NUM_NODES} --node-rank ${RANK} \
                --dist-init-addr ${PREFILL_NODE_0_IP}:${PREFILL_DIST_PORT} 2>&1
        "
    set +x
    echo
    echo "DeepGEMM cache warmed on this host (${HF_CACHE_DIR}/../dg-cache)."
    echo "Run this on BOTH prefill nodes before serving with DP_ATTENTION=1."
    exit 0
fi

# --- serve ------------------------------------------------------------------
ROLE=${2:?need ROLE: prefill | decode}
RANK=${3:?need NODE_RANK (0 or 1)}

case "$ROLE" in
    prefill)
        SERVE_PORT=$PREFILL_PORT
        DIST_ADDR="${PREFILL_NODE_0_IP}:${PREFILL_DIST_PORT}"
        # DeepEP's high-throughput ("normal") dispatch is what prefill wants.
        DEEPEP_MODE=normal
        MEMFRAC=${PREFILL_MEM_FRACTION_STATIC:-0.82}
        ROLE_FLAGS=(--chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-16384}") ;;
    decode)
        SERVE_PORT=$DECODE_PORT
        DIST_ADDR="${DECODE_NODE_0_IP}:${DECODE_DIST_PORT}"
        DEEPEP_MODE=low_latency
        MEMFRAC=${DECODE_MEM_FRACTION_STATIC:-0.82}
        ROLE_FLAGS=() ;;
    *) echo "ERROR: ROLE must be prefill or decode" >&2; exit 1 ;;
esac

case "$MOE_BACKEND" in
    tp)
        PARALLEL=(--tp-size "$TP_SIZE")
        A2A=(--moe-a2a-backend none) ;;
    baseline)
        PARALLEL=(--tp-size "$TP_SIZE" --ep-size "$EP_SIZE")
        A2A=(--moe-a2a-backend none) ;;
    deepep)
        PARALLEL=(--tp-size "$TP_SIZE" --ep-size "$EP_SIZE")
        # Pinned per role rather than "auto" (which recipe/serve.sh uses for the
        # colocated case, where one server does both stages). Pinning is also a
        # hard requirement for UCCL: with --deepep-mode auto its prefill path
        # segfaults during startup.
        A2A=(--moe-a2a-backend deepep --deepep-mode "$DEEPEP_MODE") ;;
    *) echo "ERROR: MOE_BACKEND must be deepep, baseline or tp" >&2; exit 1 ;;
esac

DP_FLAGS=()
if [[ "$DP_ATTENTION" == "1" ]]; then
    DP_FLAGS=(--dp-size "$DP_SIZE" --enable-dp-attention --enable-dp-lm-head
              --moe-dense-tp-size 1)
    if [[ "$ROLE" == "decode" ]]; then
        # Without a pinned graph batch-size list, each of the 16 DP ranks
        # captures the whole default list and CUDA-graph capture OOMs; the
        # symptom is "scheduler died" during startup, not an OOM message.
        DP_FLAGS+=(--cuda-graph-bs "${CUDA_GRAPH_BS:-128}"
                   --max-running-requests "${MAX_RUNNING_REQUESTS:-256}")
        MEMFRAC=${DECODE_MEM_FRACTION_STATIC:-0.78}
    fi
fi

# UCCL's low_latency RDMA buffers are larger than DeepEP's: 0.82 OOMs during
# CUDA-graph capture inside uccl_ep.cc. Only lower it if the caller has not.
if [[ "$UCCL" == "1" && "$ROLE" == "decode" && -z "${DECODE_MEM_FRACTION_STATIC:-}" ]]; then
    MEMFRAC=0.70
fi

NAME="${NAME_PREFIX}-${ROLE}"
docker rm -f "$NAME" 2>/dev/null || true
set -x
docker run -d --name "$NAME" \
    "${DOCKER_FLAGS[@]}" "${COMMON_MOUNTS[@]}" "${COMMON_ENV[@]}" \
    --entrypoint bash "$IMAGE_URI" -c "
        python3 -m sglang.launch_server \
            --model-path ${MODEL} --trust-remote-code \
            ${PARALLEL[*]} ${A2A[*]} ${DP_FLAGS[*]} ${ROLE_FLAGS[*]} \
            --disaggregation-mode ${ROLE} \
            --disaggregation-transfer-backend mooncake \
            --disaggregation-bootstrap-port ${BOOTSTRAP_PORT} \
            --nnodes ${NUM_NODES} --node-rank ${RANK} \
            --dist-init-addr ${DIST_ADDR} \
            --host 0.0.0.0 --port ${SERVE_PORT} \
            `# Unconditional, and not a tuning choice -- see recipe/serve.sh. The` \
            `# sweeps pin --seed, so cached prefixes from one point silently turn` \
            `# the next point's prefill into a cache read (measured 12x).` \
            --disable-radix-cache \
            --mem-fraction-static ${MEMFRAC} \
            --watchdog-timeout ${WATCHDOG_TIMEOUT:-1200} 2>&1
    "
set +x

echo
echo "Launched ${NAME} (backend=${MOE_BACKEND} role=${ROLE} rank=${RANK} dp=${DP_ATTENTION})."
echo "Follow startup:  docker logs -f ${NAME}"
echo "R1 takes several minutes to load; wait for 'The server is fired up'."
echo
echo "Confirm the KV path is EFA and not TCP before trusting any number:"
echo "  docker logs ${NAME} 2>&1 | grep -E 'EfaTransport|Installing TCP transport'"
echo "  want: 'EfaTransport: Initialized EFA device ...'  /  must NOT see TCP"
