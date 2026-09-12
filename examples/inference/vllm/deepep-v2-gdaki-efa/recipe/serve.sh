#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0
# serve.sh — vLLM DP/EP serve with the DeepEP-V2 all-to-all backend over AWS EFA
# (NCCL-GIN GDAKI, GPU-initiated / kernel-posted WQE). One invocation per NODE
# (leader on node 0, worker on the rest). This is the NCCL_GIN_TYPE=3 counterpart to
# ../deepep-v2-efa/recipe/serve.sh (NCCL_GIN_TYPE=2 CPU-proxy) — the transport env is the
# only delta; the eager/non-eager decision surface is identical (same vLLM pin).
#
#   leader:  serve.sh leader <leader-pod-ip>
#   worker:  serve.sh worker <leader-pod-ip> <start-rank>     # 8 / 16 / 24 ...
#
# Scale is env-driven (the proven 2-node DP16 default is byte-identical unless overridden):
#   SERVE_DP=16|32...    total data-parallel size across all nodes (= EP size)
#   SERVE_DP_LOCAL=8     ranks per node (= GPUs per node)
#   SERVE_MODEL=...      any MoE whose n_routed_experts % SERVE_DP == 0 (preflighted below)
#
# ── EAGER vs DEFAULT COMPILATION (see README "eager vs non-eager") ──────────────
# SERVE_ENFORCE_EAGER=1 (DEFAULT) — the KNOWN-GOOD path, measured GREEN over GDAKI
#   (2026-08-14, 2× p5en, DP16/EP16, Qwen3-30B-A3B-FP8: 121/121 HTTP 200 over c=1..64).
# SERVE_ENFORCE_EAGER=0 — KNOWN-CRASH as of vLLM e2f993dc4: default compilation dies
#   deterministically during startup profile_run with a Triton illegal-memory-access in
#   deepep_v2.py:347 (buffer.combine). This is transport-independent (inherited from the
#   proxy package's root-cause). This script refuses =0 unless you also set
#   SERVE_I_UNDERSTAND_NONEAGER_CRASHES=1. Non-eager is pending the empty-ExpertTokensMetadata guard, filed upstream (vLLM PR #52632);
#   once merged, bump the vLLM pin past it — no patch step.
set -uo pipefail
ROLE="$1"; DP_MASTER_IP="$2"
DP_MASTER_PORT="${DP_MASTER_PORT:-29500}"

# ---- GDAKI (GPU-initiated) GIN + EFA env contract (the package delta vs ../deepep-v2-efa) ----
# OFI_NCCL_GDAKI_EFA_HW_COUNTER is honored FROM THE ENVIRONMENT (the kubernetes/ YAML sets
# it "off" on nodes with efa.ko < 3.3.0) — it is a STRING enum (auto|on|off); numeric 0/1
# abort plugin init. Not forced here.
export NCCL_GIN_TYPE=${NCCL_GIN_TYPE:-3} NCCL_GIN_ENABLE=1 OFI_NCCL_GIN_GDAKI=${OFI_NCCL_GIN_GDAKI:-1} OFI_NCCL_GIN_MAX_REQUESTS=512
export NCCL_CUMEM_ENABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IGNORE_DISABLED_P2P=1
export FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 FI_EFA_ENABLE_SHM_TRANSFER=0 FI_EFA_FORK_SAFE=1
export OFI_NCCL_PROTOCOL=RDMA DEEP_EP_BACKEND=nccl
export NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-/opt/aws-ofi-nccl-gdaki/lib/libnccl-net-ofi.so}
export EP_REUSE_NCCL_COMM=0   # DeepEP creates its own comm; torch's is lazy/null under vLLM (segfault rootcause)
export NCCL_DEBUG=${SERVE_NCCL_DEBUG:-WARN}   # SERVE_NCCL_DEBUG=INFO to see the efa-direct banner + GDAKI createContext
export EP_EFA_MAX_QPS=${EP_EFA_MAX_QPS:-2} EP_EFA_RDMA_GBS=${EP_EFA_RDMA_GBS:-25.0}

# ---- vLLM PR#41183 (DeepEPV2All2AllManager) envs — V2-native, shim OFF ----
export DEEP_EP_USE_V2_SHIM=0
export VLLM_USE_FLASHINFER_MOE_FP8=0
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-1800}
export VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE=1
export VLLM_DEEPEP_V2_PREFER_OVERLAP=0
export VLLM_DEEPEP_V2_ALLOW_MULTIPLE_REDUCTION=0

# ---- HF cache on the pod-local writable volume (NOT the image layer) ----
export HF_HOME=${HF_HOME:-/work/hf}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
mkdir -p "$HUGGINGFACE_HUB_CACHE"

# ---- lib path: nvshmem + nccl wheels (namespace-pkg safe) + GDAKI plugin + libfabric + rdma-core + efa + cuda ----
NVSHMEM_LIB="$(python3 -c 'import importlib.util,os
s=importlib.util.find_spec("nvidia.nvshmem")
p=(s.submodule_search_locations[0] if s and s.submodule_search_locations else None)
print(os.path.join(p,"lib") if p else "")' 2>/dev/null || true)"
NCCL_LIB="$(python3 -c 'import importlib.util,os
s=importlib.util.find_spec("nvidia.nccl")
p=(s.submodule_search_locations[0] if s and s.submodule_search_locations else None)
print(os.path.join(p,"lib") if p else "")' 2>/dev/null || true)"
export LD_LIBRARY_PATH="${NVSHMEM_LIB}:${NCCL_LIB}:/opt/aws-ofi-nccl-gdaki/lib:/opt/libfabric-gdaki/lib:/opt/rdma-core-gdaki/lib:/opt/amazon/efa/lib:/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PATH=/opt/amazon/efa/bin:$PATH

# ---- scale + model (defaults = the measured EP16 2-node shape) ----
SERVE_MODEL="${SERVE_MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
SERVE_DP="${SERVE_DP:-16}"
SERVE_DP_LOCAL="${SERVE_DP_LOCAL:-8}"
SERVE_TP="${SERVE_TP:-1}"
START_RANK="${3:-8}"
SERVE_MAX_MODEL_LEN="${SERVE_MAX_MODEL_LEN:-4096}"
SERVE_MAX_NUM_SEQS="${SERVE_MAX_NUM_SEQS:-16}"
SERVE_MAX_BATCHED_TOKENS="${SERVE_MAX_BATCHED_TOKENS:-256}"
SERVE_GPU_MEM_UTIL="${SERVE_GPU_MEM_UTIL:-0.70}"

# ---- EP-divisibility preflight (the one per-model gate): n_routed_experts % DP == 0 ----
# Qwen3 family=128 (ok 16/32), DeepSeek-V3/R1=256 (ok), Kimi-K2=384 (ok), DeepSeek-V2-Lite=64 (ok),
# Qwen1.5-MoE=60 (FAILS 16/32). Reads the model's config.json via huggingface_hub.
if [ "${SKIP_EP_PREFLIGHT:-0}" != "1" ]; then
  python3 - "$SERVE_MODEL" "$SERVE_DP" <<'PY' || exit 3
import json, sys
from huggingface_hub import hf_hub_download
model, dp = sys.argv[1], int(sys.argv[2])
cfg = json.load(open(hf_hub_download(model, "config.json")))
n = cfg.get("n_routed_experts") or cfg.get("num_experts")
assert n, f"{model}: no n_routed_experts/num_experts in config.json — not a routed-MoE model?"
assert n % dp == 0, (f"EP-DIVISIBILITY GATE FAILED: {model} has {n} routed experts, "
                     f"not divisible by DP/EP={dp}. Pick a DP that divides {n} or another model.")
print(f"EP preflight OK: {model} n_routed_experts={n} % DP={dp} == 0 ({n//dp} experts/rank)")
PY
fi

# ---- eager / non-eager selection (see header + README "eager vs non-eager") ----
SERVE_ENFORCE_EAGER="${SERVE_ENFORCE_EAGER:-1}"
EAGER_FLAG="--enforce-eager"
if [ "$SERVE_ENFORCE_EAGER" != "1" ]; then
  if [ "${SERVE_I_UNDERSTAND_NONEAGER_CRASHES:-0}" != "1" ]; then
    echo "REFUSING SERVE_ENFORCE_EAGER=0: default compilation is KNOWN-CRASH at vLLM"
    echo "e2f993dc4 (Triton IMA, deepep_v2.py:347 combine during profile_run). Run"
    echo "wait for the upstream guard (vLLM PR #52632) + a pin bump, or set"
    echo "SERVE_I_UNDERSTAND_NONEAGER_CRASHES=1 — or keep eager (the default)."
    exit 4
  fi
  echo "WARNING: running WITHOUT --enforce-eager — requires a vLLM pin past the upstream guard (see README)."
  EAGER_FLAG=""
fi

COMMON="--tensor-parallel-size ${SERVE_TP} --data-parallel-size ${SERVE_DP} --data-parallel-size-local ${SERVE_DP_LOCAL} \
  --data-parallel-address ${DP_MASTER_IP} --data-parallel-rpc-port ${DP_MASTER_PORT} \
  --data-parallel-backend mp --enable-expert-parallel --all2all-backend deepep_v2 \
  ${EAGER_FLAG} --max-model-len ${SERVE_MAX_MODEL_LEN} --max-num-seqs ${SERVE_MAX_NUM_SEQS} \
  --max-num-batched-tokens ${SERVE_MAX_BATCHED_TOKENS} \
  --gpu-memory-utilization ${SERVE_GPU_MEM_UTIL} --trust-remote-code --dtype bfloat16"

echo "===== vLLM serve model=$SERVE_MODEL role=$ROLE tp=${SERVE_TP} dp=${SERVE_DP}/local${SERVE_DP_LOCAL} eager=${SERVE_ENFORCE_EAGER} gin_type=${NCCL_GIN_TYPE} gdaki=${OFI_NCCL_GIN_GDAKI} start_rank=${START_RANK} dp_master=$DP_MASTER_IP $(hostname) $(date -u +%FT%TZ) ====="
python3 -c "import vllm; print('vllm', vllm.__version__)"
python3 -c "from vllm.distributed.device_communicators.all2all import DeepEPV2All2AllManager; print('PR#41183 symbol OK')"
python3 -c "import deep_ep; assert hasattr(deep_ep,'ElasticBuffer'); print('deep_ep ElasticBuffer OK')"

if [ "$ROLE" = "leader" ]; then
  exec vllm serve "$SERVE_MODEL" $COMMON --host 0.0.0.0 --port 8000
else
  # worker start-rank: 2-node DP16 -> 8 ; 4-node DP32 -> 8/16/24 (one worker node each)
  exec vllm serve "$SERVE_MODEL" $COMMON --data-parallel-start-rank "$START_RANK" --headless
fi
