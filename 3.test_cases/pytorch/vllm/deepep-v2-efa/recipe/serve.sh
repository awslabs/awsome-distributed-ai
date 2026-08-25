#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0
# serve.sh — vLLM DP/EP serve with the DeepEP-V2 all-to-all backend over AWS EFA
# (NCCL-GIN CPU-proxy). One invocation per NODE (leader on node 0, worker on the rest).
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
# SERVE_ENFORCE_EAGER=1 (DEFAULT) — the KNOWN-GOOD path. Every measured E2E validation of
#   this stack ran eager: our H200 EP16/EP32 (2026-08-01, 16/16 chat 200) AND the
#   independent B200 EP16 run (0/384 request failures) — both at vLLM e2f993dc4.
# SERVE_ENFORCE_EAGER=0 — KNOWN-CRASH as of vLLM e2f993dc4: default compilation dies
#   deterministically during startup profile_run with a Triton illegal-memory-access in
#   deepep_v2.py:347 (buffer.combine, reached via finalize_async) on 12+/16 ranks
#   (bounded on B200 EP16, 2/2 repro; Xid 43; ECC clean). NOT a DeepEP kernel bug
#   (standalone test_ep.py passes the same dims) and NOT cudagraphs alone (compilation
#   mode NONE without --enforce-eager still crashes). Track the upstream item before
#   flipping this. This script therefore refuses =0 unless you also set
#   SERVE_I_UNDERSTAND_NONEAGER_CRASHES=1.
set -euo pipefail   # -e: preflight failures below must STOP the launch, not fall through to vllm serve
ROLE="${1:?usage: serve.sh {leader|worker} <leader-ip> [start-rank]}"; DP_MASTER_IP="${2:?need leader ip}"
case "$ROLE" in leader|worker) ;; *) echo "FATAL: unrecognized role '$ROLE' (leader|worker)"; exit 2 ;; esac
DP_MASTER_PORT="${DP_MASTER_PORT:-29500}"

# ---- proxy-Gin + EFA env contract (identical to the measured runs + deploy YAML) ----
export NCCL_GIN_TYPE=2 NCCL_GIN_ENABLE=1 OFI_NCCL_GIN_GDAKI=0 OFI_NCCL_GIN_MAX_REQUESTS=512
export NCCL_CUMEM_ENABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IGNORE_DISABLED_P2P=1
export FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 FI_EFA_ENABLE_SHM_TRANSFER=0 FI_EFA_FORK_SAFE=1
export OFI_NCCL_PROTOCOL=RDMA DEEP_EP_BACKEND=nccl
export NCCL_NET_PLUGIN=/opt/aws-ofi-nccl/lib/libnccl-net-ofi.so
export OFI_NCCL_GDRCOPY_FORCED_PCIE_COPY=1   # PR#1351: assert forced-PCIe on gdrdrv-2.4-kernel nodes
export EP_REUSE_NCCL_COMM=0   # DeepEP creates its own comm; torch's is lazy/null under vLLM (segfault rootcause 2026-08-14)
export NCCL_DEBUG=${SERVE_NCCL_DEBUG:-WARN}
# EP_EFA_MAX_QPS=2 is the value the published benchmarks/ numbers were measured with
# (DeepEP PR#612's conservative EFA default: its commit message caps auto-QP at 2 to avoid
# a 128-slot GIN request-ring overflow) — kept as the default so the sample reproduces its
# own tables. It may leave throughput on the table on newer aws-ofi-nccl: that 128-slot ring
# was replaced by the seq-window design upstream (6e504db), and the plugin pinned here
# (9c44d34) is 76 commits PAST that redesign (github.com/aws/aws-ofi-nccl/compare/6e504db...9c44d34),
# so the image is not in the condition the cap was written for; a 2x B200 A/B through this
# same vLLM path measured +29% throughput / -23% p50 uncapped (=129) with 0/384 failures, so
# uncapping is worth testing on p5en. If you tune it, re-measure at YOUR concurrency and
# record the value — both knobs are part of the benchmark provenance table.
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

# ---- lib path: nvshmem + nccl wheels (namespace-pkg safe) + plugin + efa + cuda ----
NVSHMEM_LIB="$(python3 -c 'import importlib.util,os
s=importlib.util.find_spec("nvidia.nvshmem")
p=(s.submodule_search_locations[0] if s and s.submodule_search_locations else None)
print(os.path.join(p,"lib") if p else "")' 2>/dev/null || true)"
NCCL_LIB="$(python3 -c 'import importlib.util,os
s=importlib.util.find_spec("nvidia.nccl")
p=(s.submodule_search_locations[0] if s and s.submodule_search_locations else None)
print(os.path.join(p,"lib") if p else "")' 2>/dev/null || true)"
export LD_LIBRARY_PATH="${NVSHMEM_LIB}:${NCCL_LIB}:/opt/aws-ofi-nccl/lib:/opt/amazon/efa/lib:/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
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
    echo "e2f993dc4 (Triton IMA, deepep_v2.py:347 combine during profile_run — bounded"
    echo "repro on B200 EP16; fix filed upstream as vLLM PR #52632 — bump the pin once merged)."
    echo "Set SERVE_I_UNDERSTAND_NONEAGER_CRASHES=1 to try anyway"
    echo "(useful ONLY for reproducing/triaging the crash), or keep eager (the default)."
    exit 4
  fi
  echo "WARNING: running WITHOUT --enforce-eager — expect the profile_run crash (~48s in)."
  EAGER_FLAG=""
fi

# --trust-remote-code is ON because the DeepSeek-V3/R1 + Kimi-K2 models serve.sh's preflight
# supports execute repo-side modeling code from their HF repos; the default Qwen3-30B-A3B does
# NOT need it. Kept unconditional because removing it breaks the DeepSeek/Kimi SERVE_MODEL path;
# a SERVE_MODEL you do not trust should not be pointed at this serve.
COMMON="--tensor-parallel-size ${SERVE_TP} --data-parallel-size ${SERVE_DP} --data-parallel-size-local ${SERVE_DP_LOCAL} \
  --data-parallel-address ${DP_MASTER_IP} --data-parallel-rpc-port ${DP_MASTER_PORT} \
  --data-parallel-backend mp --enable-expert-parallel --all2all-backend deepep_v2 \
  ${EAGER_FLAG} --max-model-len ${SERVE_MAX_MODEL_LEN} --max-num-seqs ${SERVE_MAX_NUM_SEQS} \
  --max-num-batched-tokens ${SERVE_MAX_BATCHED_TOKENS} \
  --gpu-memory-utilization ${SERVE_GPU_MEM_UTIL} --trust-remote-code --dtype bfloat16"

echo "===== vLLM serve model=$SERVE_MODEL role=$ROLE tp=${SERVE_TP} dp=${SERVE_DP}/local${SERVE_DP_LOCAL} eager=${SERVE_ENFORCE_EAGER} start_rank=${START_RANK} dp_master=$DP_MASTER_IP $(hostname) $(date -u +%FT%TZ) ====="
python3 -c "import vllm; print('vllm', vllm.__version__)"
python3 -c "from vllm.distributed.device_communicators.all2all import DeepEPV2All2AllManager; print('PR#41183 symbol OK')"
python3 -c "import deep_ep; assert hasattr(deep_ep,'ElasticBuffer'); print('deep_ep ElasticBuffer OK')"

if [ "$ROLE" = "leader" ]; then
  exec vllm serve "$SERVE_MODEL" $COMMON --host 0.0.0.0 --port 8000
else
  # worker start-rank: 2-node DP16 -> 8 ; 4-node DP32 -> 8/16/24 (one worker node each)
  exec vllm serve "$SERVE_MODEL" $COMMON --data-parallel-start-rank "$START_RANK" --headless
fi
