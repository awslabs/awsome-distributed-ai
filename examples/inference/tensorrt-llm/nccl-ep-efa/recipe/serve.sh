#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0
# serve.sh — trtllm-serve with the NcclEP MoE all-to-all backend over AWS EFA
# (NCCL-GIN CPU-proxy). SINGLE-NODE (tp_size == ep_size == GPUs on the node).
#
# Why single-node: a node-spanning trtllm-serve needs mpirun, and mpirun's OOB cannot
# route between EKS pods — AWS VPC-CNI gives each pod a /32 eth0, so OMPI's
# opal_net_samenetwork() never matches a peer address and the HNP reports "no route to"
# the remote orted. torchrun does not share that limitation, which is why the CROSS-NODE
# proof lives in run-kernel-test.sh (16-rank factory/dispatch/combine probe), while the
# served-completion proof is this script at EP8. See README "Known limitations".
#
#   serve.sh            # in-pod (the kubernetes manifest runs this on ordinal 0)
#
# Knobs (env; defaults = the measured EP8 shape):
#   SERVE_MODEL=...   any MoE whose routed-expert count % SERVE_EP == 0 (preflighted below)
#   SERVE_TP=8 SERVE_EP=8   single-node: both == GPUs on the node
#   TRTLLM_NCCL_EP_ALGO=LOW_LATENCY   non-default values need the patched image (guard below)
set -euo pipefail   # -e: preflight failures below must STOP the launch, not fall through to trtllm-serve

# ---- NCCL-GIN proxy + EFA env contract (identical to the measured runs + deploy YAML) ----
export NCCL_GIN_TYPE=2 NCCL_GIN_ENABLE=1   # 2 = CPU-proxy GIN, the EFA-viable GIN mode
export NCCL_CUMEM_ENABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IGNORE_DISABLED_P2P=1
export FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 FI_EFA_ENABLE_SHM_TRANSFER=0 FI_EFA_FORK_SAFE=1
export OFI_NCCL_PROTOCOL=RDMA
export NCCL_NET_PLUGIN=/opt/aws-ofi-nccl/lib/libnccl-net-ofi.so
export NCCL_DEBUG=${SERVE_NCCL_DEBUG:-WARN}  # INFO to see the efa-direct banner proof

# ---- NcclEP selection — the point of this sample ----
# TRTLLM_FORCE_COMM_METHOD=NCCL_EP makes CommunicationFactory.create_strategy() return
# NcclEP — but ONLY once attention DP is on (the extra_llm_api_options YAML below);
# without it the factory short-circuits to None BEFORE reading this env.
export TRTLLM_FORCE_COMM_METHOD=NCCL_EP
export NCCL_EP_NUM_QP_PER_RANK="${NCCL_EP_NUM_QP_PER_RANK:-32}"   # QPs per rank — part of benchmark provenance
export ENABLE_CONFIGURABLE_MOE=1
# TLLM_LOG_LEVEL (not TRTLLM_LOG_LEVEL — that one is silently ignored) is what
# tensorrt_llm/logger.py reads; without info the "NCCL EP group created ..." line that
# constitutes engagement PROOF is suppressed at the default 'error'. README trap 1.
export TLLM_LOG_LEVEL=info

# ---- algorithm knob guard: honored ONLY on a patched image (TensorRT-LLM PR#17715) ----
# Unpatched upstream hardcodes LOW_LATENCY+RANK_MAJOR (the measured-clean baseline on this
# substrate). Refusing beats silently serving a different algorithm than the one requested.
export TRTLLM_NCCL_EP_ALGO="${TRTLLM_NCCL_EP_ALGO:-LOW_LATENCY}"
if [ "$TRTLLM_NCCL_EP_ALGO" != "LOW_LATENCY" ] && [ ! -f /opt/.ht-flat-patch-applied ]; then
  echo "REFUSING TRTLLM_NCCL_EP_ALGO=$TRTLLM_NCCL_EP_ALGO on an unpatched image: upstream"
  echo "ignores the knob and runs LOW_LATENCY+RANK_MAJOR regardless. Build with"
  echo "APPLY_HT_FLAT_PATCH=1 (bakes TensorRT-LLM PR#17715) to make it selectable."
  exit 4
fi

# ---- HF cache on the pod-local writable volume (NOT the image layer) ----
export HF_HOME=${HF_HOME:-/work/hf}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
mkdir -p "$HUGGINGFACE_HUB_CACHE"

# ---- lib path: pip NCCL 2.30.4 first (the container's baked NCCL must NOT win) ----
# No `2>/dev/null || true`: that swallowed every failure, left NCCL_LIB empty, silently
# dropped the "pip NCCL wins" contract (README trap 3), and put a leading empty element on
# LD_LIBRARY_PATH (the loader then searches $PWD first). Fail loud instead — the whole
# sample depends on this resolving, and the downstream symptom otherwise misdirects
# ("nccl-ep is not installed" points at nccl4py when the real cause is a lost lib path).
NCCL_LIB="$(python3 -c 'import importlib.util,os
s=importlib.util.find_spec("nvidia.nccl")
p=(s.submodule_search_locations[0] if s and s.submodule_search_locations else None)
print(os.path.join(p,"lib") if p else "")')"
: "${NCCL_LIB:?could not locate the pip nvidia-nccl-cu13 lib dir — NcclEP needs it first on LD_LIBRARY_PATH (README trap 3)}"
export LD_LIBRARY_PATH="${NCCL_LIB}:/opt/aws-ofi-nccl/lib:/opt/amazon/efa/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PATH=/opt/amazon/efa/bin:$PATH

# ---- scale + model (defaults = the measured single-node EP8 shape) ----
SERVE_MODEL="${SERVE_MODEL:-Qwen/Qwen3-30B-A3B}"
# --trust_remote_code executes arbitrary repo Python in a privileged root container, so pin
# WHAT executes: SERVE_REVISION (a commit SHA/tag) makes the model an immutable artifact like
# every other pin here, instead of "whatever main points at today". Empty = HF default branch.
SERVE_REVISION="${SERVE_REVISION:-}"
SERVE_TP="${SERVE_TP:-8}"
SERVE_EP="${SERVE_EP:-8}"
SERVE_HOST="${SERVE_HOST:-0.0.0.0}"
SERVE_PORT="${SERVE_PORT:-8000}"
SERVE_MAX_BATCH_SIZE="${SERVE_MAX_BATCH_SIZE:-4}"
SERVE_MAX_NUM_TOKENS="${SERVE_MAX_NUM_TOKENS:-2048}"
SERVE_MAX_SEQ_LEN="${SERVE_MAX_SEQ_LEN:-2048}"
SERVE_KV_FRACTION="${SERVE_KV_FRACTION:-0.5}"

# ---- EP-divisibility preflight (the one per-model gate): routed experts % EP == 0 ----
# Qwen3 family=128 (ok 8/16/32), DeepSeek-V3/R1=256 (ok), Qwen1.5-MoE=60 (FAILS 8/16).
# Reads the model's config.json via huggingface_hub.
if [ "${SKIP_EP_PREFLIGHT:-0}" != "1" ]; then
  python3 - "$SERVE_MODEL" "$SERVE_EP" "$SERVE_REVISION" <<'PY' || exit 3
import json, sys
from huggingface_hub import hf_hub_download
model, ep = sys.argv[1], int(sys.argv[2])
rev = sys.argv[3] or None   # validate the SAME pinned revision the serve will run
cfg = json.load(open(hf_hub_download(model, "config.json", revision=rev)))
n = cfg.get("n_routed_experts") or cfg.get("num_experts")
assert n, f"{model}: no n_routed_experts/num_experts in config.json — not a routed-MoE model?"
assert n % ep == 0, (f"EP-DIVISIBILITY GATE FAILED: {model} has {n} routed experts, "
                     f"not divisible by EP={ep}. Pick an EP that divides {n} or another model.")
print(f"EP preflight OK: {model} routed_experts={n} % EP={ep} == 0 ({n//ep} experts/rank)")
PY
fi

# ---- attention DP: REQUIRED for NcclEP (see the selection comment above) ----
CFG=/tmp/extra-llm-api-config.yml
cat > "$CFG" <<'YML'
enable_attention_dp: true
YML

echo "===== trtllm-serve model=$SERVE_MODEL tp=${SERVE_TP} ep=${SERVE_EP} algo=${TRTLLM_NCCL_EP_ALGO} qp/rank=${NCCL_EP_NUM_QP_PER_RANK} $(hostname) $(date -u +%FT%TZ) ====="
python3 -c "import tensorrt_llm; print('tensorrt_llm', tensorrt_llm.__version__)"
python3 -c "import nccl.ep; print('nccl.ep OK (nccl4py)')"

# Engagement PROOF once up (requires TLLM_LOG_LEVEL=info above):
#   grep "NCCL EP group created" <server log>
#   -> "... layout=RANK_MAJOR, algorithm=LOW_LATENCY" on every rank (or FLAT/HIGH_THROUGHPUT
#      on a patched image with TRTLLM_NCCL_EP_ALGO=HIGH_THROUGHPUT)
# --trust_remote_code: MoE models whose HF repos ship modeling code (DeepSeek family) need
# it; a SERVE_MODEL you do not trust should not be pointed at this serve.
exec trtllm-serve "$SERVE_MODEL" \
  ${SERVE_REVISION:+--revision "$SERVE_REVISION"} \
  --host "$SERVE_HOST" --port "$SERVE_PORT" \
  --tp_size "$SERVE_TP" --ep_size "$SERVE_EP" --pp_size 1 \
  --max_batch_size "$SERVE_MAX_BATCH_SIZE" --max_num_tokens "$SERVE_MAX_NUM_TOKENS" \
  --max_seq_len "$SERVE_MAX_SEQ_LEN" \
  --kv_cache_free_gpu_memory_fraction "$SERVE_KV_FRACTION" \
  --trust_remote_code \
  --extra_llm_api_options "$CFG"
