#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0
# serve.sh — NVIDIA Dynamo (frontend ingress + dynamo.vllm worker) DP/EP serve with the DeepEP-V2
# all-to-all backend over AWS EFA (NCCL-GIN CPU-proxy). One invocation per NODE.
#
#   leader:  serve.sh leader <leader-pod-ip>                # node 0: dynamo.frontend + dynamo.vllm
#   worker:  serve.sh worker <leader-pod-ip> <start-rank>   # node 1+: dynamo.vllm --headless (8 / 16 / 24 ...)
#
# Topology — how Dynamo composes with the proven vLLM-DeepEP-V2 substrate (nothing about the
# transport changes; only the serving front is added):
#   LEADER (node 0):
#     (1) `dynamo.frontend` = the OpenAI-compatible HTTP ingress on :8000. It discovers the local
#         engine via the file backend (DYN_DISCOVERY_BACKEND=file, DYN_FILE_KV) — a NODE-0-LOCAL
#         path; it is NOT shared across nodes.
#     (2) `dynamo.vllm` (non-headless) = the leader engine (DP ranks 0..local-1). It registers with
#         the frontend over that file backend, then vLLM's OWN data-parallel coordinator (mp backend
#         + --data-parallel-address) fans the expert all-to-all across ALL nodes over EFA.
#   WORKER (node 1+):
#     `dynamo.vllm --headless` -> vLLM run_headless(): the worker runs vLLM ONLY (no discovery, no
#     dynamo endpoints); it joins the DP group purely through vLLM's data-parallel RPC to the leader
#     — byte-identical to the `vllm serve --headless` worker in the sibling vLLM sample.
# The ONLY delta vs the vLLM-DeepEP-V2 twin (../../vllm/deepep-v2-efa) is the dynamo.vllm wrapper +
# dynamo.frontend ingress. The proxy-Gin/EFA env contract, the DP-coordinator flags, the model, and
# the EP-divisibility preflight below are IDENTICAL to that sample (see README "Relationship to the
# vLLM sample").
#
# Scale is env-driven (the proven 2-node DP16 default is byte-identical unless overridden):
#   SERVE_DP=16|32...    total data-parallel size across all nodes (= EP size)
#   SERVE_DP_LOCAL=8     ranks per node (= GPUs per node)
#   SERVE_MODEL=...      any MoE whose n_routed_experts % SERVE_DP == 0 (preflighted below)
#
# ── EAGER vs DEFAULT COMPILATION (see README "eager vs non-eager") ──────────────
# SERVE_ENFORCE_EAGER=1 (DEFAULT, and the ONLY supported mode at the shipped pin) — eager is
#   what every measured E2E validation of this stack ran on: our H200 EP16/EP32 (2026-08-01,
#   16/16 chat 200) AND the independent B200 EP16 run (0/384 request failures) — both at the
#   shipped vLLM pin e2f993dc4. The benchmarks/ tables are this mode at this pin.
# SERVE_ENFORCE_EAGER=0 — default (CUDA-graph) compilation. NOT supported at the shipped pin:
#   at e2f993dc4, non-eager crashes deterministically ~48 s into startup in profile_run
#   (Triton IMA in deepep_v2.py combine, via finalize_async). The empty-ExpertTokensMetadata
#   guard that fixes it (vLLM #52632) landed only on the 0.26 line — and 0.26's deepep_v2
#   combine separately faults CUDA_ERROR_LAUNCH_FAILED (719) in profile_run on this exact
#   DeepEP/EFA substrate (measured 2026-09-04), so we cannot pin forward to get #52632 either.
#   The historical non-eager benchmarks/ tables were taken on this pin with #52632 applied as
#   an UNMERGED cherry-pick; reproducing them needs that patch. Leave this =1.
set -euo pipefail   # -e: preflight failures below must STOP the launch, not fall through to the serve
ROLE="${1:?usage: serve.sh leader|worker <leader-ip> [start-rank]}"; DP_MASTER_IP="${2:?need leader ip}"
case "$ROLE" in leader|worker) ;; *) echo "FATAL: unrecognized role '$ROLE' (leader|worker)"; exit 2 ;; esac
DP_MASTER_PORT="${DP_MASTER_PORT:-29500}"

# ---- proxy-Gin + EFA env contract (identical to the measured runs + deploy YAML) ----
export NCCL_GIN_TYPE=2 NCCL_GIN_ENABLE=1 OFI_NCCL_GIN_GDAKI=0 OFI_NCCL_GIN_MAX_REQUESTS=512
export NCCL_CUMEM_ENABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IGNORE_DISABLED_P2P=1
export FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 FI_EFA_ENABLE_SHM_TRANSFER=0 FI_EFA_FORK_SAFE=1
export OFI_NCCL_PROTOCOL=RDMA DEEP_EP_BACKEND=nccl
export NCCL_NET_PLUGIN=/opt/aws-ofi-nccl/lib/libnccl-net-ofi.so
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-^lo,docker,veth}   # exclusion, never positive selection: EFA nodes expose efa*/enp* and CNI adds bridges; auto-select can pick a non-routing iface -> rendezvous hang. Repo convention (nccl-tests Dockerfile).
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

# ---- Dynamo discovery: file backend, NODE-0-LOCAL (frontend <-> leader engine only) ----
# The worker is --headless and never touches this path; cross-node fan-out is vLLM's native DP
# coordinator (--data-parallel-address), so DYN_FILE_KV does NOT need a shared/cross-pod volume.
export DYN_DISCOVERY_BACKEND=${DYN_DISCOVERY_BACKEND:-file}
export DYN_FILE_KV=${DYN_FILE_KV:-/work/dynamo_store_kv}
mkdir -p "$DYN_FILE_KV"

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

# ---- scale + model (defaults = the measured EP16 2-node shape; identical to the vLLM sample) ----
SERVE_MODEL="${SERVE_MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
# Pin the model REVISION too: --trust-remote-code (below) executes repo-side modeling code
# in-pod, so an unpinned repo could change the code between preflight and load, or between
# runs. Default = the measured commit of the default model; empty for any other SERVE_MODEL
# (set SERVE_MODEL_REVISION to pin yours). Threaded into BOTH the preflight fetch and serve.
if [ "$SERVE_MODEL" = "Qwen/Qwen3-30B-A3B-FP8" ]; then
  SERVE_MODEL_REVISION="${SERVE_MODEL_REVISION:-d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
else
  SERVE_MODEL_REVISION="${SERVE_MODEL_REVISION:-}"
fi
SERVE_DP="${SERVE_DP:-16}"
SERVE_DP_LOCAL="${SERVE_DP_LOCAL:-8}"
SERVE_TP="${SERVE_TP:-1}"
START_RANK="${3:-$SERVE_DP_LOCAL}"   # derive from ranks/node (assigned above), not a hardcoded 8: a 4-GPU node wants worker ranks 4-7, not 8-11
SERVE_MAX_MODEL_LEN="${SERVE_MAX_MODEL_LEN:-4096}"
SERVE_MAX_NUM_SEQS="${SERVE_MAX_NUM_SEQS:-16}"
SERVE_MAX_BATCHED_TOKENS="${SERVE_MAX_BATCHED_TOKENS:-256}"
SERVE_GPU_MEM_UTIL="${SERVE_GPU_MEM_UTIL:-0.70}"
HTTP_PORT="${HTTP_PORT:-8000}"       # dynamo.frontend OpenAI HTTP port

# ---- EP-divisibility preflight (the one per-model gate): n_routed_experts % DP == 0 ----
# Qwen3 family=128 (ok 16/32), DeepSeek-V3/R1=256 (ok), Kimi-K2=384 (ok), DeepSeek-V2-Lite=64 (ok),
# Qwen1.5-MoE=60 (FAILS 16/32). Reads the model's config.json via huggingface_hub.
if [ "${SKIP_EP_PREFLIGHT:-0}" != "1" ]; then
  python3 - "$SERVE_MODEL" "$SERVE_DP" "$SERVE_MODEL_REVISION" <<'PY' || exit 3
import json, sys
from huggingface_hub import hf_hub_download
model, dp = sys.argv[1], int(sys.argv[2])
rev = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None   # pin the same revision the serve loads
cfg = json.load(open(hf_hub_download(model, "config.json", revision=rev)))
n = cfg.get("n_routed_experts") or cfg.get("num_experts")
assert n, f"{model}: no n_routed_experts/num_experts in config.json — not a routed-MoE model?"
assert n % dp == 0, (f"EP-DIVISIBILITY GATE FAILED: {model} has {n} routed experts, "
                     f"not divisible by DP/EP={dp}. Pick a DP that divides {n} or another model.")
print(f"EP preflight OK: {model} n_routed_experts={n} % DP={dp} == 0 ({n//dp} experts/rank)")
PY
fi

# ---- eager / non-eager selection (see header + README "eager vs non-eager") ----
# SERVE_ENFORCE_EAGER=1 (default) = eager, the ONLY supported mode at the shipped pin
# (e2f993dc4) and the mode the published tables were measured with. =0 = default (CUDA-graph)
# compilation, which at this pin crashes in profile_run (needs vLLM #52632, on the 0.26 line
# only — and 0.26 regresses deepep_v2 combine here; see header + README "eager vs non-eager").
SERVE_ENFORCE_EAGER="${SERVE_ENFORCE_EAGER:-1}"
EAGER_FLAG="--enforce-eager"
if [ "$SERVE_ENFORCE_EAGER" != "1" ]; then
  echo "WARNING: SERVE_ENFORCE_EAGER=0 (default CUDA-graph compilation) is NOT supported at the shipped"
  echo "  vLLM pin e2f993dc4 — it crashes deterministically in profile_run (needs the #52632 guard, which"
  echo "  is only on the deepep_v2-combine-regressing 0.26 line). Proceeding as requested, but expect a crash."
  EAGER_FLAG=""
fi

# --trust-remote-code is ON because the DeepSeek-V3/R1 + Kimi-K2 models serve.sh's preflight
# supports execute repo-side modeling code from their HF repos; the default Qwen3-30B-A3B does
# NOT need it. Kept unconditional because removing it breaks the DeepSeek/Kimi SERVE_MODEL path;
# a SERVE_MODEL you do not trust should not be pointed at this serve. Because that code runs
# in-pod, SERVE_MODEL_REVISION (above) pins WHICH commit's code executes — passed as --revision
# so the served code matches the preflighted code and cannot drift between runs.
REV_FLAG=""; [ -n "$SERVE_MODEL_REVISION" ] && REV_FLAG="--revision ${SERVE_MODEL_REVISION}"
# COMMON = the DP/EP + AsyncEngineArgs flags. dynamo.vllm forwards every flag it does not itself
# define straight into vLLM's AsyncEngineArgs (parse_known_args -> AsyncEngineArgs.from_cli_args),
# so this block is IDENTICAL to the vLLM sample's COMMON — the DeepEP-V2 backend selection,
# DP-coordinator wiring, and eager/revision flags all pass through unchanged.
COMMON="--tensor-parallel-size ${SERVE_TP} --data-parallel-size ${SERVE_DP} --data-parallel-size-local ${SERVE_DP_LOCAL} \
  --data-parallel-address ${DP_MASTER_IP} --data-parallel-rpc-port ${DP_MASTER_PORT} \
  --data-parallel-backend mp --enable-expert-parallel --all2all-backend deepep_v2 \
  ${EAGER_FLAG} ${REV_FLAG} --max-model-len ${SERVE_MAX_MODEL_LEN} --max-num-seqs ${SERVE_MAX_NUM_SEQS} \
  --max-num-batched-tokens ${SERVE_MAX_BATCHED_TOKENS} \
  --gpu-memory-utilization ${SERVE_GPU_MEM_UTIL} --trust-remote-code --dtype bfloat16"

echo "===== Dynamo serve model=$SERVE_MODEL role=$ROLE tp=${SERVE_TP} dp=${SERVE_DP}/local${SERVE_DP_LOCAL} eager=${SERVE_ENFORCE_EAGER} start_rank=${START_RANK} dp_master=$DP_MASTER_IP http_port=${HTTP_PORT} $(hostname) $(date -u +%FT%TZ) ====="
# import dynamo.vllm.MAIN (the serve entrypoint), not bare dynamo.vllm (whose __init__ imports no vLLM):
# this resolves the full serve-path vLLM API surface against the pinned wheel, so a 0.22↔dynamo mismatch
# fails HERE (seconds) rather than mid-load. main() is behind `if __name__` so importing it does not serve.
python3 -c "import dynamo.vllm.main, dynamo.frontend, vllm, deep_ep; print('dynamo.vllm.main/frontend OK | vllm', vllm.__version__, '| deep_ep OK')"
python3 -c "from vllm.distributed.device_communicators.all2all import DeepEPV2All2AllManager; print('PR#41183 symbol OK')"
python3 -c "import deep_ep; assert hasattr(deep_ep,'ElasticBuffer'); print('deep_ep ElasticBuffer OK')"

if [ "$ROLE" = "leader" ]; then
  # (1) ingress: dynamo.frontend on 0.0.0.0:$HTTP_PORT. The pod runs WITHOUT hostNetwork and the
  #     shipped headless Service is clusterIP:None with no external LB/NodePort — so 0.0.0.0 binds
  #     only inside the pod's network namespace (reachable via cluster DNS / a `kubectl port-forward`,
  #     NOT the host or the internet). The readinessProbe curls 127.0.0.1:$HTTP_PORT/health.
  #     --discovery-backend file is passed EXPLICITLY (the arg default is etcd; the flag beats the
  #     DYN_DISCOVERY_BACKEND env default and matches the proven invocation) so no etcd is needed.
  LOG_DIR="${LOG_DIR:-/work/dynamo-serve}"; mkdir -p "$LOG_DIR"
  python3 -m dynamo.frontend \
    --discovery-backend file --http-host 0.0.0.0 --http-port "$HTTP_PORT" \
    > "$LOG_DIR/frontend.log" 2>&1 &
  FRONTEND_PID=$!
  echo "dynamo.frontend PID $FRONTEND_PID -> $LOG_DIR/frontend.log (http 0.0.0.0:$HTTP_PORT)"
  # If the frontend dies, the engine is unreachable — fail the pod rather than serve a black hole.
  trap 'kill "$FRONTEND_PID" 2>/dev/null || true' EXIT
  sleep 6
  kill -0 "$FRONTEND_PID" 2>/dev/null || { echo "FATAL: dynamo.frontend exited during startup — see $LOG_DIR/frontend.log"; tail -20 "$LOG_DIR/frontend.log" || true; exit 4; }
  # (2) leader engine: dynamo.vllm NON-headless (DP ranks 0..local-1). Registers with the frontend
  #     over the file backend; vLLM's DP coordinator fans expert A2A across nodes over EFA.
  #     --request-plane tcp + --event-plane zmq are the arg defaults, passed explicitly to match the
  #     proven invocation. --kv-events-config disables KV-cache event publishing (no KV router here;
  #     leaving it on makes the engine publish over zmq that nothing consumes).
  exec python3 -m dynamo.vllm --model "$SERVE_MODEL" \
    --discovery-backend file --request-plane tcp --event-plane zmq \
    --kv-events-config '{"enable_kv_cache_events": false}' $COMMON
else
  # worker: dynamo.vllm --headless -> vLLM run_headless() (no discovery/endpoints, bypasses
  # DistributedRuntime entirely — no NATS/etcd). --headless IS the multi-node mechanism for
  # --data-parallel-backend mp (vLLM asserts nnodes>1 only under mp). It joins the DP group via
  # vLLM's data-parallel RPC to the leader. start-rank = this node's DP rank offset
  # (2-node DP16 -> 8 ; 4-node DP32 -> 8/16/24, one worker node each). No discovery flags: the
  # headless worker never touches the discovery backend.
  exec python3 -m dynamo.vllm --model "$SERVE_MODEL" --data-parallel-start-rank "$START_RANK" --headless $COMMON
fi
