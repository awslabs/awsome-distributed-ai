#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0
# run-kernel-test.sh — prove the factory-selected NcclEP actually moves bytes over EFA
# BEFORE any model load. This is the "smoke-test the transport" gate: it drives TRT-LLM's
# own CommunicationFactory.create_strategy() + dispatch/combine across the nodes with the
# SAME NcclEP/EFA env the serve uses (probe_nccl_ep.py), and exits non-zero on failure.
# Cheap, and — unlike a node-spanning serve, which needs mpirun (see the kubernetes
# manifest header) — this torchrun probe DOES cross the node boundary.
#
#   leader:  run-kernel-test.sh leader <leader-ip>
#   worker:  run-kernel-test.sh worker <leader-ip> <node-rank>   # node-rank 1,2,3 for the worker nodes
#
# Env (shared with serve.sh): NNODES (default 2), GPUS_PER_NODE (default 8),
# TRTLLM_NCCL_EP_ALGO (default LOW_LATENCY — see the patched-image guard below).
set -uo pipefail

# NOTE: keep the usage message brace-free. A literal '}' inside a ${1:?...} message (e.g.
# "{leader|worker}") closes the parameter expansion early, so ROLE captures the junk tail and
# the case below never matches -> FATAL on every invocation. Check role in a separate statement.
ROLE="${1:-}"; : "${ROLE:?usage: run-kernel-test.sh leader-or-worker <leader-ip> [node-rank]}"
case "$ROLE" in leader|worker) ;; *) echo "FATAL: unrecognized role '$ROLE' (expected leader|worker)"; exit 2 ;; esac
LEADER_IP="${2:?need leader ip}"
if [ "$ROLE" = "worker" ]; then NODE_RANK_ARG="${3:?worker requires an explicit node-rank (1,2,...)}"; else NODE_RANK_ARG=0; fi
NNODES="${NNODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
PROBE="/opt/probe_nccl_ep.py"
[ -f "$PROBE" ] || { echo "FAIL: $PROBE not in image — rebuild from the Dockerfile"; exit 3; }

# ---- NCCL-GIN proxy + EFA contract, VERBATIM from serve.sh (same transport under test) ----
export NCCL_GIN_TYPE=2 NCCL_GIN_ENABLE=1   # 2 = CPU-proxy GIN, the EFA-viable GIN mode
export NCCL_CUMEM_ENABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IGNORE_DISABLED_P2P=1
export FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 FI_EFA_ENABLE_SHM_TRANSFER=0 FI_EFA_FORK_SAFE=1
export OFI_NCCL_PROTOCOL=RDMA
export NCCL_NET_PLUGIN=/opt/aws-ofi-nccl/lib/libnccl-net-ofi.so
# ---- NcclEP selection (same knobs the serve exports) ----
export TRTLLM_FORCE_COMM_METHOD=NCCL_EP
export NCCL_EP_NUM_QP_PER_RANK="${NCCL_EP_NUM_QP_PER_RANK:-32}"
export ENABLE_CONFIGURABLE_MOE=1   # match serve.sh: the correctness gate must exercise the
                                   # SAME configurable-MoE path the serve runs, not a variant
export TLLM_LOG_LEVEL=info   # TLLM_, not TRTLLM_ — the latter is silently ignored (README trap 1)
export TRTLLM_NCCL_EP_ALGO="${TRTLLM_NCCL_EP_ALGO:-LOW_LATENCY}"
if [ "$TRTLLM_NCCL_EP_ALGO" != "LOW_LATENCY" ] && [ ! -f /opt/.ht-flat-patch-applied ]; then
  echo "FATAL: TRTLLM_NCCL_EP_ALGO=$TRTLLM_NCCL_EP_ALGO needs an image built with"
  echo "APPLY_HT_FLAT_PATCH=1 (TensorRT-LLM PR#17715). Unpatched upstream ignores the knob"
  echo "and runs LOW_LATENCY+RANK_MAJOR — refusing rather than probing the wrong algorithm."
  exit 4
fi
export NCCL_DEBUG=${KERNEL_TEST_NCCL_DEBUG:-INFO}   # INFO so the efa-direct banner prints = transport proof

# ---- lib path: pip NCCL 2.30.4 first (the container's baked NCCL must NOT win) ----
# Fail loud rather than swallow (see serve.sh): an empty NCCL_LIB would silently drop the
# "pip NCCL wins" contract (README trap 3) and prepend an empty LD_LIBRARY_PATH element.
# The :? guard fires even though set -e is off here (parameter expansion exits regardless).
NCCL_LIB="$(python3 -c 'import importlib.util,os
s=importlib.util.find_spec("nvidia.nccl")
p=(s.submodule_search_locations[0] if s and s.submodule_search_locations else None)
print(os.path.join(p,"lib") if p else "")')"
: "${NCCL_LIB:?could not locate the pip nvidia-nccl-cu13 lib dir — NcclEP needs it first on LD_LIBRARY_PATH (README trap 3)}"
export LD_LIBRARY_PATH="${NCCL_LIB}:/opt/aws-ofi-nccl/lib:/opt/amazon/efa/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PATH=/opt/amazon/efa/bin:$PATH

NODE_RANK="$NODE_RANK_ARG"; [ "$ROLE" = "leader" ] && NODE_RANK=0
echo "===== NcclEP kernel smoke: role=$ROLE node_rank=$NODE_RANK nnodes=$NNODES gpus/node=$GPUS_PER_NODE algo=$TRTLLM_NCCL_EP_ALGO leader=$LEADER_IP $(hostname) $(date -u +%FT%TZ) ====="

set -o pipefail
# 8 torchrun procs per node (one per GPU): probe_nccl_ep.py is a per-rank script that
# reads RANK/WORLD_SIZE/LOCAL_RANK — 2 nodes x 8 = the 16-rank EP16 shape the measured
# runs used. The probe's MPI shim backs TRT-LLM's bootstrap collectives with
# torch.distributed, which is why this crosses nodes where mpirun cannot.
torchrun --nnodes="$NNODES" --nproc-per-node="$GPUS_PER_NODE" --node-rank="$NODE_RANK" \
  --master-addr="$LEADER_IP" --master-port=29501 "$PROBE" 2>&1 | tee /tmp/kernel-test.$NODE_RANK.log
rc=${PIPESTATUS[0]}

# fail-loud contract: gate on the exit code (the probe all-reduces a MIN verdict — one bad
# rank fails every rank), then separately require the EFA transport banner.
if [ "$rc" -eq 0 ]; then
  # confirm EFA (not TCP/SHM) actually carried it — only provider-specific banners count;
  # a bare "NET/OFI" line also prints for tcp;ofi_rxm fallback, the exact case to rule out
  if grep -qiE "efa-direct|Selected Provider is efa" /tmp/kernel-test.$NODE_RANK.log; then
    echo "KERNEL-TEST PASS (node_rank=$NODE_RANK) — factory-selected NcclEP dispatch/combine over EFA verified"
    exit 0
  fi
  echo "KERNEL-TEST INCONCLUSIVE: probe passed but no EFA-provider banner in log — confirm transport before trusting"
  exit 2
fi
echo "KERNEL-TEST FAIL (node_rank=$NODE_RANK, torchrun rc=$rc) — see /tmp/kernel-test.$NODE_RANK.log"
exit 1
