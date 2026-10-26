#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Serve the model in each layout and benchmark it with vllm bench serve:
#   tp8     tensor parallelism 8 on the head instance alone
#   pp2tp8  pipeline parallelism 2 x tensor parallelism 8, one stage per instance
#   tp16    tensor parallelism 16 across both instances
# Two-instance layouts run with the near and the far second instance, alternating within each round (ABBA), so a
# drift during the run can't favour one placement. The head runs the API server, the engine core and the benchmark
# client in every run. A two-instance run counts only if every NCCL process on both instances reports
# "Using network Libfabric" and none reports the TCP socket fallback.
#
# Usage: source env_vars && ./3.benchmark-layouts.sh           then: python3 4.analyze.py <result dir>
set -uo pipefail
: "${NAMESPACE:?source env_vars}" "${MODEL_REPO:?}" "${GPU_PER_NODE:?}"
LAYOUTS=${LAYOUTS:-"tp8 pp2tp8 tp16"}
ROUNDS=${ROUNDS:-2}
CONCS=${CONCS:-"1 8 32 64"}
INLEN=${INLEN:-1024}; OUTLEN=${OUTLEN:-128}
MAXLEN=${MAXLEN:-8192}; GPUUTIL=${GPUUTIL:-0.92}
PORT=${PORT:-8000}; MPORT=${MPORT:-29500}
READY_MAX=${READY_MAX:-2400}; BENCH_TMO=${BENCH_TMO:-3600}
NET_PLUGIN=${NET_PLUGIN:-/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so}
SOCKET_IFNAME=${SOCKET_IFNAME:-eth0}
MODEL=/model/$(basename "$MODEL_REPO")
OUT=${OUT:-./results/layouts-$(date -u +%Y%m%dT%H%MZ)}
mkdir -p "$OUT/raw"
k() { kubectl -n "$NAMESPACE" "$@"; }
HEAD=vllm-head; NEAR=vllm-near; FAR=vllm-far
retry() { local _; for _ in 1 2 3 4 5; do "$@" && return 0; sleep 5; done; return 1; }
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$OUT/run.log"; }

# Placement check: NEAR must share all three network-node layers with HEAD, FAR only layers 1 and 2.
kubectl get nodes -o json > "$OUT/.nodes.json"
for p in $HEAD $NEAR $FAR; do echo "$p $(k get pod "$p" -o jsonpath='{.spec.nodeName}')"; done > "$OUT/.pods.txt"
python3 - "$OUT" > "$OUT/placement.txt" <<'PY' || { cat "$OUT/placement.txt"; exit 1; }
import json
import sys

out = sys.argv[1]
with open(f"{out}/.nodes.json", encoding="utf-8") as fh:
    labels = {n["metadata"]["name"]: n["metadata"]["labels"] for n in json.load(fh)["items"]}
with open(f"{out}/.pods.txt", encoding="utf-8") as fh:
    pods = dict(line.split() for line in fh)
key = "topology.k8s.aws/network-node-layer-%d"


def shared(a, b):
    d = 0
    for i in (1, 2, 3):
        x, y = labels[pods[a]].get(key % i), labels[pods[b]].get(key % i)
        if x and x == y:
            d += 1
        else:
            break
    return d


near, far = shared("vllm-head", "vllm-near"), shared("vllm-head", "vllm-far")
print(f"layers shared with vllm-head: vllm-near {near}, vllm-far {far}")
sys.exit(0 if (near, far) == (3, 2) else 1)
PY
rm -f "$OUT/.nodes.json" "$OUT/.pods.txt"; cat "$OUT/placement.txt"

HEAD_IP=$(k get pod $HEAD -o jsonpath='{.status.podIP}')
ENVS="NCCL_NET_PLUGIN=$NET_PLUGIN NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET NCCL_SOCKET_IFNAME=$SOCKET_IFNAME \
GLOO_SOCKET_IFNAME=$SOCKET_IFNAME VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_LOGGING_LEVEL=INFO HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1"
COMMON="$MODEL --served-model-name model --trust-remote-code --no-enable-prefix-caching \
--max-model-len $MAXLEN --gpu-memory-utilization $GPUUTIL --port $PORT"
{ echo "utc=$(date -u +%FT%TZ) model=$MODEL_REPO@${MODEL_REVISION:-} layouts=$LAYOUTS rounds=$ROUNDS concs=$CONCS"
  echo "input=$INLEN output=$OUTLEN max_model_len=$MAXLEN gpu_util=$GPUUTIL prompts_per_point=8x concurrency (min 16)"
  k exec $HEAD -- python3 -c 'import vllm, torch; print("vllm", vllm.__version__, "torch", torch.__version__, "nccl", torch.cuda.nccl.version())'
} > "$OUT/meta.txt" 2>&1; cat "$OUT/meta.txt"

stop_all() {
  for p in $HEAD $NEAR $FAR; do
    k exec "$p" -- pkill -9 -f '[v]llm bench' >/dev/null 2>&1
    k exec "$p" -- pkill -9 -f '[v]llm serve' >/dev/null 2>&1
    k exec "$p" -- pkill -9 -f '[V]LLM::' >/dev/null 2>&1
  done
  sleep 30
}

launch() {   # pod role tag "serve args": run vLLM detached inside the pod
  local pod=$1 role=$2 tag=$3 args=$4
  { echo '#!/bin/bash'; echo 'cd /tmp'
    for kv in $ENVS; do echo "export $kv"; done
    echo "exec vllm serve $args"
  } > "$OUT/raw/launch-$role-$tag.sh"
  retry k exec -i "$pod" -- tee "/tmp/launch-$role-$tag.sh" > /dev/null < "$OUT/raw/launch-$role-$tag.sh"
  retry k exec "$pod" -- bash -c "chmod +x /tmp/launch-$role-$tag.sh; \
setsid /tmp/launch-$role-$tag.sh </dev/null > /tmp/server-$role-$tag.log 2>&1 &" </dev/null
}

start() {    # layout worker-pod-or-empty tag
  local layout=$1 worker=$2 tag=$3 t=0
  log "start $tag (layout $layout, second instance ${worker:-none})"
  case $layout in
    tp8) launch $HEAD head "$tag" "$COMMON --tensor-parallel-size $GPU_PER_NODE" ;;
    pp2tp8|tp16)
      local tp=$GPU_PER_NODE pp=2
      [ "$layout" = tp16 ] && { tp=$((2 * GPU_PER_NODE)); pp=1; }
      local multi="$COMMON --tensor-parallel-size $tp --pipeline-parallel-size $pp --nnodes 2 \
--master-addr $HEAD_IP --master-port $MPORT"
      launch "$worker" worker "$tag" "$multi --node-rank 1 --headless"; sleep 5
      launch $HEAD head "$tag" "$multi --node-rank 0" ;;
    *) log "unknown layout $layout"; return 1 ;;
  esac
  while [ $t -lt "$READY_MAX" ]; do
    if k exec $HEAD -- curl -sf -m 5 "localhost:$PORT/health" > /dev/null 2>&1; then log "  ready after ${t}s"; return 0; fi
    if ! retry k exec $HEAD -- pgrep -f '[v]llm serve' > /dev/null 2>&1; then log "  head server exited"; break; fi
    sleep 20; t=$((t + 20))
  done
  log "  FAILED to become ready"
  k exec $HEAD -- tail -30 "/tmp/server-head-$tag.log" >> "$OUT/run.log" 2>&1
  return 1
}

transport_ok() {   # tag worker-pod-or-empty: every NCCL process on both instances on EFA, none on sockets
  local tag=$1 worker=$2 pr pod role efa tcp ok=0
  for pr in "$HEAD:head" ${worker:+"$worker:worker"}; do
    pod=${pr%%:*}; role=${pr##*:}
    k exec "$pod" -- cat "/tmp/server-$role-$tag.log" > "$OUT/raw/server-$role-$tag.log" 2>/dev/null
    efa=$(grep -oE "[A-Za-z0-9.-]+:[0-9]+:[0-9]+ \[[0-9]+\] NCCL INFO Using network Libfabric" \
      "$OUT/raw/server-$role-$tag.log" | cut -d: -f1,2 | sort -u | wc -l | tr -d ' ')
    tcp=$(grep -c "Using network Socket" "$OUT/raw/server-$role-$tag.log")
    echo "$tag $role nccl_processes_on_efa=$efa socket_lines=$tcp" >> "$OUT/transport.txt"
    if [ -n "$worker" ] && { [ "$efa" -lt "$GPU_PER_NODE" ] || [ "$tcp" -gt 0 ]; }; then ok=1; fi
  done
  return $ok
}

bench() {    # layout round arm
  local layout=$1 r=$2 arm=$3 c np tag t
  for c in $CONCS; do
    np=$((c * 8)); [ "$np" -lt 16 ] && np=16
    tag="$layout-r$r-$arm-c$c"; t=0
    { echo '#!/bin/bash'; echo 'cd /tmp'
      echo "vllm bench serve --model model --tokenizer $MODEL --trust-remote-code --base-url http://localhost:$PORT \
--dataset-name random --random-input-len $INLEN --random-output-len $OUTLEN --max-concurrency $c --num-prompts $np \
--ignore-eos --percentile-metrics ttft,tpot,itl --metric-percentiles 50,99 --save-result \
--result-filename /tmp/bench-$tag.json > /tmp/bench-$tag.log 2>&1"
      echo "echo \$? > /tmp/bench-$tag.rc"; } > "$OUT/raw/bench-$tag.sh"
    retry k exec -i $HEAD -- tee "/tmp/bench-$tag.sh" > /dev/null < "$OUT/raw/bench-$tag.sh"
    retry k exec $HEAD -- bash -c "chmod +x /tmp/bench-$tag.sh; setsid /tmp/bench-$tag.sh </dev/null >/dev/null 2>&1 &" </dev/null
    while [ $t -lt "$BENCH_TMO" ]; do
      sleep 15; t=$((t + 15))
      k exec $HEAD -- test -f "/tmp/bench-$tag.rc" 2>/dev/null && break
    done
    retry k exec $HEAD -- cat "/tmp/bench-$tag.json" > "$OUT/raw/bench-$tag.json" 2>/dev/null
    python3 - "$OUT/raw/bench-$tag.json" "$layout" "$r" "$arm" "$c" >> "$OUT/serving.csv" <<'PY'
import json
import sys

path, layout, rnd, arm, conc = sys.argv[1:6]
keys = ["median_tpot_ms", "p99_tpot_ms", "median_itl_ms", "p99_itl_ms", "median_ttft_ms", "output_throughput"]
try:
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    vals = [f"{d[k]:.4f}" for k in keys] + [str(d["completed"])]
except (OSError, ValueError, KeyError):
    vals = ["NA"] * (len(keys) + 1)
print(",".join([layout, rnd, arm, conc] + vals))
PY
    tail -1 "$OUT/serving.csv" | sed 's/^/  /' | tee -a "$OUT/run.log"
  done
}

echo "layout,round,arm,concurrency,tpot_p50_ms,tpot_p99_ms,itl_p50_ms,itl_p99_ms,ttft_p50_ms,output_tps,completed" \
  > "$OUT/serving.csv"
for layout in $LAYOUTS; do
  if [ "$layout" = tp8 ]; then
    stop_all
    if start tp8 "" tp8-r1-one; then transport_ok tp8-r1-one ""; bench tp8 1 one; fi
    continue
  fi
  for r in $(seq 1 "$ROUNDS"); do
    if [ $((r % 2)) -eq 1 ]; then order="near far"; else order="far near"; fi
    for arm in $order; do
      worker=$NEAR; [ "$arm" = far ] && worker=$FAR
      stop_all
      if start "$layout" "$worker" "$layout-r$r-$arm"; then
        if transport_ok "$layout-r$r-$arm" "$worker"; then
          bench "$layout" "$r" "$arm"
        else
          log "  DISCARDED $layout-r$r-$arm: transport not proven on EFA"
        fi
      fi
    done
  done
done
stop_all
log "results: $OUT"
