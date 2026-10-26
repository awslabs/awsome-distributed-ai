#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Compare two ways to serve one model with two instances, with the same vLLM configuration (TP 8) everywhere:
#   replicas  one complete server per instance; the client sends half the prompts at half the concurrency to each
#   disagg    vllm-prefill prefills every prompt and vllm-decode decodes every answer; the KV cache of every
#             prompt moves from prefill to decode over EFA with NIXL's LIBFABRIC backend (vLLM's NixlConnector),
#             and vLLM's example proxy on vllm-decode sends each request to both
# The arms alternate within each round (ABBA). A disaggregated run counts only if every vLLM worker on both
# instances instantiated NIXL's LIBFABRIC backend and the decode workers connected to the prefill instance.
# kv_load_failure_policy=fail makes a failed KV transfer an error instead of a silent local re-prefill.
#
# WORKLOADS lists input:output tokens@concurrencies. Concurrencies must be even (the replicas arm splits them).
# Usage: source env_vars && ./2.benchmark.sh                 then: python3 3.analyze.py <result dir>
set -uo pipefail
: "${NAMESPACE:?source env_vars}" "${MODEL_REPO:?}" "${GPU_PER_NODE:?}" "${VLLM_COMMIT:?}" "${PROXY_SHA256:?}"
WORKLOADS=${WORKLOADS:-"2048:256@8,32,64 8192:256@8,32,64 100000:256@2,4,8"}
ROUNDS=${ROUNDS:-2}
MAXLEN=${MAXLEN:-102400}; GPUUTIL=${GPUUTIL:-0.90}
PROMPTS_PER_CONC=${PROMPTS_PER_CONC:-4}; MIN_PROMPTS=${MIN_PROMPTS:-8}
READY_MAX=${READY_MAX:-2400}; BENCH_TMO=${BENCH_TMO:-3600}
MODEL=/model/$(basename "$MODEL_REPO")
OUT=${OUT:-./results/disagg-vs-replicas-$(date -u +%Y%m%dT%H%MZ)}
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)-$$   # names this invocation's launch markers in the pods
mkdir -p "$OUT/raw"
k() { kubectl -n "$NAMESPACE" "$@"; }
retry() { local _; for _ in 1 2 3 4 5; do "$@" && return 0; sleep 5; done; return 1; }
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$OUT/run.log"; }
P=vllm-prefill; D=vllm-decode
PIP=$(k get pod $P -o jsonpath='{.status.podIP}'); DIP=$(k get pod $D -o jsonpath='{.status.podIP}')

# vLLM's example proxy for disaggregated serving, at the commit the image was built from, checked against a hash.
PROXY_URL=https://raw.githubusercontent.com/vllm-project/vllm/$VLLM_COMMIT/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py
k exec $D -- bash -c "curl -fsSL --retry 3 -o /tmp/toy_proxy_server.py $PROXY_URL && \
echo '$PROXY_SHA256  /tmp/toy_proxy_server.py' | sha256sum -c -" || { log "proxy download or checksum failed"; exit 1; }

{ echo "utc=$(date -u +%FT%TZ) model=$MODEL_REPO@${MODEL_REVISION:-} workloads=$WORKLOADS rounds=$ROUNDS"
  echo "max_model_len=$MAXLEN gpu_util=$GPUUTIL prompts_per_point=${PROMPTS_PER_CONC}x concurrency (min $MIN_PROMPTS)"
  k exec $D -- python3 -c 'import importlib.metadata as m, vllm; print("vllm", vllm.__version__, "nixl", m.version("nixl"))'
} > "$OUT/meta.txt" 2>&1; cat "$OUT/meta.txt"

ENVS="FI_PROVIDER=efa VLLM_NIXL_SIDE_CHANNEL_PORT=5600 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 \
VLLM_LOGGING_LEVEL=INFO NIXL_LOG_LEVEL=INFO"
KVCFG='{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_buffer_device":"cuda",'
KVCFG+='"kv_connector_extra_config":{"backends":["LIBFABRIC"],"kv_load_failure_policy":"fail"}}'
COMMON="$MODEL --served-model-name model --trust-remote-code --tensor-parallel-size $GPU_PER_NODE \
--no-enable-prefix-caching --max-model-len $MAXLEN --gpu-memory-utilization $GPUUTIL"

stop_all() {
  for p in $P $D; do
    k exec "$p" -- pkill -9 -f '[v]llm bench' >/dev/null 2>&1
    k exec "$p" -- pkill -9 -f '[t]oy_proxy' >/dev/null 2>&1
    k exec "$p" -- pkill -9 -f '[v]llm serve' >/dev/null 2>&1
    k exec "$p" -- pkill -9 -f '[V]LLM::' >/dev/null 2>&1
  done
  sleep 20
}

launch() {   # pod tag command extra-env: run detached inside the pod
  local pod=$1 tag=$2 cmd=$3 xenv=${4:-}
  { echo '#!/bin/bash'; echo 'cd /tmp'
    for kv in $ENVS $xenv; do echo "export $kv"; done
    echo "exec $cmd"; } > "$OUT/raw/launch-$tag.sh"
  retry k exec -i "$pod" -- tee "/tmp/launch-$tag.sh" > /dev/null < "$OUT/raw/launch-$tag.sh"
  # Launch once: if a kubectl exec stalls and is retried, the marker stops a second copy from starting.
  retry k exec "$pod" -- bash -c "chmod +x /tmp/launch-$tag.sh; if mkdir /tmp/started-$RUN_ID-$tag 2>/dev/null; \
then setsid /tmp/launch-$tag.sh </dev/null > /tmp/$tag.log 2>&1 & fi" </dev/null
}

ready() {    # pod port
  local t=0
  while [ $t -lt "$READY_MAX" ]; do
    k exec "$1" -- curl -sf -m 5 "localhost:$2/health" > /dev/null 2>&1 && { log "  $1 ready after ${t}s"; return 0; }
    sleep 20; t=$((t + 20))
  done
  return 1
}

start_arm() {   # arm tag
  local arm=$1 tag=$2
  if [ "$arm" = replicas ]; then
    launch $P "srv-$tag-p" "vllm serve $COMMON --port 8100"
    launch $D "srv-$tag-d" "vllm serve $COMMON --port 8100"
    ready $P 8100 && ready $D 8100
  else
    launch $P "srv-$tag-p" "vllm serve $COMMON --port 8100 --kv-transfer-config '$KVCFG'" "VLLM_NIXL_SIDE_CHANNEL_HOST=$PIP"
    launch $D "srv-$tag-d" "vllm serve $COMMON --port 8100 --kv-transfer-config '$KVCFG'" "VLLM_NIXL_SIDE_CHANNEL_HOST=$DIP"
    ready $P 8100 && ready $D 8100 || return 1
    launch $D "proxy-$tag" "python3 /tmp/toy_proxy_server.py --port 8000 --prefiller-hosts $PIP --prefiller-ports 8100 \
--decoder-hosts localhost --decoder-ports 8100"
    sleep 15
  fi
}

nixl_ok() {   # tag: LIBFABRIC on every worker of both instances, and decode workers connected to the prefill peer
  local tag=$1 p fab conn ok=0
  for p in $P $D; do
    k exec "$p" -- bash -c "cat /tmp/srv-$tag-*.log" > "$OUT/raw/srv-$tag-$p.log" 2>/dev/null
    fab=$(grep -c "Backend LIBFABRIC was instantiated" "$OUT/raw/srv-$tag-$p.log")
    conn=$(grep -c "Successfully created connection" "$OUT/raw/srv-$tag-$p.log")
    echo "$tag $p libfabric_workers=$fab connections=$conn" | tee -a "$OUT/transport.txt"
    [ "$fab" -ge "$GPU_PER_NODE" ] || ok=1
    if [ "$p" = "$D" ] && [ "$conn" -lt "$GPU_PER_NODE" ]; then ok=1; fi
  done
  return $ok
}

bg_bench() {    # pod name command: run detached in the pod, poll, copy back the log and the result
  local pod=$1 name=$2 cmd=$3 t=0
  { echo '#!/bin/bash'; echo 'cd /tmp'; echo "$cmd > /tmp/$name.log 2>&1"; echo "echo \$? > /tmp/$name.rc"; } \
    > "$OUT/raw/$name.sh"
  retry k exec -i "$pod" -- tee "/tmp/$name.sh" > /dev/null < "$OUT/raw/$name.sh"
  retry k exec "$pod" -- bash -c "chmod +x /tmp/$name.sh; if mkdir /tmp/started-$RUN_ID-$name 2>/dev/null; \
then rm -f /tmp/$name.rc /tmp/$name.json; setsid /tmp/$name.sh </dev/null >/dev/null 2>&1 & fi" </dev/null
  while [ $t -lt "$BENCH_TMO" ]; do
    sleep 15; t=$((t + 15))
    k exec "$pod" -- test -f "/tmp/$name.rc" 2>/dev/null && break
  done
  [ $t -ge "$BENCH_TMO" ] && { log "  TIMEOUT $name"; k exec "$pod" -- pkill -9 -f '[v]llm bench' > /dev/null 2>&1; }
  retry k exec "$pod" -- cat "/tmp/$name.log" > "$OUT/raw/$name.log" 2>/dev/null
  retry k exec "$pod" -- cat "/tmp/$name.json" > "$OUT/raw/$name.json" 2>/dev/null
}

bench_one() {   # arm round input output concurrency
  local arm=$1 r=$2 in=$3 outl=$4 c=$5 tag="$1-r$2-i$3-c$5"
  local np=$((c * PROMPTS_PER_CONC)); [ "$np" -lt "$MIN_PROMPTS" ] && np=$MIN_PROMPTS
  local b="vllm bench serve --model model --tokenizer $MODEL --trust-remote-code --dataset-name random \
--random-input-len $in --random-output-len $outl --ignore-eos --percentile-metrics ttft,tpot,itl \
--metric-percentiles 50,99 --save-result"
  log "  bench $tag prompts=$np"
  if [ "$arm" = replicas ]; then
    bg_bench $P "b-$tag-p" "$b --base-url http://localhost:8100 --max-concurrency $((c / 2)) --num-prompts $((np / 2)) \
--result-filename /tmp/b-$tag-p.json" &
    bg_bench $D "b-$tag-d" "$b --base-url http://localhost:8100 --max-concurrency $((c / 2)) --num-prompts $((np / 2)) \
--result-filename /tmp/b-$tag-d.json"
    wait
  else
    bg_bench $D "b-$tag" "$b --base-url http://localhost:8000 --max-concurrency $c --num-prompts $np \
--result-filename /tmp/b-$tag.json"
  fi
  python3 - "$OUT" "$tag" "$arm" "$r" "$in" "$c" "$np" >> "$OUT/serving.csv" <<'PY'
import json
import sys

out, tag, arm, rnd, inl, conc, np_ = sys.argv[1:8]
if arm == "replicas":
    files = [f"{out}/raw/b-{tag}-p.json", f"{out}/raw/b-{tag}-d.json"]
else:
    files = [f"{out}/raw/b-{tag}.json"]
keys = ["median_ttft_ms", "p99_ttft_ms", "median_tpot_ms", "p99_tpot_ms", "median_itl_ms", "p99_itl_ms"]
try:
    ds = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            ds.append(json.load(fh))
    tps = sum(d["output_throughput"] for d in ds)
    # Replicas: the mean of the two halves' percentiles.
    vals = [tps] + [sum(d[k] for d in ds) / len(ds) for k in keys] + [max(d["duration"] for d in ds)]
    row = [f"{v:.4f}" for v in vals] + [str(sum(d["completed"] for d in ds))]
except (OSError, ValueError, KeyError):
    row = ["NA"] * (len(keys) + 3)
print(",".join([inl, rnd, arm, conc, np_] + row))
PY
  tail -1 "$OUT/serving.csv" | sed 's/^/  /' | tee -a "$OUT/run.log"
}

echo "workload,round,arm,concurrency,prompts,output_tps,ttft_p50_ms,ttft_p99_ms,tpot_p50_ms,tpot_p99_ms,\
itl_p50_ms,itl_p99_ms,duration_s,completed" > "$OUT/serving.csv"
for r in $(seq 1 "$ROUNDS"); do
  if [ $((r % 2)) -eq 1 ]; then order="disagg replicas"; else order="replicas disagg"; fi
  for arm in $order; do
    stop_all; log "start $arm r$r"
    if ! start_arm "$arm" "$arm-r$r"; then log "  FAILED to start $arm r$r"; continue; fi
    for w in $WORKLOADS; do
      io=${w%@*}; cs=${w#*@}
      for c in ${cs//,/ }; do bench_one "$arm" "$r" "${io%%:*}" "${io##*:}" "$c"; done
    done
    if [ "$arm" = disagg ] && ! nixl_ok "$arm-r$r"; then
      log "  DISCARDED $arm-r$r: NIXL LIBFABRIC transfer not proven"
      sed -i.bak "/^[0-9]*,$r,disagg,/d" "$OUT/serving.csv" && rm -f "$OUT/serving.csv.bak"
    fi
  done
done
stop_all
log "results: $OUT"
