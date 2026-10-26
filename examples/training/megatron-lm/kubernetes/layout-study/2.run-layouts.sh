#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Train the same GPT model in each parallelism layout and placement, and record step time and throughput. The
# global batch, and so the work per step, is the same in every layout (8-GPU instances):
#   tp8-dp2    tensor parallelism (TP) 8 inside each of two instances, data parallelism (DP) 2 across them
#   tp8-pp2    TP 8 inside each instance, pipeline parallelism (PP) 2 across them, one stage per instance
#   tp16       TP 16 across both instances
#   tp8-split  one TP 8 group split 4 + 4 across two instances, DP 1
#   tp8        TP 8 on the head instance alone, DP 1: the baseline for tp8-split
#   tp8-half   TP 8 on the head instance alone with half the attention heads, feed-forward width and vocabulary,
#              so each GPU does tp16's per-GPU work without crossing instances: the baseline for tp16
# Two-instance layouts run with megatron-near or megatron-far as the second instance. ARMS lists
# layout:placement:sequence-length, and even rounds run the list backwards, so a drift during the run can't favour
# one arm. A two-instance run counts only if every rank's process on both instances reports "Using network
# Libfabric" and none reports the TCP socket fallback.
#
# Usage: source env_vars && ./2.run-layouts.sh           then: python3 3.analyze.py <result dir>
set -uo pipefail
: "${NAMESPACE:?source env_vars}" "${GPU_PER_NODE:?}"
[ "$GPU_PER_NODE" -eq 8 ] || { echo "the layouts assume 8 GPUs per instance"; exit 1; }
ARMS=${ARMS:-"tp8-dp2:near:4096 tp8-pp2:near:4096 tp16:near:4096 tp8-split:near:4096 tp8-split:near:8192 \
tp8-half:one:4096 tp8:one:4096 tp8:one:8192 tp8-split:far:8192 tp16:far:4096"}
ROUNDS=${ROUNDS:-2}
NUM_LAYERS=${NUM_LAYERS:-24}; HIDDEN_SIZE=${HIDDEN_SIZE:-4096}; NUM_HEADS=${NUM_HEADS:-32}
FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-$((4 * HIDDEN_SIZE))}; VOCAB_SIZE=${VOCAB_SIZE:-32000}
GLOBAL_BATCH=${GLOBAL_BATCH:-16}; TRAIN_ITERS=${TRAIN_ITERS:-80}
RUN_TMO=${RUN_TMO:-1200}; MPORT=${MPORT:-29500}
MEGATRON_DIR=${MEGATRON_DIR:-/workspace/Megatron-LM}
NET_PLUGIN=${NET_PLUGIN:-/opt/amazon/aws-ofi-nccl/lib/libnccl-net-ofi.so}
SOCKET_IFNAME=${SOCKET_IFNAME:-eth0}
EXTRA_ENV=${EXTRA_ENV:-}   # space-separated NAME=value pairs added to every rank, for example NCCL_ALGO=...
OUT=${OUT:-./results/layouts-$(date -u +%Y%m%dT%H%MZ)}
mkdir -p "$OUT/raw"
k() { kubectl -n "$NAMESPACE" "$@"; }
HEAD=megatron-head; NEAR=megatron-near; FAR=megatron-far
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


near = shared("megatron-head", "megatron-near")
far = shared("megatron-head", "megatron-far")
print(f"layers shared with megatron-head: megatron-near {near}, megatron-far {far}")
sys.exit(0 if (near, far) == (3, 2) else 1)
PY
rm -f "$OUT/.nodes.json" "$OUT/.pods.txt"; cat "$OUT/placement.txt"

ENVS="NCCL_NET_PLUGIN=$NET_PLUGIN NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET NCCL_SOCKET_IFNAME=$SOCKET_IFNAME \
GLOO_SOCKET_IFNAME=$SOCKET_IFNAME CUDA_DEVICE_MAX_CONNECTIONS=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8"
{ echo "utc=$(date -u +%FT%TZ) rounds=$ROUNDS arms=$ARMS"
  echo "layers=$NUM_LAYERS hidden=$HIDDEN_SIZE heads=$NUM_HEADS ffn=$FFN_HIDDEN_SIZE vocab=$VOCAB_SIZE" \
    "global_batch=$GLOBAL_BATCH train_iters=$TRAIN_ITERS extra_env=$EXTRA_ENV"
  k exec $HEAD -- bash -c "cd $MEGATRON_DIR && python3 -c 'import torch, megatron.core as m; \
print(\"megatron-core\", m.__version__, \"torch\", torch.__version__, \"nccl\", torch.cuda.nccl.version())'"
} > "$OUT/meta.txt" 2>&1; cat "$OUT/meta.txt"

layout_cfg() {   # layout placement: sets PODS GPN TP PP MODEL
  local second=$NEAR
  [ "$2" = far ] && second=$FAR
  GPN=8; PP=1
  MODEL="--num-layers $NUM_LAYERS --hidden-size $HIDDEN_SIZE --num-attention-heads $NUM_HEADS \
--ffn-hidden-size $FFN_HIDDEN_SIZE --vocab-size $VOCAB_SIZE"
  case $1:$2 in
    tp8-dp2:near | tp8-dp2:far) PODS="$HEAD $second"; TP=8 ;;
    tp8-pp2:near | tp8-pp2:far) PODS="$HEAD $second"; TP=8; PP=2 ;;
    tp16:near | tp16:far) PODS="$HEAD $second"; TP=16 ;;
    tp8-split:near | tp8-split:far) PODS="$HEAD $second"; TP=8; GPN=4 ;;
    tp8:one) PODS=$HEAD; TP=8 ;;
    tp8-half:one)
      PODS=$HEAD; TP=8
      MODEL="--num-layers $NUM_LAYERS --hidden-size $HIDDEN_SIZE --num-attention-heads $((NUM_HEADS / 2)) \
--kv-channels $((HIDDEN_SIZE / NUM_HEADS)) --ffn-hidden-size $((FFN_HIDDEN_SIZE / 2)) \
--vocab-size $((VOCAB_SIZE / 2))" ;;
    *) log "unknown arm $1:$2"; return 1 ;;
  esac
}

alive() {   # pod: is a training process still running? (unknown counts as running)
  local out
  out=$(retry k exec "$1" -- bash -c 'pgrep -f "[p]retrain_gpt.py" > /dev/null; echo "rc=$?"' 2>/dev/null)
  case $out in *rc=1*) return 1 ;; *) return 0 ;; esac
}

RUN_NO=0
run_arm() {   # round layout placement seqlen
  local r=$1 layout=$2 placement=$3 seqlen=$4
  layout_cfg "$layout" "$placement" || return 1
  local pods n master cvd dp tag i=0 p t=0 port
  read -r -a pods <<< "$PODS"
  n=${#pods[@]}; dp=$((n * GPN / (TP * PP))); tag="r$r-$layout-$placement-s$seqlen"
  master=$(k get pod "${pods[0]}" -o jsonpath='{.status.podIP}')
  cvd=$(seq -s, 0 $((GPN - 1))); port=$((MPORT + RUN_NO % 100)); RUN_NO=$((RUN_NO + 1))
  log "run $tag: ${pods[*]} tp=$TP pp=$PP dp=$dp gpus_per_instance=$GPN"
  for p in "${pods[@]}"; do
    { echo '#!/bin/bash'; echo 'cd /tmp'
      for kv in $ENVS $EXTRA_ENV; do echo "export $kv"; done
      echo "export CUDA_VISIBLE_DEVICES=$cvd"
      echo "exec torchrun --nnodes $n --nproc-per-node $GPN --node-rank $i --master-addr $master \\"
      echo "  --master-port $port $MEGATRON_DIR/pretrain_gpt.py \\"
      echo "  --tensor-model-parallel-size $TP --pipeline-model-parallel-size $PP $MODEL \\"
      echo "  --seq-length $seqlen --max-position-embeddings $seqlen --micro-batch-size 1 \\"
      echo "  --global-batch-size $GLOBAL_BATCH --train-iters $TRAIN_ITERS --lr 1.5e-4 --min-lr 1.0e-5 \\"
      echo "  --lr-decay-style cosine --lr-warmup-iters 5 --weight-decay 0.1 --clip-grad 1.0 --bf16 \\"
      echo "  --tokenizer-type NullTokenizer --mock-data --log-interval 5 --eval-iters 0 \\"
      echo "  --distributed-timeout-minutes 6"
    } > "$OUT/raw/launch-$tag-$p.sh"
    retry k exec -i "$p" -- tee "/tmp/launch-$tag.sh" > /dev/null < "$OUT/raw/launch-$tag-$p.sh"
    retry k exec "$p" -- bash -c "chmod +x /tmp/launch-$tag.sh; \
setsid /tmp/launch-$tag.sh </dev/null > /tmp/train-$tag.log 2>&1 &" </dev/null
    i=$((i + 1))
  done
  while [ $t -lt "$RUN_TMO" ]; do
    sleep 20; t=$((t + 20))
    local any=0
    for p in "${pods[@]}"; do alive "$p" && any=1; done
    [ $any = 0 ] && break
  done
  [ $t -ge "$RUN_TMO" ] && log "  TIMEOUT after ${RUN_TMO}s"
  for p in "${pods[@]}"; do
    k exec "$p" -- pkill -9 -f '[p]retrain_gpt.py' > /dev/null 2>&1
    retry k exec "$p" -- cat "/tmp/train-$tag.log" > "$OUT/raw/train-$tag-$p.log" 2>/dev/null
  done
  sleep 10
  python3 - "$OUT" "$tag" "$r" "$layout" "$placement" "$seqlen" "$n" "$GPN" "$TP" "$PP" "$dp" "$GLOBAL_BATCH" \
    "${pods[@]}" >> "$OUT/training.csv" <<'PY'
import re
import statistics
import sys

out, tag, rnd, layout, placement, seq, n, gpn, tp, pp, dp, gbs, *pods = sys.argv[1:]
logs = []
for pod in pods:
    with open(f"{out}/raw/train-{tag}-{pod}.log", encoding="utf-8", errors="replace") as fh:
        logs.append(fh.read())
# Megatron logs step times from the last rank, which is on the last instance.
times = []
for text in logs:
    t = re.findall(r"elapsed time per iteration \(ms\):\s*([\d.]+)", text)
    if len(t) > len(times):
        times = [float(x) for x in t]
times = times[1:]  # the first logged interval includes warm-up
procs = int(n) * int(gpn)
efa_re = r"(\S+:\d+):\d+ \[\d+\] NCCL INFO Using network Libfabric"
efa = sum(len(set(re.findall(efa_re, text))) for text in logs)
sock = sum(text.count("Using network Socket") for text in logs)
if int(n) == 1:
    transport = "one-instance"
elif efa >= procs and sock == 0:
    transport = "EFA"
else:
    transport = f"NOT_PROVEN(efa_processes={efa};socket_lines={sock})"
tokens = int(gbs) * int(seq)
step = tps = "NA"
if times and transport in ("EFA", "one-instance"):
    med = statistics.median(times[-8:])
    step, tps = f"{med:.1f}", f"{tokens / (med / 1000):.0f}"
row = [rnd, layout, placement, seq, n, str(procs), tp, pp, dp, gbs, str(tokens), step, tps, transport]
print(",".join([*row, str(len(times))]))
PY
  tail -1 "$OUT/training.csv" | sed 's/^/  /' | tee -a "$OUT/run.log"
}

echo "round,layout,placement,seq_len,instances,gpus,tp,pp,dp,global_batch,tokens_per_step,step_ms,tokens_per_sec,\
transport,intervals" > "$OUT/training.csv"
for r in $(seq 1 "$ROUNDS"); do
  list=$ARMS
  [ $((r % 2)) -eq 0 ] && list=$(echo "$ARMS" | awk '{for (i = NF; i > 0; i--) printf "%s ", $i}')
  for arm in $list; do
    IFS=: read -r layout placement seqlen <<< "$arm"
    run_arm "$r" "$layout" "$placement" "$seqlen"
  done
done
log "results: $OUT"
