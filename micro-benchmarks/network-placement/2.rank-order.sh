#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Does rank order matter when a job spans more than one layer-3 network node? NCCL numbers instances in the order
# their ranks appear and builds its ring between instances in that order, so the node order you give the launcher
# decides how many ring hops cross between layer-3 nodes. This script times NCCL collectives across the probe pods
# in two node orders, chosen from the network-node labels:
#   sorted:      instances grouped by layer-3 node (as a topology-sorted hostfile would give)
#   interleaved: neighbours under different layer-3 nodes wherever possible
# It launches torchrun with the static rendezvous and an explicit --node-rank, which keeps the order you pass
# (torchrun's c10d rendezvous assigns node ranks itself). The orders alternate within each round (ABBA), and a run
# is kept only if every NCCL process used EFA.
#
# Usage: NAMESPACE=default [PODS=pod-a,pod-b,pod-c,pod-d] ./2.rank-order.sh     then: python3 analyze_rank_order.py <dir>
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
NAMESPACE=${NAMESPACE:-default}; SELECTOR=${SELECTOR:-app=placement-probe}; PODS=${PODS:-}
ROUNDS=${ROUNDS:-4}; GPUS=${GPUS_PER_NODE:-8}; PORT=${PORT:-29611}; RUN_TIMEOUT=${RUN_TIMEOUT:-1800}
COLLS=${COLLS:-all_reduce,all_gather,all_to_all}; MAX_BYTES=${MAX_BYTES:-$((4 << 30))}
SOCKET_IFNAME=${SOCKET_IFNAME:-eth0}
NET_PLUGIN=${NET_PLUGIN:-/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so}
TUNER_PLUGIN=${TUNER_PLUGIN:-/opt/amazon/ofi-nccl/lib/libnccl-ofi-tuner.so}
OUT=${OUT:-./results/rank-order-$(date -u +%Y%m%dT%H%MZ)}
mkdir -p "$OUT/raw"
K="kubectl -n $NAMESPACE"
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$OUT/run.log"; }

$K get pods -l "$SELECTOR" -o json > "$OUT/.pods.json"
kubectl get nodes -o json > "$OUT/.nodes.json"
python3 "$HERE/topology.py" orders --nodes "$OUT/.nodes.json" --pods "$OUT/.pods.json" ${PODS:+--only "$PODS"} \
  > "$OUT/orders.txt" || exit 1
rm -f "$OUT/.pods.json" "$OUT/.nodes.json"
cat "$OUT/orders.txt"
read -r -a SORTED <<< "$(awk '$1 == "sorted" {for (i = 4; i <= NF; i++) printf "%s ", $i}' "$OUT/orders.txt")"
read -r -a INTERLEAVED <<< "$(awk '$1 == "interleaved" {for (i = 4; i <= NF; i++) printf "%s ", $i}' "$OUT/orders.txt")"
read -r XS NHOPS <<< "$(awk '$1 == "sorted" {print $2, $3}' "$OUT/orders.txt")"
XI=$(awk '$1 == "interleaved" {print $2}' "$OUT/orders.txt")
[ "$XS" -lt "$XI" ] || { echo "these pods give both orders the same number of layer-3 crossings ($XS of $NHOPS); \
pick pods from at least two layer-3 groups, with two or more pods in at least two of them"; exit 1; }
log "ring hops crossing layer 3: sorted $XS of $NHOPS, interleaved $XI of $NHOPS"

for p in "${SORTED[@]}"; do
  $K exec -i "$p" -- tee /tmp/bench_coll.py > /dev/null < "$HERE/bench_coll.py" || exit 1
done

ENVS="NCCL_NET_PLUGIN=$NET_PLUGIN NCCL_TUNER_PLUGIN=$TUNER_PLUGIN NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET \
NCCL_SOCKET_IFNAME=$SOCKET_IFNAME GLOO_SOCKET_IFNAME=$SOCKET_IFNAME PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8"
RUN=0
run_order() {   # tag, then pods in node-rank order
  local tag=$1; shift
  local first=$1 n=$# i=0 t=0 ok=1 master p
  master=$($K get pod "$first" -o jsonpath='{.status.podIP}')
  RUN=$((RUN + 1))
  log "run $tag order=$*"
  for p in "$@"; do
    {
      echo '#!/bin/bash'
      echo 'cd /tmp'
      for kv in $ENVS; do echo "export $kv"; done
      echo "exec torchrun --nnodes $n --nproc-per-node $GPUS --node-rank $i --master-addr $master \\"
      echo "  --master-port $((PORT + RUN)) /tmp/bench_coll.py --colls $COLLS --max-bytes $MAX_BYTES --tag $tag"
    } > "$OUT/raw/launch-$tag-$p.sh"
    $K exec -i "$p" -- tee "/tmp/run-$tag.sh" > /dev/null < "$OUT/raw/launch-$tag-$p.sh"
    $K exec "$p" -- bash -c "chmod +x /tmp/run-$tag.sh; setsid /tmp/run-$tag.sh </dev/null > /tmp/out-$tag.log 2>&1 &" \
      </dev/null >> "$OUT/run.log" 2>&1
    i=$((i + 1))
  done
  sleep 15
  while [ $t -lt "$RUN_TIMEOUT" ]; do
    local running=0
    for p in "$@"; do
      $K exec "$p" -- pgrep -f '[b]ench_coll.py' > /dev/null 2>&1 && running=1
    done
    [ $running -eq 0 ] && break
    sleep 15; t=$((t + 15))
  done
  for p in "$@"; do
    [ $t -ge "$RUN_TIMEOUT" ] && $K exec "$p" -- pkill -9 -f '[b]ench_coll.py' > /dev/null 2>&1
    $K exec "$p" -- cat "/tmp/out-$tag.log" > "$OUT/raw/out-$tag-$p.log" 2>/dev/null || ok=0
  done
  if [ $ok -eq 1 ] && "$HERE/check_efa_transport.sh" "$OUT"/raw/out-"$tag"-*.log $((n * GPUS)) | tee -a "$OUT/transport.txt"; then
    grep -E '^(all_reduce|all_gather|all_to_all),[0-9]+,[0-9]+,[0-9.]+,[0-9.]+,[0-9.]+$' "$OUT/raw/out-$tag-$first.log" \
      | sed "s/^/$tag,/" >> "$OUT/coll.csv"
    grep -m1 -E 'NCCL INFO Channel 00/[0-9]+ :' "$OUT/raw/out-$tag-$first.log" | sed "s/^/$tag /" >> "$OUT/rings.txt"
    for p in "$@"; do   # the lowest NCCL rank on each pod is node rank x GPUs, so this shows the order NCCL saw
      echo "$tag $p $(grep -oE 'NCCL INFO comm 0x[0-9a-f]+ rank [0-9]+ nRanks' "$OUT/raw/out-$tag-$p.log" \
        | awk '{print $6}' | sort -n | head -1)"
    done >> "$OUT/node-ranks.txt"
    log "  kept: $(grep -c "^$tag," "$OUT/coll.csv") rows"
  else
    log "  DISCARDED $tag: transport not proven on EFA, or a log is missing"
  fi
}

echo "tag,coll,bytes,iters,time_us,algbw_GBps,busbw_GBps" > "$OUT/coll.csv"
for r in $(seq 1 "$ROUNDS"); do
  if [ $((r % 2)) -eq 1 ]; then
    run_order "r$r-sorted" "${SORTED[@]}"; run_order "r$r-interleaved" "${INTERLEAVED[@]}"
  else
    run_order "r$r-interleaved" "${INTERLEAVED[@]}"; run_order "r$r-sorted" "${SORTED[@]}"
  fi
done
log "results: $OUT"
