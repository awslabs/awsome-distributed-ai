#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Measure one-way EFA latency between every pair of probe pods, and a within-host floor on every pod, with
# fi_pingpong from libfabric. Pairs are labelled with the number of network-node layers their nodes share
# (3 = same layer-3 network node), so you can see what each layer costs on your instances.
#
#   - The same EFA device name is used on every host (DEV), so device-to-device differences don't pose as distance.
#   - The floor runs a server on DEV and a client on a second device (DEV2) in the same pod: no traffic leaves the
#     instance, so a slow host can't look like a far one. FI_EFA_ENABLE_SHM_TRANSFER=0 keeps it on the EFA path.
#   - Each pair runs ROUNDS times per message size and the first round is discarded as warm-up. Pair order is
#     shuffled with a fixed SEED, so near and far pairs interleave in time.
#
# Usage: NAMESPACE=default ./1.efa-latency-matrix.sh            then: python3 analyze_latency.py <result dir>
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
NAMESPACE=${NAMESPACE:-default}; SELECTOR=${SELECTOR:-app=placement-probe}
SIZES=${SIZES:-"8 1024"}; ITERS=${ITERS:-2000}; ROUNDS=${ROUNDS:-4}; FLOOR_ROUNDS=${FLOOR_ROUNDS:-3}; SEED=${SEED:-42}
FI_BIN=${FI_BIN:-/opt/amazon/efa/bin}
OUT=${OUT:-./results/latency-$(date -u +%Y%m%dT%H%MZ)}
mkdir -p "$OUT/raw"
K="kubectl -n $NAMESPACE"
FIENV="FI_EFA_ENABLE_SHM_TRANSFER=0"

$K get pods -l "$SELECTOR" -o json > "$OUT/.pods.json"
kubectl get nodes -o json > "$OUT/.nodes.json"
python3 "$HERE/topology.py" groups --nodes "$OUT/.nodes.json" --pods "$OUT/.pods.json" > "$OUT/groups.txt" || exit 1
python3 "$HERE/topology.py" pairs --nodes "$OUT/.nodes.json" --pods "$OUT/.pods.json" --seed "$SEED" > "$OUT/pairs.txt" || exit 1
rm -f "$OUT/.pods.json" "$OUT/.nodes.json"
cat "$OUT/groups.txt"
PODS=$(awk '{print $1; print $2}' "$OUT/pairs.txt" | sort -u); FIRST=$(echo "$PODS" | head -1)
[ "$(echo "$PODS" | grep -c .)" -ge 2 ] || { echo "need at least 2 running probe pods ($SELECTOR)"; exit 1; }

# Pick two EFA RDM domains on the first pod; every pod must expose both under the same names.
domains() { $K exec "$1" -- "$FI_BIN/fi_info" -p efa 2>/dev/null | awk '/domain:/ && /-rdm/ {print $2}' | sort -u; }
DEV=${DEV:-$(domains "$FIRST" | sed -n 1p)}; DEV2=${DEV2:-$(domains "$FIRST" | sed -n 2p)}
[ -n "$DEV" ] && [ -n "$DEV2" ] || { echo "could not read two EFA RDM domains with fi_info in $FIRST"; exit 1; }
: > "$OUT/.ips"
for p in $PODS; do
  d=$(domains "$p")
  echo "$d" | grep -qx "$DEV" && echo "$d" | grep -qx "$DEV2" || { echo "$p does not expose $DEV and $DEV2"; exit 1; }
  echo "$p $($K get pod "$p" -o jsonpath='{.status.podIP}')" >> "$OUT/.ips"
done
ip_of() { awk -v p="$1" '$1 == p {print $2; exit}' "$OUT/.ips"; }
{ echo "utc=$(date -u +%FT%TZ) pods=$(echo "$PODS" | grep -c .) sizes=$SIZES iters=$ITERS rounds=$ROUNDS (round 1 discarded) seed=$SEED"
  echo "fi_pingpong -p efa -e rdm -d $DEV -S <size> -I $ITERS ($FIENV); floor client device $DEV2"
  $K exec "$FIRST" -- "$FI_BIN/fi_info" --version 2>/dev/null | head -2; } > "$OUT/meta.txt"
cat "$OUT/meta.txt"

pingpong() {   # server-pod server-dev client-pod client-dev size tag -> prints usec/xfer or NA
  local sp=$1 sd=$2 cp=$3 cd=$4 sz=$5 tag=$6 u
  $K exec "$sp" -- env $FIENV timeout 120 "$FI_BIN/fi_pingpong" -p efa -e rdm -d "$sd" -S "$sz" -I "$ITERS" \
    > "$OUT/raw/srv-$tag.out" 2>&1 &
  local srv=$!; sleep 3
  $K exec "$cp" -- env $FIENV timeout 60 "$FI_BIN/fi_pingpong" -p efa -e rdm -d "$cd" -S "$sz" -I "$ITERS" "$(ip_of "$sp")" \
    > "$OUT/raw/cli-$tag.out" 2>&1
  wait "$srv" 2>/dev/null
  $K exec "$sp" -- pkill -f '[f]i_pingpong' >/dev/null 2>&1
  u=$(awk -f "$HERE/pingpong_usec.awk" "$OUT/raw/cli-$tag.out"); echo "${u:-NA}"
}

echo "pod,server_dev,client_dev,size_bytes,round,usec_per_xfer" > "$OUT/floor.csv"
for p in $PODS; do
  for sz in $SIZES; do
    for ((r = 1; r <= FLOOR_ROUNDS; r++)); do
      echo "$p,$DEV,$DEV2,$sz,$r,$(pingpong "$p" "$DEV" "$p" "$DEV2" "$sz" "${p}_floor_s${sz}_r$r")" | tee -a "$OUT/floor.csv"
    done
  done
done

echo "order,pod_a,pod_b,shared_layers,size_bytes,round,usec_per_xfer,discarded" > "$OUT/latency.csv"
n=0
while read -r a b depth; do
  n=$((n + 1))
  for sz in $SIZES; do
    for ((r = 1; r <= ROUNDS; r++)); do
      u=$(pingpong "$a" "$DEV" "$b" "$DEV" "$sz" "${a}__${b}_s${sz}_r$r")
      echo "$n,$a,$b,$depth,$sz,$r,$u,$([ $r -eq 1 ] && echo yes || echo no)" | tee -a "$OUT/latency.csv"
    done
  done
done < "$OUT/pairs.txt"
rm -f "$OUT/.ips"
echo "results: $OUT"
