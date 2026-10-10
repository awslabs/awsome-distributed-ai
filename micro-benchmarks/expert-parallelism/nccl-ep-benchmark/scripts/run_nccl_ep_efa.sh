#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# run_nccl_ep_efa.sh - NCCL EP dispatch/combine over EFA on two pods: correctness first, then a
# timing loop, with transport evidence and a clean teardown. Runs where kubectl runs.
#
#   run_nccl_ep_efa.sh up     check the namespace exists, render ../kubernetes/nccl-ep-2node.yaml, refuse busy
#                             nodes, apply, wait
#   run_nccl_ep_efa.sh run    preflight, ep_test HT+FLAT (correctness), ep_bench (timing), checks
#   run_nccl_ep_efa.sh down   delete the two pods and confirm they are gone
#   run_nccl_ep_efa.sh all    up, run, down (down also runs after a failed run, once harvested)
#
# Required: KUBE_CONTEXT NAMESPACE (an existing namespace; this script never creates or deletes one);
#           for up/all also IMAGE_URI NODE_A NODE_B (kubernetes/env_vars.example)
# Optional: INSTANCE_TYPE GPU_PER_NODE EFA_PER_NODE BUILD_IN_POD OUT_ROOT SSH_PORT MPI_CIDR
#           EP_TEST_TIMEOUT EP_BENCH_TIMEOUT EP_BENCH_ARGS KEEP_PODS
# Exit 0 only when every required gate in RESULT.json passes.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
EXAMPLE="$(cd "$HERE/.." && pwd)"
: "${KUBE_CONTEXT:?set KUBE_CONTEXT (kubernetes/env_vars.example)}"
: "${NAMESPACE:?set NAMESPACE}"
export NAMESPACE
export INSTANCE_TYPE="${INSTANCE_TYPE:-p5en.48xlarge}"
export GPU_PER_NODE="${GPU_PER_NODE:-8}"
export EFA_PER_NODE="${EFA_PER_NODE:-16}"
BUILD_IN_POD="${BUILD_IN_POD:-0}"
SSH_PORT="${SSH_PORT:-2222}"
EP_TEST_TIMEOUT="${EP_TEST_TIMEOUT:-1200}"
EP_BENCH_TIMEOUT="${EP_BENCH_TIMEOUT:-1800}"
EP_BENCH_ARGS="${EP_BENCH_ARGS:---algorithm ht --layout flat --tokens 4096 --hidden 7168 --top-k 8 --experts 256 --warmup 10 --iters 50 --validate}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT_ROOT:-$PWD/results}/nccl-ep-efa-$STAMP"
P0=nccl-ep-0
P1=nccl-ep-1
RD="/work/nccl-ep-run/$STAMP"           # per-run directory inside both pods
K=(kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE")
NP=$((2 * GPU_PER_NODE))

[ -e "$OUT" ] && { echo "$OUT exists; never reused" >&2; exit 2; }
mkdir -p "$OUT"
# Bind every result to the exact recipe files that produced it (and the commit, when in git). In a git checkout the
# recipe is the tracked files, so caches and earlier results never enter the digest; otherwise every file except
# env_vars, __pycache__/ and results/ at any depth.
if git -C "$EXAMPLE" rev-parse --is-inside-work-tree > /dev/null 2>&1; then
  (cd "$EXAMPLE" && git ls-files -z -- . | sed -z 's|^|./|' | sort -z | xargs -0 sha256sum)
else
  (cd "$EXAMPLE" && find . -type f ! -path '*/results/*' ! -path '*/__pycache__/*' ! -name env_vars -print0 | sort -z | xargs -0 sha256sum)
fi > "$OUT/example-files.sha256"
if git -C "$EXAMPLE" rev-parse HEAD > "$OUT/example-git-head.txt" 2>/dev/null; then
  git -C "$EXAMPLE" status --porcelain -- . >> "$OUT/example-git-head.txt"
fi
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$OUT/run.log"; }
die() { log "STOP: $*"; exit 1; }
X() { local p="$1"; shift; "${K[@]}" exec "$p" -- "$@"; }
XR() { local p="$1"; shift; "${K[@]}" exec "$p" -- env RUN_DIR="$RD" bash "$RD/pod_tools.sh" "$@"; }
check() { python3 -B "$HERE/check_run.py" "$@"; }   # -B: no __pycache__ beside the recipe

cmd_up() {
  : "${IMAGE_URI:?set IMAGE_URI}" "${NODE_A:?set NODE_A}" "${NODE_B:?set NODE_B}"
  export IMAGE_URI NODE_A NODE_B
  [ "$NODE_A" != "$NODE_B" ] || die "NODE_A and NODE_B must be two different nodes"
  # The manifest holds only the two pods; the namespace must already exist (kubectl's own error is printed above).
  kubectl --context "$KUBE_CONTEXT" get namespace "$NAMESPACE" -o name > /dev/null \
    || die "namespace $NAMESPACE not readable or not found; if missing: kubectl --context $KUBE_CONTEXT create namespace $NAMESPACE"
  local n used
  for n in "$NODE_A" "$NODE_B"; do
    used=$(kubectl --context "$KUBE_CONTEXT" get pods -A --field-selector "spec.nodeName=$n" -o json | check gpu-requests) \
      || die "could not read pods on $n"
    [ "$used" = 0 ] || die "node $n has $used GPUs requested by live pods; pick GPU-free nodes (never preempt)"
  done
  if "${K[@]}" get pod "$P0" "$P1" -o name 2>/dev/null | grep . > /dev/null; then
    die "pods already exist; run 'down' first (every run starts on fresh pods)"
  fi
  # shellcheck disable=SC2016  # the variable list is for envsubst, not the shell
  envsubst '$NAMESPACE $IMAGE_URI $INSTANCE_TYPE $NODE_A $NODE_B $GPU_PER_NODE $EFA_PER_NODE' \
    < "$EXAMPLE/kubernetes/nccl-ep-2node.yaml" > "$OUT/rendered.yaml"
  kubectl --context "$KUBE_CONTEXT" apply -f "$OUT/rendered.yaml" | tee -a "$OUT/run.log"
  "${K[@]}" wait --for=condition=Ready "pod/$P0" "pod/$P1" --timeout=30m || die "pods not Ready within 30 min"
  log "up: $P0 and $P1 Ready"
}

stage_scripts() {
  local p f
  for p in "$P0" "$P1"; do
    X "$p" bash -c "test ! -e '$RD' && mkdir -p '$RD/out'" || die "$p: $RD exists (never reused) or cannot be created"
    for f in pod_tools.sh efa_counters.sh sshd_setup.sh; do
      "${K[@]}" cp "$HERE/$f" "$p:$RD/$f" || die "$p: staging $f failed"
    done
    if [ "$BUILD_IN_POD" = 1 ]; then
      "${K[@]}" cp "$EXAMPLE/setup_nccl_ep_efa.sh" "$p:$RD/setup_nccl_ep_efa.sh" || die "$p: staging setup script failed"
    fi
  done
}

preflight() {
  local p ip0 ip1
  for p in "$P0" "$P1"; do "${K[@]}" get pod "$p" -o json > "$OUT/pod-$p.json" || die "cannot read pod $p"; done
  read -r ip0 ip1 < <(check pods --dir "$OUT" --p0 "$P0" --p1 "$P1") || die "pod check failed (run.log)"
  IP0="$ip0"; IP1="$ip1"
  MPI_CIDR="${MPI_CIDR:-$(echo "$IP0" | awk -F. '{print $1"."$2".0.0/16"}')}"
  log "pods: $P0=$IP0 $P1=$IP1 MPI_CIDR=$MPI_CIDR run_dir=$RD"
  stage_scripts
  if [ "$BUILD_IN_POD" = 1 ]; then
    for p in "$P0" "$P1"; do
      if ! X "$p" test -f /opt/nccl-ep/BUILD-RECORD.txt; then
        log "$p: building in-pod with setup_nccl_ep_efa.sh all (log: build-$p.log)"
        ( X "$p" env SETUP_WORK_DIR=/work/nccl-ep-setup bash "$RD/setup_nccl_ep_efa.sh" all > "$OUT/build-$p.log" 2>&1
          echo $? > "$OUT/build-$p.exit" ) &
      fi
    done
    wait
    for p in "$P0" "$P1"; do
      if [ -f "$OUT/build-$p.exit" ] && [ "$(cat "$OUT/build-$p.exit")" != 0 ]; then die "$p: in-pod build failed (build-$p.log)"; fi
    done
  fi
  for p in "$P0" "$P1"; do XR "$p" preflight > "$OUT/preflight-$p.txt" 2>&1 || die "$p: preflight failed"; done
  check preflight --dir "$OUT" --p0 "$P0" --p1 "$P1" --gpus "$GPU_PER_NODE" --efa "$EFA_PER_NODE" \
    --build-in-pod "$BUILD_IN_POD" | tee -a "$OUT/run.log"
  [ "${PIPESTATUS[0]}" = 0 ] || die "preflight gates failed (preflight-*.txt)"
}

sshd_up() {
  local pub
  pub=$(X "$P0" env RUN_DIR="$RD" bash "$RD/sshd_setup.sh" keygen) || die "keygen failed"
  X "$P1" env RUN_DIR="$RD" bash "$RD/sshd_setup.sh" server "$IP1" "$SSH_PORT" "$IP0" "$pub" > "$OUT/sshd-server.txt" 2>&1 \
    || die "sshd on $P1 failed (sshd-server.txt)"
  X "$P0" env RUN_DIR="$RD" bash "$RD/sshd_setup.sh" check "$IP1" "$SSH_PORT" > "$OUT/sshd-check.txt" 2>&1 \
    || die "ssh round trip failed (sshd-check.txt)"
  log "sshd: $(tail -1 "$OUT/sshd-check.txt")"
}

counters() { # $1 tag  $2 before|after
  local p
  for p in "$P0" "$P1"; do
    X "$p" env RUN_DIR="$RD" bash "$RD/efa_counters.sh" "$2-$1" > "$OUT/$1/efa-$2-$p.json" 2>/dev/null \
      || log "WARN: $p EFA counter read failed ($2-$1)"
  done
}

harvest() { # $1 tag
  local p
  for p in "$P0" "$P1"; do
    mkdir -p "$OUT/$1/$p"
    X "$p" tar -C "$RD/out/$1" -cf - . | tar -C "$OUT/$1/$p" -xf - || log "WARN: harvest of $1 from $p incomplete"
  done
}

run_case() { # $1 tag  $2 timeout  $3... exe and args
  local tag="$1" tmo="$2" p rc
  shift 2
  mkdir -p "$OUT/$tag"
  for p in "$P0" "$P1"; do XR "$p" prepare "$tag" || die "$p: prepare $tag failed"; done
  counters "$tag" before
  XR "$P0" mpirun "$tag" "$NP" "localhost:$GPU_PER_NODE,$IP1:$GPU_PER_NODE" "$MPI_CIDR" "$SSH_PORT" "$tmo" "$@" \
    > "$OUT/$tag/launch.console" 2>&1
  rc=$?
  counters "$tag" after
  for p in "$P0" "$P1"; do XR "$p" reap > "$OUT/$tag/reap-$p.txt" 2>&1; done
  harvest "$tag"
  log "case $tag: mpirun exit $rc; reap: $(cat "$OUT/$tag"/reap-*.txt | tr '\n' ' ')"
}

close_out() {
  local p
  for p in "$P0" "$P1"; do XR "$p" gpu > "$OUT/gpu-after-$p.txt" 2>&1; done
  for p in "$P1" "$P0"; do X "$p" env RUN_DIR="$RD" bash "$RD/sshd_setup.sh" stop >> "$OUT/sshd-stop.txt" 2>&1; done
  log "closed: GPU readback and sshd stop recorded"
}

cmd_run() {
  # Any early stop still records GPU state and stops the per-run sshd (best effort).
  trap 'rc=$?; if [ "${CLOSED:-0}" != 1 ]; then close_out; fi; exit $rc' EXIT
  preflight
  sshd_up
  run_case correctness "$EP_TEST_TIMEOUT" /opt/nccl-ep/test/nccl_ep/ep_test -a ht -L fl
  if check correctness --dir "$OUT" --world "$NP" >> "$OUT/run.log" 2>&1; then
    log "correctness PASS: running the timing loop"
    # shellcheck disable=SC2086  # EP_BENCH_ARGS is a word list by design
    run_case timing "$EP_BENCH_TIMEOUT" /opt/nccl-ep/test/nccl_ep/ep_bench $EP_BENCH_ARGS
  else
    log "correctness FAIL: timing loop not run (correctness first; see run.log)"
  fi
  close_out
  CLOSED=1
  check summarize --dir "$OUT" --out "$OUT/RESULT.json" --world "$NP" | tee -a "$OUT/run.log"
  return "${PIPESTATUS[0]}"
}

cmd_down() {
  "${K[@]}" delete pod "$P0" "$P1" --ignore-not-found --wait=true --timeout=5m | tee -a "$OUT/run.log"
  if "${K[@]}" get pod "$P0" "$P1" -o name 2>/dev/null | grep . > /dev/null; then die "pods still present after delete"; fi
  log "down: pods deleted; namespace $NAMESPACE kept (delete it if it was created only for this test)"
}

case "${1:-}" in
  up) cmd_up ;;
  run) cmd_run ;;
  down) cmd_down ;;
  all)
    cmd_up
    ( cmd_run ); rc=$?    # a subshell, so a stop inside run still reaches down below
    if [ "${KEEP_PODS:-0}" = 1 ]; then log "KEEP_PODS=1: pods left running"; else cmd_down; fi
    exit "$rc" ;;
  *) echo "usage: $0 up|run|down|all" >&2; exit 2 ;;
esac
