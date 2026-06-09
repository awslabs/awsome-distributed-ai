#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Drive the full with/without-UCCL A/B campaign for BOTH models, serially (each run uses all
# 256 GPUs), preserving every log on FSx Lustre with NO overwrite under one CAMPAIGN_ID.
#
#   matrix = MODELS {dsv3, kimi-k2} x CELLS {(mb1,off),(mb4,off),(mb4,on),(mb1,on)} x ARMS {alltoall, deepep}
#          = 16 runs.
#
# Per run: launch via ../run-ab-rawpods.sh, wait for rank-0 to finish, assert the EFA-active
# gate on all ranks, then delete the run's pods (free the 256 GPUs) before the next run. A
# persistent `bench-util` pod (0 GPU, FSx-mounted) is used for all FSx file ops + final parse.
#
# Usage:  CTX=<ctx> IMG=<ecr-uri> bash run-campaign.sh
# Override the matrix with MODELS / CELLS / ARMS env (space-separated; CELLS items are "mb:ovl").
set -uo pipefail

CTX="${CTX:?set CTX to your kubectl context}"
IMG="${IMG:?set IMG to your megatron-bridge-uccl ECR image URI}"
NS="${NS:-kimi-k2-bench}"
NNODES="${NNODES:-32}"
export TRAIN_ITERS="${TRAIN_ITERS:-24}"
export GLOBAL_BATCH="${GLOBAL_BATCH:-256}"
RUN_TIMEOUT="${RUN_TIMEOUT:-2400}"   # seconds to wait for a single run's rank-0 to finish

MODELS="${MODELS:-dsv3 kimi-k2}"
CELLS="${CELLS:-1:off 4:off 4:on 1:on}"
ARMS="${ARMS:-alltoall deepep}"

export CAMPAIGN_ID="${CAMPAIGN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export CTX IMG NS
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
LAUNCH="${SELF_DIR}/../run-ab-rawpods.sh"
REPO_ROOT="$(cd "${SELF_DIR}/../../../.." && pwd)"
GIT_REV="$(git -C "${SELF_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
export GIT_REV
K="kubectl --context ${CTX} -n ${NS}"
CAMPAIGN_FS="/fsx/megatron-bridge-bench/${CAMPAIGN_ID}"

echo "############################################################"
echo "# CAMPAIGN ${CAMPAIGN_ID}   git=${GIT_REV}"
echo "#   models=[${MODELS}] cells=[${CELLS}] arms=[${ARMS}]  -> $(echo ${MODELS} | wc -w)x$(echo ${CELLS} | wc -w)x$(echo ${ARMS} | wc -w) runs"
echo "#   logs (no overwrite): ${CAMPAIGN_FS}/<model>/<arm>-mb<m>-ovl<on|off>/"
echo "############################################################"

# ---- persistent util pod for FSx file ops + parsing (0 GPU, bench-tolerated) --------------
UTIL=bench-util
ensure_util() {
  if $K get pod ${UTIL} >/dev/null 2>&1; then return; fi
  cat <<EOF | $K apply -f - >/dev/null
apiVersion: v1
kind: Pod
metadata: {name: ${UTIL}}
spec:
  restartPolicy: Never
  nodeSelector: {workload: bench}
  tolerations:
    - {key: workload, value: bench, operator: Equal, effect: NoSchedule}
    - {key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}
    - {key: capacity-reservation, operator: Exists, effect: NoSchedule}
  containers:
    - name: u
      image: ${IMG}
      command: ["bash","-lc","mkdir -p ${CAMPAIGN_FS}; sleep infinity"]
      volumeMounts: [{name: fsx, mountPath: /fsx}]
  volumes:
    - name: fsx
      persistentVolumeClaim: {claimName: fsx-kimi-k2}
EOF
  echo "   waiting for ${UTIL} ..."
  $K wait --for=condition=Ready pod/${UTIL} --timeout=300s
}
uexec() { $K exec ${UTIL} -- bash -lc "$1"; }

ensure_util

# ---- stage bench entrypoints + parser onto FSx -------------------------------------------
echo "== staging bench scripts + parser to /fsx/kimi-k2 =="
$K cp "${REPO_ROOT}/3.test_cases/megatron/megatron-bridge/dsv3/benchmarks/bench_dsv3_pretrain.py" ${UTIL}:/fsx/kimi-k2/bench_dsv3_pretrain.py
$K cp "${REPO_ROOT}/3.test_cases/megatron/megatron-bridge/kimi-k2/benchmarks/bench_kimi_k2_pretrain.py" ${UTIL}:/fsx/kimi-k2/bench_kimi_k2_pretrain.py
uexec "mkdir -p /fsx/kimi-k2/bench"
$K cp "${SELF_DIR}/parse-runs.py" ${UTIL}:/fsx/kimi-k2/bench/parse-runs.py
uexec "ls -la /fsx/kimi-k2/bench_*.py /fsx/kimi-k2/bench/parse-runs.py"

# ---- one run ------------------------------------------------------------------------------
run_one() {
  local MODEL="$1" MB="$2" OVL="$3" ARM="$4"
  local JOB="abrun-${MODEL}-${ARM}"
  local RUN_DIR="${CAMPAIGN_FS}/${MODEL}/${ARM}-mb${MB}-ovl${OVL}"
  echo ""
  echo ">>> RUN  model=${MODEL} arm=${ARM} mb=${MB} overlap=${OVL}"
  MODEL="${MODEL}" MICRO_BATCH="${MB}" MOE_A2A_OVERLAP="${OVL}" NNODES="${NNODES}" \
    bash "${LAUNCH}" "${ARM}" "${NNODES}" || { echo "   launch failed"; return 1; }

  # wait for rank-0 to finish (Succeeded/Failed) or timeout
  local t=0
  while true; do
    local ph
    ph=$($K get pod "${JOB}-0" -o jsonpath='{.status.phase}' 2>/dev/null || echo "")
    [ "$ph" = "Succeeded" ] && { echo "   rank-0 Succeeded (${t}s)"; break; }
    [ "$ph" = "Failed" ]    && { echo "   rank-0 FAILED (${t}s) — see ${RUN_DIR}/logs/rank-0.log"; break; }
    [ "$t" -ge "$RUN_TIMEOUT" ] && { echo "   TIMEOUT ${RUN_TIMEOUT}s waiting on rank-0 (phase=${ph})"; break; }
    sleep 20; t=$((t+20))
  done

  # validity gates from FSx
  local efa status
  efa=$(uexec "grep -l 'Selected provider is efa' ${RUN_DIR}/logs/rank-*.log 2>/dev/null | wc -l" | tr -d '[:space:]')
  status=$(uexec "cat ${RUN_DIR}/STATUS 2>/dev/null" | tr -d '\r')
  echo "   STATUS: ${status:-<none>} | EFA-active ranks: ${efa}/${NNODES}"
  [ "${efa}" != "${NNODES}" ] && echo "   !! WARNING: EFA not active on all ranks — treat this run as INVALID (rerun)."

  # free the 256 GPUs before the next run
  for r in $(seq 0 $((NNODES-1))); do $K delete pod "${JOB}-${r}" --ignore-not-found --wait=false >/dev/null 2>&1; done
  $K delete svc "${JOB}" --ignore-not-found >/dev/null 2>&1
  echo "   waiting for ${JOB} pods to terminate (free GPUs) ..."
  $K wait --for=delete pod -l app="${JOB}" --timeout=240s >/dev/null 2>&1 || sleep 20
}

# ---- matrix -------------------------------------------------------------------------------
for MODEL in ${MODELS}; do
  for CELL in ${CELLS}; do
    MB="${CELL%%:*}"; OVL="${CELL##*:}"
    for ARM in ${ARMS}; do
      run_one "${MODEL}" "${MB}" "${OVL}" "${ARM}"
    done
  done
done

# ---- parse the whole campaign into index.csv + per-run loss_curve.csv ----------------------
echo ""
echo "== parsing campaign =="
uexec "cd /fsx/kimi-k2/bench && python3 parse-runs.py ${CAMPAIGN_FS} --warmup 4"
$K cp ${UTIL}:${CAMPAIGN_FS}/index.csv "${SELF_DIR}/last-campaign-index.csv" 2>/dev/null \
  && echo "   pulled index.csv -> ${SELF_DIR}/last-campaign-index.csv"
echo ""
echo "Campaign ${CAMPAIGN_ID} done. Raw logs preserved under ${CAMPAIGN_FS} (no overwrite)."
echo "Util pod ${UTIL} left running for inspection; delete with: ${K} delete pod ${UTIL}"
