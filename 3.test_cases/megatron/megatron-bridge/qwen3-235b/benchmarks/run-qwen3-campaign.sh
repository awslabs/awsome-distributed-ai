#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Drive the Qwen3-235B-A22B THREE-WAY MoE dispatcher comparison on up to 8x p6-b300:
#
#   arms   = { alltoall (NCCL all-to-all),
#              deepep-uccl    (DeepEP over UCCL/EFA),
#              deepep-nvshmem (NVIDIA DeepEP over NVSHMEM-libfabric/EFA) }
#   EP     = { 16, 32 }                 # the all-to-all fan-out is the variable that matters
#   cells  = { mb:overlap }            # default: 1:off 4:off 4:on
#
# matrix = 3 arms x |EP| x |cells|  (default 3 x 2 x 3 = 18 runs), run serially: each run
# uses all 64 GPUs. Per run: launch via ../../run-ab-rawpods.sh, wait for rank-0, assert
# the EFA-active gate on all ranks, delete the run's pods, then the next run.
#
# The two deepep arms differ ONLY in the image (which `deep_ep` is importable); alltoall
# uses the UCCL image but never imports deep_ep. Each arm writes to a distinct, never-
# overwritten dir tagged <arm>-ep<EP>-mb<m>-ovl<on|off> under one CAMPAIGN_ID.
#
# Usage:
#   CTX=<ctx> UCCL_IMG=<uccl-ecr-uri> NVSHMEM_IMG=<nvshmem-ecr-uri> bash run-qwen3-campaign.sh
# Override the matrix with EPS / CELLS / ARMS env (space-separated; CELLS items are "mb:ovl").
set -uo pipefail

CTX="${CTX:?set CTX to your kubectl context}"
UCCL_IMG="${UCCL_IMG:?set UCCL_IMG to the megatron-bridge-uccl (UCCL) ECR image URI}"
NVSHMEM_IMG="${NVSHMEM_IMG:?set NVSHMEM_IMG to the deepep-nvshmem ECR image URI}"
NS="${NS:-kimi-k2-bench}"
PVC="${PVC:-fsx-kimi-k2}"
NNODES="${NNODES:-8}"
GPUS_PER_NODE=8
WORLD=$(( NNODES * GPUS_PER_NODE ))

export TRAIN_ITERS="${TRAIN_ITERS:-24}"
export GLOBAL_BATCH="${GLOBAL_BATCH:-256}"
export SEQ_LEN="${SEQ_LEN:-4096}"
export MOE_FORCE_BALANCE="${MOE_FORCE_BALANCE:-on}"
export RECOMPUTE="${RECOMPUTE:-}"     # full|selective|"" — held identical across arms
RUN_TIMEOUT="${RUN_TIMEOUT:-2400}"     # seconds to wait for one run's rank-0 to finish

EPS="${EPS:-16 32}"
CELLS="${CELLS:-1:off 4:off 4:on}"
# arm spec = "label:dispatcher:imagevar:backend" (imagevar in {UCCL,NVSHMEM}).
ARMS="${ARMS:-alltoall:alltoall:UCCL: deepep-uccl:deepep:UCCL:uccl deepep-nvshmem:deepep:NVSHMEM:nvshmem}"

export CAMPAIGN_ID="${CAMPAIGN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export CTX NS
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
LAUNCH="${SELF_DIR}/../../run-ab-rawpods.sh"
PARSER="${SELF_DIR}/../../bench/parse-runs.py"
BENCH_PY_SRC="${SELF_DIR}/bench_qwen3_pretrain.py"
GIT_REV="$(git -C "${SELF_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
export GIT_REV
MODEL=qwen3-235b
STAGE=/fsx/kimi-k2
HF_CACHE="${STAGE}/hf-cache"
K="kubectl --context ${CTX} -n ${NS}"
CAMPAIGN_FS="/fsx/megatron-bridge-bench/${CAMPAIGN_ID}"

echo "############################################################"
echo "# QWEN3-235B 3-WAY CAMPAIGN ${CAMPAIGN_ID}  git=${GIT_REV}"
echo "#   arms=[${ARMS}]"
echo "#   EP=[${EPS}] cells=[${CELLS}] nnodes=${NNODES} world=${WORLD}"
echo "#   uccl=${UCCL_IMG}"
echo "#   nvshmem=${NVSHMEM_IMG}"
echo "#   logs (no overwrite): ${CAMPAIGN_FS}/${MODEL}/<arm>-ep<EP>-mb<m>-ovl<on|off>/"
echo "############################################################"

# ---- persistent util pod for FSx file ops + HF staging + parsing (0 GPU) ------------------
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
      image: ${UCCL_IMG}
      command: ["bash","-lc","mkdir -p ${CAMPAIGN_FS}; sleep infinity"]
      volumeMounts: [{name: fsx, mountPath: /fsx}]
  volumes:
    - name: fsx
      persistentVolumeClaim: {claimName: ${PVC}}
EOF
  echo "   waiting for ${UTIL} ..."
  $K wait --for=condition=Ready pod/${UTIL} --timeout=300s
}
uexec() { $K exec ${UTIL} -- bash -lc "$1"; }

ensure_util

# ---- stage bench entrypoint + parser onto FSx --------------------------------------------
echo "== staging bench + parser to ${STAGE} =="
$K cp "${BENCH_PY_SRC}" ${UTIL}:${STAGE}/bench_qwen3_pretrain.py
uexec "mkdir -p ${STAGE}/bench"
$K cp "${PARSER}" ${UTIL}:${STAGE}/bench/parse-runs.py
uexec "ls -la ${STAGE}/bench_qwen3_pretrain.py ${STAGE}/bench/parse-runs.py"

# ---- stage the Qwen3 HF config + tokenizer (config only; load_weights=False) -------------
# The recipe builds its provider via AutoBridge.from_hf_pretrained(...). Pre-download the
# small config/tokenizer into an FSx HF cache so runs work without per-pod egress. If the
# util pod has no egress this is a no-op and runs fall back to online (HF_HUB_OFFLINE=0).
HF_OFFLINE=0
echo "== staging Qwen3 HF config/tokenizer to ${HF_CACHE} (config only) =="
if uexec "HF_HOME=${HF_CACHE} python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-235B-A22B', allow_patterns=['*.json','*.txt','tokenizer*','merges*','vocab*'])\" "; then
  HF_OFFLINE=1
  echo "   HF config staged — runs will use HF_HUB_OFFLINE=1"
else
  echo "   WARN: HF prestage failed (no egress on util pod?). Runs will try online (HF_HUB_OFFLINE=0)."
fi
export HF_HOME="${HF_CACHE}" HF_HUB_OFFLINE="${HF_OFFLINE}"

# ---- one run ------------------------------------------------------------------------------
run_one() {
  local LABEL="$1" DISP="$2" IMG="$3" BACKEND="$4" EP="$5" MB="$6" OVL="$7"
  local PP=$(( WORLD / EP ))          # TP8 fixed: EP=TP*DP -> PP=WORLD/EP (EP32->PP2, EP16->PP4)
  local ARM_LABEL="${LABEL}-ep${EP}"
  local JOB="abrun-${MODEL}-${ARM_LABEL}"
  local RUN_DIR="${CAMPAIGN_FS}/${MODEL}/${ARM_LABEL}-mb${MB}-ovl${OVL}"
  echo ""
  echo ">>> RUN  arm=${LABEL} backend=${BACKEND:-nccl} EP=${EP} (TP8/PP${PP}) mb=${MB} overlap=${OVL}"
  MODEL="${MODEL}" IMG="${IMG}" ARM_LABEL="${ARM_LABEL}" EP_BACKEND="${BACKEND}" \
    TENSOR_PARALLEL=8 PIPELINE_PARALLEL="${PP}" EXPERT_PARALLEL="${EP}" \
    MICRO_BATCH="${MB}" MOE_A2A_OVERLAP="${OVL}" NNODES="${NNODES}" \
    HF_HOME="${HF_HOME}" HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" \
    bash "${LAUNCH}" "${DISP}" "${NNODES}" || { echo "   launch failed"; return 1; }

  # wait for rank-0 to finish (Succeeded/Failed) or timeout
  local t=0 ph
  while true; do
    ph=$($K get pod "${JOB}-0" -o jsonpath='{.status.phase}' 2>/dev/null || echo "")
    [ "$ph" = "Succeeded" ] && { echo "   rank-0 Succeeded (${t}s)"; break; }
    [ "$ph" = "Failed" ]    && { echo "   rank-0 FAILED (${t}s) — see ${RUN_DIR}/logs/rank-0.log"; break; }
    [ "$t" -ge "$RUN_TIMEOUT" ] && { echo "   TIMEOUT ${RUN_TIMEOUT}s (phase=${ph})"; break; }
    sleep 20; t=$((t+20))
  done

  # validity gates from FSx
  local efa status
  efa=$(uexec "grep -l 'Selected provider is efa' ${RUN_DIR}/logs/rank-*.log 2>/dev/null | wc -l" | tr -d '[:space:]')
  status=$(uexec "cat ${RUN_DIR}/STATUS 2>/dev/null" | tr -d '\r')
  echo "   STATUS: ${status:-<none>} | EFA-active ranks: ${efa}/${NNODES}"
  [ "${efa}" != "${NNODES}" ] && echo "   !! WARNING: EFA not active on all ranks — treat as INVALID (rerun)."

  # free the GPUs before the next run
  for r in $(seq 0 $((NNODES-1))); do $K delete pod "${JOB}-${r}" --ignore-not-found --wait=false >/dev/null 2>&1; done
  $K delete svc "${JOB}" --ignore-not-found >/dev/null 2>&1
  echo "   waiting for ${JOB} pods to terminate (free GPUs) ..."
  $K wait --for=delete pod -l app="${JOB}" --timeout=240s >/dev/null 2>&1 || sleep 20
}

# ---- matrix: EP (outer) x cell x arm ------------------------------------------------------
for EP in ${EPS}; do
  for CELL in ${CELLS}; do
    MB="${CELL%%:*}"; OVL="${CELL##*:}"
    for SPEC in ${ARMS}; do
      IFS=':' read -r LABEL DISP IMGVAR BACKEND <<< "${SPEC}"
      case "${IMGVAR}" in
        UCCL)    IMG="${UCCL_IMG}" ;;
        NVSHMEM) IMG="${NVSHMEM_IMG}" ;;
        *) echo "bad imagevar ${IMGVAR} in arm spec ${SPEC}"; continue ;;
      esac
      run_one "${LABEL}" "${DISP}" "${IMG}" "${BACKEND:-}" "${EP}" "${MB}" "${OVL}"
    done
  done
done

# ---- parse the whole campaign into index.csv + per-run loss_curve.csv ---------------------
echo ""
echo "== parsing campaign =="
uexec "cd ${STAGE}/bench && python3 parse-runs.py ${CAMPAIGN_FS} --warmup 4"
$K cp ${UTIL}:${CAMPAIGN_FS}/index.csv "${SELF_DIR}/last-campaign-index.csv" 2>/dev/null \
  && echo "   pulled index.csv -> ${SELF_DIR}/last-campaign-index.csv"
echo ""
echo "Campaign ${CAMPAIGN_ID} done. Raw logs preserved under ${CAMPAIGN_FS} (no overwrite)."
echo "Util pod ${UTIL} left running; delete with: ${K} delete pod ${UTIL}"
