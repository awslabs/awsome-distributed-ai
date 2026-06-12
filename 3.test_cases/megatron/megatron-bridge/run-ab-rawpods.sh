#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Model-agnostic MoE dispatcher A/B launcher via RAW ranked Pods + headless Service.
# This cluster's kubeflow PyTorchJob CRD is absent, so we wire static torchrun
# rendezvous ourselves: 1 headless Service ${JOB} + ${NNODES} Pods ${JOB}-0..N-1,
# each torchrun with --node_rank from its ordinal, master_addr=${JOB}-0.
#
# Runs ONE arm (alltoall|deepep) of ONE model per invocation. Model/data/parallelism
# are byte-identical across arms; only MOE_DISPATCHER differs.
#
#   MODEL=dsv3      -> DeepSeek-V3 256-expert recipe     (BENCH_PY bench_dsv3_pretrain.py)
#   MODEL=kimi-k2   -> Kimi-K2 384-expert via AutoBridge (BENCH_PY bench_kimi_k2_pretrain.py)
#
# NO-OVERWRITE LOGGING: every run writes to a unique directory on FSx Lustre under
#   /fsx/megatron-bridge-bench/${CAMPAIGN_ID}/${MODEL}/${ARM}-mb${MICRO_BATCH}-ovl${MOE_A2A_OVERLAP}/
# (logs/rank-<r>.log for all ranks, env.txt, STATUS). A run whose dir already has a
# completed STATUS is REFUSED (rank-0 aborts) so a retro is never clobbered. CAMPAIGN_ID
# defaults to a fresh UTC timestamp; the campaign driver passes one shared id for all runs.
#
# Usage:  MODEL=<dsv3|kimi-k2> CTX=<ctx> IMG=<ecr-uri> ./run-ab-rawpods.sh <alltoall|deepep> [NNODES]
set -uo pipefail

ARM="${1:?usage: MODEL=<dsv3|kimi-k2> ./run-ab-rawpods.sh <alltoall|deepep> [NNODES]}"
NNODES="${2:-32}"

CTX="${CTX:?set CTX to your kubectl context}"
NS="${NS:-kimi-k2-bench}"
IMG="${IMG:?set IMG to your megatron-bridge-uccl ECR image URI}"
MODEL="${MODEL:-dsv3}"
GPUS_PER_NODE=8
EFA_PER_NODE=16
WORLD=$(( NNODES * GPUS_PER_NODE ))

# Parallelism. TP MUST be >1 (recipe enables sequence_parallel). EP = DP*TP = 32 (ETP=1) at 256 GPU.
TP="${TENSOR_PARALLEL:-8}"
PP="${PIPELINE_PARALLEL:-8}"
EP="${EXPERT_PARALLEL:-32}"
TRAIN_ITERS="${TRAIN_ITERS:-24}"
GLOBAL_BATCH="${GLOBAL_BATCH:-256}"
MICRO_BATCH="${MICRO_BATCH:-1}"
SEQ_LEN="${SEQ_LEN:-4096}"
MOE_A2A_OVERLAP="${MOE_A2A_OVERLAP:-on}"
MOE_FORCE_BALANCE="${MOE_FORCE_BALANCE:-on}"
LOSS_PROBE="${LOSS_PROBE:-0}"

# Staging dir on FSx holds the bench entrypoints + the Kimi-K2 HF config dir (hf/).
STAGE="${STAGE:-/fsx/kimi-k2}"
case "${MODEL}" in
  dsv3)    DEFAULT_BENCH="${STAGE}/bench_dsv3_pretrain.py" ;;
  kimi-k2) DEFAULT_BENCH="${STAGE}/bench_kimi_k2_pretrain.py" ;;
  *) echo "MODEL must be 'dsv3' or 'kimi-k2', got '${MODEL}'" >&2; exit 2 ;;
esac
BENCH_PY="${BENCH_PY:-${DEFAULT_BENCH}}"

# No-overwrite run tree on Lustre. One CAMPAIGN_ID groups a whole 16-run campaign.
CAMPAIGN_ID="${CAMPAIGN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_TAG="${ARM}-mb${MICRO_BATCH}-ovl${MOE_A2A_OVERLAP}"
RUN_DIR="${RUN_DIR:-/fsx/megatron-bridge-bench/${CAMPAIGN_ID}/${MODEL}/${RUN_TAG}}"
LOGDIR="${RUN_DIR}/logs"

GIT_REV="${GIT_REV:-$(git -C "$(dirname "$0")" rev-parse --short HEAD 2>/dev/null || echo unknown)}"
JOB="abrun-${MODEL}-${ARM}"
PORT=12355
K="kubectl --context ${CTX} -n ${NS}"

echo "== raw-pod A/B  model=${MODEL} arm=${ARM} nnodes=${NNODES} world=${WORLD} TP${TP}/PP${PP}/EP${EP} mb=${MICRO_BATCH} ovl=${MOE_A2A_OVERLAP} =="
echo "   img=${IMG}"
echo "   bench=${BENCH_PY} iters=${TRAIN_ITERS} gbs=${GLOBAL_BATCH} seq=${SEQ_LEN}"
echo "   RUN_DIR=${RUN_DIR}  (logs/rank-<r>.log, no overwrite)"

# Clean prior pods of THIS job by explicit name (avoids label-selector ambiguity).
for r in $(seq 0 $(( NNODES - 1 ))); do $K delete pod "${JOB}-${r}" --ignore-not-found --wait=false >/dev/null 2>&1; done
$K delete svc "${JOB}" --ignore-not-found >/dev/null 2>&1
sleep 3

cat <<EOF | $K apply -f - >/dev/null
apiVersion: v1
kind: Service
metadata: {name: ${JOB}}
spec:
  clusterIP: None
  selector: {app: ${JOB}}
  ports: [{name: rdzv, port: ${PORT}}]
EOF

# rank-0 owns the run dir: no-overwrite guard + env.txt + STATUS. Other ranks only mkdir + log.
RANK0_PREAMBLE="
          if [ -f ${RUN_DIR}/STATUS ]; then echo 'REFUSE: ${RUN_DIR} already has STATUS (completed run); not overwriting' ; exit 3 ; fi ;
          mkdir -p ${LOGDIR} ;
          { echo run_dir=${RUN_DIR} ; echo model=${MODEL} arm=${ARM} ; echo nnodes=${NNODES} world=${WORLD} ;
            echo TP=${TP} PP=${PP} EP=${EP} mb=${MICRO_BATCH} gbs=${GLOBAL_BATCH} seq=${SEQ_LEN} iters=${TRAIN_ITERS} ;
            echo overlap=${MOE_A2A_OVERLAP} force_balance=${MOE_FORCE_BALANCE} loss_probe=${LOSS_PROBE} ;
            echo image=${IMG} ; echo bench_py=${BENCH_PY} ; echo git_rev=${GIT_REV} ; echo started=\$(date -u +%FT%TZ) ; } > ${RUN_DIR}/env.txt ;"

launch_pod() {
  local R="$1"
  local PREAMBLE="mkdir -p ${LOGDIR} ;"
  local EPILOGUE=""
  if [ "$R" = "0" ]; then
    PREAMBLE="${RANK0_PREAMBLE}"
    EPILOGUE="; echo \"exit=\$? finished=\$(date -u +%FT%TZ)\" > ${RUN_DIR}/STATUS"
  fi
  cat <<EOF | $K apply -f - >/dev/null
apiVersion: v1
kind: Pod
metadata:
  name: ${JOB}-${R}
  labels: {app: ${JOB}, rank: "${R}"}
spec:
  restartPolicy: Never
  hostname: ${JOB}-${R}
  subdomain: ${JOB}
  nodeSelector:
    node.kubernetes.io/instance-type: p6-b300.48xlarge
  tolerations:
    - {key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}
    - {key: workload, value: bench, operator: Equal, effect: NoSchedule}
    - {key: capacity-reservation, operator: Exists, effect: NoSchedule}
  containers:
    - name: c
      image: ${IMG}
      command: ["bash","-lc"]
      args:
        - >
          ${PREAMBLE}
          export PYTHONPATH=${STAGE} KIMI_K2_HF_PATH=${STAGE}/hf
          MOE_DISPATCHER=${ARM} MOE_A2A_OVERLAP=${MOE_A2A_OVERLAP} MOE_FORCE_BALANCE=${MOE_FORCE_BALANCE}
          TENSOR_PARALLEL=${TP} PIPELINE_PARALLEL=${PP} EXPERT_PARALLEL=${EP}
          TRAIN_ITERS=${TRAIN_ITERS} GLOBAL_BATCH=${GLOBAL_BATCH} MICRO_BATCH=${MICRO_BATCH} SEQ_LEN=${SEQ_LEN}
          LOSS_PROBE=${LOSS_PROBE}
          FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 FI_EFA_FORK_SAFE=1
          NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET NCCL_SOCKET_IFNAME=^docker,lo,veth ;
          torchrun --nnodes=${NNODES} --nproc_per_node=${GPUS_PER_NODE}
          --node_rank=${R} --master_addr=${JOB}-0.${JOB}.${NS}.svc.cluster.local
          --master_port=${PORT} ${BENCH_PY} > ${LOGDIR}/rank-${R}.log 2>&1 ${EPILOGUE}
      resources:
        requests: {nvidia.com/gpu: ${GPUS_PER_NODE}, vpc.amazonaws.com/efa: ${EFA_PER_NODE}}
        limits:   {nvidia.com/gpu: ${GPUS_PER_NODE}, vpc.amazonaws.com/efa: ${EFA_PER_NODE}}
      volumeMounts:
        - {name: fsx, mountPath: /fsx}
        - {name: shmem, mountPath: /dev/shm}
  volumes:
    - name: fsx
      persistentVolumeClaim: {claimName: fsx-kimi-k2}
    - name: shmem
      emptyDir: {medium: Memory, sizeLimit: 32Gi}
EOF
}

for r in $(seq 0 $(( NNODES - 1 ))); do launch_pod "$r"; done
echo "   launched ${NNODES} pods: ${JOB}-0..$(( NNODES - 1 ))"
echo "   tail rank-0:  ${LOGDIR}/rank-0.log   STATUS: ${RUN_DIR}/STATUS"
