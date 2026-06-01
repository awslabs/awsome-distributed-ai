#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Tier-B end-to-end MoE dispatcher A/B via RAW ranked Pods + headless Service.
# This cluster's kubeflow PyTorchJob CRD is absent, so we wire static torchrun
# rendezvous ourselves: 1 headless Service ${JOB} + ${NNODES} Pods ${JOB}-0..N-1,
# each torchrun with --node_rank from its ordinal, master_addr=${JOB}-0.
#
# Runs ONE arm (alltoall|deepep) per invocation. Model/data/parallelism are
# byte-identical across arms; only MOE_DISPATCHER differs. Native DeepSeek-V3
# 32-node shape: TP8/PP8/DP4/EP32 = 256 GPU, full 61-layer depth, 256 experts.
# Each pod's torchrun output is redirected to FSx (kubectl exec/logs stdout
# garbles); read with a reader pod `cat ${LOGDIR}/${JOB}-<r>.log`.
#
# Usage:  ./run-ab-rawpods.sh <alltoall|deepep> [NNODES]   (default NNODES=32)
set -uo pipefail

ARM="${1:?usage: run-ab-rawpods.sh <alltoall|deepep> [NNODES]}"
NNODES="${2:-32}"

CTX="${CTX:?set CTX to your kubectl context}"
NS="${NS:-kimi-k2-bench}"
IMG="${IMG:?set IMG to your megatron-bridge-uccl ECR image URI}"
GPUS_PER_NODE=8
EFA_PER_NODE=16
WORLD=$(( NNODES * GPUS_PER_NODE ))

# Native DSV3 32-node parallelism. TP MUST be >1 (recipe enables sequence_parallel).
# EP = DP*TP = 32 with ETP=1 at 256 GPU.
TP="${TENSOR_PARALLEL:-8}"
PP="${PIPELINE_PARALLEL:-8}"
EP="${EXPERT_PARALLEL:-32}"
TRAIN_ITERS="${TRAIN_ITERS:-20}"
GLOBAL_BATCH="${GLOBAL_BATCH:-512}"
MICRO_BATCH="${MICRO_BATCH:-1}"
SEQ_LEN="${SEQ_LEN:-4096}"
MOE_A2A_OVERLAP="${MOE_A2A_OVERLAP:-on}"
LOSS_PROBE="${LOSS_PROBE:-0}"

BENCH_PY="${BENCH_PY:-/fsx/kimi-k2/bench_dsv3_pretrain.py}"
LOGDIR="${LOGDIR:-/fsx/kimi-k2/bench/logs}"
JOB="abrun-${ARM}"
PORT=12355
K="kubectl --context ${CTX} -n ${NS}"

echo "== Tier-B raw-pod A/B arm=${ARM} nnodes=${NNODES} world=${WORLD} TP${TP}/PP${PP}/EP${EP} =="
echo "   img=${IMG}"
echo "   bench=${BENCH_PY} iters=${TRAIN_ITERS} gbs=${GLOBAL_BATCH} seq=${SEQ_LEN}  log=${LOGDIR}/${JOB}-<r>.log"

# Clean prior pods of THIS arm by explicit name (avoids label-selector ambiguity).
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

launch_pod() {
  local R="$1"
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
          mkdir -p ${LOGDIR} ;
          export PYTHONPATH=/fsx/kimi-k2 MOE_DISPATCHER=${ARM} MOE_A2A_OVERLAP=${MOE_A2A_OVERLAP}
          TENSOR_PARALLEL=${TP} PIPELINE_PARALLEL=${PP} EXPERT_PARALLEL=${EP}
          TRAIN_ITERS=${TRAIN_ITERS} GLOBAL_BATCH=${GLOBAL_BATCH} MICRO_BATCH=${MICRO_BATCH} SEQ_LEN=${SEQ_LEN}
          LOSS_PROBE=${LOSS_PROBE}
          FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 FI_EFA_FORK_SAFE=1
          NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET NCCL_SOCKET_IFNAME=^docker,lo,veth ;
          torchrun --nnodes=${NNODES} --nproc_per_node=${GPUS_PER_NODE}
          --node_rank=${R} --master_addr=${JOB}-0.${JOB}.${NS}.svc.cluster.local
          --master_port=${PORT} ${BENCH_PY} > ${LOGDIR}/${JOB}-${R}.log 2>&1
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
