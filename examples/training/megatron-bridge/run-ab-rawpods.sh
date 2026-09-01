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
#   MODEL=dsv3       -> DeepSeek-V3 256-expert recipe     (BENCH_PY bench_dsv3_pretrain.py)
#   MODEL=kimi-k2    -> Kimi-K2 384-expert via AutoBridge (BENCH_PY bench_kimi_k2_pretrain.py)
#   MODEL=qwen3-235b -> Qwen3-235B-A22B 128-expert recipe (BENCH_PY bench_qwen3_pretrain.py)
#
# THREE-WAY DISPATCHER COMPARISON (NCCL / DeepEP+UCCL / DeepEP+NVSHMEM): the ARM
# positional is what the bench reads as MOE_DISPATCHER (alltoall|deepep). The deepep
# arm's *transport* is set by which IMG is passed (UCCL image -> UCCL deep_ep;
# NVSHMEM image -> NVIDIA DeepEP). Set ARM_LABEL to disambiguate the two deepep arms
# in run dirs / pod names (e.g. ARM_LABEL=deepep-nvshmem IMG=<nvshmem-image>).
#
# NO-OVERWRITE LOGGING: every run writes to a unique directory on FSx Lustre under
#   /fsx/megatron-bridge-bench/${CAMPAIGN_ID}/${MODEL}/${ARM_LABEL}-mb${MICRO_BATCH}-ovl${MOE_A2A_OVERLAP}/
# (logs/rank-<r>.log for all ranks, env.txt, STATUS). A run whose dir already has a
# completed STATUS is REFUSED (rank-0 aborts) so a retro is never clobbered. CAMPAIGN_ID
# defaults to a fresh UTC timestamp; the campaign driver passes one shared id for all runs.
#
# Usage:  MODEL=<dsv3|kimi-k2|qwen3-235b> CTX=<ctx> IMG=<ecr-uri> ./run-ab-rawpods.sh <alltoall|deepep> [NNODES]
set -uo pipefail

ARM="${1:?usage: MODEL=<dsv3|kimi-k2|qwen3-235b> ./run-ab-rawpods.sh <alltoall|deepep> [NNODES]}"
NNODES="${2:-32}"
# ARM_LABEL names the run dir / pod set; defaults to ARM. Use it to split the two deepep
# transports (deepep-uccl vs deepep-nvshmem) into distinct, non-clobbering run dirs.
ARM_LABEL="${ARM_LABEL:-${ARM}}"

CTX="${CTX:?set CTX to your kubectl context}"
NS="${NS:-kimi-k2-bench}"
IMG="${IMG:?set IMG to your megatron-bridge-uccl ECR image URI}"
MODEL="${MODEL:-dsv3}"
GPUS_PER_NODE=8
# Node type + EFA NIC count per node. Defaults to p6-b300 (16 EFA); set INSTANCE_TYPE=
# p5.48xlarge EFA_PER_NODE=32 for the H100 runs.
INSTANCE_TYPE="${INSTANCE_TYPE:-p6-b300.48xlarge}"
EFA_PER_NODE="${EFA_PER_NODE:-16}"
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
# Optional activation recompute (full|selective|""), passed to the bench. Lets a large
# per-stage layer count fit on few nodes (e.g. EP32 at PP1). Held identical across arms.
RECOMPUTE="${RECOMPUTE:-}"
# Transport label recorded in env.txt for the 3-way comparison (uccl|nvshmem|"" for the
# NCCL alltoall arm). Informational only — the actual transport is fixed by the IMG.
EP_BACKEND="${EP_BACKEND:-}"
# Staging dir on FSx holds the bench entrypoints + the Kimi-K2 HF config dir (hf/).
STAGE="${STAGE:-/fsx/kimi-k2}"

# HF hub config/tokenizer access. The qwen3-235b recipe builds its model provider via
# AutoBridge.from_hf_pretrained(...) (config + tokenizer only; load_weights=False), so it
# needs either pod egress to huggingface.co OR a pre-staged offline cache. Point HF_HOME at
# a staged cache on FSx and set HF_HUB_OFFLINE=1 to run without egress. (No-op for dsv3.)
HF_HOME="${HF_HOME:-${STAGE}/hf-cache}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

# Volume backing /fsx. Default is the FSx Lustre PVC (shared across all nodes). On clusters
# without FSx (e.g. local-zone capacity blocks), set STORAGE=hostpath to back /fsx with
# node-local NVMe (HOSTPATH_ROOT). In hostpath mode ${STAGE} must be pre-staged on EVERY
# node and each node holds only its own rank's logs — harvest with a utility DaemonSet
# after every cell (see kimi-k2/README.md).
STORAGE="${STORAGE:-pvc}"
FSX_PVC="${FSX_PVC:-fsx-kimi-k2}"
HOSTPATH_ROOT="${HOSTPATH_ROOT:-/mnt/k8s-disks/0/bench-fsx}"
case "${STORAGE}" in
  pvc)      FSX_VOLUME_SRC="persistentVolumeClaim: {claimName: ${FSX_PVC}}" ;;
  hostpath) FSX_VOLUME_SRC="hostPath: {path: ${HOSTPATH_ROOT}, type: DirectoryOrCreate}" ;;
  *) echo "STORAGE must be 'pvc' or 'hostpath', got '${STORAGE}'" >&2; exit 2 ;;
esac

# GDRCOPY_DEV=on mounts the host's /dev/gdrdrv into the pod (privileged). Needed for the
# NVSHMEM arm when its symmetric heap outgrows the init chunk: NVSHMEM's dynamic (CUDA-VMM)
# heap growth registers chunks over libfabric/EFA via GDRCopy, and on clusters whose nvidia
# container toolkit does not honor NVIDIA_GDRCOPY=enabled the device is absent in-container
# and every rank dies at register_mem_handle (mem_heap.cpp:1361). No-op for UCCL/alltoall.
GDRCOPY_DEV="${GDRCOPY_DEV:-off}"
GDR_MOUNT_LINE=""; GDR_VOLUME_LINE=""; SECURITY_LINE=""
if [ "${GDRCOPY_DEV}" = "on" ]; then
  GDR_MOUNT_LINE='- {name: gdrdrv, mountPath: /dev/gdrdrv}'
  GDR_VOLUME_LINE='- {name: gdrdrv, hostPath: {path: /dev/gdrdrv, type: CharDevice}}'
  SECURITY_LINE='securityContext: {privileged: true}'
fi
case "${MODEL}" in
  dsv3)       DEFAULT_BENCH="${STAGE}/bench_dsv3_pretrain.py" ;;
  kimi-k2)    DEFAULT_BENCH="${STAGE}/bench_kimi_k2_pretrain.py" ;;
  # Both qwen3 sizes share one bench, differentiated by QWEN3_SIZE (235b on B300, 30b on H100).
  qwen3-235b) DEFAULT_BENCH="${STAGE}/bench_qwen3_pretrain.py"; QWEN3_SIZE="${QWEN3_SIZE:-235b}" ;;
  qwen3-30b)  DEFAULT_BENCH="${STAGE}/bench_qwen3_pretrain.py"; QWEN3_SIZE="${QWEN3_SIZE:-30b}" ;;
  *) echo "MODEL must be 'dsv3', 'kimi-k2', 'qwen3-235b', or 'qwen3-30b', got '${MODEL}'" >&2; exit 2 ;;
esac
BENCH_PY="${BENCH_PY:-${DEFAULT_BENCH}}"
QWEN3_SIZE="${QWEN3_SIZE:-}"

# No-overwrite run tree on Lustre. One CAMPAIGN_ID groups a whole campaign.
CAMPAIGN_ID="${CAMPAIGN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_TAG="${ARM_LABEL}-mb${MICRO_BATCH}-ovl${MOE_A2A_OVERLAP}"
RUN_DIR="${RUN_DIR:-/fsx/megatron-bridge-bench/${CAMPAIGN_ID}/${MODEL}/${RUN_TAG}}"
LOGDIR="${RUN_DIR}/logs"

GIT_REV="${GIT_REV:-$(git -C "$(dirname "$0")" rev-parse --short HEAD 2>/dev/null || echo unknown)}"
JOB="abrun-${MODEL}-${ARM_LABEL}"
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
          { echo run_dir=${RUN_DIR} ; echo model=${MODEL} arm=${ARM} arm_label=${ARM_LABEL} ep_backend=${EP_BACKEND} ; echo nnodes=${NNODES} world=${WORLD} ;
            echo TP=${TP} PP=${PP} EP=${EP} mb=${MICRO_BATCH} gbs=${GLOBAL_BATCH} seq=${SEQ_LEN} iters=${TRAIN_ITERS} ;
            echo overlap=${MOE_A2A_OVERLAP} force_balance=${MOE_FORCE_BALANCE} loss_probe=${LOSS_PROBE} ;
            echo image=${IMG} ; echo bench_py=${BENCH_PY} ; echo git_rev=${GIT_REV} ; echo started=\$(date -u +%FT%TZ) ; } > ${RUN_DIR}/env.txt ;"

launch_pod() {
  local R="$1"
  # ALL ranks must skip a completed cell, not just rank-0. If only rank-0 REFUSE-exits, ranks
  # 1..N-1 still start torchrun, fail rendezvous (no rank-0), and OVERWRITE their rank logs —
  # corrupting a previously-good run when a campaign is re-run with the same CAMPAIGN_ID.
  # The skip key is STATUS *existing*, deliberately NOT exit==0: the NVSHMEM arm writes STATUS
  # then exits 1 at NVSHMEM finalize (validity is judged by efa_ok + n_steady, not exit code —
  # see RESULTS.md), so a same-CAMPAIGN_ID re-run treats such a cell as complete and skips it.
  local PREAMBLE="if [ -f ${RUN_DIR}/STATUS ]; then echo 'skip: completed run' ; exit 0 ; fi ; mkdir -p ${LOGDIR} ;"
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
    node.kubernetes.io/instance-type: ${INSTANCE_TYPE}
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
          LOSS_PROBE=${LOSS_PROBE} RECOMPUTE=${RECOMPUTE} NUM_LAYERS=${NUM_LAYERS:-} QWEN3_SIZE=${QWEN3_SIZE}
          HF_HOME=${HF_HOME} HF_HUB_OFFLINE=${HF_HUB_OFFLINE}
          FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 FI_EFA_FORK_SAFE=1
          NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET NCCL_SOCKET_IFNAME=^docker,lo,veth ;
          torchrun --nnodes=${NNODES} --nproc_per_node=${GPUS_PER_NODE}
          --node_rank=${R} --master_addr=${JOB}-0.${JOB}.${NS}.svc.cluster.local
          --master_port=${PORT} ${BENCH_PY} > ${LOGDIR}/rank-${R}.log 2>&1 ${EPILOGUE}
      resources:
        requests: {nvidia.com/gpu: ${GPUS_PER_NODE}, vpc.amazonaws.com/efa: ${EFA_PER_NODE}}
        limits:   {nvidia.com/gpu: ${GPUS_PER_NODE}, vpc.amazonaws.com/efa: ${EFA_PER_NODE}}
      ${SECURITY_LINE}
      volumeMounts:
        - {name: fsx, mountPath: /fsx}
        - {name: shmem, mountPath: /dev/shm}
        ${GDR_MOUNT_LINE}
  volumes:
    - name: fsx
      ${FSX_VOLUME_SRC}
    - name: shmem
      emptyDir: {medium: Memory, sizeLimit: 32Gi}
    ${GDR_VOLUME_LINE}
EOF
}

for r in $(seq 0 $(( NNODES - 1 ))); do launch_pod "$r"; done
echo "   launched ${NNODES} pods: ${JOB}-0..$(( NNODES - 1 ))"
echo "   tail rank-0:  ${LOGDIR}/rank-0.log   STATUS: ${RUN_DIR}/STATUS"
