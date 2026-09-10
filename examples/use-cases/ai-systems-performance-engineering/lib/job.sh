#!/usr/bin/env bash
# shellcheck disable=SC2016
set -euo pipefail
# Container-side variables intentionally expand in the child shell.
# shellcheck disable=SC2016
source "${LAB_DIR:?}/lib/common.sh"
action=${1:?}
# shellcheck source=instance-type.sh
source "$LAB_DIR/lib/instance-type.sh"
export INSTANCE_TYPE=$(detect_instance_type)
apply_g7_protocol
GPU_COUNT=unknown
# One launcher per node receives the whole CPU allocation. MPI GPU ranks retain
# the batch allocation's CPUs per task.
NODE_CPUS=${SLURM_CPUS_ON_NODE:?}
srun --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 --cpus-per-task="$NODE_CPUS" --mpi=none --cpu-bind=none mkdir -p "$LAB_DIR/results/$RUN_ID"
scontrol show job "$SLURM_JOB_ID"
srun --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 --cpus-per-task="$NODE_CPUS" --mpi=none --cpu-bind=none bash "$LAB_DIR/lib/allocation-evidence.sh"
if [[ "$action" =~ ^v[0-3]$ ]]; then
    out="$LAB_DIR/results/$RUN_ID/$action"
    mkdir -p "$out"
    [[ ! -e "$out/summary.json" ]] || { echo 'Choose a new RUN_ID to preserve the completed run' >&2; exit 2; }
    started=$(date +%s.%N)
    trap 'status=$?; python3 "$LAB_DIR/lib/allocation.py" "$out" "$started" "$(date +%s.%N)" "$status" "$INSTANCE_TYPE" "$GPU_COUNT"' EXIT
fi
NNODES=${SLURM_JOB_NUM_NODES:?Run inside a Slurm allocation}
inventory="$LAB_DIR/results/$RUN_ID/$action-resources.jsonl"
srun --ntasks="$NNODES" --ntasks-per-node=1 --cpus-per-task="$NODE_CPUS" --cpu-bind=none bash "$LAB_DIR/lib/resources.sh" > "$inventory"
counts=$(python3 "$LAB_DIR/lib/resource_plan.py" "$inventory" --nodes "$NNODES" --output "$LAB_DIR/results/$RUN_ID/$action-resources.json")
read -r NNODES GPUS_PER_NODE GPU_COUNT <<< "$counts"
export NNODES GPUS_PER_NODE GPU_COUNT
first_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_ADDR=$(scontrol show node "$first_node" -o | tr ' ' '\n' | sed -n 's/^NodeAddr=//p')
[[ -n $MASTER_ADDR ]] || { echo 'Slurm did not return the first node address' >&2; exit 2; }
export MASTER_ADDR
export MASTER_PORT=29547
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET FI_PROVIDER=efa
# Bind NCCL bootstrap to the assigned hosts' routable private interface.
: "${NCCL_SOCKET_IFNAME:?Set NCCL_SOCKET_IFNAME to the private NIC, for example =ens5}"
export NCCL_SOCKET_IFNAME
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# Keep PyTorch's NCCL aligned with the pinned nccl-tests and OFI plugin stack.
mounts="$LAB_DIR:/opt/aim347,$DATA_DIR:/data"
args=(--container-image="$LAB_IMAGE" --container-mounts="$mounts" --container-workdir=/opt/aim347
    --container-env=NCCL_SOCKET_IFNAME,FI_EFA_IFACE,OFI_NCCL_PROTOCOL --no-container-remap-root --cpu-bind=none)
if [[ "$action" == gemm ]]; then
    srun --ntasks="$NNODES" --ntasks-per-node=1 --cpus-per-task="$NODE_CPUS" "${args[@]}" bash -c 'nvidia-smi -q > /opt/aim347/results/"$RUN_ID"/gpu-"$SLURM_PROCID".txt; /usr/local/bin/aim347-gemm > /opt/aim347/results/"$RUN_ID"/gemm-node-"$SLURM_PROCID".json'
    if [[ ${LAB_NODE_LOCAL:-0} == 1 ]]; then python3 "$LAB_DIR/lib/gather-node-results.py" "$LAB_DIR/results/$RUN_ID"; fi
    exit
fi
if [[ "$action" == bandwidth ]]; then
    srun --ntasks="$NNODES" --ntasks-per-node=1 --cpus-per-task="$NODE_CPUS" "${args[@]}" bash -c '
      for ((gpu=0; gpu<GPUS_PER_NODE; gpu++)); do
        /usr/local/bin/aim347-dram "$gpu" > /opt/aim347/results/"$RUN_ID"/dram-node-"$SLURM_PROCID"-gpu-"$gpu".json
      done'
    if [[ ${LAB_NODE_LOCAL:-0} == 1 ]]; then python3 "$LAB_DIR/lib/gather-node-results.py" "$LAB_DIR/results/$RUN_ID"; fi
    exit
fi
if [[ "$action" == serving ]]; then
    # Independent tensor-parallel replicas, one per node, using the same allocation.
    srun --ntasks="$NNODES" --ntasks-per-node=1 --cpus-per-task="$NODE_CPUS" --container-image="$VLLM_IMAGE" --container-mounts="$DATA_DIR:/data" \
      --container-env=NCCL_SOCKET_IFNAME,FI_EFA_IFACE,OFI_NCCL_PROTOCOL --cpu-bind=none \
      vllm serve /data/serving-model --served-model-name aim347 --tensor-parallel-size "$GPUS_PER_NODE" \
      --dtype bfloat16 --max-model-len 2048 --gpu-memory-utilization 0.8 --host 0.0.0.0 --port 8000
    exit
fi
network=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["network"])' "$LAB_DIR/configs/$action.json")
if [[ "$network" == Socket ]]; then
    export NCCL_NET=Socket NCCL_NET_PLUGIN=none
else
    [[ "$network" == "AWS Libfabric" ]] || { echo "Unknown NCCL network: $network" >&2; exit 2; }
    export NCCL_NET="$network"
    unset NCCL_NET_PLUGIN
    srun --ntasks="$NNODES" --ntasks-per-node=1 --cpus-per-task="$NODE_CPUS" "${args[@]}" fi_info -p efa
fi
if [[ "$action" == v0 || "$action" == v1 ]]; then
    # MPI tasks are ranks, so override both ntasks and ntasks-per-node.
    srun --ntasks="$GPU_COUNT" --ntasks-per-node="$GPUS_PER_NODE" --mpi=pmix "${args[@]}" \
      bash lib/nccl-rank.sh
fi
: "${DENSE_TFLOPS:?Run the GEMM step and set DENSE_TFLOPS in .env}"
export DENSE_TFLOPS
out="$LAB_DIR/results/$RUN_ID/$action"
mkdir -p "$out"
[[ ! -e "$out/summary.json" ]] || { echo 'Choose a new RUN_ID to avoid overwriting a completed run' >&2; exit 2; }
status=0
srun --ntasks="$NNODES" --ntasks-per-node=1 --cpus-per-task="$NODE_CPUS" "${args[@]}" bash -c '
  exec env LD_PRELOAD=/opt/nccl/build/lib/libnccl.so torchrun --nnodes="$NNODES" --nproc-per-node="$GPUS_PER_NODE" --node-rank="$SLURM_PROCID" \
    --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
    lib/train.py --config="configs/'"$action"'.json" --data=/data/tokens \
    --output="results/$RUN_ID/'"$action"'" --instance-type="$INSTANCE_TYPE" --dense-tflops="$DENSE_TFLOPS" \
    --steps="$STEPS" --warmup="$WARMUP" --microbatch="$MICROBATCH" --cpu-rounds="$CPU_ROUNDS" \
    --pushgateway="$PUSHGATEWAY_URL" --run-id="$RUN_ID"
' || status=$?
# Record failed attempts too; no retry or automatic useful-work credit is invented.
exit "$status"
