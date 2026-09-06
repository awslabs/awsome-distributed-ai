#!/usr/bin/env bash
# shellcheck disable=SC2016
set -euo pipefail
# Container-side variables intentionally expand in the child shell.
# shellcheck disable=SC2016
source "${LAB_DIR:?}/lib/common.sh"
action=${1:?}
if [[ "$action" =~ ^v[0-3]$ ]]; then
    out="$LAB_DIR/results/$RUN_ID/$action"
    mkdir -p "$out"
    [[ ! -e "$out/summary.json" ]] || { echo 'Choose a new RUN_ID to preserve the completed run' >&2; exit 2; }
    started=$(date +%s.%N)
    trap 'status=$?; python3 "$LAB_DIR/lib/allocation.py" "$out" "$started" "$(date +%s.%N)" "$status" "$INSTANCE_TYPE"' EXIT
fi
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR
export MASTER_PORT=29547
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET FI_PROVIDER=efa
# Bind NCCL bootstrap to the assigned hosts' routable private interface.
: "${NCCL_SOCKET_IFNAME:?Set NCCL_SOCKET_IFNAME to the private NIC, for example =ens5}"
export NCCL_SOCKET_IFNAME
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# Keep PyTorch's NCCL aligned with the pinned nccl-tests and OFI plugin stack.
mounts="$LAB_DIR:/opt/aim347,$DATA_DIR:/data"
args=(--container-image="$LAB_IMAGE" --container-mounts="$mounts" --container-workdir=/opt/aim347 --cpu-bind=none)
if [[ "$action" == gemm ]]; then
    srun --ntasks=2 "${args[@]}" bash -c 'nvidia-smi -q > /opt/aim347/results/"$RUN_ID"/gpu-"$SLURM_PROCID".txt; /usr/local/bin/aim347-gemm > /opt/aim347/results/"$RUN_ID"/gemm-node-"$SLURM_PROCID".json'
    exit
fi
if [[ "$action" == serving ]]; then
    # Independent tensor-parallel replicas, one per node, using the same allocation.
    srun --ntasks=2 --container-image="$VLLM_IMAGE" --container-mounts="$DATA_DIR:/data" --cpu-bind=none \
      vllm serve /data/serving-model --served-model-name aim347 --tensor-parallel-size 2 \
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
    srun --ntasks=2 "${args[@]}" fi_info -p efa
fi
if [[ "$action" == v0 || "$action" == v1 ]]; then
    # MPI tasks are ranks, so override both ntasks and ntasks-per-node.
    srun --ntasks=4 --ntasks-per-node=2 --mpi=pmix "${args[@]}" \
      env LD_PRELOAD=/opt/nccl/build/lib/libnccl.so /opt/nccl-tests/build/all_reduce_perf -b 8M -e 256M -f 2 -g 1 -w 5 -n 20 -c 1
fi
: "${DENSE_TFLOPS:?Run the GEMM step and set DENSE_TFLOPS in .env}"
export DENSE_TFLOPS
out="$LAB_DIR/results/$RUN_ID/$action"
mkdir -p "$out"
[[ ! -e "$out/summary.json" ]] || { echo 'Choose a new RUN_ID to avoid overwriting a completed run' >&2; exit 2; }
status=0
srun --ntasks=2 "${args[@]}" bash -c '
  if [[ "$INSTANCE_TYPE" == g7e.12xlarge ]]; then
    [[ "$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)" -eq 24 && "$SLURM_CPUS_ON_NODE" -eq 24 ]] || { echo "PCS core preflight failed" >&2; exit 1; }
  fi
  exec env LD_PRELOAD=/opt/nccl/build/lib/libnccl.so torchrun --nnodes=2 --nproc-per-node=2 --node-rank="$SLURM_PROCID" \
    --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
    lib/train.py --config="configs/'"$action"'.json" --data=/data/tokens \
    --output="results/$RUN_ID/'"$action"'" --instance-type="$INSTANCE_TYPE" --dense-tflops="$DENSE_TFLOPS" \
    --steps="$STEPS" --warmup="$WARMUP" --microbatch="$MICROBATCH" --cpu-rounds="$CPU_ROUNDS" \
    --pushgateway="$PUSHGATEWAY_URL" --run-id="$RUN_ID"
' || status=$?
# Record failed attempts too; no retry or automatic useful-work credit is invented.
exit "$status"
