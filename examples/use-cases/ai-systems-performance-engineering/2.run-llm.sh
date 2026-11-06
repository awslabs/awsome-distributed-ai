#!/usr/bin/env bash
# Invoke inside an explicitly assigned Slurm allocation. No implicit sbatch.
set -euo pipefail
lab=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${SLURM_JOB_ID:?Run inside an assigned Slurm allocation}"
: "${SLURM_JOB_NUM_NODES:?}" "${GPUS_PER_NODE:?Set the assigned visible GPU count per node}"
: "${LAB_IMAGE:?Set the pinned training squashfs path}"
: "${LLM_DATA_DIR:?Set the prepared model/tokens directory}"
: "${LLM_RESULTS_DIR:?Set the result root on shared storage, or explicitly use node-local output}"
: "${NCCL_SOCKET_IFNAME:?Set the verified private interface}"
: "${SLURM_JOB_CPUS_PER_NODE:?}"
cpus_per_node=$(PYTHONPATH="$lab/lib" python3 -c 'import os; from serve_llm import allocation_cpus; print(allocation_cpus(os.environ["SLURM_JOB_CPUS_PER_NODE"], int(os.environ["SLURM_JOB_NUM_NODES"])))')
[[ $GPUS_PER_NODE =~ ^[1-9][0-9]*$ ]] || { echo 'invalid GPU count' >&2; exit 2; }
for path in "$lab" "$LLM_DATA_DIR" "$LLM_RESULTS_DIR"; do
    [[ "$path" != *[[:space:],:]* ]] || { echo 'mount paths cannot contain whitespace, comma or colon' >&2; exit 2; }
done
mounts="$lab:/opt/aim347,$LLM_DATA_DIR:/data,$LLM_RESULTS_DIR:/results,$LLM_RESULTS_DIR:$LLM_RESULTS_DIR"
if [[ -n ${LLM_PYTHON_HEADERS_DIR:-} ]]; then
    [[ "$LLM_PYTHON_HEADERS_DIR" != *[[:space:],:]* ]] || exit 2
    test -f "$LLM_PYTHON_HEADERS_DIR/usr/include/python3.10/Python.h"
    mounts+=",$LLM_PYTHON_HEADERS_DIR/usr/include/python3.10:/usr/include/python3.10"
    mounts+=",$LLM_PYTHON_HEADERS_DIR/usr/include/x86_64-linux-gnu/python3.10:/usr/include/x86_64-linux-gnu/python3.10"
fi
if [[ -n ${LLM_COMPILER_CACHE_DIR:-} ]]; then
    [[ "$LLM_COMPILER_CACHE_DIR" != *[[:space:],:]* ]] || exit 2
    mounts+=",$LLM_COMPILER_CACHE_DIR:/compiler-cache"
    export TORCHINDUCTOR_CACHE_DIR=/compiler-cache/inductor TRITON_CACHE_DIR=/compiler-cache/triton
fi
export TORCHINDUCTOR_COMPILE_THREADS=1
first_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_ADDR=$(scontrol show node "$first_node" -o | tr ' ' '\n' | sed -n 's/^NodeAddr=//p')
[[ -n $MASTER_ADDR ]] || { echo 'missing Slurm node address' >&2; exit 2; }
export MASTER_ADDR MASTER_PORT=${MASTER_PORT:-29547} GPUS_PER_NODE
export NNODES=$SLURM_JOB_NUM_NODES
# ENV logs "NCCL_PROTO set by environment to Simple" on every rank of an EFA run.
# A caller's NCCL_DEBUG (for example `NCCL_DEBUG=info bash tools/run.sh efa`) and
# NCCL_DEBUG_SUBSYS are kept as given; NCCL 2.30.4-1 compares the level case-insensitively
# (src/debug.cc, strcasecmp). Only NCCL's levels pass; the log stays in the stage's launch.log.
case ${NCCL_DEBUG:-INFO} in
    [Vv][Ee][Rr][Ss][Ii][Oo][Nn]|[Ww][Aa][Rr][Nn]|[Ii][Nn][Ff][Oo]|[Aa][Bb][Oo][Rr][Tt]|[Tt][Rr][Aa][Cc][Ee]) ;;
    *) echo "NCCL_DEBUG must be one of VERSION, WARN, INFO, ABORT, TRACE (it says '$NCCL_DEBUG')" >&2; exit 2 ;;
esac
export NCCL_DEBUG=${NCCL_DEBUG:-INFO} NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET,ENV}
# On g6e.48xlarge (test events; AIM347_INSTANCE_TYPE comes from aim347.env), the plugin's
# RDMA protocol never completes the first large collective (aws-ofi-nccl 1.19.0, fabric efa,
# no GPUDirect RDMA). SENDRECV, the plugin's own choice for g6e.48xlarge, completes it.
ofi_protocol=RDMA
[[ ${AIM347_INSTANCE_TYPE:-} != g6e.48xlarge ]] || ofi_protocol=SENDRECV
case ${LLM_TRANSPORT:-efa} in
    # Under RDMA, NCCL runs all-reduce from 64 KiB to 8 MiB on two g7.48xlarge
    # as LL, which is many times slower there than Simple (numbers in
    # lib/instance-type.sh). NCCL_PROTO=^LL would enable LL128 over PCIe.
    # An NCCL_PROTO set by the caller is kept; an empty one lets NCCL choose.
    efa) export FI_PROVIDER=efa NCCL_NET='AWS Libfabric' OFI_NCCL_PROTOCOL=$ofi_protocol NCCL_PROTO=${NCCL_PROTO-Simple}
         [[ -n $NCCL_PROTO ]] || unset NCCL_PROTO
         unset NCCL_NET_PLUGIN NCCL_GIN_PLUGIN ;;
    socket) export FI_PROVIDER=efa NCCL_NET=Socket OFI_NCCL_PROTOCOL=RDMA NCCL_NET_PLUGIN=none NCCL_GIN_PLUGIN=none ;;
    *) printf 'LLM_TRANSPORT must be efa or socket\n' >&2; exit 2 ;;
esac
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
# A named container stays unpacked for the rest of the job, so later steps in
# the same allocation skip the image import.
container_name=()
if [[ -n ${LLM_CONTAINER_NAME:-} ]]; then
    [[ $LLM_CONTAINER_NAME =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'invalid LLM_CONTAINER_NAME' >&2; exit 2; }
    container_name=(--container-name="$LLM_CONTAINER_NAME")
fi
exec srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 \
    --cpus-per-task="$cpus_per_node" --mpi=none --cpu-bind=none \
    --container-image="$LAB_IMAGE" ${container_name[@]+"${container_name[@]}"} \
    --container-mounts="$mounts" \
    --container-workdir=/opt/aim347 --no-container-remap-root \
    --container-env=NCCL_SOCKET_IFNAME,NCCL_DEBUG,NCCL_DEBUG_SUBSYS,NCCL_PROTO,NCCL_ALGO,FI_PROVIDER,NCCL_NET,NCCL_NET_PLUGIN,NCCL_GIN_PLUGIN,OFI_NCCL_PROTOCOL,MASTER_ADDR,MASTER_PORT,GPUS_PER_NODE,NNODES,TORCHINDUCTOR_COMPILE_THREADS,TORCHINDUCTOR_CACHE_DIR,TRITON_CACHE_DIR \
    bash lib/llm-rank.sh "$@"
