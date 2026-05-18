#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --job-name=nccl-tests-ami # name of your job
#SBATCH --nodes=2 # number of nodes to use, 
#SBATCH --ntasks-per-node 8 # Number of GPU per node (e.g 8 H200 for p5en.48xlarge)
#SBATCH --output logs/%x_%j.out
#SBATCH --error logs/%x_%j.err
#SBATCH --exclusive
#SBATCH --wait-all-nodes=1

set -ex

# Create logs directory if it doesn't exist
mkdir -p logs

### Disable hyperthreading by setting the tasks per core to 1
#SBATCH --ntasks-per-core=1

# This script is designed to run by default on the Deep Learning AMI, Ubuntu 20.04
# See https://aws.amazon.com/releasenotes/aws-deep-learning-base-gpu-ami-ubuntu-20-04/

# Supported NCCL collective operations:
# - allreduce     : AllReduce collective (default)
# - allgather     : AllGather collective  
# - reducescatter : ReduceScatter collective
# - alltoall      : AllToAll collective
# - gather        : Gather collective
# - reduce        : Reduce collective
# - scatter       : Scatter collective
# - broadcast     : Broadcast collective
# - hypercube     : Hypercube collective
# - sendrecv      : SendRecv point-to-point

TEST_TYPE=${1:-allreduce}
ADDITIONAL_LD_LIBRARY_PATH=${2:-/usr/local/cuda-12.9/lib64}
SPLIT_MASK=${3:-0x0}
TOPO_SORTED_FILE=${4:-""}
ENABLE_NCCL_DEBUG=${5:-false}

# Set binary path based on test type
CUDA_TEST_DIR="/opt/nccl-tests/build"
case ${TEST_TYPE} in
    allreduce)
        TEST_BINARY="${CUDA_TEST_DIR}/all_reduce_perf"
        ;;
    allgather)
        TEST_BINARY="${CUDA_TEST_DIR}/all_gather_perf"
        ;;
    reducescatter)
        TEST_BINARY="${CUDA_TEST_DIR}/reduce_scatter_perf"
        ;;
    alltoall)
        TEST_BINARY="${CUDA_TEST_DIR}/alltoall_perf"
        ;;
    gather)
        TEST_BINARY="${CUDA_TEST_DIR}/gather_perf"
        ;;
    reduce)
        TEST_BINARY="${CUDA_TEST_DIR}/reduce_perf"
        ;;
    scatter)
        TEST_BINARY="${CUDA_TEST_DIR}/scatter_perf"
        ;;
    broadcast)
        TEST_BINARY="${CUDA_TEST_DIR}/broadcast_perf"
        ;;
    hypercube)
        TEST_BINARY="${CUDA_TEST_DIR}/hypercube_perf"
        ;;
    sendrecv)
        TEST_BINARY="${CUDA_TEST_DIR}/sendrecv_perf"
        ;;
    *)
        echo "Error: Unsupported test type '${TEST_TYPE}'"
        echo "Supported types: allreduce, allgather, reducescatter, alltoall, gather, reduce, scatter, broadcast, hypercube, sendrecv"
        exit 1
        ;;
esac

echo "Running NCCL ${TEST_TYPE} test in ami with split mask ${SPLIT_MASK}"
echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting NCCL ${TEST_TYPE} test"


# Get Hostname to Instance ID mapping
mpirun -N 1 bash -c '
  HOSTNAME=$(hostname)
  ASSET_TAG=$(cat /sys/devices/virtual/dmi/id/board_asset_tag | tr -d " ")
  NCCL_VER=$(strings /opt/nccl/build/lib/libnccl.so | grep "NCCL version" | head -1 | awk "{print \$3}" 2>/dev/null || echo "N/A")
  AWS_OFI_NCCL_VER=$(strings /opt/amazon/ofi-nccl/lib/libnccl-net.so | grep "NET/OFI Initializing aws-ofi-nccl" | awk "{print \$4}" 2>/dev/null || echo "N/A")
  EFA_VER=$(grep -E "(version|installer)" /opt/amazon/efa_installed_packages 2>/dev/null | head -1 || echo "N/A")
  echo "hostname=$HOSTNAME ➡️ $ASSET_TAG | NCCL: $NCCL_VER | AWS-OFI-NCCL: $AWS_OFI_NCCL_VER | EFA: $EFA_VER"
'
# In the echo statement above , don't remove the string hostname=, it is used in converting the nccl output to csv in process_nccl_results.sh


time_stamp=$(date +%Y%m%d_%H%M%S)

# Expand the compact node list into a full list of hostnames,
# and save it to a file.
scontrol show hostnames $SLURM_JOB_NODELIST > /tmp/nccl_test_ami_slurm_hostfile_${time_stamp}.txt


# Set up hostfile options based on topology sorting
if [ -n "$TOPO_SORTED_FILE" ]; then
    HOSTFILE="/tmp/nccl_test_ami_seq_hostfile_${time_stamp}.txt"
    rm -f "$HOSTFILE"

    # Filter TOPO_SORTED_FILE to only include hosts that exist in the slurm hostfile
    # and repeat each hostname SLURM_NTASKS_PER_NODE times for processes per node
    while read -r hostname; do
        if grep -q "^${hostname}$" /tmp/nccl_test_ami_slurm_hostfile_${time_stamp}.txt; then
            for i in $(seq 1 $SLURM_NTASKS_PER_NODE); do
                echo "$hostname" >> "$HOSTFILE"
            done
        fi
    done < $TOPO_SORTED_FILE

    echo "Created sequential hostfile with repeated hostnames (filtered to match SLURM allocation):"
    echo "Total lines: $(wc -l < "$HOSTFILE")"
    
    HOSTFILE_OPTS="--hostfile $HOSTFILE --mca rmaps seq"
else
    HOSTFILE_OPTS=""
fi



# Set NCCL debug flag conditionally
if [ "$ENABLE_NCCL_DEBUG" = "true" ]; then
    NCCL_DEBUG_FLAG="-x NCCL_DEBUG=INFO"
else
    NCCL_DEBUG_FLAG=""
fi

mpirun -n $((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES)) -N $SLURM_NTASKS_PER_NODE \
        -x LD_LIBRARY_PATH=/opt/nccl/build/lib:$ADDITIONAL_LD_LIBRARY_PATH:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/opt/amazon/ofi-nccl/lib:/usr/local/lib:/usr/lib:$LD_LIBRARY_PATH \
        -x NCCL_SOCKET_IFNAME=^docker,lo,veth \
        -x NCCL_TESTS_SPLIT_MASK=${SPLIT_MASK} \
        ${NCCL_DEBUG_FLAG} \
        ${HOSTFILE_OPTS} \
        --bind-to none ${TEST_BINARY} -b 8 -e 16G -f 2 -g 1 -c 1 -n 100
