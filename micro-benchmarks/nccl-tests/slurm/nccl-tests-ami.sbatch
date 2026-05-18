#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --job-name=nccl-all_reduce_perf # name of your job
#SBATCH --nodes=2 # number of nodes to use, 24 p4d(e) = 192 A100 GPUs
#SBATCH --ntasks-per-node 8 # Number of GPU per node
#SBATCH --output %x_%j.out
#SBATCH --error %x_%j.err
#SBATCH --exclusive

set -ex

# This script is designed to run by default on the Deep Learning AMI, Ubuntu 20.04
# See https://aws.amazon.com/releasenotes/aws-deep-learning-base-gpu-ami-ubuntu-20-04/
ALL_REDUCE_BINARY=${1:-/usr/local/cuda-13.0/efa/test-cuda-13.0/all_reduce_perf}
ADDITIONAL_LD_LIBRARY_PATH=${2:-/usr/local/cuda-13.0/lib}

# Get Hostname to Instance ID mapping
mpirun -N 1 bash -c 'echo $(hostname) ➡️ $(cat /sys/devices/virtual/dmi/id/board_asset_tag | tr -d " ")'


# run all_reduce test
mpirun -n $((8 * SLURM_JOB_NUM_NODES)) -N 8 \
        -x FI_PROVIDER=efa \
	-x FI_EFA_FORK_SAFE=1 \
	-x LD_LIBRARY_PATH=$ADDITIONAL_LD_LIBRARY_PATH:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/opt/amazon/ofi-nccl/lib:/usr/local/lib:/usr/lib:$LD_LIBRARY_PATH \
	-x NCCL_DEBUG=INFO \
	-x NCCL_SOCKET_IFNAME=^docker,lo,veth \
	--bind-to none ${ALL_REDUCE_BINARY} -b 8 -e 16G -f 2 -g 1 -c 1 -n 100

