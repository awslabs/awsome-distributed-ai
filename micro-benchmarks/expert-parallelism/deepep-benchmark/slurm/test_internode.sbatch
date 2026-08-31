#!/bin/bash

#SBATCH --job-name=internode
#SBATCH --nodes=2
#SBATCH --ntasks-per-node 1
#SBATCH --exclusive
#SBATCH --wait-all-nodes=1

export FI_PROVIDER=efa
export NVSHMEM_REMOTE_TRANSPORT=libfabric
export NVSHMEM_LIBFABRIC_PROVIDER=efa
export NVSHMEM_NETDEVS_POLICY=EXTERNAL_SHARING_PCIE_SWITCH_NIC_EXCLUSIVE

export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export WORLD_SIZE=$SLURM_NNODES

mpirun -N 1 bash -c 'echo $(hostname): $(cat /sys/devices/virtual/dmi/id/board_asset_tag | tr -d " ")'

srun --mpi=pmix --cpu-bind=none --container-image ./deepep.sqsh bash -c 'RANK=$SLURM_PROCID python3 /DeepEP/tests/test_internode.py'
