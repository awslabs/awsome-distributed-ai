#!/bin/bash

#SBATCH --job-name=intranode
#SBATCH --nodes=1
#SBATCH --ntasks-per-node 1
#SBATCH --exclusive
#SBATCH --wait-all-nodes=1

export FI_PROVIDER=efa
export NVSHMEM_REMOTE_TRANSPORT=libfabric
export NVSHMEM_LIBFABRIC_PROVIDER=efa

mpirun -N 1 bash -c 'echo $(hostname): $(cat /sys/devices/virtual/dmi/id/board_asset_tag | tr -d " ")'

srun --mpi=pmix --cpu-bind=none --container-image ./deepep.sqsh bash -c 'python3 /DeepEP/tests/test_intranode.py'
