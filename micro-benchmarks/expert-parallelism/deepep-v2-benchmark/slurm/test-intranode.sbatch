#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Single-node DeepEP V2 intranode test via enroot+pyxis.
#
# Submit:
#   sbatch test-intranode.sbatch
#
#SBATCH --job-name=deepep_intranode
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --wait-all-nodes=1
#SBATCH -t 30
#SBATCH -o %x_%j.out
set +u

export PATH=/opt/slurm/bin:$PATH
: "${SQSH:=$HOME/deepep-v2.sqsh}"
[ -f "$SQSH" ] || { echo "ERROR: container image not found: $SQSH (build and enroot-import it first; see README.md)"; exit 1; }
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(( 29500 + RANDOM % 5000 ))
export WORLD_SIZE=1

srun --mpi=pmix --cpu-bind=none \
    --container-image "$SQSH" \
    --container-mounts /dev/infiniband:/dev/infiniband,/dev/gdrdrv:/dev/gdrdrv \
    bash -lc '
        set -x
        export RANK=0

        export EP_NCCL_ROOT_DIR=$NCCL_HOME
        export TORCHDYNAMO_DISABLE=1
        export TORCH_COMPILE_DISABLE=1

        export NCCL_GIN_TYPE=5
        export NCCL_SYM_GIN_KERNELS_ENABLE=0

        echo "=== container: $(hostname) intranode WORLD_SIZE=$WORLD_SIZE ==="
        python3 /opt/amazon/deepep/tests/elastic/test_ep.py \
            --test-first-only --num-tokens=8192 --num-topk=8 \
            --num-sms=64
        rc=$?
        echo "RUN_EXIT=$rc"
        exit $rc
    '
