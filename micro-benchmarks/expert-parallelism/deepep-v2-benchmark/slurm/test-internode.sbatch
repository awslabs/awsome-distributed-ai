#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Two-node DeepEP V2 internode test via enroot+pyxis.
#
# Submit:
#   sbatch test-internode.sbatch
#
#SBATCH --job-name=deepep_internode_efa_gda
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --wait-all-nodes=1
#SBATCH -t 30
#SBATCH -o %x_%j.out
set +u


export PATH=/opt/slurm/bin:$PATH
: "${SQSH:=$HOME/deepep-v2.sqsh}"
[ -f "$SQSH" ] || { echo "ERROR: container image not found: $SQSH (build and enroot-import it first; see README.md)"; exit 1; }
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=$(( 29500 + RANDOM % 5000 ))
export WORLD_SIZE=$SLURM_NNODES


srun --mpi=pmix --cpu-bind=none \
    --container-image "$SQSH" \
    --container-mounts /dev/infiniband:/dev/infiniband,/dev/gdrdrv:/dev/gdrdrv \
    bash -lc '
        set -x
        export RANK=${SLURM_PROCID:-0}     # per-NODE rank (0 or 1)

        # Pin the EFA-GDA GIN backend (type 5); unset, NCCL selects the CPU-proxy backend.
        export NCCL_GIN_TYPE=5
        export NCCL_SYM_GIN_KERNELS_ENABLE=0

        export TORCHDYNAMO_DISABLE=1
        export TORCH_COMPILE_DISABLE=1



        echo "=== container: $(hostname) node-RANK=$RANK WORLD_SIZE=$WORLD_SIZE MASTER=$MASTER_ADDR:$MASTER_PORT ==="
        python3 /opt/amazon/deepep/tests/elastic/test_ep.py \
            --test-first-only --num-tokens=8192 --num-topk=8 \
            --num-sms=12
        rc=$?
        echo "RUN_EXIT=$rc"
        exit $rc
    '
