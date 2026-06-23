#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Disaggregated prefill-decode on P6e-GB200 (Slurm), heterogeneous prefill + decode nodes.
# Co-resident by default (KV over NVLink). For split across two UltraServers, set --nodes
# accordingly and ensure NIXL uses the LIBFABRIC/EFA path (gated on #1609).
#SBATCH --job-name=gb200-disagg
#SBATCH --nodes=18
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --output %x_%j.out
#SBATCH --error  %x_%j.err
set -euo pipefail
: "${APPS_PATH:=/fsx}"
: "${IMAGE:=$APPS_PATH/gb200-disagg-arm64.sqsh}"
: "${MODEL:=deepseek-ai/DeepSeek-V3}"
: "${KV_TRANSFER_PORT:=7100}"

export FI_PROVIDER=efa FI_HMEM_CUDA_USE_DMABUF=0   # NIXL #1609 workaround

# First half of nodes = prefill, second half = decode. NIXL discovers peers via the
# proxy; PREFILL_IP/DECODE_IP are derived from the allocation.
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
half=$(( ${#nodes[@]} / 2 ))
export PREFILL_IP="${nodes[0]}" DECODE_IP="${nodes[$half]}" KV_TRANSFER_PORT

srun --container-image "$IMAGE" --cpu-bind=none bash -lc '
  cd /opt/disagg
  ROLE=$([ "$SLURM_NODEID" -lt '"$half"' ] && echo prefill || echo decode)
  echo "node $SLURM_NODEID role=$ROLE"
  vllm serve '"$MODEL"' \
    --kv-transfer-config "$(cat nixl-config.yaml | python3 -c "import sys,yaml,json;print(json.dumps(yaml.safe_load(sys.stdin)))")" \
    --disagg-role "$ROLE"
'
