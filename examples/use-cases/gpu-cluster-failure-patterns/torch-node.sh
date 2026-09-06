#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
set -euo pipefail
: "${MASTER_ADDR:?Set the primary address of the first node}"
NODE_RANK=${NODE_RANK:-${SLURM_PROCID:?Set NODE_RANK outside Slurm}}
iface=${AIM344_SOCKET_IFNAME:-$(ip -4 route show default | awk 'NR == 1 {print $5}')}
[[ -n $iface ]] || exit 1
unset NCCL_NET NCCL_NET_PLUGIN NCCL_IB_DISABLE FI_EFA_IFACE FI_EFA_DEVICE_NAME FI_EFA_FORK_SAFE
export FI_PROVIDER=efa NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME="=$iface"
export OMP_NUM_THREADS=1
exec python3 -m torch.distributed.run --nnodes=2 --nproc-per-node=8 \
    --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port="${MASTER_PORT:-29534}" \
    /opt/aim344/workload.py "$@"
