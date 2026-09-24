#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
set -euo pipefail
# shellcheck source=instance-type.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/instance-type.sh"
apply_g7_protocol
: "${MASTER_ADDR:?Set the primary address of the first node}"
NODE_RANK=${NODE_RANK:-${SLURM_PROCID:?Set NODE_RANK outside Slurm}}
GPUS_PER_NODE=${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | awk '/^GPU [0-9]+:/ {n++} END {print n+0}')}}
[[ $GPUS_PER_NODE =~ ^[1-9][0-9]*$ ]] || exit 1
iface=${AIM344_SOCKET_IFNAME:-$(ip -4 route show default | awk 'NR == 1 {print $5}')}
[[ -n $iface ]] || exit 1
unset NCCL_NET NCCL_NET_PLUGIN NCCL_IB_DISABLE FI_EFA_IFACE FI_EFA_DEVICE_NAME FI_EFA_FORK_SAFE
# A stand-in allocation may expose more host EFAs than the target instance.
# Preserve an explicit physical-device selection for every NCCL rank.
if [[ -n ${AIM344_EFA_IFACE:-} ]]; then export FI_EFA_IFACE=$AIM344_EFA_IFACE; fi
export FI_PROVIDER=efa NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME="=$iface"
export OMP_NUM_THREADS=1
exec python3 -m torch.distributed.run --nnodes=2 --nproc-per-node="$GPUS_PER_NODE" \
    --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port="${MASTER_PORT:-29534}" \
    /opt/aim344/workload.py "$@"
