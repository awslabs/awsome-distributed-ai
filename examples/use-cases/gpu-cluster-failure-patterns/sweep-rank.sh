#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
set -euo pipefail
# Apply the same environment to baseline, injection and recovery. Plugin
# availability is the independent variable; do not force NCCL_NET=Socket.
unset NCCL_NET NCCL_NET_PLUGIN NCCL_IB_DISABLE FI_EFA_IFACE FI_EFA_DEVICE_NAME
# A stand-in allocation may expose more host EFAs than the target instance.
# Preserve an explicit physical-device selection for every NCCL rank.
if [[ -n ${AIM344_EFA_IFACE:-} ]]; then export FI_EFA_IFACE=$AIM344_EFA_IFACE; fi
export FI_PROVIDER=efa NCCL_DEBUG=INFO
# Select after the Enroot hook, only inside an MPI rank. Exporting a PMIX_ name
# around --mpi=none steps makes the hook try to mount nonexistent PMIx paths.
export PMIX_MCA_gds=hash
# MPI carries process setup only over TCP, so MPI's OFI MTL cannot mask the
# NCCL transport result with an independent OFI error.
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=tcp,self
# EKS nodes may also expose a link-local pod-identity interface. Select the
# primary routed interface explicitly for MPI and NCCL socket bootstrap.
iface=${AIM344_SOCKET_IFNAME:-$(ip -4 route show default | awk 'NR == 1 {print $5}')}
[[ -n $iface ]] || { echo 'Set AIM344_SOCKET_IFNAME to the primary IP interface.' >&2; exit 1; }
export NCCL_SOCKET_IFNAME="=$iface" OMPI_MCA_btl_tcp_if_include="$iface"
unset OMPI_MCA_btl_tcp_if_exclude
exec /opt/nccl-tests/build/all_reduce_perf -b 8 -e 2G -f 2 -g 1 -c 1
