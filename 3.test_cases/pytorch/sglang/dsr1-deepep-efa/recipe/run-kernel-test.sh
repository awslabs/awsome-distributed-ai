#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Run one DeepEP kernel test inside the image on ONE node. Run it on every node
# with the same TEST and WORLD_SIZE, varying RANK.
#
# This is a smoke test that the EFA transport in the image works end to end
# before spending time on a 640 GB model load. For the full kernel benchmark
# matrix (Slurm / EKS launchers, 1-32 nodes) use
# micro-benchmarks/expert-parallelism/deepep-benchmark instead.
#
# Usage:
#   source setup/env_vars
#   recipe/run-kernel-test.sh <TEST> <RANK> [extra test args...]
#     TEST = intranode | internode | low_latency
#     RANK = this node's rank, 0..NUM_NODES-1  (rank 0 must be NODE_0_IP)
#
# WORLD_SIZE here is the NODE count, matching what DeepEP's tests expect; each
# node spawns one process per GPU. intranode is single-node (NVLink only, no
# NVSHMEM) and is forced to WORLD_SIZE=1 below; internode and low_latency use
# NVSHMEM over EFA/libfabric and take the node count from NUM_NODES.

set -euo pipefail

TEST=${1:?need TEST: intranode | internode | low_latency}
RANK=${2:?need RANK (0..NUM_NODES-1)}
shift 2

: "${IMAGE_URI:?source setup/env_vars first}"
: "${NODE_0_IP:?source setup/env_vars first}"
: "${IFACE:?source setup/env_vars first}"
PORT=${MASTER_PORT:-8361}

case "$TEST" in
    intranode|internode|low_latency) ;;
    *) echo "ERROR: TEST must be intranode, internode or low_latency" >&2; exit 1 ;;
esac

# WORLD_SIZE is a NODE count, and intranode must get 1 regardless of NUM_NODES.
# DeepEP's tests/utils.py::init_dist computes
#   world_size = int(os.getenv('WORLD_SIZE')) * num_local_ranks
# so with setup/env_vars sourced (NUM_NODES=2) the documented single-node command
# `run-kernel-test.sh intranode 0` opens a 16-rank rendezvous with only 8
# processes present and blocks in init_process_group forever -- and this is the
# pre-flight check users run BEFORE the 640 GB model load, so it has to be the
# one thing that cannot hang.
if [[ "$TEST" == "intranode" ]]; then
    WS=1
else
    WS=${NUM_NODES:-2}
fi

# CUDA VMM follows the kernel regime:
#   - low_latency: VMM must be ENABLED, else the RDMA buffer cudaMemset fails
#     with "invalid argument" (deep_ep.cpp:371). Measured on p5, 2 nodes: exit 1
#     with that exact error when VMM is off. This is the direction that matters.
#   - normal (intranode/internode): VMM off is belt-and-braces, NOT a
#     requirement. internode also runs clean with VMM enabled (60.97 vs 61.07
#     GB/s BF16 dispatch RDMA, no NVSHMEM/topology/transport-map error). Kept
#     because it costs nothing and an NVSHMEM init failure was seen during
#     bring-up on a host that has not been re-identified. See benchmarks/README.
if [[ "$TEST" == "low_latency" ]]; then
    VMM_ENV=()                                    # leave CUDA VMM enabled
else
    VMM_ENV=(-e NVSHMEM_DISABLE_CUDA_VMM=1)
fi

# Belt and braces: preload the v3.7.0 host lib so it wins over any copy the base
# image's torch might pull in. The Dockerfile already overwrites the pip copy, so
# this only matters if you rebuild on a base image that ships a different one.
NVSHMEM_HOST_LIB=/opt/nvshmem/install/lib/libnvshmem_host.so.3.7.0

# NCCL gets the exclusion form below (repo convention, matches
# micro-benchmarks/nccl-tests): instance-type-agnostic, so it needs no NIC name and
# keeps working on p5, p5en, p6-b200, p6-b300 and whatever comes next. Gloo does not
# support `^` exclusion -- it takes concrete comma-separated names only -- so it gets
# $IFACE, which setup/env_vars derives from the default route.
set -x
docker run --rm \
    --gpus all --network host --ipc host --privileged \
    --device /dev/infiniband --device /dev/gdrdrv \
    --ulimit memlock=-1 --shm-size 32g \
    -e MASTER_ADDR="$NODE_0_IP" -e MASTER_PORT="$PORT" \
    -e WORLD_SIZE="$WS" -e RANK="$RANK" \
    -e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^docker,lo,veth}" -e GLOO_SOCKET_IFNAME="$IFACE" \
    -e FI_PROVIDER=efa -e FI_EFA_USE_DEVICE_RDMA=1 \
    -e NVSHMEM_REMOTE_TRANSPORT=libfabric \
    -e NVSHMEM_LIBFABRIC_PROVIDER=efa \
    -e NVSHMEM_NETDEVS_POLICY=EXTERNAL_SHARING_PCIE_SWITCH_NIC_EXCLUSIVE \
    "${VMM_ENV[@]}" \
    -e NVSHMEM_BOOTSTRAP=UID \
    -e LD_PRELOAD="$NVSHMEM_HOST_LIB" \
    --entrypoint bash "$IMAGE_URI" \
    -c 'cd /opt/DeepEP && python3 tests/test_'"${TEST}"'.py "$@"' -- "$@"
