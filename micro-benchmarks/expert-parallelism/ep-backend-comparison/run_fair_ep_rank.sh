#!/usr/bin/env bash
set -euo pipefail

: "${EP_ARM:?Set EP_ARM}"
: "${EP_NODES:?Set EP_NODES}"
: "${EP_SERVICE:?Set EP_SERVICE}"
: "${EP_MASTER_ADDR:?Set EP_MASTER_ADDR}"
: "${POD_NAME:?Set POD_NAME}"
: "${EP_RUN_INDEX:?Set EP_RUN_INDEX}"
: "${EP_DISPATCH_DTYPES:?Set EP_DISPATCH_DTYPES}"
: "${EP_WARMUPS:=20}"
: "${EP_ITERATIONS:=100}"
: "${EP_SEED:=20260824}"
: "${EP_NCCL_DEBUG:=WARN}"

ordinal="${POD_NAME##*-}"
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_SOCKET_IFNAME='^lo,docker,veth'
export NCCL_NET_PLUGIN=ofi
export NCCL_DEBUG="${EP_NCCL_DEBUG}"
export CUDA_DEVICE_MAX_CONNECTIONS=1

case "${EP_ARM}" in
    uccl)
        export PER_EXPERT_BATCHING=1
        export UCCL_SOCKET_IFNAME='^lo,docker,veth'
        ;;
    deepep-v1-nvshmem)
        export NVSHMEM_REMOTE_TRANSPORT=libfabric
        export NVSHMEM_LIBFABRIC_PROVIDER=efa
        export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME='^lo,docker,veth'
        export NVSHMEM_NETDEVS_POLICY=EXTERNAL_SHARING_PCIE_SWITCH_NIC_EXCLUSIVE
        ;;
    deepep-v2-gin-gda)
        export FI_EFA_USE_HW_CNTR=1
        export NCCL_GIN_TYPE=5
        export NCCL_SYM_GIN_KERNELS_ENABLE=0
        export EP_BUFFER_DEBUG=1
        if [[ "${NCCL_DEBUG}" == INFO ]]; then
            export NCCL_DEBUG_SUBSYS=INIT,ENV,NET
            export FI_LOG_LEVEL=info
            export FI_LOG_PROV=efa
            export FI_LOG_SUBSYS=cntr
        fi
        ;;
    *)
        printf 'Unsupported EP_ARM=%s\n' "${EP_ARM}" >&2
        exit 2
        ;;
esac

printf 'ADAI_FAIR_LAUNCH arm=%s nodes=%s ranks=%s run_index=%s_dimensionless dtype_order=%s warmups=%s_iterations measured=%s_iterations\n' \
    "${EP_ARM}" "${EP_NODES}" "$((EP_NODES * 8))" "${EP_RUN_INDEX}" \
    "${EP_DISPATCH_DTYPES}" "${EP_WARMUPS}" "${EP_ITERATIONS}"

set +e
torchrun \
    --nnodes="${EP_NODES}" \
    --nproc-per-node=8 \
    --node-rank="${ordinal}" \
    --master-addr="${EP_MASTER_ADDR}" \
    --master-port=29400 \
    /opt/benchmark/fair_ep_benchmark.py \
    --arm="${EP_ARM}" \
    --tokens=128 \
    --hidden=7168 \
    --top-k=8 \
    --experts=256 \
    --seed="${EP_SEED}" \
    --warmups="${EP_WARMUPS}" \
    --iterations="${EP_ITERATIONS}" \
    --run-index="${EP_RUN_INDEX}" \
    --dispatch-dtypes="${EP_DISPATCH_DTYPES}"
status=$?
set -e

printf 'ADAI_FAIR_EXIT_STATUS=%s_dimensionless\n' "${status}"
if ((status == 0)); then
    printf 'ADAI_FAIR_COMPLETE\n'
else
    printf 'ADAI_FAIR_FAILED\n'
fi
sleep infinity
