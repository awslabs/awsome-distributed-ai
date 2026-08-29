#!/usr/bin/env bash
set -euo pipefail

: "${PR5_DOMAINS:?}"
: "${PR5_EXPECT:?}"
: "${PR5_SERVICE:?}"
: "${PR5_MASTER_ADDR:?}"
: "${POD_NAME:?}"

ordinal="${POD_NAME##*-}"
export WORLD_SIZE="${PR5_DOMAINS}"
export RANK="${ordinal}"
export MASTER_ADDR="${PR5_MASTER_ADDR}"
export MASTER_PORT=29500
export LOCAL_WORLD_SIZE=8
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_SOCKET_IFNAME='^lo,docker,veth'
export NCCL_NET_PLUGIN=ofi
export NCCL_GIN_TYPE=5
export NCCL_SYM_GIN_KERNELS_ENABLE=0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export EP_BUFFER_DEBUG=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

printf 'ADAI_PR5_CONFIG domains=%s_domains ranks=%s_ranks expectation=%s node_rank=%s_dimensionless\n' \
    "${PR5_DOMAINS}" "$((PR5_DOMAINS * 8))" "${PR5_EXPECT}" "${ordinal}"
set +e
python3 /opt/benchmark/pr5-gate.py --domains "${PR5_DOMAINS}" 2>&1 | tee /tmp/pr5-gate.log
status="${PIPESTATUS[0]}"
set -e

case "${PR5_EXPECT}" in
    refuse)
        if ((status != 0)) && \
           grep -Fq 'GIN indexed-signal budget cannot give each peer rail team a dedicated signal' /tmp/pr5-gate.log; then
            printf 'ADAI_PR5_GATE_PASS result=EXPECTED_REFUSAL domains=%s_domains\n' "${PR5_DOMAINS}"
        else
            printf 'ADAI_PR5_GATE_FAIL expected=refusal exit_status=%s_dimensionless\n' "${status}" >&2
            exit 1
        fi
        ;;
    pass)
        if ((status == 0)) && grep -Fq 'ADAI_PR5_CORRECTNESS_PASS' /tmp/pr5-gate.log; then
            printf 'ADAI_PR5_GATE_PASS result=CORRECTNESS_PASS domains=%s_domains\n' "${PR5_DOMAINS}"
        else
            printf 'ADAI_PR5_GATE_FAIL expected=pass exit_status=%s_dimensionless\n' "${status}" >&2
            exit 1
        fi
        ;;
    *) printf 'Unsupported PR5_EXPECT=%s\n' "${PR5_EXPECT}" >&2; exit 2 ;;
esac
sleep infinity
