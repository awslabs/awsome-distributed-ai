#!/usr/bin/env bash
# One MPI rank of tools/nccl-protocols.sh: all_reduce_perf on one GPU over EFA.
# Usage: bash lib/nccl-protocols-rank.sh nccl-choice|simple LOG_DIR
set -euo pipefail
variant=${1:?} logs=${2:?}
# MPI bootstrap only, as in lib/nccl-rank.sh, set after the image's environment.
export PMIX_MCA_gds=hash
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=tcp,self
export OMPI_MCA_btl_tcp_if_include="${NCCL_SOCKET_IFNAME#=}"
unset OMPI_MCA_btl_tcp_if_exclude
# The EFA settings of 2.run-llm.sh; only NCCL_PROTO differs between the two runs.
export FI_PROVIDER=efa NCCL_NET='AWS Libfabric' OFI_NCCL_PROTOCOL=RDMA
unset NCCL_NET_PLUGIN NCCL_GIN_PLUGIN NCCL_ALGO
case $variant in
    nccl-choice) unset NCCL_PROTO ;;
    simple) export NCCL_PROTO=Simple ;;
    *) echo "unknown run $variant; use nccl-choice or simple" >&2; exit 2 ;;
esac
# With TUNING, rank 0 logs the algorithm and protocol of every collective. The
# log goes to one file per process, so the table on stdout stays readable.
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,ENV,TUNING NCCL_DEBUG_FILE="$logs/nccl-%h-%p.log"
exec env LD_PRELOAD=/opt/nccl/build/lib/libnccl.so \
    /opt/nccl-tests/build/all_reduce_perf -b 1M -e 64M -f 2 -g 1 -w 5 -n 20 -c 1
