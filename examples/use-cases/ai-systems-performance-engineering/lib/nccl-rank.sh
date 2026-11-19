#!/usr/bin/env bash
set -euo pipefail
# PCS PMIx 5 advertises shmem2, absent from this image's PMIx 3 client. Set this
# inside the MPI rank so non-MPI Enroot steps do not inherit a PMIX_ variable.
export PMIX_MCA_gds=hash
# Apply after container environment loading. MPI is bootstrap only; the
# configured NCCL transport remains the independent variable.
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=tcp,self
export OMPI_MCA_btl_tcp_if_include="${NCCL_SOCKET_IFNAME#=}"
unset OMPI_MCA_btl_tcp_if_exclude
exec env LD_PRELOAD=/opt/nccl/build/lib/libnccl.so \
    /opt/nccl-tests/build/all_reduce_perf -b 8M -e 256M -f 2 -g 1 -w 5 -n 20 -c 1
