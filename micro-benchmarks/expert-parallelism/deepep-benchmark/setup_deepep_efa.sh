#!/bin/bash
# Build DeepEP with NVSHMEM EFA (libfabric) transport for AWS EFA clusters.
#
# This script patches an existing DeepEP source tree for EFA, builds NVSHMEM
# from an existing source tree, and builds DeepEP against it.
#
# Assumes it runs on the build machine (with CUDA, libfabric, Python+PyTorch).
#
# Usage:
#   ./setup_deepep_efa.sh --deepep-src <path> --nvshmem-src <path> [options]
#
# Required:
#   --deepep-src <path>       Path to existing DeepEP source tree
#   --nvshmem-src <path>      Path to existing NVSHMEM source tree
#   --venv <path>             Python venv to activate before build
#
# Build environment:
#   --cuda-home <path>        CUDA toolkit (default: /usr/local/cuda)
#   --libfabric-home <path>   Libfabric install (default: /opt/amazon/efa)
#   --gdrcopy-home <path>     GDRCopy install (default: /usr)
#   --gpu-arch <arch>         CUDA architecture (default: 90)

set -euo pipefail

# --- Defaults ---
DEEPEP_SRC=""
NVSHMEM_SRC=""
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
LIBFABRIC_HOME="${LIBFABRIC_HOME:-/opt/amazon/efa}"
GDRCOPY_HOME="${GDRCOPY_HOME:-/usr}"
GPU_ARCH="${GPU_ARCH:-90}"
VENV=""

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --deepep-src)       DEEPEP_SRC="$2"; shift 2 ;;
        --nvshmem-src)      NVSHMEM_SRC="$2"; shift 2 ;;
        --cuda-home)        CUDA_HOME="$2"; shift 2 ;;
        --libfabric-home)   LIBFABRIC_HOME="$2"; shift 2 ;;
        --gdrcopy-home)     GDRCOPY_HOME="$2"; shift 2 ;;
        --gpu-arch)         GPU_ARCH="$2"; shift 2 ;;
        --venv)             VENV="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Derive TORCH_CUDA_ARCH_LIST from GPU_ARCH (e.g. 90 -> 9.0, 100 -> 10.0)
TORCH_CUDA_ARCH_LIST="${GPU_ARCH%?}.${GPU_ARCH: -1}"

# --- Validate required args ---
if [[ -z "$DEEPEP_SRC" ]]; then
    echo "ERROR: --deepep-src is required"
    exit 1
fi
if [[ -z "$NVSHMEM_SRC" ]]; then
    echo "ERROR: --nvshmem-src is required"
    exit 1
fi
if [[ -z "$VENV" ]]; then
    echo "ERROR: --venv is required (PEP 668 — use a virtual environment)"
    exit 1
fi
if [[ ! -f "$DEEPEP_SRC/csrc/kernels/internode.cu" ]]; then
    echo "ERROR: $DEEPEP_SRC does not look like a DeepEP source tree"
    exit 1
fi
if [[ ! -f "$NVSHMEM_SRC/CMakeLists.txt" ]]; then
    echo "ERROR: $NVSHMEM_SRC does not look like an NVSHMEM source tree"
    exit 1
fi

DEEPEP_SRC="$(cd "$DEEPEP_SRC" && pwd)"
NVSHMEM_SRC="$(cd "$NVSHMEM_SRC" && pwd)"
NVSHMEM_INSTALL_DIR="$NVSHMEM_SRC/install"

# --- Activate venv ---
echo "Activating venv: $VENV"
# venv activate scripts may reference unset vars (e.g. LD_LIBRARY_PATH)
set +u
source "$VENV/bin/activate"
set -u

# --- Validate environment ---
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"

if ! "$CUDA_HOME/bin/nvcc" --version &>/dev/null; then
    echo "ERROR: nvcc not found at $CUDA_HOME/bin/nvcc. Set --cuda-home."
    exit 1
fi

if [[ ! -d "$LIBFABRIC_HOME/lib" ]]; then
    echo "ERROR: libfabric not found at $LIBFABRIC_HOME. Set --libfabric-home."
    exit 1
fi

CUDA_VERSION=$("$CUDA_HOME/bin/nvcc" --version | grep -oP 'release \K[0-9]+\.[0-9]+')

echo "=== DeepEP EFA Setup ==="
echo "  DeepEP source:  $DEEPEP_SRC"
echo "  NVSHMEM source: $NVSHMEM_SRC"
echo "  CUDA:           $CUDA_HOME (CUDA $CUDA_VERSION)"
echo "  Libfabric:      $LIBFABRIC_HOME"
echo "  GDRCopy:        $GDRCOPY_HOME"
echo "  GPU arch:       $GPU_ARCH"
echo ""

########################################################################
# Step 1: Patch DeepEP
########################################################################
# Tested with DeepEP commit 567632d (pre-EPv2, main branch).
# EPv2 (merged Apr 29 2026) restructures the kernel sources and switches to
# the NCCL GIN backend.  This script targets the legacy NVSHMEM code path.
DEEPEP_COMPAT_COMMIT="567632d"

echo ">>> Step 1: Patch DeepEP for EFA"

cd "$DEEPEP_SRC"

# Verify commit compatibility
CURRENT_COMMIT=$(git rev-parse --short=7 HEAD 2>/dev/null || echo "unknown")
if [[ "$CURRENT_COMMIT" != "$DEEPEP_COMPAT_COMMIT" ]]; then
    echo "ERROR: DeepEP HEAD is $CURRENT_COMMIT, expected $DEEPEP_COMPAT_COMMIT."
    echo "       This script is tested with commit $DEEPEP_COMPAT_COMMIT (pre-EPv2)."
    exit 1
fi

# 0. Apply combined put+signal patch to internode.cu
#    Use a per-invocation temp file (not a fixed /tmp path) so concurrent users on
#    a shared node don't collide on a sticky-bit-owned /tmp/deepep-put-signal.patch.
PUT_SIGNAL_PATCH="$(mktemp -t deepep-put-signal.XXXXXX.patch)"
cat > "$PUT_SIGNAL_PATCH" << 'PATCHEOF'
diff --git a/csrc/kernels/internode.cu b/csrc/kernels/internode.cu
index 48c6c00..1cce923 100644
--- a/csrc/kernels/internode.cu
+++ b/csrc/kernels/internode.cu
@@ -815,13 +815,14 @@ __global__ void __launch_bounds__(((kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NV
                         reinterpret_cast<uint64_t>(rdma_channel_data.recv_buffer(rdma_rank) + dst_slot_idx * num_bytes_per_token);
                     const auto src_ptr =
                         reinterpret_cast<uint64_t>(rdma_channel_data.send_buffer(dst_rdma_rank) + dst_slot_idx * num_bytes_per_token);
-                    nvshmemi_ibgda_put_nbi_warp<true>(dst_ptr,
+                    nvshmemi_ibgda_put_signal_nbi_warp<true>(dst_ptr,
                                                       src_ptr,
                                                       num_bytes_per_msg,
+                                                      rdma_channel_tail.buffer(rdma_rank),
+                                                      num_tokens_to_issue,
                                                       translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
                                                       channel_id,
-                                                      lane_id,
-                                                      0);
+                                                      lane_id);
                 } else {
                     // Lighter fence for local RDMA rank
                     memory_fence();
@@ -832,11 +833,9 @@ __global__ void __launch_bounds__(((kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NV
                 if (lane_id == dst_rdma_rank) {
                     last_issued_tail += num_tokens_to_issue;
                     num_tokens_to_send -= num_tokens_to_issue;
-                    nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_tail.buffer(rdma_rank),
-                                                    num_tokens_to_issue,
-                                                    translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
-                                                    channel_id,
-                                                    dst_rdma_rank == rdma_rank);
+                    if (dst_rdma_rank == rdma_rank) {
+                        atomicAdd(reinterpret_cast<unsigned long long*>(rdma_channel_tail.buffer(rdma_rank)), static_cast<unsigned long long>(num_tokens_to_issue));
+                    }
                 }
                 __syncwarp();
             }
@@ -2115,13 +2114,14 @@ __global__ void __launch_bounds__((kNumForwarders + 1) * 32, 1) combine(int4* co
                             reinterpret_cast<uint64_t>(rdma_channel_data.recv_buffer(rdma_rank) + rdma_slot_idx * num_bytes_per_token);
                         const auto src_ptr =
                             reinterpret_cast<uint64_t>(rdma_channel_data.send_buffer(dst_rdma_rank) + rdma_slot_idx * num_bytes_per_token);
-                        nvshmemi_ibgda_put_nbi_warp<true>(dst_ptr,
+                        nvshmemi_ibgda_put_signal_nbi_warp<true>(dst_ptr,
                                                           src_ptr,
                                                           num_bytes_per_msg,
+                                                          rdma_channel_tail.buffer(rdma_rank),
+                                                          num_chunked_tokens,
                                                           translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
                                                           channel_id,
-                                                          lane_id,
-                                                          0);
+                                                          lane_id);
                     } else {
                         memory_fence();
                     }
@@ -2129,11 +2129,10 @@ __global__ void __launch_bounds__((kNumForwarders + 1) * 32, 1) combine(int4* co
                     // Write new RDMA tail
                     __syncwarp();
                     if (elect_one_sync()) {
-                        nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_tail.buffer(rdma_rank),
-                                                        num_chunked_tokens,
-                                                        translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
-                                                        channel_id,
-                                                        dst_rdma_rank == rdma_rank);
+                        if (dst_rdma_rank == rdma_rank) {
+                            atomicAdd(reinterpret_cast<unsigned long long int*>(rdma_channel_tail.buffer(rdma_rank)),
+                                      static_cast<unsigned long long int>(num_chunked_tokens));
+                        }
                     }
                 }
             }
PATCHEOF
git apply "$PUT_SIGNAL_PATCH"
rm -f "$PUT_SIGNAL_PATCH"
echo "  Applied put+signal patch to internode.cu"

# 1. Replace ibgda_device.cuh with EFA shim (same filename, no include changes needed)
cat > csrc/kernels/ibgda_device.cuh << 'PATCH_EOF'
// Portions derived from NVSHMEM (https://developer.nvidia.com/nvshmem)
// Copyright (c) NVIDIA Corporation.
// Licensed under the NVSHMEM Software License Agreement.
//
// EFA-compatible replacement for ibgda_device.cuh.
// Replaces IBGDA GPU-initiated RDMA with NVSHMEM host-proxy QP APIs.
#pragma once

#include <type_traits>

#include "configs.cuh"
#include "exception.cuh"
#include "nvshmemx.h"
#include "utils.cuh"

namespace deep_ep {

/*
 * Fake IBGDA device state for EFA.
 *
 * ibgda_get_state()->num_rc_per_pe and num_devices_initialized are used by
 * internode.cu / internode_ll.cu to iterate over QPs per PE for quiet calls
 * and to satisfy IBGDA-specific asserts:
 *   internode.cu:    num_rc_per_pe == num_channels OR num_rc_per_pe >= num_sms
 *   internode_ll.cu: num_rc_per_pe >= num_local_experts  (up to 288)
 *
 * Libfabric has 1 proxy endpoint per PE, so repeated quiet calls are
 * idempotent and cheap (early-exit in nvshmemt_libfabric_quiet when
 * submitted_ops == completed_ops + completed_staged_atomics).  Setting
 * num_rc_per_pe = 288 satisfies both asserts without needing to patch
 * them out.
 */
extern __device__ nvshmemi_ibgda_device_state_t g_fake_ibgda_device_state;
extern __device__ int g_fake_ibgda_state_initialized;

__device__ static __forceinline__ void init_fake_ibgda_device_state_if_needed() {
    if (__ldg(&g_fake_ibgda_state_initialized))
        return;

    if (atomicCAS(&g_fake_ibgda_state_initialized, 0, 1) == 0) {
        g_fake_ibgda_device_state.version = (1 << 16) + sizeof(nvshmemi_ibgda_device_state_t);
        g_fake_ibgda_device_state.num_shared_dcis = 0;
        g_fake_ibgda_device_state.num_exclusive_dcis = 0;
        g_fake_ibgda_device_state.dci_map_type = NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_INVALID;
        g_fake_ibgda_device_state.ndcts_per_pe = 0;
        g_fake_ibgda_device_state.num_qp_groups = 0;
        g_fake_ibgda_device_state.num_dct_groups = 0;
        g_fake_ibgda_device_state.num_rc_per_pe = 288;
        g_fake_ibgda_device_state.num_devices_initialized = 1;
        g_fake_ibgda_device_state.rc_map_type = NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_INVALID;
        g_fake_ibgda_device_state.num_requests_in_batch = 0;
        g_fake_ibgda_device_state.log2_cumem_granularity = 0;
        g_fake_ibgda_device_state.nic_buf_on_gpumem = false;
        g_fake_ibgda_device_state.support_half_av_seg = false;
        g_fake_ibgda_device_state.may_skip_cst = false;
        g_fake_ibgda_device_state.use_async_postsend = false;
        g_fake_ibgda_device_state.globalmem.qp_group_switches = nullptr;
        g_fake_ibgda_device_state.globalmem.cqs = nullptr;
        g_fake_ibgda_device_state.globalmem.dcis = nullptr;
        g_fake_ibgda_device_state.globalmem.rcs = nullptr;
        g_fake_ibgda_device_state.globalmem.local_only_mhandle_head = nullptr;
        g_fake_ibgda_device_state.globalmem.dcts = nullptr;
        g_fake_ibgda_device_state.globalmem.lkeys = nullptr;
        g_fake_ibgda_device_state.globalmem.rkeys = nullptr;
        g_fake_ibgda_device_state.extra = nullptr;
        __threadfence();
    }

    while (!__ldg(&g_fake_ibgda_state_initialized)) {
    }
    __threadfence();
}

__device__ static __forceinline__ nvshmemi_ibgda_device_state_t* ibgda_get_state() {
    init_fake_ibgda_device_state_if_needed();
    return &g_fake_ibgda_device_state;
}

__device__ static __forceinline__ void nvshmemi_ibgda_quiet(int dst_pe, int _qp_id) {
    int host_qp = NVSHMEMX_QP_DEFAULT;
    nvshmemx_qp_quiet(dst_pe, &host_qp, 1);
}

template <bool kAlwaysDoPostSend = false>
__device__ static __forceinline__ void nvshmemi_ibgda_put_nbi_warp(uint64_t req_rptr,
                                                                   uint64_t req_lptr,
                                                                   size_t bytes,
                                                                   int dst_pe,
                                                                   int _qp_id,
                                                                   int lane_id,
                                                                   int message_idx,
                                                                   nvshmemx_qp_handle_t _qp_handle = NVSHMEMX_QP_DEFAULT) {
    nvshmemx_qp_char_put_nbi_warp(reinterpret_cast<char*>(req_rptr), reinterpret_cast<char*>(req_lptr), bytes, dst_pe, NVSHMEMX_QP_DEFAULT);
}

__device__ __forceinline__ void nvshmemi_ibgda_amo_nonfetch_add(
    void* rptr, const int& value, int pe, int _qp_id, bool is_local_copy = false) {
    if (is_local_copy) {
        atomicAdd(static_cast<unsigned long long*>(rptr), value);
    } else {
        nvshmem_int_atomic_add(static_cast<int*>(rptr), value, pe);
    }
}

__device__ __forceinline__ uint64_t nvshmemi_get_p2p_ptr(const uint64_t& ptr, const int& rank, const int& dst_rank) {
    if (rank == dst_rank)
        return ptr;
    auto peer_base = __ldg(reinterpret_cast<uint64_t*>(nvshmemi_device_state_d.peer_heap_base_p2p) + dst_rank);
    if (peer_base == 0)
        return 0;
    return peer_base + (ptr - reinterpret_cast<uint64_t>(nvshmemi_device_state_d.heap_base));
}

// Combined put + signal: posts data and signals the remote tail in one call.
template <bool kAlwaysDoPostSend = false>
__device__ static __forceinline__ void nvshmemi_ibgda_put_signal_nbi_warp(
    uint64_t req_rptr, uint64_t req_lptr, size_t bytes,
    void* signal_rptr, const int signal_value,
    int dst_pe, int qp_id, int lane_id,
    nvshmemx_qp_handle_t qp_handle = NVSHMEMX_QP_DEFAULT) {
    nvshmemx_qp_char_put_signal_nbi_warp(reinterpret_cast<char*>(req_rptr),
        reinterpret_cast<const char*>(req_lptr), bytes,
        reinterpret_cast<uint64_t*>(signal_rptr), signal_value, NVSHMEM_SIGNAL_ADD,
        dst_pe, qp_handle);
}

__device__ static __forceinline__ void nvshmemi_ibgda_rma_p(
    int* rptr, const int value, int dst_pe, int _qp_id, uint32_t imm = std::numeric_limits<uint32_t>::max()) {
    nvshmemx_qp_int_p(rptr, value, dst_pe, NVSHMEMX_QP_DEFAULT);
}

}  // namespace deep_ep
PATCH_EOF

# 2. Inject fake IBGDA state into runtime.cu
#    (No assert comment-outs needed — num_rc_per_pe=288 satisfies both asserts.)
if ! grep -q 'g_fake_ibgda_device_state' csrc/kernels/runtime.cu; then
    sed -i '/^namespace deep_ep {/a \
\
#ifndef DISABLE_NVSHMEM\
__device__ nvshmemi_ibgda_device_state_t g_fake_ibgda_device_state;\
__device__ int g_fake_ibgda_state_initialized = 0;\
#endif' csrc/kernels/runtime.cu
fi

# 3. Raise DeepEP's baked-in NVSHMEM_MAX_TEAMS from 7 to 8.
#    buffer.py hard-codes NVSHMEM_MAX_TEAMS=7 ("6 default teams + 1 extra"), sized
#    for an older NVSHMEM that stood up 6 internal teams at init. NVSHMEM 3.7 adds a
#    7th (NVSHMEM_TEAM_MC_SHARED for NVLS multicast), so all 7 slots are taken before
#    DeepEP's low-latency path splits its RDMA sub-team and nvshmem_team_split_strided
#    fails with "No more teams available (max=7)". Bump it to 8 (7 internal + 1 for the
#    split) so a freshly built tree runs out of the box with no runtime env var.
if [[ -f deep_ep/buffer.py ]]; then
    sed -i "s/os.environ\['NVSHMEM_MAX_TEAMS'\] = '7'/os.environ['NVSHMEM_MAX_TEAMS'] = '8'/" deep_ep/buffer.py
    echo "  Set NVSHMEM_MAX_TEAMS to 8 in buffer.py (NVSHMEM 3.7 needs >7)"
fi

echo "  DeepEP patched at $DEEPEP_SRC"
echo ""

########################################################################
# Step 2: Build NVSHMEM
########################################################################
echo ">>> Step 2: Build NVSHMEM"

cd "$NVSHMEM_SRC"
rm -rf build && mkdir build && cd build

cmake .. \
    -DCMAKE_INSTALL_PREFIX="$NVSHMEM_INSTALL_DIR" \
    -DCUDA_HOME="$CUDA_HOME" \
    -DLIBFABRIC_HOME="$LIBFABRIC_HOME" \
    -DNVSHMEM_LIBFABRIC_SUPPORT=ON \
    -DNVSHMEM_MPI_SUPPORT=OFF \
    -DNVSHMEM_IBRC_SUPPORT=OFF \
    -DNVSHMEM_IBGDA_SUPPORT=OFF \
    -DNVSHMEM_IBDEVX_SUPPORT=OFF \
    -DNVSHMEM_UCX_SUPPORT=OFF \
    -DNVSHMEM_SHMEM_SUPPORT=OFF \
    -DNVSHMEM_PMIX_SUPPORT=OFF \
    -DNVSHMEM_USE_NCCL=OFF \
    -DNVSHMEM_USE_GDRCOPY=ON \
    -DGDRCOPY_HOME="$GDRCOPY_HOME" \
    -DNVSHMEM_USE_MLX5DV=OFF \
    -DNVSHMEM_BUILD_TESTS=OFF \
    -DNVSHMEM_BUILD_EXAMPLES=OFF \
    -DNVSHMEM_BUILD_PYTHON_LIB=OFF \
    -DNVSHMEM_BUILD_BITCODE_LIBRARY=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="$GPU_ARCH"

make -j"$(nproc)"
make install

echo "  NVSHMEM installed to $NVSHMEM_INSTALL_DIR"
echo ""

########################################################################
# Step 3: Build DeepEP
########################################################################
echo ">>> Step 3: Build DeepEP"

export NVSHMEM_DIR="$NVSHMEM_INSTALL_DIR"
export NVSHMEM_HOME="$NVSHMEM_INSTALL_DIR"
export LIBFABRIC_HOME
export LD_LIBRARY_PATH="${LIBFABRIC_HOME}/lib:${NVSHMEM_INSTALL_DIR}/lib:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

TORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
echo "  NVSHMEM:      $NVSHMEM_INSTALL_DIR"
echo "  PyTorch CUDA: $TORCH_CUDA"

if [[ "$TORCH_CUDA" != "unknown" && "$CUDA_VERSION" != "$TORCH_CUDA" ]]; then
    echo "  WARNING: CUDA mismatch — nvcc=$CUDA_VERSION, PyTorch=$TORCH_CUDA"
fi

cd "$DEEPEP_SRC"

pip uninstall -y deep_ep 2>/dev/null || true
python setup.py clean --all 2>/dev/null || true

# DeepEP's setup.py requires DISABLE_AGGRESSIVE_PTX_INSTRS=1 on sm_100 (Blackwell);
# Hopper (sm_90) builds with it off. Derive from --gpu-arch instead of hard-coding.
if [[ "$GPU_ARCH" == "90" ]]; then
    DISABLE_AGGRESSIVE_PTX_INSTRS=0
else
    DISABLE_AGGRESSIVE_PTX_INSTRS=1
fi

DISABLE_AGGRESSIVE_PTX_INSTRS=$DISABLE_AGGRESSIVE_PTX_INSTRS \
TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
    python setup.py install

echo ""
echo "  DeepEP installed. Verify: python -c 'import deep_ep; print(\"OK\")'"
echo ""

echo "=== Setup complete ==="
echo "  DeepEP source:   $DEEPEP_SRC"
echo "  NVSHMEM install: $NVSHMEM_INSTALL_DIR"
