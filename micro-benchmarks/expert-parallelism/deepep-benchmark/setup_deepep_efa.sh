#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

set -euo pipefail

log() {
    printf '%s\n' "$*" >&2
}

warn() {
    printf 'WARNING: %s\n' "$*" >&2
}

die() {
    printf 'ERROR: %s\n' "$*" >&2 || true
    exit 1
}

PROG_NAME="${0##*/}"
if [[ -z "$PROG_NAME" || "$PROG_NAME" == "bash" || "$PROG_NAME" == "-bash" ]]; then
    PROG_NAME="setup_deepep_efa.sh"
fi

print_help() {
    cat <<EOF
Usage: ${PROG_NAME} [options]

Install NVSHMEM and build DeepEP for AWS EFA (libfabric).

With no arguments, downloads a prebuilt NVSHMEM release, clones DeepEP at the
only supported commit, applies the embedded AWS EFA patch, and installs DeepEP
into the active Python environment.

Options:
  --nvshmem-version <v>     NVSHMEM version to download, MAJOR.MINOR.PATCH form
                            (default: 3.7.0)
  --nvshmem-prefix <path>   NVSHMEM install location
                            (default: /opt/amazon/nvshmem)
  --nvshmem-src <path>      Build NVSHMEM from this existing source tree instead
                            of downloading the prebuilt release
                            (default: unset; download prebuilt)
  --deepep-commit <sha>     DeepEP commit to checkout, 7-40 hex chars
                            (default: 567632d)
  --deepep-prefix <path>    DeepEP clone/build location
                            (default: /opt/amazon/deepep)
  --deepep-src <path>       Build DeepEP from this existing source tree instead
                            of cloning from GitHub
                            (default: unset; clone from GitHub)
  --wheel-only              Build a DeepEP wheel but do not install it
                            (default: off; install into the active environment)
  --wheel-output-dir <path> Directory to write the DeepEP wheel into
                            (default: /opt/amazon/wheels)
  --libfabric-home <path>   Libfabric install dir (NVSHMEM source build only)
                            (default: /opt/amazon/efa)
  --gdrcopy-home <path>     GDRCopy install dir (NVSHMEM source build only)
                            (default: /usr)
  --cuda-home <path>        CUDA toolkit directory
                            (default: /usr/local/cuda)
  --gen-ldconfig            Write /etc/ld.so.conf.d/nvshmem.conf and refresh the
                            dynamic linker cache (default: off)
  --python <cmd|path>       Python interpreter command or absolute path
                            (default: python3)
  --pip <cmd|path>          pip command or absolute path
                            (default: pip3)
  --force                   Skip the DeepEP commit-compatibility check and build
                            at your own risk (default: off)
  --skip-checks             Skip Python prerequisite validation (torch, pip,
                            ninja checks) (default: off)
  -h, --help                Print this usage information and exit

Environment:
  TORCH_CUDA_ARCH_LIST      GPU architectures, dotted and ;-separated
                            (default: 9.0;10.0)
EOF
    exit 0
}

parse_args() {
    NVSHMEM_VERSION="3.7.0"
    NVSHMEM_PREFIX="/opt/amazon/nvshmem"
    NVSHMEM_SRC=""
    DEEPEP_COMMIT="567632d"
    DEEPEP_PREFIX="/opt/amazon/deepep"
    DEEPEP_SRC=""
    WHEEL_ONLY="false"
    WHEEL_OUTPUT_DIR="/opt/amazon/wheels"
    LIBFABRIC_HOME="/opt/amazon/efa"
    GDRCOPY_HOME="/usr"
    CUDA_HOME="/usr/local/cuda"
    GEN_LDCONFIG="false"
    FORCE="false"
    SKIP_CHECKS="false"
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST-9.0;10.0}"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --nvshmem-version)
                [[ $# -ge 2 ]] || die "option '$1' requires a value"
                NVSHMEM_VERSION="$2"; shift 2 ;;
            --nvshmem-prefix)
                [[ $# -ge 2 ]] || die "option '$1' requires a value"
                NVSHMEM_PREFIX="$2"; shift 2 ;;
            --nvshmem-src)
                [[ $# -ge 2 ]] || die "option '$1' requires a value"
                NVSHMEM_SRC="$2"; shift 2 ;;
            --deepep-commit)
                [[ $# -ge 2 ]] || die "option '$1' requires a value"
                DEEPEP_COMMIT="$2"; shift 2 ;;
            --deepep-prefix)
                [[ $# -ge 2 ]] || die "option '$1' requires a value"
                DEEPEP_PREFIX="$2"; shift 2 ;;
            --deepep-src)
                [[ $# -ge 2 ]] || die "option '$1' requires a value"
                DEEPEP_SRC="$2"; shift 2 ;;
            --wheel-only)
                WHEEL_ONLY="true"; shift ;;
            --wheel-output-dir)
                [[ $# -ge 2 ]] || die "option '$1' requires a value"
                WHEEL_OUTPUT_DIR="$2"; shift 2 ;;
            --libfabric-home)
                [[ $# -ge 2 ]] || die "option '$1' requires a value"
                LIBFABRIC_HOME="$2"; shift 2 ;;
            --gdrcopy-home)
                [[ $# -ge 2 ]] || die "option '$1' requires a value"
                GDRCOPY_HOME="$2"; shift 2 ;;
            --cuda-home)
                [[ $# -ge 2 ]] || die "option '$1' requires a value"
                CUDA_HOME="$2"; shift 2 ;;
            --python)
                [[ $# -ge 2 ]] || die "option '$1' requires a value"
                PYTHON_CMD="$2"; shift 2 ;;
            --pip)
                [[ $# -ge 2 ]] || die "option '$1' requires a value"
                PIP_CMD="$2"; shift 2 ;;
            --gen-ldconfig)
                GEN_LDCONFIG="true"; shift ;;
            --force)
                FORCE="true"; shift ;;
            --skip-checks)
                SKIP_CHECKS="true"; shift ;;
            -h|--help)
                print_help ;;
            *)
                die "unrecognized argument: '$1'" ;;
        esac
    done
}

validate_version() {
    local version="$1"
    if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        die "invalid NVSHMEM version: '$version' (expected MAJOR.MINOR.PATCH, e.g. 3.7.0)"
    fi
}

validate_commit() {
    local commit="$1"
    if [[ ! "$commit" =~ ^[0-9a-fA-F]{7,40}$ ]]; then
        die "invalid DeepEP commit: '$commit' (expected 7-40 hexadecimal characters)"
    fi
}

validate_arch_list() {
    local arch_list="$1"
    if [[ ! "$arch_list" =~ ^[0-9]+\.[0-9]+(;[0-9]+\.[0-9]+)*$ ]]; then
        die "invalid TORCH_CUDA_ARCH_LIST: '$arch_list' (expected non-empty, dotted, ;-separated architectures, e.g. 9.0;10.0)"
    fi
}

compute_disable_ptx() {
    local arch_list="${1-${TORCH_CUDA_ARCH_LIST:-}}"
    if [[ "$arch_list" == "9.0" ]]; then
        printf '0\n'
    else
        printf '1\n'
    fi
}

arch_list_to_cmake() {
    local arch_list="${1-${TORCH_CUDA_ARCH_LIST:-}}"
    local out="" entry first=1
    local IFS=';'
    for entry in $arch_list; do
        if (( first )); then
            first=0
        else
            out+=";"
        fi
        out+="${entry/./}"
    done
    printf '%s\n' "$out"
}

detect_arch() {
    local machine
    machine="$(uname -m)"

    case "$machine" in
        aarch64|arm64)
            ARCH_KEY="linux-sbsa"
            ;;
        x86_64)
            ARCH_KEY="linux-x86_64"
            ;;
        *)
            die "unsupported architecture: '$machine' (expected aarch64, arm64, or x86_64)"
            ;;
    esac

    case "$machine" in
        aarch64|arm64)
            if [[ "$ARCH_KEY" != "linux-sbsa" ]]; then
                die "architecture mismatch: host is '$machine' but selected key is '$ARCH_KEY' (expected linux-sbsa)"
            fi
            ;;
        x86_64)
            if [[ "$ARCH_KEY" != "linux-x86_64" ]]; then
                die "architecture mismatch: host is '$machine' but selected key is '$ARCH_KEY' (expected linux-x86_64)"
            fi
            ;;
    esac

    log "Detected arch: $ARCH_KEY"
    printf '%s\n' "$ARCH_KEY"
}

detect_cuda_major() {
    local nvcc_path=""

    if command -v nvcc >/dev/null 2>&1; then
        nvcc_path="$(command -v nvcc)"
    elif [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
        nvcc_path="${CUDA_HOME}/bin/nvcc"
    else
        die "cannot find nvcc: not on PATH and not at \${CUDA_HOME}/bin/nvcc (is the CUDA toolkit installed?)"
    fi

    local version_output major
    version_output="$("$nvcc_path" --version 2>/dev/null)" || \
        die "failed to run '$nvcc_path --version'"

    major="$(printf '%s\n' "$version_output" | \
        sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p')"

    if [[ -z "$major" ]]; then
        die "unable to parse CUDA major version from nvcc output"
    fi

    CUDA_MAJOR="$major"
    log "Detected CUDA major version: $CUDA_MAJOR"
    printf '%s\n' "$CUDA_MAJOR"
}

check_python_prereqs() {
    # Validate that PYTHON_CMD and PIP_CMD are resolvable before checking prereqs.
    local python_cmd="${PYTHON_CMD:-python3}"
    local pip_cmd="${PIP_CMD:-pip3}"

    if [[ "$python_cmd" == /* ]]; then
        [[ -x "$python_cmd" ]] || die "python command not found: '$python_cmd' is not an executable file"
    else
        command -v "$python_cmd" >/dev/null 2>&1 || die "python command not found: '$python_cmd' is not on PATH"
    fi

    if [[ "$pip_cmd" == /* ]]; then
        [[ -x "$pip_cmd" ]] || die "pip command not found: '$pip_cmd' is not an executable file"
    else
        command -v "$pip_cmd" >/dev/null 2>&1 || die "pip command not found: '$pip_cmd' is not on PATH"
    fi

    local missing=()

    if ! command -v "$python_cmd" >/dev/null 2>&1; then
        missing+=("$python_cmd")
    else
        if ! "$python_cmd" -c "import torch" >/dev/null 2>&1; then
            missing+=("torch (Python module)")
        fi
        if ! "$python_cmd" -m pip --version >/dev/null 2>&1; then
            missing+=("pip")
        fi
    fi

    if ! command -v ninja >/dev/null 2>&1; then
        missing+=("ninja")
    fi

    if (( ${#missing[@]} > 0 )); then
        local IFS=', '
        die "missing prerequisites: ${missing[*]} (install them before running this script)"
    fi

    return 0
}

download_with_retry() {
    local url="$1"
    local dest="$2"
    local max_attempts=3
    local attempt

    for (( attempt = 1; attempt <= max_attempts; attempt++ )); do
        if command -v curl >/dev/null 2>&1; then
            if curl -fsSL -o "$dest" "$url"; then
                return 0
            fi
        elif command -v wget >/dev/null 2>&1; then
            if wget -q -O "$dest" "$url"; then
                return 0
            fi
        else
            die "Neither curl nor wget is available; cannot download ${url}"
        fi

        if (( attempt < max_attempts )); then
            log "Download attempt ${attempt}/${max_attempts} failed for ${url}; retrying in 2s..."
            sleep 2
        fi
    done

    die "Failed to download ${url} after ${max_attempts} attempts"
}

verify_sha256() {
    local file="$1"
    local expected="$2"
    local actual

    if command -v sha256sum >/dev/null 2>&1; then
        actual="$(sha256sum "$file" | awk '{print $1}')"
    elif command -v shasum >/dev/null 2>&1; then
        actual="$(shasum -a 256 "$file" | awk '{print $1}')"
    else
        die "Neither sha256sum nor shasum is available; cannot verify integrity of ${file}"
    fi

    if [[ "$actual" != "$expected" ]]; then
        die "SHA256 mismatch for ${file}: expected=${expected}, actual=${actual}"
    fi
}

readonly NVSHMEM_REDIST_ROOT="https://developer.download.nvidia.com/compute/nvshmem/redist/"

resolve_nvshmem_archive() {
    local manifest_url="${NVSHMEM_REDIST_ROOT}redistrib_${NVSHMEM_VERSION}.json"
    local manifest_tmp
    manifest_tmp="$(mktemp)"

    download_with_retry "$manifest_url" "$manifest_tmp"

    local parsed
    if ! parsed="$("$PYTHON_CMD" - "$manifest_tmp" "$ARCH_KEY" "$CUDA_MAJOR" <<'PY'
import json, sys
manifest, arch_key, major = sys.argv[1], sys.argv[2], sys.argv[3]
with open(manifest) as f:
    data = json.load(f)
obj = data["libnvshmem"][arch_key]["cuda" + major]
print(obj["relative_path"], obj["sha256"])
PY
    )"; then
        rm -f "$manifest_tmp"
        die "NVSHMEM manifest does not contain an entry for arch='${ARCH_KEY}', cuda='${CUDA_MAJOR}' (version ${NVSHMEM_VERSION})"
    fi

    rm -f "$manifest_tmp"
    read -r NVSHMEM_REL_PATH NVSHMEM_SHA256 <<< "$parsed"
}

install_prebuilt_nvshmem() {
    log "Installing prebuilt NVSHMEM ${NVSHMEM_VERSION} for ${ARCH_KEY}/cuda${CUDA_MAJOR}..."

    resolve_nvshmem_archive

    local archive_url="${NVSHMEM_REDIST_ROOT}${NVSHMEM_REL_PATH}"
    local archive_tmp
    archive_tmp="$(mktemp)"

    download_with_retry "$archive_url" "$archive_tmp"
    verify_sha256 "$archive_tmp" "$NVSHMEM_SHA256"

    mkdir -p "$NVSHMEM_PREFIX"
    tar -xf "$archive_tmp" --strip-components=1 -C "$NVSHMEM_PREFIX"
    rm -f "$archive_tmp"

    log "Prebuilt NVSHMEM ${NVSHMEM_VERSION} installed to ${NVSHMEM_PREFIX}"
}

build_nvshmem_from_source() {
    if [[ ! -d "$NVSHMEM_SRC" ]]; then
        die "NVSHMEM source path does not exist or is not a directory: '$NVSHMEM_SRC'"
    fi
    if [[ ! -f "$NVSHMEM_SRC/CMakeLists.txt" ]]; then
        die "NVSHMEM source path does not contain CMakeLists.txt: '$NVSHMEM_SRC' does not look like an NVSHMEM source tree"
    fi
    if [[ ! -d "$LIBFABRIC_HOME" ]]; then
        die "libfabric not found at LIBFABRIC_HOME: '$LIBFABRIC_HOME' does not exist or is not a directory"
    fi

    log "Building NVSHMEM from source at ${NVSHMEM_SRC}..."

    local cmake_archs
    cmake_archs="$(arch_list_to_cmake)"

    local build_dir="${NVSHMEM_SRC}/build"
    rm -rf "$build_dir"
    mkdir -p "$build_dir"

    cmake -S "$NVSHMEM_SRC" -B "$build_dir" \
        -DCMAKE_INSTALL_PREFIX="$NVSHMEM_PREFIX" \
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
        -DCMAKE_CUDA_ARCHITECTURES="$cmake_archs" \
        || die "NVSHMEM CMake configure failed"

    make -C "$build_dir" -j"$(nproc)" \
        || die "NVSHMEM build (make) failed"

    make -C "$build_dir" install \
        || die "NVSHMEM install (make install) failed"

    log "NVSHMEM built from source and installed to ${NVSHMEM_PREFIX}"
}

acquire_deepep() {
    if [[ -z "${DEEPEP_SRC:-}" ]]; then
        log "Cloning DeepEP into ${DEEPEP_PREFIX} at commit ${DEEPEP_COMMIT}..."

        rm -rf "$DEEPEP_PREFIX"
        mkdir -p "$(dirname "$DEEPEP_PREFIX")"

        git clone https://github.com/deepseek-ai/DeepEP "$DEEPEP_PREFIX" \
            || die "failed to clone DeepEP from https://github.com/deepseek-ai/DeepEP"

        git -C "$DEEPEP_PREFIX" checkout "$DEEPEP_COMMIT" \
            || die "failed to checkout DeepEP commit '${DEEPEP_COMMIT}'"

        log "DeepEP cloned and checked out at commit ${DEEPEP_COMMIT}"
    else
        log "Using existing DeepEP source tree at ${DEEPEP_SRC}..."

        if [[ ! -d "$DEEPEP_SRC" ]]; then
            die "DeepEP source path does not exist or is not a directory: '${DEEPEP_SRC}'"
        fi

        if [[ ! -f "$DEEPEP_SRC/csrc/kernels/internode.cu" ]]; then
            die "DeepEP source path does not contain csrc/kernels/internode.cu: '${DEEPEP_SRC}' does not look like a valid DeepEP source tree"
        fi

        DEEPEP_PREFIX="$DEEPEP_SRC"
        log "DeepEP source tree validated at ${DEEPEP_SRC}"
    fi
}

readonly DEEPEP_SUPPORTED_COMMIT="567632d"

commit_compat_gate() {
    if [[ "$DEEPEP_COMMIT" == "$DEEPEP_SUPPORTED_COMMIT" ]]; then
        log "DeepEP commit ${DEEPEP_COMMIT} matches the supported commit"
        return 0
    fi

    if [[ "$FORCE" != "true" ]]; then
        die "DeepEP commit '${DEEPEP_COMMIT}' is not supported; the only supported commit is ${DEEPEP_SUPPORTED_COMMIT}. Use --force to override at your own risk."
    fi

    warn "DeepEP commit '${DEEPEP_COMMIT}' differs from the supported commit ${DEEPEP_SUPPORTED_COMMIT}; proceeding at your own risk (--force)"
    return 0
}

apply_efa_patch() {
    cd "$DEEPEP_PREFIX"

    log "Applying EFA patch to DeepEP at ${DEEPEP_PREFIX}..."

    local efa_patch
    efa_patch="$(mktemp -t deepep-efa.XXXXXX.patch)"

    cat > "$efa_patch" << 'PATCHEOF'
diff --git a/csrc/kernels/ibgda_device.cuh b/csrc/kernels/ibgda_device.cuh
index 421ec2a..b3473db 100644
--- a/csrc/kernels/ibgda_device.cuh
+++ b/csrc/kernels/ibgda_device.cuh
@@ -1,506 +1,132 @@
 // Portions derived from NVSHMEM (https://developer.nvidia.com/nvshmem)
 // Copyright (c) NVIDIA Corporation.
-// Licensed under the NVSHMEM Software License Agreement (version: September 3, 2019).
-// See full license at: https://docs.nvidia.com/nvshmem/api/sla.html
+// Licensed under the NVSHMEM Software License Agreement.
 //
-// Modified from original source:
-//  - nvshmem/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh
+// EFA-compatible replacement for ibgda_device.cuh.
+// Replaces IBGDA GPU-initiated RDMA with NVSHMEM host-proxy QP APIs.
 #pragma once

 #include <type_traits>

 #include "configs.cuh"
 #include "exception.cuh"
+#include "nvshmemx.h"
 #include "utils.cuh"

 namespace deep_ep {

-EP_STATIC_ASSERT(NVSHMEMI_IBGDA_MIN_QP_DEPTH >= 64, "Invalid QP minimum depth");
-
-__device__ static __forceinline__ uint64_t HtoBE64(uint64_t x) {
-    uint64_t ret;
-    asm("{\n\t"
-        ".reg .b32 ign;\n\t"
-        ".reg .b32 lo;\n\t"
-        ".reg .b32 hi;\n\t"
-        ".reg .b32 new_lo;\n\t"
-        ".reg .b32 new_hi;\n\t"
-        "mov.b64 {lo,hi}, %1;\n\t"
-        "prmt.b32 new_hi, lo, ign, 0x0123;\n\t"
-        "prmt.b32 new_lo, hi, ign, 0x0123;\n\t"
-        "mov.b64 %0, {new_lo,new_hi};\n\t"
-        "}"
-        : "=l"(ret)
-        : "l"(x));
-    return ret;
-}
-
-__device__ static __forceinline__ uint32_t HtoBE32(uint32_t x) {
-    uint32_t ret;
-    asm("{\n\t"
-        ".reg .b32 ign;\n\t"
-        "prmt.b32 %0, %1, ign, 0x0123;\n\t"
-        "}"
-        : "=r"(ret)
-        : "r"(x));
-    return ret;
-}
-
-__device__ static __forceinline__ uint16_t HtoBE16(uint16_t x) {
-    // TODO: simplify PTX using 16-bit instructions
-    auto a = static_cast<uint32_t>(x);
-    uint32_t d;
-    asm volatile(
-        "{\n\t"
-        ".reg .b32 mask;\n\t"
-        ".reg .b32 ign;\n\t"
-        "mov.b32 mask, 0x4401;\n\t"
-        "mov.b32 ign, 0x0;\n\t"
-        "prmt.b32 %0, %1, ign, mask;\n\t"
-        "}"
-        : "=r"(d)
-        : "r"(a));
-    return static_cast<uint16_t>(d);
-}
-
-typedef struct mlx5_wqe_ctrl_seg __attribute__((__aligned__(8))) ibgda_ctrl_seg_t;
-
-typedef struct {
-    uint32_t add_data;
-    uint32_t field_boundary;
-    uint64_t reserved;
-} __attribute__((__packed__)) ibgda_atomic_32_masked_fa_seg_t;
-
-__device__ static __forceinline__ nvshmemi_ibgda_device_state_t* ibgda_get_state() {
-    return &nvshmemi_ibgda_device_state_d;
-}
-
-// Template helper to get RC - uses compile-time type checking with if constexpr (C++17)
-template <typename StateType>
-__device__ static __forceinline__ nvshmemi_ibgda_device_qp_t* ibgda_get_rc_impl(StateType* state, int pe, int id) {
-    const auto num_rc_per_pe = state->num_rc_per_pe;
+/*
+ * Fake IBGDA device state for EFA.
+ *
+ * ibgda_get_state()->num_rc_per_pe and num_devices_initialized are used by
+ * internode.cu / internode_ll.cu to iterate over QPs per PE for quiet calls
+ * and to satisfy IBGDA-specific asserts:
+ *   internode.cu:    num_rc_per_pe == num_channels OR num_rc_per_pe >= num_sms
+ *   internode_ll.cu: num_rc_per_pe >= num_local_experts  (up to 288)
+ *
+ * Libfabric has 1 proxy endpoint per PE, so repeated quiet calls are
+ * idempotent and cheap (early-exit in nvshmemt_libfabric_quiet when
+ * submitted_ops == completed_ops + completed_staged_atomics).  Setting
+ * num_rc_per_pe = 288 satisfies both asserts without needing to patch
+ * them out.
+ */
+extern __device__ nvshmemi_ibgda_device_state_t g_fake_ibgda_device_state;
+extern __device__ int g_fake_ibgda_state_initialized;
+
+__device__ static __forceinline__ void init_fake_ibgda_device_state_if_needed() {
+    if (__ldg(&g_fake_ibgda_state_initialized))
+        return;

-    if constexpr (std::is_same_v<StateType, nvshmemi_ibgda_device_state_v1>) {
-        // v1 implementation
-        return &state->globalmem
-                    .rcs[pe * num_rc_per_pe * state->num_devices_initialized + id % (num_rc_per_pe * state->num_devices_initialized)];
-    } else {
-        // v2 implementation (or any other type)
-        return &state->globalmem.rcs[pe + nvshmemi_device_state_d.npes * id];
+    if (atomicCAS(&g_fake_ibgda_state_initialized, 0, 1) == 0) {
+        g_fake_ibgda_device_state.version = (1 << 16) + sizeof(nvshmemi_ibgda_device_state_t);
+        g_fake_ibgda_device_state.num_shared_dcis = 0;
+        g_fake_ibgda_device_state.num_exclusive_dcis = 0;
+        g_fake_ibgda_device_state.dci_map_type = NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_INVALID;
+        g_fake_ibgda_device_state.ndcts_per_pe = 0;
+        g_fake_ibgda_device_state.num_qp_groups = 0;
+        g_fake_ibgda_device_state.num_dct_groups = 0;
+        g_fake_ibgda_device_state.num_rc_per_pe = 288;
+        g_fake_ibgda_device_state.num_devices_initialized = 1;
+        g_fake_ibgda_device_state.rc_map_type = NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_INVALID;
+        g_fake_ibgda_device_state.num_requests_in_batch = 0;
+        g_fake_ibgda_device_state.log2_cumem_granularity = 0;
+        g_fake_ibgda_device_state.nic_buf_on_gpumem = false;
+        g_fake_ibgda_device_state.support_half_av_seg = false;
+        g_fake_ibgda_device_state.may_skip_cst = false;
+        g_fake_ibgda_device_state.use_async_postsend = false;
+        g_fake_ibgda_device_state.globalmem.qp_group_switches = nullptr;
+        g_fake_ibgda_device_state.globalmem.cqs = nullptr;
+        g_fake_ibgda_device_state.globalmem.dcis = nullptr;
+        g_fake_ibgda_device_state.globalmem.rcs = nullptr;
+        g_fake_ibgda_device_state.globalmem.local_only_mhandle_head = nullptr;
+        g_fake_ibgda_device_state.globalmem.dcts = nullptr;
+        g_fake_ibgda_device_state.globalmem.lkeys = nullptr;
+        g_fake_ibgda_device_state.globalmem.rkeys = nullptr;
+        g_fake_ibgda_device_state.extra = nullptr;
+        __threadfence();
     }
-}
-
-__device__ static __forceinline__ nvshmemi_ibgda_device_qp_t* ibgda_get_rc(int pe, int id) {
-    auto state = ibgda_get_state();
-    return ibgda_get_rc_impl(state, pe, id);
-}
-
-__device__ static __forceinline__ void ibgda_lock_acquire(int* lock) {
-    while (atomicCAS(lock, 0, 1) == 1)
-        ;
-
-    // Prevent reordering before the lock is acquired
-    memory_fence_cta();
-}
-
-__device__ static __forceinline__ void ibgda_lock_release(int* lock) {
-    memory_fence_cta();
-
-    // Prevent reordering before lock is released
-    st_na_relaxed(lock, 0);
-}
-
-__device__ static __forceinline__ void ibgda_update_dbr(nvshmemi_ibgda_device_qp_t* qp, uint32_t dbrec_head) {
-    // `DBREC` contains the index of the next empty `WQEBB`
-    __be32 dbrec_val;
-    __be32* dbrec_ptr = qp->tx_wq.dbrec;
-
-    // This is equivalent to `WRITE_ONCE(dbrec_ptr, HtoBE32(dbrec_head & 0xffff))`
-    asm("{\n\t"
-        ".reg .b32 dbrec_head_16b;\n\t"
-        ".reg .b32 ign;\n\t"
-        "and.b32 dbrec_head_16b, %1, 0xffff;\n\t"
-        "prmt.b32 %0, dbrec_head_16b, ign, 0x123;\n\t"
-        "}"
-        : "=r"(dbrec_val)
-        : "r"(dbrec_head));
-    st_na_release(dbrec_ptr, dbrec_val);
-}
-
-__device__ static __forceinline__ void ibgda_ring_db(nvshmemi_ibgda_device_qp_t* qp, uint16_t prod_idx) {
-    auto bf_ptr = reinterpret_cast<uint64_t*>(qp->tx_wq.bf);
-    ibgda_ctrl_seg_t ctrl_seg = {.opmod_idx_opcode = HtoBE32(prod_idx << 8), .qpn_ds = HtoBE32(qp->qpn << 8)};
-
-    EP_STATIC_ASSERT(sizeof(decltype(&ctrl_seg)) == sizeof(uint64_t), "");
-    st_na_release(bf_ptr, *(reinterpret_cast<uint64_t*>(&ctrl_seg)));
-}

-__device__ static __forceinline__ void ibgda_post_send(nvshmemi_ibgda_device_qp_t* qp, uint64_t new_prod_idx) {
-    nvshmemi_ibgda_device_qp_management_t* mvars = &qp->mvars;
-    uint64_t old_prod_idx;
-
-    // Update `prod_idx` before ringing the doorbell, so that we know which index is needed in quiet/fence
-    ibgda_lock_acquire(&mvars->post_send_lock);
-
-    old_prod_idx = atomicMax(reinterpret_cast<unsigned long long int*>(&mvars->tx_wq.prod_idx), new_prod_idx);
-    if (new_prod_idx > old_prod_idx) {
-        ibgda_update_dbr(qp, new_prod_idx);
-        ibgda_ring_db(qp, new_prod_idx);
+    while (!__ldg(&g_fake_ibgda_state_initialized)) {
     }
-    ibgda_lock_release(&mvars->post_send_lock);
-}
-
-template <bool kAlwaysDoPostSend>
-__device__ static __forceinline__ void ibgda_submit_requests(nvshmemi_ibgda_device_qp_t* qp,
-                                                             uint64_t base_wqe_idx,
-                                                             uint32_t num_wqes,
-                                                             int message_idx = 0) {
-    auto state = ibgda_get_state();
-    nvshmemi_ibgda_device_qp_management_t* mvars = &qp->mvars;
-    uint64_t new_wqe_idx = base_wqe_idx + num_wqes;
-
-    // WQE writes must be finished first
     __threadfence();
-
-    unsigned long long int* ready_idx =
-        (unsigned long long int*)(state->use_async_postsend ? qp->tx_wq.prod_idx : &mvars->tx_wq.ready_head);
-
-    // Wait for prior WQE slots to be filled first
-    while (atomicCAS(ready_idx, base_wqe_idx, new_wqe_idx) != base_wqe_idx)
-        ;
-
-    // Always post, not in batch
-    if (!state->use_async_postsend) {
-        constexpr int kNumRequestInBatch = 4;
-        if (kAlwaysDoPostSend or (message_idx + 1) % kNumRequestInBatch == 0)
-            ibgda_post_send(qp, new_wqe_idx);
-    }
 }

-__device__ static __forceinline__ void ibgda_write_rdma_write_inl_wqe(
-    nvshmemi_ibgda_device_qp_t* qp, const uint32_t* val, uint64_t raddr, __be32 rkey, uint16_t wqe_idx, void** out_wqes, uint32_t imm) {
-    ibgda_ctrl_seg_t ctrl_seg;
-    struct mlx5_wqe_raddr_seg raddr_seg;
-    struct mlx5_wqe_inl_data_seg inl_seg;
-
-    auto* ctrl_seg_ptr = reinterpret_cast<ibgda_ctrl_seg_t*>(out_wqes[0]);
-    auto* raddr_seg_ptr = reinterpret_cast<mlx5_wqe_raddr_seg*>(reinterpret_cast<uintptr_t>(ctrl_seg_ptr) + sizeof(*ctrl_seg_ptr));
-    auto* inl_seg_ptr = reinterpret_cast<mlx5_wqe_inl_data_seg*>(reinterpret_cast<uintptr_t>(raddr_seg_ptr) + sizeof(*raddr_seg_ptr));
-    auto* wqe_data_ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(inl_seg_ptr) + sizeof(*inl_seg_ptr));
-
-    raddr_seg.raddr = HtoBE64(raddr);
-    raddr_seg.rkey = rkey;
-    raddr_seg.reserved = 0;
-
-    inl_seg.byte_count = HtoBE32(4 | MLX5_INLINE_SEG);
-
-    // `imm == std::numeric_limits<uint32_t>::max()` means no imm writes
-    ctrl_seg = {0};
-    ctrl_seg.qpn_ds = HtoBE32((qp->qpn << 8) | 3);
-    ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
-    ctrl_seg.opmod_idx_opcode =
-        HtoBE32((wqe_idx << 8) | (imm != std::numeric_limits<uint32_t>::max() ? MLX5_OPCODE_RDMA_WRITE_IMM : MLX5_OPCODE_RDMA_WRITE));
-    if (imm != std::numeric_limits<uint32_t>::max())
-        ctrl_seg.imm = HtoBE32(imm);
-
-    EP_STATIC_ASSERT(sizeof(*ctrl_seg_ptr) == 16, "sizeof(*ctrl_seg_ptr) == 16");
-    EP_STATIC_ASSERT(sizeof(*raddr_seg_ptr) == 16, "sizeof(*raddr_seg_ptr) == 16");
-    EP_STATIC_ASSERT(sizeof(*inl_seg_ptr) == 4, "sizeof(*inl_seg_ptr) == 4");
-    st_na_relaxed(reinterpret_cast<int4*>(ctrl_seg_ptr), *reinterpret_cast<const int4*>(&ctrl_seg));
-    st_na_relaxed(reinterpret_cast<int4*>(raddr_seg_ptr), *reinterpret_cast<const int4*>(&raddr_seg));
-    st_na_relaxed(reinterpret_cast<uint32_t*>(inl_seg_ptr), *reinterpret_cast<const uint32_t*>(&inl_seg));
-    st_na_relaxed(reinterpret_cast<uint32_t*>(wqe_data_ptr), *reinterpret_cast<const uint32_t*>(val));
-}
-
-__device__ static __forceinline__ uint64_t
-ibgda_get_lkey_and_rkey(uint64_t laddr, __be32* lkey, uint64_t raddr, int dst_pe, uint64_t* out_raddr, __be32* out_rkey, uint32_t dev_idx) {
-    auto state = ibgda_get_state();
-    auto heap_start = reinterpret_cast<uint64_t>(nvshmemi_device_state_d.heap_base);
-    auto log2_cumem_granularity = state->log2_cumem_granularity;
-
-    // Local key
-    uint64_t idx = ((laddr - heap_start) >> log2_cumem_granularity) * state->num_devices_initialized + dev_idx;
-    auto device_key = state->constmem.lkeys[idx];
-    auto lchunk_size = device_key.next_addr - laddr;
-    *lkey = device_key.key;
-
-    // Remote key
-    uint64_t roffset = raddr - heap_start;
-
-    idx = ((roffset >> log2_cumem_granularity) * nvshmemi_device_state_d.npes) * state->num_devices_initialized +
-        dst_pe * state->num_devices_initialized + dev_idx;
-    if (idx < NVSHMEMI_IBGDA_MAX_CONST_RKEYS) {
-        device_key = state->constmem.rkeys[idx];
-    } else {
-        device_key = state->globalmem.rkeys[idx - NVSHMEMI_IBGDA_MAX_CONST_RKEYS];
-    }
-    *out_raddr = reinterpret_cast<uint64_t>(nvshmemi_device_state_d.peer_heap_base_remote[dst_pe]) + roffset;
-    *out_rkey = device_key.key;
-
-    // Return the minimum of local and remote chunk sizes
-    auto rchunk_size = device_key.next_addr - roffset;
-    return min(lchunk_size, rchunk_size);
-}
-
-__device__ static __forceinline__ void ibgda_get_rkey(uint64_t addr, int dst_pe, uint64_t* out_raddr, __be32* out_rkey, uint32_t dev_idx) {
-    auto state = ibgda_get_state();
-    auto heap_start = reinterpret_cast<uint64_t>(nvshmemi_device_state_d.heap_base);
-
-    uint64_t roffset = addr - heap_start;
-    uint64_t idx = ((roffset >> state->log2_cumem_granularity) * nvshmemi_device_state_d.npes * state->num_devices_initialized) +
-        dst_pe * state->num_devices_initialized + dev_idx;
-    nvshmemi_ibgda_device_key_t device_key;
-    if (idx < NVSHMEMI_IBGDA_MAX_CONST_RKEYS)
-        device_key = state->constmem.rkeys[idx];
-    else
-        device_key = state->globalmem.rkeys[idx - NVSHMEMI_IBGDA_MAX_CONST_RKEYS];
-    *out_raddr = reinterpret_cast<uint64_t>(nvshmemi_device_state_d.peer_heap_base_remote[dst_pe]) + roffset;
-    *out_rkey = device_key.key;
-}
-
-__device__ static __forceinline__ uint64_t ibgda_reserve_wqe_slots(nvshmemi_ibgda_device_qp_t* qp, uint32_t num_wqes) {
-    auto mvars = &qp->mvars;
-    return atomicAdd(reinterpret_cast<unsigned long long*>(&mvars->tx_wq.resv_head), static_cast<unsigned long long>(num_wqes));
-}
-
-__device__ static __forceinline__ void* ibgda_get_wqe_ptr(nvshmemi_ibgda_device_qp_t* qp, uint16_t wqe_idx) {
-    uint16_t cnt = qp->tx_wq.nwqes;
-    uint16_t idx = wqe_idx & (cnt - 1);
-    return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(qp->tx_wq.wqe) + (idx << MLX5_SEND_WQE_SHIFT));
-}
-
-__device__ static __forceinline__ void nvshmemi_ibgda_rma_p(
-    int* rptr, const int value, int dst_pe, int qp_id, uint32_t imm = std::numeric_limits<uint32_t>::max()) {
-    // Get rkey
-    // NOTES: the `p` operation will not cross multiple remote chunks
-    __be32 rkey;
-    uint64_t raddr;
-    auto qp = ibgda_get_rc(dst_pe, qp_id);
-    ibgda_get_rkey(reinterpret_cast<uint64_t>(rptr), dst_pe, &raddr, &rkey, qp->dev_idx);
-
-    // Write WQEs
-    uint64_t base_wqe_idx = ibgda_reserve_wqe_slots(qp, 1);
-    void* wqe_ptrs;
-    wqe_ptrs = ibgda_get_wqe_ptr(qp, base_wqe_idx);
-    ibgda_write_rdma_write_inl_wqe(qp, reinterpret_cast<const uint32_t*>(&value), raddr, rkey, base_wqe_idx, &wqe_ptrs, imm);
-
-    // Submit requests
-    ibgda_submit_requests<true>(qp, base_wqe_idx, 1);
-}
-
-__device__ static __forceinline__ void ibgda_write_rdma_write_wqe(nvshmemi_ibgda_device_qp_t* qp,
-                                                                  uint64_t laddr,
-                                                                  __be32 lkey,
-                                                                  uint64_t raddr,
-                                                                  __be32 rkey,
-                                                                  uint32_t bytes,
-                                                                  uint16_t wqe_idx,
-                                                                  void** out_wqes) {
-    ibgda_ctrl_seg_t ctrl_seg;
-    struct mlx5_wqe_raddr_seg raddr_seg;
-    struct mlx5_wqe_data_seg data_seg;
-
-    auto* ctrl_seg_ptr = reinterpret_cast<ibgda_ctrl_seg_t*>(out_wqes[0]);
-    void* av_seg_ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(ctrl_seg_ptr) + sizeof(*ctrl_seg_ptr));
-    struct mlx5_wqe_raddr_seg* raddr_seg_ptr;
-    struct mlx5_wqe_data_seg* data_seg_ptr;
-
-    raddr_seg_ptr = reinterpret_cast<mlx5_wqe_raddr_seg*>(reinterpret_cast<uintptr_t>(av_seg_ptr));
-    data_seg_ptr = reinterpret_cast<mlx5_wqe_data_seg*>(reinterpret_cast<uintptr_t>(raddr_seg_ptr) + sizeof(*raddr_seg_ptr));
-
-    raddr_seg.raddr = HtoBE64(raddr);
-    raddr_seg.rkey = rkey;
-    raddr_seg.reserved = 0;
-
-    data_seg.byte_count = HtoBE32(bytes);
-    data_seg.lkey = lkey;
-    data_seg.addr = HtoBE64(laddr);
-
-    ctrl_seg = {0};
-    ctrl_seg.qpn_ds = HtoBE32((qp->qpn << 8) | 3);
-    ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
-    ctrl_seg.opmod_idx_opcode = HtoBE32((wqe_idx << 8) | MLX5_OPCODE_RDMA_WRITE);
-
-    EP_STATIC_ASSERT(sizeof(*ctrl_seg_ptr) == 16, "sizeof(*ctrl_seg_ptr) == 16");
-    EP_STATIC_ASSERT(sizeof(*raddr_seg_ptr) == 16, "sizeof(*raddr_seg_ptr) == 16");
-    EP_STATIC_ASSERT(sizeof(*data_seg_ptr) == 16, "sizeof(*data_seg_ptr) == 16");
-    st_na_relaxed(reinterpret_cast<int4*>(ctrl_seg_ptr), *reinterpret_cast<const int4*>(&ctrl_seg));
-    st_na_relaxed(reinterpret_cast<int4*>(raddr_seg_ptr), *reinterpret_cast<const int4*>(&raddr_seg));
-    st_na_relaxed(reinterpret_cast<int4*>(data_seg_ptr), *reinterpret_cast<const int4*>(&data_seg));
+__device__ static __forceinline__ nvshmemi_ibgda_device_state_t* ibgda_get_state() {
+    init_fake_ibgda_device_state_if_needed();
+    return &g_fake_ibgda_device_state;
 }

-__device__ static __forceinline__ void ibgda_write_empty_recv_wqe(void* out_wqe) {
-    auto* data_seg_ptr = reinterpret_cast<struct mlx5_wqe_data_seg*>(out_wqe);
-    struct mlx5_wqe_data_seg data_seg;
-
-    // Make the first segment in the WQE invalid, then the entire list will be invalid
-    data_seg.byte_count = 0;
-    data_seg.lkey = HtoBE64(MLX5_INVALID_LKEY);
-    data_seg.addr = 0;
-
-    EP_STATIC_ASSERT(sizeof(mlx5_wqe_data_seg) == sizeof(int4), "Invalid data type length");
-    st_na_relaxed(reinterpret_cast<int4*>(data_seg_ptr), *reinterpret_cast<const int4*>(&data_seg));
+__device__ static __forceinline__ void nvshmemi_ibgda_quiet(int dst_pe, int _qp_id) {
+    int host_qp = NVSHMEMX_QP_DEFAULT;
+    nvshmemx_qp_quiet(dst_pe, &host_qp, 1);
 }

 template <bool kAlwaysDoPostSend = false>
-__device__ static __forceinline__ void nvshmemi_ibgda_put_nbi_warp(
-    uint64_t req_rptr, uint64_t req_lptr, size_t bytes, int dst_pe, int qp_id, int lane_id, int message_idx) {
-    // Get lkey and rkey, store them into lanes
-    uint32_t num_wqes = 0;
-    __be32 my_lkey = 0;
-    uint64_t my_laddr = 0;
-    __be32 my_rkey = 0;
-    uint64_t my_raddr = 0;
-    uint64_t my_chunk_size = 0;
-
-    auto qp = ibgda_get_rc(dst_pe, qp_id);
-
-    // Decide how many messages (theoretically 3 for maximum)
-    auto remaining_bytes = bytes;
-    while (remaining_bytes > 0) {
-        if (lane_id == num_wqes) {
-            my_chunk_size = min(remaining_bytes,
-                                ibgda_get_lkey_and_rkey(my_laddr = req_lptr, &my_lkey, req_rptr, dst_pe, &my_raddr, &my_rkey, qp->dev_idx));
-        }
-
-        // Move one more message
-        auto chunk_size = __shfl_sync(0xffffffff, my_chunk_size, static_cast<int>(num_wqes));
-        remaining_bytes -= chunk_size;
-        req_lptr += chunk_size;
-        req_rptr += chunk_size;
-        ++num_wqes;
-    }
-    EP_DEVICE_ASSERT(num_wqes <= 32);
-
-    // Process WQE
-    uint64_t base_wqe_idx = 0;
-    if (lane_id == 0)
-        base_wqe_idx = ibgda_reserve_wqe_slots(qp, num_wqes);
-    base_wqe_idx = __shfl_sync(0xffffffff, base_wqe_idx, 0);
-    if (lane_id < num_wqes) {
-        auto wqe_idx = base_wqe_idx + lane_id;
-        auto wqe_ptr = ibgda_get_wqe_ptr(qp, wqe_idx);
-        ibgda_write_rdma_write_wqe(qp, my_laddr, my_lkey, my_raddr, my_rkey, my_chunk_size, wqe_idx, &wqe_ptr);
-    }
-    __syncwarp();
-
-    // Submit
-    if (lane_id == 0)
-        ibgda_submit_requests<kAlwaysDoPostSend>(qp, base_wqe_idx, num_wqes, message_idx);
-    __syncwarp();
-}
-
-__device__ static __forceinline__ void ibgda_write_amo_add_wqe(nvshmemi_ibgda_device_qp_t* qp,
-                                                               const int& value,
-                                                               uint64_t laddr,
-                                                               __be32 lkey,
-                                                               uint64_t raddr,
-                                                               __be32 rkey,
-                                                               uint16_t wqe_idx,
-                                                               void** out_wqes) {
-    ibgda_ctrl_seg_t ctrl_seg = {0};
-    struct mlx5_wqe_raddr_seg raddr_seg;
-    struct mlx5_wqe_atomic_seg atomic_seg_1;
-    struct mlx5_wqe_data_seg data_seg;
-
-    auto ctrl_seg_ptr = reinterpret_cast<ibgda_ctrl_seg_t*>(out_wqes[0]);
-    auto raddr_seg_ptr = reinterpret_cast<mlx5_wqe_raddr_seg*>(reinterpret_cast<uintptr_t>(ctrl_seg_ptr) + sizeof(*ctrl_seg_ptr));
-    auto atomic_seg_ptr = reinterpret_cast<mlx5_wqe_atomic_seg*>(reinterpret_cast<uintptr_t>(raddr_seg_ptr) + sizeof(*raddr_seg_ptr));
-    auto data_seg_ptr = reinterpret_cast<mlx5_wqe_data_seg*>(reinterpret_cast<uintptr_t>(atomic_seg_ptr) + sizeof(*atomic_seg_ptr));
-
-    raddr_seg.raddr = HtoBE64(raddr);
-    raddr_seg.rkey = rkey;
-    raddr_seg.reserved = 0;
-
-    // NOTES: `0x08000000` means `IBGDA_4_BYTE_EXT_AMO_OPMOD`
-    ctrl_seg.opmod_idx_opcode = HtoBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_idx << 8) | 0x08000000);
-    auto atomic_32_masked_fa_seg = reinterpret_cast<ibgda_atomic_32_masked_fa_seg_t*>(&atomic_seg_1);
-    atomic_32_masked_fa_seg->add_data = HtoBE32(value);
-    atomic_32_masked_fa_seg->field_boundary = 0;
-
-    ctrl_seg.qpn_ds = HtoBE32((qp->qpn << 8) | 4);
-    ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
-
-    data_seg.byte_count = HtoBE32(sizeof(int));
-    data_seg.lkey = lkey;
-    data_seg.addr = HtoBE64(laddr);
-
-    EP_STATIC_ASSERT(sizeof(*ctrl_seg_ptr) == sizeof(int4), "Invalid vectorization");
-    EP_STATIC_ASSERT(sizeof(*raddr_seg_ptr) == sizeof(int4), "Invalid vectorization");
-    EP_STATIC_ASSERT(sizeof(*atomic_seg_ptr) == sizeof(int4), "Invalid vectorization");
-    EP_STATIC_ASSERT(sizeof(*data_seg_ptr) == sizeof(int4), "Invalid vectorization");
-    st_na_relaxed(reinterpret_cast<int4*>(ctrl_seg_ptr), *reinterpret_cast<int4*>(&ctrl_seg));
-    st_na_relaxed(reinterpret_cast<int4*>(raddr_seg_ptr), *reinterpret_cast<int4*>(&raddr_seg));
-    st_na_relaxed(reinterpret_cast<int4*>(atomic_seg_ptr), *reinterpret_cast<int4*>(&atomic_seg_1));
-    st_na_relaxed(reinterpret_cast<int4*>(data_seg_ptr), *reinterpret_cast<int4*>(&data_seg));
+__device__ static __forceinline__ void nvshmemi_ibgda_put_nbi_warp(uint64_t req_rptr,
+                                                                   uint64_t req_lptr,
+                                                                   size_t bytes,
+                                                                   int dst_pe,
+                                                                   int _qp_id,
+                                                                   int lane_id,
+                                                                   int message_idx,
+                                                                   nvshmemx_qp_handle_t _qp_handle = NVSHMEMX_QP_DEFAULT) {
+    nvshmemx_qp_char_put_nbi_warp(reinterpret_cast<char*>(req_rptr), reinterpret_cast<char*>(req_lptr), bytes, dst_pe, NVSHMEMX_QP_DEFAULT);
 }

 __device__ __forceinline__ void nvshmemi_ibgda_amo_nonfetch_add(
-    void* rptr, const int& value, int pe, int qp_id, bool is_local_copy = false) {
+    void* rptr, const int& value, int pe, int _qp_id, bool is_local_copy = false) {
     if (is_local_copy) {
         atomicAdd(static_cast<unsigned long long*>(rptr), value);
     } else {
-        nvshmemi_ibgda_device_qp_t* qp = ibgda_get_rc(pe, qp_id);
-
-        __be32 rkey;
-        uint64_t raddr;
-        ibgda_get_rkey(reinterpret_cast<uint64_t>(rptr), pe, &raddr, &rkey, qp->dev_idx);
-
-        uint64_t my_wqe_idx = ibgda_reserve_wqe_slots(qp, 1);
-        void* wqe_ptrs = ibgda_get_wqe_ptr(qp, my_wqe_idx);
-
-        ibgda_write_amo_add_wqe(qp, value, reinterpret_cast<uint64_t>(qp->ibuf.buf), qp->ibuf.lkey, raddr, rkey, my_wqe_idx, &wqe_ptrs);
-
-        ibgda_submit_requests<true>(qp, my_wqe_idx, 1);
+        nvshmem_int_atomic_add(static_cast<int*>(rptr), value, pe);
     }
 }

 __device__ __forceinline__ uint64_t nvshmemi_get_p2p_ptr(const uint64_t& ptr, const int& rank, const int& dst_rank) {
-    // Local rank, no need for mapping
     if (rank == dst_rank)
         return ptr;
     auto peer_base = __ldg(reinterpret_cast<uint64_t*>(nvshmemi_device_state_d.peer_heap_base_p2p) + dst_rank);
-
-    // RDMA connected
     if (peer_base == 0)
         return 0;
-
-    // NVLink P2P is enabled
     return peer_base + (ptr - reinterpret_cast<uint64_t>(nvshmemi_device_state_d.heap_base));
 }

-// This is a simplified version of NVSHMEM's `ibgda_poll_cq`.
-// Note that this implementation does not guarantee thread safety,
-// so we must ensure that no other threads are concurrently using the same QP.
-__device__ static __forceinline__ void ibgda_poll_cq(nvshmemi_ibgda_device_cq_t* cq, uint64_t idx) {
-    const auto cqe64 = static_cast<mlx5_cqe64*>(cq->cqe);
-    const uint32_t ncqes = cq->ncqes;
-    memory_fence_cta();
-    if (*cq->cons_idx >= idx)
-        return;
-    // NOTES: this while loop is part of do-while below.
-    // `wqe_counter` is the HW consumer index. However, we always maintain `index + 1`.
-    // To be able to compare with the index, we need to use `wqe_counter + 1`.
-    // Because `wqe_counter` is `uint16_t`, it may be overflow. Still, we know for
-    // sure that if `idx - wqe_counter - 1 < ncqes`, `wqe_counter + 1 is less than
-    // idx, and thus we need to wait. We don't need to wait when `idx == wqe_counter + 1`
-    // That's why we use `- 2` here to make this case overflow.
-    uint16_t wqe_counter;
-    do {
-        wqe_counter = HtoBE16(ld_na_relaxed(&cqe64->wqe_counter));
-    } while ((static_cast<uint16_t>(static_cast<uint16_t>(idx) - wqe_counter - static_cast<uint16_t>(2)) < ncqes));
-    *cq->cons_idx = idx;
-
-    // Prevent reordering of this function and later instructions
-    memory_fence_cta();
+// Combined put + signal: posts data and signals the remote tail in one call.
+template <bool kAlwaysDoPostSend = false>
+__device__ static __forceinline__ void nvshmemi_ibgda_put_signal_nbi_warp(
+    uint64_t req_rptr, uint64_t req_lptr, size_t bytes,
+    void* signal_rptr, const int signal_value,
+    int dst_pe, int qp_id, int lane_id,
+    nvshmemx_qp_handle_t qp_handle = NVSHMEMX_QP_DEFAULT) {
+    nvshmemx_qp_char_put_signal_nbi_warp(reinterpret_cast<char*>(req_rptr),
+        reinterpret_cast<const char*>(req_lptr), bytes,
+        reinterpret_cast<uint64_t*>(signal_rptr), signal_value, NVSHMEM_SIGNAL_ADD,
+        dst_pe, qp_handle);
 }

-// Wait until wqe `idx - 1` is completed.
-__device__ static __forceinline__ void nvshmemi_ibgda_quiet(int dst_pe, int qp_id) {
-    auto qp = ibgda_get_rc(dst_pe, qp_id);
-    auto state = ibgda_get_state();
-    uint64_t prod_idx = state->use_async_postsend ? ld_na_relaxed(qp->tx_wq.prod_idx) : ld_na_relaxed(&qp->mvars.tx_wq.ready_head);
-    ibgda_poll_cq(qp->tx_wq.cq, prod_idx);
+__device__ static __forceinline__ void nvshmemi_ibgda_rma_p(
+    int* rptr, const int value, int dst_pe, int _qp_id, uint32_t imm = std::numeric_limits<uint32_t>::max()) {
+    nvshmemx_qp_int_p(rptr, value, dst_pe, NVSHMEMX_QP_DEFAULT);
 }

 }  // namespace deep_ep
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
diff --git a/csrc/kernels/runtime.cu b/csrc/kernels/runtime.cu
index c4fbb8e..c8c8ec3 100644
--- a/csrc/kernels/runtime.cu
+++ b/csrc/kernels/runtime.cu
@@ -13,6 +13,11 @@

 namespace deep_ep {

+#ifndef DISABLE_NVSHMEM
+__device__ nvshmemi_ibgda_device_state_t g_fake_ibgda_device_state;
+__device__ int g_fake_ibgda_state_initialized = 0;
+#endif
+
 namespace intranode {

 template <int kNumRanks>
diff --git a/deep_ep/buffer.py b/deep_ep/buffer.py
index 37512ee..b29ff63 100644
--- a/deep_ep/buffer.py
+++ b/deep_ep/buffer.py
@@ -106,7 +106,6 @@ class Buffer:
             # Enable IBGDA
             assert num_qps_per_rank > 0
             os.environ['NVSHMEM_DISABLE_P2P'] = '0' if allow_nvlink_for_low_latency_mode else '1'
-            os.environ['NVSHMEM_IB_ENABLE_IBGDA'] = '1'
             os.environ['NVSHMEM_IBGDA_NUM_RC_PER_PE'] = f'{num_qps_per_rank}'

             # Make sure QP depth is always larger than the number of on-flight WRs, so that we can skip WQ slot check
@@ -115,7 +114,7 @@ class Buffer:

             # Reduce gpu memory usage
             # 6 default teams + 1 extra team
-            os.environ['NVSHMEM_MAX_TEAMS'] = '7'
+            os.environ['NVSHMEM_MAX_TEAMS'] = '8'
             # Disable NVLink SHArP
             os.environ['NVSHMEM_DISABLE_NVLS'] = '1'
             # NOTES: NVSHMEM initialization requires at least 256 MiB

PATCHEOF

    if ! git apply --check "$efa_patch" 2>/dev/null; then
        rm -f "$efa_patch"
        die "EFA patch does not apply cleanly; the patch was validated against commit ${DEEPEP_SUPPORTED_COMMIT}. Is the tree at that commit?"
    fi

    git apply "$efa_patch"
    rm -f "$efa_patch"

    log "EFA patch applied successfully (4 files patched atomically)"
}

apply_cuda13_cccl_fix() {
    if (( CUDA_MAJOR < 13 )); then
        log "CUDA major ${CUDA_MAJOR} < 13; skipping cccl include fix"
        return 0
    fi

    local setup_py="${DEEPEP_PREFIX}/setup.py"

    if [[ ! -f "$setup_py" ]]; then
        die "cannot apply CUDA 13 cccl fix: setup.py not found at '${setup_py}'"
    fi

    if grep -q "include/cccl" "$setup_py"; then
        log "CUDA 13 cccl include path already present in setup.py; skipping (idempotent)"
        return 0
    fi

    local target_pattern="f'{nvshmem_dir}/include']"

    if ! grep -q "$target_pattern" "$setup_py"; then
        die "cannot apply CUDA 13 cccl fix: target include-list pattern not found in ${setup_py} (expected line containing: ${target_pattern})"
    fi

    sed -i'' -e "s|f'{nvshmem_dir}/include']|f'{nvshmem_dir}/include', '${CUDA_HOME}/include/cccl']|" "$setup_py"

    log "Applied CUDA 13 cccl include fix to ${setup_py} (added ${CUDA_HOME}/include/cccl)"
}

warn_nvshmem_pip_conflict() {
    local pkg="nvidia-nvshmem-cu${CUDA_MAJOR}"

    if "$PIP_CMD" show "$pkg" >/dev/null 2>&1; then
        warn "pip package '${pkg}' is installed and conflicts with the standalone NVSHMEM library at Python runtime. Please remove it manually: $PIP_CMD uninstall ${pkg}"
    fi
}

build_deepep() {
    warn_nvshmem_pip_conflict

    local disable_ptx
    disable_ptx="$(compute_disable_ptx)"

    export LD_LIBRARY_PATH="${NVSHMEM_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    if [[ "$WHEEL_ONLY" == "true" ]]; then
        log "Building DeepEP wheel (wheel-only mode)..."

        mkdir -p "$WHEEL_OUTPUT_DIR" \
            || die "failed to create wheel output directory: '${WHEEL_OUTPUT_DIR}'"

        cd "$DEEPEP_PREFIX"
        DISABLE_AGGRESSIVE_PTX_INSTRS="$disable_ptx" \
        TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
        NVSHMEM_DIR="$NVSHMEM_PREFIX" \
            "$PIP_CMD" wheel -vv --no-build-isolation --wheel-dir "$WHEEL_OUTPUT_DIR" . \
            || die "DeepEP wheel build failed"

        local wheel_file
        wheel_file="$(find "$WHEEL_OUTPUT_DIR" -maxdepth 1 -name '*.whl' -type f | sort | tail -n1)"
        if [[ -z "$wheel_file" ]]; then
            die "DeepEP wheel build completed but no .whl file found in '${WHEEL_OUTPUT_DIR}'"
        fi

        printf '%s\n' "$wheel_file"
        log "DeepEP wheel written to ${wheel_file}"
    else
        log "Building and installing DeepEP (install mode)..."

        "$PIP_CMD" uninstall -y deep_ep 2>/dev/null || true

        cd "$DEEPEP_PREFIX"
        DISABLE_AGGRESSIVE_PTX_INSTRS="$disable_ptx" \
        TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
        NVSHMEM_DIR="$NVSHMEM_PREFIX" \
            "$PIP_CMD" install -vv --no-build-isolation . \
            || die "DeepEP build/install failed"

        log "DeepEP installed successfully"
    fi
}

print_completion_notes() {
    if [[ "$GEN_LDCONFIG" == "true" ]]; then
        local ldconfig_file="/etc/ld.so.conf.d/nvshmem.conf"
        local lib_path="${NVSHMEM_PREFIX}/lib"

        if ! printf '%s\n' "$lib_path" > "$ldconfig_file" 2>/dev/null; then
            die "insufficient permissions to write ${ldconfig_file}; elevated privileges are required for --gen-ldconfig"
        fi

        if ldconfig 2>/dev/null; then
            log "Wrote ${ldconfig_file} and refreshed the linker cache successfully"
        else
            die "wrote ${ldconfig_file} but failed to refresh the linker cache (ldconfig); elevated privileges may be required"
        fi
    fi

    log ""
    log "=== DeepEP setup complete ==="
    log ""
    log "To use DeepEP at runtime, configure the following environment variables:"
    log ""
    log "  export LD_LIBRARY_PATH=${NVSHMEM_PREFIX}/lib:\${LD_LIBRARY_PATH}"
    log "  export PATH=${NVSHMEM_PREFIX}/bin:\${PATH}"
    log ""
    log "  export NVSHMEM_REMOTE_TRANSPORT=libfabric"
    log "  export NVSHMEM_LIBFABRIC_PROVIDER=efa"
    log "  export NVSHMEM_NETDEVS_POLICY=EXTERNAL_SHARING_PCIE_SWITCH_NIC_EXCLUSIVE"
    log ""
    log "NOTE: If the nvidia-nvshmem-cu${CUDA_MAJOR} Python package is installed in your environment,"
    log "      you must uninstall it to avoid conflicts with the standalone NVSHMEM library at Python runtime"
    log "      built by this script:"
    log ""
    log "  $PIP_CMD uninstall -y nvidia-nvshmem-cu${CUDA_MAJOR}"
    log ""
}

main() {
    parse_args "$@"

    if [[ -z "${NVSHMEM_SRC}" ]]; then
        validate_version "$NVSHMEM_VERSION"
    fi
    validate_commit "$DEEPEP_COMMIT"
    validate_arch_list "$TORCH_CUDA_ARCH_LIST"

    detect_arch
    detect_cuda_major
    if [[ "$SKIP_CHECKS" != "true" ]]; then
        check_python_prereqs
    fi

    if [[ -n "${NVSHMEM_SRC}" ]]; then
        build_nvshmem_from_source
    else
        install_prebuilt_nvshmem
    fi

    acquire_deepep
    commit_compat_gate
    apply_efa_patch
    apply_cuda13_cccl_fix
    build_deepep
    print_completion_notes
}

if [[ "${SETUP_DEEPEP_EFA_LIB:-}" != "1" ]]; then
    main "$@"
fi
