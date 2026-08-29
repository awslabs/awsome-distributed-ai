#!/usr/bin/env bash
set -euo pipefail

: "${NCCL_TAG:?}"

PYTORCH_SOURCE=/opt/pytorch/pytorch
NCCL_ROOT=/opt/nccl/build
EXPECTED_NCCL=23102
PYTORCH_COMMIT=8145d630e811c9ec098a632d9ea7f4d52a2ea16e
ONNX_COMMIT=e709452ef2bbc1d113faf678c24e6d3467696e83
ONNX_SOURCE="${PYTORCH_SOURCE}/third_party/onnx"

test -f "${PYTORCH_SOURCE}/setup.py"
test -f "${PYTORCH_SOURCE}/version.txt"
test -f "${NCCL_ROOT}/include/nccl.h"
test -f "${NCCL_ROOT}/lib/libnccl.so.2.31.2"
test "${NCCL_TAG}" = v2.31.2-1
test -f /usr/local/include/mkl.h
for mkl_component in intel_lp64 gnu_thread core; do
  test -f "/usr/local/lib/libmkl_${mkl_component}.so.1"
  ln -sfn "libmkl_${mkl_component}.so.1" "/usr/local/lib/libmkl_${mkl_component}.so"
done

# NVIDIA's source layer omits this one submodule. The full PyTorch commit is
# encoded in the shipped package version, and its public gitlink supplies the
# exact ONNX commit rather than an inferred compatible release.
if [[ ! -f "${ONNX_SOURCE}/CMakeLists.txt" ]]; then
  rm -rf "${ONNX_SOURCE}"
  git clone --filter=blob:none https://github.com/onnx/onnx.git "${ONNX_SOURCE}"
  git -C "${ONNX_SOURCE}" checkout --detach "${ONNX_COMMIT}"
  git -C "${ONNX_SOURCE}" submodule update --init --recursive
fi
test "$(git -C "${ONNX_SOURCE}" rev-parse HEAD)" = "${ONNX_COMMIT}"
test -f "${ONNX_SOURCE}/third_party/pybind11/CMakeLists.txt"
printf '%s\n' "${PYTORCH_COMMIT}" > /opt/benchmark/pytorch-upstream-commit.txt
printf '%s\n' "${ONNX_COMMIT}" > /opt/benchmark/pytorch-onnx-commit.txt

# The NeMo container retains the exact source tree used for its packaged Torch,
# but strips the parent repository that the pytorch submodule's .git file names.
# Hash that shipped tree before building so the source remains independently
# identifiable even though torch.version.git_version is unavailable.
find "${PYTORCH_SOURCE}" -type f \
  ! -path '*/.git' \
  ! -path '*/.git/*' \
  ! -path '*/build/*' \
  ! -path '*/dist/*' \
  ! -path '*/torch.egg-info/*' \
  -print0 | sort -z | xargs -0 sha256sum | sha256sum |
  tee /opt/benchmark/pytorch-source-tree.sha256

export BUILD_TEST=0
export CMAKE_BUILD_TYPE=Release
export CMAKE_GENERATOR=Ninja
export CMAKE_PREFIX_PATH="${NCCL_ROOT}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
export CMAKE_INCLUDE_PATH="${NCCL_ROOT}/include:/usr/local/include${CMAKE_INCLUDE_PATH:+:${CMAKE_INCLUDE_PATH}}"
export CMAKE_LIBRARY_PATH="${NCCL_ROOT}/lib:/usr/local/lib${CMAKE_LIBRARY_PATH:+:${CMAKE_LIBRARY_PATH}}"
export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
export MAX_JOBS="${MAX_JOBS:-96}"
export MKLROOT=/usr/local
export NCCL_INCLUDE_DIR="${NCCL_ROOT}/include"
export NCCL_LIB_DIR="${NCCL_ROOT}/lib"
export NCCL_ROOT
export NCCL_ROOT_DIR="${NCCL_ROOT}"
export NCCL_VERSION=2.31.2
export PYTORCH_BUILD_NUMBER=1
export PYTORCH_BUILD_VERSION=2.13.0a0+8145d630e8.nv26.06
export TORCH_CUDA_ARCH_LIST=10.0
export USE_CUDA=1
export USE_CUDNN=1
export USE_CUSPARSELT=1
export USE_DISTRIBUTED=1
export USE_GLOO=1
export USE_KINETO=1
export USE_MKL=1
export USE_MKLDNN=1
export USE_MPI=1
export USE_NCCL=1
export USE_NNPACK=1
export USE_ROCM=0
export USE_STATIC_NCCL=0
export USE_SYSTEM_NCCL=1
export USE_XPU=0

ccache --max-size 100G
rm -rf "${PYTORCH_SOURCE}/build" "${PYTORCH_SOURCE}/dist" "${PYTORCH_SOURCE}/torch.egg-info"
python3 -m pip install --no-cache-dir --upgrade wheel
(
  cd "${PYTORCH_SOURCE}"
  python3 setup.py bdist_wheel
)

mapfile -t wheels < <(find "${PYTORCH_SOURCE}/dist" -maxdepth 1 -type f -name 'torch-*.whl' | sort)
test "${#wheels[@]}" -eq 1
python3 -m pip install --no-cache-dir --no-deps --force-reinstall "${wheels[0]}"

LD_PRELOAD="${NCCL_ROOT}/lib/libnccl.so.2" python3 - "${EXPECTED_NCCL}" <<'PY'
import ctypes
import pathlib
import sys

import torch

expected = int(sys.argv[1])
build_tuple = torch.cuda.nccl.version()
build = build_tuple[0] * 10000 + build_tuple[1] * 100 + build_tuple[2]
runtime = ctypes.c_int()
nccl_path = pathlib.Path("/opt/nccl/build/lib/libnccl.so.2").resolve()
nccl = ctypes.CDLL(str(nccl_path))
assert nccl.ncclGetVersion(ctypes.byref(runtime)) == 0
assert build == expected, (build_tuple, expected)
assert runtime.value == expected, (runtime.value, expected)
print(f"PYTORCH_NCCL_REBUILD_OK build={build} runtime={runtime.value} path={nccl_path}")
PY
