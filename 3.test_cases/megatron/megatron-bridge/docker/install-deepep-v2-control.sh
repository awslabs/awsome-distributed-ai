#!/usr/bin/env bash
set -euo pipefail
: "${DEEPEP_V2_BASE:?}"
: "${DEEPEP_PR3_PATCH_SHA256:?}"
: "${DEEPEP_CONTROL_TREE:?}"
: "${DEEPEP_CONTROL_COMMIT:?}"
: "${NVSHMEM_VERSION:?}"
: "${NVSHMEM_SHA256:?}"
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST='10.0' DISABLE_AGGRESSIVE_PTX_INSTRS=1 MAX_JOBS="${MAX_JOBS:-16}"
export EP_NCCL_ROOT_DIR=/opt/nccl/build EP_NVSHMEM_ROOT_DIR=/opt/amazon/nvshmem NVSHMEM_DIR=/opt/amazon/nvshmem
export LIBRARY_PATH="/usr/local/cuda/lib64/stubs${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cuda_stub=/usr/local/cuda/lib64/stubs/libcuda.so
test -f "${cuda_stub}"
[[ -e "${cuda_stub}.1" ]] || ln -s "${cuda_stub}" "${cuda_stub}.1"
trap 'rm -f /usr/local/cuda/lib64/stubs/libcuda.so.1' EXIT
archive="libnvshmem-linux-x86_64-${NVSHMEM_VERSION}_cuda13-archive.tar.xz"
curl --fail --location --retry 5 \
  "https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-x86_64/${archive}" \
  --output "/tmp/${archive}"
echo "${NVSHMEM_SHA256}  /tmp/${archive}" | sha256sum --check --strict
mkdir -p /opt/amazon/nvshmem
tar -xJf "/tmp/${archive}" --strip-components=1 -C /opt/amazon/nvshmem
echo "${DEEPEP_PR3_PATCH_SHA256}  /opt/benchmark/patches/deepep-pr3.patch" | sha256sum --check --strict
git clone --no-checkout https://github.com/amazon-contributing/DeepEP.git /opt/amazon/deepep-v2
repo=/opt/amazon/deepep-v2
git -C "${repo}" fetch --no-tags --depth=1 origin "${DEEPEP_V2_BASE}"
git -C "${repo}" checkout --detach "${DEEPEP_V2_BASE}"
git -C "${repo}" apply --index /opt/benchmark/patches/deepep-pr3.patch
test "$(git -C "${repo}" write-tree)" = "${DEEPEP_CONTROL_TREE}"
export GIT_AUTHOR_NAME='AWSome Distributed AI Benchmark' GIT_AUTHOR_EMAIL='adai-benchmark@amazon.com'
export GIT_COMMITTER_NAME="${GIT_AUTHOR_NAME}" GIT_COMMITTER_EMAIL="${GIT_AUTHOR_EMAIL}"
export GIT_AUTHOR_DATE='2026-08-23T18:41:17Z' GIT_COMMITTER_DATE='2026-08-23T18:41:17Z'
message='DeepEP v2 PR #5 control: base + PR #3

base: 02efc268a37802fc00812ede8f5ad7f535ceea0e
pr3-head: dd0f87261a80cf0ce8aa66e4ab2041843851d810'
commit="$(printf '%s\n' "${message}" | git -C "${repo}" commit-tree "${DEEPEP_CONTROL_TREE}" -p "${DEEPEP_V2_BASE}")"
test "${commit}" = "${DEEPEP_CONTROL_COMMIT}"
git -C "${repo}" reset --hard "${commit}"
python3 -m pip install --no-build-isolation -v "${repo}"
