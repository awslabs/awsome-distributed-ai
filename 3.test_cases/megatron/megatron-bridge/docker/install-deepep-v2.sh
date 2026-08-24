#!/usr/bin/env bash
set -euo pipefail
: "${DEEPEP_V2_BASE:?}"
: "${DEEPEP_PR3_HEAD:?}"
: "${DEEPEP_PR3_PATCH_SHA256:?}"
: "${DEEPEP_PR5_HEAD:?}"
: "${DEEPEP_PR5_PATCH_SHA256:?}"
: "${DEEPEP_SYNTHETIC_TREE:?}"
: "${DEEPEP_SYNTHETIC_COMMIT:?}"
: "${NVSHMEM_VERSION:?}"
: "${NVSHMEM_SHA256:?}"
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST='10.0'
export DISABLE_AGGRESSIVE_PTX_INSTRS=1
export EP_NCCL_ROOT_DIR=/opt/nccl/build EP_NVSHMEM_ROOT_DIR=/opt/amazon/nvshmem
export NVSHMEM_DIR=/opt/amazon/nvshmem MAX_JOBS="${MAX_JOBS:-16}"

archive="libnvshmem-linux-x86_64-${NVSHMEM_VERSION}_cuda13-archive.tar.xz"
curl --fail --location --retry 5 \
  "https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-x86_64/${archive}" \
  --output "/tmp/${archive}"
echo "${NVSHMEM_SHA256}  /tmp/${archive}" | sha256sum --check --strict
mkdir -p /opt/amazon/nvshmem
tar -xJf "/tmp/${archive}" --strip-components=1 -C /opt/amazon/nvshmem
rm -f "/tmp/${archive}"

echo "${DEEPEP_PR3_PATCH_SHA256}  /opt/benchmark/patches/deepep-pr3.patch" | sha256sum --check --strict
echo "${DEEPEP_PR5_PATCH_SHA256}  /opt/benchmark/patches/deepep-pr5.patch" | sha256sum --check --strict
git clone --no-checkout https://github.com/amazon-contributing/DeepEP.git /opt/amazon/deepep-v2
repo=/opt/amazon/deepep-v2
git -C "${repo}" fetch --no-tags --depth=1 origin "${DEEPEP_V2_BASE}"
git -C "${repo}" checkout --detach "${DEEPEP_V2_BASE}"
git -C "${repo}" apply --index /opt/benchmark/patches/deepep-pr3.patch
git -C "${repo}" apply --index /opt/benchmark/patches/deepep-pr5.patch
tree="$(git -C "${repo}" write-tree)"
test "${tree}" = "${DEEPEP_SYNTHETIC_TREE}"
export GIT_AUTHOR_NAME='AWSome Distributed AI Benchmark'
export GIT_AUTHOR_EMAIL='adai-benchmark@amazon.com'
export GIT_COMMITTER_NAME="${GIT_AUTHOR_NAME}" GIT_COMMITTER_EMAIL="${GIT_AUTHOR_EMAIL}"
export GIT_AUTHOR_DATE='2026-08-23T18:41:17Z' GIT_COMMITTER_DATE='2026-08-23T18:41:17Z'
message='DeepEP v2 synthetic: base + PR #3 + PR #5

base: 02efc268a37802fc00812ede8f5ad7f535ceea0e
pr3-head: dd0f87261a80cf0ce8aa66e4ab2041843851d810
pr5-head: 2542d9641f2ec280213e875feb04be7862dda57c'
synthetic="$(printf '%s\n' "${message}" | git -C "${repo}" commit-tree "${tree}" -p "${DEEPEP_V2_BASE}")"
test "${synthetic}" = "${DEEPEP_SYNTHETIC_COMMIT}"
git -C "${repo}" reset --hard "${synthetic}"
(cd "${repo}" && find deep_ep/include -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}' > /opt/benchmark/deepep-v2-jit-headers.sha256)
python3 -m pip install --no-build-isolation -v "${repo}"
python3 - <<'PY'
import deep_ep
assert hasattr(deep_ep, "ElasticBuffer")
print("DeepEP v2 ElasticBuffer API: PASS")
PY

