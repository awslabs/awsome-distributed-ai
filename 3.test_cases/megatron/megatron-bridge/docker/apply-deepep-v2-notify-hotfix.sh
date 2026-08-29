#!/usr/bin/env bash
set -euo pipefail
: "${DEEPEP_NOTIFY_WARPS_PATCH_SHA256:?}"
: "${DEEPEP_SYNTHETIC_TREE:?}"
: "${DEEPEP_SYNTHETIC_COMMIT:?}"

export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST='10.0'
export DISABLE_AGGRESSIVE_PTX_INSTRS=1
export EP_NCCL_ROOT_DIR=/opt/nccl/build EP_NVSHMEM_ROOT_DIR=/opt/amazon/nvshmem
export NVSHMEM_DIR=/opt/amazon/nvshmem MAX_JOBS="${MAX_JOBS:-16}"
export LIBRARY_PATH="/usr/local/cuda/lib64/stubs${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cuda_stub=/usr/local/cuda/lib64/stubs/libcuda.so
test -f "${cuda_stub}"
[[ -e "${cuda_stub}.1" ]] || ln -s "${cuda_stub}" "${cuda_stub}.1"
trap 'rm -f /usr/local/cuda/lib64/stubs/libcuda.so.1' EXIT

patch=/opt/benchmark/patches/deepep-v2-notify-warps.patch
echo "${DEEPEP_NOTIFY_WARPS_PATCH_SHA256}  ${patch}" | sha256sum --check --strict
repo=/opt/amazon/deepep-v2
test "$(git -C "${repo}" rev-parse HEAD)" = b56ebf8bb4ece24cd78aa8c12550b24e35ac255b
test -z "$(git -C "${repo}" status --porcelain)"
git -C "${repo}" apply --check "${patch}"
git -C "${repo}" apply --index "${patch}"
tree="$(git -C "${repo}" write-tree)"
test "${tree}" = "${DEEPEP_SYNTHETIC_TREE}"

export GIT_AUTHOR_NAME='AWSome Distributed AI Benchmark'
export GIT_AUTHOR_EMAIL='adai-benchmark@amazon.com'
export GIT_COMMITTER_NAME="${GIT_AUTHOR_NAME}" GIT_COMMITTER_EMAIL="${GIT_AUTHOR_EMAIL}"
export GIT_AUTHOR_DATE='2026-08-24T06:42:00Z' GIT_COMMITTER_DATE='2026-08-24T06:42:00Z'
message='DeepEP v2 synthetic: base + PR #3 + PR #5 + notify-warp fix

base: 02efc268a37802fc00812ede8f5ad7f535ceea0e
pr3-head: dd0f87261a80cf0ce8aa66e4ab2041843851d810
pr5-head: 2542d9641f2ec280213e875feb04be7862dda57c
notify-warp-patch-sha256: 198fe6148b14b3c4533542f43e043835c47a861af37021628f4942f5b114a039'
synthetic="$(printf '%s\n' "${message}" | git -C "${repo}" commit-tree "${tree}" -p 02efc268a37802fc00812ede8f5ad7f535ceea0e)"
test "${synthetic}" = "${DEEPEP_SYNTHETIC_COMMIT}"
git -C "${repo}" reset --hard "${synthetic}"

printf '%s  %s\n' "${DEEPEP_NOTIFY_WARPS_PATCH_SHA256}" "${patch}" \
  > /opt/benchmark/deepep-v2-notify-warps.patch.sha256
(cd "${repo}" && find deep_ep/include -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}' > /opt/benchmark/deepep-v2-jit-headers.sha256)
python3 -m pip install --force-reinstall --no-deps --no-build-isolation -v "${repo}"
python3 - <<'PY'
import deep_ep
assert hasattr(deep_ep, "ElasticBuffer")
print("DeepEP v2 dynamic notify-warp hotfix: PASS")
PY
