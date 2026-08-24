#!/usr/bin/env bash
set -euo pipefail
: "${DEEPEP_V1_COMMIT:?}"
: "${NVSHMEM_VERSION:?}"
export TORCH_CUDA_ARCH_LIST='10.0'
export NVSHMEM_REMOTE_TRANSPORT=libfabric NVSHMEM_LIBFABRIC_PROVIDER=efa
export LIBRARY_PATH="/usr/local/cuda/lib64/stubs${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cuda_stub=/usr/local/cuda/lib64/stubs/libcuda.so
test -f "${cuda_stub}"
[[ -e "${cuda_stub}.1" ]] || ln -s "${cuda_stub}" "${cuda_stub}.1"
trap 'rm -f /usr/local/cuda/lib64/stubs/libcuda.so.1' EXIT
/opt/benchmark/setup_deepep_efa.sh \
  --nvshmem-version "${NVSHMEM_VERSION}" \
  --nvshmem-prefix /opt/amazon/nvshmem \
  --deepep-commit "${DEEPEP_V1_COMMIT}" \
  --deepep-prefix /opt/amazon/deepep \
  --libfabric-home /opt/amazon/efa \
  --gdrcopy-home /opt/gdrcopy \
  --cuda-home /usr/local/cuda \
  --gen-ldconfig --force
test "$(git -C /opt/amazon/deepep rev-parse HEAD)" = "${DEEPEP_V1_COMMIT}"
python3 - <<'PY'
import deep_ep
assert hasattr(deep_ep, "Buffer")
assert not hasattr(deep_ep, "ElasticBuffer")
print("DeepEP v1 Buffer API: PASS")
PY
