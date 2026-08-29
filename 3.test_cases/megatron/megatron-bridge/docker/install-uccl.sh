#!/usr/bin/env bash
set -euo pipefail
: "${UCCL_COMMIT:?}"
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST='10.0a' PER_EXPERT_BATCHING=1
export MAX_JOBS="${MAX_JOBS:-16}"
python3 -m pip install --no-cache-dir nanobind
git clone --filter=blob:none --recurse-submodules https://github.com/uccl-project/uccl.git /opt/uccl
git -C /opt/uccl checkout --detach "${UCCL_COMMIT}"
git -C /opt/uccl submodule update --init --recursive
test "$(git -C /opt/uccl rev-parse HEAD)" = "${UCCL_COMMIT}"
python3 -m pip install --no-build-isolation /opt/uccl
(cd /opt/uccl/ep && python3 setup.py install)
python3 -m pip install --no-build-isolation /opt/uccl/ep/deep_ep_wrapper
python3 - <<'PY'
import deep_ep
import uccl.ep
assert hasattr(deep_ep, "Buffer")
assert not hasattr(deep_ep, "ElasticBuffer")
print("UCCL DeepEP-compatible Buffer API: PASS")
PY

