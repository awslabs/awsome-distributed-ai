#!/usr/bin/env bash
set -euo pipefail

: "${BRIDGE_COMMIT:?}"
: "${MCORE_COMMIT:?}"
: "${NCCL_TAG:?}"
: "${NCCL_COMMIT:?}"
: "${EFA_INSTALLER_VERSION:?}"
: "${EFA_INSTALLER_SHA256:?}"
: "${GDRCOPY_TAG:?}"

test "$(git -C /opt/Megatron-Bridge rev-parse HEAD)" = "${BRIDGE_COMMIT}"
test "$(git -C /opt/Megatron-Bridge/3rdparty/Megatron-LM rev-parse HEAD)" = "${MCORE_COMMIT}"
echo '9d54dd664daded6c3e4b105a77d578fea4340121188ea3a4261e33cec1523ce0  /opt/benchmark/patches/bridge-deepep-v2.patch' | sha256sum --check --strict
echo '1f182f801ec2d086158000d9bd6300305129aecd12ce6dcc3a8d5eb966ceb0a8  /opt/benchmark/patches/mcore-deepep-v2-elastic.patch' | sha256sum --check --strict
git -C /opt/Megatron-Bridge apply --check /opt/benchmark/patches/bridge-deepep-v2.patch
git -C /opt/Megatron-Bridge apply /opt/benchmark/patches/bridge-deepep-v2.patch
git -C /opt/Megatron-Bridge/3rdparty/Megatron-LM apply --check /opt/benchmark/patches/mcore-deepep-v2-elastic.patch
git -C /opt/Megatron-Bridge/3rdparty/Megatron-LM apply /opt/benchmark/patches/mcore-deepep-v2-elastic.patch
python3 -m py_compile \
  /opt/Megatron-Bridge/src/megatron/bridge/training/flex_dispatcher_backend.py \
  /opt/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/transformer/moe/fused_a2a.py \
  /opt/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/transformer/moe/token_dispatcher.py

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  autoconf automake build-essential ca-certificates ccache cmake curl git jq \
  libibverbs-dev libnl-3-dev libnl-route-3-dev libnuma-dev libtool ninja-build \
  numactl pciutils pkg-config rdma-core wget

git clone --filter=blob:none --branch "${GDRCOPY_TAG}" https://github.com/NVIDIA/gdrcopy.git /opt/gdrcopy-src
test "$(git -C /opt/gdrcopy-src describe --tags --exact-match)" = "${GDRCOPY_TAG}"
make -C /opt/gdrcopy-src -j"$(nproc)" prefix=/opt/gdrcopy lib lib_install

curl --fail --location --retry 5 \
  "https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz" \
  --output /tmp/aws-efa-installer.tar.gz
echo "${EFA_INSTALLER_SHA256}  /tmp/aws-efa-installer.tar.gz" | sha256sum --check --strict
mkdir -p /tmp/aws-efa-installer
tar -xzf /tmp/aws-efa-installer.tar.gz --strip-components=1 -C /tmp/aws-efa-installer
(cd /tmp/aws-efa-installer && ./efa_installer.sh --yes --skip-kmod --skip-limit-conf --no-verify)
fi_info --version | grep -q 'libfabric: 2.6.0'
test -e /opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so || \
  test -e /opt/amazon/ofi-nccl/lib/libnccl-net.so || \
  test -e /opt/amazon/ofi-nccl/lib/x86_64-linux-gnu/libnccl-net.so

git clone --filter=blob:none https://github.com/NVIDIA/nccl.git /opt/nccl
git -C /opt/nccl checkout --detach "${NCCL_COMMIT}"
test "$(git -C /opt/nccl describe --tags --exact-match)" = "${NCCL_TAG}"
make -C /opt/nccl -j"$(nproc)" src.build CUDA_HOME=/usr/local/cuda \
  NVCC_GENCODE='-gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_100,code=compute_100'
test -e /opt/nccl/build/lib/libnccl.so.2

for libdir in /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64; do
  if [[ -d "${libdir}" ]]; then
    ln -sfn /opt/nccl/build/lib/libnccl.so.2 "${libdir}/libnccl.so.2"
    ln -sfn /opt/nccl/build/lib/libnccl.so "${libdir}/libnccl.so"
  fi
done
ldconfig

rm -rf /tmp/aws-efa-installer /tmp/aws-efa-installer.tar.gz
apt-get clean
rm -rf /var/lib/apt/lists/*
