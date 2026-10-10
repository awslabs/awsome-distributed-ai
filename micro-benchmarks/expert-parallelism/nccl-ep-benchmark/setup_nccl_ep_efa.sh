#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# setup_nccl_ep_efa.sh - install the EFA userspace and build NVIDIA NCCL EP (nccl-extensions nccl_ep)
# for the NCCL-GIN CPU proxy (NCCL_GIN_TYPE=2) on AWS EFA. Every input is public and pinned.
#
#   setup_nccl_ep_efa.sh deps|gdrcopy|efa|nccl|nccl-ep|all
#
# The Dockerfile runs one stage per layer. The same script can run inside a running pod built
# from the plain NGC CUDA devel image (stage "all"), which produces the same /opt layout.
# Every stage fails closed: a wrong hash, a wrong version string or a missing file stops it.
#
# Result layout:
#   /opt/nccl           NCCL_HOME: headers + libnccl.so.2 from ONE nvidia-nccl-cu13 wheel
#   /opt/nccl-ep        BUILDDIR: lib/libnccl_ep.so.0.2.0, include/nccl_ep (JIT headers),
#                       test/nccl_ep/{ep_test,ep_bench}, BUILD-RECORD.txt
#   /opt/gdrcopy        GDRCopy userspace (library + headers only; the kernel module is a host concern)
#   /opt/amazon/...     EFA installer: libfabric, rdma-core, aws-ofi-nccl, Open MPI 4
set -euo pipefail

NCCL_EXTENSIONS_REPO="${NCCL_EXTENSIONS_REPO:-https://github.com/NVIDIA/nccl-extensions.git}"
NCCL_EXTENSIONS_COMMIT="${NCCL_EXTENSIONS_COMMIT:-901c4e65d6c3c8141499902c99838573de43a253}"
# EP_-prefixed: the CUDA base image already sets ENV NCCL_VERSION (its own, removed, NCCL).
EP_NCCL_VERSION="${EP_NCCL_VERSION:-2.32.3}"
EP_NCCL_VERSION_CODE="${EP_NCCL_VERSION_CODE:-23203}"
NCCL_WHEEL_URL="${NCCL_WHEEL_URL:-https://files.pythonhosted.org/packages/5b/29/6b277e63c92d91f9cb4d1a3a554e148983de39d54baa652bb52c798af78e/nvidia_nccl_cu13-2.32.3-py3-none-manylinux_2_27_x86_64.whl}"
NCCL_WHEEL_SHA256="${NCCL_WHEEL_SHA256:-1459723080ac889d73a26edfa3e04383a7928ab31ac8f0ec43b3ea9548b04ff3}"
EFA_INSTALLER_VERSION="${EFA_INSTALLER_VERSION:-1.50.0}"
EFA_INSTALLER_SHA256="${EFA_INSTALLER_SHA256:-fa6dff8593d866866c13cb4640d9059835cd4efa427971f100ab40c97bef2841}"
AWS_OFI_NCCL_EXPECTED="${AWS_OFI_NCCL_EXPECTED:-1.21.1}"
GDRCOPY_COMMIT="${GDRCOPY_COMMIT:-c91ad9f178e5fb729fc5b6dc62a77c3bb364d6c9}"   # tag v2.5.2
NVCC_GENCODE="${NVCC_GENCODE:--gencode=arch=compute_90,code=sm_90}"           # H100/H200
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"

NCCL_HOME=/opt/nccl
EP_BUILD=/opt/nccl-ep
OFI_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so
MPI_HOME=/opt/amazon/openmpi
WORK="${SETUP_WORK_DIR:-/tmp/nccl-ep-setup}"

log() { echo "[setup_nccl_ep_efa $(date -u +%H:%M:%S)] $*"; }
die() { echo "[setup_nccl_ep_efa] FATAL: $*" >&2; exit 1; }
# Count matches without `grep -q` (a closed pipe under pipefail is a SIGPIPE flake).
count() { grep -c -- "$1" || true; }

stage_deps() {
  log "deps: remove the base image NCCL, install build and runtime packages"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y
  # The CUDA devel base ships libnccl2/libnccl-dev (2.28.3, held). Remove them so the process can
  # only ever load the one NCCL this recipe builds against.
  if dpkg -s libnccl2 >/dev/null 2>&1 || dpkg -s libnccl-dev >/dev/null 2>&1; then
    apt-get remove -y --allow-change-held-packages libnccl2 libnccl-dev
  fi
  apt-get install -y --no-install-recommends \
    ca-certificates curl git unzip pciutils environment-modules tcl \
    openssh-client openssh-server
  # Keep the apt lists: the EFA installer runs its own apt-get install.
}

stage_gdrcopy() {
  log "gdrcopy: userspace library at commit ${GDRCOPY_COMMIT}"
  rm -rf "$WORK/gdrcopy"; mkdir -p "$WORK"
  git clone -q https://github.com/NVIDIA/gdrcopy.git "$WORK/gdrcopy"
  git -C "$WORK/gdrcopy" checkout -q "$GDRCOPY_COMMIT"
  [ "$(git -C "$WORK/gdrcopy" rev-parse HEAD)" = "$GDRCOPY_COMMIT" ] || die "gdrcopy HEAD is not ${GDRCOPY_COMMIT}"
  make -C "$WORK/gdrcopy" prefix=/opt/gdrcopy lib lib_install
  echo /opt/gdrcopy/lib > /etc/ld.so.conf.d/50-gdrcopy.conf
  ldconfig
  [ -e /opt/gdrcopy/lib/libgdrapi.so.2 ] || die "libgdrapi.so.2 missing after install"
  rm -rf "$WORK/gdrcopy"
}

stage_efa() {
  log "efa: AWS EFA installer ${EFA_INSTALLER_VERSION} (userspace only)"
  local tgz="$WORK/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz"
  mkdir -p "$WORK"
  curl --retry 3 -fsSL -o "$tgz" "https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz"
  echo "${EFA_INSTALLER_SHA256}  ${tgz}" | sha256sum -c -
  tar -xzf "$tgz" -C "$WORK"
  # Same flags as micro-benchmarks/expert-parallelism/deepep-v2-benchmark. The NGC CUDA base has
  # /opt/nvidia/nvidia_entrypoint.sh, so the installer selects its NGC plugin package
  # (libnccl-ofi-ngc-v3), installed at /opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so.
  (cd "$WORK/aws-efa-installer" && ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify)
  ldconfig
  [ -f "$OFI_PLUGIN" ] || die "aws-ofi-nccl plugin not at ${OFI_PLUGIN}"
  [ "$(strings "$OFI_PLUGIN" | count "aws-ofi-nccl ${AWS_OFI_NCCL_EXPECTED}")" -ge 1 ] \
    || die "${OFI_PLUGIN} is not aws-ofi-nccl ${AWS_OFI_NCCL_EXPECTED}"
  [ "$(nm -D --defined-only "$OFI_PLUGIN" | count ncclRmaPlugin_v15)" -ge 1 ] \
    || die "plugin lacks ncclRmaPlugin_v15 (the RMA table the NCCL 2.32 GIN proxy uses)"
  [ -x "$MPI_HOME/bin/mpirun" ] || die "Open MPI 4 not at ${MPI_HOME}"
  [ -x /opt/amazon/efa/bin/fi_info ] || die "libfabric fi_info missing"
  rm -rf "$tgz" "$WORK/aws-efa-installer"
}

stage_nccl() {
  log "nccl: nvidia-nccl-cu13 ${EP_NCCL_VERSION} wheel as NCCL_HOME=${NCCL_HOME}"
  local whl
  whl="$WORK/$(basename "$NCCL_WHEEL_URL")"
  mkdir -p "$WORK"
  curl --retry 3 -fsSL -o "$whl" "$NCCL_WHEEL_URL"
  echo "${NCCL_WHEEL_SHA256}  ${whl}" | sha256sum -c -
  rm -rf "$WORK/wheel" "$NCCL_HOME"
  unzip -q "$whl" 'nvidia/nccl/*' -d "$WORK/wheel"
  mv "$WORK/wheel/nvidia/nccl" "$NCCL_HOME"
  # The wheel ships only the soname file; add the linker name so -lnccl resolves.
  ln -sfn libnccl.so.2 "$NCCL_HOME/lib/libnccl.so"
  echo "$NCCL_HOME/lib" > /etc/ld.so.conf.d/00-nccl.conf
  ldconfig
  [ "$(awk '/#define NCCL_VERSION_CODE/ {print $3}' "$NCCL_HOME/include/nccl.h")" = "$EP_NCCL_VERSION_CODE" ] \
    || die "nccl.h NCCL_VERSION_CODE is not ${EP_NCCL_VERSION_CODE}"
  [ -f "$NCCL_HOME/include/nccl_device.h" ] || die "device-API headers (nccl_device.h) missing"
  [ "$(strings "$NCCL_HOME/lib/libnccl.so.2" | count "NCCL version ${EP_NCCL_VERSION}")" -ge 1 ] \
    || die "libnccl.so.2 is not ${EP_NCCL_VERSION}"
  # One NCCL per process starts with one NCCL in the image.
  local found
  found=$(find / -xdev \( -path /proc -o -path /sys \) -prune -o -name 'libnccl.so*' -type f -print 2>/dev/null | sort)
  [ "$found" = "$NCCL_HOME/lib/libnccl.so.2" ] || die "expected exactly one libnccl file, found: ${found//$'\n'/ }"
  rm -rf "$whl" "$WORK/wheel"
}

stage_nccl_ep() {
  log "nccl-ep: nccl-extensions ${NCCL_EXTENSIONS_COMMIT} (lib, ep_test, ep_bench)"
  local src="$WORK/nccl-extensions"
  rm -rf "$src" "$EP_BUILD"; mkdir -p "$WORK"
  git clone -q "$NCCL_EXTENSIONS_REPO" "$src"
  git -C "$src" checkout -q "$NCCL_EXTENSIONS_COMMIT"
  [ "$(git -C "$src" rev-parse HEAD)" = "$NCCL_EXTENSIONS_COMMIT" ] || die "nccl-extensions HEAD is not the pin"
  local mk=(-C "$src/nccl_ep" NCCL_HOME="$NCCL_HOME" BUILDDIR="$EP_BUILD" NVCC_GENCODE="$NVCC_GENCODE")
  make "${mk[@]}" -j "$BUILD_JOBS" lib
  # With nvcc, -lnccl_ep picks libnccl_ep.a when it exists, which would compile NCCL EP into the
  # test binaries. Move the archive aside so ep_test/ep_bench load the shared library that
  # frameworks load. The two file targets depend only on their object and the shared library.
  mkdir -p "$EP_BUILD/lib/static"
  mv "$EP_BUILD/lib/libnccl_ep.a" "$EP_BUILD/lib/static/"
  make "${mk[@]}" -j "$BUILD_JOBS" MPI=1 MPI_HOME="$MPI_HOME" \
    "$EP_BUILD/test/nccl_ep/ep_test" "$EP_BUILD/test/nccl_ep/ep_bench"

  local lib="$EP_BUILD/lib/libnccl_ep.so.0.2.0" exe
  [ -f "$lib" ] || die "missing ${lib}"
  [ -d "$EP_BUILD/include/nccl_ep" ] || die "JIT kernel headers missing under ${EP_BUILD}/include/nccl_ep"
  [ "$(strings "$lib" | count 'are incompatible')" -ge 1 ] \
    || die "runtime NCCL version check (validateNcclRuntimeVersion) not compiled in"
  readelf -d "$lib" | grep -F '[libnccl.so.2]' > /dev/null || die "libnccl_ep does not link libnccl.so.2"
  for exe in ep_test ep_bench; do
    readelf -d "$EP_BUILD/test/nccl_ep/$exe" > "$WORK/$exe.dynamic"
    grep -F '[libnccl_ep.so.0]' "$WORK/$exe.dynamic" > /dev/null || die "${exe} does not load libnccl_ep.so.0"
    grep -F '[libnccl.so.2]' "$WORK/$exe.dynamic" > /dev/null || die "${exe} does not load libnccl.so.2"
    grep -F "$NCCL_HOME/lib" "$WORK/$exe.dynamic" > /dev/null || die "${exe} RUNPATH lacks ${NCCL_HOME}/lib"
  done
  # Every dependency must resolve with the runtime library path the launcher passes to each rank.
  # libcuda.so.1 is excluded: the NVIDIA container runtime injects it when a GPU is attached.
  local rt_ld="$NCCL_HOME/lib:$EP_BUILD/lib:/opt/amazon/ofi-nccl/lib:$MPI_HOME/lib:/opt/amazon/efa/lib:/opt/gdrcopy/lib:/usr/local/cuda/lib64"
  local missing obj
  for obj in "$EP_BUILD/test/nccl_ep/ep_test" "$EP_BUILD/test/nccl_ep/ep_bench" "$lib" "$OFI_PLUGIN"; do
    LD_LIBRARY_PATH="$rt_ld" ldd "$obj" > "$WORK/ldd.txt" 2>&1 || true
    missing=$(awk '/not found/ && $1 != "libcuda.so.1" {print $1}' "$WORK/ldd.txt" | tr '\n' ' ')
    [ -z "$missing" ] || die "$(basename "$obj") cannot resolve with the runtime library path: ${missing}"
  done
  # Compile-time NCCL identity of the headers actually used (a probe, not a binary heuristic).
  printf '#include <stdio.h>\n#include <nccl.h>\nint main(void){printf("%%d\\n", NCCL_VERSION_CODE);return 0;}\n' \
    > "$WORK/nccl_version_probe.c"
  gcc -I"$NCCL_HOME/include" -I/usr/local/cuda/include -o "$WORK/nccl_version_probe" "$WORK/nccl_version_probe.c"
  local hdr_code ep_ver
  hdr_code=$("$WORK/nccl_version_probe")
  [ "$hdr_code" = "$EP_NCCL_VERSION_CODE" ] || die "header probe says ${hdr_code}, want ${EP_NCCL_VERSION_CODE}"
  ep_ver=$(awk '/#define NCCL_EP_(MAJOR|MINOR|PATCH) / {printf "%s.", $3}' "$src/nccl_ep/include/nccl_ep.h")
  ep_ver=${ep_ver%.}
  [ "$ep_ver" = "0.2.0" ] || die "nccl_ep.h says ${ep_ver}, want 0.2.0"
  write_record "$hdr_code" "$ep_ver"
  rm -rf "$src" "$WORK/nccl_version_probe" "$WORK/nccl_version_probe.c"
}

sha() { sha256sum "$1" | cut -d' ' -f1; }

write_record() {
  local rec="$EP_BUILD/BUILD-RECORD.txt"
  {
    echo "verdict=PASS"
    echo "built_utc=$(date -u +%FT%TZ)"
    echo "base_image=${CUDA_IMAGE:-unrecorded}"
    echo "nccl_extensions_commit=${NCCL_EXTENSIONS_COMMIT}"
    echo "nccl_ep_version=$2"
    echo "nccl_wheel_sha256=${NCCL_WHEEL_SHA256}"
    echo "nccl_header_version_code=$1"
    echo "libnccl_sha256=$(sha "$NCCL_HOME/lib/libnccl.so.2")"
    echo "libnccl_ep_sha256=$(sha "$EP_BUILD/lib/libnccl_ep.so.0.2.0")"
    echo "ep_test_sha256=$(sha "$EP_BUILD/test/nccl_ep/ep_test")"
    echo "ep_bench_sha256=$(sha "$EP_BUILD/test/nccl_ep/ep_bench")"
    echo "nvcc_gencode=${NVCC_GENCODE}"
    echo "nvcc=$(nvcc --version | tail -1)"
    echo "gcc=$(gcc --version | head -1)"
    if [ -x "$MPI_HOME/bin/mpirun" ]; then echo "mpirun=$("$MPI_HOME/bin/mpirun" --version 2>&1 | head -1)"; fi
    if [ -f "$OFI_PLUGIN" ]; then
      echo "aws_ofi_nccl_plugin=${OFI_PLUGIN}"
      echo "aws_ofi_nccl_sha256=$(sha "$OFI_PLUGIN")"
    fi
    if [ -x /opt/amazon/efa/bin/fi_info ]; then echo "libfabric=$(/opt/amazon/efa/bin/fi_info --version 2>&1 | head -1)"; fi
    echo "efa_installer_version=${EFA_INSTALLER_VERSION}"
    echo "gdrcopy_commit=${GDRCOPY_COMMIT}"
  } > "$rec"
  log "BUILD PASS: ${rec}"
}

case "${1:-}" in
  deps) stage_deps ;;
  gdrcopy) stage_gdrcopy ;;
  efa) stage_efa ;;
  nccl) stage_nccl ;;
  nccl-ep) stage_nccl_ep ;;
  all) stage_deps; stage_gdrcopy; stage_efa; stage_nccl; stage_nccl_ep ;;
  *) echo "usage: $0 deps|gdrcopy|efa|nccl|nccl-ep|all" >&2; exit 2 ;;
esac
