#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Install Enroot and Pyxis on AWS PCS nodes
# Usage: bash install-enroot-pyxis.sh
#
# DEPENDENCIES / PREREQUISITES (must hold at runtime, or install fails):
#   1. PCS Slurm pre-installed: Pyxis is compiled against the Slurm headers at
#      /opt/aws/pcs/scheduler/slurm-${VERSION}/include/. Versions in
#      SLURM_VERSIONS whose directory is absent are SKIPPED (warning, not error),
#      leaving no plugstack config for that version. Use a PCS DLAMI base AMI.
#   2. Debian/Ubuntu family only: uses apt-get/dpkg (no yum/dnf). Tested on
#      Ubuntu 24.04 (PCS DLAMI Base).
#   3. Outbound network egress required (private subnets need a NAT path; the S3
#      VPC endpoint alone is NOT sufficient):
#        - github.com                 (Enroot .deb releases, Pyxis source)
#        - raw.githubusercontent.com  (aws-samples enroot.template.conf)
#        - nvidia.github.io           (libnvidia-container repo, GPU nodes only)
#        - Ubuntu apt mirrors
#   4. Runs as root and may contend with cloud-init / unattended-upgrades for the
#      apt lock at first boot. The script now waits for the dpkg/apt lock and
#      retries apt (wait_for_apt_lock/apt_get), so the boot-time lock race no
#      longer aborts the install.
#   5. GPU toolkit (nvidia-container-toolkit) installs ONLY when nvidia-smi
#      succeeds; CPU-only nodes intentionally skip it.
#
# EXECUTION CONTEXT:
#   Designed to run as a first-boot hook (PCS PostInstallScriptUrl) or during a
#   custom-AMI build -- i.e. BEFORE slurmd starts. It does NOT restart slurmd, so
#   the slurmd PATH it writes only takes effect on the next slurmd start (the
#   first boot). If you run it on an already-running node, restart slurmd yourself.
#
# IDEMPOTENT:
#   Safe to run more than once (e.g. PostInstallScriptUrl left at the default on a
#   BuildAMI=true node where Enroot/Pyxis is already baked in). Each component is
#   skipped when already present at the expected version; a version mismatch is
#   overwritten. So a redundant re-run is a fast no-op rather than a double install.

set -exo pipefail

echo "Starting Enroot/Pyxis installation for AWS PCS..."

ENROOT_RELEASE=3.5.0
PYXIS_RELEASE=v0.20.0
SLURM_VERSIONS="25.05 25.11"

# At first boot this script runs alongside cloud-init and Ubuntu's
# `unattended-upgrades`, which hold the dpkg/apt locks. Without waiting, the very
# first `apt-get` fails with "Could not get lock /var/lib/dpkg/lock-frontend ...
# held by ... (unattended-upgr)" and the whole script aborts under `set -e`
# (observed as post-install exit 100 on a cold boot). Wait for the locks to clear,
# and route apt through a small retry wrapper so a transient lock never fails the run.
wait_for_apt_lock() {
  local max=300 i=0
  while fuser /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock /var/lib/apt/lists/lock \
        /var/cache/apt/archives/lock >/dev/null 2>&1; do
    i=$((i+1))
    [ "$i" -ge "$max" ] && { echo "WARN: apt/dpkg lock still held after ${max}s; proceeding"; break; }
    echo "Waiting for apt/dpkg lock to be released (${i}s)..."
    sleep 1
  done
}

apt_get() {
  # Wait for the lock, then retry a few times to ride out unattended-upgrades.
  local tries=0
  wait_for_apt_lock
  until DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=300 "$@"; do
    tries=$((tries+1))
    [ "$tries" -ge 5 ] && { echo "ERROR: apt-get $* failed after $tries attempts"; return 1; }
    echo "apt-get $* failed (attempt $tries); waiting for lock and retrying..."
    wait_for_apt_lock
    sleep 5
  done
}

# Install dependencies
echo "Installing dependencies..."
apt_get update
apt_get install -y jq squashfs-tools parallel fuse-overlayfs pigz squashfuse zstd git build-essential

# Install nvidia-container-toolkit if GPU is detected
if nvidia-smi 2>/dev/null; then
  echo "GPU detected, installing nvidia-container-toolkit..."
  # gpg must not try to open a controlling terminal: post-install/cloud-init runs
  # with no tty, and `gpg --dearmor` would otherwise fail with
  # "gpg: cannot open '/dev/tty'", aborting the whole script under `set -e`.
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --batch --no-tty --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  # Use the version-agnostic 'stable/deb' repo path. The per-distribution paths
  # (e.g. ubuntu24.04) do not all exist and return an HTML 404 that, without
  # `curl -f`, would get written verbatim into the apt source list and break
  # `apt-get update`.
  curl -fsSL "https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list" | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt_get update -y
  apt_get install -y libnvidia-container-tools
fi

# Install Enroot (skip if the target version is already present, e.g. baked into
# a custom AMI; reinstall only on a version mismatch).
if [ "$(enroot version 2>/dev/null)" = "${ENROOT_RELEASE}" ]; then
  echo "Enroot ${ENROOT_RELEASE} already installed, skipping."
else
  echo "Installing Enroot ${ENROOT_RELEASE}..."
  arch=$(dpkg --print-architecture)
  mkdir -p /tmp/enroot
  cd /tmp/enroot
  curl -fSsL -O "https://github.com/NVIDIA/enroot/releases/download/v${ENROOT_RELEASE}/enroot_${ENROOT_RELEASE}-1_${arch}.deb"
  curl -fSsL -O "https://github.com/NVIDIA/enroot/releases/download/v${ENROOT_RELEASE}/enroot+caps_${ENROOT_RELEASE}-1_${arch}.deb"
  apt_get install -y ./*.deb
fi

# Configure Enroot
ln -sf /usr/share/enroot/hooks.d/50-slurm-pmi.sh /etc/enroot/hooks.d/
ln -sf /usr/share/enroot/hooks.d/50-slurm-pytorch.sh /etc/enroot/hooks.d/

mkdir -p /tmp/enroot /tmp/enroot/data /tmp/enroot/cache
chmod 1777 /tmp/enroot /tmp/enroot/data /tmp/enroot/cache

wget -O /tmp/enroot.template.conf https://raw.githubusercontent.com/aws-samples/aws-parallelcluster-post-install-scripts/main/pyxis/enroot.template.conf
# Make the cache per-user, like the data/runtime paths. The template already keys
# ENROOT_DATA_PATH / ENROOT_RUNTIME_PATH on $(id -u); a SHARED cache instead writes
# layer blobs as 0640 owned by the first importer, so a second user importing any
# image that shares a layer hits an unreadable blob and the job dies with an opaque
# "tar: Error is not recoverable". Single-quote the value so $(id -u) stays literal
# through envsubst (it only expands ${VAR} forms) and enroot evaluates it per-user
# at runtime. The sticky 1777 parent above lets each user create their own subdir.
ENROOT_CACHE_PATH='/tmp/enroot/cache/user-$(id -u)' envsubst < /tmp/enroot.template.conf > /etc/enroot/enroot.conf
chmod 0644 /etc/enroot/enroot.conf

# Install Pyxis for all Slurm versions
echo "Installing Pyxis ${PYXIS_RELEASE}..."
for SLURM_VERSION in ${SLURM_VERSIONS}; do
  echo "Installing Pyxis for Slurm ${SLURM_VERSION}..."
  SLURM_PATH="/opt/aws/pcs/scheduler/slurm-${SLURM_VERSION}"
  SLURM_ETC_PATH="/etc/aws/pcs/scheduler/slurm-${SLURM_VERSION}"

  if [ ! -d "${SLURM_PATH}" ]; then
    echo "Warning: Slurm ${SLURM_VERSION} not found at ${SLURM_PATH}, skipping..."
    continue
  fi

  # Idempotent: skip this Slurm version if Pyxis is already wired up for it (the
  # SPANK plugin built and the plugstack config symlinked). Catches the AMI-baked
  # / re-run case so we don't rebuild from source every boot.
  if [ -f /usr/local/lib/slurm/spank_pyxis.so ] && \
     [ -e "${SLURM_ETC_PATH}/plugstack.conf.d/pyxis.conf" ]; then
    echo "Pyxis already installed for Slurm ${SLURM_VERSION}, skipping."
    continue
  fi

  if [ ! -d /tmp/pyxis ]; then
    git clone --depth 1 --branch "${PYXIS_RELEASE}" https://github.com/NVIDIA/pyxis.git /tmp/pyxis
  fi

  cd /tmp/pyxis
  make clean || true
  CPPFLAGS="-I ${SLURM_PATH}/include/" make
  CPPFLAGS="-I ${SLURM_PATH}/include/" make install

  mkdir -p "${SLURM_ETC_PATH}/plugstack.conf.d"
  ln -sf /usr/local/share/pyxis/pyxis.conf "${SLURM_ETC_PATH}/plugstack.conf.d/pyxis.conf"

  echo "Pyxis installed for Slurm ${SLURM_VERSION}"
done

# Update PATH for slurmd. Derive the Slurm bin dir from what is actually present
# on this node instead of hardcoding a version -- a hardcoded slurm-25.11 silently
# mismatches clusters deployed with SlurmVersion=25.05. Include every installed
# slurm-*/bin (a single-version cluster yields exactly one). Guard the append with
# grep -q so re-running this script does not add a duplicate PATH line.
SLURM_BIN_DIRS=$(printf '%s:' /opt/aws/pcs/scheduler/slurm-*/bin 2>/dev/null)
SLURMD_PATH_LINE="PATH=${SLURM_BIN_DIRS}/usr/lib/ccache/bin:/usr/local/bin:/usr/bin:/bin"
if ! grep -qxF "${SLURMD_PATH_LINE}" /etc/default/slurmd 2>/dev/null; then
  echo "${SLURMD_PATH_LINE}" >> /etc/default/slurmd
fi

# Load GPU kernel modules if GPU detected
if nvidia-smi 2>/dev/null; then
  echo "Loading GPU kernel modules..."
  nvidia-container-cli --load-kmods info || true
fi

echo "Enroot/Pyxis installation complete!"
echo "Installed at: $(date)"
echo ""
echo "Verification:"
enroot version
ls -la /etc/aws/pcs/scheduler/slurm-*/plugstack.conf.d/ 2>/dev/null || echo "Plugstack config check skipped"
