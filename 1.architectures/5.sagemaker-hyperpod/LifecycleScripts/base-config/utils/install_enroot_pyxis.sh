#!/bin/bash

# This file assumes that install_docker.sh script was executed prior.
set -e

BIN_DIR=$(dirname $(readlink -e ${BASH_SOURCE[0]}))

# Exponential backoff function
function retry_with_backoff() {
    local max_attempts=$1
    local initial_wait=$2
    local max_wait=$3
    local command="${@:4}"
    local attempt=1
    local wait_time=$initial_wait

    while true; do
        echo "Attempt $attempt of $max_attempts: $command"
        if eval "$command"; then
            return 0
        fi

        if (( attempt == max_attempts )); then
            echo "Command failed after $max_attempts attempts: $command"
            return 1
        fi

        echo "Command failed. Retrying in $wait_time seconds..."
        sleep $wait_time

        attempt=$(( attempt + 1 ))
        wait_time=$(( wait_time * 2 ))
        if (( wait_time > max_wait )); then
            wait_time=$max_wait
        fi
    done
}

# Function for apt operations with retry
function apt_install_with_retry() {
    local package=$1
    retry_with_backoff 5 5 60 "apt-get -y -o DPkg::Lock::Timeout=120 install $package"
}

################################################################################
# Install enroot & pyxis
################################################################################
# Modify cgroup.conf to avoid runtime error due to incorrect GPU ID mapping
# https://github.com/NVIDIA/pyxis/issues/47#issuecomment-842065289
if [[ -f /opt/slurm/etc/cgroup.conf ]]; then
    grep ^ConstrainDevices /opt/slurm/etc/cgroup.conf &> /dev/null \
        || echo "ConstrainDevices=yes" >> /opt/slurm/etc/cgroup.conf
fi

# Install packages with retry
apt_install_with_retry "squashfs-tools parallel"
apt_install_with_retry "fuse-overlayfs squashfuse"

SLURM_INSTALL_DIR='/opt/slurm'
PYXIS_TMP_DIR='/tmp/pyxis'

if [ ! -d $SLURM_INSTALL_DIR ]; then
    echo "Slurm installation not found. Skipping pyxis and enroot installation.\n"
    exit 1
fi

# ------------------------------------------------------------------------------
# Detect whether the HyperPod DLAMI has already provisioned pyxis and/or enroot.
#
# Current HyperPod DLAMIs ship both pre-installed and pre-registered:
#   - pyxis SPANK plugin at /opt/slurm/lib/slurm/spank_pyxis.so, pre-registered
#     in plugstack.conf with a direct `required` line
#   - enroot as an installed apt package
# DLAMI also handles the mount-aware enroot.conf tuning for NVMe / /opt/sagemaker
# / /fsx
#
# Running `install_enroot_pyxis.sh` on top of a full DLAMI baseline produces:
#   - a duplicate SPANK registration ("srun: spank: option ... provided by both
#     spank_pyxis.so and spank_pyxis.so" on every job launch), because LCS
#     installs a second pyxis under /usr/local/lib/slurm/ and adds an
#     `include .../pyxis.conf` line on top of the DLAMI's direct `required` line;
#   - redundant apt install of the same enroot version.
#
# Detect each independently so partial-baseline DLAMIs (e.g. pyxis present but
# not enroot, or vice versa) still get the piece they are missing, and so LCS
# behavior on older Packer-built AMIs / bare Ubuntu is unchanged.
#
# What is NOT skipped, regardless of detection:
#   - `cp $BIN_DIR/enroot.conf /etc/enroot/enroot.conf` (LCS ships broader flags
#     that customer container workloads depend on: ENROOT_ROOTFS_WRITABLE=yes,
#     ENROOT_MOUNT_HOME=no, ENROOT_RESTRICT_DEV=no, `-comp lzo` squash options,
#     ENROOT_CONFIG_PATH).
#   - Mount-aware sed for /opt/dlami/nvme / /opt/sagemaker / /fsx (paired with
#     the cp above - cp brings the flags, sed rescues the paths).
#   - slurmctld / slurmd restart at the bottom of the script.
# ------------------------------------------------------------------------------
DLAMI_PYXIS_INSTALLED=0
DLAMI_ENROOT_INSTALLED=0

if [[ -f "$SLURM_INSTALL_DIR/lib/slurm/spank_pyxis.so" ]] \
   && grep -qE '^[[:space:]]*required[[:space:]]+.*spank_pyxis\.so[[:space:]]*$' \
        "$SLURM_INSTALL_DIR/etc/plugstack.conf" 2>/dev/null; then
    DLAMI_PYXIS_INSTALLED=1
    echo "[INFO] pyxis is already installed and registered by the HyperPod DLAMI"
    echo "[INFO]   binary: $SLURM_INSTALL_DIR/lib/slurm/spank_pyxis.so"
    echo "[INFO]   registration: $(grep -E 'spank_pyxis\.so' "$SLURM_INSTALL_DIR/etc/plugstack.conf" | head -1)"
    echo "[INFO] LCS will skip pyxis build/install and plugstack.conf edit to avoid a duplicate SPANK registration."
fi

if command -v enroot >/dev/null 2>&1; then
    DLAMI_ENROOT_INSTALLED=1
    echo "[INFO] enroot is already installed by the HyperPod DLAMI (version: $(enroot version 2>/dev/null | head -1))"
    echo "[INFO] LCS will skip the redundant enroot deb download/install."
    echo "[INFO] Mount-aware enroot.conf tuning still runs below, since LCS's cp overwrites the file with static paths."
fi

# Guard the destructive rm of $SLURM_INSTALL_DIR/pyxis so we don't wipe the
# DLAMI's pyxis source directory (if any); still create parent dirs (idempotent).
if [[ "$DLAMI_PYXIS_INSTALLED" -eq 0 ]]; then
    rm -fr $SLURM_INSTALL_DIR/pyxis
fi
mkdir -p $SLURM_INSTALL_DIR/enroot/ $SLURM_INSTALL_DIR/pyxis/ $PYXIS_TMP_DIR

PYXIS_VERSION=v0.19.0
ENROOT_VERSION=3.4.1
arch=$(dpkg --print-architecture)
cd $PYXIS_TMP_DIR

if [[ "$DLAMI_ENROOT_INSTALLED" -eq 0 ]]; then
    # Download enroot packages with retry
    retry_with_backoff 5 5 60 "curl -fSsL -O https://github.com/NVIDIA/enroot/releases/download/v${ENROOT_VERSION}/enroot_${ENROOT_VERSION}-1_${arch}.deb"
    retry_with_backoff 5 5 60 "curl -fSsL -O https://github.com/NVIDIA/enroot/releases/download/v${ENROOT_VERSION}/enroot+caps_${ENROOT_VERSION}-1_${arch}.deb"

    # Install enroot packages with retry
    retry_with_backoff 5 5 60 "apt install -y -o DPkg::Lock::Timeout=120 ./enroot_${ENROOT_VERSION}-1_${arch}.deb"
    retry_with_backoff 5 5 60 "apt install -y -o DPkg::Lock::Timeout=120 ./enroot+caps_${ENROOT_VERSION}-1_${arch}.deb"
fi

# Always overwrite enroot.conf: LCS ships broader flags (writable rootfs, no
# home mount, unrestricted /dev, lzo squash, ENROOT_CONFIG_PATH=$HOME/enroot)
# that customer container workloads depend on. Changing this default is a
# breaking behavior change.
cp $BIN_DIR/enroot.conf /etc/enroot/enroot.conf

if [[ "$DLAMI_PYXIS_INSTALLED" -eq 0 ]]; then
    # Clone pyxis with retry
    retry_with_backoff 5 5 60 "git clone --depth 1 --branch $PYXIS_VERSION https://github.com/NVIDIA/pyxis.git $SLURM_INSTALL_DIR/pyxis"
    cd $SLURM_INSTALL_DIR/pyxis/
    CPPFLAGS='-I /opt/slurm/include/' make -j $(nproc)
    CPPFLAGS='-I /opt/slurm/include/' make install
    mkdir -p $SLURM_INSTALL_DIR/etc/plugstack.conf.d/
    # Idempotent + wildcard-aware guard: skip the include add if pyxis is already
    # covered (either by an explicit `include .../pyxis.conf` or a wildcard
    # `include .../plugstack.conf.d/*`), so LCS re-runs and older Packer-built
    # AMIs that use the wildcard form don't produce a duplicate include.
    if ! grep -qE "^[[:space:]]*include[[:space:]]+.*plugstack\.conf\.d/(pyxis\.conf|\*)[[:space:]]*$" \
            "$SLURM_INSTALL_DIR/etc/plugstack.conf" 2>/dev/null; then
        echo -e "include $SLURM_INSTALL_DIR/etc/plugstack.conf.d/pyxis.conf" >> $SLURM_INSTALL_DIR/etc/plugstack.conf
    fi
    ln -fs /usr/local/share/pyxis/pyxis.conf $SLURM_INSTALL_DIR/etc/plugstack.conf.d/pyxis.conf
fi

mkdir -p /run/pyxis/ /tmp/enroot/data /opt/enroot/
chmod 777 -R /tmp/enroot /opt/enroot

################################################################################
# Mount-aware enroot.conf tuning.
#
# ALWAYS runs, regardless of DLAMI detection. The `cp $BIN_DIR/enroot.conf ...`
# step above unconditionally overwrites /etc/enroot/enroot.conf with LCS's
# broad flags but *static* paths (ENROOT_CACHE_PATH=/opt/enroot on the
# root volume, ENROOT_RUNTIME_PATH=/tmp/..., etc.). Without this sed pass
# immediately after, ENROOT_CACHE_PATH stays on the root volume leading to
# the first large container pull to fill the root volume and the node degrades.
#  See: https://github.com/awslabs/awsome-distributed-training/issues/427
#
# The `cp` and this sed are a *pair* by design: `cp` brings LCS's broadened
# flags (writable rootfs, no home mount, lzo squash, etc.), this sed rescues
# the paths from the `cp`'s static defaults. Do not separate them.
################################################################################
# Below while loop instituted to combat race condition when mapping enroot path to /opt/dlami/nvme
MAX_WAIT_TIME=120
ELAPSED_TIME=0
CHECK_INTERVAL=5

while true; do
    # Check the ActiveState of the lib/systemd/system/dlami-nvme.service
    ACTIVE_STATE=$(systemctl show dlami-nvme | grep "ActiveState" | cut -d '=' -f 2)
    # Check the ExecMainStatus of the lib/systemd/system/dlami-nvme.service
    RESULT_STATE=$(systemctl show dlami-nvme | grep "ExecMainStatus" | cut -d '=' -f 2)

    echo "dlami-nvme.service ActiveState: $ACTIVE_STATE"
    echo "dlami-nvme.service ExecMainStatus: $RESULT_STATE"

    if [[ "$ACTIVE_STATE" == "active" && "$RESULT_STATE" == "0" ]]; then
        echo "dlami-nvme.service is active and successful. Proceeding with Enroot configuration on /opt/dlami/nvme if available"
        break
    fi

    ELAPSED_TIME=$((ELAPSED_TIME + CHECK_INTERVAL))

    if [[ $ELAPSED_TIME -ge $MAX_WAIT_TIME ]]; then
        echo "WARN: Timeout reached: dlami-nvme.service did not become active and successful, it is possible enroot default path is /opt/sagemaker. When training larger models, dragons be here. See https://github.com/awslabs/awsome-distributed-training/issues/427 for corrective actions"
        break
    fi

    sleep $CHECK_INTERVAL
done

####################################################################################################

# Configure enroot paths based on available mounts
if [[ $(mount | grep /opt/dlami/nvme) ]]; then
    sed -i \
        -e 's|^\(ENROOT_RUNTIME_PATH  *\).*$|\1/opt/dlami/nvme/tmp/enroot/user-$(id -u)|' \
        -e 's|^\(ENROOT_CACHE_PATH  *\).*$|\1/opt/dlami/nvme/enroot|' \
        -e 's|^\(ENROOT_DATA_PATH  *\).*$|\1/opt/dlami/nvme/tmp/enroot/data/user-$(id -u)|' \
        -e 's|^#\(ENROOT_TEMP_PATH  *\).*$|\1/opt/dlami/nvme/tmp|' \
        /etc/enroot/enroot.conf

    mkdir -p /opt/dlami/nvme/tmp/enroot/
    chmod 1777 /opt/dlami/nvme/tmp
    chmod 1777 /opt/dlami/nvme/tmp/enroot/

    mkdir -p /opt/dlami/nvme/tmp/enroot/data/
    chmod 1777 /opt/dlami/nvme/tmp/enroot/data/

    mkdir -p /opt/dlami/nvme/enroot
    chmod 1777 /opt/dlami/nvme/enroot

elif [[ $(mount | grep /opt/sagemaker) ]]; then
    sed -i \
        -e 's|^\(ENROOT_RUNTIME_PATH  *\).*$|\1/opt/sagemaker/tmp/enroot/user-$(id -u)|' \
        -e 's|^\(ENROOT_CACHE_PATH  *\).*$|\1/opt/sagemaker/enroot|' \
        -e 's|^\(ENROOT_DATA_PATH  *\).*$|\1/opt/sagemaker/tmp/enroot/data/user-$(id -u)|' \
        -e 's|^#\(ENROOT_TEMP_PATH  *\).*$|\1/opt/sagemaker/tmp|' \
        /etc/enroot/enroot.conf

    mkdir -p /opt/sagemaker/tmp/enroot/
    chmod 1777 /opt/sagemaker/tmp
    chmod 1777 /opt/sagemaker/tmp/enroot/

    mkdir -p /opt/sagemaker/tmp/enroot/data/
    chmod 1777 /opt/sagemaker/tmp/enroot/data/

    mkdir -p /opt/sagemaker/enroot
    chmod 1777 /opt/sagemaker/enroot
fi

# Configure FSX for enroot cache if available
if [[ $(mount | grep /fsx) ]]; then
    sed -i -e 's|^\(ENROOT_CACHE_PATH  *\).*$|\1/fsx/enroot|' /etc/enroot/enroot.conf
    mkdir -p /fsx/enroot
    chmod 1777 /fsx/enroot
fi

# Restart Slurm services if they're running
retry_with_backoff 5 5 60 "systemctl is-active --quiet slurmctld && systemctl restart slurmctld || echo 'This instance does not run slurmctld'"
retry_with_backoff 5 5 60 "systemctl is-active --quiet slurmd && systemctl restart slurmd || echo 'This instance does not run slurmd'"
