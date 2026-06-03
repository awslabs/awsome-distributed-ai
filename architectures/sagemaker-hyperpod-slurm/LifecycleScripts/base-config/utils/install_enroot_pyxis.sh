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

rm -fr $SLURM_INSTALL_DIR/pyxis
mkdir -p $SLURM_INSTALL_DIR/enroot/ $SLURM_INSTALL_DIR/pyxis/ $PYXIS_TMP_DIR

PYXIS_VERSION=v0.19.0
ENROOT_VERSION=3.4.1
arch=$(dpkg --print-architecture)
cd $PYXIS_TMP_DIR

# Download enroot packages with retry
retry_with_backoff 5 5 60 "curl -fSsL -O https://github.com/NVIDIA/enroot/releases/download/v${ENROOT_VERSION}/enroot_${ENROOT_VERSION}-1_${arch}.deb"
retry_with_backoff 5 5 60 "curl -fSsL -O https://github.com/NVIDIA/enroot/releases/download/v${ENROOT_VERSION}/enroot+caps_${ENROOT_VERSION}-1_${arch}.deb"

# Install enroot packages with retry
retry_with_backoff 5 5 60 "apt install -y -o DPkg::Lock::Timeout=120 ./enroot_${ENROOT_VERSION}-1_${arch}.deb"
retry_with_backoff 5 5 60 "apt install -y -o DPkg::Lock::Timeout=120 ./enroot+caps_${ENROOT_VERSION}-1_${arch}.deb"
cp $BIN_DIR/enroot.conf /etc/enroot/enroot.conf

# Clone pyxis with retry
retry_with_backoff 5 5 60 "git clone --depth 1 --branch $PYXIS_VERSION https://github.com/NVIDIA/pyxis.git $SLURM_INSTALL_DIR/pyxis"
cd $SLURM_INSTALL_DIR/pyxis/
CPPFLAGS='-I /opt/slurm/include/' make -j $(nproc)
CPPFLAGS='-I /opt/slurm/include/' make install
mkdir -p $SLURM_INSTALL_DIR/etc/plugstack.conf.d/
# Check if the line exists before adding it to handle multi headnode conf
if ! grep -q "include $SLURM_INSTALL_DIR/etc/plugstack.conf.d/pyxis.conf" "$SLURM_INSTALL_DIR/etc/plugstack.conf"; then
    echo -e "include $SLURM_INSTALL_DIR/etc/plugstack.conf.d/pyxis.conf" >> $SLURM_INSTALL_DIR/etc/plugstack.conf
fi
ln -fs /usr/local/share/pyxis/pyxis.conf $SLURM_INSTALL_DIR/etc/plugstack.conf.d/pyxis.conf

mkdir -p /run/pyxis/ /tmp/enroot/data /opt/enroot/
chmod 777 -R /tmp/enroot /opt/enroot

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
        echo "WARN: Timeout reached: dlami-nvme.service did not become active and successful, it is possible enroot default path is /opt/sagemaker. When training larger models, dragons be here. See https://github.com/awslabs/awsome-distributed-ai/issues/427 for corrective actions"
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
