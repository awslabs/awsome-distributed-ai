#!/bin/bash

set -exo pipefail

# If docker is alrady installed, skip installing again
if command -v docker &> /dev/null; then
    echo "Docker is already installed and in the PATH."
    exit 0
fi

echo "
###################################
# BEGIN: install docker
###################################
"

apt-get -y -o DPkg::Lock::Timeout=120 update
apt-get -y -o DPkg::Lock::Timeout=120 install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
mkdir -m 0755 -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
"deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
$(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
apt-get -y -o DPkg::Lock::Timeout=120 update
apt-get -y -o DPkg::Lock::Timeout=120 install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
groupadd -f docker
chgrp docker $(which docker)
chmod g+s $(which docker)
systemctl enable docker.service
systemctl start docker.service

# Detect pre-installed nvidia-container-toolkit-base version or use default
if dpkg -s nvidia-container-toolkit-base &> /dev/null; then
    DETECTED_VERSION=$(dpkg -l nvidia-container-toolkit-base | awk '/^ii/ {print $3}')
    echo "Detected pre-installed nvidia-container-toolkit-base version: ${DETECTED_VERSION}"
    export NVIDIA_CONTAINER_TLK_VERSION="${DETECTED_VERSION}"
else
    # Fallback to pinned version if not pre-installed (due to known issue https://github.com/NVIDIA/nvidia-container-toolkit/issues/1093)
    export NVIDIA_CONTAINER_TLK_VERSION="1.18.1-1"
    echo "No pre-installed nvidia-container-toolkit-base found, using default version: ${NVIDIA_CONTAINER_TLK_VERSION}"
fi

# Install nvidia container toolkit with all dependencies at consistent version
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt-get install -y --allow-downgrades -o DPkg::Lock::Timeout=120 nvidia-container-toolkit=${NVIDIA_CONTAINER_TLK_VERSION} nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TLK_VERSION} libnvidia-container-tools=${NVIDIA_CONTAINER_TLK_VERSION} libnvidia-container1=${NVIDIA_CONTAINER_TLK_VERSION}
# Lock nvidia-container-toolkit version
sudo apt-mark hold nvidia-container-toolkit nvidia-container-toolkit-base libnvidia-container-tools libnvidia-container1

# Print NV_COTNAINER_TLK_VERSIONS to logs 
echo "Expected NV_TLK_VERSION: ${NVIDIA_CONTAINER_TLK_VERSION}"
echo "Installed NV_TLK_VERSION: $(dpkg -l nvidia-container-toolkit | awk '/nvidia-container-toolkit/ {print $3}')"

# add user to docker group
sudo usermod -aG docker ubuntu


# Opportunistically use /opt/sagemaker or /opt/dlami/nvme if present. Let's be extra careful in the probe.
# See: https://github.com/awslabs/awsome-distributed-ai/issues/127
#
# Docker workdir doesn't like Lustre. Tried with storage driver overlay2, fuse-overlayfs, & vfs.
if [[ $(mount | grep /opt/sagemaker) ]]; then
    cat <<EOL >> /etc/docker/daemon.json
{
    "data-root": "/opt/sagemaker/docker/data-root"
}
EOL

    sed -i \
        's|^\[Service\]$|[Service]\nEnvironment="DOCKER_TMPDIR=/opt/sagemaker/docker/tmp"|' \
        /usr/lib/systemd/system/docker.service

    sudo sed -i -e 's|^#\?root *=.*|root = "/opt/sagemaker/containerd/data-root"|' \
        /etc/containerd/config.toml
    sudo systemctl restart containerd
elif [[ $(mount | grep /opt/dlami/nvme) ]]; then
    cat <<EOL >> /etc/docker/daemon.json
{
    "data-root": "/opt/dlami/nvme/docker/data-root"
}
EOL

    sed -i \
        's|^\[Service\]$|[Service]\nEnvironment="DOCKER_TMPDIR=/opt/dlami/nvme/docker/tmp"|' \
        /usr/lib/systemd/system/docker.service

    sudo sed -i -e 's|^#\?root *=.*|root = "/opt/dlami/nvme/containerd/data-root"|' \
        /etc/containerd/config.toml
    sudo systemctl restart containerd
fi

systemctl daemon-reload
systemctl restart docker
