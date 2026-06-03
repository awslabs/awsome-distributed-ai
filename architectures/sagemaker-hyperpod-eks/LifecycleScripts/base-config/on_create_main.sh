#!/bin/bash

set -ex

# Configuration: Choose disk for containerd and kubelet
# Options: "/opt/sagemaker" or "/opt/dlami/nvme"

DISK_FOR_CONTAINERD_KUBELET="/opt/sagemaker"
#DISK_FOR_CONTAINERD_KUBELET="/opt/dlami/nvme"

logger() {
  echo "$@"
}

logger "[start] on_create_main.sh"
logger "Disk for containerd and kubelet: $DISK_FOR_CONTAINERD_KUBELET"

# Wait for disk to be mounted (max 60s)
for i in {1..12}; do
  if mount | grep -q "$DISK_FOR_CONTAINERD_KUBELET"; then
    logger "$DISK_FOR_CONTAINERD_KUBELET is mounted"
    break
  else
    logger "Waiting for $DISK_FOR_CONTAINERD_KUBELET to be mounted..."
    sleep 5
  fi
done


# Dump the status of containerd.service and kubelet.service
logger "Dumping the status of containerd.service and kubelet.service"
systemctl status containerd.service --no-pager || true
systemctl status kubelet.service --no-pager || true


if mount | grep -q "$DISK_FOR_CONTAINERD_KUBELET"; then
  logger "Setting containerd data root to $DISK_FOR_CONTAINERD_KUBELET/containerd/data-root"

  # Detect OS version reliably
  source /etc/os-release
  os_version="$VERSION_ID"
  logger "Detected OS version: $os_version"

  if [[ "$os_version" == "2" ]]; then
    # Amazon Linux 2 logic
    CONFIG_FILE="/etc/eks/containerd/containerd-config.toml"
    if [[ -f "$CONFIG_FILE" ]]; then
      logger "Amazon Linux 2 detected. Modifying $CONFIG_FILE using sed"
      sed -i -e "/^[# ]*root\s*=/c\root = \"$DISK_FOR_CONTAINERD_KUBELET/containerd/data-root\"" "$CONFIG_FILE"
    else
      logger "Amazon Linux 2 detected, but $CONFIG_FILE not found!"
    fi

  elif [[ "$os_version" == "2023" ]]; then
    # Amazon Linux 2023 logic (systemd override with custom config)
    logger "Amazon Linux 2023 detected. Creating custom containerd config and systemd override"

    # Clean up old containerd data to avoid AL2->AL23 compatibility issues
    if [[ -d "$DISK_FOR_CONTAINERD_KUBELET/containerd/data-root" ]]; then
      logger "Removing existing containerd data-root to prevent AL2/AL23 incompatibility"
      rm -rf "$DISK_FOR_CONTAINERD_KUBELET/containerd/data-root"
    fi

    # Create custom containerd config directory
    mkdir -p "$DISK_FOR_CONTAINERD_KUBELET/containerd"

    # Create complete custom containerd config
    cat <<EOF | tee "$DISK_FOR_CONTAINERD_KUBELET/containerd/config.toml"
version = 2
root = "$DISK_FOR_CONTAINERD_KUBELET/containerd/data-root"
state = "/run/containerd"

[grpc]
address = "/run/containerd/containerd.sock"

[plugins."io.containerd.grpc.v1.cri".containerd]
default_runtime_name = "nvidia"
discard_unpacked_layers = true

[plugins."io.containerd.grpc.v1.cri"]
sandbox_image = "localhost/kubernetes/pause"
enable_cdi = true

[plugins."io.containerd.grpc.v1.cri".registry]
config_path = "/etc/containerd/certs.d:/etc/docker/certs.d"

[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia]
runtime_type = "io.containerd.runc.v2"
base_runtime_spec = "/etc/containerd/base-runtime-spec.json"

[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia.options]
BinaryName = "/usr/bin/nvidia-container-runtime"
SystemdCgroup = true

[plugins."io.containerd.grpc.v1.cri".cni]
bin_dir = "/opt/cni/bin"
conf_dir = "/etc/cni/net.d"
EOF

    # Create systemd override
    mkdir -p /etc/systemd/system/containerd.service.d

    cat <<EOF | tee /etc/systemd/system/containerd.service.d/override.conf
[Service]
Environment="CONTAINERD_CONFIG=$DISK_FOR_CONTAINERD_KUBELET/containerd/config.toml"
ExecStart=
ExecStart=/usr/bin/containerd --config \$CONTAINERD_CONFIG
EOF

    systemctl daemon-reload

    cp -a /var/lib/containerd "$DISK_FOR_CONTAINERD_KUBELET/containerd/data-root"

  else
    logger "Unsupported OS version: $os_version. Skipping containerd configuration."
  fi

  logger "Creating symbolic link from /var/lib/kubelet to $DISK_FOR_CONTAINERD_KUBELET/kubelet"

  # Clean up old kubelet data to avoid AL2->AL23 compatibility issues
  if [[ -d "$DISK_FOR_CONTAINERD_KUBELET/kubelet" ]]; then
    logger "Removing existing kubelet directory to prevent AL2/AL23 incompatibility"
    rm -rf "$DISK_FOR_CONTAINERD_KUBELET/kubelet"
  fi

  mkdir -p "$DISK_FOR_CONTAINERD_KUBELET/kubelet"
  if [ "$(ls -A /var/lib/kubelet 2>/dev/null)" ]; then
    mv /var/lib/kubelet/* "$DISK_FOR_CONTAINERD_KUBELET/kubelet/"
  else
    logger "/var/lib/kubelet is empty, skipping file move"
  fi
  rmdir /var/lib/kubelet
  ln -s "$DISK_FOR_CONTAINERD_KUBELET/kubelet" /var/lib/

else
  logger "$DISK_FOR_CONTAINERD_KUBELET not mounted. Skipping containerd configuration"
fi


# ===== EFA FSx LUSTRE CLIENT SETUP =====
# NOTE: This function runs in a subshell to prevent cd from leaking into the caller's working directory.
# The EXIT trap ensures cleanup on all exit paths (success, error, or signal).

setup_efa_fsx_client() (
    logger "[INFO] Starting EFA FSx client setup"

    # Step 1: OS compatibility check
    source /etc/os-release 2>/dev/null || { logger "[INFO] Cannot detect OS, skipping"; return 0; }

    case "$ID-$VERSION_ID" in
        "amzn-2023")
            logger "[INFO] Amazon Linux 2023 - supported" ;;
        "rhel-9."[5-9]* | "rhel-1"[0-9]*)
            logger "[INFO] RHEL $VERSION_ID - supported" ;;
        "ubuntu-22.04" | "ubuntu-2"[3-9]*)
            # Proper kernel version check for Ubuntu
            local kernel_major=$(uname -r | cut -d'.' -f1)
            local kernel_minor=$(uname -r | cut -d'.' -f2)
            if [[ "$kernel_major" -gt 6 ]] || [[ "$kernel_major" -eq 6 && "$kernel_minor" -ge 8 ]]; then
                logger "[INFO] Ubuntu $VERSION_ID kernel ${kernel_major}.${kernel_minor} - supported"
            else
                logger "[INFO] Ubuntu needs kernel 6.8+, found ${kernel_major}.${kernel_minor}, skipping"
                return 0
            fi ;;
        *)
            logger "[INFO] OS $ID $VERSION_ID not supported, skipping"
            return 0 ;;
    esac

    # Step 2: EFA availability check
    if [[ ! -x "/opt/amazon/efa/bin/fi_info" ]]; then
        logger "[INFO] EFA tools not found, skipping"
        return 0
    fi

    if ! /opt/amazon/efa/bin/fi_info -p efa >/dev/null 2>&1; then
        logger "[INFO] EFA not available on this instance, skipping"
        return 0
    fi

    logger "[INFO] EFA detected - configuring for FSx Lustre"

    # Step 3: Download and setup in isolated temp directory
    WORKDIR=$(mktemp -d /tmp/efa-fsx-setup.XXXXXX) || { logger "[ERROR] Cannot create temp directory"; return 1; }
    trap 'rm -rf "$WORKDIR"' EXIT
    cd "$WORKDIR" || {
        logger "[ERROR] Cannot enter temp directory"
        return 1
    }

    logger "[INFO] Downloading EFA FSx client setup..."
    if ! curl --fail --silent --show-error --max-time 30 -o efa-setup.zip \
         "https://docs.aws.amazon.com/fsx/latest/LustreGuide/samples/configure-efa-fsx-lustre-client.zip"; then
        logger "[ERROR] Download failed"
        return 1
    fi

    logger "[INFO] Extracting setup files..."
    if ! unzip -q efa-setup.zip; then
        logger "[ERROR] Extract failed"
        return 1
    fi

    if [[ ! -f "configure-efa-fsx-lustre-client/setup.sh" ]]; then
        logger "[ERROR] Setup script not found in package"
        return 1
    fi

    chmod +x configure-efa-fsx-lustre-client/setup.sh

    logger "[INFO] Running EFA FSx client setup..."
    if ./configure-efa-fsx-lustre-client/setup.sh; then
        logger "[SUCCESS] EFA FSx client configured successfully"
    else
        logger "[ERROR] EFA FSx client setup failed"
        return 1
    fi

    # Cleanup handled by EXIT trap in Step 3
    return 0
)

# Load Lustre modules
load_lustre_modules() {
    logger "[INFO] Loading Lustre kernel modules"

    # Load lnet module
    if modprobe lnet 2>/dev/null; then
        logger "[INFO] lnet module loaded"
    else
        logger "[WARN] lnet module load failed or already loaded"
    fi

    # Load lustre module
    if modprobe lustre 2>/dev/null; then
        logger "[INFO] lustre module loaded"
    else
        logger "[WARN] lustre module load failed or already loaded"
    fi

    # Initialize LNet network
    if command -v lctl >/dev/null 2>&1; then
        if lctl network up 2>/dev/null; then
            logger "[INFO] LNet network initialized"
        else
            logger "[INFO] LNet network already active or initialization attempted"
        fi
    fi
}

# Execute EFA FSx client setup
if setup_efa_fsx_client; then
    logger "[INFO] EFA FSx client setup completed successfully"
else
    logger "[INFO] EFA FSx client setup skipped or failed - continuing with standard Lustre"
fi

# Load Lustre modules (always execute)
load_lustre_modules

logger "[INFO] FSx client setup complete"

logger "[stop] on_create_main.sh"
