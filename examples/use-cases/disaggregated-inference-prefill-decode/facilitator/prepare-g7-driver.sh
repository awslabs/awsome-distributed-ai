#!/usr/bin/env bash
# AL2023 repair exercised on the Spain G7 nodes; run before GPU workloads.
set -euo pipefail
instance_type=$(cat /sys/devices/virtual/dmi/id/product_name)
case "$instance_type" in g7.*) ;; *) exit 0 ;; esac
[[ $(id -u) == 0 ]] || { echo 'Run this host preparation helper as root' >&2; exit 2; }
source /etc/os-release
[[ "$ID" == amzn && "$VERSION_ID" == 2023 ]] || { echo 'This repair is for the observed AL2023 EKS image' >&2; exit 2; }
version=595.91.07
if [[ $(modinfo -F version nvidia) == "$version" && $(modinfo -F license nvidia) == 'Dual MIT/GPL' ]] && nvidia-smi -L; then
    printf 'Detected %s with the prepared open driver version %s\n' "$instance_type" "$version"
    exit 0
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '^[0-9]'; then
    echo 'Stop the assigned GPU workloads before host driver preparation' >&2; exit 2
fi
install -d /var/log/aim345-driver
if [[ -f /etc/dkms/nvidia.conf && ! -e /var/log/aim345-driver/nvidia.conf.before ]]; then
    cp -a /etc/dkms/nvidia.conf /var/log/aim345-driver/nvidia.conf.before
fi
cat > /etc/dkms/nvidia-595.91.07.conf <<'EOF'
MAKE[0]="'make' -j16 KERNEL_UNAME=${kernelver} IGNORE_PREEMPT_RT_PRESENCE=1 IGNORE_XEN_PRESENCE=1 modules"
EOF
dnf install -y --allowerasing "nvidia-open-$version"
dkms build -m nvidia -v "$version" -k "$(uname -r)" -j 16
dkms install -m nvidia -v "$version" -k "$(uname -r)"
modprobe nvidia
modprobe nvidia_uvm
install -d /etc/systemd/system/nvidia-kmod-load.service.d
cat > /etc/systemd/system/nvidia-kmod-load.service.d/50-aim345-g7-open.conf <<'EOF'
[Service]
ExecStart=
ExecStart=/sbin/modprobe nvidia
ExecStart=/sbin/modprobe nvidia_uvm
EOF
systemctl daemon-reload
systemctl restart nvidia-persistenced
[[ $(modinfo -F version nvidia) == "$version" && $(modinfo -F license nvidia) == 'Dual MIT/GPL' ]]
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version,compute_cap --format=csv
