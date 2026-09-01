#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Install node_exporter on a ParallelCluster compute node.
# Use as OnNodeConfigured script so that the Prometheus Agent Collector
# (prometheus-agent-collector.yaml) can scrape OS metrics from port 9100.
#
# Usage (ParallelCluster config.yaml):
#   Scheduling:
#     SlurmQueues:
#       - CustomActions:
#           OnNodeConfigured:
#             Sequence:
#               - Script: s3://<bucket>/install-node-exporter.sh
#               # Or use GitHub raw URL directly (no S3 upload required):
#               # - Script: https://raw.githubusercontent.com/awslabs/awsome-distributed-ai/main/architectures/aws-parallelcluster/post-install-scripts/install-node-exporter.sh
#
# Environment variables:
#   NODE_EXPORTER_VERSION: Version to install (default: 1.9.1)
#   NODE_EXPORTER_DOWNLOAD_URL: Base URL for downloads (default: GitHub releases)

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "[ERROR] This script must be run as root"
    exit 1
fi

NODE_EXPORTER_VERSION="${NODE_EXPORTER_VERSION:-1.9.1}"
INSTALL_DIR="/usr/local/bin"
SERVICE_FILE="/etc/systemd/system/node_exporter.service"

# Skip if already running
if systemctl is-active --quiet node_exporter; then
    echo "[INFO] node_exporter is already running. Skipping installation."
    exit 0
fi

echo "[INFO] Installing node_exporter v${NODE_EXPORTER_VERSION}"

ARCH=$(uname -m)
case "$ARCH" in
    x86_64)  ARCH_LABEL="amd64" ;;
    aarch64) ARCH_LABEL="arm64" ;;
    *)
        echo "[ERROR] Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

TARBALL="node_exporter-${NODE_EXPORTER_VERSION}.linux-${ARCH_LABEL}.tar.gz"
DOWNLOAD_BASE_URL="${NODE_EXPORTER_DOWNLOAD_URL:-https://github.com/prometheus/node_exporter/releases/download}"
DOWNLOAD_URL="${DOWNLOAD_BASE_URL}/v${NODE_EXPORTER_VERSION}/${TARBALL}"

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

CHECKSUM_URL="${DOWNLOAD_BASE_URL}/v${NODE_EXPORTER_VERSION}/sha256sums.txt"

# Download with retry logic
download_with_retry() {
    local url=$1
    local output=$2
    local max_attempts=3
    local timeout=30

    for attempt in $(seq 1 "$max_attempts"); do
        echo "[INFO] Downloading ${url} (attempt ${attempt}/${max_attempts})"
        if wget --timeout="${timeout}" --tries=1 --quiet -O "$output" "$url"; then
            echo "[INFO] Download successful"
            return 0
        fi
        echo "[WARN] Download failed (attempt ${attempt}/${max_attempts})"
        if [ "$attempt" -lt "$max_attempts" ]; then
            sleep 5
        fi
    done
    echo "[ERROR] Failed to download after ${max_attempts} attempts"
    return 1
}

echo "[INFO] Downloading ${DOWNLOAD_URL}"
download_with_retry "${DOWNLOAD_URL}" "${TMP_DIR}/${TARBALL}" || exit 1
download_with_retry "${CHECKSUM_URL}" "${TMP_DIR}/sha256sums.txt" || exit 1

echo "[INFO] Verifying checksum"
(cd "$TMP_DIR" && grep "${TARBALL}" sha256sums.txt | sha256sum --check --status)
echo "[INFO] Checksum verified"

tar -xzf "${TMP_DIR}/${TARBALL}" -C "${TMP_DIR}"
install -m 755 "${TMP_DIR}/node_exporter-${NODE_EXPORTER_VERSION}.linux-${ARCH_LABEL}/node_exporter" "${INSTALL_DIR}/node_exporter"

echo "[INFO] Creating node_exporter user"
id -u node_exporter &>/dev/null || useradd --system --no-create-home --shell /usr/sbin/nologin node_exporter

echo "[INFO] Creating systemd service"
cat > "${SERVICE_FILE}" <<EOF
[Unit]
Description=Prometheus Node Exporter
After=network.target

[Service]
Type=simple
User=node_exporter
Group=node_exporter
ExecStart=${INSTALL_DIR}/node_exporter
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now node_exporter

echo "[INFO] node_exporter started on port 9100"
