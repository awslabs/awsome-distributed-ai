#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Check if Slurm ctld service is running to identify controller node:
if systemctl is-active --quiet slurmctld; then
    # Check if Go is installed, if not, install it
    if ! command -v go &> /dev/null; then
        echo "Go is not installed. Installing Go..."
        rm -rf /usr/local/go
        wget https://go.dev/dl/go1.22.2.linux-amd64.tar.gz
        tar -C /usr/local -xzf go1.22.2.linux-amd64.tar.gz
        export PATH=$PATH:/usr/local/go/bin
    else
        echo "Go is already installed."
    fi

    # Environment variable HOME is not set in LCS. It is needed for Go.
    # Setting it only for this script & child processes.
    export HOME=$(pwd)

    # Install SLURM exporter from source.
    # slurm_exporter is licensed under GPLv3. See LICENSE_SLURM_EXPORTER.txt as well.
    echo "This was identified as the controller node because Slurmctld is running. Begining SLURM Exporter Installation"
    git clone -b v1.1.0 https://github.com/SckyzO/slurm_exporter.git
    cd slurm_exporter

    make build && cp bin/slurm_exporter /usr/bin/
    tee /etc/systemd/system/slurm_exporter.service > /dev/null <<EOF
[Unit]
Description=Prometheus SLURM Exporter

[Service]
Environment=PATH=/opt/slurm/bin:\$PATH
ExecStart=/usr/bin/slurm_exporter
Restart=on-failure
RestartSec=15
Type=simple

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload
    systemctl enable --now slurm_exporter
    echo "Prometheus SLURM Exporter installation completed successfully."
else
    echo "This was identified as a worker node because Slurmctld is not running. Did not begin Prometheus SLURM Exporter installation. Exiting gracefully."
    exit 0
fi
