#!/bin/bash

export Nsight_version=2025.1.1 # Nsight Version
export Nsight_download_url=https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_1/NsightSystems-linux-cli-public-2025.1.1.131-3554042.deb
export Nsight_cli_installer=$(basename "$Nsight_download_url")

export Nsight_Path=/fsx/ubuntu/nsight-latest

# Download Nsight CLI
wget ${Nsight_download_url}

# Install
sudo dpkg -i ${Nsight_cli_installer}

# This would place the nsys binay at /opt/nvidia/nsight-systems-cli/2025.1.1/target-linux-x64/nsys
# Move to FSx filesystem
mkdir -p ${Nsight_Path}
cp -r /opt/nvidia/nsight-systems-cli/${Nsight_version}/* ${Nsight_Path}
