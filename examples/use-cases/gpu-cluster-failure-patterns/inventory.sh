#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
set -euo pipefail
printf 'host=%s utc=%s\n' "$(hostname)" "$(date -u +%FT%TZ)"
printf 'instance_type='
cat /sys/devices/virtual/dmi/id/product_name
printf 'instance_id='
cat /sys/devices/virtual/dmi/id/board_asset_tag
uname -r
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
nvidia-smi topo -m
/opt/amazon/efa/bin/fi_info --version
/opt/amazon/efa/bin/fi_info -p efa
if [[ -d /opt/amazon/ofi-nccl/lib ]]; then
    ls -l /opt/amazon/ofi-nccl/lib/libnccl-*.so*
else
    echo 'OFI plugin library directory is absent from its normal search path.'
fi
