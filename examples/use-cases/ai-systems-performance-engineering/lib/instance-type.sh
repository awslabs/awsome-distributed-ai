#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Read the physical host, never a participant-supplied instance-type variable.
detect_instance_type() {
    local detected token
    detected=$(cat /sys/devices/virtual/dmi/id/product_name 2>/dev/null || true)
    if [[ ! $detected =~ ^[a-z][a-z0-9-]*\.[a-z0-9]+$ ]]; then
        token=$(curl -fsS --connect-timeout 1 --max-time 2 -X PUT \
            -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' http://169.254.169.254/latest/api/token 2>/dev/null || true)
        detected=$(curl -fsS --connect-timeout 1 --max-time 2 \
            -H "X-aws-ec2-metadata-token: $token" http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || true)
    fi
    [[ $detected =~ ^[a-z][a-z0-9-]*\.[a-z0-9]+$ ]] || detected=unknown
    printf '%s\n' "$detected"
}

apply_g7_protocol() {
    local detected
    detected=$(detect_instance_type)
    case "$detected" in
        g7.*)
            # Spain 2026-09-10: 4 ranks/node, 1 EFA, 2 GiB all-reduce:
            # SENDRECV 5.2 GB/s versus RDMA 20.3 GB/s. See VALIDATION.md.
            export OFI_NCCL_PROTOCOL=RDMA
            printf 'host=%s instance_type=%s OFI_NCCL_PROTOCOL=RDMA\n' "$(hostname)" "$detected" >&2
            ;;
    esac
}
