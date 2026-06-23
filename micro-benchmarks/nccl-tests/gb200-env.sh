#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Canonical NCCL + EFA environment block for GB200 / P6e-GB200 UltraServers.
# Source this from every GB200 NCCL workload (training, inference, microbenchmarks):
#   source /path/to/micro-benchmarks/nccl-tests/gb200-env.sh
#
# WHY THIS IS MINIMAL (do not add NCCL_PROTO / NCCL_ALGO / NCCL_BUFFSIZE here):
# On a P6e-GB200 UltraServer, 18 instances of 4 GB200 GPUs are federated into ONE
# 72-GPU multi-node NVLink domain (NVL36x2 wiring). The performance story splits at
# the NVLink-domain boundary:
#   * INSIDE one UltraServer (<=72 GPUs): NCCL uses NVLS -- NVLink SHARP reduction
#     performed in-fabric by the NVSwitch ASIC. Identical to a reference NVL72 system;
#     the SHARP-style reduction is fully preserved on AWS because it lives in the
#     NVSwitch fabric, not in the EC2 network.
#   * ACROSS UltraServers (>72 GPUs): traffic falls onto EFA, which has NO in-network
#     SHARP. aws-ofi-nccl implements only NCCL's point-to-point transport (ncclNet)
#     and exports no ncclCollNet_v* symbol, so NCCL_COLLNET_ENABLE=1 and
#     NCCL_ALGO=Collnet* are SILENT NO-OPS on EFA. The algorithm that crosses the
#     boundary efficiently is NVLSTree: NVLink SHARP within each 72-GPU domain, then a
#     Tree reduction between domains carried by aws-ofi-nccl over EFA.
#
# Protocol/algorithm/channel selection is owned by the bundled aws-ofi-nccl platform
# tuner (default-on for AWS). Since aws-ofi-nccl 1.17.0 it falls back to the stock
# NCCL tuner the moment you set NCCL_ALGO/NCCL_PROTO explicitly -- so hand-setting
# them from InfiniBand-era guides turns the AWS tuner OFF. Leave them unset.

# --- EFA / libfabric ---
export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1
# export FI_LOG_LEVEL=warn   # uncomment to debug EFA

# --- NCCL (GB200-specific levers) ---
export NCCL_DEBUG=INFO          # logs NIC count, plugin/tuner activation, selected algorithm
export NCCL_NVLS_ENABLE=1       # force-enable NVLink SHARP (default is 2 = "if supported")
export NCCL_MNNVL_ENABLE=1      # tell NCCL it is on a multi-node NVLink platform (NVL36x2)

# Deliberately NOT set (owned by the aws-ofi-nccl tuner): NCCL_PROTO, NCCL_ALGO,
#   NCCL_BUFFSIZE, NCCL_NCHANNELS.
# Deliberately NOT set (no-op on EFA -- there is no in-network SHARP): NCCL_COLLNET_ENABLE.
#
# Cross-UltraServer (>72 GPU) note: to bias toward the inter-domain algorithm you MAY
# set NCCL_ALGO=NVLSTree for that leg, but this disables the platform tuner's
# per-message-size selection. Prefer leaving it to the tuner.

# Version pins this block is validated against (P6e-GB200, 2026-06):
#   CUDA 13.0.2 | NCCL 2.30.4-1 | aws-ofi-nccl ~1.19.0 (bundled in EFA installer 1.48.0)
#   libfabric >= 1.22.0 (series tested at 2.4.0) | GB200 MNNVL needs NCCL >= 2.25.2
