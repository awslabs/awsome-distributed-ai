#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# IMEX setup for P6e-GB200 on ParallelCluster (OnNodeStart head-node custom action +
# Slurm prolog). Forms the cross-instance NVLink (IMEX) domain across the allocation so
# the 18 instances present as one 72-GPU NVLink domain.
#
# On Slurm >= 24.05 prefer the built-in switch/nvidia_imex plugin instead of this script.
set -euo pipefail

IMEX_CONF=/etc/nvidia-imex/nodes_config.cfg
IMEX_SERVICE=nvidia-imex

# Only relevant on GB200 compute nodes (they carry the IMEX package).
if ! command -v nvidia-imex-ctl >/dev/null 2>&1; then
  exit 0
fi

# Build the IMEX peer list from the Slurm allocation (one line per node IP).
if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
  mkdir -p "$(dirname "$IMEX_CONF")"
  scontrol show hostnames "$SLURM_JOB_NODELIST" \
    | while read -r h; do getent hosts "$h" | awk '{print $1}'; done > "$IMEX_CONF"
fi

# (Re)start the IMEX service so the domain forms with the current peer set.
systemctl restart "$IMEX_SERVICE" || nvidia-imex -c "$IMEX_CONF" &

# Brief wait, then assert the domain is UP (fail the prolog if not).
for _ in $(seq 1 30); do
  if nvidia-imex-ctl -N 2>/dev/null | grep -q "Domain State: UP"; then
    echo "IMEX domain UP on $(hostname)"
    exit 0
  fi
  sleep 2
done

echo "ERROR: IMEX domain did not reach UP on $(hostname)" >&2
exit 1
