#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Install an IMEX Slurm prolog on a PCS GB200 compute node. Fetched + run by the
# add-cng-p6e-gb200.yaml UserData at first boot. The prolog forms the cross-instance
# NVLink (IMEX) domain per allocation so the p6e-gb200.36xlarge instances in a job
# present as one 72-GPU NVLink domain.
#
# PCS managed Slurm does not wire IMEX natively. On Slurm >= 24.05 you can instead use
# the built-in switch/nvidia_imex plugin (set in slurm.conf) and skip this prolog.
set -euo pipefail

if ! command -v nvidia-imex-ctl >/dev/null 2>&1; then
  echo "nvidia-imex-ctl not present -- not a GB200 IMEX node; nothing to do."
  exit 0
fi

PROLOG_DIR=/etc/slurm/prolog.d
mkdir -p "$PROLOG_DIR"
PROLOG="$PROLOG_DIR/91_nvidia_imex.sh"

cat > "$PROLOG" <<'PROLOG_EOF'
#!/bin/bash
# Slurm prolog: form the IMEX (cross-instance NVLink) domain for this allocation.
set -euo pipefail
command -v nvidia-imex-ctl >/dev/null 2>&1 || exit 0
[ -n "${SLURM_JOB_NODELIST:-}" ] || exit 0

mkdir -p /etc/nvidia-imex
scontrol show hostnames "$SLURM_JOB_NODELIST" \
  | while read -r h; do getent hosts "$h" | awk '{print $1}'; done \
  > /etc/nvidia-imex/nodes_config.cfg

systemctl restart nvidia-imex 2>/dev/null || nvidia-imex -c /etc/nvidia-imex/nodes_config.cfg &

for _ in $(seq 1 30); do
  if nvidia-imex-ctl -N 2>/dev/null | grep -q "Domain State: UP"; then
    echo "IMEX domain UP on $(hostname)"
    exit 0
  fi
  sleep 2
done
echo "ERROR: IMEX domain did not reach UP on $(hostname)" >&2
exit 1
PROLOG_EOF

chmod +x "$PROLOG"
echo "Installed IMEX Slurm prolog at $PROLOG (GB200 NVLink-domain formation)."
