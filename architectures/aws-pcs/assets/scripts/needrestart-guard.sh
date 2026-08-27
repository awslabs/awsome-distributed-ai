#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# needrestart-guard.sh — protect running Slurm jobs from unattended-upgrades.
#
# apt-daily-upgrade updates base libraries (e.g. glibc). needrestart then
# auto-restarts every service linked against them. slurmd links libc, so it
# gets restarted mid-upgrade — which KILLS the jobs running under it (the step
# is torn down and the job requeues from scratch). Exclude slurmd from
# needrestart's automatic restart so a security upgrade can never take down a
# running job. Security packages still install; only the slurmd auto-restart
# is suppressed. qr(^slurmd) also covers the versioned units (e.g. slurmd-25.11).
#
# Runs as root, first-boot only. Idempotent (overwrite is fine).

set -euo pipefail

LOG=/var/log/pcs-needrestart-guard.log
mkdir -p /etc/needrestart/conf.d
cat > /etc/needrestart/conf.d/90-pcs-slurm.conf <<'NRCONF'
# AWS PCS: never auto-restart slurmd — restarting it kills the jobs running
# under it. Managed by needrestart-guard.sh (first-boot lifecycle action).
# Set only the slurmd key; do NOT reassign the whole %nrconf override_rc hash,
# which would discard needrestart's shipped defaults.
$nrconf{override_rc}{qr(^slurmd)} = 0;
NRCONF

echo "needrestart: slurmd excluded from auto-restart" | tee "$LOG"
