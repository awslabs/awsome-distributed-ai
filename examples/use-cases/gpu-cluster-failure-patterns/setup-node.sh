#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Prepare one GPU node for the lab. Run it with sudo on each GPU node:
#   sudo bash setup-node.sh PARTICIPANT_USER HEALTHCHECK_TGZ
set -euo pipefail
participant=${1:?Usage: sudo bash setup-node.sh PARTICIPANT_USER HEALTHCHECK_TGZ}
healthcheck=${2:?Give the health suite archive}
lab=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

install -d /opt/aim344-healthcheck /opt/aim344/.prolog
tar -xzf "$healthcheck" -C /opt/aim344-healthcheck
# The Slurm Prolog of the lab's partition: Checks 0 and 2 before every job.
install -m 0755 "$lab/prejob-prolog.sh" /opt/aim344/.prolog/dispatch.sh

# enroot creates these as whichever account runs it first; keep them shared.
printf 'd %s 1777 root root -\n' /tmp/enroot /tmp/enroot/cache /tmp/enroot/data > /etc/tmpfiles.d/aim344-enroot.conf
systemd-tmpfiles --create /etc/tmpfiles.d/aim344-enroot.conf

# The 32 MiB tmpfs that Incident 2 fills. A mount unit brings it back after a reboot.
unit=$(systemd-escape -p --suffix=mount /run/aim344-checkpoints)
cat > "/etc/systemd/system/$unit" <<UNIT
[Unit]
Description=AIM344 checkpoint directory

[Mount]
What=tmpfs
Where=/run/aim344-checkpoints
Type=tmpfs
Options=size=32M,mode=0700,uid=$(id -u "$participant"),gid=$(id -g "$participant")

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now "$unit"
findmnt /run/aim344-checkpoints
