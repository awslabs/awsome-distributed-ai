#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
set -euo pipefail
[[ $(id -u) == 0 ]] || { echo 'Run with sudo on each allocated compute node' >&2; exit 1; }
participant=${1:?Usage: prepare-compute.sh PARTICIPANT_USER}
id "$participant"
lab=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
stage=${AIM344_STAGE_DIR:-/fsx/aim344}
findmnt -T "$stage" -n -o FSTYPE | grep -qx lustre
install -d -o root -g root -m 0755 /opt/aim344-healthcheck /opt/aim344
tar -xzf "$stage/healthcheck-pinned.tgz" -C /opt/aim344-healthcheck
chown -R root:root /opt/aim344-healthcheck
install -o root -g root -m 0755 "$lab/prejob-prolog.sh" /opt/aim344/prejob-prolog.sh
install -d -m 0755 /run/aim344-checkpoints
if ! mountpoint -q /run/aim344-checkpoints; then
    mount -t tmpfs -o size=32M,mode=0700 tmpfs /run/aim344-checkpoints
fi
[[ $(findmnt -n -o FSTYPE /run/aim344-checkpoints) == tmpfs ]]
chown "$participant:$(id -gn "$participant")" /run/aim344-checkpoints
chmod 0700 /run/aim344-checkpoints
printf '%s\n' 'AIM344 isolated checkpoint fixture' > /run/aim344-checkpoints/.aim344-fixture
findmnt /run/aim344-checkpoints
sha256sum /opt/aim344/prejob-prolog.sh "$stage/healthcheck-pinned.tgz"
