#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
set -euo pipefail
[[ $(id -u) == 0 ]] || { echo 'Run with sudo on each allocated compute node' >&2; exit 1; }
participant=${1:?Usage: prepare-compute.sh PARTICIPANT_USER}
id "$participant"
lab=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
stage=${AIM344_STAGE_DIR:-/fsx/aim344}
# Node-local staging is an explicit facilitator choice. Copy the prepared tree
# and images to the same absolute path on every assigned compute node.
if [[ ${AIM344_NODE_LOCAL:-0} == 1 ]]; then
    [[ $stage == /opt/* && -d $stage ]] || { echo 'Node-local staging requires an existing /opt directory' >&2; exit 2; }
else
    findmnt -T "$stage" -n -o FSTYPE | grep -qx lustre
fi
install -d -o root -g root -m 0755 /opt/aim344-healthcheck /opt/aim344
tar -xzf "$stage/healthcheck-pinned.tgz" -C /opt/aim344-healthcheck
chown -R root:root /opt/aim344-healthcheck
install -o root -g root -m 0755 "$lab/prejob-prolog.sh" /opt/aim344/prejob-prolog.sh
install -d -o root -g root -m 0755 /var/lib/aim344-device-recovery
install -o root -g root -m 0755 "$lab/facilitator/restore-runtime.sh" /var/lib/aim344-device-recovery/restore-runtime.sh
install -o root -g root -m 0755 "$lab/facilitator/device-fault.sh" /usr/local/sbin/aim344-device-fault
bash /var/lib/aim344-device-recovery/restore-runtime.sh "$participant" "$stage"
sha256sum /opt/aim344/prejob-prolog.sh "$stage/healthcheck-pinned.tgz"
