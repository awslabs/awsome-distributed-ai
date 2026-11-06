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
# The health Prolog on each GPU node takes the assigned EFA count only from
# /etc/aim344-maintenance.json. prepare-target.py writes the target's whole
# file. This sets the count on both nodes and keeps any other settings.
if [[ -n ${AIM344_EXPECTED_EFA_DEVICES:-} ]]; then
    python3 - "$AIM344_EXPECTED_EFA_DEVICES" <<'PY'
import json, os, sys
path = '/etc/aim344-maintenance.json'
config = {}
if os.path.exists(path):
    with open(path) as source:
        config = json.load(source)
config['expected_efa_devices'] = int(sys.argv[1])
with os.fdopen(os.open(path + '.new', os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'w') as output:
    json.dump(config, output)
os.chmod(path + '.new', 0o600)
os.replace(path + '.new', path)
PY
fi
install -d -o root -g root -m 0755 /var/lib/aim344-device-recovery
install -o root -g root -m 0755 "$lab/facilitator/restore-runtime.sh" /var/lib/aim344-device-recovery/restore-runtime.sh
install -o root -g root -m 0755 "$lab/facilitator/device-fault.sh" /usr/local/sbin/aim344-device-fault
# enroot creates /tmp/enroot and its cache and data directories as whichever
# account runs it first, so a run as root would leave them closed to the
# participant. systemd-tmpfiles keeps them shared, now and at every boot.
printf 'd %s 1777 root root -\n' /tmp/enroot /tmp/enroot/cache /tmp/enroot/data > /etc/tmpfiles.d/aim344-enroot.conf
systemd-tmpfiles --create /etc/tmpfiles.d/aim344-enroot.conf
bash /var/lib/aim344-device-recovery/restore-runtime.sh "$participant" "$stage"
# Read both images into the page cache, so that the first container start after
# this boot does not read them from the volume. Run this script on every node at
# the same time; each node reads its own copy.
python3 "$lab/facilitator/image-cache.py" read "$stage/aim344.sqsh" "$stage/nccl-baseline.sqsh"
sha256sum /opt/aim344/prejob-prolog.sh "$stage/healthcheck-pinned.tgz"
