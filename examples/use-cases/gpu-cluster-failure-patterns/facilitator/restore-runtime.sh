#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Run from the external maintenance route after the PCS-coordinated Slurm reboot.
# Validate the provisioned staging layout first (NVMe bind or root directory).
# This script never formats storage.
set -euo pipefail
[[ $(id -u) == 0 ]] || { echo 'Use the participant recovery command; this helper requires the trusted maintenance route.' >&2; exit 1; }
participant=${1:?Usage: restore-runtime.sh PARTICIPANT_USER STAGE_DIR}
stage=${2:?Provide the validated staging path}
[[ $stage == /* && -d $stage && $stage != / ]] || exit 2
id "$participant"
findmnt -T "$stage"
for image in aim344.sqsh nccl-baseline.sqsh; do
    [[ -s $stage/$image ]] || { echo "Missing staged image: $stage/$image" >&2; exit 1; }
done
# The 32 MiB tmpfs is the full disk of the storage incident. It is mounted for
# the participant, who owns the directory and everything the job writes in it.
install -d -m 0755 /run/aim344-checkpoints
if ! mountpoint -q /run/aim344-checkpoints; then
    mount -t tmpfs -o "size=32M,mode=0700,uid=$(id -u "$participant"),gid=$(id -g "$participant")" \
        tmpfs /run/aim344-checkpoints
fi
[[ $(findmnt -n -o FSTYPE /run/aim344-checkpoints) == tmpfs ]]
[[ $(df -B1 --output=size /run/aim344-checkpoints | tail -1 | tr -d ' ') == 33554432 ]]
# Initialize fixture records as the participant, preserving an existing checkpoint.
runuser -u "$participant" -- python3 - <<'PYFIXTURE'
import json
from pathlib import Path
path = Path('/run/aim344-checkpoints')
marker = path / '.aim344-fixture'
if marker.is_symlink():
    raise SystemExit('Refusing a symlink fixture marker.')
try:
    with marker.open('x') as output:
        output.write('AIM344 isolated checkpoint fixture\n')
except FileExistsError:
    if marker.read_text().strip() != 'AIM344 isolated checkpoint fixture':
        raise SystemExit('Unexpected existing fixture marker.')
checkpoint = path / 'last.json'
if checkpoint.is_symlink():
    raise SystemExit('Refusing a symlink checkpoint record.')
try:
    with checkpoint.open('x') as output:
        json.dump({'next_step': 0}, output)
        output.write('\n')
except FileExistsError:
    saved = json.loads(checkpoint.read_text())
    if not isinstance(saved.get('next_step'), int) or saved['next_step'] < 0:
        raise SystemExit('Invalid existing checkpoint step.')
PYFIXTURE
[[ -x /opt/aim344/prejob-prolog.sh ]]
systemctl is-active slurmd.service
findmnt /run/aim344-checkpoints
stat -c '%U %a %n' /run/aim344-checkpoints
printf 'Runtime paths restored; run the pinned health suite before resuming the drained node.\n'
