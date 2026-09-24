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
# Replacement nodes retain the checked bootstrap helper. Revalidate runtime
# parents before handing over fixtures or declaring recovery ready. Never chown
# a parent that a participant recreated after /tmp cleanup; replace that node.
# Legacy nodes without this initializer keep their existing provisioning path.
if [[ -f /usr/local/sbin/aim344-replacement-bootstrap ]]; then
    /usr/bin/python3 /usr/local/sbin/aim344-replacement-bootstrap --prepare-enroot
fi
# /run is a root-owned tmpfs, so the mountpoint's own name is not participant
# controlled. It is still created and checked without following a link, because
# `install -d` on an existing symlink-to-directory applies the mode to the link's
# target (measured on this host: `install -d -m 0755 <link>` left the pointed-to
# directory at 0755). A pre-existing non-directory here is refused rather than
# mounted over.
if [[ -L /run/aim344-checkpoints ]]; then
    echo 'Refusing: /run/aim344-checkpoints is a symlink.' >&2
    exit 1
fi
install -d -m 0755 /run/aim344-checkpoints
if ! mountpoint -q /run/aim344-checkpoints; then
    mount -t tmpfs -o size=32M,mode=0700 tmpfs /run/aim344-checkpoints
fi
[[ $(findmnt -n -o FSTYPE /run/aim344-checkpoints) == tmpfs ]]
[[ $(df -B1 --output=size /run/aim344-checkpoints | tail -1 | tr -d ' ') == 33554432 ]]
# Hand the fixture and anything a previous round left inside it to this table.
# The directory alone is not enough: a fixture marker or checkpoint owned by the
# previous table's account stays unwritable for this one, and the workload then
# fails inside the container with no indication that ownership is the cause.
#
# Every operation below is performed on an open descriptor, never on a name that
# root re-resolves. The fixture's contents are deliberately participant writable,
# so an entry there can be a symlink pointing anywhere on the node, and a
# path-based `chown` dereferences a symlink argument by default: measured on this
# host, `find /fixture -mindepth 1 -maxdepth 1 -exec chown ... {} +` hands the
# link's *target* to chown, while `fchownat(..., AT_SYMLINK_NOFOLLOW)` and
# `os.chown(..., follow_symlinks=False)` leave the target untouched. So each entry
# is opened O_NOFOLLOW, checked to be a regular file or directory on the fixture's
# own filesystem, and then changed through its descriptor, which refers to the
# inode that was opened rather than to a name a participant can swap afterwards.
python3 - "$participant" /run/aim344-checkpoints <<'PYHANDOVER'
import grp
import os
import pwd
import stat
import sys

participant, fixture_path = sys.argv[1], sys.argv[2]
entry = pwd.getpwnam(participant)
uid, gid = entry.pw_uid, entry.pw_gid
group = grp.getgrgid(gid).gr_name

# The mountpoint's own name lives under root-owned /run, so its ancestors are not
# participant controlled; it is still opened O_NOFOLLOW so that neither the
# ownership nor the mode below can be applied through a substituted link.
fixture = os.open(fixture_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
info = os.fstat(fixture)
if not stat.S_ISDIR(info.st_mode):
    raise SystemExit('The checkpoint fixture is not a directory.')
os.fchown(fixture, uid, gid)
os.fchmod(fixture, 0o700)
device = info.st_dev

handed, refused = [], []
for name in sorted(os.listdir(fixture)):
    try:
        handle = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                         dir_fd=fixture)
    except OSError as error:
        # ELOOP is a symlink refusing to be followed, which is the case this
        # exists to stop. Anything else is reported rather than worked around.
        refused.append(f'{name} ({os.strerror(error.errno)})')
        continue
    try:
        record = os.fstat(handle)
        if record.st_dev != device:
            refused.append(f'{name} (outside the fixture filesystem)')
            continue
        if not (stat.S_ISREG(record.st_mode) or stat.S_ISDIR(record.st_mode)):
            refused.append(f'{name} (not a regular file or directory)')
            continue
        # The st_dev comparison is what makes owning a regular file safe here. A
        # hard link would otherwise let a participant expose a root-owned inode
        # under a name inside the fixture, and a descriptor cannot tell a link
        # from an original. Hard links cannot cross filesystems, and the fixture
        # is its own dedicated tmpfs (asserted above), whose only contents are
        # this exercise's, so an inode on that device is fixture content by
        # construction.
        os.fchown(handle, uid, gid)
        handed.append(name)
    finally:
        os.close(handle)
os.close(fixture)

print(f'fixture_owner={participant}:{group} handed_over={len(handed)} '
      f'entries={",".join(handed) if handed else "none"}')
if refused:
    # Not a failure of the restoration: the fixture is participant writable, so
    # an unusable entry is the participant's own leftover. It is named, left
    # exactly as it is, and never followed.
    print('left_untouched=' + '; '.join(refused))
PYHANDOVER
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
