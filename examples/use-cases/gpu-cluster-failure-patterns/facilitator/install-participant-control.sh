#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Install the participant control path on one assigned pair. Facilitator runs
# this once per pair before the session, from the coordinator, as root.
#
#   sudo bash install-participant-control.sh <assignment-id> <participant-user> \
#        <target-node> <target-instance-id> <target-host>
#
# What it creates:
#   an unprivileged OS user for the participant, with no sudo entry
#   an SSH key pair for that participant, and a forced-command control user
#   /etc/aim344-device-session.json describing that one assignment
#   a root key pair for the coordinator's maintenance route to the target
#
# It does not grant a shell anywhere: both keys are pinned to forced commands.
set -euo pipefail

[[ $(id -u) == 0 ]] || { echo 'Run this with sudo on the coordinator node.' >&2; exit 1; }

assignment=${1:?assignment id, for example table-1}
participant=${2:?participant OS user to create, for example aim344-t1}
target_node=${3:?Slurm node name of the exercise target}
target_instance=${4:?EC2 instance id of the exercise target}
target_host=${5:?address of the exercise target}

[[ $assignment =~ ^[a-z0-9-]+$ ]] || { echo 'Bad assignment id.' >&2; exit 2; }
[[ $participant =~ ^[a-z][a-z0-9-]*$ ]] || { echo 'Bad participant user name.' >&2; exit 2; }
[[ $target_node =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Bad Slurm node name.' >&2; exit 2; }
[[ $target_instance =~ ^i-[0-9a-f]+$ ]] || { echo 'Bad instance id.' >&2; exit 2; }

lab=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

# ---------------------------------------------------------------------------
# Privileged path preparation.
#
# Every directory this installer creates inside an account's home is created BY
# that account, and every shared directory it creates outside one is created
# with the path components checked rather than followed. The rejected version
# used `install -d -o "$participant" ... "$home/.config/enroot"` as root, and
# reruns are explicitly allowed on an existing account (see the account block
# below), so that account could replace `.config` or `.ssh` with a symlink to a
# privileged directory and have root apply the requested ownership and mode to
# the target. GNU install resolves symlinks in the destination path: measured on
# this stack, `install -d` through a directory symlink returned 0 and left the
# real directory 0755 and owned by the named user
# (participant-revision/test-logs/rework4-r1-primitive-probe.log:14-18).
#
# Two rules follow, and they are what the functions below implement.
#
#   1. A directory that will belong to an unprivileged account is prepared while
#      running as that account. Then a substituted component grants nothing the
#      account did not already have, and no privilege is available to donate.
#   2. A directory outside any such account -- the shared container-runtime
#      parent under /tmp or /var/tmp -- is prepared as root, but every component
#      is opened O_NOFOLLOW and its ownership is checked before the mode is set,
#      and it is created with mkdir, which fails rather than reusing a name
#      another local user precreated.
# ---------------------------------------------------------------------------

# Prepare "$base/$relative" as $owner, with $mode. $base is a root-controlled
# path (an account's home directory as recorded in passwd); only the components
# below it are treated as untrusted and opened O_NOFOLLOW.
prepare_owned_dir() {
    local owner=$1 base=$2 relative=$3 mode=$4
    runuser -u "$owner" -- python3 - "$base" "$relative" "$mode" <<'PY'
import os
import stat
import sys

base, relative, mode = sys.argv[1], sys.argv[2], int(sys.argv[3], 8)
# The base is opened normally: it is root-provisioned and named by passwd.
handle = os.open(base, os.O_RDONLY | os.O_DIRECTORY)
try:
    for name in [part for part in relative.split('/') if part]:
        try:
            # mkdir is the atomic part: it either creates this component or
            # tells us it already exists. It never follows a symlink.
            os.mkdir(name, 0o700, dir_fd=handle)
        except FileExistsError:
            pass
        # O_NOFOLLOW|O_DIRECTORY refuses a component that is a symlink (ELOOP)
        # or not a directory (ENOTDIR) instead of resolving it.
        nested = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                         dir_fd=handle)
        os.close(handle)
        handle = nested
    info = os.fstat(handle)
    if info.st_uid != os.geteuid():
        raise SystemExit(f'{base}/{relative} is owned by uid {info.st_uid}, not '
                         f'by the account preparing it')
    os.fchmod(handle, mode)
    now = os.fstat(handle)
    print(f'   {base}/{relative} mode {oct(stat.S_IMODE(now.st_mode))} '
          f'uid {now.st_uid} gid {now.st_gid}')
finally:
    os.close(handle)
PY
}

# Prepare one shared container-runtime parent as root, refusing any name another
# local user already owns or replaced with a link.
prepare_shared_parent() {
    python3 - "$1" <<'PY'
import os
import stat
import sys

path = sys.argv[1].rstrip('/')
base, name = os.path.split(path)
if not name:
    raise SystemExit(f'refusing: {sys.argv[1]} names no directory to prepare')
# The immediate parent must be a root-owned sticky directory (/tmp, /var/tmp),
# so no other local user can rename or replace the entry we are about to touch.
parent = os.open(base, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
try:
    info = os.fstat(parent)
    if info.st_uid != 0 or not info.st_mode & stat.S_ISVTX:
        raise SystemExit(f'refusing to prepare {path}: {base} is not a '
                         f'root-owned sticky directory')
    try:
        os.mkdir(name, 0o1777, dir_fd=parent)
        created = True
    except FileExistsError:
        created = False
    handle = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                     dir_fd=parent)
    try:
        entry = os.fstat(handle)
        if entry.st_uid != 0:
            # This is the one case the old code silently "fixed": a participant
            # who ran the container runtime first owns this directory 0700 and
            # locks every other table out. Changing its mode as root is exactly
            # the privileged operation being removed here, so report it and let
            # the facilitator remove the directory deliberately.
            raise SystemExit(
                f'refusing to change {path}: it exists and is owned by uid '
                f'{entry.st_uid}, not root. Remove it (it is a cache) and rerun.')
        os.fchmod(handle, 0o1777)
        now = os.fstat(handle)
        print(f'   {path} {"created" if created else "existing"} mode '
              f'{oct(stat.S_IMODE(now.st_mode))} uid {now.st_uid}')
    finally:
        os.close(handle)
finally:
    os.close(parent)
PY
}

# One authorized_keys entry built from an account's own public key file, with
# that file treated as untrusted input. The key lives in the participant's
# .ssh and is writable by them, and a rerun reads it rather than regenerating
# it, so a file holding two lines -- or one line carrying its own options --
# would have appended an entry to the control user's authorized_keys with no
# forced command at all. Only the algorithm and the base64 key material are
# taken, and the comment is this installer's own.
#
# The read itself runs as the key's owner, not as root, and refuses a symlink at
# that name: root reading a participant-named path is the same class of defect as
# root writing one, and its refusal messages would otherwise describe a file the
# participant cannot read.
forced_command_entry() {
    local owner=$1 pubkey=$2 forced=$3 options=$4
    runuser -u "$owner" -- python3 - "$pubkey" "$forced" "$options" <<'PY'
import os
import re
import sys

pubkey_path, forced, options = sys.argv[1], sys.argv[2], sys.argv[3]
descriptor = os.open(pubkey_path, os.O_RDONLY | os.O_NOFOLLOW)
with os.fdopen(descriptor, 'r', newline='') as handle:
    raw = handle.read()
# split('\n') rather than splitlines(): splitlines() also breaks on a bare \r,
# \v and \f, which would let a crafted file hide a second entry from this check
# while ssh still reads it as one line.
lines = [line for line in raw.split('\n') if line.strip()]
if len(lines) != 1:
    raise SystemExit(f'refusing: {pubkey_path} holds {len(lines)} key lines, '
                     f'expected exactly one')
fields = lines[0].split()
if len(fields) < 2:
    raise SystemExit(f'refusing: {pubkey_path} is not an OpenSSH public key line')
algorithm, material = fields[0], fields[1]
if algorithm != 'ssh-ed25519':
    raise SystemExit(f'refusing: {pubkey_path} is a {algorithm} key; this '
                     f'installer generates ssh-ed25519')
if not re.fullmatch(r'[A-Za-z0-9+/]+={0,3}', material):
    raise SystemExit(f'refusing: {pubkey_path} has unusable key material')
for char in ('\r', '\v', '\f', '\x00'):
    if char in lines[0]:
        raise SystemExit(f'refusing: {pubkey_path} contains a control character')
print(f'{forced},{options} {algorithm} {material}')
PY
}

# The session helper may already be installed by the staging step. Prefer an
# explicit source, then the companion layout, then the installed copy.
if [[ -n ${AIM344_SESSION_SOURCE:-} ]]; then
    session_source=$AIM344_SESSION_SOURCE
elif [[ -f $lab/facilitator/device-session.py ]]; then
    session_source=$lab/facilitator/device-session.py
elif [[ -f /usr/local/sbin/aim344-device-session ]]; then
    session_source=/usr/local/sbin/aim344-device-session
else
    echo 'Cannot find device-session.py; set AIM344_SESSION_SOURCE.' >&2
    exit 2
fi
[[ -f $session_source ]] || { echo "Not a file: $session_source" >&2; exit 2; }
slurm_bin=${AIM344_SLURM_BIN:-/opt/aws/pcs/scheduler/slurm-25.05/bin}
control_user=${AIM344_CONTROL_USER:-aim344-control}
state_dir=/var/lib/aim344-device-session
session_config=/etc/aim344-device-session.json
suite_entry=${AIM344_SUITE_ENTRY:?Set AIM344_SUITE_ENTRY to the pinned suite entry point on the target}
export AIM344_PARTITION=${AIM344_PARTITION:?Set AIM344_PARTITION to the assigned queue}

echo "== installing the session helper"
installed=/usr/local/sbin/aim344-device-session
if [[ $(readlink -f "$session_source") == "$(readlink -f "$installed")" ]]; then
    echo "The session helper is already staged at $installed; keeping it."
    chown root:root "$installed"
    chmod 0755 "$installed"
else
    install -o root -g root -m 0755 "$session_source" "$installed"
fi
sha256sum "$installed"

echo "== creating the control user that owns the forced command"
if ! id "$control_user" >/dev/null 2>&1; then
    useradd --system --create-home --shell /bin/bash "$control_user"
fi
# The control user runs the helper through sudo, restricted to that one program
# with no arguments of the caller's choosing beyond the fixed assignment.
install -d -m 0755 /etc/sudoers.d
# sudo resets the environment, which would drop SSH_ORIGINAL_COMMAND and leave
# the helper with no request at all. Keep exactly that one variable; the helper
# validates it against a five-verb grammar, and the assignment stays fixed in
# the forced command rather than coming from the connection.
{
    printf 'Defaults:%s env_keep += "SSH_ORIGINAL_COMMAND"\n' "$control_user"
    printf '%s ALL=(root) NOPASSWD: /usr/local/sbin/aim344-device-session --assignment %s\n' \
        "$control_user" "$assignment"
} > "/etc/sudoers.d/aim344-$assignment"
chmod 0440 "/etc/sudoers.d/aim344-$assignment"
visudo -cf "/etc/sudoers.d/aim344-$assignment"

echo "== creating the unprivileged participant user"
if ! id "$participant" >/dev/null 2>&1; then
    useradd --create-home --shell /bin/bash "$participant"
fi
# Assert the participant holds no administrative rights. A participant who can
# sudo would make the whole exercise's authorization test meaningless.
for group in sudo admin wheel; do
    if id -nG "$participant" | tr ' ' '\n' | grep -qx "$group"; then
        echo "Refusing: $participant is in the $group group." >&2
        exit 3
    fi
done
if sudo -n -l -U "$participant" 2>/dev/null | grep -q '(ALL'; then
    echo "Refusing: $participant already has broad sudo rights." >&2
    exit 3
fi

# The container runtime needs a per-user path under a shared parent. Whichever
# account runs enroot first creates /tmp/enroot, and it is created 0700, so a
# single earlier run by another account locks every participant out: Check 5
# then fails with "mkdir: cannot create directory '/tmp/enroot'" reported as a
# NCCL timeout, which reads as a device fault and is not one. Observed on the
# exercise target after an administrative run. Make the parent shared and
# sticky, so each participant owns its own subdirectory and cannot remove
# another's. Paths come from enroot.conf rather than being assumed.
#
# The extraction is a python parse rather than a sed expression. The previous
# `sed -n 's|^ENROOT_\(RUNTIME\|CACHE\|DATA\)_PATH...|\2|p'` never matched a real
# enroot.conf, because `|` was the s-command delimiter, so each `\|` was an
# escaped delimiter rather than a BRE alternation: the whole pattern only matched
# the literal text `ENROOT_RUNTIME|CACHE|DATA_PATH`. Measured against a real
# configuration shape, it printed nothing and the installer reported "no enroot
# paths configured; skipping". The per-user suffix was also stripped with
# `s|/user-.*$||` alone, which leaves ENROOT_CACHE_PATH's `group-` suffix in
# place and would have prepared `/tmp/enroot/cache` rather than `/tmp/enroot`.
echo "== container runtime paths for the participant"
enroot_parents=$(python3 - /etc/enroot/enroot.conf <<'PY'
import re
import sys

try:
    with open(sys.argv[1]) as handle:
        text = handle.read()
except OSError:
    raise SystemExit(0)
parents = []
for line in text.split('\n'):
    stripped = line.strip()
    if stripped.startswith('#'):
        continue
    match = re.match(r'^ENROOT_(?:RUNTIME|CACHE|DATA)_PATH\s+(\S+)', stripped)
    if not match:
        continue
    path = match.group(1)
    # enroot's own per-identity suffixes. Cut at the first of them so the shared
    # parent is what gets prepared, not a per-user or per-group subdirectory.
    parts = []
    for component in path.split('/'):
        if component.startswith(('user-', 'group-')) or component in ('cache', 'data'):
            break
        parts.append(component)
    candidate = '/'.join(parts)
    if candidate and candidate not in parents:
        parents.append(candidate)
for parent in sorted(parents):
    print(parent)
PY
)
if [[ -z $enroot_parents ]]; then
    echo "   no enroot paths configured; skipping"
else
    for parent in $enroot_parents; do
        case $parent in
            /tmp/*|/run/*|/var/tmp/*) ;;
            *) echo "   refusing to adjust unexpected path: $parent" >&2; continue ;;
        esac
        prepare_shared_parent "$parent"
        # Root's first enroot invocation may also leave these intermediates
        # 0700. The same descriptor/owner checks apply below the sticky parent.
        prepare_shared_parent "$parent/cache"
        prepare_shared_parent "$parent/data"
    done
fi
prepare_owned_dir "$participant" \
    "$(getent passwd "$participant" | cut -d: -f6)" '.config/enroot' 0755

echo "== participant key and forced command"
participant_home=$(getent passwd "$participant" | cut -d: -f6)
prepare_owned_dir "$participant" "$participant_home" '.ssh' 0700
key="$participant_home/.ssh/aim344-exercise"
if [[ ! -s $key ]]; then
    runuser -u "$participant" -- ssh-keygen -t ed25519 -N '' -q -f "$key" \
        -C "aim344-$assignment"
fi
control_home=$(getent passwd "$control_user" | cut -d: -f6)
prepare_owned_dir "$control_user" "$control_home" '.ssh' 0700
control_keys="$control_home/.ssh/authorized_keys"
forced="command=\"sudo /usr/local/sbin/aim344-device-session --assignment $assignment\""
options='no-port-forwarding,no-agent-forwarding,no-X11-forwarding,no-pty,no-user-rc'
entry=$(forced_command_entry "$participant" "$key.pub" "$forced" "$options")
entry="$entry aim344-$assignment"
# Replace only this assignment's entry; other tables keep theirs. Done as the
# control user through a descriptor opened O_NOFOLLOW, for the same reason the
# home directories above are: the previous `touch`/`>`/`mv`/`chown`/`chmod`
# sequence resolved `authorized_keys` as root, so a symlink at that name would
# have had root write this file's contents, ownership and mode onto its target.
runuser -u "$control_user" -- python3 - "$control_home/.ssh" "$entry" \
        "aim344-$assignment" <<'PY'
import os
import sys

ssh_dir, entry, marker = sys.argv[1], sys.argv[2], sys.argv[3]
directory = os.open(ssh_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
try:
    try:
        existing = os.open('authorized_keys', os.O_RDONLY | os.O_NOFOLLOW,
                           dir_fd=directory)
    except FileNotFoundError:
        text = ''
    else:
        with os.fdopen(existing, 'r', newline='') as handle:
            text = handle.read()
    # split('\n'), not splitlines(): splitlines() also breaks on a bare \r, and
    # reassembling on that boundary would silently reshape the file.
    kept = [line for line in text.split('\n')
            if line.strip() and not line.rstrip().endswith(marker)]
    kept.append(entry)
    body = '\n'.join(kept) + '\n'
    # A fresh file, then rename over the name: the entry either lands whole or
    # not at all, and sshd never reads a half-written authorized_keys.
    temporary = f'authorized_keys.new-{os.getpid()}'
    handle = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=directory)
    try:
        os.write(handle, body.encode())
        os.fchmod(handle, 0o600)
    finally:
        os.close(handle)
    os.rename(temporary, 'authorized_keys', src_dir_fd=directory,
              dst_dir_fd=directory)
    print(f'   {ssh_dir}/authorized_keys now holds {len(kept)} entr'
          f'{"y" if len(kept) == 1 else "ies"}')
finally:
    os.close(directory)
PY

echo "== maintenance key for the coordinator's route to the target"
install -d -o root -g root -m 0700 /root/.ssh
maintenance_key=/root/.ssh/aim344-maintenance
if [[ ! -s $maintenance_key ]]; then
    ssh-keygen -t ed25519 -N '' -q -f "$maintenance_key" -C 'aim344-maintenance'
fi
echo 'Install this public key on the TARGET as root, with the forced command:'
printf '  command="/usr/local/sbin/aim344-maintenance",%s %s\n' \
    "$options" "$(cat "$maintenance_key.pub")"
# Slurm resolves the submitting uid on every allocated node. Without a matching
# account on the target the participant's job still runs, but id -un returns a
# bare number there, and anything name-based inside the workload sees no user.
# Observed as "id: cannot find name for user ID 1001" on rank 0.
printf 'Also create the participant on the TARGET with the SAME uid and gid, no sudo:\n'
printf '  groupadd -g %s %s 2>/dev/null; useradd -u %s -g %s -m -s /bin/bash %s\n' \
    "$(id -g "$participant")" "$participant" \
    "$(id -u "$participant")" "$(id -g "$participant")" "$participant"
printf '  then confirm: sudo -n -l -U %s   ->   must report not allowed\n' "$participant"
# The target decides which accounts may own its restored checkpoint fixture and
# hold its one allowed job. Both were a single fixed name, `ubuntu`, so once the
# exercise ran as aim344-t1 the fixture came back 0700 owned by the wrong account
# and the participant could not read it. Add this table to both allowlists there.
printf 'Add this table to the TARGET allowlists, so its fixture and job are its own:\n'
printf '  /etc/aim344-maintenance.json  "participants": [..., "%s"]\n' "$participant"
printf '  /etc/aim344-device-fault.json "participant_users": [..., "%s"]\n' "$participant"
printf '  then confirm: runuser -u %s -- test -w /run/aim344-checkpoints\n' "$participant"

echo "== session configuration"
install -d -o root -g root -m 0700 "$state_dir"
python3 - "$session_config" "$assignment" "$participant" "$target_node" \
        "$target_instance" "$target_host" "$state_dir" "$slurm_bin" "$suite_entry" \
        <<'PY'
import json, os, pwd, sys
(path, assignment, participant, node, instance, host, state_dir, slurm_bin,
 suite_entry) = sys.argv[1:10]
control_user = os.environ.get('AIM344_CONTROL_USER', 'aim344-control')
config = {}
if os.path.exists(path):
    with open(path) as handle:
        config = json.load(handle)
config.setdefault('state_dir', state_dir)
config.setdefault('assignments', {})
config['assignments'][assignment] = {
    'participant_user': participant,
    'participant_uid': pwd.getpwnam(participant).pw_uid,
    # The participant's own primary group. Results are written by a process that
    # has dropped to this uid and gid, so both are needed; without the gid the
    # helper would have to guess one.
    'participant_gid': pwd.getpwnam(participant).pw_gid,
    # The account sshd runs the forced command as. The per-table binding is the
    # key plus its forced command; this only confirms the expected caller.
    'caller_uid': pwd.getpwnam(control_user).pw_uid,
    'answers_dir': f'{pwd.getpwnam(participant).pw_dir}/aim344-results',
    'target_node': node,
    'target_instance_id': instance,
    'target_host': host,
    'coordinator_node': os.uname().nodename,
    'partition': os.environ['AIM344_PARTITION'],
    'drain_reason': 'aim344-device-recovery',
    'job_name_prefix': 'aim344-',
    'slurm_bin': slurm_bin,
    'suite_entry': suite_entry,
    'allowed_checks': {'gpu': [0, 3], 'efa': [2, 6]},
    'recovery_deadline_seconds': int(os.environ.get('AIM344_DEADLINE', '1800')),
    'reboot_wait_seconds': 900,
    'reboot_poll_seconds': 10,
    'maintenance_identity_file': '/root/.ssh/aim344-maintenance',
}
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(descriptor, 'w') as handle:
    json.dump(config, handle, indent=2, sort_keys=True)
    handle.write('\n')
os.chmod(path, 0o600)
print(f'Wrote {assignment} into {path}')
PY

prepare_owned_dir "$participant" "$participant_home" 'aim344-results' 0700

# The deadline safeguard, scheduled rather than waiting for somebody to type it.
# The hook existed but nothing ever invoked it, so an abandoned round had no
# recovery path. This runs on the coordinator, which holds the session state and
# is not the fault target, so a timer here survives the exercise node rebooting.
# It is the existing helper on a schedule, not a new service: no daemon, no port,
# no state of its own, and it exits after one pass.
echo "== deadline sweep timer"
cat > /etc/systemd/system/aim344-deadline-sweep.service <<UNIT
[Unit]
Description=AIM344 exercise deadline safeguard, one pass over every assignment
Documentation=file://$lab/facilitator/DEVICE-RECOVERY.md
ConditionPathExists=$session_config

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/aim344-device-session --sweep
# One pass, then exit. A pass that hangs is killed rather than left holding the
# per-node lock, and systemd does not restart it: the next timer tick is the
# retry, and the helper itself bounds how many times one round is retried.
TimeoutStartSec=900
UNIT
cat > /etc/systemd/system/aim344-deadline-sweep.timer <<'UNIT'
[Unit]
Description=Run the AIM344 exercise deadline safeguard periodically

[Timer]
OnBootSec=5min
OnUnitInactiveSec=5min
AccuracySec=30s
Unit=aim344-deadline-sweep.service

[Install]
WantedBy=timers.target
UNIT
systemctl daemon-reload
systemctl enable --now aim344-deadline-sweep.timer
systemctl list-timers --all aim344-deadline-sweep.timer --no-pager

echo "== participant-facing summary"
cat <<SUMMARY
Assignment:        $assignment
Participant user:  $participant (no sudo)
Exercise node:     $target_node ($target_instance)
Control endpoint:  $control_user@localhost, forced command only
Participant key:   $key
Results directory: $participant_home/aim344-results

Give the participant a terminal as $participant, then:
  AIM344_CONTROL_USER=$control_user ./12.device-exercise.sh status
SUMMARY
