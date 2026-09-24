# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Repairs for the re-review's safety findings S1 to S4.

Each test below was written against the rejected implementation and fails on it.
The rejected revision is commit `dd1d00c6`; where a test needs the pre-repair
behaviour to prove it would have caught the defect, it reconstructs that exact
construction and asserts the unsafe outcome, so nothing here is validated only on
the input that passes.

S1 is an installer defect and cannot be established by reading. `InstallerAsRoot`
runs the real `install-participant-control.sh` inside a container, as root, with
real accounts and real symlinks in the destinations, and then reads the
filesystem. Those tests skip when no container runtime is available; the skip
message says so rather than quietly passing.

S2, S3 and S4 are controller and maintenance defects and are exercised through
the same recorded fake executor the existing suites use, plus, for S4, the real
`maintenance.gpu_restore` function against real recorded-state files with the
`systemctl`/`nvidia-smi` binaries it actually calls replaced by recording stubs
on PATH. No test asserts a return code a mock was told to produce.

Run: python3 -m unittest discover -s <lab>/tests -p 'test_*.py'
"""
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest

LAB = Path(__file__).resolve().parents[1]
HELPER = LAB / 'facilitator' / 'device-session.py'
MAINTENANCE = LAB / 'facilitator' / 'maintenance.py'
INSTALLER = LAB / 'facilitator' / 'install-participant-control.sh'

_rework_spec = importlib.util.spec_from_file_location(
    'aim344_rework_safety3',
    Path(__file__).resolve().parent / 'test_device_session_rework.py')
if _rework_spec is None or _rework_spec.loader is None:        # pragma: no cover
    raise unittest.SkipTest('test_device_session_rework.py not importable')
_rework = importlib.util.module_from_spec(_rework_spec)
_rework_spec.loader.exec_module(_rework)
# Same module object as the shared Base, so `session.Refusal` is the class the
# handler actually raises rather than a second copy of it.
session = _rework.session
Base = _rework.Base
ReworkExecutor = _rework.ReworkExecutor

_maint_spec = importlib.util.spec_from_file_location('aim344_maintenance_safety3',
                                                    MAINTENANCE)
if _maint_spec is None or _maint_spec.loader is None:          # pragma: no cover
    raise unittest.SkipTest(f'Maintenance helper not importable: {MAINTENANCE}')
maintenance = importlib.util.module_from_spec(_maint_spec)
_maint_spec.loader.exec_module(maintenance)


def container_runtime():
    """A usable rootful container runtime, or None.

    The installer creates OS accounts and applies ownership, so it cannot be
    exercised anywhere but as root on a throwaway filesystem.
    """
    for name in ('docker', 'podman'):
        path = shutil.which(name)
        if not path:
            continue
        probe = subprocess.run([path, 'info'], capture_output=True, text=True)
        if probe.returncode == 0:
            return path
    return None


IMAGE = os.environ.get('AIM344_TEST_IMAGE', 'ubuntu:24.04')


# ---------------------------------------------------------------------------
# S1: the installer's privileged writes
# ---------------------------------------------------------------------------

# What the container does. Written as a script rather than a chain of -c
# arguments so the actual installer invocation is readable, and so the negative
# and positive cases run under one root filesystem.
#
# Each case resets the privileged decoy first, so one case's outcome cannot be
# read out of another's. The decoy is /root/aim344-secret, 0700 root:root, with a
# 0700 child.
#
# On which component the attack targets. `install -d -o U -g G -m M a/b/c` applies
# the ownership and mode to the components it CREATES. So the substituted
# component that matters is the LAST one: if `.config` is a real directory and
# `.config/enroot` is a symlink to a privileged directory, `install -d` resolves
# that symlink and applies the requested ownership and mode to its target. That is
# the shape of the author's own probe
# (participant-revision/test-logs/rework4-r1-primitive-probe.log:14-18), and it is
# the shape used below.
CONTAINER_SCRIPT = r'''
set -u
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq python3 openssh-client sudo >/dev/null 2>&1

# The installer calls these; they are not what S1 is about, and a container has
# no systemd, no Slurm and no visudo worth running. Stub them so the run reaches
# the filesystem operations under test. Each stub records that it was called.
mkdir -p /opt/stubs /records
for tool in systemctl visudo; do
    printf '#!/bin/sh\necho "$0 $*" >> /records/stub-calls.txt\nexit 0\n' > /opt/stubs/$tool
    chmod 0755 /opt/stubs/$tool
done
export PATH=/opt/stubs:$PATH

mkdir -p /lab/facilitator
cp /mnt/installer.sh /lab/facilitator/install-participant-control.sh
printf '#!/usr/bin/env python3\n' > /lab/facilitator/device-session.py
mkdir -p /etc/enroot
# A realistic enroot.conf, including a commented-out line and the per-group cache
# path, so the installer's own extraction is exercised rather than a shape chosen
# to make it succeed.
cat > /etc/enroot/enroot.conf <<'CONF'
#ENROOT_RUNTIME_PATH        /run/enroot/user-$(id -u)
ENROOT_RUNTIME_PATH        /tmp/enroot/user-$(id -u)
ENROOT_CACHE_PATH          /tmp/enroot/cache/group-$(id -g)
ENROOT_DATA_PATH           /tmp/enroot/data/user-$(id -u)
CONF
mkdir -p /opt/suite && printf '#!/bin/bash\ntrue\n' > /opt/suite/gpu-healthcheck.sh
export AIM344_SUITE_ENTRY=/opt/suite/gpu-healthcheck.sh
export AIM344_PARTITION=assigned-queue

reset_decoy() {
    rm -rf /root/aim344-secret
    mkdir -p /root/aim344-secret/inside
    chmod 0700 /root/aim344-secret
    chmod 0700 /root/aim344-secret/inside
}

run_installer() {
    bash /lab/facilitator/install-participant-control.sh table-1 aim344-t1 \
        gpu-g7-1 i-0123456789abcdef0 10.0.0.1 > "/records/$1.out" 2> "/records/$1.err"
    echo "$?" > "/records/$1.rc"
}

record_state() {
    python3 - "$1" <<'PY'
import json, os, stat, sys
name = sys.argv[1]
def describe(path):
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return {'exists': False}
    return {'exists': True, 'symlink': stat.S_ISLNK(info.st_mode),
            'dir': stat.S_ISDIR(info.st_mode), 'uid': info.st_uid,
            'gid': info.st_gid, 'mode': oct(stat.S_IMODE(info.st_mode)),
            'target': os.readlink(path) if stat.S_ISLNK(info.st_mode) else None}
paths = ['/home/aim344-t1/.config', '/home/aim344-t1/.config/enroot',
         '/home/aim344-t1/.ssh', '/home/aim344-t1/aim344-results',
         '/tmp/enroot', '/root/aim344-secret', '/root/aim344-secret/inside',
         '/home/aim344-control/.ssh',
         '/home/aim344-control/.ssh/authorized_keys']
state = {p: describe(p) for p in paths}
keys = '/home/aim344-control/.ssh/authorized_keys'
if os.path.exists(keys):
    with open(keys, newline='') as handle:
        state['authorized_keys_lines'] = [
            line for line in handle.read().split('\n') if line.strip()]
with open(f'/records/{name}.json', 'w') as handle:
    json.dump(state, handle, indent=2, sort_keys=True)
PY
}

echo "=== case 1: first install, new account"
reset_decoy
run_installer case1
record_state case1

echo "=== case 2: rerun with real directories already present"
reset_decoy
run_installer case2
record_state case2

# Keep the key pair the successful run generated: the later cases must rerun on a
# real existing account, which is the condition S1 turns on.
cp -a /home/aim344-t1/.ssh /root/keep-ssh

echo "=== case 3a: .config/enroot is a symlink, with the shared parent VALID"
# The installer prepares the shared /tmp parent (line ~321) BEFORE either home
# directory, and it runs under `set -euo pipefail`. A case that also substitutes
# /tmp/enroot therefore stops there and never reaches this branch at all -- which is
# the S1 evidence gap. So here /tmp/enroot is left as the installer's own valid
# root-owned sticky directory and only the participant's `.config/enroot` is
# substituted.
reset_decoy
rm -rf /home/aim344-t1/.config /home/aim344-t1/.ssh /tmp/enroot
cp -a /root/keep-ssh /home/aim344-t1/.ssh
chown -R aim344-t1:aim344-t1 /home/aim344-t1/.ssh
mkdir -p /home/aim344-t1/.config
chown aim344-t1:aim344-t1 /home/aim344-t1/.config
runuser -u aim344-t1 -- ln -s /root/aim344-secret /home/aim344-t1/.config/enroot
run_installer case3a
record_state case3a

echo "=== case 3b: .ssh is a symlink, with the shared parent AND .config/enroot valid"
# .ssh is prepared after .config/enroot, so that one is left a real directory here.
reset_decoy
rm -rf /home/aim344-t1/.config /home/aim344-t1/.ssh /tmp/enroot
mkdir -p /home/aim344-t1/.config/enroot
chown -R aim344-t1:aim344-t1 /home/aim344-t1/.config
ln -s /root/aim344-secret /home/aim344-t1/.ssh
chown -h aim344-t1:aim344-t1 /home/aim344-t1/.ssh
run_installer case3b
record_state case3b

echo "=== case 3c: the participant's results directory is a symlink"
# Prepared last of the three (line ~463), after the session configuration is written,
# so everything before it has to be valid for this branch to be reached at all.
reset_decoy
rm -rf /home/aim344-t1/.config /home/aim344-t1/.ssh /tmp/enroot \
    /home/aim344-t1/aim344-results
mkdir -p /home/aim344-t1/.config/enroot
chown -R aim344-t1:aim344-t1 /home/aim344-t1/.config
cp -a /root/keep-ssh /home/aim344-t1/.ssh
chown -R aim344-t1:aim344-t1 /home/aim344-t1/.ssh
runuser -u aim344-t1 -- ln -s /root/aim344-secret /home/aim344-t1/aim344-results
run_installer case3c
record_state case3c

echo "=== case 3: every destination substituted at once (the combined layout)"
# Kept because it is the layout the previous evidence used, and it is what shows
# that the shared parent is checked FIRST: this run must stop at /tmp/enroot.
reset_decoy
rm -rf /home/aim344-t1/.config
mkdir -p /home/aim344-t1/.config
chown aim344-t1:aim344-t1 /home/aim344-t1/.config
runuser -u aim344-t1 -- ln -s /root/aim344-secret /home/aim344-t1/.config/enroot
rm -rf /home/aim344-t1/.ssh
ln -s /root/aim344-secret /home/aim344-t1/.ssh
chown -h aim344-t1:aim344-t1 /home/aim344-t1/.ssh
rm -rf /tmp/enroot
ln -s /root/aim344-secret /tmp/enroot
run_installer case3
record_state case3

echo "=== case 3b: what the REJECTED construction does to the same layout"
# Not the installer: the exact primitives the rejected revision used, against the
# exact same layout, so this suite is not validated only on the repaired path.
reset_decoy
rm -rf /home/aim344-t1/.config
mkdir -p /home/aim344-t1/.config
runuser -u aim344-t1 -- ln -s /root/aim344-secret /home/aim344-t1/.config/enroot 2>/dev/null \
    || ln -s /root/aim344-secret /home/aim344-t1/.config/enroot
rm -rf /tmp/enroot
ln -s /root/aim344-secret /tmp/enroot
install -d -o aim344-t1 -g aim344-t1 -m 0755 /home/aim344-t1/.config/enroot \
    > /records/rejected.out 2>&1
echo "$?" > /records/rejected.rc
install -d -m 1777 /tmp/enroot >> /records/rejected.out 2>&1
record_state rejected

echo "=== case 3b-prim: the REJECTED primitive on the .ssh destination"
# `install -d -o aim344-t1 -g aim344-t1 -m 0700 "$participant_home/.ssh"` is the exact
# line the rejected installer used. Run against the same layout case3b uses, so that
# case's refusal is paired with a demonstration that the pre-repair code did follow
# the link.
reset_decoy
rm -rf /home/aim344-t1/.ssh
ln -s /root/aim344-secret /home/aim344-t1/.ssh
chown -h aim344-t1:aim344-t1 /home/aim344-t1/.ssh
install -d -o aim344-t1 -g aim344-t1 -m 0700 /home/aim344-t1/.ssh \
    > /records/rejected-ssh.out 2>&1
echo "$?" > /records/rejected-ssh.rc
record_state rejected-ssh

echo "=== case 3c-prim: the REJECTED primitive on the results destination"
reset_decoy
rm -rf /home/aim344-t1/aim344-results
runuser -u aim344-t1 -- ln -s /root/aim344-secret /home/aim344-t1/aim344-results
install -d -o aim344-t1 -g aim344-t1 -m 0700 /home/aim344-t1/aim344-results \
    > /records/rejected-results.out 2>&1
echo "$?" > /records/rejected-results.rc
record_state rejected-results

echo "=== case 4: a second key line in the participant's own public key file"
reset_decoy
rm -rf /home/aim344-t1/.config /home/aim344-t1/.ssh /tmp/enroot
cp -a /root/keep-ssh /home/aim344-t1/.ssh
chown -R aim344-t1:aim344-t1 /home/aim344-t1/.ssh
runuser -u aim344-t1 -- bash -c 'printf "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIINJECTEDKEYMATERIALINJECTEDKEYMATER injected\n" >> /home/aim344-t1/.ssh/aim344-exercise.pub'
run_installer case4
record_state case4

echo "=== case 5: a shared runtime parent precreated by another local user"
reset_decoy
rm -rf /home/aim344-t1/.config /home/aim344-t1/.ssh /tmp/enroot
cp -a /root/keep-ssh /home/aim344-t1/.ssh
chown -R aim344-t1:aim344-t1 /home/aim344-t1/.ssh
python3 - <<'PY'
import os
os.setgid(65534); os.setuid(65534)
os.mkdir('/tmp/enroot', 0o700)
PY
run_installer case5
record_state case5

echo "=== stub calls"
cat /records/stub-calls.txt 2>/dev/null | sort | uniq -c
'''


@unittest.skipUnless(container_runtime(),
                     'no usable container runtime; the installer needs root on a '
                     'throwaway filesystem and is not exercised by reading it')
class InstallerAsRoot(unittest.TestCase):
    """S1: the real installer, run as root, with real symlink negatives.

    One container run produces every case; the assertions below read its recorded
    filesystem state. Sharing the run keeps the class from paying the image cost
    per test while each test still asserts one thing.
    """

    records = {}
    container_output = ''
    workdir = ''

    @classmethod
    def setUpClass(cls):
        runtime = container_runtime()
        assert runtime is not None                        # guarded by skipUnless
        cls.workdir = tempfile.mkdtemp(prefix='aim344-s1-')
        script = Path(cls.workdir) / 'run.sh'
        script.write_text(CONTAINER_SCRIPT)
        finished = subprocess.run(
            [runtime, 'run', '--rm', '-v', f'{INSTALLER}:/mnt/installer.sh:ro',
             '-v', f'{cls.workdir}:/records', IMAGE, 'bash', '/records/run.sh'],
            capture_output=True, text=True, timeout=900)
        cls.container_output = finished.stdout + finished.stderr
        cls.records = {}
        for name in ('case1', 'case2', 'case3', 'case3a', 'case3b', 'case3c',
                     'case4', 'case5', 'rejected', 'rejected-ssh',
                     'rejected-results'):
            path = Path(cls.workdir) / f'{name}.json'
            if path.is_file():
                cls.records[name] = json.loads(path.read_text())
            rc = Path(cls.workdir) / f'{name}.rc'
            if rc.is_file():
                cls.records[f'{name}_rc'] = int(rc.read_text().strip())
            for stream in ('out', 'err'):
                log = Path(cls.workdir) / f'{name}.{stream}'
                if log.is_file():
                    cls.records[f'{name}_{stream}'] = log.read_text()
        if not cls.records:
            raise unittest.SkipTest(
                'the container produced no records; output was:\n'
                + cls.container_output[-4000:])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.workdir, ignore_errors=True)

    def test_a_first_install_succeeds_and_the_participant_owns_its_directories(self):
        """The normal path has to keep working; a refusal is not a repair."""
        self.assertEqual(0, self.records.get('case1_rc'),
                         f'installer failed:\n{self.records.get("case1_err", "")}')
        state = self.records['case1']
        for path in ('/home/aim344-t1/.config/enroot', '/home/aim344-t1/.ssh',
                     '/home/aim344-t1/aim344-results'):
            with self.subTest(path=path):
                self.assertTrue(state[path]['exists'], f'{path} was not created')
                self.assertTrue(state[path]['dir'])
                self.assertNotEqual(0, state[path]['uid'],
                                    f'{path} is owned by root, not the participant')
        self.assertEqual('0o700', state['/home/aim344-t1/.ssh']['mode'])
        self.assertEqual('0o700', state['/home/aim344-t1/aim344-results']['mode'])
        self.assertEqual('0o755', state['/home/aim344-t1/.config/enroot']['mode'])

    def test_the_forced_command_entry_is_installed_for_the_assignment(self):
        lines = self.records['case1'].get('authorized_keys_lines', [])
        self.assertEqual(1, len(lines), f'authorized_keys holds {lines}')
        self.assertTrue(lines[0].startswith(
            'command="sudo /usr/local/sbin/aim344-device-session '
            '--assignment table-1"'), lines[0])
        self.assertIn('no-pty', lines[0])
        self.assertTrue(lines[0].rstrip().endswith('aim344-table-1'), lines[0])

    def test_a_rerun_on_an_existing_account_still_succeeds(self):
        """Reruns are explicitly supported, so refusing them is not the repair."""
        self.assertEqual(0, self.records.get('case2_rc'),
                         f'rerun failed:\n{self.records.get("case2_err", "")}')
        lines = self.records['case2'].get('authorized_keys_lines', [])
        self.assertEqual(1, len(lines),
                         f'a rerun duplicated or dropped the entry: {lines}')

    def test_a_symlinked_destination_does_not_change_the_privileged_target(self):
        """S1, the combined layout: root must not follow the participant's symlink.

        This case substitutes every destination at once, so it establishes the
        shared-parent refusal and nothing beyond it -- the installer's `set -euo
        pipefail` stops it at `/tmp/enroot`. The per-destination cases below are what
        show each home-path branch refusing, with the preceding path valid.
        """
        state = self.records['case3']
        secret = state['/root/aim344-secret']
        self.assertTrue(secret['exists'])
        self.assertEqual(0, secret['uid'],
                         'the privileged directory changed owner through a symlink')
        self.assertEqual('0o700', secret['mode'],
                         'the privileged directory changed mode through a symlink')
        inside = state['/root/aim344-secret/inside']
        self.assertEqual(0, inside['uid'])
        self.assertEqual('0o700', inside['mode'])
        self.assertNotEqual(0, self.records.get('case3_rc'),
                            'the installer reported success on a substituted path')
        # And the honest limit of this case, asserted rather than assumed: it stops
        # at the shared parent, so it says nothing about the home paths. The
        # observed shape of that stop is `prepare_shared_parent`'s descriptor open
        # failing with ENOTDIR on the substituted entry, before any home-path
        # preparation prints anything.
        out = self.records.get('case3_out', '')
        err = self.records.get('case3_err', '')
        self.assertIn("Not a directory: 'enroot'", err)
        self.assertNotIn('/tmp/enroot created', out)
        self.assertNotIn('/tmp/enroot existing', out)
        self.assertNotIn('/home/aim344-t1/.config/enroot mode', out,
                         'the combined case did reach a home path, so its scope '
                         'assertion here is wrong')

    def _assert_home_destination_refused(self, case, destination, reached_marker,
                                        component):
        """One home-path negative: the branch was reached, and it refused.

        `reached` is the part the previous evidence was missing. The installer runs
        under `set -euo pipefail`, so an earlier refusal ends the run; a case whose
        output never shows the preceding preparation succeeding proves nothing about
        the branch under test. `reached_marker` is a line the installer prints only
        after everything before this destination has been prepared, and `component`
        is the path component whose O_NOFOLLOW open must fail.
        """
        out = self.records.get(f'{case}_out', '')
        err = self.records.get(f'{case}_err', '')
        self.assertIn(reached_marker, out,
                      f'{case} stopped before {destination} was attempted, so this '
                      f'case does not exercise that branch')
        self.assertIn(f"Not a directory: '{component}'", err,
                      f'{case} did not fail on {destination} itself: {err[-400:]}')
        self.assertNotEqual(0, self.records.get(f'{case}_rc'),
                            f'the installer reported success with {destination} '
                            f'substituted')
        state = self.records[case]
        secret = state['/root/aim344-secret']
        self.assertEqual(0, secret['uid'],
                         f'{destination} donated ownership of the privileged '
                         f'directory')
        self.assertEqual('0o700', secret['mode'],
                         f'{destination} widened the privileged directory')
        self.assertEqual(0, state['/root/aim344-secret/inside']['uid'])
        self.assertEqual('0o700', state['/root/aim344-secret/inside']['mode'])
        # The substituted name is still the participant's symlink: it was refused,
        # not replaced or followed.
        self.assertTrue(state[destination]['symlink'],
                        f'{destination} is no longer the symlink that was placed '
                        f'there, so this case did not exercise the refusal')
        self.assertEqual('/root/aim344-secret', state[destination]['target'])

    def test_a_symlinked_config_enroot_is_refused_with_the_parent_valid(self):
        """The `.config/enroot` branch, actually reached.

        `/tmp/enroot` is left absent so `prepare_shared_parent` creates it and
        succeeds, and the run gets as far as the participant's home. This is the case
        the review asked for: the combined layout above exits before this point.
        """
        self._assert_home_destination_refused(
            'case3a', '/home/aim344-t1/.config/enroot',
            reached_marker='/tmp/enroot created', component='enroot')
        # `.config` itself is the participant's own real directory and is untouched.
        self.assertFalse(self.records['case3a']['/home/aim344-t1/.config']['symlink'])

    def test_a_symlinked_participant_ssh_is_refused_with_the_parent_valid(self):
        """The participant `.ssh` branch, with everything before it valid."""
        self._assert_home_destination_refused(
            'case3b', '/home/aim344-t1/.ssh',
            reached_marker='/home/aim344-t1/.config/enroot mode 0o755',
            component='.ssh')

    def test_a_symlinked_results_directory_is_refused(self):
        """The `aim344-results` branch, which is prepared last.

        Everything before it is valid in this case -- the shared parent, both home
        directories, the key pair and the control user's `authorized_keys` -- so
        reaching it at all is part of what the assertion shows.
        """
        self._assert_home_destination_refused(
            'case3c', '/home/aim344-t1/aim344-results',
            reached_marker='Wrote table-1 into /etc/aim344-device-session.json',
            component='aim344-results')

    def test_the_rejected_construction_would_have_changed_it(self):
        """The negative control: the same layout, the pre-repair primitive.

        Without this, the test above could be passing because the layout is
        harmless rather than because the code is fixed.
        """
        state = self.records['rejected']
        secret = state['/root/aim344-secret']
        self.assertNotEqual(
            0, secret['uid'],
            'the pre-repair `install -d -o participant` did NOT change the '
            'privileged directory, so this suite has no negative control')
        self.assertEqual(0, self.records.get('rejected_rc'),
                         'the pre-repair primitive is expected to report success')

    def test_the_rejected_primitive_would_have_changed_the_other_destinations(self):
        """The same control for the two home paths the new cases exercise.

        Each of `.ssh` and `aim344-results` gets the exact `install -d -o … -m …`
        line the rejected installer used, on the layout its own case uses. Both
        report success and both change the privileged decoy, so the refusals above
        are refusals of an operation that really was dangerous.
        """
        for case, destination in (('rejected-ssh', '/home/aim344-t1/.ssh'),
                                  ('rejected-results',
                                   '/home/aim344-t1/aim344-results')):
            with self.subTest(case=case):
                self.assertEqual(0, self.records.get(f'{case}_rc'),
                                 f'the pre-repair primitive for {destination} did '
                                 f'not report success')
                secret = self.records[case]['/root/aim344-secret']
                self.assertNotEqual(
                    0, secret['uid'],
                    f'the pre-repair primitive for {destination} did NOT change the '
                    f'privileged directory, so that case has no negative control')

    def test_a_second_key_line_does_not_reach_authorized_keys(self):
        """A participant-writable public key file is untrusted input."""
        lines = self.records['case4'].get('authorized_keys_lines', [])
        self.assertNotIn(
            'INJECTEDKEYMATERIAL', ' '.join(lines),
            'a second line from the participant-writable .pub file was installed')
        self.assertNotEqual(0, self.records.get('case4_rc'),
                            'the installer accepted a two-line public key file')

    def test_a_foreign_owned_shared_parent_is_refused_not_relabelled(self):
        """The shared /tmp parent: report it, do not chmod another user's directory."""
        state = self.records['case5']
        parent = state['/tmp/enroot']
        self.assertTrue(parent['exists'])
        self.assertEqual(65534, parent['uid'],
                         'the installer took ownership decisions on a directory '
                         'another local user created')
        self.assertEqual('0o700', parent['mode'],
                         'the installer widened another local user\'s directory')
        self.assertNotEqual(0, self.records.get('case5_rc'))
        self.assertIn('owned by uid', self.records.get('case5_err', '')
                      + self.records.get('case5_out', ''))

    def test_the_shared_parent_extraction_actually_reads_enroot_conf(self):
        """A skipped preparation cannot be a passing one.

        The rejected extraction used `|` as both the s-command delimiter and the
        intended alternation, so `\\|` was an escaped delimiter and the pattern
        only matched the literal `ENROOT_RUNTIME|CACHE|DATA_PATH`. Against a real
        configuration it printed nothing, and the installer said "no enroot paths
        configured; skipping" -- which would make every shared-parent assertion in
        this class vacuous.
        """
        output = self.records.get('case1_out', '')
        self.assertNotIn('no enroot paths configured', output,
                         'the installer did not read the enroot configuration, so '
                         'the shared-parent path was never exercised')
        self.assertIn('/tmp/enroot', output)
        # Shared cache/data intermediates must also be prepared; per-identity
        # paths remain exclusively the user's responsibility.
        for expected in ('/tmp/enroot/cache', '/tmp/enroot/data'):
            self.assertIn(expected, output)
        for wrong in ('user-', 'group-'):
            with self.subTest(wrong=wrong):
                self.assertNotIn(f'{wrong} created', output)
        self.assertTrue(self.records['case1']['/tmp/enroot']['exists'])
        self.assertEqual('0o1777', self.records['case1']['/tmp/enroot']['mode'])
        self.assertEqual(0, self.records['case1']['/tmp/enroot']['uid'])

    def test_no_case_leaves_the_privileged_decoy_altered(self):
        """After each refusal, the decoy is still root-owned and 0700.

        Each case resets the decoy first, so this reads one case at a time rather
        than the accumulated end state.
        """
        for case in ('case1', 'case2', 'case3', 'case3a', 'case3b', 'case3c',
                     'case4', 'case5'):
            with self.subTest(case=case):
                state = self.records[case]
                self.assertEqual(0, state['/root/aim344-secret']['uid'])
                self.assertEqual('0o700', state['/root/aim344-secret']['mode'])
                self.assertEqual(0, state['/root/aim344-secret/inside']['uid'])
                self.assertEqual('0o700', state['/root/aim344-secret/inside']['mode'])


class InstallerSource(unittest.TestCase):
    """The audit S1 asked for: no privileged write through a participant path.

    A source assertion cannot establish behaviour, which is why the container
    tests above exist. What it can do is fail if a future edit reintroduces the
    construction anywhere in the file, including a line the review did not name.
    """

    # Tokens that mean "this path is, or is under, a directory an unprivileged
    # account controls". `/root/.ssh` and `/etc/...` are root-owned throughout and
    # are not reachable by such an account, so they are not in scope.
    UNTRUSTED = ('participant_home', 'control_home', '.config', '$parent',
                 'getent passwd')

    @classmethod
    def logical_lines(cls, text):
        """Backslash continuations joined, so a wrapped command is one line.

        The first version of this detector read physical lines and therefore
        missed `install -d -o "$participant" ... \\` with the destination on the
        following line -- which is the exact shape the rejected installer used.
        Its own test caught that, which is the point of keeping the negative case
        below.
        """
        joined = []
        pending = ''
        start = 0
        for number, line in enumerate(text.splitlines(), 1):
            if pending:
                pending += ' ' + line.strip()
            else:
                start = number
                pending = line
            if pending.rstrip().endswith('\\'):
                pending = pending.rstrip()[:-1].rstrip()
                continue
            joined.append((start, pending))
            pending = ''
        if pending:
            joined.append((start, pending))
        return joined

    @classmethod
    def offending(cls, line):
        stripped = line.strip()
        if stripped.startswith('#') or 'install -d' not in stripped:
            return False
        if any(token in stripped for token in cls.UNTRUSTED):
            return True
        return '.ssh' in stripped and '/root/.ssh' not in stripped

    def test_no_install_d_names_a_home_or_shared_temporary_directory(self):
        offenders = [f'{number}: {line.strip()}'
                     for number, line in self.logical_lines(INSTALLER.read_text())
                     if self.offending(line)]
        self.assertEqual([], offenders,
                         'privileged install -d through an account-controlled path')

    def test_the_detector_above_catches_the_rejected_lines(self):
        """The detector is run on the lines it must catch, not only on a clean file.

        Without this the assertion above could pass because the detector never
        matches anything -- and in fact it did miss the wrapped form until this
        test failed on it.
        """
        rejected = (
            'install -d -o "$participant" -g "$participant" -m 0755 \\\n'
            '    "$(getent passwd "$participant" | cut -d: -f6)/.config/enroot"\n',
            'install -d -o "$participant" -g "$participant" -m 0700 "$participant_home/.ssh"\n',
            'install -d -m 1777 "$parent"\n',
            'install -d -o "$control_user" -g "$control_user" -m 0700 \\\n'
            '        "$(getent passwd "$control_user" | cut -d: -f6)/.ssh"\n',
        )
        for text in rejected:
            with self.subTest(text=text.splitlines()[0]):
                caught = any(self.offending(line)
                             for _, line in self.logical_lines(text))
                self.assertTrue(caught, f'the detector would have missed: {text}')

    def test_the_detector_accepts_the_legitimate_root_only_lines(self):
        """A detector that flags everything is as useless as one that flags nothing."""
        for text in ('install -d -o root -g root -m 0700 /root/.ssh\n',
                     'install -d -m 0755 /etc/sudoers.d\n',
                     'install -d -o root -g root -m 0700 "$state_dir"\n',
                     '# install -d -o "$participant" ... was the rejected form\n'):
            with self.subTest(text=text.strip()):
                caught = any(self.offending(line)
                             for _, line in self.logical_lines(text))
                self.assertFalse(caught, f'the detector rejects: {text}')

    def test_the_owned_and_shared_preparations_are_the_only_creators(self):
        text = INSTALLER.read_text()
        self.assertIn('prepare_owned_dir', text)
        self.assertIn('prepare_shared_parent', text)
        self.assertIn('O_NOFOLLOW', text)
        # Each account-owned directory the installer needs is prepared as that
        # account, named here so a dropped call is visible.
        for call in ("prepare_owned_dir \"$participant\" \\\n    \"$(getent passwd \"$participant\" | cut -d: -f6)\" '.config/enroot' 0755",
                     "prepare_owned_dir \"$participant\" \"$participant_home\" '.ssh' 0700",
                     "prepare_owned_dir \"$participant\" \"$participant_home\" 'aim344-results' 0700",
                     "prepare_owned_dir \"$control_user\" \"$control_home\" '.ssh' 0700"):
            with self.subTest(call=call.split('\n')[0]):
                self.assertIn(call, text)


# ---------------------------------------------------------------------------
# S2: a post-recovery WARN is qualified against the round's own pre-fault
# warning CONDITIONS, exactly, and only check 6's two documented cumulative
# counters are read as quantities.
# ---------------------------------------------------------------------------

# The exact lines the DEPLOYED suite produces. Captured by running
# `gpu-healthcheck.sh --check 6 --verbose` on the target through the maintenance
# route, recorded verbatim in
# participant-revision/runs/rework7-read-warn-format/output.log:
#
#   stdout  '[WARN] 6-efa-loopback: EFA loopback completed for 2 domain(s); ...'
#   stderr  '[WARN] EFA retransmission timeouts detected (2)'
#
# and zero ESC bytes in either stream. The controller concatenates stdout and
# stderr, so both lines reach it. The aggregate sentence is identical whether the
# condition is drops or retransmissions (checks/6-efa-loopback.sh:194-195 at the
# pinned revision), which is precisely why the label cannot distinguish them.
WARN6_AGGREGATE = ('[WARN] 6-efa-loopback: EFA loopback completed for 2 '
                   'domain(s); cumulative EFA statistics contain drops or '
                   'retransmission timeouts (see logs)\n')
# The two per-condition `log_warn` lines, quoted from the PINNED suite rather than
# paraphrased: checks/6-efa-loopback.sh:171 and :175 at
# a4ba07eb15e6f277063b4000346f9109c98de843. The rx_drops line carries the
# ' -- possible network issues' tail, which the earlier constants here dropped.
WARN6_RETRANS_2 = WARN6_AGGREGATE + '[WARN] EFA retransmission timeouts detected (2)\n'
WARN6_RETRANS_9 = WARN6_AGGREGATE + '[WARN] EFA retransmission timeouts detected (9)\n'
WARN6_RETRANS_1 = WARN6_AGGREGATE + '[WARN] EFA retransmission timeouts detected (1)\n'
WARN6_DROPS_4 = (WARN6_AGGREGATE
                 + '[WARN] EFA rx_drops detected (4) -- possible network issues\n')
WARN6_BOTH = (WARN6_AGGREGATE
              + '[WARN] EFA rx_drops detected (4) -- possible network issues\n'
              + '[WARN] EFA retransmission timeouts detected (2)\n')
# The two check-6 PASS shapes, which differ ONLY in the statistics step's own DEBUG
# line. Both are quoted from the pinned suite: the verdict sentence is
# checks/6-efa-loopback.sh@a4ba07eb:197 (reached whenever stats_warning is 0), and the
# DEBUG lines are :179 for a successful clean read and :182/:185 for the two branches
# where `rdma -p statistic show` failed or the tool was absent, all of which leave
# stats_warning=0. log_verbose prints them to stderr under VERBOSE=1
# (lib/common.sh:49-53), which is how maintenance.collect runs the suite, and the
# controller concatenates stdout and stderr. A real captured check-2 run over the
# maintenance route carries its own DEBUG line in exactly that way
# (participant-revision/runs/rework7-read-warn-format/output.log:107,
# '[DEBUG] Memory lock limit OK'), which is the stored evidence that this stream
# reaches the controller.
#
# A check-6 capture with no statistics line at all is a THIRD shape, used by fixtures
# whose subject is not the statistics step; it is an incomplete observation for a round
# whose baseline reported a counter condition, and the controller treats it as one.
PASS6_STATISTICS_CLEAN = (
    _rework.PASS_6 + '[DEBUG] EFA statistics clean -- no drops or retransmissions\n')
PASS6_STATISTICS_SKIPPED = (
    _rework.PASS_6 + '[DEBUG] rdma statistic show failed -- EFA statistics skipped\n')
# A check-6 warning line this helper does not recognise as either counter. The
# suite has other `log_warn` lines in that check (:48, :63, :83), so this shape is
# not hypothetical; what matters here is that an unrecognised line is compared as
# a condition rather than mined for numbers.
WARN6_UNKNOWN = (WARN6_AGGREGATE
                 + '[WARN] Domain rdmap176s0-rdm: fi_pingpong client exit 3\n')

# Check 0, at the same pinned revision. Both of these are WARN-severity Xid
# reports built at checks/0-nvidia-smi-check.sh:239 and printed at :275, where
# `check_warn` gives them the '[WARN] <check>: ' prefix (lib/common.sh:185-190).
# Xid 94 and Xid 13 are BOTH classified MONITOR/RESTART_APP (:219-220), so the two
# sentences differ only in the identifier -- which is exactly why the rejected
# all-digit normalisation made them the same warning with a 'decreased counter'.
WARN0_XID_94 = ('[WARN] 0-nvidia-smi: Found 1 Xid message(s): '
                'Xid 94 (MONITOR/RESTART_APP)\n')
WARN0_XID_13 = ('[WARN] 0-nvidia-smi: Found 1 Xid message(s): '
                'Xid 13 (MONITOR/RESTART_APP)\n')
# The ECC/retired-pages warning from the same check, :373 -> :379. It carries a GPU
# INDEX, a count and a THRESHOLD in one sentence. Only the GPU index differs
# between these two, so under digit normalisation they were the same warning.
WARN0_ECC_GPU_3 = ('[WARN] 0-nvidia-smi: ECC/retired pages: GPU 3: 61 SBE '
                   'retired pages (threshold: 60); \n')
WARN0_ECC_GPU_5 = ('[WARN] 0-nvidia-smi: ECC/retired pages: GPU 5: 61 SBE '
                   'retired pages (threshold: 60); \n')
# Check 2, :189-191 at the pinned revision, printed through `warn_efa` -> check_warn
# (:21). The condition is `memlock -lt 16777216`, so a LOWER limit is WORSE. Under
# 'no number rose' the second of these was accepted as equivalent to the first.
WARN2_MEMLOCK_8192 = ('[WARN] 2-efa-enumeration: Memory lock limit 8192 KB is '
                      'below 16 GiB -- EFA performance may be degraded\n')
WARN2_MEMLOCK_1024 = ('[WARN] 2-efa-enumeration: Memory lock limit 1024 KB is '
                      'below 16 GiB -- EFA performance may be degraded\n')
# The COMPLETE check-2 advisory output, which is what a real run prints: `warn_efa`
# emits the advisory through check_warn AND increments `advisory_count`, and the final
# branch then prints the aggregate with that count
# (checks/2-efa-enumeration.sh@a4ba07eb, `warn_efa` and the `advisory_count -gt 0`
# branch). The earlier fixtures above carry only the detail line, which is why the
# review asked for a positive control holding both.
WARN2_ADVISORY_AGGREGATE_1 = ('[WARN] 2-efa-enumeration: EFA devices enumerated with '
                              '1 advisories; inspect raw output\n')
WARN2_MEMLOCK_COMPLETE = WARN2_MEMLOCK_8192 + WARN2_ADVISORY_AGGREGATE_1
# The same aggregate with none of the advisories it counts. A capture can look like
# this when the detail did not survive into the round's record, and it is an incomplete
# observation of conditions the sentence itself says exist.
WARN2_ADVISORY_SUMMARY_ONLY = WARN2_ADVISORY_AGGREGATE_1
# Check 3's two NVLink warnings, at the same pinned revision. The first sums Replay,
# Recovery and CRC errors ACROSS links (`awk -F: '{sum += $NF}'` per class), so it is an
# accumulation naming no link; the second is the check reporting that it could not read
# the counters at all (`log_warn` on the `nvidia-smi nvlink -e` failure branch), which
# is the absence of an observation rather than an observation of a condition.
WARN3_NVLINK_ERRORS = ('[WARN] 3-topology-check: NVLink errors detected: 4 replay, '
                       '0 recovery, 2 CRC errors across links\n')
WARN3_NVLINK_UNAVAILABLE = ('[WARN] nvidia-smi nvlink -e failed -- NVLink error '
                            'counters not available\n')


class S2WarningQualification(Base):
    """The recovery decision, driven through the real `start`/`recover` verbs.

    Each test runs a real round on the recorded fake executor: `start efa` captures
    the round's own baseline from that executor's check output, then `recover`
    judges the post-recovery output. Nothing hand-writes a baseline record, so the
    comparison is on what the controller itself stored.
    """

    def qualify(self, baseline_text, recovered_text, check='6', kind='efa'):
        """One round: baseline says `baseline_text`, recovery says `recovered_text`.

        Returns (result-or-None, refusal-or-None) plus the recorded state, so a
        test can assert both the decision and what the node was left in.
        """
        self.executor.check_output[check] = baseline_text
        self.make().start(kind)
        self.executor.check_output[check] = recovered_text
        try:
            result = self.make().recover()
        except session.Refusal as refusal:
            return None, refusal
        return result, None

    def test_an_unchanged_memlock_warning_requires_replacement(self):
        """Final finding 5: this source-shaped fixture is not supported evidence."""
        result, refusal = self.qualify(WARN2_MEMLOCK_COMPLETE, WARN2_MEMLOCK_COMPLETE,
                                      check='2')
        self.assertIsNone(result)
        self.assertIsNotNone(refusal)
        self.assertEqual('replacement-required', self.make().store.read().phase)
        self.assertEqual([], self.executor.resumes())
        self.assertIn('auto-qualification is disabled', str(refusal))

    def test_a_different_condition_under_the_same_label_keeps_the_node_drained(self):
        """S2's original case: retransmissions before, drops after. Same WARN label.

        Both observations report a cumulative counter, so this is refused on the
        continuity rule; the point preserved here is that the node is NOT returned to
        service when the condition behind one WARN label changed.
        """
        result, refusal = self.qualify(WARN6_RETRANS_2, WARN6_DROPS_4)
        self.assertIsNone(result)
        assert refusal is not None
        self.assertIn('cumulative EFA counter', str(refusal))
        self.assertEqual([], self.executor.resumes(),
                         'the node was returned to service on a new warning')
        self.assertEqual('replacement-required', self.make().store.read().phase)

    def test_an_additional_condition_keeps_the_node_drained(self):
        """Retransmissions before; retransmissions AND drops after."""
        result, refusal = self.qualify(WARN6_RETRANS_2, WARN6_BOTH)
        self.assertIsNone(result)
        assert refusal is not None
        self.assertEqual([], self.executor.resumes())

    def test_a_risen_counter_keeps_the_node_drained(self):
        """Same condition, worse: 2 retransmissions before, 9 after."""
        result, refusal = self.qualify(WARN6_RETRANS_2, WARN6_RETRANS_9)
        self.assertIsNone(result)
        assert refusal is not None
        self.assertIn('cumulative EFA counter', str(refusal))
        self.assertEqual([], self.executor.resumes())

    def test_a_fallen_counter_also_keeps_the_node_drained(self):
        """The review's S2 point, on the one counter that IS a counter.

        The rejected code accepted any number that had not risen. A fall means the
        accumulation started again, and the current rule is stronger: an EQUAL count
        does not establish continuity either, because the accumulation epoch is the
        instance launch or the last driver reset and this round rebinds the driver.
        """
        result, refusal = self.qualify(WARN6_RETRANS_2, WARN6_RETRANS_1)
        self.assertIsNone(result)
        assert refusal is not None
        self.assertIn('cumulative EFA counter', str(refusal))
        self.assertEqual([], self.executor.resumes())

    def test_a_dropped_counter_line_keeps_the_node_drained(self):
        """Both conditions before, only one named after.

        Nothing new appeared and nothing rose, so the rejected comparison saw no
        problem.
        """
        result, refusal = self.qualify(WARN6_BOTH, WARN6_RETRANS_2)
        self.assertIsNone(result)
        assert refusal is not None
        self.assertEqual([], self.executor.resumes())

    # -- S2: a cumulative EFA counter is never auto-qualified --
    def test_an_unchanged_counter_warning_is_refused_and_says_why(self):
        """The fatal finding: equality is not continuity.

        Identical text before and after, on the SAME recorded boot, with a successful
        rebind and no reboot -- everything the rejected predicate checked. It must
        still refuse, because the counters restart at the instance launch or at the
        last EFA driver reset and this round rebound that driver
        (device-fault.sh:143-146). The refusal must also say what it does not mean:
        stricter rejection is not a finding that the hardware is viable or faulty.
        """
        result, refusal = self.qualify(WARN6_RETRANS_2, WARN6_RETRANS_2)
        self.assertIsNone(result, 'an equal counter was accepted as continuous')
        assert refusal is not None
        message = str(refusal)
        self.assertIn('cumulative EFA counter', message)
        self.assertIn('retrans_timeout_events=2', message)
        self.assertIn('last EFA driver reset', message)
        self.assertIn('cannot be resumed automatically', message)
        self.assertIn('not a finding that the hardware is faulty', message)
        self.assertEqual([], self.executor.resumes())
        self.assertEqual('replacement-required', self.make().store.read().phase)

    def test_no_reset_epoch_is_invented_from_the_state_record(self):
        """The refusal must not become 'unless the boot id matches'.

        The rejected repair compared the baseline's cached boot id with the cached
        state boot id, both written from the same pre-start read, so a boot outside
        this controller's recorded request path left them equal. Here they are equal
        AND the round recorded no reboot -- the most favourable state the old predicate
        could see -- and the decision is still a refusal.
        """
        self.executor.check_output['6'] = WARN6_RETRANS_2
        self.make().start('efa')
        state = self.make().store.read()
        recorded = state.baseline_check_results['round-1/baseline/check-6']
        self.assertEqual(state.boot_id, recorded.get('boot_id'),
                         'this test needs the two ids equal to have any force')
        self.assertFalse(state.reboot_completed)
        handler = self.make()
        allowed, detail = handler._baseline_allows_warning(
            state, '6', WARN6_RETRANS_2)
        self.assertFalse(allowed, detail)
        self.assertIn('cumulative EFA counter', detail)

    def test_the_refusal_does_not_depend_on_the_recorded_boot_id_at_all(self):
        """Deleting the recorded boot id changes nothing.

        A decision that still consulted it would behave differently here. This is the
        assertion that the epoch is not being inferred from state the controller
        happens to hold.
        """
        self.executor.check_output['6'] = WARN6_RETRANS_2
        self.make().start('efa')
        state = self.make().store.read()
        for spelling in (None, 'a-different-boot'):
            with self.subTest(boot_id=spelling):
                state.baseline_check_results[
                    'round-1/baseline/check-6']['boot_id'] = spelling
                handler = self.make()
                allowed, detail = handler._baseline_allows_warning(
                    state, '6', WARN6_RETRANS_2)
                self.assertFalse(allowed)
                self.assertIn('cumulative EFA counter', detail)

    def test_a_clean_pass_after_recovery_still_resumes(self):
        """The counter rule must not make a healthy EFA round unrecoverable.

        The ordinary case: the pre-fault run warned about a counter, and after recovery
        check 6 reads those counters again and finds them clean, so there is no warning
        to qualify and the round resumes. This is what keeps 'refuse every counter
        warning' from meaning 'no EFA round can finish'.

        The recovery capture carries the check's own clean-statistics sentence, which is
        what a real clean run prints and what this test previously left out. That
        matters because it is the only thing in the capture that distinguishes a run
        which read the counters from one whose statistics step never ran: the pinned
        check emits the same '[PASS] 6-efa-loopback: EFA loopback OK' either way
        (checks/6-efa-loopback.sh@a4ba07eb:179 against :182,185, all three falling
        through to check_pass at :197). A verdict-only fixture therefore describes an
        incomplete observation, and the controller now refuses to qualify one against a
        recorded counter condition -- see the statistics-skipped case immediately below,
        which is the same round with the same verdict and no counter observation.
        """
        self.executor.check_output['6'] = WARN6_RETRANS_2
        self.make().start('efa')
        self.executor.check_output['6'] = PASS6_STATISTICS_CLEAN
        result = self.make().recover()
        self.assertEqual('runtime-ready', result.state.phase)
        self.assertEqual(1, len(self.executor.resumes()))

    def test_a_statistics_skipped_pass_does_not_qualify_a_counter_baseline(self):
        """The complement, and the reason the control above needs its DEBUG line.

        Same round, same PASS verdict, same loopback sentence -- but the statistics step
        reports that it was skipped, so the counters this round's baseline warned about
        were not re-read. A PASS whose own text says the observation did not happen
        cannot establish that the recorded condition is unchanged, and the node keeps
        its drain instead of resuming on a label.
        """
        self.executor.check_output['6'] = WARN6_RETRANS_2
        self.make().start('efa')
        self.executor.check_output['6'] = PASS6_STATISTICS_SKIPPED
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        message = str(caught.exception)
        self.assertIn('statistics collection was skipped', message)
        self.assertIn('not a finding that the hardware is faulty', message)
        self.assertEqual([], self.executor.resumes())
        self.assertEqual('recovery-failed', self.make().store.read().phase)

    # -- S2-B: the aggregate on its own is an incomplete observation --
    def test_two_identical_aggregate_only_captures_do_not_qualify_each_other(self):
        """The S2-B finding, on the qualification decision itself.

        Both observations hold only the check-6 aggregate, so the condition sets are
        identical and both counter maps are empty. The rejected code skipped every
        counter and boot check and returned True. The aggregate asserts that a
        cumulative-statistics condition exists -- `stats_warning` is set only by the two
        per-counter log_warn lines -- and names neither, so this is an incomplete
        observation of a condition it says exists, not an observation of the same one.
        """
        result, refusal = self.qualify(WARN6_AGGREGATE, WARN6_AGGREGATE)
        self.assertIsNone(result, 'aggregate-only captures qualified each other')
        assert refusal is not None
        message = str(refusal)
        self.assertIn('without reporting either counter', message)
        self.assertIn('see logs', message)
        self.assertEqual([], self.executor.resumes())
        self.assertEqual('replacement-required', self.make().store.read().phase)

    def test_the_aggregate_only_refusal_is_the_decision_not_the_parse(self):
        """Asserted on `_baseline_allows_warning`, and on equal parses.

        The old test only checked that parsing the aggregate twice gave the same
        result, which it does and always did. What was missing was any assertion about
        the DECISION. So this asserts both: the parses are equal, and the decision is
        still a refusal.
        """
        first = session.DeviceSession.warning_details(WARN6_AGGREGATE)
        second = session.DeviceSession.warning_details(WARN6_AGGREGATE)
        self.assertEqual(first, second)
        self.assertEqual([], [item for item in first if item['counter']],
                         'the aggregate was read as a counter')
        self.executor.check_output['6'] = WARN6_AGGREGATE
        self.make().start('efa')
        handler = self.make()
        allowed, detail = handler._baseline_allows_warning(
            handler.store.read(), '6', WARN6_AGGREGATE)
        self.assertFalse(allowed, detail)
        self.assertIn('without reporting either counter', detail)

    def test_an_aggregate_only_recovery_against_a_complete_baseline_refuses(self):
        """One side incomplete is still incomplete.

        The baseline named the counter; the post-recovery capture reports only the
        aggregate. The conditions differ, so this was already refused -- asserted here
        so the aggregate rule is exercised in both directions rather than only on the
        symmetric case.
        """
        result, refusal = self.qualify(WARN6_RETRANS_2, WARN6_AGGREGATE)
        self.assertIsNone(result)
        assert refusal is not None
        self.assertEqual([], self.executor.resumes())

    def test_a_complete_aggregate_plus_counter_capture_is_not_refused_as_incomplete(self):
        """The aggregate rule must not fire when the detail IS present.

        The aggregate accompanies the counter line in the real output
        (runs/rework7-read-warn-format/output.log:18), so refusing every capture that
        contains the aggregate would refuse every real check-6 warning for the wrong
        reason. Such a capture is refused on the COUNTER rule, and the message says so.
        """
        result, refusal = self.qualify(WARN6_RETRANS_2, WARN6_RETRANS_2)
        self.assertIsNone(result)
        assert refusal is not None
        self.assertIn('cumulative EFA counter', str(refusal))
        self.assertNotIn('without reporting either counter', str(refusal))

    def test_a_check6_warning_line_this_helper_cannot_read_is_refused(self):
        """An unrecognised warning is not a counter and not a match.

        It is compared as a condition, so an unrecognised line that the pre-fault
        run did not produce refuses -- rather than being mined for numbers, which is
        how a `fi_pingpong client exit 3` line became 'a counter that fell'.
        """
        result, refusal = self.qualify(WARN6_RETRANS_2, WARN6_UNKNOWN)
        self.assertIsNone(result)
        assert refusal is not None
        self.assertIn('fi_pingpong client exit 3', str(refusal))
        self.assertEqual([], self.executor.resumes())

    # -- the two cases the review named, on the pinned suite's own text --
    def test_a_changed_xid_identifier_keeps_the_node_drained(self):
        """The review's first S2 counterexample, run through a real GPU round.

        Xid 94 before the fault, Xid 13 after. Both are MONITOR/RESTART_APP at the
        pinned revision, so both sentences reach the controller as WARN with the
        same shape once digits are normalised, and the rejected comparison read the
        changed identifier as a counter that had decreased. It is a different fault
        condition and the node must stay out of service.
        """
        result, refusal = self.qualify(WARN0_XID_94, WARN0_XID_13,
                                       check='0', kind='gpu')
        self.assertIsNone(result)
        assert refusal is not None
        self.assertIn('not the one the pre-fault run produced', str(refusal))
        self.assertIn('Xid 13', str(refusal))
        self.assertEqual([], self.executor.resumes(),
                         'a node resumed on a different Xid than it started with')
        self.assertEqual('replacement-required', self.make().store.read().phase)

    def test_a_changed_gpu_identity_keeps_the_node_drained(self):
        """The same defect on a device identifier rather than a fault code.

        The ECC warning names the GPU index. GPU 5 before, GPU 3 after, with an
        identical page count and threshold: a different device reporting the same
        condition, which the pre-fault observation says nothing about. The direction
        matters for the negative control below -- under digit normalisation the index
        was a counter, so 5 -> 3 was "a counter that decreased" and was admitted.
        """
        result, refusal = self.qualify(WARN0_ECC_GPU_5, WARN0_ECC_GPU_3,
                                       check='0', kind='gpu')
        self.assertIsNone(result)
        assert refusal is not None
        self.assertIn('GPU 3', str(refusal))
        self.assertEqual([], self.executor.resumes())

    def test_a_worsened_limit_keeps_the_node_drained(self):
        """The review's second S2 counterexample, on check 2's memory-lock limit.

        The condition is `memlock -lt 16777216`, so 1024 KB is worse than 8192 KB.
        No number rose, and the rejected comparison accepted it.
        """
        result, refusal = self.qualify(WARN2_MEMLOCK_8192, WARN2_MEMLOCK_1024,
                                       check='2', kind='efa')
        self.assertIsNone(result)
        assert refusal is not None
        self.assertIn('1024 KB', str(refusal))
        self.assertEqual([], self.executor.resumes(),
                         'a node resumed on a memory-lock limit that got worse')
        self.assertEqual('replacement-required', self.make().store.read().phase)

    def test_an_unchanged_limit_and_xid_warning_both_refuse(self):
        """Final finding 5 removes unproven memlock acceptance, not Xid refusal."""
        result, refusal = self.qualify(WARN2_MEMLOCK_COMPLETE, WARN2_MEMLOCK_COMPLETE,
                                       check='2', kind='efa')
        self.assertIsNone(result)
        self.assertIsNotNone(refusal)
        self.assertEqual('replacement-required', self.make().store.read().phase)
        self.assertEqual([], self.executor.resumes())

        self.setUp()
        result, refusal = self.qualify(WARN0_XID_94, WARN0_XID_94,
                                       check='0', kind='gpu')
        self.assertIsNone(result, 'an equal Xid summary resumed the node')
        assert refusal is not None
        self.assertIn('will not qualify automatically', str(refusal))
        self.assertEqual([], self.executor.resumes())

    def test_the_rejected_normalisation_would_have_resumed_on_all_of_them(self):
        """The negative control, on the same recorded observations.

        Reconstructs the REJECTED comparison -- every digit run replaced by '#', the
        numbers kept as counters, a value accepted unless it rose -- and evaluates it
        on each pair the tests above refuse. It admits every one, including the two
        the review named. Without this, those refusals could be passing for an
        unrelated reason.
        """
        number = re.compile(r'\d+')
        line = re.compile(r'\[WARN\]\s+(?:(\d+-[a-z0-9-]+):\s*)?(.*\S)\s*$')

        def rejected_details(text):
            out = []
            for row in (text or '').splitlines():
                found = line.search(row)
                if not found:
                    continue
                name, body = found.group(1), found.group(2)
                prefix = f'[WARN] {name}: ' if name else '[WARN] '
                out.append((prefix + number.sub('#', body),
                            [int(n) for n in number.findall(body)]))
            return out

        def rejected_allows(before_text, after_text):
            before, after = {}, {}
            for shape, counters in rejected_details(before_text):
                before[shape] = max(before.get(shape, counters), counters)
            for shape, counters in rejected_details(after_text):
                after[shape] = max(after.get(shape, counters), counters)
            if any(shape not in before for shape in after):
                return False
            for shape, counters in after.items():
                was = before[shape]
                if len(counters) != len(was):
                    return False
                if any(now > then for now, then in zip(counters, was)):
                    return False
            return True

        cases = (
            ('a changed Xid identifier', WARN0_XID_94, WARN0_XID_13),
            ('a changed GPU identity', WARN0_ECC_GPU_5, WARN0_ECC_GPU_3),
            ('a worsened memory-lock limit', WARN2_MEMLOCK_8192, WARN2_MEMLOCK_1024),
            ('a fallen cumulative counter', WARN6_RETRANS_2, WARN6_RETRANS_1),
            ('a dropped counter line', WARN6_BOTH, WARN6_RETRANS_2),
        )
        handler = self.make()
        for name, before_text, after_text in cases:
            with self.subTest(case=name):
                self.assertTrue(
                    rejected_allows(before_text, after_text),
                    f'the pre-repair comparison did NOT admit {name}, so this '
                    f'suite has no negative control for it')
                # And the repaired parser distinguishes them, on the same text.
                before = session.DeviceSession.warning_details(before_text)
                after = session.DeviceSession.warning_details(after_text)
                self.assertNotEqual(
                    [item['condition'] for item in before]
                    + [item['counter'] for item in before],
                    [item['condition'] for item in after]
                    + [item['counter'] for item in after],
                    f'the repaired parser cannot tell {name} apart')
        # The repaired decision, exercised through the method on a real round for
        # one case of each class, so this control is paired with real refusals.
        self.assertTrue(callable(handler._baseline_allows_warning))

    def test_a_baseline_warn_without_recorded_details_is_not_a_baseline(self):
        """An older or truncated record must fail closed, not permissively."""
        self.executor.check_output['6'] = WARN6_RETRANS_2
        self.make().start('efa')
        state = self.make().store.read()
        # Exactly what a pre-repair state file holds: a WARN label, rc 0, no detail.
        state.baseline_check_results['round-1/baseline/check-6'] = {
            'verdict': 'WARN', 'returncode': 0, 'log': 'baseline-check-6.log'}
        self.make().store.write(state)
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        self.assertIn('warning details were not recorded', str(caught.exception))
        self.assertEqual([], self.executor.resumes())

    def test_a_baseline_recorded_in_the_old_normalised_format_is_refused(self):
        """A record written by the rejected helper cannot be compared.

        Its entries hold 'shape' and 'counters', where every number was replaced by
        '#'. Reading those as conditions would compare a normalised sentence with a
        real one, and every such comparison would refuse for the wrong reason -- or,
        worse, a future edit could make them match. It fails closed explicitly.
        """
        self.executor.check_output['6'] = WARN6_RETRANS_2
        self.make().start('efa')
        state = self.make().store.read()
        state.baseline_check_results['round-1/baseline/check-6'] = {
            'verdict': 'WARN', 'returncode': 0, 'log': 'baseline-check-6.log',
            'boot_id': state.boot_id,
            'warnings': [
                {'shape': '[WARN] 6-efa-loopback: EFA loopback completed for # '
                          'domain(s); cumulative EFA statistics contain drops or '
                          'retransmission timeouts (see logs)',
                 'counters': [2]},
                {'shape': '[WARN] EFA retransmission timeouts detected (#)',
                 'counters': [2]}],
        }
        self.make().store.write(state)
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        self.assertIn('format this helper cannot compare', str(caught.exception))
        self.assertEqual([], self.executor.resumes())

    def test_a_warn_whose_warning_lines_are_unreadable_is_refused(self):
        """The verdict says WARN but no comparable warning line arrived.

        Two remarks about honesty here. First, a bare `[WARN]` with nothing after
        it does not reach this branch through a full round: `check_verdict` requires
        `[WARN]\\s+(\\S+)`, so such output has no readable verdict and is refused one
        step earlier. That is asserted below, because it is what the round actually
        does. Second, the uninterpretable-line branch in `_baseline_allows_warning`
        is therefore reached directly on the method here rather than through a round.
        """
        self.executor.check_output['6'] = WARN6_RETRANS_2
        self.make().start('efa')
        self.executor.check_output['6'] = '6-efa-loopback finished\n[WARN]\n'
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        self.assertEqual([], self.executor.resumes())
        self.assertIn('no readable verdict', str(caught.exception))

        handler = self.make()
        allowed, detail = handler._baseline_allows_warning(
            handler.store.read(), '6', '6-efa-loopback finished\n[WARN]\n')
        self.assertFalse(allowed)
        self.assertIn('could not read as a condition', detail)

    def test_the_baseline_record_holds_the_conditions_not_only_the_label(self):
        """What makes the comparison possible, asserted on the stored record."""
        self.executor.check_output['6'] = WARN6_BOTH
        self.make().start('efa')
        recorded = self.make().store.read().baseline_check_results[
            'round-1/baseline/check-6']
        self.assertEqual('WARN', recorded['verdict'])
        conditions = sorted(item['condition'] for item in recorded['warnings'])
        self.assertIn('[WARN] EFA rx_drops detected (4) -- possible network issues',
                      conditions)
        self.assertIn('[WARN] EFA retransmission timeouts detected (2)', conditions)
        counters = {item['counter']['name']: item['counter']['value']
                    for item in recorded['warnings'] if item['counter']}
        self.assertEqual({'rx_drops': 4, 'retrans_timeout_events': 2}, counters)
        # And the boot the observation was made on, which is what makes the counter
        # comparison meaningful at all.
        self.assertEqual(self.make().store.read().boot_id, recorded['boot_id'])

    def test_the_recovered_record_also_holds_the_conditions(self):
        """So the round's record shows which condition was accepted or refused.

        Recorded whatever the decision was, which is why this uses the check-6 capture
        even though it is now refused: the record's job is to show what was observed.
        """
        result, refusal = self.qualify(WARN6_RETRANS_2, WARN6_RETRANS_2)
        self.assertIsNone(result)
        assert refusal is not None
        recorded = self.make().store.read().check_results[
            'round-1/recovered/check-6']
        self.assertEqual('WARN', recorded['verdict'])
        self.assertEqual(
            ['[WARN] 6-efa-loopback: EFA loopback completed for 2 domain(s); '
             'cumulative EFA statistics contain drops or retransmission timeouts '
             '(see logs)',
             '[WARN] EFA retransmission timeouts detected (2)'],
            sorted(item['condition'] for item in recorded['warnings']))
        # And the counter is recorded as a quantity, so the round's evidence shows
        # what could not be qualified rather than only that something was refused.
        self.assertEqual(
            {'retrans_timeout_events': 2},
            {item['counter']['name']: item['counter']['value']
             for item in recorded['warnings'] if item['counter']})

    def test_the_recovered_record_holds_the_unsupported_memlock_condition(self):
        """Final finding 5 refuses memlock but retains the same warning details."""
        result, refusal = self.qualify(WARN2_MEMLOCK_COMPLETE, WARN2_MEMLOCK_COMPLETE,
                                       check='2')
        self.assertIsNone(result)
        self.assertIsNotNone(refusal)
        self.assertEqual([], self.executor.resumes())
        recorded = self.make().store.read().check_results['round-1/recovered/check-2']
        self.assertEqual(
            [line for line in WARN2_MEMLOCK_COMPLETE.splitlines() if line.strip()],
            [item['condition'] for item in recorded['warnings']])

    def test_the_participant_log_is_not_what_authorizes_the_resume(self):
        """The comparison reads the round's own recorded text, not a mutable file.

        The participant owns the log file named in the record and can rewrite it.
        Here the recorded log path is pointed at a file whose content would qualify
        anything, and the decision is unchanged: still refused, because the
        comparison never opens it. Uses check 2's limit warning, so the refusal is
        the changed condition rather than S2's counter rule -- a forged log that had
        been read would have made this pair look identical and resumed the node.
        """
        self.executor.check_output['2'] = WARN2_MEMLOCK_8192
        self.make().start('efa')
        state = self.make().store.read()
        forged = Path(self.state_dir) / 'forged-baseline.log'
        forged.write_text(WARN2_MEMLOCK_1024)
        state.baseline_check_results['round-1/baseline/check-2']['log'] = str(forged)
        self.make().store.write(state)
        self.executor.check_output['2'] = WARN2_MEMLOCK_1024
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        self.assertIn('1024 KB', str(caught.exception))
        self.assertIn('not the one the pre-fault run produced',
                      str(caught.exception))
        self.assertEqual([], self.executor.resumes())


class S2QualifiableConditionsOnly(Base):
    """The review's FATAL finding: equal warning text is not equal evidence.

    The rejected fallback returned True for any two captures whose condition texts
    matched. That is sound only for a condition which describes the node AS IT IS NOW,
    carries its own detail, and is re-read from the system on every run. Most of this
    suite's warnings have none of those properties, and for those two identical captures
    can be two different situations.

    Each case below is the pinned suite's own text, and each drives the real
    `start`/`recover` decision so what is asserted is whether the node was returned to
    service -- not what a parser returned.
    """

    def qualify(self, baseline_text, recovered_text, check, kind):
        self.executor.check_output[check] = baseline_text
        self.make().start(kind)
        self.executor.check_output[check] = recovered_text
        try:
            return self.make().recover(), None
        except session.Refusal as refusal:
            return None, refusal

    def test_an_equal_xid_summary_with_a_new_event_does_not_resume(self):
        """Check 0's Xid summary is an event history reduced to counts and classes.

        `xid_errors` comes from a kernel-log window (`journalctl -k ... -n
        ${KERNEL_LOG_LINES}` matching `NVRM.*Xid`), the codes are deduplicated with
        `sort -un`, and the sentence printed is `Found ${xid_count} Xid message(s):
        ${severity_details}`. No event time and no PCI identity survive into it. So one
        NEW Xid 94 after the reboot produces the same sentence as the pre-fault one,
        and the pre-fault observation says nothing about it.

        The two captures here are character-for-character identical, which is exactly
        the condition the rejected code accepted.
        """
        result, refusal = self.qualify(WARN0_XID_94, WARN0_XID_94,
                                       check='0', kind='gpu')
        self.assertIsNone(result, 'an equal Xid summary returned the node to service')
        assert refusal is not None
        self.assertEqual([], self.executor.resumes())
        self.assertIn('will not qualify automatically', str(refusal))
        self.assertIn('Xid 94', str(refusal))
        # The refusal must not read as a hardware verdict.
        self.assertIn('not a finding that the hardware is faulty', str(refusal))
        self.assertEqual('replacement-required', self.make().store.read().phase)

    def test_a_summary_only_check2_capture_does_not_resume(self):
        """Check 2's advisory aggregate, with none of the advisories it counts.

        `warn_efa` increments `advisory_count` and calls `check_warn` for each advisory,
        and the final branch prints `EFA devices enumerated with ${advisory_count}
        advisories; inspect raw output`. The aggregate therefore asserts detail that is
        not in it, and two captures holding only the aggregate are two incomplete
        observations rather than one condition observed twice.
        """
        result, refusal = self.qualify(WARN2_ADVISORY_SUMMARY_ONLY,
                                       WARN2_ADVISORY_SUMMARY_ONLY,
                                       check='2', kind='efa')
        self.assertIsNone(result, 'a summary-only capture returned the node to service')
        assert refusal is not None
        self.assertEqual([], self.executor.resumes())
        self.assertIn('does not match', str(refusal))

    def test_a_cumulative_nvlink_error_warning_does_not_resume(self):
        """Check 3 sums NVLink replay, recovery and CRC errors ACROSS links.

        `nvidia-smi nvlink -e` is summed per class with `awk -F: '{sum += $NF}'`, so the
        sentence reports accumulated totals and names no link. Nothing this controller
        observes establishes that two equal totals accumulated from the same start, and
        `nvidia-smi` documents a reset operation for exactly these counters.
        """
        result, refusal = self.qualify(WARN3_NVLINK_ERRORS, WARN3_NVLINK_ERRORS,
                                       check='3', kind='gpu')
        self.assertIsNone(result, 'a cumulative NVLink total returned the node')
        assert refusal is not None
        self.assertEqual([], self.executor.resumes())
        self.assertIn('will not qualify automatically', str(refusal))

    def test_an_unavailable_observation_does_not_resume(self):
        """Check 3's own report that it could NOT read the counters.

        `log_warn "nvidia-smi nvlink -e failed -- NVLink error counters not available"`
        is the absence of an observation. Two runs equally unable to look have not
        observed the same condition; they have observed nothing about it, so this cannot
        establish continuity in either direction.
        """
        result, refusal = self.qualify(WARN3_NVLINK_UNAVAILABLE,
                                       WARN3_NVLINK_UNAVAILABLE,
                                       check='3', kind='gpu')
        self.assertIsNone(result, 'an unavailable observation returned the node')
        assert refusal is not None
        self.assertEqual([], self.executor.resumes())
        self.assertIn('will not qualify automatically', str(refusal))

    def test_an_unknown_warning_from_a_future_pin_does_not_resume(self):
        """A sentence this route has never classified is refused, not accepted.

        The whitelist is the decision, so a warning the suite adds later has to be
        judged by a person before it can resume a node. This is the property that makes
        the repair survive a pin advance rather than depending on today's text.
        """
        invented = ('[WARN] 2-efa-enumeration: Some condition this route has never '
                    'classified\n')
        result, refusal = self.qualify(invented, invented, check='2', kind='efa')
        self.assertIsNone(result, 'an unclassified warning returned the node')
        assert refusal is not None
        self.assertEqual([], self.executor.resumes())
        self.assertIn('will not qualify automatically', str(refusal))

    # -- final finding 5 disables the unproven candidate; clean PASS remains --
    def test_the_full_point_in_time_fixture_remains_unsupported(self):
        """Detail plus aggregate is still not a real supported producer capture."""
        result, refusal = self.qualify(WARN2_MEMLOCK_COMPLETE, WARN2_MEMLOCK_COMPLETE,
                                       check='2', kind='efa')
        self.assertIsNone(result)
        self.assertIsNotNone(refusal)
        self.assertEqual('replacement-required', self.make().store.read().phase)
        self.assertEqual([], self.executor.resumes())
        self.assertIn('auto-qualification is disabled', str(refusal))

    def test_a_module_warning_is_no_longer_qualified_at_all(self):
        """The producer's own success is not established, so equality is not evidence.

        This case previously resumed, and the fixture it used could not have come from
        the pinned check: `EFA kernel module gdrdrv not loaded` names gdrdrv in the
        required-module sentence, while the pin's required array is efa, ib_uverbs,
        ib_core and gdrdrv has its own different sentence. Both are corrected here.

        The reason it is refused is a property of the check, not of the sentence: step 5
        is `if ! lsmod | grep -qw "${mod}"`, which takes the warning branch when lsmod
        SUCCEEDS and the module is absent, and equally when lsmod fails or is missing.
        kmod's own `do_lsmod` returns early on `could not get list of modules` (kmod
        v34, tools/lsmod.c), and the check still completes with rc 0 and a consistent
        aggregate, so two captures can agree while both are failed observations.
        """
        module = ('[WARN] 2-efa-enumeration: EFA kernel module efa not loaded\n'
                  '[WARN] 2-efa-enumeration: EFA devices enumerated with 1 '
                  'advisories; inspect raw output\n')
        result, refusal = self.qualify(module, module, check='2', kind='efa')
        self.assertIsNone(result, 'a module warning resumed the node')
        assert refusal is not None
        self.assertIn('cannot confirm succeeded', str(refusal))
        self.assertEqual([], self.executor.resumes())
        self.assertEqual('replacement-required', self.make().store.read().phase)

    def test_a_clean_pass_after_recovery_still_resumes(self):
        """The whitelist must not make a healthy round unrecoverable.

        The pre-fault run warned about an Xid the route will not qualify, and after
        recovery check 0 passes. There is no warning to judge, so the round resumes.
        """
        self.executor.check_output['0'] = WARN0_XID_94
        self.make().start('gpu')
        self.executor.check_output['0'] = _rework.PASS_0
        result = self.make().recover()
        self.assertEqual('runtime-ready', result.state.phase)
        self.assertEqual(1, len(self.executor.resumes()))

    def test_a_changed_qualifiable_condition_still_keeps_the_node_drained(self):
        """The whitelist admits a CONDITION, not a check: equality is still required."""
        result, refusal = self.qualify(WARN2_MEMLOCK_COMPLETE,
                                       WARN2_MEMLOCK_COMPLETE.replace('8192', '1024'),
                                       check='2', kind='efa')
        self.assertIsNone(result)
        assert refusal is not None
        self.assertIn('1024 KB', str(refusal))
        self.assertEqual([], self.executor.resumes())


class S2WarningDetailParsing(unittest.TestCase):
    """`warning_details` on the real captured text, both shapes and neither.

    A parser validated only on the input that must be rejected is half-verified, so
    the accepted shapes are asserted too.
    """

    def test_both_emitter_shapes_are_read(self):
        details = session.DeviceSession.warning_details(WARN6_BOTH)
        self.assertEqual(3, len(details), details)
        conditions = [item['condition'] for item in details]
        # check_warn's '[WARN] <name>: <details>' and log_warn's '[WARN] <details>'
        self.assertTrue(any(condition.startswith('[WARN] 6-efa-loopback:')
                            for condition in conditions))
        self.assertIn('[WARN] EFA rx_drops detected (4) -- possible network issues',
                      conditions)

    def test_a_clean_pass_yields_no_warning(self):
        for text in ('[PASS] 2-efa-enumeration: EFA OK: 2 PCI devices\n',
                     '[SKIP] 6-efa-loopback: fi_pingpong not available\n',
                     '[FAIL] 0-nvidia-smi: Expected 8 GPUs, found 7 '
                     '(severity: ISOLATE)\n',
                     '', 'no verdict here at all\n'):
            with self.subTest(text=text.strip()[:40]):
                self.assertEqual([], session.DeviceSession.warning_details(text))

    def test_the_condition_keeps_its_numbers(self):
        """The core of the S2 repair: identifiers and limits are not erased."""
        for text in (WARN0_XID_94, WARN0_ECC_GPU_3, WARN2_MEMLOCK_8192):
            with self.subTest(text=text.strip()[:50]):
                details = session.DeviceSession.warning_details(text)
                self.assertEqual(1, len(details))
                self.assertEqual(text.strip(), details[0]['condition'])
                self.assertIsNone(details[0]['counter'],
                                  'a number outside check 6 was read as a counter')
        # And the identifiers really do distinguish the pairs the review named.
        self.assertNotEqual(
            session.DeviceSession.warning_details(WARN0_XID_94)[0]['condition'],
            session.DeviceSession.warning_details(WARN0_XID_13)[0]['condition'])
        self.assertNotEqual(
            session.DeviceSession.warning_details(WARN0_ECC_GPU_3)[0]['condition'],
            session.DeviceSession.warning_details(WARN0_ECC_GPU_5)[0]['condition'])
        self.assertNotEqual(
            session.DeviceSession.warning_details(WARN2_MEMLOCK_8192)[0]['condition'],
            session.DeviceSession.warning_details(WARN2_MEMLOCK_1024)[0]['condition'])

    def test_only_the_two_documented_check6_counters_are_read_as_counters(self):
        drops = session.DeviceSession.warning_details(
            '[WARN] EFA rx_drops detected (4) -- possible network issues\n')
        self.assertEqual({'name': 'rx_drops', 'value': 4}, drops[0]['counter'])
        retrans = session.DeviceSession.warning_details(
            '[WARN] EFA retransmission timeouts detected (2)\n')
        self.assertEqual({'name': 'retrans_timeout_events', 'value': 2},
                         retrans[0]['counter'])
        # The aggregate sentence names a domain COUNT, not a counter, and it arrives
        # with a check name in front of it, so it is never read as one.
        aggregate = session.DeviceSession.warning_details(WARN6_AGGREGATE)
        self.assertEqual(1, len(aggregate))
        self.assertIsNone(aggregate[0]['counter'])
        # Nor is a check-6 warning line the helper does not recognise.
        unknown = session.DeviceSession.warning_details(
            '[WARN] Domain rdmap176s0-rdm: fi_pingpong client exit 3\n')
        self.assertEqual(1, len(unknown))
        self.assertIsNone(unknown[0]['counter'])
        # A counter sentence with the same words but a different tail is not the
        # documented line either, so it stays a plain condition.
        altered = session.DeviceSession.warning_details(
            '[WARN] EFA rx_drops detected (4) -- probably fine\n')
        self.assertIsNone(altered[0]['counter'])

    def test_a_counter_line_is_a_counter_under_either_emitter(self):
        """The counter sentence is recognised by its text, not by its emitter.

        This test previously asserted the OPPOSITE -- that a counter line carrying a
        check name is not a counter -- on the reasoning that the counter lines are
        `log_warn` and so never carry one. That is true only of the PINNED suite. The
        same two sentences are emitted by `check_warn "${CHECK_NAME}"` on riv-aim344
        `main` (checks/6-efa-loopback.sh:166,169 at 39848da1), and `check_warn`
        prefixes the check name (lib/common.sh:186-189).

        So the old rule silently disabled the entire S2 refusal against that suite: a
        prefixed counter sentence parsed as counter=None and an unchanged count then
        qualified as an ordinary exact-match condition, auto-resuming the node. The
        invariant that actually matters is that the COUNTER SENTENCE is interpreted as
        a counter however the suite chose to print it.
        """
        for label, line in (
                ('log_warn, pinned suite',
                 '[WARN] EFA rx_drops detected (4) -- possible network issues\n'),
                ('check_warn, upstream suite',
                 '[WARN] 6-efa-loopback: EFA rx_drops detected (4) -- possible '
                 'network issues\n')):
            with self.subTest(emitter=label):
                details = session.DeviceSession.warning_details(line)
                self.assertEqual(1, len(details))
                self.assertEqual({'name': 'rx_drops', 'value': 4},
                                 details[0]['counter'],
                                 f'the counter sentence was not read as a counter '
                                 f'when emitted by {label}')

    def test_the_aggregate_is_still_not_a_counter_under_either_emitter(self):
        """The control for the test above: matching is on the sentence, not the prefix.

        Making the check-name prefix optional must not turn the aggregate into a
        counter. It is a different sentence and matches neither pattern, prefixed or
        not.
        """
        for line in (
                WARN6_AGGREGATE,
                '[WARN] EFA loopback completed for 2 domain(s); cumulative EFA '
                'statistics contain drops or retransmission timeouts (see logs)\n'):
            details = session.DeviceSession.warning_details(line)
            self.assertEqual(1, len(details))
            self.assertIsNone(details[0]['counter'],
                              'the aggregate sentence was read as a counter')
            self.assertTrue(
                session.DeviceSession._claims_cumulative_statistics(details),
                'the aggregate stopped being recognised as the aggregate')

    def test_an_unreadable_warning_line_is_recorded_not_skipped(self):
        """A '[WARN]' with nothing comparable after it must not vanish."""
        details = session.DeviceSession.warning_details(
            '6-efa-loopback finished\n[WARN]\n')
        self.assertEqual(1, len(details))
        self.assertIsNone(details[0]['condition'])
        self.assertIsNone(details[0]['counter'])

    def test_the_aggregate_sentence_alone_cannot_distinguish_the_conditions(self):
        """Why the per-condition lines are required, stated as a test.

        The deployed check prints one aggregate sentence for either condition
        (checks/6-efa-loopback.sh:194-195). On the aggregate line alone, drops and
        retransmissions are indistinguishable -- which is the S2 defect.
        """
        drops = session.DeviceSession.warning_details(WARN6_AGGREGATE)
        self.assertEqual(1, len(drops))
        self.assertEqual(
            session.DeviceSession.warning_details(WARN6_AGGREGATE),
            drops,
            'the aggregate line is identical for both conditions, so the '
            'per-condition log_warn lines are what carry the distinction')


# ---------------------------------------------------------------------------
# S3: an outstanding reboot is reconciled on every attempt, whatever the fault
# kind and whatever the latest rebind returned.
# ---------------------------------------------------------------------------


class S3EfaRebootReconciliation(Base):
    """The EFA branch the R4 tests never entered.

    R4's tests all start with `start('gpu')`, where `needs_reboot` is
    unconditionally true, so the reconciliation inside `if needs_reboot:` always
    ran. On an EFA round `needs_reboot` comes from the latest rebind's outcome, so a
    rebind that succeeds on the retry skipped reconciliation altogether. That branch
    is reachable in reality because the device helper's rebind succeeds when the
    device is already bound (device-fault.sh:145-147), which is the state a
    completed reboot leaves behind.

    Each test below drives a real `start efa` and then real `recover` attempts on
    the recorded fake executor.
    """

    def reach_an_efa_reboot_whose_completion_was_missed(self, came_back):
        """Historical issued request, retained for supported reconciliation.

        Final finding 2 forbids newly issuing this via failed EFA rebind. Seed the
        persisted request and recording scheduler explicitly instead; the recovery
        under test must still reconcile it. This fixture is not fallback approval.
        """
        handler = self.make()
        handler.start('efa')
        held = self.executor.boot_id
        state = handler.store.read()
        state.reboot_requests = 1
        state.reboot_requested_at = 900.0
        handler.store.write(state)
        handler._scontrol('reboot', 'reason=aim344-device-recovery', 'gpu-g7-1')
        self.assertFalse(state.reboot_completed)
        self.assertEqual(1, state.reboot_requests)
        self.assertEqual(1, len(self.executor.reboots()))
        if came_back:
            self.executor.boot_id = 'cccccccc-0000-0000-0000-000000000003'
            self.executor.node_state = 'IDLE+DRAIN'
        else:
            self.executor.boot_id = held
            self.executor.node_state = 'IDLE+DRAIN+REBOOT_REQUESTED'
        return state

    def test_a_pending_efa_reboot_is_not_left_outstanding_by_a_good_rebind(self):
        """S3's first case: the scheduler still holds the reboot.

        The round must refuse rather than resume, and must not queue a second
        reboot, even though this attempt's rebind would have succeeded.
        """
        self.reach_an_efa_reboot_whose_completion_was_missed(came_back=False)
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        message = str(caught.exception)
        self.assertIn('already requested', message)
        self.assertIn('no second reboot', message)
        self.assertEqual([], self.executor.resumes(),
                         'the node was resumed with a reboot still outstanding')
        self.assertEqual(1, len(self.executor.reboots()),
                         'a second reboot was queued')
        self.assertEqual(1, self.make().store.read().reboot_requests)

    def test_a_completed_but_unobserved_efa_reboot_is_recorded_and_remounted(self):
        """S3's second case: it came back, unnoticed.

        The round must record the boot, re-establish the staging mount, and resume
        once -- not skip remount-staging because `rebooted` was still false.

        Check 6 reports PASS after the boot, because the two counters it warns on are
        read from `rdma -p statistic show` and accumulate per boot
        (checks/6-efa-loopback.sh:166-175). A counter WARN that still appears after a
        boot cannot be qualified against a pre-boot baseline, and that refusal is
        asserted in test_device_session_rework.py rather than mixed into this
        reconciliation case.
        """
        self.reach_an_efa_reboot_whose_completion_was_missed(came_back=True)
        self.executor.check_output['6'] = _rework.PASS_6
        result = self.make().recover()
        self.assertEqual('runtime-ready', result.state.phase)
        self.assertTrue(result.state.reboot_completed)
        self.assertEqual('cccccccc-0000-0000-0000-000000000003',
                         result.state.boot_id_after_reboot)
        self.assertIn('completed while this round was not watching',
                      ' '.join(result.state.notes.split()))
        self.assertIn('remount-staging', self.executor.actions(),
                      'the staging bind mount was not re-established after a '
                      'reboot this round did not observe')
        self.assertEqual(1, len(self.executor.reboots()),
                         'the node was rebooted twice')
        self.assertEqual(1, len(self.executor.resumes()))

    def test_the_reconciliation_runs_before_the_rebind_is_attempted(self):
        """Ordering, not just presence.

        A round with a reboot still pending must refuse before issuing any device
        mutation, so the recorded maintenance actions of that attempt contain no
        rebind at all.
        """
        self.reach_an_efa_reboot_whose_completion_was_missed(came_back=False)
        before = len(self.executor.maintenance_calls)
        with self.assertRaises(session.Refusal):
            self.make().recover()
        during = [call[0] for call in self.executor.maintenance_calls[before:]]
        self.assertNotIn('efa-rebind', during,
                         f'a device mutation was issued with a reboot pending: '
                         f'{during}')

    def test_a_completed_reboot_makes_the_retry_skip_the_rebind(self):
        """The device is already bound after a reboot, so no mutation is needed.

        This is the same ordering property on the other branch: reconciliation puts
        the round into restoration, and `if not state.reboot_completed` then keeps
        the rebind from running again. Check 6 reports PASS after the boot, for the
        per-boot-counter reason given above.
        """
        self.reach_an_efa_reboot_whose_completion_was_missed(came_back=True)
        self.executor.check_output['6'] = _rework.PASS_6
        before = len(self.executor.maintenance_calls)
        self.make().recover()
        during = [call[0] for call in self.executor.maintenance_calls[before:]]
        self.assertNotIn('efa-rebind', during,
                         f'the device was rebound after a completed reboot: '
                         f'{during}')
        self.assertIn('remount-staging', during)

    def test_the_rejected_nesting_would_have_skipped_both_cases(self):
        """The negative control, on the same recorded state.

        The rejected control flow was:

            if needs_reboot:              # from the latest rebind
                if state.reboot_requests: # reconciliation, nested inside

        Reconstructed as a predicate here and evaluated on the state each case
        above reaches, with the retry's rebind succeeding, so `needs_reboot` is
        false. It reconciles in neither case, which is the defect. Without this the
        two tests above could be passing for an unrelated reason.
        """
        for came_back in (False, True):
            with self.subTest(came_back=came_back):
                state = self.reach_an_efa_reboot_whose_completion_was_missed(
                    came_back=came_back)
                # The retry's rebind succeeds, so the rejected code's needs_reboot
                # is false for an EFA round.
                needs_reboot = False
                reconciled = bool(needs_reboot and state.reboot_requests)
                self.assertFalse(
                    reconciled,
                    'the rejected nesting DID reconcile, so this suite has no '
                    'negative control')
                # And the repaired code does reconcile on the same state.
                handler = self.make()
                notes = []
                if came_back:
                    self.assertTrue(
                        handler._reconcile_outstanding_reboot(
                            handler.store.read(), notes))
                    self.assertTrue(any('was not watching' in note
                                        for note in notes), notes)
                else:
                    with self.assertRaises(session.Refusal):
                        handler._reconcile_outstanding_reboot(
                            handler.store.read(), notes)
                self.setUp()

    def test_a_normal_efa_round_still_rebinds_and_never_reboots(self):
        """The positive control: reconciliation must not become 'always reboot'."""
        self.make().start('efa')
        result = self.make().recover()
        self.assertEqual('runtime-ready', result.state.phase)
        self.assertIn('efa-rebind', self.executor.actions())
        self.assertEqual([], self.executor.reboots(),
                         'a successful rebind should not reboot the node')
        self.assertEqual(0, result.state.reboot_requests)
        self.assertNotIn('remount-staging', self.executor.actions(),
                         'nothing was remounted on a round that never rebooted')
        self.assertEqual(1, len(self.executor.resumes()))

    def test_a_normal_gpu_round_still_reboots_once_and_remounts(self):
        """The repaired GPU behaviour is retained, not traded for the EFA one."""
        self.make().start('gpu')
        result = self.make().recover()
        self.assertEqual('runtime-ready', result.state.phase)
        self.assertEqual(1, len(self.executor.reboots()))
        self.assertTrue(result.state.reboot_completed)
        self.assertIn('remount-staging', self.executor.actions())
        self.assertEqual(1, len(self.executor.resumes()))

    def test_a_safety_refusal_on_the_efa_rebind_still_stops_before_a_reboot(self):
        """An identity or safety refusal is not a qualified rebind failure."""
        self.make().start('efa')
        self.executor.maintenance_failures['efa-rebind'] = (
            3, 'this node is not the provisioned fault target\n')
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        self.assertIn('stopped before changing anything', str(caught.exception))
        self.assertEqual([], self.executor.reboots())
        self.assertEqual([], self.executor.resumes())
        self.assertEqual('recovery-failed', self.make().store.read().phase)

    def test_a_recorded_request_that_is_no_longer_pending_does_not_block_forever(self):
        """A request the scheduler dropped, with no boot change, must not deadlock.

        The node is readable, its boot id is what the round started on, and no
        REBOOT_* flag remains. Nothing establishes that a reboot happened, so the
        round proceeds -- and, for a GPU round, requests the reboot it still needs.
        The recorded count keeps both requests, so the history is not rewritten.
        """
        handler = self.make()
        handler.start('gpu')
        state = self.make().store.read()
        state.reboot_requests = 1
        state.reboot_requested_at = 900.0
        self.make().store.write(state)
        self.executor.node_state = 'IDLE+DRAIN'
        result = self.make().recover()
        self.assertEqual('runtime-ready', result.state.phase)
        self.assertIn('no longer pending', ' '.join(result.state.notes.split()))
        self.assertEqual(2, result.state.reboot_requests,
                         'the earlier request was erased from the record')
        self.assertEqual(1, len(self.executor.reboots()))

    # -- an unread boot is not an unchanged boot --
    def unreadable_boot(self, answer):
        """Make the target's boot-id request answer with `answer`.

        `answer` is (returncode, stdout). The two cases are a request that failed and
        one that succeeded with nothing usable, because `_current_boot_id` returns None
        for both and the rejected branch read None as 'the boot has not changed'.
        """
        original = self.executor.run_maintenance

        def answering(argv, timeout=None):
            if argv and argv[0] == 'boot-id':
                return session.Completed(answer[0], answer[1], '')
            return original(argv, timeout=timeout)

        self.executor.run_maintenance = answering

    def reach_an_outstanding_request(self, shorten_wait=True):
        """A GPU round with a recorded, unreconciled reboot request.

        `shorten_wait` sets the reboot observation window to zero. The repaired code
        never reaches `_await_boot` in the refusal tests, but the PRE-REPAIR code does
        -- it read an unread boot as unchanged and went on to request a second reboot
        -- and with the default 900 s window the falsification bench would sleep
        rather than report. A test whose subject IS the second reboot completing
        passes False and keeps the ordinary window.
        """
        self.make().start('gpu')
        if shorten_wait:
            self.config_body['assignments']['table-1']['reboot_wait_seconds'] = 0
            self.config_body['assignments']['table-1']['reboot_poll_seconds'] = 0
            self.write_config()
        state = self.make().store.read()
        state.reboot_requests = 1
        state.reboot_requested_at = 900.0
        self.make().store.write(state)
        self.executor.node_state = 'IDLE+DRAIN'
        return state

    def test_an_unreadable_boot_does_not_count_as_an_unchanged_boot(self):
        """The residual the review's reconciled comments name.

        `_current_boot_id` returns None when the target does not answer. The rejected
        branch required `current and state.boot_id and current != state.boot_id` to
        record a completed reboot, so None fell through to 'this node did not change
        boot id; continuing recovery' -- and a node that HAD come back then had its
        staging bind mount missing while restoration ran, or a second reboot queued.
        An unread boot is not an observation, so this attempt refuses and retries.
        """
        self.reach_an_outstanding_request()
        self.unreadable_boot((255, ''))
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        message = str(caught.exception)
        self.assertIn('could not establish whether it completed', message)
        self.assertIn('did not report its current boot', message)
        self.assertEqual([], self.executor.reboots(),
                         'a second reboot was queued on an unread boot')
        self.assertEqual([], self.executor.resumes())
        self.assertNotIn('restore-runtime', self.executor.actions())
        state = self.make().store.read()
        self.assertEqual(1, state.reboot_requests,
                         'the outstanding request was erased')
        self.assertFalse(state.reboot_completed)

    def test_an_empty_boot_answer_is_refused_the_same_way(self):
        """Same failure class: rc 0 with nothing usable in stdout.

        `_current_boot_id` returns None for this too, and the return code differs, so
        asserting only the transport failure above would leave half the branch
        unexercised.
        """
        self.reach_an_outstanding_request()
        self.unreadable_boot((0, '   \n'))
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        self.assertIn('could not establish whether it completed',
                      str(caught.exception))
        self.assertEqual([], self.executor.reboots())
        self.assertEqual([], self.executor.resumes())

    def test_a_round_with_no_recorded_boot_is_refused_rather_than_assumed(self):
        """Nothing to compare against is also not 'unchanged'."""
        self.reach_an_outstanding_request()
        state = self.make().store.read()
        state.boot_id = None
        self.make().store.write(state)
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        self.assertIn('no recorded boot to compare against', str(caught.exception))
        self.assertEqual([], self.executor.reboots())

    def test_the_next_attempt_reconciles_once_the_boot_can_be_read(self):
        """The refusal is retryable, not terminal.

        This is the positive control: the same round, with the target answering, goes
        on to reconcile the completed reboot and finish. Without it, 'refuse on an
        unread boot' could be stranding the round.
        """
        self.reach_an_outstanding_request()
        original = self.executor.run_maintenance
        answer = {'boot': None}

        def answering(argv, timeout=None):
            if argv and argv[0] == 'boot-id':
                if answer['boot'] is None:
                    return session.Completed(255, '', '')
                return session.Completed(0, answer['boot'] + '\n', '')
            return original(argv, timeout=timeout)

        self.executor.run_maintenance = answering
        with self.assertRaises(session.Refusal):
            self.make().recover()
        # The node had in fact come back; now it answers.
        answer['boot'] = 'cccccccc-0000-0000-0000-000000000003'
        result = self.make().recover()
        self.assertTrue(result.state.reboot_completed)
        self.assertIn('completed while this round was not watching',
                      result.state.notes)
        self.assertIn('remount-staging', self.executor.actions())
        self.assertEqual([], self.executor.reboots(),
                         'the reconciled reboot was repeated')
        self.assertEqual('runtime-ready', result.state.phase)

    def test_a_readable_unchanged_boot_still_continues(self):
        """The other positive control: a real, readable, unchanged boot.

        `_current_boot_id` answering with the recorded boot is an observation, and it
        means what it says. Without this test the repair could have been 'refuse
        whenever a request is outstanding'. The ordinary reboot-observation window is
        kept, because this round really does go on to request and observe its reboot.
        """
        self.reach_an_outstanding_request(shorten_wait=False)
        result = self.make().recover()
        self.assertIn('no longer pending', ' '.join(result.state.notes.split()))
        self.assertEqual('runtime-ready', result.state.phase)
        self.assertEqual(1, len(self.executor.reboots()))

    def test_a_round_with_no_request_reconciles_the_actual_boot_and_pending_state(self):
        """A round that requested no reboot still has to observe the node's state.

        This test previously asserted the opposite -- that an EFA round with no reboot
        request of its own reaches runtime-ready even when the target cannot report its
        current boot -- and the safety review named that expectation itself as the
        defect: 'test_a_round_with_no_request_is_not_affected_by_the_boot_read starts
        EFA, makes current boot unreadable and requires runtime-ready. That assertion
        conflicts with this card's actual-boot/retry requirement when the node's boot or
        pending state is no longer established.'

        The scope of the old expectation was 'this round did not request a reboot', and
        the fact it stood in for was 'no reboot concerns this node'. Those differ: a
        request can arrive from outside the round between attempts, and a node whose
        boot cannot be read has not established that it is still the boot the round
        observed. So the refusal is not scoped to a locally recorded request any more,
        and an unreadable boot keeps the node reserved instead of restoring under it.

        The readable ordinary EFA recovery is the positive control, immediately below.
        """
        self.make().start('efa')
        self.unreadable_boot((255, ''))
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        message = str(caught.exception)
        self.assertIn('could not establish which boot', message)
        self.assertEqual([], self.executor.reboots(),
                         'an unreadable boot must not become a reboot request')
        self.assertEqual([], self.executor.resumes())

    def test_a_readable_round_with_no_request_still_recovers_normally(self):
        """The positive control for the test above: nothing odd, so nothing refused.

        The same EFA round with a target that answers its boot-id request reaches
        runtime-ready without a reboot, which is what keeps the repair above from
        meaning 'an EFA round can never finish'.
        """
        self.make().start('efa')
        result = self.make().recover()
        self.assertEqual('runtime-ready', result.state.phase)
        self.assertEqual([], self.executor.reboots())
        self.assertEqual(1, len(self.executor.resumes()))


class S3InitialBootValidity(Base):
    """The initial boot observation, before any mutation and before the first wait.

    The safety review's second finding. `_boot_id` returns rc-0 empty stdout as the
    empty string and `start` stores it unvalidated, so the round's baseline boot is a
    value no observation established. The first recovery then requests a reboot and
    calls `_await_boot(state.boot_id)`, which accepts ANY nonempty current boot that
    differs from that empty baseline -- including the boot the node is still running,
    while the scheduler has not yet acted on the request. The caller records
    `reboot_completed=True`, restores, qualifies the checks and can resume a node whose
    reboot is still outstanding.

    The route that reaches a resume without a downstream check failing is
    capture-valid-then-preparation-failed: the GPU is still present, so check 0 passes.
    That is the route these tests take, so the decision under test is the boot
    comparison rather than a GPU count that happens to fail for another reason.
    """

    def empty_initial_boot(self):
        """The target answers the boot-id request with rc 0 and nothing usable.

        This is `_boot_id`'s own permissive shape, not an invented one: the same
        `(0, '   \\n')` answer the existing outstanding-request tests use for
        `_current_boot_id`.
        """
        original = self.executor.run_maintenance
        answers = {'boot': '   \n'}
        self.boot_answers = answers

        def answering(argv, timeout=None):
            if argv and argv[0] == 'boot-id':
                return session.Completed(0, answers['boot'], '')
            return original(argv, timeout=timeout)

        self.executor.run_maintenance = answering
        return answers

    def one_boot_observation(self):
        """Let `_await_boot` evaluate its comparison exactly once.

        The window must be positive or the loop never runs: `_await_boot`'s condition is
        `self.clock() < deadline` and this bench's clock is constant, so a zero window
        returns None and the caller reports 'has not returned yet'. That is the wrong
        reason for these tests to pass -- measured: with a zero window the pre-repair
        controller refused on that message rather than recording a completion. A
        one-second window with no poll delay makes the comparison happen, once.
        """
        self.config_body['assignments']['table-1']['reboot_wait_seconds'] = 1
        self.config_body['assignments']['table-1']['reboot_poll_seconds'] = 0
        self.write_config()

    def attempt_recovery(self, handler=None):
        """Run one recovery, returning its refusal message or ''.

        Captured rather than required, so the assertions that fail on the unrepaired
        controller are the behavioural ones -- the resume and the recorded completion --
        rather than 'Refusal not raised'.
        """
        try:
            (handler or self.make()).recover()
        except session.Refusal as refusal:
            return str(refusal)
        return ''

    def test_a_round_cannot_start_on_a_boot_it_never_read(self):
        """Validate before mutation: the transition table's start-preflight row.

        `start` records the initial boot before it drains and before any device call,
        and every later completion decision is relative to that value. An unusable
        answer there must stop the round while nothing has been changed, rather than
        become a baseline that a later comparison reads as 'the boot changed'.
        """
        self.empty_initial_boot()
        try:
            self.make().start('gpu')
            message = ''
        except session.Refusal as refusal:
            message = str(refusal)
        # The behavioural consequence first: no mutation was issued at all.
        actions = self.executor.actions()
        for mutation in ('gpu-prepare', 'gpu-remove', 'efa-unbind'):
            self.assertNotIn(mutation, actions,
                             f'{mutation} ran for a round with no usable boot')
        self.assertEqual([], self.executor.reboots())
        state = self.make().store.read()
        self.assertNotEqual('', state.boot_id,
                            'an unusable boot answer was stored as this round\'s '
                            'baseline boot')
        self.assertIn('current boot', message)

    def test_the_first_wait_does_not_manufacture_a_completed_reboot(self):
        """The review's counterexample, on the first recovery of a real round.

        The round's recorded boot is emptied to reproduce a record started by the
        permissive read, the node keeps running the boot it always had, and the
        scheduler still holds the request. `_await_boot` must not report that boot as a
        change, and the caller must not record a completion.
        """
        self.make().start('gpu')
        state = self.make().store.read()
        state.boot_id = ''
        self.make().store.write(state)
        # The node never reboots: boot-id keeps answering the boot it started on, and
        # the scheduler still shows the request outstanding.
        original = self.executor.run_maintenance
        unchanged = 'aaaaaaaa-0000-0000-0000-000000000001'

        def answering(argv, timeout=None):
            if argv and argv[0] == 'boot-id':
                return session.Completed(0, unchanged + '\n', '')
            return original(argv, timeout=timeout)

        self.executor.run_maintenance = answering
        self.one_boot_observation()

        message = self.attempt_recovery()
        after = self.make().store.read()
        # The behavioural consequences first: no resume, no recorded completion.
        self.assertEqual([], self.executor.resumes(),
                         'a node was returned to service with its reboot outstanding')
        self.assertFalse(after.reboot_completed,
                         'a reboot completion was recorded against an empty baseline '
                         'boot')
        self.assertNotEqual(unchanged, after.boot_id_after_reboot,
                            'the boot the node never left was recorded as the boot it '
                            'came back on')
        self.assertNotEqual('runtime-ready', after.phase)
        self.assertIn('boot', message.lower())

    def test_a_pending_scheduler_request_is_not_a_completed_reboot(self):
        """Same round, with the scheduler's own flag still set.

        `REBOOT_ISSUED` is one of the two flags this Slurm build emits, read out of the
        deployed libraries rather than assumed (see REBOOT_PENDING_FLAGS). A completion
        decision must not be reached while the request the round made is still visibly
        outstanding.
        """
        self.make().start('gpu')
        state = self.make().store.read()
        state.boot_id = ''
        self.make().store.write(state)
        original_slurm = self.executor.run_slurm
        unchanged = 'aaaaaaaa-0000-0000-0000-000000000001'

        def slurm(argv, timeout=None):
            result = original_slurm(argv, timeout=timeout)
            if Path(argv[0]).name == 'scontrol' and argv[1:2] == ['reboot']:
                # The scheduler accepted the request and has not acted on it: the flag
                # is set and the node is still running the boot it always had.
                self.executor.node_state = 'IDLE+DRAIN+REBOOT_ISSUED'
                self.executor.boot_id = unchanged
            return result

        self.executor.run_slurm = slurm
        self.one_boot_observation()
        self.attempt_recovery()
        after = self.make().store.read()
        self.assertEqual([], self.executor.resumes())
        self.assertFalse(after.reboot_completed)
        self.assertNotEqual('runtime-ready', after.phase)

    def advancing_handler(self, step=0.4):
        """A handler whose clock advances, so a bounded wait actually ends.

        `Base.make` pins the clock to a constant, which is right for every decision that
        does not wait: it makes the tests deterministic. `_await_boot` is a real bounded
        loop against that clock, though, so with a constant clock and a zero poll delay
        it cannot terminate unless the very first reading satisfies it -- the reason the
        earlier bench in this class set the window to zero and, as a result, exercised
        'the node has not returned' rather than the comparison. A clock that advances a
        little on each reading ends the window after a few polls, which is what the
        deployed helper's wall clock does.
        """
        config = session.load_config(self.config_path, require_root_owned=False)
        ticks = {'now': 1000.0}

        def clock():
            ticks['now'] += step
            return ticks['now']

        return session.DeviceSession(
            config=config, assignment_id='table-1', peer_uid=os.getuid(),
            executor=self.executor, clock=clock)

    def test_a_current_answer_that_is_not_a_boot_id_is_not_a_change(self):
        """The other side of the comparison: what establishes the change.

        The recorded boot is real and the node answers rc 0 with a diagnostic line
        instead of a boot id. That is not an observation of the boot, so it must not
        satisfy 'the boot changed' -- and it must not be recorded as the boot the node
        came back on, because the round's evidence would then hold a value the node
        never reported.

        Where that unusable answer is now caught is EARLIER than it was, and the
        refusal message changed with it. A round with no recorded reboot request
        reconciles the node's actual boot and pending state before any mutation
        (`_reconcile_outstanding_reboot`, the no-request branch), so this round is
        refused before it requests the reboot rather than after waiting for a boot it
        could not read. The load-bearing assertions are unchanged and are asserted
        below: nothing resumed, no completion recorded, the diagnostic never became the
        round's boot -- and, now, no reboot was requested at all, which is strictly
        less mutation than the previous behaviour.
        """
        self.make().start('gpu')
        before = self.make().store.read().boot_id
        original = self.executor.run_maintenance
        diagnostic = 'Failed to read /proc/sys/kernel/random/boot_id'

        def answering(argv, timeout=None):
            if argv and argv[0] == 'boot-id':
                return session.Completed(0, diagnostic + '\n', '')
            return original(argv, timeout=timeout)

        self.executor.run_maintenance = answering
        self.one_boot_observation()
        message = self.attempt_recovery(self.advancing_handler())
        after = self.make().store.read()
        self.assertEqual([], self.executor.resumes())
        self.assertFalse(after.reboot_completed)
        self.assertEqual(before, after.boot_id,
                         'a reading that is not a boot id was recorded as the boot')
        self.assertNotEqual(diagnostic, after.boot_id_after_reboot)
        self.assertEqual([], self.executor.reboots(),
                         'a boot that could not be read became a reboot request')
        self.assertIn('could not establish which boot', message)

    # -- the valid cases, which must keep working --
    def test_an_ordinary_gpu_round_still_reboots_and_resumes(self):
        """The positive control: a readable initial boot and a real boot change.

        Without this, 'refuse whenever the boot looks odd' would pass the tests above
        while making every GPU round unrecoverable. The fake scheduler changes the boot
        id on `scontrol reboot`, so this is the ordinary path end to end.
        """
        self.make().start('gpu')
        before = self.make().store.read().boot_id
        self.assertTrue(before, 'this control needs a readable initial boot')
        result = self.make().recover()
        self.assertEqual('runtime-ready', result.state.phase)
        self.assertTrue(result.state.reboot_completed)
        self.assertNotEqual(before, result.state.boot_id)
        self.assertEqual(before, result.state.boot_id_before_reboot)
        self.assertEqual(1, len(self.executor.resumes()))

    def test_a_readable_initial_boot_is_still_recorded_at_start(self):
        """The other side of the start-time validation.

        A round whose boot answers normally records it and proceeds to the device
        mutation, so the new check is a validity requirement rather than a refusal of
        the path.
        """
        self.make().start('gpu')
        state = self.make().store.read()
        self.assertEqual('aaaaaaaa-0000-0000-0000-000000000001', state.boot_id)
        self.assertIn('gpu-remove-attempted', state.prepared)

    def test_an_efa_round_with_a_readable_boot_is_unaffected(self):
        """The EFA path takes the same start-time read and must not be blocked."""
        self.make().start('efa')
        self.executor.check_output['6'] = _rework.PASS_6
        result = self.make().recover()
        self.assertEqual('runtime-ready', result.state.phase)
        self.assertEqual([], self.executor.reboots())


# ---------------------------------------------------------------------------
# S4: telemetry restoration is bound to a unique operation, and every success
# has observed the state it restored to.
# ---------------------------------------------------------------------------

# Recording stubs for the two binaries maintenance.gpu_prepare/gpu_restore actually
# invoke. They are not mocks of the functions under test: the real functions run,
# build their real argv, and these programs record what arrived and answer as the
# deployed tools do.
#
# `systemctl` semantics come from measurement, not documentation. On the target,
# `is-active nvidia-dcgm.service` prints `active` with rc 0, and a unit that does not
# exist at all prints `inactive` with rc 4
# (participant-revision/runs/rework7-read-telemetry-semantics/output.log:3-5). On
# systemd 255.4-1ubuntu8.17, `show -p LoadState -p ActiveState` separates those cases
# while `is-active` does not: a stopped loaded unit is loaded/inactive, a nonexistent
# one is not-found/inactive, a masked one is masked/inactive, and `is-active` prints
# the word `inactive` for all three
# (participant-revision/runs/rework9-read-loadstate/output.log). The stub reproduces
# both interfaces, including the return codes, so a helper that trusted rc alone, or
# the word alone, is caught.
#
# It can also be told to answer with a transitional word, an empty string, or a
# scripted sequence of words, because those are the observations the review's S4-A
# finding is about: 'activating' is not 'inactive', and neither is ''.
SYSTEMCTL_STUB = r'''#!/usr/bin/env python3
import os
import pathlib
import sys

root = pathlib.Path(os.environ['AIM344_STUB_DIR'])
state_file = root / 'unit-state.txt'
load_file = root / 'unit-load-state.txt'
calls = root / 'systemctl-calls.txt'
argv = sys.argv[1:]
with calls.open('a') as handle:
    handle.write(' '.join(argv) + '\n')
verb = argv[0] if argv else ''


def next_active_state():
    """The next reading, honouring a scripted sequence if one is present.

    A scripted sequence is one word per line, consumed one reading at a time. What
    is left after the last line is the settled answer. This is how a unit caught
    mid-transition is reproduced without a real systemd.
    """
    script = root / 'is-active-script.txt'
    if script.is_file():
        words = [line for line in script.read_text().split('\n') if line != '']
        if words:
            state = words[0]
            script.write_text('\n'.join(words[1:]) + ('\n' if words[1:] else ''))
            return state
    return state_file.read_text().strip()


def load_state():
    return (load_file.read_text().strip() if load_file.is_file() else 'loaded')


if verb == 'show':
    state = next_active_state()
    load = load_state()
    # systemd prints `LoadState=not-found ActiveState=inactive` for a unit that does
    # not exist, and exits 0: the query succeeded, the unit is simply absent. A
    # masked unit reports LoadState=masked with the same ActiveState. Measured, see
    # runs/rework9-read-loadstate/output.log.
    if load != 'loaded':
        state = 'inactive'
    elif state == 'EMPTY':
        # A failed invocation prints nothing usable. The helper must not read that as
        # a state; the rejected code's `!= 'active'` test read it as inactive.
        sys.exit(3)
    wanted = [argv[i + 1] for i, a in enumerate(argv)
              if a == '-p' and i + 1 < len(argv)]
    wanted = wanted or ['LoadState', 'ActiveState']
    values = {'LoadState': load, 'ActiveState': state,
              'SubState': 'running' if state == 'active' else 'dead'}
    for name in wanted:
        sys.stdout.write(f'{name}={values.get(name, "")}\n')
    sys.exit(0)
if verb == 'is-active':
    state = next_active_state()
    load = load_state()
    if load != 'loaded':
        # not-found and masked both print the word `inactive`; only the status
        # differs (4 and 3 respectively on the measured build).
        sys.stdout.write('inactive\n')
        sys.exit(4 if load == 'not-found' else 3)
    if state == 'EMPTY':
        sys.exit(3)
    sys.stdout.write(state + '\n')
    # rc 0 only for 'active'; 4 for everything else, as measured on the target.
    sys.exit(0 if state == 'active' else 4)
if verb == 'stop':
    if (root / 'refuse-stop').exists():
        sys.stderr.write('Job for the unit failed\n')
        sys.exit(1)
    if (root / 'stop-leaves').is_file():
        state_file.write_text((root / 'stop-leaves').read_text())
        sys.exit(0)
    state_file.write_text('inactive\n')
    sys.exit(0)
if verb == 'start':
    if (root / 'refuse-start').exists():
        sys.stderr.write('Job for the unit failed\n')
        sys.exit(1)
    if (root / 'start-leaves').is_file():
        state_file.write_text((root / 'start-leaves').read_text())
        sys.exit(0)
    state_file.write_text('active\n')
    sys.exit(0)
sys.exit(0)
'''

NVIDIA_SMI_STUB = r'''#!/usr/bin/env python3
import os
import pathlib
import sys

root = pathlib.Path(os.environ['AIM344_STUB_DIR'])
mode_file = root / 'persistence-mode.txt'
sticky = root / 'persistence-sticky'
calls = root / 'nvidia-smi-calls.txt'
argv = sys.argv[1:]
with calls.open('a') as handle:
    handle.write(' '.join(argv) + '\n')
for item in argv:
    if item.startswith('--persistence-mode='):
        # `persistence-sticky` reproduces a GPU that accepts the write and keeps
        # answering the old value, which is the shape of the real helper's
        # 'Persistence is still Enabled' refusal: it happens AFTER both
        # original-state records were written.
        if sticky.is_file():
            mode_file.write_text(sticky.read_text())
        elif item == '--persistence-mode=1':
            mode_file.write_text('Enabled\n')
        else:
            mode_file.write_text('Disabled\n')
        sys.exit(0)
if '--query-gpu=persistence_mode' in argv:
    sys.stdout.write(mode_file.read_text().strip() + '\n')
    sys.exit(0)
sys.exit(0)
'''

# Operation identifiers of the shape the coordinator's helper generates,
# `<assignment>/<round>/<32 hex>`. TABLE1_R1 and TABLE2_R1 are the S4-B case: two
# assignments sharing one exercise node, both on their own round 1.
TABLE1_R1 = 'table-1/1/' + 'a' * 32
TABLE2_R1 = 'table-2/1/' + 'b' * 32
TABLE1_R2 = 'table-1/2/' + 'c' * 32


class S4TelemetryRestoration(unittest.TestCase):
    """The real maintenance functions, with real record files on disk.

    Nothing here asserts a return code a mock was told to produce: `gpu_prepare` and
    `gpu_restore` run for real, write and read real files under a temporary
    RECORD_DIR, and drive stub `systemctl`/`nvidia-smi` programs that record their
    arguments and answer as the deployed tools do.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.records = root / 'records'
        self.records.mkdir()
        self.stubs = root / 'stubs'
        self.stubs.mkdir()
        self.stub_state = root / 'stub-state'
        self.stub_state.mkdir()
        (self.stub_state / 'unit-state.txt').write_text('active\n')
        (self.stub_state / 'persistence-mode.txt').write_text('Enabled\n')
        for name, body in (('systemctl', SYSTEMCTL_STUB),
                           ('nvidia-smi', NVIDIA_SMI_STUB)):
            path = self.stubs / name
            path.write_text(body)
            path.chmod(0o755)
        self.fault_config = root / 'aim344-device-fault.json'
        self.fault_config.write_text(json.dumps({
            'gpu_uuid': 'GPU-11111111-2222-4333-8444-555555555555',
            'efa_rdma_device': 'rdmap176s0',
            'slurm_node': 'gpu-g7-1',
        }))
        # The settle wait is short here so a bounded-wait test does not spend real
        # seconds; the bound itself is what is under test, not its length.
        self.config = {'slurm_bin': '/opt/aws/pcs/scheduler/slurm-25.05/bin',
                       'telemetry_settle_attempts': 3,
                       'telemetry_settle_pause_seconds': 0}

        # The helper hardcodes /usr/bin/systemctl and /usr/bin/nvidia-smi, and it
        # reads /etc/aim344-device-fault.json. Redirect all three at the module
        # level rather than editing the source under test.
        self.saved_record_dir = maintenance.RECORD_DIR
        maintenance.RECORD_DIR = self.records
        self.saved_run = subprocess.run
        stubs, stub_state = self.stubs, self.stub_state
        fault_config = self.fault_config

        real_run = subprocess.run

        def redirected(argv, **kwargs):
            argv = list(argv)
            if argv and argv[0].startswith('/usr/bin/'):
                name = Path(argv[0]).name
                if (stubs / name).is_file():
                    argv[0] = str(stubs / name)
            env = dict(kwargs.pop('env', None) or os.environ)
            env['AIM344_STUB_DIR'] = str(stub_state)
            return real_run(argv, env=env, **kwargs)

        self.redirected = redirected
        maintenance.subprocess.run = redirected

        self.saved_read_text = Path.read_text

        def read_text(path, *args, **kwargs):
            if str(path) == '/etc/aim344-device-fault.json':
                return self.saved_read_text(fault_config, *args, **kwargs)
            return self.saved_read_text(path, *args, **kwargs)

        Path.read_text = read_text

    def tearDown(self):
        maintenance.RECORD_DIR = self.saved_record_dir
        maintenance.subprocess.run = self.saved_run
        Path.read_text = self.saved_read_text
        self.tmp.cleanup()

    # -- helpers the tests read --
    def unit_state(self):
        return (self.stub_state / 'unit-state.txt').read_text().strip()

    def persistence(self):
        return (self.stub_state / 'persistence-mode.txt').read_text().strip()

    def systemctl_calls(self):
        path = self.stub_state / 'systemctl-calls.txt'
        return path.read_text().splitlines() if path.is_file() else []

    def script_is_active(self, *words):
        """The next readings of the unit's active state, in order."""
        (self.stub_state / 'is-active-script.txt').write_text(
            '\n'.join(words) + '\n')

    def set_load_state(self, value):
        """What systemd reports as the unit's LoadState.

        `loaded` for a unit that exists, `not-found` for one that does not, `masked`
        for a masked one. All three report ActiveState=inactive when not running, and
        `systemctl is-active` prints the word `inactive` for all three, which is the
        S4 finding (runs/rework9-read-loadstate/output.log).
        """
        (self.stub_state / 'unit-load-state.txt').write_text(value + '\n')

    def capture(self, function, *args):
        """Run one real maintenance function, returning (rc-or-Refusal, output)."""
        # Legacy direct-helper fixtures model the trusted controller's request.
        # Supply its new first-capture flag explicitly, including on repeated
        # calls: replaying that flag must never overwrite surviving originals.
        # Retry/legacy tests opt out with fresh_capture=False; never infer this
        # authorization from record presence or from the helper's own decision.
        if function is maintenance.gpu_prepare and len(args) == 2:
            config, options = args
            args = (config, {'fresh_capture': True, **options})
        stream = io.StringIO()
        saved = sys.stdout
        sys.stdout = stream
        try:
            code = function(*args)
        except maintenance.Refusal as refusal:
            return refusal, stream.getvalue()
        finally:
            sys.stdout = saved
        return code, stream.getvalue()

    # -- the operation-bound record --
    def test_prepare_records_the_operation_and_both_original_states(self):
        code, output = self.capture(maintenance.gpu_prepare, self.config,
                                    {'operation': TABLE1_R1})
        self.assertEqual(0, code, output)
        self.assertIn(f'operation={TABLE1_R1}', output)
        record = json.loads((self.records / 'native-dcgm-state.txt').read_text())
        self.assertEqual('active', record['value'])
        self.assertEqual(TABLE1_R1, record['operation'])
        mode = json.loads((self.records / 'selected-persistence-mode.txt').read_text())
        self.assertEqual('Enabled', mode['value'])
        self.assertEqual(TABLE1_R1, mode['operation'])
        # And it really paused the service and disabled persistence.
        self.assertEqual('inactive', self.unit_state())
        self.assertEqual('Disabled', self.persistence())

    def test_an_active_original_is_restored_and_confirmed(self):
        self.capture(maintenance.gpu_prepare, self.config,
                     {'operation': TABLE1_R1})
        self.assertEqual('inactive', self.unit_state())
        code, output = self.capture(maintenance.gpu_restore, self.config,
                                    {'operation': TABLE1_R1})
        self.assertEqual(0, code, output)
        self.assertEqual('active', self.unit_state(),
                         'the telemetry owner was left paused')
        self.assertEqual('Enabled', self.persistence())
        self.assertIn('recorded active', output)
        self.assertIn('start', ' '.join(self.systemctl_calls()))

    def test_an_inactive_original_is_restored_to_inactive_explicitly(self):
        """S4's second half: a deliberately stopped service must stay stopped.

        The round records `inactive`, the reboot starts the service, and restoration
        must put it back rather than leave the node changed.
        """
        (self.stub_state / 'unit-state.txt').write_text('inactive\n')
        code, output = self.capture(maintenance.gpu_prepare, self.config,
                                    {'operation': TABLE1_R1})
        self.assertEqual(0, code, output)
        record = json.loads((self.records / 'native-dcgm-state.txt').read_text())
        self.assertEqual('inactive', record['value'])
        # A reboot started it.
        (self.stub_state / 'unit-state.txt').write_text('active\n')
        code, output = self.capture(maintenance.gpu_restore, self.config,
                                    {'operation': TABLE1_R1})
        self.assertEqual(0, code, output)
        self.assertEqual('inactive', self.unit_state(),
                         'a service the round recorded as inactive was left running')
        self.assertIn('recorded inactive', output)
        self.assertIn('stop', ' '.join(self.systemctl_calls()))

    def test_an_inactive_original_already_inactive_needs_no_stop(self):
        """The other valid success case: nothing to put back, and it is confirmed."""
        (self.stub_state / 'unit-state.txt').write_text('inactive\n')
        self.capture(maintenance.gpu_prepare, self.config,
                     {'operation': TABLE1_R1})
        before = len([c for c in self.systemctl_calls() if c.startswith('stop')])
        code, output = self.capture(maintenance.gpu_restore, self.config,
                                    {'operation': TABLE1_R1})
        self.assertEqual(0, code, output)
        self.assertIn('nvidia-dcgm.service=inactive', output)
        after = len([c for c in self.systemctl_calls() if c.startswith('stop')])
        self.assertEqual(before, after,
                         'a service that already read inactive was stopped again')

    def test_an_inactive_original_that_cannot_be_stopped_is_a_failure(self):
        (self.stub_state / 'unit-state.txt').write_text('inactive\n')
        self.capture(maintenance.gpu_prepare, self.config,
                     {'operation': TABLE1_R1})
        (self.stub_state / 'unit-state.txt').write_text('active\n')
        (self.stub_state / 'refuse-stop').write_text('')
        code, output = self.capture(maintenance.gpu_restore, self.config,
                                    {'operation': TABLE1_R1})
        self.assertEqual(4, code, output)
        self.assertIn('recorded it inactive', output)
        self.assertIn('preserve this state', output)

    def test_an_active_original_that_does_not_come_back_is_a_failure(self):
        """The existing nonzero result for a failed restart is preserved."""
        self.capture(maintenance.gpu_prepare, self.config,
                     {'operation': TABLE1_R1})
        (self.stub_state / 'refuse-start').write_text('')
        code, output = self.capture(maintenance.gpu_restore, self.config,
                                    {'operation': TABLE1_R1})
        self.assertEqual(4, code, output)
        self.assertIn('did not return to active', output)

    # -- S4-A: an observation that establishes nothing is not a restoration --
    def test_an_inactive_original_observed_activating_is_a_failure(self):
        """The review's S4-A case, on the branch that returned 0 for it.

        The round recorded `inactive`. After the reboot the unit is `activating` and
        stays there. `activating` is not `inactive`: the rejected code returned 0
        because the word was not exactly `active`, and the controller then resumed a
        node whose telemetry owner was coming up.
        """
        (self.stub_state / 'unit-state.txt').write_text('inactive\n')
        self.capture(maintenance.gpu_prepare, self.config,
                     {'operation': TABLE1_R1})
        (self.stub_state / 'unit-state.txt').write_text('activating\n')
        # The stop leaves it activating too, so the bounded wait ends unsettled.
        (self.stub_state / 'stop-leaves').write_text('activating\n')
        code, output = self.capture(maintenance.gpu_restore, self.config,
                                    {'operation': TABLE1_R1})
        self.assertEqual(4, code, output)
        self.assertIn('does not read inactive', output)
        self.assertIn('preserve this state', output)

    def test_an_inactive_original_that_settles_to_inactive_still_succeeds(self):
        """The bounded wait is a wait, not a refusal: a settling unit is accepted.

        Two readings of `deactivating` and then `inactive`. This is the positive
        control for the test above -- without it, 'refuse anything transitional' would
        pass both.
        """
        (self.stub_state / 'unit-state.txt').write_text('inactive\n')
        self.capture(maintenance.gpu_prepare, self.config,
                     {'operation': TABLE1_R1})
        self.script_is_active('deactivating', 'deactivating', 'inactive')
        (self.stub_state / 'unit-state.txt').write_text('inactive\n')
        code, output = self.capture(maintenance.gpu_restore, self.config,
                                    {'operation': TABLE1_R1})
        self.assertEqual(0, code, output)
        self.assertIn('nvidia-dcgm.service=inactive', output)

    def test_an_active_original_observed_activating_is_a_failure(self):
        """The same rule on the other branch, which already compared to 'active'.

        Asserted so the pair is symmetric and so a future edit cannot relax one side
        of it: an `activating` unit is not a restored active service.
        """
        self.capture(maintenance.gpu_prepare, self.config,
                     {'operation': TABLE1_R1})
        (self.stub_state / 'start-leaves').write_text('activating\n')
        code, output = self.capture(maintenance.gpu_restore, self.config,
                                    {'operation': TABLE1_R1})
        self.assertEqual(4, code, output)
        self.assertIn('did not return to active', output)

    def test_an_unreadable_observation_is_a_failure_not_an_inactive_unit(self):
        """An empty answer is not a state.

        `systemctl is-active` printing nothing is a failed invocation. The rejected
        code's `state != 'active'` test read that as 'not running, nothing to do'.
        """
        (self.stub_state / 'unit-state.txt').write_text('inactive\n')
        self.capture(maintenance.gpu_prepare, self.config,
                     {'operation': TABLE1_R1})
        (self.stub_state / 'unit-state.txt').write_text('EMPTY\n')
        (self.stub_state / 'stop-leaves').write_text('EMPTY\n')
        code, output = self.capture(maintenance.gpu_restore, self.config,
                                    {'operation': TABLE1_R1})
        self.assertEqual(4, code, output)
        self.assertIn('unreadable', output)

    def test_a_transient_state_at_prepare_time_refuses_before_any_change(self):
        """The missing case is prevented rather than invented later.

        A unit this helper cannot restore to must not become a round's original
        state. `activating` at prepare time refuses, and nothing is paused or
        disabled -- so there is no round whose restoration has to be guessed.
        """
        (self.stub_state / 'unit-state.txt').write_text('activating\n')
        outcome, output = self.capture(maintenance.gpu_prepare, self.config,
                                       {'operation': TABLE1_R1})
        self.assertIsInstance(outcome, maintenance.Refusal)
        self.assertIn('did not settle', str(outcome))
        self.assertEqual('activating', self.unit_state(),
                         'the unit was changed by a refused preparation')
        self.assertEqual('Enabled', self.persistence(),
                         'persistence was changed by a refused preparation')
        self.assertFalse((self.records / 'native-dcgm-state.txt').exists(),
                         'an unrestorable state was recorded as an original')

    def test_a_failed_unit_at_prepare_time_refuses_before_any_change(self):
        """`failed` has no defined restoration, so no round starts on it.

        The rejected code recorded it and then reported 'left as it is' at restore
        time whatever the unit was doing by then, including `active`.
        """
        (self.stub_state / 'unit-state.txt').write_text('failed\n')
        outcome, _ = self.capture(maintenance.gpu_prepare, self.config,
                                  {'operation': TABLE1_R1})
        self.assertIsInstance(outcome, maintenance.Refusal)
        self.assertIn('no defined restoration', str(outcome))
        self.assertEqual('failed', self.unit_state())
        self.assertFalse((self.records / 'native-dcgm-state.txt').exists())

    def test_an_unreadable_state_at_prepare_time_refuses(self):
        (self.stub_state / 'unit-state.txt').write_text('EMPTY\n')
        outcome, _ = self.capture(maintenance.gpu_prepare, self.config,
                                  {'operation': TABLE1_R1})
        self.assertIsInstance(outcome, maintenance.Refusal)
        self.assertIn('Unexpected', str(outcome))
        self.assertFalse((self.records / 'native-dcgm-state.txt').exists())

    # -- S4 expected-unit validity: `inactive` from a unit that is not there --
    def test_a_missing_unit_at_prepare_time_is_not_a_valid_inactive_original(self):
        """The review's S4 unit-validity case, on the recording side.

        A unit that does not exist reports ActiveState=inactive, and
        `systemctl is-active` prints the word `inactive` for it with rc 4 -- measured
        on the target (runs/rework7-read-telemetry-semantics/output.log:5) and on
        systemd 255.4 (runs/rework9-read-loadstate/output.log). Recording that as
        this round's original state would mean the round starts with an original
        state nothing established, and its restoration would later report success for
        a unit that was never there.
        """
        self.set_load_state('not-found')
        outcome, output = self.capture(maintenance.gpu_prepare, self.config,
                                       {'operation': TABLE1_R1})
        self.assertIsInstance(outcome, maintenance.Refusal)
        self.assertIn('LoadState', str(outcome))
        self.assertIn('not-found', str(outcome))
        self.assertFalse((self.records / 'native-dcgm-state.txt').exists(),
                         'a nonexistent unit was recorded as an inactive original')
        self.assertFalse((self.records / 'selected-persistence-mode.txt').exists(),
                         'a refused preparation still wrote a record')
        self.assertEqual('Enabled', self.persistence(),
                         'persistence was changed by a refused preparation')

    def test_a_masked_unit_at_prepare_time_is_refused_the_same_way(self):
        """Same word, same failure class, different LoadState.

        A masked unit reports masked/inactive and `is-active` exits 3 rather than 4,
        which is exactly why the return code cannot be the test. This is the same
        failure class as the missing unit, so it is asserted on the same path rather
        than only on the one the review named.
        """
        self.set_load_state('masked')
        outcome, _ = self.capture(maintenance.gpu_prepare, self.config,
                                  {'operation': TABLE1_R1})
        self.assertIsInstance(outcome, maintenance.Refusal)
        self.assertIn('masked', str(outcome))
        self.assertFalse((self.records / 'native-dcgm-state.txt').exists())

    def test_a_valid_inactive_unit_at_prepare_time_is_still_recorded(self):
        """The positive control: a genuinely stopped unit that exists is fine.

        Without this, 'refuse every inactive' would pass the two tests above. The
        distinction is LoadState, not the word and not the return code.
        """
        (self.stub_state / 'unit-state.txt').write_text('inactive\n')
        self.set_load_state('loaded')
        code, output = self.capture(maintenance.gpu_prepare, self.config,
                                    {'operation': TABLE1_R1})
        self.assertEqual(0, code, output)
        self.assertEqual('inactive',
                         json.loads((self.records / 'native-dcgm-state.txt')
                                    .read_text())['value'])

    def test_an_active_unit_at_prepare_time_is_still_recorded_and_paused(self):
        """The other positive control, on the branch that mutates the unit."""
        self.set_load_state('loaded')
        code, output = self.capture(maintenance.gpu_prepare, self.config,
                                    {'operation': TABLE1_R1})
        self.assertEqual(0, code, output)
        self.assertEqual('active',
                         json.loads((self.records / 'native-dcgm-state.txt')
                                    .read_text())['value'])
        self.assertEqual('inactive', self.unit_state())

    def test_a_unit_that_vanishes_before_restoration_is_not_a_restored_inactive(self):
        """The review's S4 unit-validity case, on the restoring side.

        A round records a valid `inactive` original. By recovery time the expected
        unit is gone -- an image change, a package removal, a mask. The word read back
        is still `inactive`, so the rejected helper skipped the stop, observed 'the
        same word' and returned 0; the controller then accepted that as a completed
        restoration. Nothing in that sequence established that the unit this round
        recorded still exists.
        """
        (self.stub_state / 'unit-state.txt').write_text('inactive\n')
        self.set_load_state('loaded')
        code, _ = self.capture(maintenance.gpu_prepare, self.config,
                               {'operation': TABLE1_R1})
        self.assertEqual(0, code)
        self.set_load_state('not-found')
        code, output = self.capture(maintenance.gpu_restore, self.config,
                                    {'operation': TABLE1_R1})
        self.assertEqual(4, code, output)
        self.assertIn('is not a loaded unit', output)
        self.assertIn('preserve this state', output)
        self.assertIn('LoadState=not-found', output)

    def test_a_unit_that_vanishes_before_an_active_restoration_also_fails(self):
        """The same rule on the recorded-active branch.

        `systemctl start` on a nonexistent unit fails, and the reading afterwards is
        the same `inactive` word. The failure must name the unit, not just the state.
        """
        self.set_load_state('loaded')
        self.capture(maintenance.gpu_prepare, self.config,
                     {'operation': TABLE1_R1})
        self.set_load_state('masked')
        code, output = self.capture(maintenance.gpu_restore, self.config,
                                    {'operation': TABLE1_R1})
        self.assertEqual(4, code, output)
        self.assertIn('is not a loaded unit', output)
        self.assertIn('LoadState=masked', output)

    def test_a_valid_unit_still_restores_after_the_unit_check(self):
        """The positive control for the two restoration tests above.

        Both original states still restore on a loaded unit, so the new check is a
        validity requirement rather than a refusal of restoration itself.
        """
        self.set_load_state('loaded')
        self.capture(maintenance.gpu_prepare, self.config,
                     {'operation': TABLE1_R1})
        code, output = self.capture(maintenance.gpu_restore, self.config,
                                    {'operation': TABLE1_R1})
        self.assertEqual(0, code, output)
        self.assertEqual('active', self.unit_state())
        self.assertIn('LoadState=loaded', output)

    def test_the_observation_reads_the_supported_property_interface(self):
        """What the helper actually asked systemd, on the calls the stub recorded.

        The finding is that `is-active` alone cannot answer the question, so the
        repair has to consult an interface that reports unit presence. This asserts
        the request that was made rather than only its outcome.
        """
        self.set_load_state('loaded')
        self.capture(maintenance.gpu_prepare, self.config,
                     {'operation': TABLE1_R1})
        calls = self.systemctl_calls()
        shows = [c for c in calls if c.startswith('show ')]
        self.assertTrue(shows, f'no property query was made: {calls}')
        self.assertIn('-p LoadState', shows[0])
        self.assertIn('-p ActiveState', shows[0])
        self.assertIn(maintenance.TELEMETRY_UNIT, shows[0])

    def test_a_return_code_is_not_read_as_a_state(self):
        """rc 4 does not mean 'valid inactive', and rc 3 does not mean 'masked'.

        The three cases that share the word `inactive` are separated here on the same
        function, with the return codes the measured build produces: 3 for a stopped
        loaded unit, 4 for not-found, 3 for masked. Only the loaded one is accepted.
        """
        for load, accepted in (('loaded', True), ('not-found', False),
                               ('masked', False)):
            with self.subTest(load_state=load):
                for name in ('native-dcgm-state.txt',
                             'selected-persistence-mode.txt'):
                    path = self.records / name
                    if path.exists():
                        path.unlink()
                (self.stub_state / 'unit-state.txt').write_text('inactive\n')
                (self.stub_state / 'persistence-mode.txt').write_text('Enabled\n')
                self.set_load_state(load)
                outcome, output = self.capture(maintenance.gpu_prepare, self.config,
                                               {'operation': TABLE1_R1})
                if accepted:
                    self.assertEqual(0, outcome, output)
                else:
                    self.assertIsInstance(outcome, maintenance.Refusal)

    def test_a_transient_unit_is_still_waited_on_when_the_unit_is_loaded(self):
        """The unit check must not short-circuit the settle wait.

        A loaded unit reading `deactivating` twice and then `inactive` is still
        accepted, which is the S4-A behaviour the review accepted and this repair must
        preserve.
        """
        (self.stub_state / 'unit-state.txt').write_text('inactive\n')
        self.set_load_state('loaded')
        self.capture(maintenance.gpu_prepare, self.config,
                     {'operation': TABLE1_R1})
        self.script_is_active('deactivating', 'deactivating', 'inactive')
        code, output = self.capture(maintenance.gpu_restore, self.config,
                                    {'operation': TABLE1_R1})
        self.assertEqual(0, code, output)
        self.assertIn('nvidia-dcgm.service=inactive', output)

    # -- a repeat of one operation's preparation must not overwrite its originals --
    def test_a_repeated_prepare_keeps_the_first_recorded_originals(self):
        """The residual the review's reconciled comments name.

        A helper reachable through a forced command must be idempotent per operation:
        the first accepted call is the only one that saw the node before this round
        changed it. By the second call telemetry is already `inactive` and persistence
        already `Disabled`, so re-recording would replace the round's original state
        with the state the round itself produced -- and restoration would then put back
        `inactive`/`Disabled` as though that were how the node was found. That is
        silent, permanent loss of the original.

        What is NOT claimed here is a caller that retries: the shipped coordinator does
        not re-send `gpu-prepare` for a round in progress. `DeviceSession.start` refuses
        or returns `already_started`, and recovery calls `gpu-restore`.
        """
        self.set_load_state('loaded')
        code, first = self.capture(maintenance.gpu_prepare, self.config,
                                   {'operation': TABLE1_R1})
        self.assertEqual(0, code, first)
        self.assertIn('recorded persistence_mode=Enabled '
                      'nvidia-dcgm.service=active', first)
        # The node as the first call left it: this is what a naive re-read would see.
        self.assertEqual('inactive', self.unit_state())
        self.assertEqual('Disabled', self.persistence())
        recorded_before = {
            name: json.loads((self.records / name).read_text())
            for name in ('native-dcgm-state.txt', 'selected-persistence-mode.txt')}

        code, second = self.capture(maintenance.gpu_prepare, self.config,
                                    {'operation': TABLE1_R1})
        self.assertEqual(0, code, second)
        recorded_after = {
            name: json.loads((self.records / name).read_text())
            for name in ('native-dcgm-state.txt', 'selected-persistence-mode.txt')}
        self.assertEqual('active', recorded_after['native-dcgm-state.txt']['value'],
                         'the repeat recorded the state this round itself produced')
        self.assertEqual('Enabled',
                         recorded_after['selected-persistence-mode.txt']['value'])
        self.assertEqual(recorded_before, recorded_after,
                         'the repeat rewrote the original-state records')
        self.assertIn('kept from this operation', second)

        # And restoration still puts back what the node was FOUND in.
        code, output = self.capture(maintenance.gpu_restore, self.config,
                                    {'operation': TABLE1_R1})
        self.assertEqual(0, code, output)
        self.assertEqual('active', self.unit_state())
        self.assertEqual('Enabled', self.persistence())

    # -- S4 provenance: a lost operation token must not overwrite the original --
    def test_a_repeat_whose_record_lost_its_operation_does_not_recapture(self):
        """The review's fourth finding, on the real helper's record files.

        One record survives with this operation's token; the other has lost its
        `operation` field -- a truncated write, a partial restore from a backup, a
        hand-edit. A record naming no operation cannot be attributed to another round,
        so it may be this one's own, and by now telemetry is paused and persistence
        disabled. Recapturing would write the state this round PRODUCED as the state it
        FOUND, permanently, and the later restoration would report success for putting
        back `inactive`/`Disabled`.

        The behavioural assertions come first: what the records hold afterwards, and
        what a restoration then does.
        """
        self.set_load_state('loaded')
        code, _ = self.capture(maintenance.gpu_prepare, self.config,
                               {'operation': TABLE1_R1})
        self.assertEqual(0, code)
        self.assertEqual('inactive', self.unit_state())
        self.assertEqual('Disabled', self.persistence())
        kept = json.loads((self.records / 'native-dcgm-state.txt').read_text())
        # The persistence record loses its operation field, keeping its value.
        mode_record = json.loads(
            (self.records / 'selected-persistence-mode.txt').read_text())
        mode_record.pop('operation')
        (self.records / 'selected-persistence-mode.txt').write_text(
            json.dumps(mode_record))

        outcome, output = self.capture(maintenance.gpu_prepare, self.config,
                                       {'operation': TABLE1_R1})
        # The surviving original is untouched and no replacement was invented.
        self.assertEqual(kept,
                         json.loads((self.records / 'native-dcgm-state.txt')
                                    .read_text()),
                         'the surviving original was overwritten')
        self.assertNotIn('recorded persistence_mode=Disabled', output,
                         'the state this round produced was written as the original')
        self.assertIsInstance(outcome, maintenance.Refusal)
        self.assertIn('can no longer be read', str(outcome))

    def test_a_repeat_after_both_records_vanish_does_not_look_like_a_first_call(self):
        """Both records gone is not a fresh operation.

        With no records at all, the rejected code could not tell this from a first
        call, so it observed the already-changed node and wrote `inactive`/`Disabled`
        as the round's original state. The capture marker is written before the first
        record and survives, so the repeat knows a capture happened here and refuses:
        the originals are gone and nothing on the node can reconstruct them.
        """
        self.set_load_state('loaded')
        code, _ = self.capture(maintenance.gpu_prepare, self.config,
                               {'operation': TABLE1_R1})
        self.assertEqual(0, code)
        for name in ('native-dcgm-state.txt', 'selected-persistence-mode.txt'):
            (self.records / name).unlink()

        outcome, output = self.capture(maintenance.gpu_prepare, self.config,
                                       {'operation': TABLE1_R1})
        self.assertFalse((self.records / 'native-dcgm-state.txt').exists(),
                         'the round recaptured its own effect as the original state')
        self.assertFalse((self.records / 'selected-persistence-mode.txt').exists())
        self.assertIsInstance(outcome, maintenance.Refusal)
        self.assertIn('can no longer be read', str(outcome))

    def test_the_capture_marker_is_written_before_the_first_record(self):
        """The evidence has to exist before the side effects, or it cannot help.

        A marker written after the records could not distinguish 'captured, record
        lost' from 'never captured' in exactly the case that matters -- a crash between
        the two writes. Asserted on the ordering the helper actually performs.
        """
        self.set_load_state('loaded')
        seen = []
        real_write = maintenance._write_record

        def watching(name, value, operation):
            seen.append(name)
            return real_write(name, value, operation)

        maintenance._write_record = watching
        try:
            code, _ = self.capture(maintenance.gpu_prepare, self.config,
                                   {'operation': TABLE1_R1})
        finally:
            maintenance._write_record = real_write
        self.assertEqual(0, code)
        self.assertEqual(maintenance._capture_marker_name(TABLE1_R1), seen[0],
                         f'the capture marker was not written first: {seen}')
        self.assertIn('native-dcgm-state.txt', seen)

    def test_one_operation_s_capture_marker_does_not_answer_for_another(self):
        """The marker is per operation, so a second round cannot erase the first's.

        A single marker naming the latest operation would be overwritten by a second
        round's capture, and the first round's repeat would then read 'never captured'
        and be free to recapture from a node it had already changed. Each operation has
        its own file, so table 2's capture leaves table 1's evidence intact.
        """
        self.set_load_state('loaded')
        self.capture(maintenance.gpu_prepare, self.config,
                     {'operation': TABLE1_R1})
        self.assertTrue(maintenance._has_captured_before(TABLE1_R1))
        self.assertFalse(maintenance._has_captured_before(TABLE2_R1))
        # Table 2 takes the node next and captures its own originals.
        self.capture(maintenance.gpu_prepare, self.config,
                     {'operation': TABLE2_R1})
        self.assertTrue(maintenance._has_captured_before(TABLE2_R1))
        self.assertTrue(maintenance._has_captured_before(TABLE1_R1),
                        'a later operation erased the earlier one\'s capture evidence')
        # So table 1's own repeat still refuses rather than recapturing: its records now
        # belong to table 2, and its own original is gone.
        outcome, _ = self.capture(maintenance.gpu_prepare, self.config,
                                  {'operation': TABLE1_R1})
        self.assertIsInstance(outcome, maintenance.Refusal)
        record = json.loads((self.records / 'native-dcgm-state.txt').read_text())
        self.assertEqual(TABLE2_R1, record['operation'],
                         'table 1\'s repeat overwrote table 2\'s record')

    def test_an_untagged_record_cannot_be_attributed_to_another_round(self):
        """PI contract: no marker does not prove that an untagged record is foreign.

        Previously this test required overwrite. That was the reviewed defect,
        not a valid positive control. Even an explicitly fresh request must refuse
        this ambiguity; the tagged-foreign and empty-first controls remain.
        """
        self.set_load_state('loaded')
        (self.records / 'selected-persistence-mode.txt').write_text(
            json.dumps({'value': 'Enabled'}))
        before = (self.records / 'selected-persistence-mode.txt').read_text()
        code, output = self.capture(maintenance.gpu_prepare, self.config,
                                    {'operation': TABLE1_R1, 'fresh_capture': True})
        self.assertIsInstance(code, maintenance.Refusal, output)
        self.assertEqual(before, (self.records / 'selected-persistence-mode.txt').read_text())
        self.assertEqual('active', self.unit_state())
        self.assertEqual('Enabled', self.persistence())
        self.assertFalse((self.records / 'native-dcgm-state.txt').exists())

    def test_a_complete_prior_record_is_still_usable_by_its_own_operation(self):
        """The other positive control: an intact pair still resumes and restores."""
        self.set_load_state('loaded')
        self.capture(maintenance.gpu_prepare, self.config,
                     {'operation': TABLE1_R1})
        before = {name: json.loads((self.records / name).read_text())
                  for name in ('native-dcgm-state.txt',
                               'selected-persistence-mode.txt')}
        code, output = self.capture(maintenance.gpu_prepare, self.config,
                                    {'operation': TABLE1_R1})
        self.assertEqual(0, code, output)
        self.assertEqual(before,
                         {name: json.loads((self.records / name).read_text())
                          for name in ('native-dcgm-state.txt',
                                       'selected-persistence-mode.txt')})
        code, output = self.capture(maintenance.gpu_restore, self.config,
                                    {'operation': TABLE1_R1})
        self.assertEqual(0, code, output)
        self.assertEqual('active', self.unit_state())
        self.assertEqual('Enabled', self.persistence())

    # -- S4 repeat preparation: decide from the node, not from the record --
    def test_a_repeat_prepares_a_node_that_drifted_from_its_recorded_original(self):
        """The review's fifth finding, on the branch that reported success.

        The first call records `inactive`/`Disabled` -- a node that was already stopped
        and already had persistence off -- and therefore performs neither action.
        Something then starts the service and re-enables persistence. The repeat must
        pause and disable again, because what needs doing is a fact about the node now;
        the rejected code read the RECORDED values here and returned 0 with the node
        left active and Enabled, so the device removal that follows would run against a
        GPU still held by telemetry.

        The recorded originals must survive unchanged: they are still what restoration
        has to put back.
        """
        (self.stub_state / 'unit-state.txt').write_text('inactive\n')
        (self.stub_state / 'persistence-mode.txt').write_text('Disabled\n')
        self.set_load_state('loaded')
        code, first = self.capture(maintenance.gpu_prepare, self.config,
                                   {'operation': TABLE1_R1})
        self.assertEqual(0, code, first)
        self.assertEqual('inactive',
                         json.loads((self.records / 'native-dcgm-state.txt')
                                    .read_text())['value'])
        recorded = {name: json.loads((self.records / name).read_text())
                    for name in ('native-dcgm-state.txt',
                                 'selected-persistence-mode.txt')}

        # The node drifts: the service comes back and persistence is re-enabled.
        (self.stub_state / 'unit-state.txt').write_text('active\n')
        (self.stub_state / 'persistence-mode.txt').write_text('Enabled\n')

        code, second = self.capture(maintenance.gpu_prepare, self.config,
                                    {'operation': TABLE1_R1})
        # The behavioural consequence first: the node is actually prepared.
        self.assertEqual('inactive', self.unit_state(),
                         'the repeat left telemetry running')
        self.assertEqual('Disabled', self.persistence(),
                         'the repeat left persistence enabled')
        self.assertEqual(0, code, second)
        # And the originals are still the ones the first call observed.
        self.assertEqual(recorded,
                         {name: json.loads((self.records / name).read_text())
                          for name in ('native-dcgm-state.txt',
                                       'selected-persistence-mode.txt')},
                         'the repeat rewrote the recorded originals')
        code, output = self.capture(maintenance.gpu_restore, self.config,
                                    {'operation': TABLE1_R1})
        self.assertEqual(0, code, output)
        self.assertEqual('inactive', self.unit_state(),
                         'restoration did not put back the recorded inactive original')
        self.assertEqual('Disabled', self.persistence())

    def test_a_preparation_that_cannot_stop_telemetry_does_not_report_success(self):
        """Prepared is an observation, not an intention.

        The stop is issued and RETURNS SUCCESS, and the unit is still active
        afterwards -- a restart triggered by something else, a unit that comes straight
        back. Returning 0 there would tell the controller the GPU is free while the
        telemetry owner still holds it, and the device removal that follows requires
        exactly the opposite (device-fault.sh refuses an idle removal while a compute
        process holds the GPU). A stop that exits nonzero already raised before this
        change; this is the case that did not.
        """
        self.set_load_state('loaded')
        (self.stub_state / 'stop-leaves').write_text('active\n')
        outcome, output = self.capture(maintenance.gpu_prepare, self.config,
                                       {'operation': TABLE1_R1})
        self.assertEqual('active', self.unit_state(),
                         'this test needs the unit to stay active to have any force')
        self.assertIsInstance(outcome, maintenance.Refusal)
        self.assertIn('did not stop', str(outcome))
        # The original was recorded before the attempt, so restoration is still defined.
        self.assertEqual('active',
                         json.loads((self.records / 'native-dcgm-state.txt')
                                    .read_text())['value'])

    def test_a_repeated_prepare_still_re_applies_the_pause(self):
        """Idempotent is not inert.

        The retry exists because the first call's effect may not have completed. If
        something restarted the service in between, the repeat must pause it again --
        while still keeping the original it recorded the first time.
        """
        self.set_load_state('loaded')
        self.capture(maintenance.gpu_prepare, self.config,
                     {'operation': TABLE1_R1})
        (self.stub_state / 'unit-state.txt').write_text('active\n')
        code, output = self.capture(maintenance.gpu_prepare, self.config,
                                    {'operation': TABLE1_R1})
        self.assertEqual(0, code, output)
        self.assertEqual('inactive', self.unit_state(),
                         'the repeat did not re-apply the pause')
        self.assertEqual('active',
                         json.loads((self.records / 'native-dcgm-state.txt')
                                    .read_text())['value'])

    def test_another_operation_does_not_inherit_this_one_s_records(self):
        """The keep is per operation, not per file.

        A different operation's `gpu-prepare` must take its own observation rather
        than treat the records it found as its own -- otherwise the S4-B repair would
        be undone by the idempotence repair.
        """
        self.set_load_state('loaded')
        self.capture(maintenance.gpu_prepare, self.config,
                     {'operation': TABLE1_R1})
        self.assertEqual('inactive', self.unit_state())
        code, output = self.capture(maintenance.gpu_prepare, self.config,
                                    {'operation': TABLE2_R1})
        self.assertEqual(0, code, output)
        self.assertNotIn('kept from this operation', output)
        record = json.loads((self.records / 'native-dcgm-state.txt').read_text())
        self.assertEqual(TABLE2_R1, record['operation'])
        # Table 2 observed the node as IT found it: telemetry already stopped.
        self.assertEqual('inactive', record['value'])

    def test_a_lost_half_of_this_round_s_capture_is_refused_not_re_taken(self):
        """A record this round wrote and lost is not re-recorded.

        Losing one of the two records after the round has already changed the node is
        the case that must NOT be repaired by observing again: by then telemetry reads
        `inactive` and persistence reads `Disabled` because this round made them so, so
        a fresh reading would write the round's own effect as its original state and
        the later restoration would report success for putting back the wrong thing.
        That is the loss becoming permanent and invisible, which is the defect class
        this whole finding is about. The round stops with its drain in place instead,
        and names which record is unusable.
        """
        self.set_load_state('loaded')
        self.capture(maintenance.gpu_prepare, self.config,
                     {'operation': TABLE1_R1})
        self.assertEqual('inactive', self.unit_state())
        self.assertEqual('Disabled', self.persistence())
        kept = json.loads((self.records / 'native-dcgm-state.txt').read_text())
        (self.records / 'selected-persistence-mode.txt').unlink()

        outcome, output = self.capture(maintenance.gpu_prepare, self.config,
                                       {'operation': TABLE1_R1})
        self.assertIsInstance(outcome, maintenance.Refusal)
        self.assertIn('the persistence mode can no longer be read', str(outcome))
        self.assertIn('what this round did rather than what it found', str(outcome))
        # The surviving record is untouched, and no replacement was invented.
        self.assertEqual(kept,
                         json.loads((self.records / 'native-dcgm-state.txt')
                                    .read_text()))
        self.assertFalse((self.records / 'selected-persistence-mode.txt').exists(),
                         'the missing original was re-recorded from the changed node')
        self.assertNotIn('recorded persistence_mode=', output)

    def test_an_unreadable_record_of_this_round_is_refused_too(self):
        """Same failure class, reached by corruption rather than by loss.

        A record for this operation that cannot be parsed, or that holds a value this
        helper cannot restore to, is as unusable as an absent one -- and equally must
        not be replaced by a reading of the already-changed node.
        """
        for name, body in (('selected-persistence-mode.txt', 'not json at all'),
                           ('native-dcgm-state.txt',
                            json.dumps({'value': 'activating',
                                        'operation': TABLE1_R1}))):
            with self.subTest(record=name):
                self.set_load_state('loaded')
                for existing in self.records.iterdir():
                    existing.unlink()
                (self.stub_state / 'unit-state.txt').write_text('active\n')
                (self.stub_state / 'persistence-mode.txt').write_text('Enabled\n')
                self.capture(maintenance.gpu_prepare, self.config,
                             {'operation': TABLE1_R1})
                (self.records / name).write_text(body)
                outcome, _ = self.capture(maintenance.gpu_prepare, self.config,
                                          {'operation': TABLE1_R1})
                self.assertIsInstance(outcome, maintenance.Refusal)
                self.assertIn('can no longer be read', str(outcome))

    def test_a_lost_record_does_not_make_restoration_report_success(self):
        """The consequence the refusal exists to prevent, asserted end to end.

        With one record gone, restoration must fail rather than put back a state
        nobody recorded. `gpu_restore` refuses on the missing record, so the node keeps
        its drain and its telemetry stays as this round left it -- reported, not
        silently accepted.
        """
        self.set_load_state('loaded')
        self.capture(maintenance.gpu_prepare, self.config,
                     {'operation': TABLE1_R1})
        (self.records / 'selected-persistence-mode.txt').unlink()
        outcome, _ = self.capture(maintenance.gpu_restore, self.config,
                                  {'operation': TABLE1_R1})
        self.assertIsInstance(outcome, maintenance.Refusal)
        self.assertIn('No recorded persistence mode', str(outcome))
        self.assertEqual('inactive', self.unit_state(),
                         'the unit was changed on the strength of a lost record')

    def test_a_foreign_record_is_still_overwritten_by_a_new_operation(self):
        """The refusal must not extend to another round's records.

        The S4-B repair depends on a new operation being able to record over the
        files a previous round left: those files are not this round's to preserve, and
        the previous round's restore refuses on the operation token. Asserted so the
        idempotence repair does not quietly turn every second round into a refusal.
        """
        self.set_load_state('loaded')
        self.capture(maintenance.gpu_prepare, self.config,
                     {'operation': TABLE1_R1})
        (self.records / 'selected-persistence-mode.txt').unlink()
        # Table 2 starts with only table 1's telemetry record present.
        code, output = self.capture(maintenance.gpu_prepare, self.config,
                                    {'operation': TABLE2_R1})
        self.assertEqual(0, code, output)
        record = json.loads((self.records / 'native-dcgm-state.txt').read_text())
        self.assertEqual(TABLE2_R1, record['operation'])

    def test_the_unit_check_still_applies_to_a_repeat(self):
        """A repeat must not skip the validity check by having records.

        If the expected unit has gone missing between the two calls, the repeat cannot
        establish that the unit its records name is still there.
        """
        self.set_load_state('loaded')
        self.capture(maintenance.gpu_prepare, self.config,
                     {'operation': TABLE1_R1})
        self.set_load_state('not-found')
        outcome, _ = self.capture(maintenance.gpu_prepare, self.config,
                                  {'operation': TABLE1_R1})
        self.assertIsInstance(outcome, maintenance.Refusal)
        self.assertIn('LoadState', str(outcome))

    def test_a_recorded_state_outside_the_restorable_set_is_refused(self):
        """Defence in depth: a record holding `failed` cannot be restored either.

        Prepare refuses to write one, so this hand-writes the record the rejected
        helper would have left and confirms the restore refuses rather than reporting
        'left as it is' and returning 0.
        """
        self.capture(maintenance.gpu_prepare, self.config,
                     {'operation': TABLE1_R1})
        (self.records / 'native-dcgm-state.txt').write_text(json.dumps(
            {'value': 'failed', 'operation': TABLE1_R1}))
        outcome, _ = self.capture(maintenance.gpu_restore, self.config,
                                  {'operation': TABLE1_R1})
        self.assertIsInstance(outcome, maintenance.Refusal)
        self.assertIn('Unusable recorded', str(outcome))
        self.assertEqual('inactive', self.unit_state(),
                         'the unit was changed on the strength of an unusable record')

    # -- the failures that used to pass as success --
    def test_a_missing_telemetry_record_is_a_failure(self):
        """S4's core: absent evidence is not evidence that nothing is needed."""
        self.capture(maintenance.gpu_prepare, self.config,
                     {'operation': TABLE1_R1})
        (self.records / 'native-dcgm-state.txt').unlink()
        outcome, _ = self.capture(maintenance.gpu_restore, self.config,
                                  {'operation': TABLE1_R1})
        self.assertIsInstance(outcome, maintenance.Refusal)
        self.assertIn('No recorded', str(outcome))
        self.assertEqual('inactive', self.unit_state(),
                         'the service was started on the strength of no record')

    def test_an_empty_telemetry_record_is_a_failure(self):
        self.capture(maintenance.gpu_prepare, self.config,
                     {'operation': TABLE1_R1})
        (self.records / 'native-dcgm-state.txt').write_text('')
        outcome, _ = self.capture(maintenance.gpu_restore, self.config,
                                  {'operation': TABLE1_R1})
        self.assertIsInstance(outcome, maintenance.Refusal)
        self.assertIn('is empty', str(outcome))

    def test_an_unrecognised_telemetry_value_is_a_failure(self):
        self.capture(maintenance.gpu_prepare, self.config,
                     {'operation': TABLE1_R1})
        (self.records / 'native-dcgm-state.txt').write_text(
            json.dumps({'value': 'probably-fine', 'operation': TABLE1_R1}))
        outcome, _ = self.capture(maintenance.gpu_restore, self.config,
                                  {'operation': TABLE1_R1})
        self.assertIsInstance(outcome, maintenance.Refusal)
        self.assertIn('Unusable recorded', str(outcome))

    # -- S4-B: the operation, not the round number --
    def test_a_record_from_another_operation_is_a_failure(self):
        """A later round of the same table cannot restore from an earlier one."""
        self.capture(maintenance.gpu_prepare, self.config,
                     {'operation': TABLE1_R1})
        outcome, _ = self.capture(maintenance.gpu_restore, self.config,
                                  {'operation': TABLE1_R2})
        self.assertIsInstance(outcome, maintenance.Refusal)
        self.assertIn('belongs to operation', str(outcome))
        self.assertEqual('inactive', self.unit_state())

    def test_table_two_round_one_cannot_restore_from_table_one_round_one(self):
        """The review's S4-B case, exactly.

        Two assignments share this exercise node, and both number their own rounds
        from 1 (device-session.py State defaults, StateStore files named per
        assignment). Table 1 finishes preparing round 1 and leaves an `active`
        original. Table 2 then starts ITS round 1 while telemetry is inactive, and
        its prepare fails -- or its controller dies -- before writing any record. Its
        recovery still runs `gpu-restore`, because the intent marker was persisted
        first. Under the rejected numeric label the target accepted table 1's
        round-1 record and would have started a service that was NOT active before
        table 2's attempt.
        """
        # Table 1's round 1: telemetry was active, so it is recorded and paused.
        code, _ = self.capture(maintenance.gpu_prepare, self.config,
                               {'operation': TABLE1_R1})
        self.assertEqual(0, code)
        self.assertEqual('active',
                         json.loads((self.records / 'native-dcgm-state.txt')
                                    .read_text())['value'])
        self.assertEqual('inactive', self.unit_state())

        # Table 2's round 1 recovery, whose own prepare never wrote a record.
        outcome, _ = self.capture(maintenance.gpu_restore, self.config,
                                  {'operation': TABLE2_R1})
        self.assertIsInstance(outcome, maintenance.Refusal)
        self.assertIn('belongs to operation', str(outcome))
        self.assertIn(TABLE1_R1, str(outcome))
        self.assertEqual('inactive', self.unit_state(),
                         'telemetry was started from another table\'s record')

        # The negative control: the rejected label DID match, because both rounds
        # are numbered 1. Reconstructed on the same on-disk record.
        rejected_label_of = lambda token: token.split('/')[1]
        self.assertEqual(rejected_label_of(TABLE1_R1),
                         rejected_label_of(TABLE2_R1),
                         'the two tables do not share a round number, so this test '
                         'has no negative control')

        # And table 1's own restore, with the same token, still works.
        code, output = self.capture(maintenance.gpu_restore, self.config,
                                    {'operation': TABLE1_R1})
        self.assertEqual(0, code, output)
        self.assertEqual('active', self.unit_state())

    def test_a_record_naming_no_operation_is_a_failure(self):
        """Including one written before this change, which named only a round."""
        self.capture(maintenance.gpu_prepare, self.config,
                     {'operation': TABLE1_R1})
        for body in (json.dumps({'value': 'active', 'round': '1'}),
                     json.dumps({'value': 'active'}),
                     'active'):
            with self.subTest(body=body[:40]):
                (self.records / 'native-dcgm-state.txt').write_text(body)
                outcome, _ = self.capture(maintenance.gpu_restore, self.config,
                                          {'operation': TABLE1_R1})
                self.assertIsInstance(outcome, maintenance.Refusal)
                self.assertEqual('inactive', self.unit_state())

    def test_the_same_token_restores_after_a_retry(self):
        """A retry reuses the token, so a second restore attempt still matches."""
        self.capture(maintenance.gpu_prepare, self.config,
                     {'operation': TABLE1_R1})
        (self.stub_state / 'refuse-start').write_text('')
        code, _ = self.capture(maintenance.gpu_restore, self.config,
                               {'operation': TABLE1_R1})
        self.assertEqual(4, code)
        (self.stub_state / 'refuse-start').unlink()
        code, output = self.capture(maintenance.gpu_restore, self.config,
                                    {'operation': TABLE1_R1})
        self.assertEqual(0, code, output)
        self.assertEqual('active', self.unit_state())

    def test_a_missing_persistence_record_is_still_a_failure(self):
        self.capture(maintenance.gpu_prepare, self.config,
                     {'operation': TABLE1_R1})
        (self.records / 'selected-persistence-mode.txt').unlink()
        outcome, _ = self.capture(maintenance.gpu_restore, self.config,
                                  {'operation': TABLE1_R1})
        self.assertIsInstance(outcome, maintenance.Refusal)
        self.assertIn('persistence mode', str(outcome))

    def test_the_records_are_read_before_any_device_change(self):
        """A refusal must not leave the GPU half-restored."""
        self.capture(maintenance.gpu_prepare, self.config,
                     {'operation': TABLE1_R1})
        self.assertEqual('Disabled', self.persistence())
        (self.records / 'native-dcgm-state.txt').unlink()
        outcome, _ = self.capture(maintenance.gpu_restore, self.config,
                                  {'operation': TABLE1_R1})
        self.assertIsInstance(outcome, maintenance.Refusal)
        self.assertEqual('Disabled', self.persistence(),
                         'persistence was changed before the records were checked')

    def test_the_rejected_condition_would_have_returned_success(self):
        """The negative control, on the same on-disk states.

        The rejected code restored telemetry only under
        `telemetry_file.is_file() and read_text().strip() == 'active'` and returned
        0 otherwise. Evaluated here on each state the tests above refuse: it takes
        the "nothing to do" path in every one, and returns 0.
        """
        telemetry = self.records / 'native-dcgm-state.txt'
        self.capture(maintenance.gpu_prepare, self.config,
                     {'operation': TABLE1_R1})
        cases = {}
        cases['from another operation'] = telemetry.read_text()
        cases['empty'] = ''
        cases['unrecognised'] = json.dumps({'value': 'probably-fine',
                                            'operation': TABLE2_R1})
        cases['naming only a round'] = json.dumps({'value': 'active', 'round': '1'})
        for name, body in list(cases.items()) + [('absent', None)]:
            with self.subTest(case=name):
                if body is None:
                    if telemetry.exists():
                        telemetry.unlink()
                    rejected_restores = False
                else:
                    telemetry.write_text(body)
                    rejected_restores = (telemetry.is_file()
                                         and telemetry.read_text().strip() == 'active')
                self.assertFalse(
                    rejected_restores,
                    f'the rejected condition DID restore for {name}, so this suite '
                    f'has no negative control')
                # It returned 0 in all of these, which is the defect: the caller
                # then resumed the node.

    # -- the grammar the round travels through --
    def test_the_maintenance_grammar_accepts_and_bounds_the_operation(self):
        for argv, expected in (
                (['gpu-restore', '--operation', TABLE1_R1],
                 {'operation': TABLE1_R1}),
                (['gpu-prepare', '--operation', TABLE2_R1],
                 {'operation': TABLE2_R1})):
            with self.subTest(argv=argv):
                action, options = maintenance.parse(argv)
                self.assertEqual(argv[0], action)
                self.assertEqual(expected, options)
        for argv in (['gpu-restore', '--operation'],
                     ['gpu-restore'],
                     ['gpu-prepare'],
                     ['gpu-restore', '--round', '7'],
                     ['gpu-restore', TABLE1_R1],
                     ['gpu-prepare', '--confirm', 'x']):
            with self.subTest(argv=argv):
                with self.assertRaises(maintenance.Refusal):
                    maintenance.parse(argv)

    def test_a_duplicated_operation_option_is_refused_on_its_own(self):
        """S4-C: the duplicate itself, with no other invalid option present.

        The rejected parser assigned each occurrence in turn, so the last value
        silently won, and the test that claimed to cover this passed because of an
        unrelated `--extra`.
        """
        with self.assertRaises(maintenance.Refusal) as caught:
            maintenance.parse(['gpu-restore', '--operation', TABLE1_R1,
                               '--operation', TABLE2_R1])
        self.assertIn('more than once', str(caught.exception))
        # The negative control: the rejected parser accepted exactly this input and
        # kept the SECOND value.
        rejected = {}
        remaining = ['--operation', TABLE1_R1, '--operation', TABLE2_R1]
        while remaining:
            name = remaining.pop(0)
            self.assertEqual('--operation', name)
            rejected['operation'] = remaining.pop(0)
        self.assertEqual(TABLE2_R1, rejected['operation'],
                         'the rejected parser did not silently take the last value, '
                         'so this test has no negative control')

    def test_a_malformed_operation_identifier_is_refused(self):
        for token in ('', 'table-1', 'table-1/1', 'table-1/1/nothex',
                      'table-1/1/' + 'a' * 31, '../1/' + 'a' * 32,
                      'table-1/x/' + 'a' * 32):
            with self.subTest(token=token):
                outcome, _ = self.capture(maintenance.gpu_restore, self.config,
                                         {'operation': token})
                self.assertIsInstance(outcome, maintenance.Refusal)

    def test_the_controller_generates_a_unique_operation_per_start(self):
        """The token really is per-start, and it names the assignment and round.

        Read off the controller's own generator rather than assumed from its name:
        two calls for the same round must differ, and the shape must be the one the
        target's grammar accepts.
        """
        spec = importlib.util.spec_from_file_location('aim344_session_s4', HELPER)
        assert spec is not None and spec.loader is not None
        controller = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(controller)
        handler = controller.DeviceSession.__new__(controller.DeviceSession)
        handler.assignment_id = 'table-2'
        first = handler._new_operation(3)
        second = handler._new_operation(3)
        self.assertNotEqual(first, second, 'the operation identifier is not unique')
        for token in (first, second):
            self.assertTrue(token.startswith('table-2/3/'), token)
            action, options = maintenance.parse(['gpu-prepare', '--operation', token])
            self.assertEqual({'operation': token}, options)


if __name__ == '__main__':
    unittest.main()