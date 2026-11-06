# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Real root filesystem regression for the shared enroot directories.
Cloud/devices are not used. Root effects stay inside a disposable container.
"""
import pathlib
import subprocess
import tempfile
import unittest
import test_safety_rework3 as safety

LAB = pathlib.Path(__file__).resolve().parents[1]
# The boot-time directories prepare-compute.sh installs, after a first enroot run
# as root has left the cache and data levels root-owned and closed.
TMPFILES_PROBE = r'''
set -eu
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null && apt-get install -y -qq systemd >/dev/null
mkdir -p -m 0755 /tmp/enroot/cache /tmp/enroot/data
chmod 0700 /tmp/enroot/data
bash /probe-lines.sh
stat -c '%U %a %n' /tmp/enroot /tmp/enroot/cache /tmp/enroot/data
runuser -u ubuntu -- mkdir /tmp/enroot/user-1000 /tmp/enroot/cache/group-1000 /tmp/enroot/data/user-1000
rm -rf /tmp/enroot
systemd-tmpfiles --create /etc/tmpfiles.d/aim344-enroot.conf
stat -c 'after boot: %U %a %n' /tmp/enroot /tmp/enroot/cache /tmp/enroot/data
'''


class BootTimeEnrootDirectories(unittest.TestCase):
    def test_the_participant_can_use_enroot_after_a_root_run_and_after_a_boot(self):
        runtime = safety.container_runtime()
        if runtime is None:
            self.skipTest('no usable rootful container runtime')
        source = (LAB / 'facilitator/prepare-compute.sh').read_text().splitlines()
        lines = [line for line in source
                 if line.startswith(("printf 'd %s 1777", 'systemd-tmpfiles --create'))]
        self.assertEqual(2, len(lines), lines)
        with tempfile.TemporaryDirectory() as temporary:
            probe = pathlib.Path(temporary) / 'probe.sh'
            probe.write_text(TMPFILES_PROBE)
            (pathlib.Path(temporary) / 'probe-lines.sh').write_text('\n'.join(lines) + '\n')
            result = subprocess.run(
                [runtime, 'run', '--rm', '-v', f'{temporary}/probe.sh:/probe.sh:ro',
                 '-v', f'{temporary}/probe-lines.sh:/probe-lines.sh:ro', safety.IMAGE,
                 'bash', '/probe.sh'], text=True, capture_output=True, timeout=600)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        for prefix in ('', 'after boot: '):
            for path in ('/tmp/enroot', '/tmp/enroot/cache', '/tmp/enroot/data'):
                self.assertIn(f'{prefix}root 1777 {path}\n', result.stdout)
