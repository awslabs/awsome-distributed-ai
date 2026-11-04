# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Real root filesystem regression for the live nested enroot-cache failure.
Cloud/devices are not used. Root effects stay inside a disposable container.
"""
import pathlib
import subprocess
import tempfile
import unittest
import importlib.util
from unittest import mock
import test_safety_rework3 as safety

LAB = pathlib.Path(__file__).resolve().parents[1]
PROBE = r'''
import importlib.util, pathlib, os, stat, subprocess, tempfile
from unittest import mock
from types import SimpleNamespace
spec=importlib.util.spec_from_file_location('bootstrap','/lab/facilitator/replacement-bootstrap.py')
bootstrap=importlib.util.module_from_spec(spec);spec.loader.exec_module(bootstrap)
for implementation in ('bootstrap',):
 for case in ('root0700','missing','symlink','foreign','foreign-parent'):
  with tempfile.TemporaryDirectory(dir='/tmp') as temp:
   parent=pathlib.Path(temp);parent.chmod(0o1777)
   # Bootstrap supports exactly one root-controlled shared parent below /tmp.
   shared=parent
   for name in ('cache','data'):
    p=shared/name
    if case!='missing':p.mkdir(mode=0o700)
   decoy=pathlib.Path(temp)/'decoy';decoy.mkdir(mode=0o700)
   if case=='symlink':
    (shared/'cache').rmdir();(shared/'cache').symlink_to(decoy,target_is_directory=True)
   if case=='foreign':os.chown(shared/'cache',65534,65534)
   if case=='foreign-parent':os.chown(shared,65534,65534)
   def invoke():
    config=SimpleNamespace(read_text=lambda:'ENROOT_RUNTIME_PATH '+str(shared)+'/user-$(id -u)\nENROOT_CACHE_PATH '+str(shared)+'/cache/user-$(id -u)\nENROOT_DATA_PATH '+str(shared)+'/data/user-$(id -u)\n')
    real=pathlib.Path
    with mock.patch.object(bootstrap,'Path',side_effect=lambda value:config if value=='/etc/enroot/enroot.conf' else real(value)):
     with mock.patch.object(bootstrap.sys,'argv',['bootstrap','--prepare-enroot']):
      bootstrap.main()
   if case in ('symlink','foreign','foreign-parent'):
    try:invoke()
    except (RuntimeError,OSError,subprocess.CalledProcessError):pass
    else:raise AssertionError((implementation,case,'did not refuse'))
    assert stat.S_IMODE(decoy.stat().st_mode)==0o700
    if case=='foreign':
     assert (shared/'cache').stat().st_uid==65534
     assert stat.S_IMODE((shared/'cache').stat().st_mode)==0o700
    if case=='foreign-parent':assert shared.stat().st_uid==65534
   else:
    invoke();invoke()
    for name in ('cache','data'):
     p=shared/name
     assert p.stat().st_uid==0 and stat.S_IMODE(p.stat().st_mode)==0o1777,(implementation,case,name)
     # Actual unprivileged child creation, not just a permission-mode assertion.
     pid=os.fork()
     if pid==0:
      try:
       os.setgid(65534);os.setuid(65534);(p/'user-65534').mkdir();os._exit(0)
      except BaseException:os._exit(1)
     assert os.waitpid(pid,0)[1]==0,(implementation,case,'non-root creation failed')
   print('PASS',implementation,case,flush=True)
print('ALL_NESTED_BOUNDARY_CASES_PASS')
'''


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


class NestedEnrootFilesystem(unittest.TestCase):
    def test_prepare_action_requires_root_and_exact_arguments(self):
        spec = importlib.util.spec_from_file_location('bootstrap_enroot', LAB / 'facilitator/replacement-bootstrap.py')
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with mock.patch.object(module, 'prepare_enroot') as prepare:
            for uid, argv in ((1001, ['bootstrap', '--prepare-enroot']),
                              (0, ['bootstrap', '--prepare-enroot', '/untrusted'])):
                with mock.patch.object(module.os, 'geteuid', return_value=uid), mock.patch.object(module.sys, 'argv', argv):
                    with self.assertRaises(RuntimeError):
                        module.main()
                prepare.assert_not_called()
            with mock.patch.object(module.os, 'geteuid', return_value=0), mock.patch.object(module.sys, 'argv', ['bootstrap', '--prepare-enroot']):
                module.main()
            prepare.assert_called_once_with()

    def test_real_root_and_unprivileged_nested_paths(self):
        runtime = safety.container_runtime()
        if runtime is None:
            self.skipTest('no usable rootful container runtime')
        with tempfile.TemporaryDirectory() as temporary:
            probe = pathlib.Path(temporary) / 'probe.py'
            probe.write_text(PROBE)
            result = subprocess.run(
                [runtime, 'run', '--rm', '-v', f'{LAB}:/lab:ro',
                 '-v', f'{probe}:/probe.py:ro', safety.IMAGE, 'bash', '-ec',
                 'apt-get update -qq && apt-get install -y -qq python3 >/dev/null && python3 /probe.py'],
                text=True, capture_output=True, timeout=300)
            print(result.stdout, result.stderr)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('ALL_NESTED_BOUNDARY_CASES_PASS', result.stdout)
