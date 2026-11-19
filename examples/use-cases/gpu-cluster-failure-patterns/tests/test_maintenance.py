# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Grammar and refusal tests for the target-side maintenance forced command.

These run without hardware. They exercise the parser and configuration checks
that stand between the coordinator's maintenance key and root on the fault
target. Whether a device operation succeeds is the existing helper's business,
and a separate hardware acceptance track.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

LAB = Path(__file__).resolve().parents[1]
HELPER = LAB / 'facilitator' / 'maintenance.py'


def load():
    spec = importlib.util.spec_from_file_location('aim344_maintenance', HELPER)
    if spec is None or spec.loader is None:
        raise unittest.SkipTest(f'Helper not importable: {HELPER}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


maintenance = load()
TOKEN_GPU = 'i-0123456789abcdef0/gpu-remove/GPU-11111111-2222-4333-8444-555555555555'
TOKEN_UNBIND = 'i-0123456789abcdef0/efa-unbind/0000:b0:00.0'
TOKEN_REBIND = 'i-0123456789abcdef0/efa-rebind/0000:b0:00.0'


class Grammar(unittest.TestCase):
    def test_accepted_actions(self):
        for argv, action in [(['inspect'], 'inspect'),
                             (['boot-id'], 'boot-id'),
                             (['restore-runtime'], 'restore-runtime'),
                             (['collect', '0'], 'collect'),
                             (['collect', '6'], 'collect'),
                             (['gpu-remove', '--confirm', TOKEN_GPU], 'gpu-remove'),
                             (['efa-unbind', '--confirm', TOKEN_UNBIND], 'efa-unbind'),
                             (['efa-unbind', '--job', '95', '--confirm', TOKEN_UNBIND],
                              'efa-unbind'),
                             (['efa-rebind', '--confirm', TOKEN_REBIND], 'efa-rebind')]:
            with self.subTest(argv=argv):
                parsed, _ = maintenance.parse(argv)
                self.assertEqual(action, parsed)

    def test_no_free_form_command_is_accepted(self):
        for argv in [['bash'], ['sh', '-c', 'id'], ['/bin/bash'], ['nvidia-smi'],
                     ['scontrol', 'update', 'NodeName=gpu-g7-2', 'State=DOWN'],
                     ['collect', '0;', 'id'], ['inspect', '&&', 'id'],
                     ['gpu-remove', '--confirm', TOKEN_GPU, ';', 'id']]:
            with self.subTest(argv=argv):
                with self.assertRaises(maintenance.Refusal):
                    maintenance.parse(argv)

    def test_empty_request_is_refused(self):
        with self.assertRaises(maintenance.Refusal):
            maintenance.parse([])

    def test_token_must_match_the_requested_operation(self):
        """A gpu-remove token must not authorize an efa-unbind."""
        with self.assertRaises(maintenance.Refusal) as caught:
            maintenance.parse(['efa-unbind', '--confirm', TOKEN_GPU])
        self.assertIn('different operation', str(caught.exception))

    def test_malformed_token_is_refused(self):
        for token in ['nonsense', 'i-abc/gpu-remove/', '/gpu-remove/GPU-1',
                      'i-0123456789abcdef0/reboot/GPU-041fb113',
                      'i-0123456789abcdef0/gpu-remove/0000:zz:00.0']:
            with self.subTest(token=token):
                with self.assertRaises(maintenance.Refusal):
                    maintenance.parse(['gpu-remove', '--confirm', token])

    def test_mutation_without_a_token_is_refused(self):
        for action in ['gpu-remove', 'efa-unbind', 'efa-rebind']:
            with self.subTest(action=action):
                with self.assertRaises(maintenance.Refusal) as caught:
                    maintenance.parse([action])
                self.assertIn('confirmation token', str(caught.exception))

    def test_job_option_only_applies_to_efa_unbind(self):
        with self.assertRaises(maintenance.Refusal):
            maintenance.parse(['gpu-remove', '--job', '95', '--confirm', TOKEN_GPU])

    def test_job_must_be_numeric(self):
        for value in ['abc', '9;id', '-1', '']:
            with self.subTest(value=value):
                with self.assertRaises(maintenance.Refusal):
                    maintenance.parse(['efa-unbind', '--job', value,
                                       '--confirm', TOKEN_UNBIND])

    def test_collect_takes_exactly_one_numeric_check(self):
        for argv in [['collect'], ['collect', '0', '3'], ['collect', 'five'],
                     ['collect', '../0'], ['collect', '100']]:
            with self.subTest(argv=argv):
                with self.assertRaises(maintenance.Refusal):
                    maintenance.parse(argv)

    def test_read_only_actions_take_no_argument(self):
        for argv in [['inspect', 'x'], ['boot-id', '0'], ['restore-runtime', '/']]:
            with self.subTest(argv=argv):
                with self.assertRaises(maintenance.Refusal):
                    maintenance.parse(argv)

    def test_unknown_action_is_refused(self):
        for action in ['gpu-add', 'reboot', 'shutdown', 'efa_unbind', 'Inspect']:
            with self.subTest(action=action):
                with self.assertRaises(maintenance.Refusal):
                    maintenance.parse([action])

    def test_environment_is_fixed_and_carries_the_slurm_bin(self):
        env = maintenance.environment({'slurm_bin': '/opt/aws/pcs/scheduler/slurm-25.05/bin'})
        self.assertIn('/opt/aws/pcs/scheduler/slurm-25.05/bin', env['PATH'])
        for leaked in ['LD_PRELOAD', 'PYTHONPATH', 'BASH_ENV', 'SSH_ORIGINAL_COMMAND']:
            self.assertNotIn(leaked, env)

    def test_assigned_efa_count_is_root_config_not_caller_environment(self):
        from unittest.mock import patch
        config = {'slurm_bin': '/opt/aws/pcs/scheduler/slurm-25.05/bin',
                  'expected_efa_devices': 2}
        with patch.dict(os.environ, HEALTHCHECK_EXPECTED_EFA_COUNT='1', INSTANCE_TYPE='g7.48xlarge'):
            env = maintenance.environment(config)
        self.assertEqual(env['HEALTHCHECK_EXPECTED_EFA_COUNT'], '2')
        self.assertNotIn('INSTANCE_TYPE', env)
        for bad in (0, -1, 33, True, '2', None):
            with self.subTest(bad=bad), self.assertRaises(maintenance.Refusal):
                maintenance.environment(dict(config, expected_efa_devices=bad))

    def test_health_prolog_uses_private_assignment_and_refuses_invalid_count(self):
        import subprocess
        # Relocate this script's fixed read-only config path, then execute its
        # actual assignment-loading prefix as our ordinary test UID.
        prefix = (LAB / 'prejob-prolog.sh').read_text().split('marker=/run/aim344-unhealthy')[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / 'maintenance.json'
            script = root / 'prolog.sh'
            text = prefix.replace('/etc/aim344-maintenance.json', str(config))
            text = text.replace('info.st_uid != 0', f'info.st_uid != {os.getuid()}')
            script.write_text(text + '\nprintf "%s|%s" "${HEALTHCHECK_EXPECTED_EFA_COUNT:-unset}" "${INSTANCE_TYPE:-unset}"\n')
            def invoke():
                return subprocess.run(['bash', str(script)],
                    env=dict(os.environ, HEALTHCHECK_EXPECTED_EFA_COUNT='1', INSTANCE_TYPE='spoofed'),
                    text=True, capture_output=True, timeout=10)
            self.assertEqual(invoke().stdout, 'unset|unset')
            for count in (2, 0, True, '2', 33):
                config.write_text(json.dumps({'expected_efa_devices': count}))
                config.chmod(0o600)
                result = invoke()
                if type(count) is int and count == 2:
                    self.assertEqual((result.returncode, result.stdout), (0, '2|unset'))
                else:
                    self.assertNotEqual(result.returncode, 0)
            config.write_text('{"expected_efa_devices": 2}')
            config.chmod(0o644)
            self.assertNotEqual(invoke().returncode, 0)
            config.unlink()
            config.symlink_to(root / 'missing')
            self.assertNotEqual(invoke().returncode, 0)

    def test_node_preparation_gives_the_prolog_on_both_nodes_the_assigned_count(self):
        # prepare-compute.sh runs on both GPU nodes, prepare-target.py only on
        # the target. Run the count block of the one and the assignment-loading
        # prefix of the Prolog against a relocated configuration path.
        source = (LAB / 'facilitator' / 'prepare-compute.sh').read_text()
        start = source.index('if [[ -n ${AIM344_EXPECTED_EFA_DEVICES:-} ]]; then')
        block = source[start:source.index('\nfi\n', start) + 4]
        prolog = (LAB / 'prejob-prolog.sh').read_text().split('marker=/run/aim344-unhealthy')[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / 'maintenance.json'
            prepare = root / 'prepare.sh'
            prepare.write_text(block.replace('/etc/aim344-maintenance.json', str(config)))
            read = root / 'prolog.sh'
            read.write_text(prolog.replace('/etc/aim344-maintenance.json', str(config))
                            .replace('info.st_uid != 0', f'info.st_uid != {os.getuid()}')
                            + '\nprintf %s "${HEALTHCHECK_EXPECTED_EFA_COUNT:-unset}"\n')
            environment = {k: v for k, v in os.environ.items() if k != 'AIM344_EXPECTED_EFA_DEVICES'}

            def run(script, **extra):
                return subprocess.run(['bash', str(script)], env=dict(environment, **extra),
                                      text=True, capture_output=True, timeout=10)
            self.assertEqual(0, run(prepare).returncode)
            self.assertFalse(config.exists())
            # The peer has no file before preparation.
            result = run(prepare, AIM344_EXPECTED_EFA_DEVICES='2')
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual({'expected_efa_devices': 2}, json.loads(config.read_text()))
            self.assertEqual(0o600, config.stat().st_mode & 0o777)
            self.assertEqual('2', run(read).stdout)
            # On the target the other settings stay.
            config.write_text(json.dumps({'slurm_bin': '/opt/aws/pcs/scheduler/slurm-25.05/bin',
                                          'checks': [0, 2, 3, 6], 'expected_efa_devices': 4}))
            self.assertEqual(0, run(prepare, AIM344_EXPECTED_EFA_DEVICES='2').returncode)
            self.assertEqual({'slurm_bin': '/opt/aws/pcs/scheduler/slurm-25.05/bin',
                              'checks': [0, 2, 3, 6], 'expected_efa_devices': 2},
                             json.loads(config.read_text()))

    def test_environment_carries_the_efa_tools_the_suite_calls(self):
        """Omitting /opt/amazon/efa/bin made check 2 report 'fi_info not found'
        and FAIL on a node whose EFA counts were all correct."""
        env = maintenance.environment({'slurm_bin': '/opt/aws/pcs/scheduler/slurm-25.05/bin'})
        self.assertIn('/opt/amazon/efa/bin', env['PATH'])
        directories = env['PATH'].split(':')
        self.assertLess(directories.index('/opt/amazon/efa/bin'),
                        directories.index('/usr/bin'))


class Configuration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'aim344-maintenance.json'
        self.body = {
            'slurm_bin': '/opt/aws/pcs/scheduler/slurm-25.05/bin',
            'participant_user': 'ubuntu',
            'stage_dir': '/opt/aim344',
            'checks': [0, 2, 3, 6],
            'suite_entry': ('/opt/aim344/device-recovery/candidate-v2/validation/'
                            'gpu-cluster-healthcheck/gpu-healthcheck.sh'),
        }
        self.write()
        self.original = maintenance.CONFIG
        maintenance.CONFIG = self.path

    def write(self):
        self.path.write_text(json.dumps(self.body))
        os.chmod(self.path, 0o600)

    def tearDown(self):
        maintenance.CONFIG = self.original
        self.tmp.cleanup()

    def test_valid_configuration_loads(self):
        config = maintenance.load_config()
        self.assertEqual([0, 2, 3, 6], config['checks'])

    def test_missing_keys_are_named(self):
        for key in ('slurm_bin', 'participant_user', 'stage_dir', 'checks', 'suite_entry'):
            with self.subTest(key=key):
                body = dict(self.body)
                body.pop(key)
                self.path.write_text(json.dumps(body))
                os.chmod(self.path, 0o600)
                with self.assertRaises(maintenance.Refusal) as caught:
                    maintenance.load_config()
                self.assertIn(key, str(caught.exception))

    def test_unexpected_slurm_installation_is_refused(self):
        self.body['slurm_bin'] = '/tmp/evil/bin'
        self.write()
        with self.assertRaises(maintenance.Refusal) as caught:
            maintenance.load_config()
        self.assertIn('Slurm installation', str(caught.exception))

    def test_collect_refuses_a_check_outside_the_node_allowlist(self):
        config = dict(self.body)
        with self.assertRaises(maintenance.Refusal) as caught:
            maintenance.collect('5', config)
        self.assertIn('not enabled', str(caught.exception))

    def test_collect_refuses_a_relative_or_symlinked_suite_entry(self):
        config = dict(self.body, suite_entry='relative/gpu-healthcheck.sh')
        with self.assertRaises(maintenance.Refusal):
            maintenance.collect('0', config)
        victim = Path(self.tmp.name) / 'real.sh'
        victim.write_text('#!/bin/bash\ntrue\n')
        link = Path(self.tmp.name) / 'link.sh'
        link.symlink_to(victim)
        config = dict(self.body, suite_entry=str(link))
        with self.assertRaises(maintenance.Refusal) as caught:
            maintenance.collect('0', config)
        self.assertIn('regular file', str(caught.exception))


class ImageDigest(unittest.TestCase):
    def test_ordered_batches_and_short_reads(self):
        # Cross a batch boundary, with distinct chunks and a partial last chunk.
        data = b''.join(bytes([i]) * (1024 * 1024) for i in range(33)) + b'end'
        with tempfile.TemporaryFile() as image:
            image.write(data)
            image.flush()
            real_pread = os.pread

            def short_read(fd, size, offset):
                return real_pread(fd, min(size, 131071), offset)
            with mock.patch.object(maintenance.os, 'pread', side_effect=short_read):
                self.assertEqual(maintenance._image_digest(image.fileno(), len(data)),
                                 hashlib.sha256(data).hexdigest())
            self.assertEqual(image.tell(), len(data))

    def test_premature_eof_and_io_error_refuse(self):
        with tempfile.TemporaryFile() as image:
            image.write(b'short')
            image.flush()
            with self.assertRaisesRegex(maintenance.Refusal, 'changed size'):
                maintenance._image_digest(image.fileno(), 100)
            with mock.patch.object(maintenance.os, 'pread', side_effect=OSError('read failure')):
                with self.assertRaisesRegex(OSError, 'read failure'):
                    maintenance._image_digest(image.fileno(), 5)


class RootStaging(unittest.TestCase):
    """Recovery's remount-staging step with the root-directory staging policy."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        parent = Path(self.tmp.name) / 'opt'
        self.stage = parent / 'aim344'
        self.stage.mkdir(parents=True)
        parent.chmod(0o755)
        self.stage.chmod(0o755)
        pins = {}
        for name in ('aim344.sqsh', 'nccl-baseline.sqsh'):
            data = f'pinned {name}'.encode()
            (self.stage / name).write_bytes(data)
            (self.stage / name).chmod(0o644)
            pins[name] = hashlib.sha256(data).hexdigest()
        self.config = {'stage_dir': str(self.stage), 'staging_policy': 'root-directory',
                       'staging_image_sha256': pins,
                       'slurm_bin': '/opt/aws/pcs/scheduler/slurm-25.05/bin'}
        self.mount_target = '/'

    def tearDown(self):
        self.tmp.cleanup()

    def staging(self):
        def findmnt(argv, **kwargs):
            self.assertEqual('/usr/bin/findmnt', argv[0])
            return subprocess.CompletedProcess(argv, 0, self.mount_target + '\n', '')
        out = io.StringIO()
        with mock.patch.object(maintenance, 'STAGING_UID', os.getuid()), \
                mock.patch.object(maintenance.subprocess, 'run', side_effect=findmnt), \
                contextlib.redirect_stdout(out):
            code = maintenance.remount_staging(self.config)
        return code, out.getvalue()

    def test_pinned_images_on_the_root_filesystem_need_no_bind_mount(self):
        code, out = self.staging()
        self.assertEqual(0, code)
        self.assertIn('verified pinned root staging', out)

    def test_a_missing_changed_or_writable_image_or_an_overlaid_source_refuses(self):
        for damage in ('missing', 'changed', 'writable', 'overlaid', 'unpinned'):
            with self.subTest(damage=damage):
                self.tearDown()
                self.setUp()
                image = self.stage / 'aim344.sqsh'
                if damage == 'missing':
                    image.unlink()
                if damage == 'changed':
                    image.write_bytes(b'another image')
                if damage == 'writable':
                    image.chmod(0o666)
                if damage == 'overlaid':
                    self.mount_target = str(self.stage)
                if damage == 'unpinned':
                    del self.config['staging_image_sha256']['nccl-baseline.sqsh']
                with self.assertRaises(maintenance.Refusal):
                    self.staging()


class GpuRecoveryQuery(unittest.TestCase):
    """The read-only `gpu-recovery` action: one nvidia-smi query, printed as one
    JSON record with this node's identity. No device is touched."""

    CONFIG = {'slurm_bin': '/opt/aws/pcs/scheduler/slurm-25.05/bin'}
    CSV = ('index, uuid, pci.bus_id, gpu_recovery_action\n'
           '0, GPU-11111111-2222-4333-8444-555555555555, 00000000:57:00.0, None\n')

    def test_the_action_parses_and_takes_no_argument(self):
        self.assertEqual(('gpu-recovery', {}), maintenance.parse(['gpu-recovery']))
        for argv in (['gpu-recovery', '0'], ['gpu-recovery', '--reset'], ['gpu-recovery', '-r']):
            with self.subTest(argv=argv):
                with self.assertRaises(maintenance.Refusal):
                    maintenance.parse(argv)

    def query(self, rc=0, stdout=CSV, stderr='', process_rc=0,
              process_stdout='gpu_uuid, pid, process_name\n', boot='aaaaaaaa-0000-0000-0000-000000000001',
              instance='i-0123456789abcdef0', error=None):
        calls = []

        def run(argv, **kwargs):
            calls.append(list(argv))
            if error is not None:
                raise error
            if '--query-compute-apps=gpu_uuid,pid,process_name' in argv:
                return subprocess.CompletedProcess(argv, process_rc, process_stdout, '')
            return subprocess.CompletedProcess(argv, rc, stdout, stderr)
        out = io.StringIO()
        with mock.patch.object(maintenance.subprocess, 'run', side_effect=run), \
                mock.patch.object(maintenance, 'instance_identity', return_value=instance), \
                mock.patch.object(maintenance, '_boot_id', return_value=boot), \
                contextlib.redirect_stdout(out):
            code = maintenance.gpu_recovery(self.CONFIG)
        return code, out.getvalue(), calls

    def test_the_query_is_the_fixed_read_only_nvidia_smi_call(self):
        code, out, calls = self.query()
        self.assertEqual(0, code)
        self.assertEqual([['/usr/bin/nvidia-smi',
                           '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv'],
                          ['/usr/bin/nvidia-smi',
                           '--query-gpu=index,uuid,pci.bus_id,gpu_recovery_action',
                           '--format=csv']], calls)
        record = json.loads(out)
        self.assertEqual({'action': 'gpu-recovery', 'instance_id': 'i-0123456789abcdef0',
                          'boot_id': 'aaaaaaaa-0000-0000-0000-000000000001',
                          'query': 'nvidia-smi --query-gpu=index,uuid,pci.bus_id,'
                                   'gpu_recovery_action --format=csv',
                          'returncode': 0, 'stdout': self.CSV, 'stderr': '',
                          'process_returncode': 0, 'process_stdout': 'gpu_uuid, pid, process_name\n',
                          'process_stderr': ''},
                         {k: v for k, v in record.items() if k != 'observed_at'})
        self.assertIsInstance(record['observed_at'], float)
        self.assertEqual(1, out.count('\n'))

    def test_a_failed_query_is_reported_in_the_record_not_as_a_refusal(self):
        code, out, _ = self.query(rc=9, stdout='', stderr='NVIDIA-SMI has failed\n')
        self.assertEqual(0, code)
        record = json.loads(out)
        self.assertEqual((9, '', 'NVIDIA-SMI has failed\n'),
                         (record['returncode'], record['stdout'], record['stderr']))
        code, out, _ = self.query(error=FileNotFoundError(2, 'No such file', '/usr/bin/nvidia-smi'))
        self.assertEqual(0, code)
        record = json.loads(out)
        self.assertEqual(127, record['returncode'])
        self.assertIn('nvidia-smi', record['stderr'])
        code, out, _ = self.query(error=subprocess.TimeoutExpired(['nvidia-smi'], 20))
        record = json.loads(out)
        self.assertEqual(124, record['returncode'])

    def test_compute_process_query_failure_is_preserved(self):
        code, out, _ = self.query(process_rc=9, process_stdout='')
        self.assertEqual(0, code)
        record = json.loads(out)
        self.assertEqual(9, record['process_returncode'])
        self.assertEqual('', record['process_stdout'])
        self.assertEqual(0, record['returncode'])

    def test_timeout_preserves_partial_output_and_error_streams(self):
        for stdout, stderr in [('partial table\n', 'driver diagnostic\n'),
                               (b'partial table\n', b'driver diagnostic\n')]:
            with self.subTest(stream_type=type(stdout).__name__):
                code, out, _ = self.query(error=subprocess.TimeoutExpired(
                    ['nvidia-smi'], 20, output=stdout, stderr=stderr))
                record = json.loads(out)
                self.assertEqual(0, code)
                self.assertEqual(124, record['process_returncode'])
                self.assertEqual('partial table\n', record['process_stdout'])
                self.assertIn('driver diagnostic\n', record['process_stderr'])
                self.assertEqual(124, record['returncode'])
                self.assertEqual('partial table\n', record['stdout'])
                self.assertIn('driver diagnostic\n', record['stderr'])

    def test_the_record_carries_no_mutating_command(self):
        source = HELPER.read_text()
        body = source[source.index('def gpu_recovery('):]
        body = body[:body.index('\ndef ', 1)]
        for word in ('--gpu-reset', '-r ', 'persistence-mode', 'systemctl', 'reboot',
                     'DEVICE_FAULT', 'execve'):
            self.assertNotIn(word, body)


if __name__ == '__main__':
    unittest.main()
