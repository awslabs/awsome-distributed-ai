# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Grammar and refusal tests for the target-side maintenance forced command.

These run without hardware. They exercise the parser and configuration checks
that stand between the coordinator's maintenance key and root on the fault
target. Whether a device operation succeeds is the existing helper's business,
and a separate hardware acceptance track.
"""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

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
        config = maintenance.load_config(require_root_owned=False)
        self.assertEqual([0, 2, 3, 6], config['checks'])

    def test_non_root_owned_configuration_is_refused(self):
        """The deployed helper requires root ownership; this asserts that check
        exists rather than only exercising the relaxed test path."""
        self.assertNotEqual(0, os.getuid(), 'run this suite as a non-root user')
        with self.assertRaises(maintenance.Refusal) as caught:
            maintenance.load_config()
        self.assertIn('owned by root', str(caught.exception))

    def test_world_readable_configuration_is_refused(self):
        os.chmod(self.path, 0o644)
        with self.assertRaises(maintenance.Refusal) as caught:
            maintenance.load_config(require_root_owned=False)
        self.assertIn('0600', str(caught.exception))

    def test_missing_keys_are_named(self):
        for key in ('slurm_bin', 'participant_user', 'stage_dir', 'checks', 'suite_entry'):
            with self.subTest(key=key):
                body = dict(self.body)
                body.pop(key)
                self.path.write_text(json.dumps(body))
                os.chmod(self.path, 0o600)
                with self.assertRaises(maintenance.Refusal) as caught:
                    maintenance.load_config(require_root_owned=False)
                self.assertIn(key, str(caught.exception))

    def test_unexpected_slurm_installation_is_refused(self):
        self.body['slurm_bin'] = '/tmp/evil/bin'
        self.write()
        with self.assertRaises(maintenance.Refusal) as caught:
            maintenance.load_config(require_root_owned=False)
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


if __name__ == '__main__':
    unittest.main()
