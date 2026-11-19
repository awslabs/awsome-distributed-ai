# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""The lab.env generator: defaults, the values a site sets, and refusals."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import unittest

LAB = Path(__file__).resolve().parents[1]
GENERATOR = LAB / 'facilitator' / 'lab-env.py'


def load():
    spec = importlib.util.spec_from_file_location('aim344_lab_env', GENERATOR)
    if spec is None or spec.loader is None:
        raise unittest.SkipTest(f'Generator not importable: {GENERATOR}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lab_env = load()


def sourced(text, *names, **environment):
    """The named variables after sourcing `text` in a clean shell."""
    script = text + '\nprintf "%s\\n" ' + ' '.join(f'"${name}"' for name in names)
    return subprocess.check_output(['bash', '-eu', '-c', script],
                                   env={'PATH': os.environ['PATH'], **environment},
                                   text=True).splitlines()


class LabEnvironment(unittest.TestCase):
    def test_partition_is_required_and_values_are_checked(self):
        with self.assertRaisesRegex(lab_env.Refusal, 'PARTITION'):
            lab_env.lab_environment({})
        text = lab_env.lab_environment({'PARTITION': 'assigned-queue'})
        self.assertIn('/opt/aim344-healthcheck/validation/', text)
        for key, value in [('NCCL_CONTAINER', '/not/provisioned.sqsh'),
                           ('OMPI_MCA_btl_tcp_if_include', 'eth0'),
                           ('PARTITION', 'two words'),
                           ('AIM344_EFA_DEVICE', '')]:
            with self.subTest(key=key), self.assertRaises(lab_env.Refusal):
                lab_env.lab_environment({'PARTITION': 'assigned-queue', key: value})
        text = lab_env.lab_environment({'PARTITION': 'q', 'AIM344_EXPECTED_EFA_DEVICES': '2'})
        self.assertIn('HEALTHCHECK_EXPECTED_EFA_COUNT=${HEALTHCHECK_EXPECTED_EFA_COUNT:-2}', text)
        for count in ('0', '-1', '33', '02', '2;id'):
            with self.subTest(count=count), self.assertRaises(lab_env.Refusal):
                lab_env.lab_environment({'PARTITION': 'q', 'AIM344_EXPECTED_EFA_DEVICES': count})
        # The g7.48xlarge site sets the NCCL protocol for Check 5 through lab.env.
        self.assertIn('export NCCL_PROTO=${NCCL_PROTO:-Simple}\n',
                      lab_env.lab_environment({'PARTITION': 'q', 'NCCL_PROTO': 'Simple'}))

    def test_the_cpu_budget_default_overrides_and_validation(self):
        for value in (None, '12', '24', '4096'):
            with self.subTest(cpus_per_rank=value):
                overrides = {'PARTITION': 'assigned-queue'}
                if value is not None:
                    overrides['CPUS_PER_RANK'] = value
                text = lab_env.lab_environment(overrides)
                cpus = int(value or '24')
                self.assertEqual(['8', str(cpus)], sourced(text, 'GPUS_PER_NODE', 'CPUS_PER_RANK'))
                self.assertEqual(['16'], sourced(text, 'CPUS_PER_RANK', CPUS_PER_RANK='16'))
        # Match the existing allocation parser: canonical positive decimal,
        # without imposing a new site/hardware maximum (4096 above is valid).
        for value in ('0', '-1', '1.5', 'abc', '12;id', ' 12', '12 ', '012',
                      '+12', '12\n', '', 12, True):
            with self.subTest(invalid=value), self.assertRaises(lab_env.Refusal):
                lab_env.lab_environment({'PARTITION': 'assigned-queue', 'CPUS_PER_RANK': value})

    def test_the_seat_values_reach_the_login_profile(self):
        text = lab_env.lab_environment({
            'PARTITION': 'gpu-g7', 'AIM344_TARGET_NODE': 'gpu-g7-1',
            'AIM344_PEER_NODE': 'gpu-g7-2', 'AIM344_EFA_DEVICE': 'rdmap83s0',
            'AIM344_SLURM_BIN': '/opt/aws/pcs/scheduler/slurm-25.11/bin'})
        self.assertEqual(['gpu-g7-1', 'gpu-g7-2', 'rdmap83s0',
                          '/opt/aws/pcs/scheduler/slurm-25.11/bin'],
                         sourced(text, 'AIM344_TARGET_NODE', 'AIM344_PEER_NODE',
                                 'AIM344_EFA_DEVICE', 'AIM344_SLURM_BIN'))
        self.assertEqual(['/opt/aws/pcs/scheduler/slurm-25.05/bin'],
                         sourced(lab_env.lab_environment({'PARTITION': 'gpu-g7'}),
                                 'AIM344_SLURM_BIN'))

    def test_the_command_line_prints_the_file_or_refuses(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = lab_env.main(['PARTITION=gpu-g7', 'NCCL_SOCKET_IFNAME==enp71s0'])
        self.assertEqual(0, code, err.getvalue())
        self.assertEqual(['gpu-g7', '=enp71s0'],
                         sourced(out.getvalue(), 'PARTITION', 'NCCL_SOCKET_IFNAME'))
        for argv in (['PARTITION'], ['PARTITION=a', 'PARTITION=b'], ['NO_SUCH_VALUE=1']):
            out, err = io.StringIO(), io.StringIO()
            with self.subTest(argv=argv), contextlib.redirect_stdout(out), \
                    contextlib.redirect_stderr(err):
                self.assertEqual(2, lab_env.main(argv))
                self.assertEqual('', out.getvalue())
                self.assertNotEqual('', err.getvalue())


if __name__ == '__main__':
    unittest.main()
