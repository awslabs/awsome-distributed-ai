"""Command-output fixtures for Check 5. These do not replace hardware qualification."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SUITE = Path(__file__).resolve().parents[1]
ROW = (Path(__file__).parent / 'nccl-valid-rows.txt').read_text()
PROVIDER = 'NET/OFI Selected Provider is efa\n'


class NcclResultTests(unittest.TestCase):
    def run_check(self, output, *, instance='g7.48xlarge', exit_code=0, threshold=None, job_cpus=None, step_cpus="192"):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, body in {
                'curl': 'printf "%s\\n" "$TEST_INSTANCE"',
                'nvidia-smi': 'echo 595.91.07',
                'srun': 'if [ "$1" = --help ]; then echo container-image; else printf "%s\\n" "$@" > "$TEST_ARGS"; cat "$TEST_OUTPUT"; exit "$TEST_EXIT"; fi',
            }.items():
                p = root / name
                p.write_text('#!/bin/sh\n' + body + '\n')
                p.chmod(0o755)
            (root / 'output').write_text(output)
            env = dict(os.environ, PATH=f'{root}:/usr/bin:/bin', SLURM_JOB_NUM_NODES='2',
                       SLURM_NTASKS='16', TEST_INSTANCE=instance,
                       TEST_OUTPUT=str(root / 'output'), TEST_ARGS=str(root / 'args'),
                       TEST_EXIT=str(exit_code), RESULTS_DIR=str(root / 'results'))
            for key, value in {'SLURM_CPUS_ON_NODE': step_cpus, 'SLURM_JOB_CPUS_PER_NODE': job_cpus}.items():
                if value is None:
                    env.pop(key, None)
                else:
                    env[key] = value
            if threshold is not None:
                env['NCCL_MIN_BUS_BW'] = str(threshold)
            else:
                env.pop('NCCL_MIN_BUS_BW', None)
            p = subprocess.run(['bash', str(SUITE / 'checks/5-nccl-allreduce.sh')],
                               env=env, text=True, capture_output=True, timeout=20)
            report = json.loads((root / 'results/check-5-nccl-allreduce.json').read_text())
            args = (root / 'args').read_text().splitlines() if (root / 'args').exists() else []
            return p.returncode, report, args

    def test_real_success_row_and_explicit_rank_mapping(self):
        rc, report, args = self.run_check(PROVIDER + ROW)
        self.assertEqual((rc, report['status']), (0, 'PASS'))
        self.assertIn('--ntasks=2', args)
        self.assertIn('/opt/nccl-tests/build/all_reduce_perf', args)
        self.assertEqual(args[args.index('-g') + 1], '8')
        self.assertEqual(args[args.index('-c') + 1], '1')

    def test_verbose_provider_output_does_not_fail_on_sigpipe(self):
        output = PROVIDER + ('NCCL INFO channel initialized\n' * 40000) + ROW
        rc, report, _ = self.run_check(output)
        self.assertEqual((rc, report['status']), (0, 'PASS'))

    def test_verbose_error_keeps_isolate_severity(self):
        output = 'NCCL WARN unhandled system error\n' + ('NCCL INFO teardown\n' * 40000)
        rc, report, _ = self.run_check(output, exit_code=1)
        self.assertNotEqual(rc, 0)
        self.assertEqual(report['severity'], 'ISOLATE')

    def test_salloc_cpu_layout_without_step_environment(self):
        rc, report, args = self.run_check(PROVIDER + ROW, job_cpus='192(x2)', step_cpus=None)
        self.assertEqual((rc, report['status']), (0, 'PASS'))
        self.assertIn('--cpus-per-task=192', args)

    def test_heterogeneous_cpu_layout_fails_before_launch(self):
        rc, report, args = self.run_check(PROVIDER + ROW, job_cpus='192,96', step_cpus=None)
        self.assertNotEqual(rc, 0)
        self.assertEqual(report['status'], 'FAIL')
        self.assertEqual(args, [])

    def test_nonzero_incorrect_result_fails(self):
        rc, report, _ = self.run_check(PROVIDER + ROW.rsplit('0', 1)[0] + '1\n')
        self.assertNotEqual(rc, 0)
        self.assertEqual(report['status'], 'FAIL')

    def test_disabled_correctness_fails(self):
        rc, _, _ = self.run_check(PROVIDER + ROW.replace('       0', '     N/A'))
        self.assertNotEqual(rc, 0)

    def test_partial_sweep_fails(self):
        rc, _, _ = self.run_check(PROVIDER + ROW.splitlines()[-1] + '\n')
        self.assertNotEqual(rc, 0)

    def test_empty_result_fails(self):
        rc, _, _ = self.run_check(PROVIDER)
        self.assertNotEqual(rc, 0)

    def test_missing_provider_fails(self):
        rc, _, _ = self.run_check(ROW)
        self.assertNotEqual(rc, 0)

    def test_bandwidth_warning_survives_final_result(self):
        rc, report, _ = self.run_check(PROVIDER + ROW, threshold=45)
        self.assertEqual((rc, report['status']), (0, 'WARN'))

    def test_unknown_profile_never_launches_zero_gpus(self):
        rc, report, args = self.run_check(PROVIDER + ROW, instance='unknown.test')
        self.assertNotEqual(rc, 0)
        self.assertEqual(report['status'], 'FAIL')
        self.assertEqual(args, [])

    def test_missing_binary_is_not_a_pass(self):
        rc, report, _ = self.run_check('execve(): missing binary\n', exit_code=127)
        self.assertNotEqual(rc, 0)
        self.assertEqual(report['status'], 'FAIL')


if __name__ == '__main__':
    unittest.main()
