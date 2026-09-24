# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Real CPU DataLoader lifecycle tests; parent CUDA/NCCL calls are mocked."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

LAB = Path(os.environ.get('AIM344_DATALOADER_TEST_LAB', Path(__file__).resolve().parents[1]))
HAS_TORCH = importlib.util.find_spec('torch') is not None


def exercise(method, epochs):
    import multiprocessing
    from types import SimpleNamespace
    from unittest import mock
    sys.path.insert(0, str(LAB))
    import workload
    tensor = object()
    calls = []
    loader_options = []
    real_loader = workload.DataLoader

    def make_loader(*args, **kwargs):
        loader_options.append({k: kwargs.get(k) for k in
                               ('num_workers', 'persistent_workers', 'multiprocessing_context')})
        return real_loader(*args, **kwargs)

    def collective(value):
        assert value is tensor
        calls.append(1)

    with mock.patch.object(workload.dist, 'get_rank', return_value=0), \
            mock.patch.object(workload, 'collective', side_effect=collective), \
            mock.patch.object(workload, 'DataLoader', side_effect=make_loader):
        workload.dataloader(SimpleNamespace(start_method=method, epochs=epochs), tensor)
    print('TEST_RESULT ' + json.dumps({'collectives': len(calls), 'loaders': loader_options,
                                      'active_children': [p.pid for p in multiprocessing.active_children()]}),
          flush=True)


@unittest.skipUnless(HAS_TORCH, 'CPU torch required; lifecycle/worker recreation not tested')
class DataLoaderEpochs(unittest.TestCase):
    def run_method(self, method, epochs):
        result = subprocess.run([sys.executable, __file__, '--exercise', method, str(epochs)],
                                text=True, capture_output=True, timeout=60,
                                env={**os.environ, 'OMP_NUM_THREADS': '1'})
        print(f'--- actual CPU {method} epochs={epochs} rc={result.returncode} ---\n'
              + result.stdout + result.stderr, flush=True)
        self.assertEqual(0, result.returncode)
        events = [json.loads(line.split('DataLoader lifecycle: ', 1)[1])
                  for line in result.stdout.splitlines() if line.startswith('DataLoader lifecycle: ')]
        summary = json.loads(next(line[12:] for line in result.stdout.splitlines()
                                  if line.startswith('TEST_RESULT ')))
        self.assertEqual(epochs * 4, summary['collectives'])
        self.assertEqual([{'num_workers': 2, 'persistent_workers': False,
                           'multiprocessing_context': method}], summary['loaders'])
        self.assertEqual([], summary['active_children'])
        worker_pids = []
        for epoch in range(1, epochs + 1):
            rows = [row for row in events if row['epoch'] == epoch]
            workers = [row for row in rows if row['event'] == 'worker_init']
            self.assertEqual({0, 1}, {row['worker_id'] for row in workers})
            self.assertEqual(2, len(workers))
            self.assertTrue(all(row['rank'] == 0 and row['host'] for row in rows))
            pids = {row['pid'] for row in workers}
            self.assertEqual(2, len(pids))
            worker_pids.append(pids)
            parent = [row for row in rows if row['event'] != 'worker_init']
            self.assertEqual(['iterator_start', 'iterator_created'] +
                             ['batch_received', 'collective_complete'] * 4 + ['iterator_end'],
                             [row['event'] for row in parent])
            self.assertTrue(all(row['expected_batches'] == 4 for row in parent))
            self.assertEqual(4, parent[-1]['batches'])
            self.assertEqual([1, 2, 3, 4], [row['batch'] for row in parent
                                          if row['event'] == 'collective_complete'])
            self.assertTrue(all(row['parent_pid'] == parent[0]['pid'] for row in workers))
        if epochs == 2:
            self.assertTrue(worker_pids[0].isdisjoint(worker_pids[1]), worker_pids)
        self.assertIn(f'{epochs * 4} batches; 0 mismatches', result.stdout)

    def test_fork_two_epochs_recreates_workers(self):
        self.run_method('fork', 2)

    def test_spawn_two_epochs_recreates_workers(self):
        self.run_method('spawn', 2)

    def test_one_epoch_compatibility(self):
        self.run_method('spawn', 1)

    def test_cli_default_and_invalid_epoch_counts(self):
        help_result = subprocess.run([sys.executable, str(LAB / 'workload.py'), '--help'],
                                     capture_output=True, text=True, timeout=20)
        self.assertEqual(0, help_result.returncode)
        self.assertIn('default: 1', help_result.stdout)
        for value in ('0', '-1', '1.5'):
            result = subprocess.run([sys.executable, str(LAB / 'workload.py'), 'dataloader',
                                     '--epochs', value], capture_output=True, text=True, timeout=20)
            self.assertEqual(2, result.returncode)
            self.assertIn('positive integer', result.stderr)
            self.assertNotIn('LOCAL_RANK', result.stderr)

    def test_bad_batch_fails_before_collective(self):
        from types import SimpleNamespace
        from unittest import mock
        sys.path.insert(0, str(LAB))
        import workload
        loader = mock.MagicMock()
        loader.__iter__.return_value = iter([[workload.torch.arange(8) + 1]])
        with mock.patch.object(workload.dist, 'get_rank', return_value=0), \
                mock.patch.object(workload, 'DataLoader', return_value=loader), \
                mock.patch.object(workload, 'collective') as collective:
            with self.assertRaisesRegex(RuntimeError, 'DataLoader batch correctness mismatch'):
                workload.dataloader(SimpleNamespace(start_method='fork', epochs=2), object())
            collective.assert_not_called()

    def test_short_epoch_fails_not_success(self):
        from types import SimpleNamespace
        from unittest import mock
        sys.path.insert(0, str(LAB))
        import workload
        loader = mock.MagicMock()
        loader.__iter__.return_value = iter([])
        with mock.patch.object(workload.dist, 'get_rank', return_value=0), \
                mock.patch.object(workload, 'DataLoader', return_value=loader):
            with self.assertRaisesRegex(RuntimeError, 'Expected 4 batches'):
                workload.dataloader(SimpleNamespace(start_method='fork', epochs=2), object())


class LauncherContract(unittest.TestCase):
    def test_numbered_launchers_select_two_epochs_and_preserve_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / 'common.sh').write_text(
                'prepare_slurm() { :; }\nrun_torch() { printf "%s\\n" "$*"; return "$TEST_RC"; }\n')
            for method, name in (('fork', '7.probe-dataloader-fork.sh'),
                                 ('spawn', '8.use-dataloader-spawn.sh')):
                (directory / name).write_text((LAB / name).read_text())
                for rc in (0, 1, 124, 137):
                    result = subprocess.run(['bash', str(directory / name)], capture_output=True, text=True,
                                            env={**os.environ, 'TEST_RC': str(rc)}, timeout=5)
                    self.assertEqual(rc, result.returncode)
                    self.assertIn(f'dataloader-{method} dataloader --start-method {method} --epochs 2',
                                  result.stdout)

    def test_existing_run_torch_preserves_nonzero_and_watchdog(self):
        # Fake srun only: no Slurm, GPU, AWS or container invocation. 124/137
        # here are injected exit statuses, not evidence of a real timed-out job.
        source = (LAB / 'common.sh').read_text()
        self.assertIn('timeout --signal=TERM --kill-after=30s 180s', source)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / 'pins.env').write_text('')
            srun = directory / 'srun'
            srun.write_text('#!/bin/bash\nprintf "fake srun\\n"\nexit "$TEST_RC"\n')
            srun.chmod(0o755)
            script = ('set -euo pipefail\nLAB_DIR=$TEST_DIR\nsource "$TEST_COMMON"\n'
                      'prepare_torch() { :; }\nRESULTS_DIR=$TEST_DIR\nAIM344_CPUS_PER_NODE=1\n'
                      'CONTAINER_ARGS=()\nrun_torch dataloader-fork dataloader --start-method fork --epochs 2\n')
            for rc in (0, 1, 124, 137):
                result = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=5,
                                        env={**os.environ, 'PATH': tmp + ':' + os.environ['PATH'],
                                             'TEST_RC': str(rc), 'TEST_DIR': tmp,
                                             'TEST_COMMON': str(LAB / 'common.sh')})
                self.assertEqual(rc, result.returncode, result.stderr)
                self.assertIn(f'launcher_exit_code={rc}', result.stdout)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--exercise':
        exercise(sys.argv[2], int(sys.argv[3]))
    else:
        unittest.main()
