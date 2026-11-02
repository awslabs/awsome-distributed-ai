# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""CPU tests of the exercise modes: DataLoader option A and its fixes, worker CUDA
under fork, the replica check with its software bit flip, and the settings the lab
controls pass through the environment. CUDA and NCCL are mocked; the DataLoader,
its workers and faulthandler are real."""
import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

LAB = Path(os.environ.get('AIM344_DATALOADER_TEST_LAB', Path(__file__).resolve().parents[1]))
HAS_TORCH = importlib.util.find_spec('torch') is not None
WORLD = 16


def load_workload():
    sys.path.insert(0, str(LAB))
    import workload
    return workload


def exercise_dataloader(spec):
    """Run workload.dataloader() in this process with the parent's NCCL calls mocked."""
    from unittest import mock
    workload = load_workload()
    workload.REFILL_SECONDS = spec.get('refill_seconds', 1.0)
    tensor = object()
    calls = []
    args = SimpleNamespace(start_method=spec['start_method'], epochs=spec.get('epochs', 2),
                           dataset_cache=spec.get('dataset_cache', 'none'),
                           dataloader_timeout=float(spec.get('dataloader_timeout', 0)),
                           stack_dump_seconds=spec.get('stack_dump_seconds', 0),
                           stack_dump_dir=spec.get('stack_dump_dir', '/nonexistent'),
                           reinit_in_worker=spec.get('reinit_in_worker', False),
                           worker_cuda=spec.get('worker_cuda', False),
                           affected_rank=spec.get('affected_rank'))
    with mock.patch.object(workload.dist, 'get_rank', return_value=spec.get('rank', 0)), \
            mock.patch.object(workload, 'collective', side_effect=lambda value: calls.append(1)):
        workload.dataloader(args, tensor)
    print('TEST_RESULT ' + json.dumps({'collectives': len(calls)}), flush=True)


def exercise_device(spec):
    """Run workload.main() in device mode on the CPU with every CUDA and NCCL call mocked.

    The fake all-gather plays the other ranks: they report the clean checksum of the
    step, computed by a replica check without the flip, and this rank reports its own.
    """
    from unittest import mock
    import torch
    workload = load_workload()
    rank, world_size = spec['rank'], spec['world_size']
    step = {'n': 0}
    peer_args = SimpleNamespace(replica_n=spec['replica_n'], replica_seed=2026, corrupt_rank=None,
                                corrupt_step=1, corrupt_bit=22, corrupt_recurring=False)
    peer = workload.ReplicaCheck(peer_args, torch.device('cpu'), rank=1, world_size=world_size)

    def all_gather(gathered, local):
        step['n'] += 1
        gathered.fill_(peer.checksum(step['n']).item())
        gathered[rank] = local[0]

    # Rank 0 ends the loop from its deadline; the fake broadcast carries that decision
    # to any rank this harness plays.
    deadline = time.monotonic() + spec['duration']

    def broadcast(tensor, src=0):
        tensor.fill_(int(time.monotonic() < deadline))

    real_ones = torch.ones
    collectives = []
    argv = ['workload.py', 'device', '--duration-seconds', str(spec['duration']),
            '--replica-n', str(spec['replica_n'])] + spec.get('argv', [])
    with mock.patch.object(sys, 'argv', argv), \
            mock.patch.object(workload.torch.cuda, 'set_device'), \
            mock.patch.object(workload.torch.cuda, 'synchronize', create=True), \
            mock.patch.object(workload.torch.cuda.nccl, 'version', return_value=(2, 27, 7)), \
            mock.patch.object(workload.torch, 'ones',
                              side_effect=lambda *a, **k: real_ones(*a, **{**k, 'device': 'cpu'})), \
            mock.patch.object(workload, 'collective', side_effect=lambda t: collectives.append(1)), \
            mock.patch.object(workload.dist, 'init_process_group'), \
            mock.patch.object(workload.dist, 'destroy_process_group'), \
            mock.patch.object(workload.dist, 'get_rank', return_value=rank), \
            mock.patch.object(workload.dist, 'get_world_size', return_value=world_size), \
            mock.patch.object(workload.dist, 'broadcast', side_effect=broadcast), \
            mock.patch.object(workload.dist, 'all_gather_into_tensor', side_effect=all_gather):
        workload.main()
    print('TEST_RESULT ' + json.dumps({'collectives': len(collectives), 'steps': step['n']}),
          flush=True)


def lifecycle(stdout):
    return [json.loads(line.split('DataLoader lifecycle: ', 1)[1])
            for line in stdout.splitlines() if line.startswith('DataLoader lifecycle: ')]


def test_result(stdout):
    return json.loads(next(line[12:] for line in stdout.splitlines()
                           if line.startswith('TEST_RESULT ')))


@unittest.skipUnless(HAS_TORCH, 'CPU torch required')
class DataLoaderOptionA(unittest.TestCase):
    """The hang, what it looks like, and the three fixes, with a real CPU DataLoader."""

    def run_spec(self, spec, timeout=90):
        result = subprocess.run([sys.executable, __file__, '--exercise-dataloader', json.dumps(spec)],
                                text=True, capture_output=True, timeout=timeout,
                                env={**os.environ, 'OMP_NUM_THREADS': '1'})
        print(f'--- dataloader {spec} rc={result.returncode} ---\n' + result.stdout + result.stderr,
              flush=True)
        return result

    def assert_two_epochs(self, result, method):
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(8, test_result(result.stdout)['collectives'])
        events = lifecycle(result.stdout)
        for epoch in (1, 2):
            parent = [row['event'] for row in events if row['epoch'] == epoch and row['event'] != 'worker_init']
            self.assertEqual(['iterator_start', 'iterator_created']
                             + ['batch_received', 'collective_complete'] * 4 + ['iterator_end'], parent)
            self.assertEqual(2, len([row for row in events if row['epoch'] == epoch and row['event'] == 'worker_init']))
        self.assertIn('8 batches; 0 mismatches', result.stdout)

    def test_rank_thread_under_fork_stops_every_rank_silently_and_dumps_stacks(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = {'start_method': 'fork', 'dataset_cache': 'rank-thread',
                    'stack_dump_seconds': 2, 'stack_dump_dir': tmp}
            process = subprocess.Popen([sys.executable, __file__, '--exercise-dataloader', json.dumps(spec)],
                                       text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       start_new_session=True, env={**os.environ, 'OMP_NUM_THREADS': '1'})
            deadline = time.monotonic() + 40

            def role(path):
                return path.name.split('-rank0-')[1].split('-pid')[0]

            def dumped():
                return {role(p) for p in Path(tmp).glob('stacks-*') if 'Thread 0x' in p.read_text()}

            try:
                # Three processes stop: the rank and its two epoch-2 workers. Each writes
                # its own stack file two seconds after its timer was armed.
                while time.monotonic() < deadline:
                    if dumped() >= {'rank', 'worker0', 'worker1'} or process.poll() is not None:
                        break
                    time.sleep(0.5)
                self.assertIsNone(process.poll(), 'the run ended on its own; the hang did not happen')
                # The run never ends by itself: the launcher's timeout ends it in the lab.
                time.sleep(2)
                self.assertIsNone(process.poll())
            finally:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                stdout, stderr = process.communicate(timeout=30)
            print('--- hang run ---\n' + stdout + stderr, flush=True)
            events = lifecycle(stdout)
            epoch1 = [row['event'] for row in events if row['epoch'] == 1 and row['event'] != 'worker_init']
            self.assertEqual(['iterator_start', 'iterator_created']
                             + ['batch_received', 'collective_complete'] * 4 + ['iterator_end'], epoch1)
            # The workers' own lines can come before the rank's iterator_created line.
            epoch2 = [row['event'] for row in events if row['epoch'] == 2]
            self.assertEqual('iterator_start', epoch2[0])
            self.assertEqual({'iterator_start': 1, 'iterator_created': 1, 'worker_init': 2}, dict(Counter(epoch2)))
            self.assertNotIn('Traceback', stderr)
            self.assertNotIn('mismatches', stdout)
            # Epoch 1's workers removed their unused files when they exited.
            files = {p.name: p.read_text() for p in Path(tmp).glob('stacks-*')}
            self.assertEqual(3, len(files), sorted(files))
            self.assertEqual({'rank', 'worker0', 'worker1'}, {role(Path(name)) for name in files}, sorted(files))
            for name, text in files.items():
                self.assertIn('Timeout (0:00:02)!', text, name)
                self.assertIn('Thread 0x', text, name)
                if '-rank-pid' in name:
                    self.assertIn('_try_get_data', text, name)
                    self.assertNotIn('__getitem__', text, name)
                else:
                    self.assertIn('__getitem__', text, name)
                    # The worker's stack still holds the frames it was forked from:
                    # the epoch-2 iter(loader) in the rank process.
                    self.assertIn('dataloader.py', text, name)

    def test_rank_thread_with_timeout_turns_the_stop_into_an_error(self):
        result = self.run_spec({'start_method': 'fork', 'dataset_cache': 'rank-thread',
                                'dataloader_timeout': 2})
        self.assertNotEqual(0, result.returncode)
        self.assertIn('RuntimeError: DataLoader timed out after 2.0 seconds', result.stderr)
        events = lifecycle(result.stdout)
        self.assertEqual(4, len([row for row in events if row['event'] == 'collective_complete']))
        self.assertNotIn('batch_received', [row['event'] for row in events if row['epoch'] == 2])

    def test_rank_thread_cannot_be_pickled_for_spawn_or_forkserver(self):
        for method in ('spawn', 'forkserver'):
            result = self.run_spec({'start_method': method, 'dataset_cache': 'rank-thread'})
            self.assertNotEqual(0, result.returncode)
            self.assertIn("TypeError: cannot pickle '_thread.lock' object", result.stderr, method)
            # Epoch 1's iter(loader) fails: no batch was ever received.
            self.assertEqual([], [row for row in lifecycle(result.stdout)
                                  if row['event'] in ('batch_received', 'worker_init')], method)

    def test_worker_thread_completes_under_every_start_method(self):
        for method in ('fork', 'spawn', 'forkserver'):
            result = self.run_spec({'start_method': method, 'dataset_cache': 'worker-thread'})
            self.assert_two_epochs(result, method)

    def test_reinit_in_worker_completes_under_fork(self):
        result = self.run_spec({'start_method': 'fork', 'dataset_cache': 'rank-thread',
                                'reinit_in_worker': True})
        self.assert_two_epochs(result, 'fork')

    def test_unaffected_rank_completes(self):
        result = self.run_spec({'start_method': 'fork', 'dataset_cache': 'rank-thread',
                                'affected_rank': 5, 'rank': 0})
        self.assert_two_epochs(result, 'fork')

    def test_stack_files_are_removed_when_the_epochs_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_spec({'start_method': 'fork', 'dataset_cache': 'worker-thread',
                                    'stack_dump_seconds': 60, 'stack_dump_dir': tmp})
            self.assert_two_epochs(result, 'fork')
            self.assertEqual([], list(Path(tmp).iterdir()))

    def test_worker_cuda_under_fork_fails_at_the_first_batch_of_epoch_1(self):
        # Without a GPU the worker raises torch's "not compiled with CUDA" assertion
        # instead of the bad-fork RuntimeError; either way the main process re-raises
        # it from the first fetch, before any collective of epoch 1.
        result = self.run_spec({'start_method': 'fork', 'dataset_cache': 'worker-thread',
                                'worker_cuda': True})
        self.assertNotEqual(0, result.returncode)
        self.assertIn('in DataLoader worker process', result.stderr)
        events = lifecycle(result.stdout)
        self.assertEqual([], [row for row in events if row['event'] in ('batch_received', 'collective_complete')])
        self.assertEqual([1], sorted({row['epoch'] for row in events}))

    def test_worker_cuda_on_an_unaffected_rank_completes(self):
        result = self.run_spec({'start_method': 'fork', 'dataset_cache': 'worker-thread',
                                'worker_cuda': True, 'affected_rank': 3, 'rank': 0})
        self.assert_two_epochs(result, 'fork')


@unittest.skipUnless(HAS_TORCH, 'CPU torch required')
class CachedDatasetUnit(unittest.TestCase):
    def test_rank_thread_refresh_returns_while_the_lock_is_held(self):
        workload = load_workload()
        workload.REFILL_SECONDS = 0.5
        dataset = workload.CachedDataset('rank-thread')
        self.assertEqual(32, len(dataset))
        self.assertEqual(7, dataset[7][0].item())
        dataset.refresh()
        self.assertTrue(dataset.lock.locked())
        dataset.refill_thread.join(5)
        self.assertFalse(dataset.lock.locked())
        self.assertEqual([i for i in range(32)], [dataset[i][0].item() for i in range(32)])

    def test_worker_thread_drops_its_objects_when_pickled(self):
        import pickle
        workload = load_workload()
        workload.REFILL_SECONDS = 0.2
        dataset = workload.CachedDataset('worker-thread')
        self.assertIsNone(dataset.lock)
        dataset.refresh()
        self.assertIsNone(dataset.lock, 'the rank process creates nothing for worker-thread')
        self.assertEqual(3, dataset[3][0].item())
        self.assertIsNotNone(dataset.lock)
        copy = pickle.loads(pickle.dumps(dataset))
        self.assertIsNone(copy.lock)
        self.assertIsNone(copy.cache)
        self.assertEqual(1, copy.generation)
        with self.assertRaises(TypeError):
            pickle.dumps(workload.CachedDataset('rank-thread'))

    def test_reinit_replaces_the_lock_and_rebuilds_the_cache(self):
        workload = load_workload()
        workload.REFILL_SECONDS = 2.0
        dataset = workload.CachedDataset('rank-thread')
        dataset.refresh()
        held = dataset.lock
        self.assertTrue(held.locked())
        self.assertLess(len(dataset.cache), 32, 'the refill is in progress')
        dataset.reinit()
        self.assertIsNot(held, dataset.lock)
        self.assertFalse(dataset.lock.locked())
        self.assertEqual(31, dataset[31][0].item())


@unittest.skipUnless(HAS_TORCH, 'CPU torch required')
class ReplicaCheckUnit(unittest.TestCase):
    def make(self, workload, rank, world_size=WORLD, **overrides):
        import torch
        from unittest import mock
        args = SimpleNamespace(replica_n=64, replica_seed=2026, corrupt_rank=None, corrupt_step=1,
                               corrupt_bit=22, corrupt_recurring=False)
        args.__dict__.update(overrides)
        with mock.patch.object(workload.torch, 'use_deterministic_algorithms') as deterministic, \
                mock.patch.dict(os.environ, {'LOCAL_RANK': '5'}):
            check = workload.ReplicaCheck(args, torch.device('cpu'), rank=rank, world_size=world_size)
        deterministic.assert_called_once_with(True)
        return check

    def test_inputs_depend_only_on_seed_and_step(self):
        workload = load_workload()
        one, two = self.make(workload, 0), self.make(workload, 9)
        self.assertEqual(one.checksum(5).item(), two.checksum(5).item())
        self.assertNotEqual(one.checksum(5).item(), one.checksum(6).item())
        self.assertNotEqual(one.checksum(5).item(), self.make(workload, 0, replica_seed=1).checksum(5).item())

    def test_flip_changes_the_checksum_by_one_power_of_two_without_an_error(self):
        workload = load_workload()
        clean = self.make(workload, 13)
        for bit in (0, 22, 31):
            flipped = self.make(workload, 13, corrupt_rank=13, corrupt_step=37, corrupt_bit=bit)
            self.assertEqual(clean.checksum(36).item(), flipped.checksum(36).item(), bit)
            self.assertEqual(1 << bit, abs(clean.checksum(37).item() - flipped.checksum(37).item()), bit)
            self.assertEqual(clean.checksum(38).item(), flipped.checksum(38).item(), 'one-time flip')
        recurring = self.make(workload, 13, corrupt_rank=13, corrupt_step=37, corrupt_recurring=True)
        self.assertEqual(clean.checksum(36).item(), recurring.checksum(36).item())
        for step in (37, 38, 200):
            self.assertNotEqual(clean.checksum(step).item(), recurring.checksum(step).item(), step)
        other_rank = self.make(workload, 12, corrupt_rank=13, corrupt_step=37)
        self.assertEqual(clean.checksum(37).item(), other_rank.checksum(37).item())
        self.assertEqual(-2147483648, workload.int32_bit_mask(31))
        self.assertEqual(1 << 22, workload.int32_bit_mask(22))

    def gather_with_peers(self, workload, check, clean):
        """A fake all-gather: every other rank reports the clean checksum."""
        def all_gather(gathered, local):
            gathered.fill_(clean.checksum(self.step).item())
            gathered[check.rank] = local[0]
        return all_gather

    def run_steps(self, workload, check, clean, steps):
        from unittest import mock
        import contextlib
        import io
        output = io.StringIO()
        with mock.patch.object(workload.dist, 'all_gather_into_tensor',
                               side_effect=self.gather_with_peers(workload, check, clean)), \
                mock.patch.dict(os.environ, {'LOCAL_RANK': '5'}), \
                contextlib.redirect_stdout(output):
            for self.step in steps:
                check.step(self.step)
        return output.getvalue().splitlines()

    def test_the_differing_rank_names_itself_once_per_transition(self):
        import socket
        workload = load_workload()
        clean = self.make(workload, 0)
        check = self.make(workload, 13, corrupt_rank=13, corrupt_step=37)
        lines = self.run_steps(workload, check, clean, range(35, 40))
        self.assertEqual(2, len(lines), lines)
        self.assertTrue(lines[0].startswith('AIM344 replica check: step=37; rank 13 ('), lines[0])
        self.assertIn(f'({socket.gethostname()}, LOCAL_RANK 5, CUDA device None, UUID none, PCI none)', lines[0])
        self.assertIn(f'checksum {check.checksum(37).item()} differs from the other 15 ranks '
                      f'({clean.checksum(37).item()}).', lines[0])
        self.assertTrue(lines[1].startswith('AIM344 replica check: step=38; rank 13 ('), lines[1])
        self.assertIn('checksum agrees again', lines[1])
        import contextlib
        import io
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            check.report(40)
        self.assertEqual('AIM344 replica check: step=40; 15/16 checksums agree over the last 5 steps; '
                         'rank 13 differed on 1 of 5 steps.', output.getvalue().strip())
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            check.report(60)
        self.assertEqual('AIM344 replica check: step=60; 16/16 checksums agree over the last 0 steps.',
                         output.getvalue().strip())

    def test_a_recurring_flip_prints_once_and_the_report_counts_every_step(self):
        import contextlib
        import io
        workload = load_workload()
        clean = self.make(workload, 0)
        check = self.make(workload, 3, corrupt_rank=3, corrupt_step=2, corrupt_recurring=True)
        lines = self.run_steps(workload, check, clean, range(1, 21))
        self.assertEqual(1, len(lines), lines)
        self.assertIn('step=2; rank 3 (', lines[0])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            check.report(20)
        self.assertEqual('AIM344 replica check: step=20; 15/16 checksums agree over the last 20 steps; '
                         'rank 3 differed on 19 of 20 steps.', output.getvalue().strip())

    def test_the_other_ranks_print_nothing(self):
        workload = load_workload()
        clean = self.make(workload, 0)
        check = self.make(workload, 0, corrupt_rank=13, corrupt_step=2)
        lines = self.run_steps(workload, check, clean, range(1, 6))
        self.assertEqual([], lines)

    def test_cuda_needs_the_cublas_workspace_setting(self):
        import torch
        workload = load_workload()
        with self.assertRaisesRegex(SystemExit, 'CUBLAS_WORKSPACE_CONFIG=:4096:8'):
            workload.require_deterministic(torch.device('cuda'), {})
        with self.assertRaises(SystemExit):
            workload.require_deterministic(torch.device('cuda'), {'CUBLAS_WORKSPACE_CONFIG': ':0:0'})
        self.assertEqual((':4096:8', ':16:8'), workload.CUBLAS_DETERMINISTIC_WORKSPACES)
        source = (LAB / 'torch-node.sh').read_text()
        self.assertIn('export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}', source)


@unittest.skipUnless(HAS_TORCH, 'CPU torch required')
class DeviceLoopWithReplicaCheck(unittest.TestCase):
    """The device loop through main(): a step per collective, a report every 20."""

    def run_device(self, spec, env=None):
        result = subprocess.run([sys.executable, __file__, '--exercise-device', json.dumps(spec)],
                                text=True, capture_output=True, timeout=120,
                                env={**os.environ, 'OMP_NUM_THREADS': '1', 'LOCAL_RANK': '5',
                                     'SLURM_JOB_ID': '777', **(env or {})})
        print(f'--- device {spec} {env} rc={result.returncode} ---\n' + result.stdout + result.stderr,
              flush=True)
        return result

    def test_healthy_run_reports_agreement_beside_each_progress_record(self):
        result = self.run_device({'rank': 0, 'world_size': WORLD, 'duration': 2, 'replica_n': 32,
                                  'argv': ['--replica-check']})
        self.assertEqual(0, result.returncode, result.stderr)
        summary = test_result(result.stdout)
        self.assertEqual(summary['collectives'], summary['steps'] + 1)
        self.assertGreaterEqual(summary['steps'], 20)
        lines = result.stdout.splitlines()
        progress = [line for line in lines if line.startswith('AIM344 collective_progress=')]
        reports = [line for line in lines if line.startswith('AIM344 replica check:')]
        self.assertEqual(len(progress), len(reports))
        for before, after in zip(progress, reports):
            self.assertEqual(lines.index(before) + 1, lines.index(after))
            n = before.split('collective_progress=')[1].split(' ')[0]
            self.assertEqual(f'AIM344 replica check: step={n}; 16/16 checksums agree over the last 20 steps.',
                             after)
            self.assertIn('; job_id=777; world_size=16; rank=0.', before)
        self.assertIn('Device workload completed:', result.stdout)
        self.assertNotIn('differs', result.stdout)

    def test_the_settings_variable_selects_the_flipped_rank_without_touching_the_command_line(self):
        settings = {'corrupt_rank': 13, 'corrupt_step': 7, 'corrupt_bit': 22, 'corrupt_recurring': False}
        result = self.run_device({'rank': 13, 'world_size': WORLD, 'duration': 2, 'replica_n': 32},
                                 env={'AIM344_INCIDENT_SETTINGS': json.dumps(settings)})
        self.assertEqual(0, result.returncode, result.stderr)
        lines = result.stdout.splitlines()
        differs = [line for line in lines if 'differs from the other 15 ranks' in line]
        self.assertEqual(1, len(differs), differs)
        self.assertTrue(differs[0].startswith('AIM344 replica check: step=7; rank 13 ('), differs[0])
        self.assertIn('LOCAL_RANK 5', differs[0])
        agrees = [line for line in lines if 'checksum agrees again' in line]
        self.assertEqual(1, len(agrees), agrees)
        self.assertIn('step=8;', agrees[0])
        # The window report is rank 0's line; this harness plays rank 13.
        self.assertEqual([], [line for line in lines if 'checksums agree' in line])
        self.assertNotIn('corrupt', result.stdout)
        self.assertNotIn('Traceback', result.stderr)

    def test_a_recurring_flip_through_the_settings_variable(self):
        settings = {'corrupt_rank': 0, 'corrupt_step': 5, 'corrupt_recurring': True}
        result = self.run_device({'rank': 0, 'world_size': 4, 'duration': 2, 'replica_n': 32},
                                 env={'AIM344_INCIDENT_SETTINGS': json.dumps(settings)})
        self.assertEqual(0, result.returncode, result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual(1, len([line for line in lines if 'differs from the other 3 ranks' in line]))
        self.assertEqual([], [line for line in lines if 'agrees again' in line])
        self.assertIn('AIM344 replica check: step=20; 3/4 checksums agree over the last 20 steps; '
                      'rank 0 differed on 16 of 20 steps.', lines)
        self.assertIn('AIM344 replica check: step=40; 3/4 checksums agree over the last 20 steps; '
                      'rank 0 differed on 20 of 20 steps.', lines)

    def test_without_the_replica_check_the_loop_is_unchanged(self):
        result = self.run_device({'rank': 0, 'world_size': WORLD, 'duration': 1, 'replica_n': 32})
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(0, test_result(result.stdout)['steps'])
        self.assertNotIn('replica check', result.stdout)
        self.assertIn('AIM344 collective_progress=20 collectives;', result.stdout)


@unittest.skipUnless(HAS_TORCH, 'CPU torch required')
class IncidentSettings(unittest.TestCase):
    def parser(self):
        workload = load_workload()
        parser = argparse.ArgumentParser()
        parser.add_argument('mode', choices=['storage', 'dataloader', 'device'])
        parser.add_argument('--start-method', choices=['fork', 'spawn', 'forkserver'], default='fork')
        workload.add_exercise_arguments(parser)
        return workload, parser

    def apply(self, settings, argv=('device',)):
        workload, parser = self.parser()
        args = parser.parse_args(list(argv))
        env = {} if settings is None else {'AIM344_INCIDENT_SETTINGS': settings}
        return workload.check_exercise_arguments(
            parser, workload.apply_incident_settings(parser, args, env, argv=list(argv)))

    def test_settings_apply_with_the_options_validation_and_imply_the_replica_check(self):
        args = self.apply(json.dumps({'corrupt_rank': 13, 'corrupt_step': 37, 'corrupt_bit': 22,
                                      'corrupt_recurring': False}))
        self.assertEqual((13, 37, 22, False, True),
                         (args.corrupt_rank, args.corrupt_step, args.corrupt_bit,
                          args.corrupt_recurring, args.replica_check))
        args = self.apply(json.dumps({'corrupt_rank': 2, 'corrupt_recurring': True}))
        self.assertEqual((2, 1, 22, True), (args.corrupt_rank, args.corrupt_step, args.corrupt_bit,
                                            args.corrupt_recurring))
        args = self.apply(json.dumps({'dataset_cache': 'rank-thread', 'stack_dump_seconds': 60}),
                          ['dataloader'])
        self.assertEqual(('rank-thread', 60.0), (args.dataset_cache, args.stack_dump_seconds))
        self.assertEqual(('none', None, False), (self.apply(None).dataset_cache,
                                                 self.apply(None).corrupt_rank,
                                                 self.apply('').replica_check))

    def test_an_option_on_the_command_line_keeps_its_value(self):
        # A participant's fix flag on a rerun wins over the incident's setting.
        settings = json.dumps({'dataset_cache': 'rank-thread', 'stack_dump_seconds': 60})
        args = self.apply(settings, ['dataloader', '--dataset-cache', 'worker-thread'])
        self.assertEqual(('worker-thread', 60.0), (args.dataset_cache, args.stack_dump_seconds))
        args = self.apply(settings, ['dataloader', '--dataset-cache=worker-thread', '--stack-dump-seconds=5'])
        self.assertEqual(('worker-thread', 5.0), (args.dataset_cache, args.stack_dump_seconds))
        args = self.apply(json.dumps({'corrupt_recurring': True}), ['device', '--corrupt-recurring'])
        self.assertTrue(args.corrupt_recurring)
        # A command-line value is still refused when a setting is invalid elsewhere.
        with self.assertRaises(SystemExit):
            self.apply(json.dumps({'dataset_cache': 'rank-thread', 'bogus': 1}),
                       ['dataloader', '--dataset-cache', 'worker-thread'])

    def test_bad_settings_are_refused(self):
        for settings, argv in (('not json', ['device']), ('[1]', ['device']),
                               (json.dumps({'mode': 'storage'}), ['device']),
                               (json.dumps({'no_such': 1}), ['device']),
                               (json.dumps({'corrupt_rank': True}), ['device']),
                               (json.dumps({'corrupt_rank': -1}), ['device']),
                               (json.dumps({'corrupt_bit': 32}), ['device']),
                               (json.dumps({'corrupt_recurring': 1}), ['device']),
                               (json.dumps({'dataset_cache': 'lock'}), ['dataloader']),
                               (json.dumps({'help': True}), ['device']),
                               (json.dumps({'corrupt_rank': 3}), ['storage']),
                               (json.dumps({'dataset_cache': 'rank-thread'}), ['device']),
                               (json.dumps({'reinit_in_worker': True}), ['dataloader', '--start-method', 'spawn'])):
            with self.subTest(settings=settings, argv=argv):
                with self.assertRaises(SystemExit) as stop:
                    self.apply(settings, argv)
                self.assertEqual(2, stop.exception.code)

    def test_command_line_consistency(self):
        workload, parser = self.parser()
        for argv in (['storage', '--replica-check'], ['device', '--dataset-cache', 'rank-thread'],
                     ['device', '--worker-cuda'], ['dataloader', '--reinit-in-worker', '--start-method', 'forkserver']):
            with self.subTest(argv=argv):
                with self.assertRaises(SystemExit):
                    workload.check_exercise_arguments(parser, parser.parse_args(argv))
        args = workload.check_exercise_arguments(parser, parser.parse_args(
            ['dataloader', '--start-method', 'forkserver', '--dataset-cache', 'worker-thread']))
        self.assertEqual('forkserver', args.start_method)


class WorkloadCommandLine(unittest.TestCase):
    """workload.py's own parser, before any CUDA call."""

    def run_workload(self, argv, env=None):
        return subprocess.run([sys.executable, str(LAB / 'workload.py')] + argv, capture_output=True,
                              text=True, timeout=30, env={**os.environ, **(env or {})})

    @unittest.skipUnless(HAS_TORCH, 'CPU torch required')
    def test_help_and_refusals_happen_before_cuda(self):
        result = self.run_workload(['--help'])
        self.assertEqual(0, result.returncode)
        for flag in ('--dataset-cache', '--dataloader-timeout', '--stack-dump-seconds', '--reinit-in-worker',
                     '--worker-cuda', '--affected-rank', '--replica-check', '--corrupt-rank',
                     '{fork,spawn,forkserver}'):
            self.assertIn(flag, result.stdout)
        for argv, env in ((['device', '--corrupt-bit', '32'], None),
                          (['storage', '--replica-check'], None),
                          (['dataloader', '--reinit-in-worker', '--start-method', 'spawn'], None),
                          (['device'], {'AIM344_INCIDENT_SETTINGS': '{"corrupt_rank": "thirteen"}'}),
                          (['device'], {'AIM344_INCIDENT_SETTINGS': '{"writer": 1}'})):
            with self.subTest(argv=argv, env=env):
                result = self.run_workload(argv, env)
                self.assertEqual(2, result.returncode, result.stderr)
                self.assertNotIn('LOCAL_RANK', result.stderr)


class LauncherContract(unittest.TestCase):
    def test_replica_check_launcher_passes_its_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / 'common.sh').write_text(
                'prepare_slurm() { :; }\nrun_torch() { printf "%s\\n" "$*"; return "$TEST_RC"; }\n')
            name = '15.replica-check.sh'
            (directory / name).write_text((LAB / name).read_text())
            for rc in (0, 1, 124):
                result = subprocess.run(['bash', str(directory / name)], capture_output=True, text=True,
                                        env={**os.environ, 'TEST_RC': str(rc)}, timeout=5)
                self.assertEqual(rc, result.returncode)
                self.assertEqual('replica-check device --replica-check --duration-seconds=60\n', result.stdout)
            result = subprocess.run(['bash', str(directory / name), '--corrupt-rank', '13', '--corrupt-step', '37'],
                                    capture_output=True, text=True, env={**os.environ, 'TEST_RC': '0'}, timeout=5)
            self.assertEqual('replica-check device --replica-check --duration-seconds=60 '
                             '--corrupt-rank 13 --corrupt-step 37\n', result.stdout)
        self.assertTrue(os.access(LAB / '15.replica-check.sh', os.X_OK))


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--exercise-dataloader':
        exercise_dataloader(json.loads(sys.argv[2]))
    elif len(sys.argv) > 2 and sys.argv[1] == '--exercise-device':
        exercise_device(json.loads(sys.argv[2]))
    else:
        unittest.main()
