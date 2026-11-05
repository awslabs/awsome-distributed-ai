# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""The checkpoint writer, the step loop, the rollback and the lab lines, on the CPU.

Distributed calls are replaced by a recording fake for one rank at a time and the
collective is counted, not run, so no GPU, NCCL or second process is involved.
These tests say what each rank does; they do not reproduce a watchdog timeout.
"""
import contextlib
import errno
import importlib.util
import io
import json
import multiprocessing
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

LAB = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec('torch') is not None
# The record the controller parses before an active device fault
# (facilitator/device-session.py, _parse_active_progress).
PROGRESS = re.compile(r'AIM344 collective_progress=(\d+) collectives; tensor_bytes=(\d+) B; '
                      r'timestamp=([0-9.]+) s since epoch; 0 mismatches; '
                      r'job_id=([1-9][0-9]*); world_size=([1-9][0-9]*); rank=0\.')


def load_workload():
    sys.path.insert(0, str(LAB))
    import workload
    return workload


class FakeDist:
    """One rank's view of the process group: broadcasts are recorded and filled."""

    def __init__(self, torch, rank, loops, world=16, start_value=0):
        self.torch, self.rank, self.world = torch, rank, world
        self.loops, self.start_value = loops, start_value
        self.seen = 0
        self.broadcasts = []

    def get_rank(self):
        return self.rank

    def get_world_size(self):
        return self.world

    def broadcast(self, tensor, src):
        if tensor.dtype == self.torch.int32:
            # keep_running from rank 0: run `loops` steps, then stop.
            self.seen += 1
            tensor.fill_(1 if self.seen <= self.loops else 0)
            self.broadcasts.append(('keep_running', src))
        else:
            # The start step comes from its source; other ranks receive it.
            if self.rank != src:
                tensor.fill_(self.start_value)
            self.broadcasts.append(('start_step', src, int(tensor.item())))


def args(**fields):
    base = dict(mode='device', duration_seconds=3600, inject=False, resume=False,
                writer_rank=0, checkpoint_every=0, lab_dir=None, end_after_save=False)
    base.update(fields)
    return SimpleNamespace(**base)


@unittest.skipUnless(HAS_TORCH, 'CPU torch required; the checkpoint workload is not tested')
class CheckpointWriter(unittest.TestCase):
    def setUp(self):
        self.workload = load_workload()
        patcher = mock.patch.object(self.workload, 'DEVICE', 'cpu')
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        (self.path / '.aim344-fixture').write_text('AIM344 isolated checkpoint fixture\n')
        self.collectives = 0

    def tearDown(self):
        self.tmp.cleanup()

    def collective(self, tensor):
        self.collectives += 1

    def run_rank(self, function, rank, options, loops=0, start_value=0, fill=None):
        fake = FakeDist(self.workload.torch, rank, loops, start_value=start_value)
        output = io.StringIO()
        fills = []

        def checkpoint_dir():
            return self.path

        def record_fill(path, given):
            fills.append(self.collectives)
            if fill:
                fill(path, given)

        with mock.patch.object(self.workload, 'dist', fake), \
                mock.patch.object(self.workload, 'collective', side_effect=self.collective), \
                mock.patch.object(self.workload, 'checkpoint_dir', side_effect=checkpoint_dir), \
                mock.patch.object(self.workload, 'fill_checkpoint_path', side_effect=record_fill), \
                contextlib.redirect_stdout(output):
            function(options, self.workload.torch.ones(4))
        return fake, output.getvalue(), fills

    def last(self):
        return json.loads((self.path / 'last.json').read_text())['next_step']

    # ---------------------------------------------------------------- step loop
    def test_device_assertion_stops_at_selected_step_and_keeps_prior_checkpoint(self):
        with mock.patch.object(self.workload.torch.cuda, 'synchronize'):
            with self.assertRaisesRegex(RuntimeError, 'batch validation condition failed'):
                self.run_rank(self.workload.run_steps, 0,
                              args(checkpoint_every=5, assert_rank=0, assert_step=6), loops=10)
        self.assertEqual(5, self.last())
        self.assertEqual(6, self.collectives)

    def test_device_assertion_options_require_device_mode_and_a_prior_checkpoint(self):
        valid = self.workload.parse_args(['device', '--checkpoint-every=5',
                                         '--assert-rank=0', '--assert-step=6'])
        self.assertEqual((0, 6), (valid.assert_rank, valid.assert_step))
        for argv in [['storage', '--assert-rank=0'],
                     ['device', '--assert-rank=0'],
                     ['device', '--checkpoint-every=5', '--assert-rank=0', '--assert-step=5'],
                     ['device', '--checkpoint-every=5', '--assert-rank=0', '--resume']]:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.workload.parse_args(argv)

    def test_other_ranks_and_corrected_resume_do_not_assert(self):
        with mock.patch.object(self.workload.torch, '_assert_async') as checked:
            self.run_rank(self.workload.run_steps, 1,
                          args(checkpoint_every=5, assert_rank=0, assert_step=6), loops=10)
            checked.assert_not_called()
            (self.path / 'last.json').write_text(json.dumps({'next_step': 5, 'saved': [0, 5]}))
            _, out, _ = self.run_rank(self.workload.run_steps, 0,
                                      args(checkpoint_every=5, resume=True), loops=6)
            checked.assert_not_called()
            self.assertIn('Resuming from last successful checkpoint: next_step=5', out)

    def test_writer_saves_the_start_and_every_nth_step_before_its_all_reduce(self):
        fake, out, _ = self.run_rank(self.workload.run_steps, 9,
                                     args(writer_rank=9, checkpoint_every=5), loops=12)
        self.assertEqual(['checkpoint saved: next_step=0 (step index)',
                          'checkpoint saved: next_step=5 (step index)',
                          'checkpoint saved: next_step=10 (step index)'],
                         [line for line in out.splitlines() if line.startswith('checkpoint')])
        self.assertEqual(10, self.last())
        self.assertEqual(12, self.collectives)
        self.assertEqual(('start_step', 9, 0), fake.broadcasts[0])
        # Only rank 0 prints the completion line.
        self.assertNotIn('Device workload completed', out)

    def test_other_ranks_never_touch_the_checkpoint_directory(self):
        def refuse():
            raise AssertionError('a non-writer rank opened the checkpoint directory')
        with mock.patch.object(self.workload, 'checkpoint_dir', side_effect=refuse):
            fake = FakeDist(self.workload.torch, 3, 12, start_value=0)
            output = io.StringIO()
            with mock.patch.object(self.workload, 'dist', fake), \
                    mock.patch.object(self.workload, 'collective', side_effect=self.collective), \
                    contextlib.redirect_stdout(output):
                self.workload.run_steps(args(writer_rank=9, checkpoint_every=5),
                                        self.workload.torch.ones(4))
        self.assertNotIn('checkpoint', output.getvalue())
        self.assertEqual(('start_step', 9, 0), fake.broadcasts[0])
        self.assertFalse((self.path / 'last.json').exists())

    def test_rank_zero_prints_progress_records_the_controller_parses(self):
        with mock.patch.dict(os.environ, {'SLURM_JOB_ID': '4242'}):
            _, out, _ = self.run_rank(self.workload.run_steps, 0,
                                      args(writer_rank=9, checkpoint_every=5), loops=40)
        records = [PROGRESS.fullmatch(line) for line in out.splitlines()
                   if line.startswith('AIM344 collective_progress=')]
        self.assertEqual(['20', '40'], [match.group(1) for match in records])
        self.assertTrue(all(match.group(4) == '4242' and match.group(5) == '16' for match in records))
        self.assertIn('Device workload completed: 40 collectives; 0 mismatches.', out)

    def test_inject_fills_the_writer_directory_just_before_its_first_periodic_save(self):
        saves = []
        real = self.workload.save_with_retry
        with mock.patch.object(self.workload, 'save_with_retry',
                               side_effect=lambda path, step, saved: (saves.append(step),
                                                                      real(path, step, saved))):
            _, out, fills = self.run_rank(self.workload.run_steps, 9,
                                          args(writer_rank=9, checkpoint_every=5, inject=True),
                                          loops=12)
        # Five steps ran before the fill, and the fill came once.
        self.assertEqual([5], fills)
        self.assertEqual([5, 10], saves)
        self.assertNotIn('injection', out)

    def test_resume_reads_the_writer_checkpoint_and_starts_every_rank_there(self):
        (self.path / 'last.json').write_text(json.dumps({'next_step': 7}))
        fake, out, _ = self.run_rank(self.workload.run_steps, 9,
                                     args(writer_rank=9, checkpoint_every=5, resume=True), loops=6)
        self.assertIn('Resuming from last successful checkpoint: next_step=7 (step index).', out)
        self.assertEqual(('start_step', 9, 7), fake.broadcasts[0])
        # Steps 7 to 12 ran; only step 10 is a checkpoint, and the start is not saved again.
        self.assertEqual(['checkpoint saved: next_step=10 (step index)'],
                         [line for line in out.splitlines() if line.startswith('checkpoint')])
        self.assertEqual(10, self.last())
        peer, _, _ = self.run_rank(self.workload.run_steps, 2,
                                   args(writer_rank=9, checkpoint_every=5, resume=True),
                                   loops=1, start_value=7)
        self.assertEqual(('start_step', 9, 7), peer.broadcasts[0])

    # ---------------------------------------------------------------- step order
    def record_steps(self, options, loops, last=None):
        """Rank 0's replica checks, saves and all-reduces, in the order the loop makes them."""
        events = []

        class Replica:
            def __init__(self, *args, **kwargs):
                pass

            def step(self, step):
                events.append(('replica', step))

            def report(self, step):
                events.append(('report', step))

        real = self.workload.save_with_retry

        def save(path, step, saved):
            events.append(('save', step))
            real(path, step, saved)

        def all_reduce(tensor):
            self.collectives += 1
            events.append(('all-reduce',))
        if last is not None:
            (self.path / 'last.json').write_text(json.dumps(last))
        self.collective = all_reduce
        with mock.patch.object(self.workload, 'ReplicaCheck', Replica), \
                mock.patch.object(self.workload, 'save_with_retry', side_effect=save):
            _, out, _ = self.run_rank(self.workload.run_steps, 0, options, loops=loops)
        return events, out

    def test_each_step_checks_the_replica_first_and_ends_with_its_all_reduce(self):
        events, _ = self.record_steps(args(checkpoint_every=5, replica_check=True), loops=20)
        self.assertEqual([('replica', 0), ('all-reduce',), ('replica', 1), ('all-reduce',)], events[:4])
        # The checkpoint comes after the step's replica check and before its all-reduce,
        # so a writer that cannot save keeps the other ranks in the all-reduce.
        start = events.index(('replica', 5))
        self.assertEqual([('replica', 5), ('save', 5), ('all-reduce',), ('replica', 6)],
                         events[start:start + 4])
        # The report after the 20th step names the last step checked, by step index.
        self.assertEqual(('report', 19), events[-1])

    def test_a_resumed_run_checks_the_replica_by_step_index(self):
        events, out = self.record_steps(args(checkpoint_every=5, replica_check=True, resume=True),
                                        loops=6, last={'next_step': 200, 'saved': [0, 200]})
        self.assertIn('Resuming from last successful checkpoint: next_step=200 (step index).', out)
        self.assertEqual([200, 201, 202, 203, 204, 205],
                         [step for kind, *step in events if kind == 'replica' for step in step])
        self.assertIn(('save', 205), events)

    def test_the_step_loop_restarts_its_stack_timer_at_each_progress_record(self):
        calls = []
        fake = SimpleNamespace(
            dump_traceback_later=lambda seconds, repeat, file: calls.append(('arm', seconds, repeat)),
            cancel_dump_traceback_later=lambda: calls.append(('cancel',)))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(self.workload, 'faulthandler', fake):
            self.run_rank(self.workload.run_steps, 3,
                          args(stack_dump_seconds=60, stack_dump_dir=tmp), loops=45)
            left = list(Path(tmp).iterdir())
        # Armed before the first step, restarted after steps 20 and 40 on every rank,
        # cancelled when the loop ends; a file that holds only its header is removed.
        self.assertEqual([('arm', 60, False)] * 3 + [('cancel',)], calls)
        self.assertEqual([], left)
        calls.clear()
        with mock.patch.object(self.workload, 'faulthandler', fake):
            self.run_rank(self.workload.run_steps, 3, args(), loops=45)
        self.assertEqual([], calls)

    def run_until_rank_zero_stops(self, options, start_value=0):
        """Rank 0 alone, with its own keep_running decision broadcast to itself."""
        fake = FakeDist(self.workload.torch, 0, 0, start_value=start_value)

        def broadcast(tensor, src):
            # Rank 0 is the source of both broadcasts, so it keeps its own values.
            fake.broadcasts.append(('keep_running' if tensor.dtype == self.workload.torch.int32
                                    else 'start_step', src, int(tensor.item())))
        fake.broadcast = broadcast
        output = io.StringIO()
        with mock.patch.object(self.workload, 'dist', fake), \
                mock.patch.object(self.workload, 'collective', side_effect=self.collective), \
                mock.patch.object(self.workload, 'checkpoint_dir', return_value=self.path), \
                contextlib.redirect_stdout(output):
            self.workload.run_steps(options, self.workload.torch.ones(4))
        return output.getvalue()

    def test_end_after_save_stops_twenty_steps_after_the_first_save_past_the_start(self):
        # The writer is rank 0 here, so one process makes the saves and the decision.
        for start, collectives, saves in ((0, 220, ['200']), (150, 70, ['200']),
                                          (200, 220, ['400'])):
            with self.subTest(start=start):
                self.collectives = 0
                (self.path / 'last.json').write_text(json.dumps({'next_step': start}))
                # The clock passes the duration only after 10,000 steps, so a loop
                # that ignores the save ends there instead of running for an hour.
                clock = lambda: 4000.0 if self.collectives >= 10000 else 0.0  # noqa: E731
                with mock.patch.object(self.workload.time, 'monotonic', side_effect=clock):
                    out = self.run_until_rank_zero_stops(
                        args(writer_rank=0, checkpoint_every=200, resume=True, end_after_save=True))
                self.assertEqual(collectives, self.collectives)
                self.assertEqual(saves, re.findall(r'^checkpoint saved: next_step=(\d+) ', out, re.M))
                self.assertIn(f'Device workload completed: {collectives} collectives; 0 mismatches.', out)
                # The save is followed by one more progress record.
                lines = out.splitlines()
                saved = lines.index(f'checkpoint saved: next_step={saves[0]} (step index)')
                self.assertTrue(any(line.startswith('AIM344 collective_progress=')
                                    for line in lines[saved:]), out)

    def test_end_after_save_is_still_bounded_by_the_duration(self):
        # The clock passes the 60-second bound after the fifth step.
        clock = lambda: 61.0 if self.collectives >= 5 else 0.0  # noqa: E731
        with mock.patch.object(self.workload.time, 'monotonic', side_effect=clock):
            out = self.run_until_rank_zero_stops(
                args(writer_rank=0, checkpoint_every=200, end_after_save=True, duration_seconds=60))
        self.assertEqual(5, self.collectives)
        self.assertNotIn('next_step=200', out)

    def test_without_end_after_save_the_duration_alone_ends_the_loop(self):
        clock = lambda: 61.0 if self.collectives >= 300 else 0.0  # noqa: E731
        with mock.patch.object(self.workload.time, 'monotonic', side_effect=clock):
            out = self.run_until_rank_zero_stops(
                args(writer_rank=0, checkpoint_every=200, duration_seconds=60))
        self.assertEqual(300, self.collectives)
        self.assertIn('checkpoint saved: next_step=200 (step index)', out)

    def test_without_checkpoints_the_loop_is_the_device_workload(self):
        fake, out, _ = self.run_rank(self.workload.run_steps, 0, args(), loops=20)
        self.assertEqual([('keep_running', 0)] * 21, fake.broadcasts)
        self.assertNotIn('checkpoint', out)
        self.assertFalse((self.path / 'last.json').exists())

    # ---------------------------------------------------------------- storage mode
    def test_storage_writer_rank_moves_the_save_the_fill_and_the_broadcast_source(self):
        fake, out, fills = self.run_rank(self.workload.storage, 9,
                                         args(mode='storage', writer_rank=9, inject=True))
        self.assertEqual(('start_step', 9, 0), fake.broadcasts[0])
        self.assertEqual([0], fills)
        self.assertEqual(1, self.last())
        self.assertEqual(1, self.collectives)
        self.assertNotIn('Storage workload completed', out)

    def test_storage_rank_zero_reports_but_does_not_write_when_another_rank_writes(self):
        fake, out, fills = self.run_rank(self.workload.storage, 0,
                                         args(mode='storage', writer_rank=9, inject=True),
                                         start_value=0)
        self.assertEqual([], fills)
        self.assertFalse((self.path / 'last.json').exists())
        self.assertEqual(('start_step', 9, 0), fake.broadcasts[0])
        self.assertIn('Storage workload completed: next_step=1 (step index); 0 mismatches.', out)

    def test_storage_resume_on_the_writer(self):
        (self.path / 'last.json').write_text(json.dumps({'next_step': 0}))
        fake, out, _ = self.run_rank(self.workload.storage, 9,
                                     args(mode='storage', writer_rank=9, resume=True))
        self.assertIn('Resuming from last successful checkpoint: next_step=0 (step index).', out)
        self.assertEqual(1, self.last())

    def test_storage_default_writer_is_rank_zero_as_before(self):
        fake, out, fills = self.run_rank(self.workload.storage, 0,
                                         args(mode='storage', inject=True))
        self.assertEqual(('start_step', 0, 0), fake.broadcasts[0])
        self.assertEqual([0], fills)
        self.assertIn('Storage workload completed: next_step=1 (step index); 0 mismatches.', out)

    # ---------------------------------------------------------------- the writer's retry
    def test_retry_prints_each_enospc_and_gives_up_after_sixty_seconds(self):
        full = OSError(errno.ENOSPC, 'No space left on device')
        clock = iter([0.0, 1.0, 30.0, 61.0])
        output = io.StringIO()
        with mock.patch.object(self.workload, 'save_checkpoint', side_effect=full), \
                mock.patch.object(self.workload.time, 'monotonic', side_effect=lambda: next(clock)), \
                mock.patch.object(self.workload.time, 'sleep'), \
                contextlib.redirect_stdout(output):
            with self.assertRaises(OSError):
                self.workload.save_with_retry(self.path, 1)
        self.assertEqual(['checkpoint write retry: [Errno 28] No space left on device'] * 3,
                         output.getvalue().splitlines())

    def test_retry_returns_once_a_save_succeeds(self):
        full = OSError(errno.ENOSPC, 'No space left on device')
        output = io.StringIO()
        with mock.patch.object(self.workload, 'save_checkpoint', side_effect=[full, full, None]), \
                mock.patch.object(self.workload.time, 'sleep'), \
                contextlib.redirect_stdout(output):
            self.workload.save_with_retry(self.path, 1)
        self.assertEqual(2, output.getvalue().count('checkpoint write retry'))

    # ---------------------------------------------------------------- the fill
    def fill(self, lab_dir):
        opened = []

        class Full:
            def __init__(self, path):
                self.written = 0
                opened.append(path)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def write(self, block):
                if self.written:
                    raise OSError(errno.ENOSPC, 'No space left on device')
                self.written += len(block)
                return len(block)

            def fileno(self):
                return 0

        def fake_open(path, mode, buffering=-1):
            self.assertEqual('xb', mode)
            return Full(path)

        output = io.StringIO()
        with mock.patch.object(self.workload.Path, 'open', fake_open), \
                contextlib.redirect_stdout(output):
            self.workload.fill_checkpoint_path(self.path, args(lab_dir=lab_dir))
        return opened, output.getvalue()

    def test_the_filler_has_a_name_a_real_job_could_leave(self):
        opened, _ = self.fill(lab_dir=None)
        self.assertEqual([self.path / 'ckpt-shard-00017.partial'], opened)
        self.assertNotIn('aim344', opened[0].name)

    def test_the_injector_line_goes_to_the_lab_directory_only_when_asked(self):
        lab = Path(self.tmp.name) / 'lab'
        _, kept = self.fill(lab_dir=str(lab))
        _, plain = self.fill(lab_dir=None)
        line = 'checkpoint injection: [Errno 28] No space left on device; filler_bytes=1048576 B'
        self.assertEqual('', kept)
        [record] = lab.iterdir()
        self.assertRegex(record.read_text(), r'^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z '
                                             + re.escape(line) + '\n$')
        self.assertEqual(line + '\n', plain)

    # ---------------------------------------------------------------- the rollback
    def test_the_writer_keeps_the_steps_of_the_runs_checkpoints(self):
        self.run_rank(self.workload.run_steps, 0, args(checkpoint_every=5), loops=12)
        self.assertEqual({'next_step': 10, 'saved': [0, 5, 10]},
                         json.loads((self.path / 'last.json').read_text()))
        # A new run from step 0 starts a new list.
        self.run_rank(self.workload.run_steps, 0, args(checkpoint_every=5), loops=3)
        self.assertEqual({'next_step': 0, 'saved': [0]}, json.loads((self.path / 'last.json').read_text()))

    def test_resume_before_rolls_back_to_the_last_checkpoint_saved_before_the_step(self):
        # Replica step S of a run from 0 is step index S - 1: next_step=5 was saved
        # before replica step 6 and holds replica step 5, so only S >= 6 keeps it.
        for before, start in ((1, 0), (5, 0), (6, 5), (7, 5), (11, 10), (40, 10)):
            with self.subTest(before=before):
                (self.path / 'last.json').write_text(json.dumps({'next_step': 10, 'saved': [0, 5, 10]}))
                self.collectives = 0
                fake, out, _ = self.run_rank(self.workload.run_steps, 0,
                                             args(checkpoint_every=5, resume=True, resume_before=before),
                                             loops=6)
                self.assertIn(f'Resuming from the last checkpoint saved before step={before}: '
                              f'next_step={start} (step index).', out)
                self.assertNotIn('last successful', out)
                self.assertEqual(('start_step', 0, start), fake.broadcasts[0])
                # The later checkpoints are dropped; the resumed run's saves follow.
                record = json.loads((self.path / 'last.json').read_text())
                expected = [step for step in (0, 5, 10) if step <= start]
                expected += [step for step in range(start + 1, start + 6) if step % 5 == 0]
                self.assertEqual(expected, record['saved'])

    def test_resume_without_resume_before_is_unchanged(self):
        (self.path / 'last.json').write_text(json.dumps({'next_step': 10, 'saved': [0, 5, 10]}))
        fake, out, _ = self.run_rank(self.workload.run_steps, 0,
                                     args(checkpoint_every=5, resume=True), loops=1)
        self.assertIn('Resuming from last successful checkpoint: next_step=10 (step index).', out)
        self.assertEqual(('start_step', 0, 10), fake.broadcasts[0])

    def test_a_checkpoint_without_saved_steps_rolls_back_only_to_itself(self):
        # The fixture's own record, and one written by storage mode, hold no list.
        self.assertEqual((0, [0]), self.workload.rollback_point({'next_step': 0}, 37))
        with self.assertRaisesRegex(SystemExit, 'No checkpoint was saved before step=37'):
            self.workload.rollback_point({'next_step': 200}, 37)

    def test_resume_before_needs_resume(self):
        parse = self.workload.parse_args
        self.assertEqual(37, parse(['device', '--checkpoint-every=200', '--resume',
                                    '--resume-before', '37']).resume_before)
        for argv in (['device', '--checkpoint-every=200', '--resume-before', '37'],
                     ['device', '--checkpoint-every=200', '--resume', '--resume-before', '0']):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as stop:
                    parse(argv)
                self.assertEqual(2, stop.exception.code)
        # job.sh passes the rollback point in the settings variable, as it does the flip.
        with mock.patch.dict(os.environ, {'AIM344_INCIDENT_SETTINGS': '{"resume_before": 37}'}):
            self.assertEqual(37, parse(['device', '--checkpoint-every=200', '--resume']).resume_before)

    # ---------------------------------------------------------------- arguments
    def test_arguments(self):
        parse = self.workload.parse_args
        self.assertEqual(9, parse(['storage', '--writer-rank', '9']).writer_rank)
        self.assertEqual(0, parse(['device']).writer_rank)
        self.assertEqual(200, parse(['device', '--checkpoint-every=200', '--inject']).checkpoint_every)
        for argv in (['storage', '--writer-rank', '-1'], ['storage', '--writer-rank', 'x'],
                     ['storage', '--checkpoint-every', '5'], ['device', '--inject'],
                     ['device', '--resume'], ['device', '--no-such-flag'],
                     ['storage', '--writer-rank'], ['device', '--end-after-save'],
                     ['storage', '--end-after-save'], ['storage', '--stack-dump-seconds', '60']):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as stop:
                    parse(argv)
                self.assertEqual(2, stop.exception.code)

    def test_dataloader_mode_refuses_the_checkpoint_options(self):
        parse = self.workload.parse_args
        self.assertEqual(0, parse(['dataloader', '--start-method', 'spawn', '--epochs', '2']).writer_rank)
        self.assertTrue(parse(['device', '--checkpoint-every=200', '--resume', '--end-after-save'])
                        .end_after_save)
        # job.sh keeps the DataLoader job's lifecycle lines off its output.
        self.assertEqual('/results/.lab-records',
                         parse(['dataloader', '--epochs', '2', '--lab-dir=/results/.lab-records']).lab_dir)
        for flags in (['--writer-rank', '9'], ['--writer-rank', '0'], ['--inject'], ['--resume'],
                      ['--checkpoint-every', '5']):
            with self.subTest(flags=flags):
                error = io.StringIO()
                with contextlib.redirect_stderr(error), self.assertRaises(SystemExit) as stop:
                    parse(['dataloader', '--epochs', '2', *flags])
                self.assertEqual(2, stop.exception.code)
                self.assertIn(flags[0], error.getvalue())

    def test_a_writer_rank_outside_the_job_stops_before_any_collective(self):
        fake = FakeDist(self.workload.torch, 0, 0)
        fake.init_process_group = mock.Mock()
        with mock.patch.object(self.workload, 'dist', fake), \
                mock.patch.object(self.workload.torch.cuda, 'set_device'), \
                mock.patch.object(self.workload, 'collective') as collective, \
                mock.patch.dict(os.environ, {'LOCAL_RANK': '0'}), \
                mock.patch.object(sys, 'argv', ['workload.py', 'storage', '--writer-rank', '16']):
            with self.assertRaisesRegex(SystemExit, 'not a rank of this 16-rank job'):
                self.workload.main()
        collective.assert_not_called()

    def test_the_recovery_removes_the_new_filler_name(self):
        text = (LAB / 'recover-checkpoint.py').read_text()
        self.assertIn("'ckpt-shard-00017.partial'", text)
        self.assertIn(self.workload.FILLER_NAME, text)


def write_lab_lines(lab_dir, worker, count):
    sys.path.insert(0, str(LAB))
    import workload
    pad = 'x' * 150
    for n in range(count):
        workload.lab_line(lab_dir, 'DataLoader lifecycle: '
                          + json.dumps({'worker': worker, 'n': n, 'pad': pad}))


@unittest.skipUnless(HAS_TORCH, 'CPU torch required to import workload.py')
class LabLine(unittest.TestCase):
    """workload.lab_line: what job.sh's processes write instead of standard output."""

    def setUp(self):
        self.workload = load_workload()

    def test_without_a_directory_the_line_prints(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.workload.lab_line(None, 'checkpoint injection: full')
        self.assertEqual('checkpoint injection: full\n', output.getvalue())

    def test_with_a_directory_nothing_prints_and_each_process_has_its_own_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            lab = Path(tmp) / 'records'
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.workload.lab_line(str(lab), 'After first collective: x')
            self.assertEqual('', output.getvalue())
            self.assertEqual(0o700, lab.stat().st_mode & 0o777)
            # Eight processes at once, as the ranks and workers of one node write.
            context = multiprocessing.get_context('fork')
            processes = [context.Process(target=write_lab_lines, args=(str(lab), worker, 200))
                         for worker in range(8)]
            for process in processes:
                process.start()
            for process in processes:
                process.join(60)
                self.assertEqual(0, process.exitcode)
            files = sorted(lab.iterdir())
            self.assertEqual(9, len(files), files)
            stamp = re.compile(r'^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z ')
            for path in files:
                self.assertRegex(path.name, r'^.+-pid\d+\.log$')
                self.assertEqual(0o600, path.stat().st_mode & 0o777)
                for line in path.read_text().splitlines():
                    self.assertRegex(line, stamp)
                    text = stamp.sub('', line)
                    if text.startswith('DataLoader lifecycle: '):
                        record = json.loads(text.split(': ', 1)[1])
                        self.assertEqual({'worker', 'n', 'pad'}, set(record))
            counts = sorted(len(path.read_text().splitlines()) for path in files)
            self.assertEqual([1] + [200] * 8, counts)


@unittest.skipUnless(HAS_TORCH, 'CPU torch required; the stack files are not tested')
class StackFiles(unittest.TestCase):
    """A run moves the stack files an earlier run in the allocation left."""

    def setUp(self):
        self.workload = load_workload()
        host = mock.patch.object(self.workload.socket, 'gethostname', return_value='h')
        host.start()
        self.addCleanup(host.stop)

    def test_on_a_shared_directory_each_node_moves_only_its_own_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for name in ('stacks-h-rank3-rank-pid10.txt', 'stacks-h2-rank9-rank-pid12.txt',
                         'stacks-h-1-rank8-rank-pid13.txt'):
                (directory / name).write_text('old\n')
            self.workload.retire_stack_files(tmp)
            self.assertEqual(['stacks-h-rank3-rank-pid10.txt'],
                             [p.name for p in (directory / 'previous-stacks').iterdir()])
            # The other nodes' files stay for their own nodes to move.
            self.assertEqual(['stacks-h-1-rank8-rank-pid13.txt', 'stacks-h2-rank9-rank-pid12.txt'],
                             sorted(p.name for p in directory.glob('stacks-*')))
            with mock.patch.object(self.workload.socket, 'gethostname', return_value='h2'):
                self.workload.retire_stack_files(tmp)
            self.assertEqual(['stacks-h-1-rank8-rank-pid13.txt'],
                             [p.name for p in directory.glob('stacks-*')])

    def test_the_earlier_files_leave_the_directory_the_stack_read_lists(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / 'stacks-h-rank3-rank-pid10.txt').write_text('old rank\n')
            (directory / 'stacks-h-rank3-worker0-pid11.txt').write_text('old worker\n')
            (directory / 'job-T.log').write_text('kept\n')
            self.workload.retire_stack_files(tmp)
            self.assertEqual([], list(directory.glob('stacks-*')))
            self.assertEqual(['job-T.log', 'previous-stacks'], sorted(p.name for p in directory.iterdir()))
            self.assertEqual('old worker\n',
                             (directory / 'previous-stacks' / 'stacks-h-rank3-worker0-pid11.txt').read_text())
            self.workload.retire_stack_files(tmp)
            self.assertEqual(2, len(list((directory / 'previous-stacks').iterdir())))
        with tempfile.TemporaryDirectory() as tmp:
            self.workload.retire_stack_files(tmp)
            self.assertEqual([], list(Path(tmp).iterdir()))

    def test_one_process_per_node_moves_them_before_the_first_collective(self):
        for local_rank, rank, moved in (('0', 8, True), ('3', 3, False)):
            with self.subTest(local_rank=local_rank), tempfile.TemporaryDirectory() as tmp:
                old = Path(tmp) / 'stacks-h-rank8-rank-pid10.txt'
                old.write_text('old\n')
                seen = []
                fake = FakeDist(self.workload.torch, rank, 0)
                fake.init_process_group = mock.Mock()
                fake.destroy_process_group = mock.Mock()
                argv = ['workload.py', 'dataloader', '--epochs', '1', '--stack-dump-seconds', '60',
                        '--stack-dump-dir', tmp]
                with mock.patch.object(self.workload, 'dist', fake), \
                        mock.patch.object(self.workload, 'DEVICE', 'cpu'), \
                        mock.patch.object(self.workload.torch.cuda, 'set_device'), \
                        mock.patch.object(self.workload, 'collective',
                                          side_effect=lambda tensor: seen.append(old.exists())), \
                        mock.patch.object(self.workload, 'dataloader'), \
                        mock.patch.dict(os.environ, {'LOCAL_RANK': local_rank}), \
                        mock.patch.object(sys, 'argv', argv):
                    self.workload.main()
                self.assertEqual([not moved], seen)
                self.assertEqual(moved, (Path(tmp) / 'previous-stacks' / old.name).exists())


def emit_lines(lab, worker, count):
    sys.path.insert(0, lab)
    import workload
    for n in range(count):
        workload.emit(f'AIM344 replica check: step={n}; worker {worker} ' + 'x' * 150)


@unittest.skipUnless(HAS_TORCH, 'CPU torch required to import workload.py')
class OneWritePerLine(unittest.TestCase):
    """workload.emit: the lines of several unbuffered processes on one pipe stay whole."""

    def test_emit_is_one_write_of_the_whole_line(self):
        workload = load_workload()
        writes = []
        recorder = SimpleNamespace(write=writes.append, flush=lambda: None)
        with mock.patch.object(sys, 'stdout', recorder):
            workload.emit('checkpoint saved: next_step=200 (step index)')
        self.assertEqual(['checkpoint saved: next_step=200 (step index)\n'], writes)

    def test_eight_python_u_processes_on_one_pipe_print_whole_lines(self):
        # torchrun starts every worker as `python -u`, and the ranks of a node share
        # one output pipe.
        code = ('import sys\nsys.path.insert(0, sys.argv[1])\nimport test_checkpoint_workload as t\n'
                't.emit_lines(sys.argv[2], int(sys.argv[3]), 300)\n')
        read, write = os.pipe()
        processes = [subprocess.Popen([sys.executable, '-u', '-c', code, str(Path(__file__).parent),
                                       str(LAB), str(worker)], stdout=write)
                     for worker in range(8)]
        os.close(write)
        with os.fdopen(read, 'rb') as stream:
            data = stream.read().decode()
        for process in processes:
            self.assertEqual(0, process.wait(120))
        lines = data.split('\n')
        self.assertEqual('', lines.pop())
        line = re.compile(r'AIM344 replica check: step=\d+; worker [0-7] x{150}')
        self.assertEqual([], [text for text in lines if not line.fullmatch(text)][:5])
        self.assertEqual(8 * 300, len(lines))


@unittest.skipUnless(HAS_TORCH, 'CPU torch required; the progress record is not tested')
class ProgressRecord(unittest.TestCase):
    def test_the_record_is_one_write_of_the_whole_line(self):
        workload = load_workload()
        writes = []

        class Recorder:
            def write(self, text):
                writes.append(text)

            def flush(self):
                pass

        with mock.patch.object(workload.dist, 'get_world_size', return_value=16), \
                mock.patch.dict(os.environ, {'SLURM_JOB_ID': '4242'}), \
                mock.patch.object(sys, 'stdout', Recorder()):
            workload.progress_record(3, workload.torch.ones(1024 * 1024))
        [line] = writes
        self.assertTrue(line.endswith('\n'))
        match = PROGRESS.fullmatch(line[:-1])
        self.assertEqual(('3', '4194304', '4242', '16'), match.group(1, 2, 4, 5))


if __name__ == '__main__':
    unittest.main()
