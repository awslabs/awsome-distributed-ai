# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""The checkpoint writer, the step loop and the lab-line filter, on the CPU.

Distributed calls are replaced by a recording fake for one rank at a time and the
collective is counted, not run, so no GPU, NCCL or second process is involved.
These tests say what each rank does; they do not reproduce a watchdog timeout.
"""
import contextlib
import errno
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import select
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
                writer_rank=0, checkpoint_every=0, mark_lab_lines=False)
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
                               side_effect=lambda path, step: (saves.append(step), real(path, step))):
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
    def fill(self, marked):
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
            self.workload.fill_checkpoint_path(self.path, args(mark_lab_lines=marked))
        return opened, output.getvalue()

    def test_the_filler_has_a_name_a_real_job_could_leave(self):
        opened, _ = self.fill(marked=True)
        self.assertEqual([self.path / 'ckpt-shard-00017.partial'], opened)
        self.assertNotIn('aim344', opened[0].name)

    def test_the_injector_line_is_marked_for_the_lab_log_only_when_asked(self):
        _, marked = self.fill(marked=True)
        _, plain = self.fill(marked=False)
        line = 'checkpoint injection: [Errno 28] No space left on device; filler_bytes=1048576 B'
        self.assertEqual('AIM344 lab: ' + line + '\n', marked)
        self.assertEqual(line + '\n', plain)

    # ---------------------------------------------------------------- arguments
    def test_arguments(self):
        parse = self.workload.parse_args
        self.assertEqual(9, parse(['storage', '--writer-rank', '9']).writer_rank)
        self.assertEqual(0, parse(['device']).writer_rank)
        self.assertEqual(200, parse(['device', '--checkpoint-every=200', '--inject']).checkpoint_every)
        for argv in (['storage', '--writer-rank', '-1'], ['storage', '--writer-rank', 'x'],
                     ['storage', '--checkpoint-every', '5'], ['device', '--inject'],
                     ['device', '--resume'], ['device', '--no-such-flag'],
                     ['storage', '--writer-rank']):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as stop:
                    parse(argv)
                self.assertEqual(2, stop.exception.code)

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


class LabLines(unittest.TestCase):
    def test_marked_lines_go_to_the_lab_log_and_the_rest_pass_through(self):
        with tempfile.TemporaryDirectory() as tmp:
            lab = Path(tmp) / 'incident-1-lab.log'
            lab.write_text('earlier\n')
            text = ('NCCL INFO Using network Libfabric\n'
                    'AIM344 lab: checkpoint injection: [Errno 28] No space left on device; filler_bytes=1 B\n'
                    'checkpoint write retry: [Errno 28] No space left on device\n'
                    'no newline at the end')
            result = subprocess.run([sys.executable, str(LAB / 'lab-lines.py'), str(lab)],
                                    input=text, capture_output=True, text=True, timeout=20)
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual('NCCL INFO Using network Libfabric\n'
                             'checkpoint write retry: [Errno 28] No space left on device\n'
                             'no newline at the end', result.stdout)
            self.assertEqual('earlier\nAIM344 lab: checkpoint injection: [Errno 28] No space '
                             'left on device; filler_bytes=1 B\n', lab.read_text())

    def test_each_line_is_passed_on_as_it_arrives(self):
        with tempfile.TemporaryDirectory() as tmp:
            process = subprocess.Popen([sys.executable, str(LAB / 'lab-lines.py'),
                                        str(Path(tmp) / 'lab.log')],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            try:
                process.stdin.write(b'AIM344 collective_progress=20 collectives\n')
                process.stdin.flush()
                # Read before the input ends: a buffering filter holds the line back.
                ready, _, _ = select.select([process.stdout], [], [], 20)
                self.assertTrue(ready, 'the filter held the line until its input ended')
                self.assertEqual(b'AIM344 collective_progress=20 collectives\n',
                                 process.stdout.readline())
            finally:
                process.stdin.close()
                process.wait(timeout=20)
                process.stdout.close()

    def test_usage(self):
        result = subprocess.run([sys.executable, str(LAB / 'lab-lines.py')],
                                capture_output=True, text=True, timeout=20)
        self.assertNotEqual(0, result.returncode)
        self.assertIn('Usage', result.stderr)


if __name__ == '__main__':
    unittest.main()
