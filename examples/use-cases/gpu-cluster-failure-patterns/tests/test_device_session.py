# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Rejection and success paths for the participant device session helper.

These tests run without cluster hardware. Scheduler and maintenance-route calls
go through a recorded fake executor, so a test asserts what the helper decided
and what it would have run, never what a GPU did. Hardware behaviour is a
separate acceptance track.

Run: python3 -m unittest discover -s <lab>/tests -p 'test_*.py'
"""
import importlib.util
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest

LAB = Path(__file__).resolve().parents[1]
HELPER = LAB / 'facilitator' / 'device-session.py'


def load_helper():
    spec = importlib.util.spec_from_file_location('aim344_device_session', HELPER)
    if spec is None or spec.loader is None:
        raise unittest.SkipTest(f'Helper not importable: {HELPER}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


session = load_helper()


class FakeExecutor:
    """Record maintenance-route and scheduler calls; answer from a script."""

    def __init__(self):
        self.maintenance_calls = []
        self.slurm_calls = []
        self.node_state = 'IDLE+CLOUD'
        self.node_reason = ''
        self.jobs = []            # list of '<id>|<user>|<name>|<state>'
        self.boot_id = 'aaaaaaaa-0000-0000-0000-000000000001'
        self.inspect = {
            'instance_id': 'i-0123456789abcdef0',
            'gpu_uuid': 'GPU-11111111-2222-4333-8444-555555555555',
            'gpu_bdf': '0000:ba:00.0',
            'efa_bdf': '0000:b0:00.0',
            'efa_rdma_device': 'rdmap176s0',
            'management_interface': 'enp71s0',
            'management_bdf': '0000:47:00.0',
            'confirmation': {
                'gpu-remove': 'i-0123456789abcdef0/gpu-remove/GPU-11111111-2222-4333-8444-555555555555',
                'efa-unbind': 'i-0123456789abcdef0/efa-unbind/0000:b0:00.0',
                'efa-rebind': 'i-0123456789abcdef0/efa-rebind/0000:b0:00.0',
            },
        }
        self.maintenance_failures = {}   # first word of argv -> (rc, stderr)
                                         # or (rc, stdout, stderr), because what the
                                         # target PRINTED before it failed is evidence
                                         # the controller reads: gpu-prepare writes
                                         # 'recorded persistence_mode=...' to stdout
                                         # once both original-state records exist and
                                         # before it changes either, so a refusal that
                                         # carries that line is a different round from
                                         # one that does not.
        # What the target's `efa-activity` observation reports. The default is a
        # qualifying active workload, so the existing active-EFA success path
        # still passes; the R9 tests vary these to drive each refusal.
        self.efa_activity_moved_bytes = 64 * 1024 * 1024
        self.efa_activity_runtime = '2:05'
        self.efa_activity_queue_readable = True
        self.efa_activity_jobs_after = None   # None means "same as before"
        self.efa_activity_no_counters = False
        # Collect output must look like the pinned suite's real output, because
        # the helper now reads the suite's own verdict line to decide whether a
        # check passed. lib/common.sh emits these with no colour escapes when
        # stdout is not a terminal, which is the case over the maintenance route;
        # confirmed against the collected logs on the target.
        self.collect_output = (
            '# suite_revision=a4ba07eb15e6f277063b4000346f9109c98de843\n'
            '[PASS] 0-nvidia-smi: nvidia-smi OK, 8 GPU(s) detected\n')

    # -- maintenance route (coordinator root -> target root, forced command) --
    def run_maintenance(self, argv, timeout=None):
        self.maintenance_calls.append(list(argv))
        action = argv[0]
        if action in self.maintenance_failures:
            scripted = self.maintenance_failures[action]
            if len(scripted) == 3:
                rc, out, err = scripted
            else:
                rc, err = scripted
                out = ''
            return session.Completed(rc, out, err)
        if action == 'inspect':
            return session.Completed(0, json.dumps(dict(self.inspect, action='inspect')) + '\n', '')
        if action == 'boot-id':
            return session.Completed(0, self.boot_id + '\n', '')
        if action == 'instance-id':
            # The identity route that does NOT depend on any device existing.
            # A GPU round removes the provisioned PCI function, so the real
            # device helper's inspect refuses during a fault; instance-id reads
            # the DMI field instead. The fake mirrors that: it answers even when
            # inspect is configured to fail.
            return session.Completed(0, self.inspect['instance_id'] + '\n', '')
        if action == 'collect':
            return session.Completed(0, self.collect_output, '')
        if action == 'efa-activity':
            return session.Completed(0, self._efa_activity_record() + '\n', '')
        return session.Completed(0, json.dumps({'action': action}) + '\n', '')

    def _efa_activity_record(self):
        """The same JSON shape maintenance.efa_activity prints on the target.

        Built from the fake's own job list, so a test that sets `jobs` does not
        also have to restate them here and cannot accidentally describe a job the
        scheduler side does not have.
        """
        device = self.inspect['efa_rdma_device']
        before = {'1/tx_bytes': 1_000_000, '1/rx_bytes': 1_000_000}
        after = {'1/tx_bytes': 1_000_000 + self.efa_activity_moved_bytes,
                 '1/rx_bytes': 1_000_000}
        deltas = {name: after[name] - before[name] for name in after}
        if self.efa_activity_no_counters:
            before, after, deltas = {}, {}, {}

        def rows(runtime):
            listing = []
            for job in self.jobs:
                parts = job.split('|')
                while len(parts) < 4:
                    parts.append('RUNNING')
                listing.append({'id': parts[0], 'user': parts[1],
                                'name': parts[2], 'state': parts[3],
                                'runtime': runtime})
            return listing

        before_rows = rows(self.efa_activity_runtime)
        after_rows = (before_rows if self.efa_activity_jobs_after is None
                      else self.efa_activity_jobs_after)
        return json.dumps({
            'action': 'efa-activity',
            'device': device,
            'slurm_node': 'gpu-g7-1',
            'interval_s': 6.0,
            'counters_before': before,
            'counters_after': after,
            'deltas_bytes': deltas,
            'jobs_before': before_rows,
            'jobs_after': after_rows,
            'queue_readable': self.efa_activity_queue_readable,
        }, sort_keys=True)

    # -- scheduler calls, run locally on the coordinator as root --
    def run_slurm(self, argv, timeout=None):
        self.slurm_calls.append(list(argv))
        program = Path(argv[0]).name
        if program == 'scontrol' and argv[1:3] == ['show', 'node']:
            reason = '\n   Reason=' + self.node_reason.replace('\n', '\n          ') if self.node_reason else ''
            return session.Completed(
                0, f'NodeName={argv[3]} State={self.node_state}{reason}\n   '
                   'InstanceId=i-0123456789abcdef0\n', '')
        if program == 'squeue':
            return session.Completed(0, ''.join(job + '\n' for job in self.jobs), '')
        if program == 'scontrol' and argv[1] == 'update':
            for item in argv[2:]:
                if item.startswith('State='):
                    if item == 'State=DRAIN':
                        self.node_state = 'IDLE+DRAIN'
                    elif item == 'State=RESUME':
                        self.node_state = 'IDLE+CLOUD'
                        self.node_reason = ''
                if item.startswith('Reason='):
                    self.node_reason = item.split('=', 1)[1]
            return session.Completed(0, '', '')
        if program == 'scontrol' and argv[1] == 'reboot':
            self.boot_id = 'bbbbbbbb-0000-0000-0000-000000000002'
            return session.Completed(0, '', '')
        return session.Completed(0, '', '')


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.state_dir = root / 'state'
        self.state_dir.mkdir()
        os.chmod(self.state_dir, 0o700)
        self.answers = root / 'answers'
        self.answers.mkdir()
        self.config_path = root / 'device-session.json'
        self.config_body = {
            'state_dir': str(self.state_dir),
            'assignments': {
                'table-1': {
                    'participant_user': 'aim344-t1',
                    'participant_uid': os.getuid(),
                    'caller_uid': os.getuid(),
                    'answers_dir': str(self.answers),
                    'target_node': 'gpu-g7-1',
                    'target_instance_id': 'i-0123456789abcdef0',
                    'coordinator_node': 'gpu-g7-2',
                    'partition': 'gpu-g7',
                    'drain_reason': 'aim344-device-recovery',
                    'job_name_prefix': 'aim344-',
                    'slurm_bin': '/opt/aws/pcs/scheduler/slurm-25.05/bin',
                    'allowed_checks': {'gpu': [0, 3], 'efa': [2, 6]},
                    'recovery_deadline_seconds': 1800,
                },
                'table-2': {
                    'participant_user': 'aim344-t2',
                    'participant_uid': os.getuid() + 4242,
                    'caller_uid': os.getuid() + 4242,
                    'answers_dir': str(root / 'answers-2'),
                    'target_node': 'gpu-g7-9',
                    'target_instance_id': 'i-0000000000000dead',
                    'coordinator_node': 'gpu-g7-2',
                    'partition': 'gpu-g7',
                    'drain_reason': 'aim344-device-recovery',
                    'job_name_prefix': 'aim344-',
                    'slurm_bin': '/opt/aws/pcs/scheduler/slurm-25.05/bin',
                    'allowed_checks': {'gpu': [0, 3], 'efa': [2, 6]},
                    'recovery_deadline_seconds': 1800,
                },
            },
        }
        self.write_config()
        self.executor = FakeExecutor()

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config_body))
        os.chmod(self.config_path, 0o600)

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, assignment='table-1', peer_uid=None, now=1000.0):
        config = session.load_config(self.config_path, require_root_owned=False)
        return session.DeviceSession(
            config=config,
            assignment_id=assignment,
            peer_uid=os.getuid() if peer_uid is None else peer_uid,
            executor=self.executor,
            clock=lambda: now,
        )


# --------------------------------------------------------------------------
# Rejection paths. These are written first and must fail before the helper
# exists, then reject for the stated reason once it does.
# --------------------------------------------------------------------------
class Rejections(Base):
    def test_config_must_be_root_owned_and_private(self):
        os.chmod(self.config_path, 0o644)
        with self.assertRaises(session.Refusal) as caught:
            session.load_config(self.config_path, require_root_owned=False)
        self.assertIn('0600', str(caught.exception))

    def test_unknown_assignment_is_refused(self):
        with self.assertRaises(session.Refusal) as caught:
            self.make(assignment='table-99')
        self.assertIn('assignment', str(caught.exception).lower())

    def test_cross_assignment_peer_uid_is_refused(self):
        """table-1's key used from another table's OS user."""
        with self.assertRaises(session.Refusal) as caught:
            self.make(assignment='table-1', peer_uid=os.getuid() + 4242)
        message = str(caught.exception).lower()
        self.assertIn('assignment', message)
        self.assertNotIn('gpu-g7-1', str(caught.exception))

    def test_participant_cannot_select_another_assignment_by_argument(self):
        """The dispatcher takes the assignment from the forced command only."""
        parsed = session.parse_participant_request('start gpu --assignment table-2')
        self.assertIsNone(parsed.error and None)
        self.assertIsNotNone(parsed.error)
        self.assertIn('not accepted', parsed.error)

    def test_participant_cannot_select_node_bdf_job_or_shell(self):
        for text in ['start gpu --node gpu-g7-2',
                     'start gpu --bdf 0000:ba:00.0',
                     'start efa --job 77',
                     'status; cat /etc/shadow',
                     'collect 0 && id',
                     'recover $(id)',
                     'start gpu\nstart efa']:
            with self.subTest(text=text):
                parsed = session.parse_participant_request(text)
                self.assertIsNotNone(parsed.error, f'accepted: {text!r}')

    def test_unknown_verb_is_refused(self):
        parsed = session.parse_participant_request('reboot')
        self.assertIsNotNone(parsed.error)

    def test_start_kind_must_be_gpu_or_efa(self):
        parsed = session.parse_participant_request('start nvlink')
        self.assertIsNotNone(parsed.error)

    def test_collect_check_must_be_in_the_allowlist_for_the_fault_kind(self):
        handler = self.make()
        handler.store.write(session.State(phase='fault-applied', kind='gpu',
                                          started_at=1.0, target_node='gpu-g7-1'))
        with self.assertRaises(session.Refusal) as caught:
            handler.collect('5')
        self.assertIn('5', str(caught.exception))
        self.assertEqual([], self.executor.maintenance_calls)

    def test_collect_rejects_non_numeric_check(self):
        handler = self.make()
        handler.store.write(session.State(phase='fault-applied', kind='gpu',
                                          started_at=1.0, target_node='gpu-g7-1'))
        for value in ['0;id', '../0', '0 3', '-3', '']:
            with self.subTest(value=value):
                with self.assertRaises(session.Refusal):
                    handler.collect(value)

    def test_gpu_start_refuses_when_any_job_holds_the_target(self):
        self.executor.jobs = ['91|aim344-t1|aim344-baseline|RUNNING']
        handler = self.make()
        with self.assertRaises(session.Refusal) as caught:
            handler.start('gpu')
        self.assertIn('idle', str(caught.exception).lower())
        self.assertNotIn(['gpu-remove'], [c[:1] for c in self.executor.maintenance_calls])

    def test_start_refuses_a_job_owned_by_another_user(self):
        self.executor.jobs = ['92|someone-else|aim344-thing|RUNNING']
        handler = self.make()
        with self.assertRaises(session.Refusal) as caught:
            handler.start('efa')
        self.assertIn('unapproved', str(caught.exception).lower())
        self.assertEqual([], [c for c in self.executor.maintenance_calls
                              if c[0].startswith('efa-unbind')])

    def test_start_refuses_a_job_outside_the_exercise_name_scope(self):
        self.executor.jobs = ['93|aim344-t1|my-own-training|RUNNING']
        handler = self.make()
        with self.assertRaises(session.Refusal):
            handler.start('efa')

    def test_start_preserves_an_unrelated_drain_reason(self):
        self.executor.node_state = 'IDLE+DRAIN'
        self.executor.node_reason = 'hardware-under-investigation'
        handler = self.make()
        with self.assertRaises(session.Refusal) as caught:
            handler.start('gpu')
        self.assertIn('hardware-under-investigation', str(caught.exception))
        self.assertEqual([], [c for c in self.executor.slurm_calls
                              if c[1:2] == ['update']])

    def test_changed_device_identity_is_surfaced_not_retried(self):
        self.executor.maintenance_failures['inspect'] = (
            1, 'GPU UUID/BDF mapping changed.\n')
        handler = self.make()
        with self.assertRaises(session.Refusal) as caught:
            handler.start('gpu')
        self.assertIn('GPU UUID/BDF mapping changed', str(caught.exception))
        actions = [c[0] for c in self.executor.maintenance_calls]
        self.assertNotIn('gpu-remove', actions)
        self.assertEqual(1, actions.count('inspect'))

    def test_changed_instance_identity_stops_the_start(self):
        """The target reporting a different instance is a replacement, not a fault
        target. Covers the guard separately from the inspect-failure path."""
        self.executor.inspect['instance_id'] = 'i-0999999999999999f'
        handler = self.make()
        with self.assertRaises(session.Refusal) as caught:
            handler.start('gpu')
        self.assertIn('instance identity', str(caught.exception).lower())
        actions = [c[0] for c in self.executor.maintenance_calls]
        self.assertNotIn('gpu-remove', actions)
        self.assertEqual([], [c for c in self.executor.slurm_calls
                              if c[1:2] == ['update']])

    def test_changed_instance_identity_stops_recovery_before_restore(self):
        handler = self.make()
        handler.start('gpu')
        self.executor.inspect['instance_id'] = 'i-0999999999999999f'
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        self.assertIn('instance', str(caught.exception).lower())
        actions = [c[0] for c in self.executor.maintenance_calls]
        self.assertNotIn('restore-runtime', actions)
        self.assertEqual([], [c for c in self.executor.slurm_calls
                              if 'State=RESUME' in c])
        # The identity check now runs before the reboot as well, not only before
        # restoration: review finding F8.
        self.assertEqual([], [c for c in self.executor.slurm_calls
                              if c[1:2] == ['reboot']])

    def test_concurrent_start_refusal_is_raised_not_returned(self):
        """A swallowed lock failure would let two connections inject; assert the
        lock helper itself raises rather than returning quietly."""
        first = self.make()
        second = self.make()
        with first.store.exclusive():
            with self.assertRaises(session.Refusal):
                second.store._acquire()
            self.assertIsNone(second.store._lock_handle)

    def test_management_interface_is_never_a_target(self):
        """No participant input reaches a device selector, and the token the
        helper sends names the provisioned fault device only."""
        handler = self.make()
        handler.start('gpu')
        sent = [c for c in self.executor.maintenance_calls if c[0] == 'gpu-remove']
        self.assertEqual(1, len(sent))
        flat = ' '.join(sent[0])
        self.assertNotIn(self.executor.inspect['management_bdf'], flat)
        self.assertNotIn(self.executor.inspect['management_interface'], flat)
        self.assertIn(self.executor.inspect['confirmation']['gpu-remove'], flat)

    def test_a_token_for_one_operation_is_not_reused_for_another(self):
        handler = self.make()
        handler.start('gpu')
        sent = [c for c in self.executor.maintenance_calls if c[0] == 'gpu-remove'][0]
        self.assertNotIn(self.executor.inspect['confirmation']['efa-unbind'], ' '.join(sent))

    def test_duplicate_start_does_not_inject_a_second_fault(self):
        handler = self.make()
        handler.start('gpu')
        first = [c[0] for c in self.executor.maintenance_calls].count('gpu-remove')
        result = self.make().start('gpu')
        second = [c[0] for c in self.executor.maintenance_calls].count('gpu-remove')
        self.assertEqual(first, second)
        self.assertTrue(result.already_started)
        self.assertEqual('fault-applied', result.state.phase)

    def test_duplicate_start_of_the_other_kind_is_refused(self):
        self.make().start('gpu')
        with self.assertRaises(session.Refusal) as caught:
            self.make().start('efa')
        self.assertIn('gpu', str(caught.exception).lower())

    def test_concurrent_start_loses_the_exclusion(self):
        first = self.make()
        second = self.make()
        with first.store.exclusive():
            with self.assertRaises(session.Refusal) as caught:
                second.start('gpu')
        self.assertIn('in progress', str(caught.exception).lower())
        self.assertEqual([], self.executor.maintenance_calls)

    def test_recover_before_start_is_refused(self):
        with self.assertRaises(session.Refusal):
            self.make().recover()

    def test_repeated_recover_is_idempotent(self):
        handler = self.make()
        handler.start('gpu')
        self.make().recover()
        reboots = len([c for c in self.executor.slurm_calls if c[1:2] == ['reboot']])
        again = self.make().recover()
        self.assertEqual(reboots,
                         len([c for c in self.executor.slurm_calls if c[1:2] == ['reboot']]))
        self.assertTrue(again.already_recovered)

    def test_collect_refuses_before_a_fault_exists(self):
        with self.assertRaises(session.Refusal):
            self.make().collect('0')

    def test_answers_directory_symlink_is_refused(self):
        target = Path(self.tmp.name) / 'sensitive'
        target.mkdir()
        (target / 'keep').write_text('original\n')
        answers = Path(self.config_body['assignments']['table-1']['answers_dir'])
        answers.rmdir()
        answers.symlink_to(target)
        handler = self.make()
        handler.store.write(session.State(phase='fault-applied', kind='gpu',
                                          started_at=1.0, target_node='gpu-g7-1'))
        with self.assertRaises(session.Refusal) as caught:
            handler.collect('0')
        self.assertIn('symlink', str(caught.exception).lower())
        self.assertEqual('original\n', (target / 'keep').read_text())

    def test_collect_output_file_symlink_is_refused(self):
        """A symlink planted at the output name must not be written through.

        The repaired writer never truncates an existing leaf: O_CREAT|O_EXCL
        refuses any existing final component, symlink included, and the capture
        goes to the next unused -attempt-N name instead. The victim file is
        therefore untouched either way, which is the property that matters.
        """
        victim = Path(self.tmp.name) / 'victim.txt'
        victim.write_text('untouched\n')
        handler = self.make()
        handler.store.write(session.State(phase='fault-applied', kind='gpu',
                                          started_at=1.0, target_node='gpu-g7-1'))
        planned = handler.collect_output_path('0', phase='fault', round=1)
        planned.parent.mkdir(parents=True, exist_ok=True)
        planned.symlink_to(victim)
        result = self.make().collect('0')
        self.assertEqual('untouched\n', victim.read_text())
        self.assertNotEqual(planned, result.saved_path)
        self.assertFalse(result.saved_path.is_symlink())
        self.assertIn('attempt-2', result.saved_path.name)

    def test_state_file_is_not_writable_by_the_participant(self):
        handler = self.make()
        handler.start('gpu')
        mode = stat.S_IMODE(handler.store.path.lstat().st_mode)
        self.assertEqual(0o600, mode)

    def test_forged_state_phase_is_refused(self):
        handler = self.make()
        handler.start('gpu')
        handler.store.path.write_text(json.dumps({'phase': 'verified', 'kind': 'gpu'}))
        with self.assertRaises(session.Refusal) as caught:
            self.make().status()
        self.assertIn('state record', str(caught.exception).lower())

    def test_environment_is_not_inherited_into_privileged_calls(self):
        env = session.safe_environment('/opt/aws/pcs/scheduler/slurm-25.05/bin')
        self.assertEqual(
            '/opt/aws/pcs/scheduler/slurm-25.05/bin:/usr/local/sbin:/usr/local/bin:'
            '/usr/sbin:/usr/bin:/sbin:/bin', env['PATH'])
        for leaked in ['LD_PRELOAD', 'LD_LIBRARY_PATH', 'PYTHONPATH', 'BASH_ENV',
                       'IFS', 'SSH_ORIGINAL_COMMAND']:
            self.assertNotIn(leaked, env)


# --------------------------------------------------------------------------
# Success paths.
# --------------------------------------------------------------------------
class SuccessPaths(Base):
    def test_gpu_start_drains_with_this_exercise_reason_then_injects(self):
        handler = self.make()
        result = handler.start('gpu')
        self.assertEqual('fault-applied', result.state.phase)
        drain = [c for c in self.executor.slurm_calls if c[1:2] == ['update']]
        self.assertEqual(1, len(drain))
        self.assertIn('NodeName=gpu-g7-1', drain[0])
        self.assertIn('State=DRAIN', drain[0])
        self.assertIn('Reason=aim344-device-recovery', drain[0])
        order = [c[0] for c in self.executor.maintenance_calls]
        self.assertLess(order.index('inspect'), order.index('gpu-remove'))

    def test_gpu_start_records_and_pauses_state_before_removal(self):
        """device-fault.sh refuses removal while persistence is enabled, so the
        prepare step must run before the mutation, not after."""
        handler = self.make()
        handler.start('gpu')
        order = [c[0] for c in self.executor.maintenance_calls]
        self.assertIn('gpu-prepare', order)
        self.assertLess(order.index('gpu-prepare'), order.index('gpu-remove'))

    def test_efa_start_does_not_touch_gpu_persistence(self):
        handler = self.make()
        handler.start('efa')
        order = [c[0] for c in self.executor.maintenance_calls]
        self.assertNotIn('gpu-prepare', order)

    def test_gpu_recovery_restores_the_recorded_state_before_resume(self):
        handler = self.make()
        handler.start('gpu')
        self.make().recover()
        order = [c[0] for c in self.executor.maintenance_calls]
        self.assertIn('gpu-restore', order)
        resume_at = [i for i, c in enumerate(self.executor.slurm_calls)
                     if 'State=RESUME' in c]
        self.assertTrue(resume_at)
        # gpu-restore is a maintenance call and RESUME a scheduler call, so
        # assert the restore happened and that RESUME came after the reboot.
        self.assertLess(order.index('restore-runtime'), order.index('gpu-restore'))

    def test_a_failed_gpu_restore_keeps_the_node_out_of_service(self):
        """Superseded by review finding F7: a failed restore is no longer a note.

        The original version of this test asserted that the failure was recorded
        in notes and the node resumed anyway. That is exactly the fail-open
        behaviour the review rejected, so the assertion is now that the node is
        kept out of service with the reason preserved.
        """
        handler = self.make()
        handler.start('gpu')
        self.executor.maintenance_failures['gpu-restore'] = (
            1, 'Persistence is Disabled, expected the recorded Enabled.\n')
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        self.assertIn('not fully restored', str(caught.exception))
        state = self.make().store.read()
        self.assertEqual('recovery-failed', state.phase)
        self.assertIn('expected the recorded Enabled', state.notes)
        self.assertEqual([], [c for c in self.executor.slurm_calls
                              if 'State=RESUME' in c])

    def test_status_is_readable_after_a_new_connection(self):
        self.make().start('gpu')
        reconnected = self.make().status()
        self.assertEqual('fault-applied', reconnected.state.phase)
        self.assertEqual('gpu', reconnected.state.kind)
        self.assertEqual('IDLE+DRAIN', reconnected.node_state)
        self.assertIn('aim344-device-recovery', reconnected.node_reason)

    def test_status_before_any_start_reports_ready(self):
        result = self.make().status()
        self.assertEqual('ready', result.state.phase)

    def test_collect_returns_raw_output_and_saves_it(self):
        handler = self.make()
        handler.start('gpu')
        result = self.make().collect('3')
        self.assertEqual(self.executor.collect_output, result.output)
        self.assertTrue(result.saved_path.exists())
        self.assertEqual(self.executor.collect_output, result.saved_path.read_text())
        call = [c for c in self.executor.maintenance_calls if c[0] == 'collect'][-1]
        self.assertEqual(['collect', '3'], call)

    def test_collect_moves_the_phase_to_investigating(self):
        handler = self.make()
        handler.start('gpu')
        self.make().collect('0')
        self.assertEqual('investigating', self.make().status().state.phase)

    def test_gpu_recover_uses_basic_scontrol_reboot_and_keeps_the_drain(self):
        handler = self.make()
        handler.start('gpu')
        result = self.make().recover()
        reboot = [c for c in self.executor.slurm_calls if c[1:2] == ['reboot']]
        self.assertEqual(1, len(reboot))
        self.assertIn('reason=aim344-device-recovery', reboot[0])
        flat = ' '.join(reboot[0])
        self.assertNotIn('nextstate', flat.lower())
        self.assertNotIn('asap', flat.lower())
        resume_index = [i for i, c in enumerate(self.executor.slurm_calls)
                        if 'State=RESUME' in c]
        reboot_index = [i for i, c in enumerate(self.executor.slurm_calls)
                        if c[1:2] == ['reboot']]
        self.assertTrue(resume_index and reboot_index)
        self.assertGreater(resume_index[0], reboot_index[0])
        self.assertEqual('runtime-ready', result.state.phase)

    def test_gpu_recover_confirms_a_changed_boot_id_on_the_same_instance(self):
        handler = self.make()
        handler.start('gpu')
        before = handler.store.read().boot_id
        result = self.make().recover()
        self.assertNotEqual(before, result.state.boot_id)
        self.assertEqual('i-0123456789abcdef0', result.state.target_instance_id)

    def test_gpu_recover_restores_runtime_before_resume(self):
        handler = self.make()
        handler.start('gpu')
        self.make().recover()
        actions = [c[0] for c in self.executor.maintenance_calls]
        self.assertIn('restore-runtime', actions)
        restore_at = actions.index('restore-runtime')
        checks_at = [i for i, c in enumerate(self.executor.maintenance_calls)
                     if c[0] == 'collect']
        self.assertTrue(checks_at)
        self.assertLess(restore_at, max(checks_at))

    def test_efa_recover_tries_rebind_before_any_reboot(self):
        handler = self.make()
        handler.start('efa')
        self.make().recover()
        actions = [c[0] for c in self.executor.maintenance_calls]
        self.assertIn('efa-rebind', actions)
        self.assertEqual([], [c for c in self.executor.slurm_calls if c[1:2] == ['reboot']])

    def test_efa_recover_requires_replacement_when_rebind_fails(self):
        """Final review finding 2 supersedes F8's unsafe fallback acceptance.

        No reboot-affected originals were captured, so even a device-level failure
        must keep isolation rather than reboot. Baseline evidence: controller.md.
        """
        handler = self.make()
        handler.start('efa')
        self.executor.maintenance_failures['efa-rebind'] = (
            1, 'write error: No such device\n')
        with self.assertRaises(session.Refusal):
            self.make().recover()
        self.assertEqual([], [c for c in self.executor.slurm_calls if c[1:2] == ['reboot']])
        self.assertEqual([], [c for c in self.executor.slurm_calls if 'State=RESUME' in c])
        state = self.make().store.read()
        self.assertEqual('replacement-required', state.phase)
        self.assertIn('No such device', state.notes)
        self.assertNotIn('restore-runtime', [c[0] for c in self.executor.maintenance_calls])

    def test_an_identity_refusal_during_rebind_does_not_reboot(self):
        """The complement of the test above, per review finding F8."""
        handler = self.make()
        handler.start('efa')
        self.executor.maintenance_failures['efa-rebind'] = (
            1, 'Unexpected driver binding.\n')
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        self.assertIn('stopped before changing anything', str(caught.exception))
        self.assertEqual([], [c for c in self.executor.slurm_calls
                              if c[1:2] == ['reboot']])
        self.assertEqual([], [c for c in self.executor.slurm_calls
                              if 'State=RESUME' in c])

    def test_runtime_ready_is_not_workload_verified(self):
        handler = self.make()
        handler.start('gpu')
        result = self.make().recover()
        self.assertEqual('runtime-ready', result.state.phase)
        self.assertNotEqual('verified', result.state.phase)
        self.assertIn('fresh', result.next_step.lower())

    def test_deadline_recovery_is_recorded_separately(self):
        handler = self.make(now=1000.0)
        handler.start('gpu')
        expired = self.make(now=1000.0 + 1801)
        result = expired.expire()
        self.assertTrue(result.expired)
        self.assertEqual('deadline', result.state.recovered_by)
        self.assertEqual('runtime-ready', result.state.phase)

    def test_participant_recovery_is_recorded_as_participant(self):
        handler = self.make()
        handler.start('gpu')
        result = self.make().recover()
        self.assertEqual('participant', result.state.recovered_by)

    def test_expire_does_nothing_before_the_deadline(self):
        handler = self.make(now=1000.0)
        handler.start('gpu')
        result = self.make(now=1000.0 + 10).expire()
        self.assertFalse(result.expired)
        self.assertEqual([], [c for c in self.executor.slurm_calls if c[1:2] == ['reboot']])

    def test_efa_start_passes_the_participants_own_job_identifier(self):
        """The --job argument, on the path that carries it.

        The participant route refuses an active EFA start on this stack for a
        measured reason (DeviceSession.ACTIVE_EFA_PARTICIPANT_PATH_ADOPTED and its
        comment: the unbind write does not return until the holding workload ends).
        The argument plumbing still has to be right for a facilitator-driven active
        trial and for the adoption decision, so this test enables that path
        explicitly rather than asserting a behaviour the shipped route refuses.
        """
        self.executor.jobs = ['95|aim344-t1|aim344-device|RUNNING']
        handler = self.make()
        handler.ACTIVE_EFA_PARTICIPANT_PATH_ADOPTED = True
        handler.start('efa')
        call = [c for c in self.executor.maintenance_calls if c[0] == 'efa-unbind'][0]
        self.assertIn('--job', call)
        self.assertEqual('95', call[call.index('--job') + 1])

    def test_efa_start_refuses_an_active_round_from_the_participant_route(self):
        """And the shipped default is the refusal, not the injection."""
        self.executor.jobs = ['95|aim344-t1|aim344-device|RUNNING']
        with self.assertRaises(session.Refusal):
            self.make().start('efa')
        self.assertEqual([], [c for c in self.executor.maintenance_calls
                              if c[0] == 'efa-unbind'])

    def test_efa_start_without_a_job_is_an_idle_unbind(self):
        handler = self.make()
        handler.start('efa')
        call = [c for c in self.executor.maintenance_calls if c[0] == 'efa-unbind'][0]
        self.assertNotIn('--job', call)

    def test_accepted_request_grammar(self):
        for text, verb, argument in [('start gpu', 'start', 'gpu'),
                                     ('start efa', 'start', 'efa'),
                                     ('status', 'status', None),
                                     ('collect 0', 'collect', '0'),
                                     ('collect 6', 'collect', '6'),
                                     ('recover', 'recover', None)]:
            with self.subTest(text=text):
                parsed = session.parse_participant_request(text)
                self.assertIsNone(parsed.error, parsed.error)
                self.assertEqual(verb, parsed.verb)
                self.assertEqual(argument, parsed.argument)

    def test_empty_request_reports_usage(self):
        parsed = session.parse_participant_request('')
        self.assertIsNotNone(parsed.error)
        self.assertIn('start', parsed.error)

    def test_a_dropped_ssh_original_command_is_reported_not_defaulted(self):
        """sudo's env_reset strips SSH_ORIGINAL_COMMAND unless it is kept. If it
        is lost the helper must say so rather than silently doing nothing, and it
        must never fall back to a default action."""
        for lost in (None, '', '   '):
            with self.subTest(lost=lost):
                parsed = session.parse_participant_request(lost)
                self.assertIsNotNone(parsed.error)
                self.assertIsNone(parsed.verb)

    def test_answers_directory_is_handed_to_the_participant(self):
        """The helper runs as root, so a file it creates must still be owned by
        the participant or their results are unreadable to them."""
        handler = self.make()
        handler.start('gpu')
        result = self.make().collect('0')
        self.assertTrue(result.saved_path.exists())
        mode = stat.S_IMODE(result.saved_path.lstat().st_mode)
        self.assertEqual(0o600, mode)

    def test_gpu_recovery_remounts_staging_before_restoring_runtime(self):
        """Measured on hardware: a reboot leaves the staging bind mount gone, so
        restore-runtime.sh fails with 'Missing staged image'. The remount must
        happen first."""
        handler = self.make()
        handler.start('gpu')
        self.make().recover()
        order = [c[0] for c in self.executor.maintenance_calls]
        self.assertIn('remount-staging', order)
        self.assertLess(order.index('remount-staging'), order.index('restore-runtime'))

    def test_efa_rebind_recovery_does_not_remount_staging(self):
        """No reboot happened, so the mount was never lost."""
        handler = self.make()
        handler.start('efa')
        self.make().recover()
        order = [c[0] for c in self.executor.maintenance_calls]
        self.assertNotIn('remount-staging', order)

    def test_a_failed_recovery_stays_retryable(self):
        """A recovery that stops part way must not strand the participant at
        'recovering', which would refuse every later attempt."""
        handler = self.make()
        handler.start('gpu')
        self.executor.maintenance_failures['restore-runtime'] = (
            1, 'Missing staged image: /opt/aim344/aim344.sqsh\n')
        with self.assertRaises(session.Refusal):
            self.make().recover()
        self.assertEqual('investigating', self.make().store.read().phase)
        # With the fault cleared the retry succeeds, without a facilitator.
        del self.executor.maintenance_failures['restore-runtime']
        result = self.make().recover()
        self.assertEqual('runtime-ready', result.state.phase)

    def test_an_unreachable_node_is_explained_not_shown_as_a_transport_error(self):
        """Observed on hardware: while the node restarts the route returns
        'Connection closed by remote host'. The participant should be told the
        node is restarting and that it stays reserved."""
        handler = self.make()
        handler.start('gpu')
        self.executor.maintenance_failures['restore-runtime'] = (
            255, 'Connection to 192.0.2.10 closed by remote host.\n')
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        message = str(caught.exception)
        self.assertIn('restarts', message)
        self.assertIn('reserved', message)
        self.assertNotIn('192.0.2.10', message)
        self.assertEqual('investigating', self.make().store.read().phase)

    def test_a_finished_round_can_be_followed_by_the_next_round(self):
        """Consecutive cycles are an acceptance requirement, so a completed round
        must not block the next start."""
        handler = self.make()
        handler.start('gpu')
        self.make().recover()
        self.assertEqual('runtime-ready', self.make().status().state.phase)
        first = [c[0] for c in self.executor.maintenance_calls].count('gpu-remove')
        result = self.make().start('gpu')
        self.assertFalse(result.already_started)
        self.assertEqual('fault-applied', result.state.phase)
        self.assertEqual(first + 1,
                         [c[0] for c in self.executor.maintenance_calls].count('gpu-remove'))
        # The new round starts with its own evidence, not the previous round's.
        self.assertEqual([], result.state.collected)
        self.assertIsNone(result.state.recovered_by)

    def test_the_other_round_can_follow_a_finished_round(self):
        handler = self.make()
        handler.start('gpu')
        self.make().recover()
        result = self.make().start('efa')
        self.assertEqual('efa', result.state.kind)
        self.assertEqual('fault-applied', result.state.phase)

    def test_a_round_still_in_progress_blocks_the_other_kind(self):
        self.make().start('gpu')
        with self.assertRaises(session.Refusal):
            self.make().start('efa')

    def test_another_table_on_the_same_node_is_refused_explicitly(self):
        """Observed on hardware: a second assignment pointing at an already
        faulted node was stopped only incidentally, by the device helper's
        'Provisioned gpu PCI function is absent: 0000:ba:00.0'. Ownership must be
        refused up front, without leaking a device identity."""
        self.config_body['assignments']['table-2'].update({
            'target_node': 'gpu-g7-1',
            'target_instance_id': 'i-0123456789abcdef0',
            'participant_uid': os.getuid(),
            'caller_uid': os.getuid(),
            'answers_dir': str(Path(self.tmp.name) / 'answers-2'),
        })
        self.write_config()
        self.make(assignment='table-1').start('gpu')
        before = len(self.executor.maintenance_calls)
        with self.assertRaises(session.Refusal) as caught:
            self.make(assignment='table-2').start('gpu')
        message = str(caught.exception)
        self.assertIn('Another table', message)
        self.assertNotIn('0000:', message)
        self.assertNotIn('gpu-g7-1', message)
        # Nothing was mutated on behalf of the second table.
        self.assertEqual(before, len(self.executor.maintenance_calls))

    def test_a_free_node_is_not_reported_as_held(self):
        """The ownership check must not block the only table using the node."""
        self.config_body['assignments']['table-2'].update({
            'target_node': 'gpu-g7-1',
            'target_instance_id': 'i-0123456789abcdef0',
            'participant_uid': os.getuid(),
            'caller_uid': os.getuid(),
            'answers_dir': str(Path(self.tmp.name) / 'answers-2'),
        })
        self.write_config()
        result = self.make(assignment='table-2').start('gpu')
        self.assertEqual('fault-applied', result.state.phase)

    def test_a_recovered_table_releases_the_node(self):
        self.config_body['assignments']['table-2'].update({
            'target_node': 'gpu-g7-1',
            'target_instance_id': 'i-0123456789abcdef0',
            'participant_uid': os.getuid(),
            'caller_uid': os.getuid(),
            'answers_dir': str(Path(self.tmp.name) / 'answers-2'),
        })
        self.write_config()
        self.make(assignment='table-1').start('gpu')
        self.make(assignment='table-1').recover()
        result = self.make(assignment='table-2').start('gpu')
        self.assertEqual('fault-applied', result.state.phase)


if __name__ == '__main__':
    unittest.main()
