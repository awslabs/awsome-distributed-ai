# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Local active-FLR authorization/streaming/recovery tests; NOT hardware proof."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import test_device_session as fixture
import test_maintenance as maintenance_fixture
import test_safety_rework3 as safety_fixture
import test_replacement as replacement_fixture

session = fixture.session
maintenance = maintenance_fixture.maintenance
LAB = Path(__file__).resolve().parents[1]


class ActiveSession(fixture.Base):
    def setUp(self):
        super().setUp()
        self.executor.jobs = ['226|aim344-t1|aim344-device|RUNNING']
        self.active_calls = []
        self.bad_identity = False
        self.rc = 0
        for kind in ('gpu', 'efa'):
            old = 'gpu-remove' if kind == 'gpu' else 'efa-unbind'
            self.executor.inspect['confirmation'][kind + '-flr'] = (
                self.executor.inspect['confirmation'][old].replace(old, kind + '-flr'))
        self.executor.run_active_fault = self.active
        self.progress_file = self.make()._answers_root() / 'device-226.log'
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        self.progress_file.write_text(self.progress(1000))

    @staticmethod
    def progress(now, job='226', sequence=40):
        return ''.join(f'AIM344 collective_progress={n} collectives; tensor_bytes=134217728 B; '
                       f'timestamp={t:.6f} s since epoch; 0 mismatches; '
                       f'job_id={job}; world_size=16; rank=0.\n'
                       for n, t in ((20, now - 2), (sequence, now - 1)))

    def active(self, argv, record):
        self.active_calls.append(argv)
        options = dict(zip(argv[1::2], argv[2::2]))
        event = {'action': 'active-flr-authorized', 'operation': options['--operation'],
                 'mutation': argv[0], 'instance_id': self.executor.inspect['instance_id'],
                 'boot_id': self.executor.boot_id, 'job_id': options['--job'],
                 'participant': options['--participant'],
                 'bdf': self.executor.inspect[argv[0].split('-')[0] + '_bdf']}
        if self.bad_identity:
            event['job_id'] = '999'
        lines = [json.dumps(event)]
        ack = record(lines[0])
        self.assertEqual('ACK ' + hashlib.sha256(lines[0].encode()).hexdigest(), ack)
        # Real StateStore readback is already populated at the ACK boundary.
        durable = self.make().store.read()
        self.assertEqual(event, durable.active_fault['events'][0])
        for action in ('mutation-start', 'mutation-returned'):
            event = {'action': action, 'operation': argv[0],
                     'instance_id': self.executor.inspect['instance_id'],
                     'boot_id': self.executor.boot_id,
                     'bdf': self.executor.inspect[argv[0].split('-')[0] + '_bdf']}
            lines.append(json.dumps(event))
            ack = record(lines[-1])
            if action == 'mutation-start':
                self.assertEqual('ACK ' + hashlib.sha256(lines[-1].encode()).hexdigest(), ack)
                self.assertEqual(action, self.make().store.read().active_fault['events'][-1]['action'])
        return session.Completed(self.rc, '\n'.join(lines), '')

    def test_gpu_active_is_not_idle_remove_or_pause(self):
        result = self.make().start('gpu', active=True)
        self.assertEqual('returned', result.state.active_fault['outcome'])
        self.assertEqual('gpu-flr', self.active_calls[0][0])
        calls = self.executor.maintenance_calls
        self.assertTrue(any(c[0] == 'gpu-prepare' and '--capture-only' in c for c in calls))
        self.assertFalse(any(c[0] in ('gpu-remove', 'collect') for c in calls))
        self.assertTrue(self.make().start('gpu', active=True).already_started)
        self.assertEqual(1, len(self.active_calls))

    def test_efa_active_reboots_and_restores_originals_not_rebind(self):
        handler = self.make()
        handler.start('efa', active=True)
        self.executor.jobs = []
        result = handler.recover()
        self.assertEqual('runtime-ready', result.state.phase)
        self.assertTrue(any(c[1] == 'reboot' for c in self.executor.slurm_calls))
        self.assertTrue(any(c[0] == 'gpu-restore' for c in self.executor.maintenance_calls))
        self.assertFalse(any(c[0] == 'efa-rebind' for c in self.executor.maintenance_calls))
        self.assertNotIn('EFA path: idle', session.render(handler.status(), 'status'))

    def test_wrong_authorization_cannot_be_acknowledged(self):
        self.bad_identity = True
        with self.assertRaisesRegex(session.Refusal, 'differs'):
            self.make().start('gpu', active=True)
        self.assertEqual([], self.make().store.read().active_fault['events'])

    def test_missing_running_job_refused_before_drain(self):
        for jobs in ([], ['226|aim344-t1|aim344-device|PENDING'],
                     ['226|aim344-t2|aim344-device|RUNNING'],
                     ['226|aim344-t1|aim344-device|RUNNING', '227|aim344-t1|aim344-device|RUNNING'],
                     ['malformed']):
            with self.subTest(jobs=jobs):
                self.executor.jobs = jobs
                with self.assertRaises(session.Refusal):
                    self.make().start('gpu', active=True)
                self.assertEqual([], self.active_calls)
                self.assertFalse(any('State=DRAIN' in c for c in self.executor.slurm_calls))

    def test_timeout_never_means_workload_exit_or_repeat(self):
        self.rc = 124
        with self.assertRaisesRegex(session.Refusal, 'Do not repeat'):
            self.make().start('gpu', active=True)
        state = self.make().store.read()
        self.assertEqual('unknown-or-refused', state.active_fault['outcome'])
        self.assertEqual('fault-applied', state.phase)
        self.make().start('gpu', active=True)
        self.assertEqual(1, len(self.active_calls))

    def assert_transport_failure_retained(self, expected_actions):
        handler = self.make()
        with self.assertRaisesRegex(session.Refusal, 'status, collect evidence, then recover'):
            handler.start('gpu', active=True)
        state = handler.store.read()
        self.assertEqual('fault-applied', state.phase)
        self.assertEqual('unknown-or-refused', state.active_fault['outcome'])
        self.assertIn('gpu-flr-attempted', state.prepared)
        self.assertEqual(expected_actions, [event['action'] for event in state.active_fault['events']])
        self.assertNotIn('not-issued', session.render(handler.status(), 'status'))
        self.assertTrue(handler.start('gpu', active=True).already_started)
        with self.assertRaises(session.Refusal):
            handler.start('efa', active=True)
        self.assertEqual(1, len(self.active_calls))
        self.executor.jobs = ['227|aim344-t1|aim344-device|COMPLETING']
        with self.assertRaisesRegex(session.Refusal, 'different job'):
            handler.recover()
        self.executor.jobs = []
        self.assertEqual('runtime-ready', handler.recover().state.phase)

    def test_eof_wait_exception_after_markers_keeps_unknown_and_recover_route(self):
        def interrupted(argv, record):
            self.active(argv, record)
            raise subprocess.TimeoutExpired('ssh', 90)
        self.executor.run_active_fault = interrupted
        self.assert_transport_failure_retained(
            ['active-flr-authorized', 'mutation-start', 'mutation-returned'])

    def test_ack_write_exception_after_start_keeps_unknown_and_recover_route(self):
        def interrupted(argv, record):
            def ack(line):
                result = record(line)
                if json.loads(line)['action'] == 'mutation-start':
                    durable = self.make().store.read()
                    self.assertEqual('unknown-or-refused', durable.active_fault['outcome'])
                    raise BrokenPipeError('ACK pipe closed after durable marker')
                return result
            return self.active(argv, ack)
        self.executor.run_active_fault = interrupted
        self.assert_transport_failure_retained(['active-flr-authorized', 'mutation-start'])

    def test_observation_exception_without_markers_is_still_unknown(self):
        def interrupted(argv, record):
            self.active_calls.append(argv)
            raise OSError('read failed after dispatch')
        self.executor.run_active_fault = interrupted
        self.assert_transport_failure_retained([])

    def test_marker_fsync_failure_never_returns_ack_and_retains_unknown(self):
        handler = self.make()
        real_write = handler.store.write
        failed, acknowledgments = [], []
        def write(state):
            events = (state.active_fault or {}).get('events', [])
            if events and events[-1]['action'] == 'mutation-start' and not failed:
                failed.append(True)
                raise OSError('marker fsync failed')
            return real_write(state)
        def active(argv, record):
            def ack(line):
                result = record(line)
                if result:
                    acknowledgments.append(json.loads(line)['action'])
                return result
            return self.active(argv, ack)
        self.executor.run_active_fault = active
        with mock.patch.object(handler.store, 'write', side_effect=write):
            with self.assertRaisesRegex(session.Refusal, 'Do not repeat'):
                handler.start('gpu', active=True)
        self.assertEqual(['active-flr-authorized'], acknowledgments)
        self.assertEqual([True], failed)
        self.assertEqual('unknown-or-refused', handler.store.read().active_fault['outcome'])
        self.assertEqual('fault-applied', handler.store.read().phase)
        self.assertTrue(handler.start('gpu', active=True).already_started)
        self.assertEqual(1, len(self.active_calls))

    def test_local_spawn_failure_is_not_issued_and_not_retried(self):
        real_executor = session.Executor('/unused', 'unused')
        self.executor.run_active_fault = real_executor.run_active_fault
        with mock.patch.object(session.subprocess, 'Popen', side_effect=OSError('exec failed')) as spawn:
            with self.assertRaisesRegex(session.Refusal, 'Do not repeat'):
                self.make().start('gpu', active=True)
            state = self.make().store.read()
            self.assertEqual('not-issued', state.active_fault['outcome'])
            self.assertEqual([], state.active_fault['events'])
            self.assertEqual('fault-applied', state.phase)
            self.assertTrue(self.make().start('gpu', active=True).already_started)
            spawn.assert_called_once()

    def test_running_job_prevents_reboot_and_foreign_job_is_not_cleanup(self):
        handler = self.make()
        handler.start('gpu', active=True)
        with self.assertRaisesRegex(session.Refusal, 'still active'):
            handler.recover()
        self.executor.jobs = ['227|aim344-t1|aim344-device|COMPLETING']
        with self.assertRaisesRegex(session.Refusal, 'different job'):
            handler.recover()
        self.assertFalse(any(c[1] == 'reboot' for c in self.executor.slurm_calls))

    def test_exact_completing_job_never_allows_resume(self):
        handler = self.make()
        state = handler.start('gpu', active=True).state
        self.executor.jobs = ['226|aim344-t1|aim344-device|COMPLETING']
        handler._require_recoverable_target(state)
        with self.assertRaisesRegex(session.Refusal, 'cleanup'):
            handler._require_current_recovery_observations(state.boot_id, 'RESUME')

    def test_cross_kind_mutation_journal_cannot_authorize_recovery(self):
        handler = self.make()
        state = handler.start('gpu', active=True).state
        state.prepared.append('efa-flr-attempted')
        with self.assertRaisesRegex(session.Refusal, 'inconsistent'):
            handler._require_coherent_restoration_obligations(state)

    @contextlib.contextmanager
    def ec2_fallback(self, result=None, error=None, post_read_error=False):
        """Real cloud wrapper/Executor, mocked subprocess only; no AWS calls."""
        self.progress_file.write_text(self.progress(2000))
        handler = self.make(now=2000)
        state = handler.start('gpu', active=True).state
        self.executor.jobs = ['226|aim344-t1|aim344-device|COMPLETING']
        state.reboot_requests, state.reboot_requested_at = 1, 1000
        handler.assignment['target_host'] = '10.0.0.1'
        handler.assignment['replacement'] = {'enabled': True, 'region': 'eu-south-2'}
        handler.store.write(state)
        node = {'InstanceId': state.target_instance_id, 'NodeAddr': '10.0.0.1'}
        real_executor = session.Executor('/unused', '10.0.0.1')
        setattr(self.executor, 'run_cloud', real_executor.run_cloud)
        result = result or subprocess.CompletedProcess([], 0, '', '')

        def run(argv, **kwargs):
            durable = handler.store.read()
            self.assertTrue(durable.active_fault['external_reboot_attempted'])
            self.assertNotIn('external_reboot_requested_at', durable.active_fault)
            self.assertEqual(['/usr/local/bin/aws', 'ec2', 'reboot-instances'], argv[:3])
            self.assertEqual({'InstanceIds': [state.target_instance_id]},
                             json.loads(argv[argv.index('--cli-input-json') + 1]))
            self.assertEqual('json', argv[argv.index('--output') + 1])
            if error:
                raise error
            return result

        instance = {'State': {'Name': 'running'}}
        with mock.patch.object(handler, '_replacement_node', return_value=node), \
             mock.patch.object(handler, '_replacement_capacity'), \
             mock.patch.object(handler, '_replacement_instance',
                               side_effect=[instance, session.Refusal('post-dispatch read failed')]
                               if post_read_error else None, return_value=instance), \
             mock.patch.object(session.subprocess, 'run', side_effect=run) as cli:
            yield handler, state, cli

    def assert_no_ec2_recovery_claim(self, handler):
        state = handler.store.read()
        self.assertFalse(state.reboot_completed)
        self.assertIsNone(state.boot_id_after_reboot)
        self.assertNotIn('reboot', state.prepared)
        self.assertFalse(any('State=RESUME' in c for c in self.executor.slurm_calls))
        self.assertFalse(any(c[0] in ('remount-staging', 'restore-runtime', 'gpu-restore')
                             for c in self.executor.maintenance_calls))

    def test_ec2_empty_cli_ack_persists_request_not_completion_or_duplicate(self):
        with self.ec2_fallback() as (handler, state, cli):
            with self.assertRaisesRegex(session.Refusal, 'completion is not established'):
                handler._active_reboot_escalation(state)
            durable = handler.store.read()
            self.assertTrue(durable.active_fault['external_reboot_requested'])
            self.assertEqual(2000, durable.active_fault['external_reboot_requested_at'])
            self.assert_no_ec2_recovery_claim(handler)
            handler.clock = lambda: 2050
            with self.assertRaisesRegex(session.Refusal, 'observation window'):
                handler.recover()
            handler.clock = lambda: 2200
            with self.assertRaisesRegex(session.Refusal, 'Replacement is required'):
                handler.recover()
            self.assertEqual('replacement-required', handler.store.read().phase)
            self.assert_no_ec2_recovery_claim(handler)
            cli.assert_called_once()

    def test_ec2_ack_survives_failed_post_dispatch_read(self):
        with self.ec2_fallback(post_read_error=True) as (handler, state, cli):
            with self.assertRaisesRegex(session.Refusal, 'post-dispatch read failed'):
                handler._active_reboot_escalation(state)
            self.assertTrue(handler.store.read().active_fault['external_reboot_requested'])
            self.assertEqual(2000, handler.store.read().active_fault['external_reboot_requested_at'])
            self.assert_no_ec2_recovery_claim(handler)
            handler.clock = lambda: 2200
            with self.assertRaisesRegex(session.Refusal, 'Replacement is required'):
                handler.recover()
            cli.assert_called_once()

    def test_ec2_lost_ack_and_invalid_response_keep_intent_without_retry(self):
        cases = [(subprocess.CompletedProcess([], 0, 'not JSON', ''), None, 'unreadable'),
                 (subprocess.CompletedProcess([], 255, '', 'AccessDenied'), None, 'did not confirm'),
                 (subprocess.CompletedProcess([], 254, '{}', 'service error'), None, 'did not confirm'),
                 (None, subprocess.TimeoutExpired('aws', 90), 'did not confirm')]
        for result, error, message in cases:
            with self.subTest(result=result, error=error):
                # Each case is a separate round; use the existing fixture setup.
                with self.ec2_fallback(result, error) as (handler, state, cli):
                    with self.assertRaisesRegex(session.Refusal, message):
                        handler._active_reboot_escalation(state)
                    durable = handler.store.read()
                    self.assertTrue(durable.active_fault['external_reboot_attempted'])
                    self.assertNotIn('external_reboot_requested_at', durable.active_fault)
                    self.assertNotIn('external_reboot_requested', durable.active_fault)
                    self.assert_no_ec2_recovery_claim(handler)
                    handler.clock = lambda: 2200
                    with self.assertRaisesRegex(session.Refusal, 'Replacement is required'):
                        handler.recover()
                    cli.assert_called_once()
                # Retire only local fixture state before the next independent case.
                handler.store.path.unlink()
                self.executor.node_state = 'IDLE+CLOUD'
                self.executor.node_reason = ''
                self.executor.jobs = ['226|aim344-t1|aim344-device|RUNNING']

    def test_ec2_ack_followup_requires_fresh_boot_and_retired_scheduler_request(self):
        with self.ec2_fallback() as (handler, state, cli):
            with self.assertRaisesRegex(session.Refusal, 'completion is not established'):
                handler._active_reboot_escalation(state)
            self.executor.jobs = []
            self.executor.boot_id = 'bbbbbbbb-0000-0000-0000-000000000002'
            self.executor.node_state = 'IDLE+DRAIN+REBOOT_REQUESTED'
            with self.assertRaisesRegex(session.Refusal, 'scheduler still holds'):
                handler.recover()
            self.assert_no_ec2_recovery_claim(handler)
            self.executor.node_state = 'IDLE+DRAIN'
            recovered = handler.recover().state
            self.assertEqual('runtime-ready', recovered.phase)
            self.assertTrue(recovered.reboot_completed)
            self.assertEqual(state.boot_id, recovered.boot_id_before_reboot)
            self.assertEqual(self.executor.boot_id, recovered.boot_id_after_reboot)
            self.assertTrue(recovered.active_fault['external_reboot_requested'])
            self.assertFalse(any(c[1] == 'reboot' for c in self.executor.slurm_calls))
            cli.assert_called_once()

    def test_ec2_ack_unreadable_boot_never_restores_or_reboots(self):
        with self.ec2_fallback() as (handler, state, cli):
            with self.assertRaisesRegex(session.Refusal, 'completion is not established'):
                handler._active_reboot_escalation(state)
            self.executor.jobs = []
            self.executor.maintenance_failures['boot-id'] = (255, 'unreachable')
            with self.assertRaisesRegex(session.Refusal, 'could not establish'):
                handler.recover()
            self.assert_no_ec2_recovery_claim(handler)
            cli.assert_called_once()

    def test_ec2_escalation_is_exact_id_once_then_replace(self):
        self.progress_file.write_text(self.progress(2000))
        handler = self.make(now=2000)
        state = handler.start('gpu', active=True).state
        self.executor.jobs = ['226|aim344-t1|aim344-device|COMPLETING']
        state.reboot_requests, state.reboot_requested_at = 1, 1000
        handler.assignment['target_host'] = '10.0.0.1'
        handler.store.write(state)
        node = {'InstanceId': state.target_instance_id, 'NodeAddr': '10.0.0.1'}
        with mock.patch.object(handler, '_replacement_node', return_value=node), \
             mock.patch.object(handler, '_replacement_capacity'), \
             mock.patch.object(handler, '_replacement_instance', return_value={'State': {'Name': 'running'}}), \
             mock.patch.object(handler, '_cloud', return_value={}) as cloud:
            with self.assertRaisesRegex(session.Refusal, 'completion is not established'):
                handler._active_reboot_escalation(state)
            cloud.assert_called_once_with('ec2', 'reboot-instances', {'InstanceIds': [state.target_instance_id]})
            self.assertTrue(handler.store.read().active_fault['external_reboot_attempted'])
            handler.clock = lambda: 2200
            with self.assertRaisesRegex(session.Refusal, 'Replacement is required'):
                handler._active_reboot_escalation(state)
            self.assertEqual(1, cloud.call_count)
        handler._replacement_idle(allow_cleanup=True)  # pre-retirement only
        with self.assertRaises(session.Refusal):
            handler._replacement_idle()  # admission must remain strictly empty
        self.executor.jobs = ['227|aim344-t1|aim344-device|COMPLETING']
        with self.assertRaises(session.Refusal):
            handler._replacement_idle(allow_cleanup=True)


class CloudResponse(fixture.Base):
    def test_only_no_output_api_accepts_empty_zero_exit(self):
        handler = self.make()
        handler.assignment['replacement'] = {'enabled': True, 'region': 'eu-south-2'}
        for body in ('', ' \n\t'):
            with self.subTest(body=body), mock.patch.object(
                    self.executor, 'run_cloud', create=True,
                    return_value=session.Completed(0, body, '')):
                self.assertEqual({}, handler._cloud('ec2', 'reboot-instances', {}))
                for service, action in (('ec2', 'describe-instances'),
                                        ('ec2', 'terminate-instances'),
                                        ('pcs', 'get-cluster'),
                                        ('pcs', 'get-compute-node-group'),
                                        ('ssm', 'send-command'),
                                        ('ssm', 'get-command-invocation')):
                    with self.subTest(action=action), self.assertRaisesRegex(
                            session.Refusal, 'unreadable'):
                        handler._cloud(service, action, {})

    def test_nonempty_json_and_errors_keep_existing_contract(self):
        handler = self.make()
        handler.assignment['replacement'] = {'enabled': True, 'region': 'eu-south-2'}
        for body, expected in (('{}', {}), ('{"Reservations": []}', {'Reservations': []})):
            with mock.patch.object(self.executor, 'run_cloud', create=True,
                                   return_value=session.Completed(0, body, '')):
                self.assertEqual(expected, handler._cloud('ec2', 'describe-instances', {}))
        for rc, body in ((0, 'not JSON'), (0, '{'), (1, ''), (124, ''), (255, '{}')):
            with self.subTest(rc=rc, body=body), mock.patch.object(
                    self.executor, 'run_cloud', create=True,
                    return_value=session.Completed(rc, body, 'fixture error')):
                with self.assertRaises(session.Refusal):
                    handler._cloud('ec2', 'reboot-instances', {})


class ProgressAndKernel(fixture.Base):
    progress_file: Path
    def setUp(self):
        bench = ActiveSession()
        bench.setUp()
        self.__dict__.update(bench.__dict__)
    progress = staticmethod(ActiveSession.progress)
    def test_wrong_job_stale_stopped_and_symlink_progress_no_ack(self):
        handler = self.make()
        state = session.State(job_id='226')
        for text in ('', self.progress(900), self.progress(1000, job='227'),
                     self.progress(1000, sequence=20)):
            self.progress_file.write_text(text)
            with self.assertRaises(session.Refusal):
                handler._active_progress(state)
        self.progress_file.unlink()
        self.progress_file.symlink_to('/does-not-exist')
        with self.assertRaises(session.Refusal):
            handler._active_progress(state)

    def test_progress_failure_before_ack_keeps_event_empty(self):
        self.progress_file.write_text(self.progress(1000, job='227'))
        with self.assertRaisesRegex(session.Refusal, 'different job'):
            self.make().start('gpu', active=True)
        self.assertEqual([], self.make().store.read().active_fault['events'])

    def test_unavailable_kernel_saved_and_not_required_for_reboot(self):
        handler = self.make()
        handler.start('gpu', active=True)
        self.executor.maintenance_failures['kernel-evidence'] = (255, 'fixture unreachable')
        result = handler.collect('kernel')
        self.assertIn('255', result.output)
        self.assertTrue(Path(result.saved_path).is_file())
        self.executor.jobs = []
        self.assertEqual('runtime-ready', handler.recover().state.phase)
        self.assertTrue(Path(result.saved_path).is_file())

    def test_journal_error_payload_preserved_across_collect_and_recover(self):
        handler = self.make()
        state = handler.start('gpu', active=True).state
        payload = json.dumps({'action': 'kernel-evidence', 'boot_id': state.boot_id,
                              'returncode': 1, 'complete_journal': False,
                              'kernel_text': 'fixture journal permission denied\n'}) + '\n'
        original = self.executor.run_maintenance

        def run_maintenance(argv, timeout=None):
            if argv[0] == 'kernel-evidence':
                self.assertEqual(['kernel-evidence', '--boot', state.boot_id], argv)
                self.assertEqual(15, timeout)
                return session.Completed(0, payload, '')
            return original(argv, timeout)

        with mock.patch.object(self.executor, 'run_maintenance', side_effect=run_maintenance):
            first = handler.collect('kernel')
            saved = Path(first.saved_path)
            before = saved.read_bytes(), saved.stat().st_mtime_ns
            second = handler.collect('kernel')
            self.assertNotEqual(first.saved_path, second.saved_path)
            self.assertIn('attempt-2', str(second.saved_path))
            self.assertIn('returncode=0; not full capture', first.output)
            self.assertIn(payload, first.output)
            self.executor.jobs = []
            self.assertEqual('runtime-ready', handler.recover().state.phase)
            self.assertEqual(before, (saved.read_bytes(), saved.stat().st_mtime_ns))
            self.assertIn(payload, Path(second.saved_path).read_text())

    def test_kernel_after_reboot_still_selects_fault_boot(self):
        handler = self.make()
        before = handler.start('gpu', active=True).state.boot_id
        self.executor.jobs = []
        handler.recover()
        handler.collect('kernel')
        self.assertEqual(['kernel-evidence', '--boot', before], self.executor.maintenance_calls[-1])

    def test_out_of_order_marker_cannot_get_ack(self):
        def bad(argv, record):
            record(json.dumps({'action': 'mutation-start'}))
            self.fail('Out-of-order marker was accepted')
        self.executor.run_active_fault = bad
        with self.assertRaisesRegex(session.Refusal, 'out of order'):
            self.make().start('gpu', active=True)

    def test_reader_timeout_is_refusal_not_authorization(self):
        import subprocess
        with mock.patch.object(session.subprocess, 'run', side_effect=subprocess.TimeoutExpired('read', 5)):
            with self.assertRaisesRegex(session.Refusal, 'no ACK'):
                self.make()._active_progress(session.State(job_id='226'))


class CleanupReplacement(fixture.Base):
    executor: replacement_fixture.ReplacementExecutor
    def setUp(self):
        bench = replacement_fixture.Replacement()
        bench.setUp()
        self.__dict__.update(bench.__dict__)
    retired = replacement_fixture.Replacement.retired
    no_roll = replacement_fixture.Replacement.no_roll

    def active_cleanup(self):
        handler = self.no_roll()
        state = handler.store.read()
        state.kind = 'gpu'
        state.phase = 'fault-applied'
        state.job_id = '226'
        state.prepared = ['drain', 'gpu-prepare-recorded', 'gpu-flr-attempted']
        state.active_fault = {'mutation': 'gpu-flr', 'outcome': 'unknown-or-refused', 'events': []}
        handler.store.write(state)
        self.executor.jobs = ['226|aim344-t1|aim344-device|COMPLETING']
        self.executor.node_state = 'ALLOCATED+COMPLETING+DRAIN+NOT_RESPONDING'
        self.executor.maintenance_failures['instance-id'] = (255, 'fixture unreachable')
        self.executor.maintenance_failures['replacement-evidence'] = (255, 'fixture unreachable')
        with self.assertRaises(session.Refusal):
            handler.recover()
        self.assertEqual('replacement-required', handler.store.read().phase)
        self.assertFalse(handler.store.read().active_fault.get('external_reboot_attempted'))
        return handler

    def test_unreachable_cg_without_reboot_receipt_retires_exact_target_and_updates_route(self):
        handler = self.active_cleanup()
        self.executor.addresses[replacement_fixture.NEW] = '10.0.0.9'
        def after(action):
            if action == 'terminate-instances':
                self.assertIn('Users=root', self.executor.reservation)
                self.executor.jobs = []
                self.executor.maintenance_failures.clear()
        self.executor.after_cloud = after
        result = handler.replace()
        self.assertEqual('runtime-ready', result.state.phase)
        self.assertEqual(replacement_fixture.NEW, result.state.target_instance_id)
        self.assertEqual('10.0.0.9', handler.assignment['target_host'])
        self.assertEqual('10.0.0.9', self.executor.target_host)
        self.assertEqual([[replacement_fixture.OLD]], self.retired())
        self.assertFalse(self.executor.reservation)
        handler.replace()
        self.assertEqual([[replacement_fixture.OLD]], self.retired())

    def test_residual_cleanup_blocks_successor_admission_not_retirement(self):
        handler = self.active_cleanup()
        self.executor.after_cloud = lambda action: (self.executor.maintenance_failures.clear()
                                                   if action == 'terminate-instances' else None)
        with self.assertRaisesRegex(session.Refusal, 'empty readable queue'):
            handler.replace()
        self.assertEqual([[replacement_fixture.OLD]], self.retired())
        self.assertNotIn(['admit-replacement'], self.executor.maintenance_calls)
        self.assertFalse(any('State=RESUME' in call for call in self.executor.slurm_calls))
        self.executor.jobs = []
        self.assertEqual('runtime-ready', handler.replace().state.phase)
        self.assertEqual([[replacement_fixture.OLD]], self.retired())

    def test_foreign_new_running_multiple_malformed_queue_never_retired(self):
        handler = self.active_cleanup()
        for jobs in (['226|aim344-t2|aim344-device|COMPLETING'],
                     ['227|aim344-t1|aim344-device|COMPLETING'],
                     ['226|aim344-t1|aim344-device|RUNNING'], ['malformed'],
                     ['226|aim344-t1|aim344-device|COMPLETING', '227|aim344-t2|aim344-device|RUNNING']):
            with self.subTest(jobs=jobs):
                self.executor.jobs = jobs
                with self.assertRaises(session.Refusal):
                    handler.replace()
                self.assertEqual([], self.retired())

    def test_field_shaped_foreign_reason_never_retires_completing_job(self):
        from test_slurm_reason import FOREIGN_SUFFIXES
        handler = self.active_cleanup()
        for suffix in FOREIGN_SUFFIXES:
            with self.subTest(suffix=suffix):
                self.executor.node_reason = 'aim344-device-recovery' + suffix
                with self.assertRaises(session.Refusal):
                    handler.replace()
                self.assertEqual([], self.retired())

    def test_changed_scheduler_or_cloud_identity_no_retirement(self):
        handler = self.active_cleanup()
        for key, value in (('node_id', replacement_fixture.PEER),
                           ('node_address', '10.0.0.99'), ('member_bad', 'SubnetId'),
                           ('node_reason', 'other-owner')):
            original = getattr(self.executor, key)
            setattr(self.executor, key, value)
            with self.subTest(key=key), self.assertRaises(session.Refusal):
                handler.replace()
            self.assertEqual([], self.retired())
            setattr(self.executor, key, original)

    def test_unreachable_evidence_exception_does_not_block_exact_replacement(self):
        import subprocess
        handler = self.active_cleanup()
        original = self.executor.run_maintenance
        def maintenance_call(argv, timeout=None):
            if argv == ['replacement-evidence']:
                raise subprocess.TimeoutExpired('ssh', 30)
            return original(argv, timeout)
        def after(action):
            if action == 'terminate-instances':
                self.executor.jobs = []
                self.executor.maintenance_failures.clear()
        self.executor.after_cloud = after
        with mock.patch.object(self.executor, 'run_maintenance', side_effect=maintenance_call):
            self.assertEqual('runtime-ready', handler.replace().state.phase)
        self.assertEqual(124, handler._replacement_record()['originals_capture']['returncode'])
        self.assertEqual([[replacement_fixture.OLD]], self.retired())

    def test_queue_changed_during_evidence_capture_prevents_retirement(self):
        handler = self.active_cleanup()
        self.executor.after_evidence = lambda: setattr(self.executor, 'jobs',
                                                       ['227|aim344-t1|aim344-device|RUNNING'])
        with self.assertRaises(session.Refusal):
            handler.replace()
        self.assertEqual([], self.retired())

    def test_lost_retirement_ack_retry_never_targets_successor(self):
        handler = self.active_cleanup()
        self.executor.terminate_ack_lost = True
        with self.assertRaises(session.Refusal):
            handler.replace()
        self.assertEqual([[replacement_fixture.OLD]], self.retired())
        self.executor.jobs = []
        self.executor.maintenance_failures.clear()
        self.assertEqual('runtime-ready', handler.replace().state.phase)
        self.assertEqual([[replacement_fixture.OLD]], self.retired())


class KernelBoundary(unittest.TestCase):
    def test_null_boot_refused_before_journal_spawn(self):
        import contextlib
        boot = '00000000-0000-0000-0000-000000000000'
        for through_parser in (True, False):
            with self.subTest(through_parser=through_parser):
                output = io.StringIO()
                with mock.patch.object(maintenance.subprocess, 'Popen',
                                       side_effect=AssertionError('journalctl must not spawn')) as spawn, \
                     mock.patch.object(maintenance.subprocess, 'run') as run, \
                     contextlib.redirect_stdout(output):
                    with self.assertRaises(maintenance.Refusal):
                        if through_parser:
                            _, options = maintenance.parse(['kernel-evidence', '--boot', boot])
                            maintenance.kernel_evidence({'slurm_bin': '/unused'}, options['boot'])
                        else:
                            maintenance.kernel_evidence({'slurm_bin': '/unused'}, boot)
                    spawn.assert_not_called()
                    run.assert_not_called()
                self.assertEqual('', output.getvalue())

    def test_fixed_boot_grammar_and_real_pipe_bounded_capture(self):
        import contextlib
        import subprocess
        boot = 'aaaaaaaa-0000-0000-0000-000000000001'
        self.assertEqual('kernel-evidence', maintenance.parse(['kernel-evidence', '--boot', boot])[0])
        self.assertIsNone(session.parse_participant_request('collect kernel').error)
        for args in (['kernel-evidence'], ['kernel-evidence', '--boot', '-1'],
                     ['kernel-evidence', '--boot', boot, '--file', '/etc/shadow']):
            with self.assertRaises(maintenance.Refusal):
                maintenance.parse(args)
        real_popen = subprocess.Popen
        for flood in (False, True):
            output = io.StringIO()
            commands = []
            def popen(command, **kwargs):
                commands.append(command)
                program = ('import os; os.write(1,b"NVRM Xid fixture\\n" * 100000)' if flood else
                           'print("efa reset fixture; Missed kernel messages")')
                return real_popen([sys.executable, '-c', program], **kwargs)
            with mock.patch.object(maintenance.subprocess, 'Popen', side_effect=popen), \
                 mock.patch.object(maintenance, 'environment', return_value=os.environ.copy()), \
                 mock.patch.object(maintenance, 'instance_identity', return_value=replacement_fixture.OLD), \
                 contextlib.redirect_stdout(output):
                maintenance.kernel_evidence({}, boot)
            record = json.loads(output.getvalue())
            self.assertLessEqual(record['bytes'], 524288)
            self.assertEqual(flood, record['bounded'])
            self.assertFalse(record['complete_journal'])
            self.assertEqual(hashlib.sha256(record['kernel_text'].encode()).hexdigest(), record['sha256'])
            self.assertEqual(boot, record['boot_id'])
            self.assertEqual([
                '/usr/bin/journalctl', '-k', '--boot=aaaaaaaa000000000000000000000001', '--no-pager',
                '--output=short-iso-precise', '--lines=512',
                '--grep=NVRM|Xid|nvidia|efa|PCI|AER|reset|blocked for|lost|Missed',
            ], commands[0])

    @unittest.skipUnless(Path('/usr/bin/journalctl').is_file(), 'system journalctl unavailable')
    def test_actual_journalctl_accepts_generated_options(self):
        import contextlib
        real_popen = subprocess.Popen
        output = io.StringIO()

        def popen(command, **kwargs):
            # Help parses the helper's actual argv without journal access/root.
            return real_popen(command + ['--help'], **kwargs)

        with mock.patch.object(maintenance.subprocess, 'Popen', side_effect=popen), \
             mock.patch.object(maintenance, 'instance_identity', return_value=replacement_fixture.OLD), \
             contextlib.redirect_stdout(output):
            self.assertEqual(0, maintenance.kernel_evidence({'slurm_bin': '/unused'},
                                                          'aaaaaaaa-0000-0000-0000-000000000001'))
        record = json.loads(output.getvalue())
        self.assertEqual(0, record['returncode'], record['kernel_text'])
        self.assertRegex(record['kernel_text'], r'-k\s+--dmesg')
        self.assertFalse(record['bounded'])
        self.assertFalse(record['complete_journal'])

    def test_journal_error_and_deadline_remain_bounded_nonblocking(self):
        import contextlib
        import selectors
        error = b'fixture journal permission denied\n'
        for deadline in (False, True):
            with self.subTest(deadline=deadline):
                reader, writer = os.pipe()
                os.write(writer, error)
                os.close(writer)
                process = mock.Mock(stdout=os.fdopen(reader, 'rb'), returncode=1)
                process.poll.return_value = None if deadline else 1
                output = io.StringIO()
                with mock.patch.object(maintenance.subprocess, 'Popen', return_value=process), \
                     mock.patch.object(maintenance, 'instance_identity', return_value=replacement_fixture.OLD), \
                     contextlib.redirect_stdout(output):
                    if deadline:
                        with mock.patch.object(selectors, 'DefaultSelector') as selector, \
                             mock.patch.object(maintenance.time, 'monotonic', side_effect=[100, 100]):
                            selector.return_value.select.return_value = []
                            self.assertEqual(0, maintenance.kernel_evidence({'slurm_bin': '/unused'},
                                                                          'aaaaaaaa-0000-0000-0000-000000000001'))
                            selector.return_value.select.assert_called_once_with(8)
                    else:
                        self.assertEqual(0, maintenance.kernel_evidence({'slurm_bin': '/unused'},
                                                                      'aaaaaaaa-0000-0000-0000-000000000001'))
                record = json.loads(output.getvalue())
                expected = b'' if deadline else error
                self.assertEqual(expected.decode(), record['kernel_text'])
                self.assertEqual(len(expected), record['bytes'])
                self.assertEqual(hashlib.sha256(expected).hexdigest(), record['sha256'])
                self.assertEqual(1, record['returncode'])
                self.assertEqual(deadline, record['bounded'])
                self.assertFalse(record['complete_journal'])
                self.assertEqual(deadline, process.kill.called)
                process.wait.assert_called_once_with()
                self.assertTrue(process.stdout.closed)


class GrammarAndTransport(unittest.TestCase):
    def test_active_grammar_is_fixed_no_node_or_function(self):
        for kind in ('gpu', 'efa'):
            self.assertIsNone(session.parse_participant_request('active ' + kind).error)
        for text in ('active gpu --job 226', 'active gpu 0000:57:00.0', 'active efa;id'):
            self.assertIsNotNone(session.parse_participant_request(text).error)
        args = ['gpu-flr', '--job', '226', '--participant', 'aim344-t1', '--operation',
                'table-1/1/' + 'a' * 32, '--boot', 'aaaaaaaa-0000-0000-0000-000000000001',
                '--confirm', 'i-0123456789abcdef0/gpu-flr/GPU-11111111-2222-4333-8444-555555555555']
        self.assertEqual('gpu-flr', maintenance.parse(args)[0])
        for bad in (args[:-2], args + ['--job', '227'], args + ['--bdf', '0000:57:00.0']):
            with self.assertRaises(maintenance.Refusal):
                maintenance.parse(bad)

    def test_real_pipe_transport_waits_for_ack_and_preserves_timeout(self):
        executor = session.Executor('/unused', 'unused')
        program = 'import sys; print("start", flush=True); a=input(); print(a, flush=True)'
        seen = []
        with mock.patch.object(executor, '_maintenance_argv', return_value=[sys.executable, '-c', program]):
            def record(line):
                seen.append(line)
                return 'ACK durable' if line == 'start' else None
            result = executor.run_active_fault([], record, timeout=2)
        self.assertEqual(0, result.returncode)
        self.assertEqual(['start', 'ACK durable'], seen)
        program = 'import time; print("start", flush=True); time.sleep(10)'
        with mock.patch.object(executor, '_maintenance_argv', return_value=[sys.executable, '-c', program]):
            result = executor.run_active_fault([], lambda line: None, timeout=0.1)
        self.assertEqual(124, result.returncode)
        self.assertIn('start', result.stdout)
        self.assertIn('unknown', result.stderr)

    def test_stream_failure_cannot_send_ack(self):
        executor = session.Executor('/unused', 'unused')
        program = 'print("start", flush=True); input(); raise SystemExit("unexpected ACK")'
        with mock.patch.object(executor, '_maintenance_argv', return_value=[sys.executable, '-c', program]):
            with self.assertRaisesRegex(OSError, 'fsync'):
                executor.run_active_fault([], mock.Mock(side_effect=OSError('fsync failed')), timeout=2)

    def test_real_pipe_eof_before_exit_returns_unknown_timeout(self):
        executor = session.Executor('/unused', 'unused')
        program = ('import os,time; print("mutation-start", flush=True); '
                   'os.close(1); os.close(2); time.sleep(10)')
        seen = []
        with mock.patch.object(executor, '_maintenance_argv', return_value=[sys.executable, '-c', program]):
            result = executor.run_active_fault([], seen.append, timeout=0.1)
        self.assertEqual(124, result.returncode)
        self.assertEqual(['mutation-start'], seen)
        self.assertIn('mutation-start', result.stdout)
        self.assertIn('unknown', result.stderr)

    def test_real_pipe_closed_ack_returns_unknown_without_close_traceback(self):
        executor = session.Executor('/unused', 'unused')
        # Close the reader before printing, so ACK write/flush deterministically
        # fails on a real pipe; close must not mask the returned failure either.
        program = ('import os,time; os.close(0); '
                   'print("mutation-start", flush=True); time.sleep(10)')
        seen = []
        def record(line):
            seen.append(line)
            return 'ACK durable'
        with mock.patch.object(executor, '_maintenance_argv', return_value=[sys.executable, '-c', program]):
            result = executor.run_active_fault([], record, timeout=2)
        self.assertEqual(255, result.returncode)
        self.assertEqual(['mutation-start'], seen)
        self.assertIn('mutation-start', result.stdout)
        self.assertIn('unknown', result.stderr)


class CaptureOnly(unittest.TestCase):
    setUp = safety_fixture.S4TelemetryRestoration.setUp
    tearDown = safety_fixture.S4TelemetryRestoration.tearDown
    capture = safety_fixture.S4TelemetryRestoration.capture
    unit_state = safety_fixture.S4TelemetryRestoration.unit_state
    persistence = safety_fixture.S4TelemetryRestoration.persistence
    systemctl_calls = safety_fixture.S4TelemetryRestoration.systemctl_calls

    def test_originals_recorded_without_pausing_active_workload(self):
        helper = safety_fixture.maintenance
        operation = 'table-1/1/' + 'a' * 32
        rc, output = self.capture(helper.gpu_prepare, self.config,
                                  {'operation': operation, 'capture_only': True})
        self.assertEqual(0, rc)
        self.assertIn('recorded persistence_mode=Enabled', output)
        self.assertEqual('active', self.unit_state())
        self.assertEqual('Enabled', self.persistence())
        self.assertFalse(any('stop' in call for call in self.systemctl_calls()))
        for name in ('selected-persistence-mode.txt', 'native-dcgm-state.txt'):
            self.assertEqual(operation, json.loads((self.records / name).read_text())['operation'])


class TargetHelper(unittest.TestCase):
    """Execute the actual embedded helper, with only OS/sysfs/Slurm boundaries fake."""
    def exercise(self, kind='gpu', foreign=False, wrong_ack=False, traffic=True,
                 reason='aim344-device-recovery', reason_after=None):
        import contextlib
        import select
        import stat
        import subprocess
        import time
        from types import SimpleNamespace
        code = (LAB / 'facilitator/device-fault.sh').read_text().split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = {'instance_id': 'i-0123456789abcdef0', 'slurm_node': 'gpu-g7-1',
                   'slurm_bin': '/opt/aws/pcs/scheduler/slurm-25.05/bin',
                   'participant_user': 'aim344-t1', 'gpu_bdf': '0000:57:00.0',
                   'efa_bdf': '0000:53:00.0', 'management_bdf': '0000:47:00.0',
                   'management_interface': 'ena0', 'efa_rdma_device': 'rdmap83s0',
                   'gpu_uuid': 'GPU-11111111-2222-4333-8444-555555555555'}
            boot = 'aaaaaaaa-0000-0000-0000-000000000001'
            def put(name, text):
                path = root / name.lstrip('/')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
                return path
            put('/sys/devices/virtual/dmi/id/board_asset_tag', cfg['instance_id'])
            put('/proc/sys/kernel/random/boot_id', boot)
            put('/proc/123/cgroup', '0::/system.slice/slurmstepd.scope/job_' + ('999' if foreign else '226') + '/step_0')
            put('/proc/123/cmdline', 'python3\0/opt/aim344/workload.py\0device\0')
            for device, driver in (('gpu', 'nvidia'), ('efa', 'efa'), ('management', 'ena')):
                folder = root / 'sys/bus/pci/devices' / cfg[device + '_bdf']
                folder.mkdir(parents=True)
                driver_path = root / 'sys/bus/pci/drivers' / driver
                driver_path.mkdir(parents=True, exist_ok=True)
                (folder / 'driver').symlink_to(driver_path)
                if device != 'management':
                    cfg[device + '_vendor'], cfg[device + '_device'] = '0x1234', '0x5678'
                    for name, value in [('vendor', '0x1234'), ('device', '0x5678'),
                                        ('reset_method', 'flr bus'), ('reset', '0')]:
                        (folder / name).write_text(value)
            net = root / 'sys/class/net/ena0'
            net.mkdir(parents=True)
            (net / 'device').symlink_to(root / 'sys/bus/pci/devices' / cfg['management_bdf'])
            put('/sys/bus/pci/devices/0000:53:00.0/infiniband/rdmap83s0/ports/1/hw_counters/rdma_write_bytes', '0')
            config_path = put('/etc/aim344-device-fault.json', json.dumps(cfg))
            config_path.chmod(0o600)
            writes, output, counter_reads = [], io.StringIO(), [0]
            real_path = Path
            class LocalPath(type(Path())):
                def __init__(self, *parts):
                    candidate = real_path(*parts)
                    if str(candidate).startswith(('/sys/', '/proc/', '/etc/')):
                        candidate = root / str(candidate).lstrip('/')
                    super().__init__(candidate)
                def lstat(self):
                    value = super().lstat()
                    if self == config_path:
                        return SimpleNamespace(st_mode=value.st_mode, st_uid=0)
                    return value
                def read_text(self, *args, **kwargs):
                    if self.name == 'rdma_write_bytes':
                        counter_reads[0] += 1
                        return str(counter_reads[0] * 16777216 if traffic else 0)
                    return super().read_text(*args, **kwargs)
                def write_text(self, text, *args, **kwargs):
                    writes.append((self.name, text, output.getvalue()))
                    return super().write_text(text, *args, **kwargs)
            node_reads = [0]
            def command(args, **kwargs):
                if args[0].endswith('/ip'):
                    return '[{"dev":"ena0"}]'
                if args[0].endswith('/scontrol'):
                    if 'job' in args:
                        return 'JobId=226 TimeLimit=00:06:00'
                    node_reads[0] += 1
                    observed_reason = reason_after if reason_after is not None and node_reads[0] > 1 else reason
                    return ('State=ALLOCATED+DRAIN\n   Reason=' + observed_reason.replace('\n', '\n          ') +
                            '\n   InstanceId=' + cfg['instance_id'])
                if args[0].endswith('/squeue'):
                    return '226|aim344-t1|aim344-device' + ('|RUNNING' if '%T' in args[-1] else '')
                if '--query-compute-apps=pid' in args:
                    return '123'
                return cfg['gpu_uuid'] + ', 00000000:57:00.0'
            action = kind + '-flr'
            target = cfg['gpu_uuid'] if kind == 'gpu' else cfg['efa_bdf']
            argv = ['helper', action, '--job', '226', '--participant', 'aim344-t1',
                    '--operation', 'table-1/1/' + 'a' * 32, '--boot', boot,
                    '--confirm', cfg['instance_id'] + '/' + action + '/' + target]
            def ack(*args):
                last = output.getvalue().splitlines()[-1]
                return ('ACK ' + ('bad' if wrong_ack else hashlib.sha256(last.encode()).hexdigest()) + '\n').encode()
            error = None
            real_import = __import__
            def imports(name, *args, **kwargs):
                if name == 'pathlib':
                    return SimpleNamespace(Path=LocalPath)
                return real_import(name, *args, **kwargs)
            with mock.patch('builtins.__import__', side_effect=imports), mock.patch.object(os, 'geteuid', return_value=0), \
                 mock.patch.object(subprocess, 'check_output', side_effect=command), \
                 mock.patch.object(select, 'select', return_value=([3], [], [])), \
                 mock.patch.object(os, 'read', side_effect=ack), mock.patch.object(time, 'sleep'), \
                 mock.patch.object(sys, 'argv', argv), contextlib.redirect_stdout(output):
                try:
                    exec(compile(code, str(LAB / 'facilitator/device-fault.sh'), 'exec'), {})
                except SystemExit as caught:
                    error = str(caught)
            return writes, output.getvalue(), error

    def test_gpu_exact_flr_after_both_markers_no_idle_remove(self):
        writes, output, error = self.exercise()
        self.assertIsNone(error)
        self.assertEqual([('reset_method', 'flr\n'), ('reset', '1\n')], [(n, v) for n, v, _ in writes])
        self.assertIn('mutation-start', writes[-1][2])
        self.assertIn('mutation-returned', output)

    def test_efa_selected_rdma_write_traffic_is_counted(self):
        writes, output, error = self.exercise(kind='efa')
        self.assertIsNone(error)
        self.assertEqual('reset', writes[-1][0])
        self.assertIn('rdma_write_bytes', output)

    def test_foreign_cgroup_bad_ack_and_zero_efa_traffic_never_reset(self):
        for options in ({'foreign': True}, {'wrong_ack': True}, {'kind': 'efa', 'traffic': False}):
            with self.subTest(options=options):
                writes, _, error = self.exercise(**options)
                self.assertIsNotNone(error)
                self.assertEqual([], writes)

    def test_field_shaped_reason_never_reaches_fake_sysfs(self):
        from test_slurm_reason import FOREIGN_SUFFIXES
        for suffix in FOREIGN_SUFFIXES:
            with self.subTest(suffix=suffix):
                reason = 'aim344-device-recovery' + suffix
                writes, _, error = self.exercise(reason=reason)
                self.assertIn('drain reason', error)
                self.assertIn(reason, error)
                self.assertEqual([], writes)
                writes, _, error = self.exercise(reason_after=reason)
                self.assertIsNotNone(error)
                self.assertEqual([], writes)

    def test_annotated_reason_and_spoofed_ownership_at_actual_helper(self):
        stamp = ' [root@2026-09-24T00:46:47]'  # Recorded annotation, mutated reason body.
        writes, _, error = self.exercise(reason='aim344-device-recovery' + stamp)
        self.assertIsNone(error)
        self.assertEqual('reset', writes[-1][0])
        for reason in ('administrator maintenance' + stamp,
                       'aim344-device-recovery extra text' + stamp,
                       'aim344-device-recovery foreign=value' + stamp,
                       'aim344-device-recovery-foreign' + stamp,
                       'aim344-device-recovery [root@bad]'):
            with self.subTest(reason=reason):
                writes, _, error = self.exercise(reason=reason)
                self.assertIn('drain reason', error)
                self.assertEqual([], writes)


if __name__ == '__main__':
    unittest.main()
