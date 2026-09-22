# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Final review findings 1--3 on the existing recorded controller bench.

These execute controller decisions, not hardware. Each case prints issued calls
so refusal after an unsafe action cannot be mistaken for a safe early refusal.
"""
from unittest.mock import patch
import test_device_session_rework as bench

session = bench.session


class ControllerBench(bench.Base):
    def attempt(self, deadline=False):
        before = len(self.executor.maintenance_calls)
        try:
            handler = self.make(now=9999.0 if deadline else 1000.0)
            result = handler.expire() if deadline else handler.recover()
            refusal = None
        except session.Refusal as error:
            result, refusal = None, error
        print('maintenance:', self.executor.maintenance_calls[before:])
        print('scheduler:', self.executor.slurm_calls)
        print('state:', self.make().store.read().to_dict())
        return result, refusal

    def assert_no_recovery_mutation(self, before):
        self.assertEqual([], self.executor.reboots())
        self.assertEqual([], self.executor.resumes())
        for action in ('remount-staging', 'restore-runtime', 'gpu-restore', 'efa-rebind'):
            self.assertNotIn(action, self.executor.actions()[before:])


class RestorationObligations(ControllerBench):
    def record(self, markers, operation: object = bench.Base.OPERATION, phase='investigating'):
        self.make().start('gpu')
        state = self.make().store.read()
        state.prepared = ['drain', *markers]
        state.operation = operation
        state.phase = phase
        self.make().store.write(state)
        return len(self.executor.actions())

    def test_preparation_only_missing_token_refuses_before_actions(self):
        before = self.record(['gpu-prepare'], operation=None)
        result, refusal = self.attempt()
        self.assertIsNotNone(refusal)
        self.assert_no_recovery_mutation(before)

    def test_malformed_nonempty_token_refuses_before_actions(self):
        before = self.record(['gpu-prepare', 'gpu-remove-attempted'], operation='not-a-token')
        result, refusal = self.attempt()
        self.assertIsNotNone(refusal)
        self.assert_no_recovery_mutation(before)

    def test_preparation_only_malformed_token_deadline_refuses(self):
        before = self.record(['gpu-prepare-attempted'], operation='table-1/1/bad')
        result, refusal = self.attempt(deadline=True)
        self.assertIsNotNone(refusal)
        self.assert_no_recovery_mutation(before)

    def test_recorded_marker_requires_actual_restore(self):
        self.record(['gpu-remove-attempted', 'gpu-prepare-recorded'])
        self.executor.maintenance_failures['gpu-restore'] = (1, 'Original state unavailable')
        result, refusal = self.attempt()
        self.assertIn('gpu-restore', self.executor.actions())
        self.assertIsNotNone(refusal)
        self.assertEqual([], self.executor.resumes())

    def test_recorded_only_preparation_restores_successfully(self):
        self.record(['gpu-prepare-recorded'])
        result, refusal = self.attempt()
        self.assertIsNone(refusal)
        self.assertIn('gpu-restore', self.executor.actions())
        assert result is not None
        self.assertEqual('runtime-ready', result.state.phase)

    def test_mutation_after_refused_capture_is_incoherent(self):
        before = self.record(['gpu-remove-attempted', 'gpu-prepare-refused-before-capture'])
        result, refusal = self.attempt()
        self.assertIsNotNone(refusal)
        self.assert_no_recovery_mutation(before)

    def test_unknown_preparation_outcome_still_restores(self):
        self.record(['gpu-prepare-attempted'])
        result, refusal = self.attempt()
        self.assertIsNone(refusal)
        self.assertIn('gpu-restore', self.executor.actions())
        self.assertEqual(1, len(self.executor.resumes()))

    def test_token_grammar_rejects_whitespace_wrong_types_and_bad_components(self):
        self.make().start('gpu')
        for token in (None, 123, [], {}, '', ' ' + self.OPERATION,
                      self.OPERATION + '\n', 'Table-1/1/' + 'a' * 32,
                      'table-1/1234567/' + 'a' * 32, 'table-1/1/' + 'g' * 32):
            with self.subTest(token=token):
                state = self.make().store.read()
                state.operation = token
                self.make().store.write(state)
                before = len(self.executor.actions())
                result, refusal = self.attempt()
                self.assertIsNotNone(refusal)
                self.assert_no_recovery_mutation(before)

    def test_mixed_fault_kind_markers_refuse(self):
        before = self.record(['gpu-prepare-recorded', 'efa-unbind-attempted'])
        result, refusal = self.attempt()
        self.assertIsNotNone(refusal)
        self.assert_no_recovery_mutation(before)

    def test_later_recorded_capture_after_refusal_restores(self):
        self.record(['gpu-prepare-refused-before-capture', 'gpu-prepare-recorded',
                     'gpu-remove-attempted'])
        result, refusal = self.attempt()
        self.assertIsNone(refusal)
        self.assertIn('gpu-restore', self.executor.actions())
        self.assertEqual(1, len(self.executor.resumes()))

    def test_phase_does_not_remove_restoration_obligation(self):
        self.record(['gpu-prepare-recorded', 'gpu-remove-attempted'], phase='preparing')
        result, refusal = self.attempt()
        self.assertIsNone(refusal)
        self.assertIn('gpu-restore', self.executor.actions())


class EfaReplacementBoundary(ControllerBench):
    def test_failed_rebind_isolates_without_reboot_or_runtime_restore(self):
        self.make().start('efa')
        self.executor.maintenance_failures['efa-rebind'] = (1, 'Driver bind failed')
        for deadline in (False, False, True):
            before = len(self.executor.actions())
            result, refusal = self.attempt(deadline=deadline)
            self.assertIsNotNone(refusal)
            self.assertEqual([], self.executor.reboots())
            self.assertEqual([], self.executor.resumes())
            self.assertNotIn('restore-runtime', self.executor.actions()[before:])
            self.assertNotIn('remount-staging', self.executor.actions()[before:])
            self.assertEqual('replacement-required', self.make().store.read().phase)
            self.assertIn('DRAIN', self.executor.node_state)

    def test_successful_rebind_restores_and_resumes_without_reboot(self):
        self.make().start('efa')
        result, refusal = self.attempt()
        self.assertIsNone(refusal)
        self.assertEqual([], self.executor.reboots())
        self.assertIn('efa-rebind', self.executor.actions())
        self.assertIn('restore-runtime', self.executor.actions())
        self.assertEqual(1, len(self.executor.resumes()))

    def test_already_issued_reboot_is_reconciled_without_another_reboot(self):
        self.make().start('efa')
        state = self.make().store.read()
        state.reboot_requests = 1
        self.make().store.write(state)
        self.executor.boot_id = 'cccccccc-0000-0000-0000-000000000003'
        result, refusal = self.attempt()
        self.assertIsNone(refusal)
        assert result is not None
        self.assertTrue(result.state.reboot_completed)
        self.assertEqual([], self.executor.reboots())
        self.assertIn('remount-staging', self.executor.actions())
        self.assertEqual(1, len(self.executor.resumes()))


    def test_replacement_retains_pair_and_cannot_be_bypassed_by_display_phase(self):
        self.make().start('efa')
        self.executor.maintenance_failures['efa-rebind'] = (1, 'Driver bind failed')
        result, refusal = self.attempt()
        self.assertIsNotNone(refusal)
        before = len(self.executor.actions())
        with self.assertRaises(session.Refusal):
            self.make(assignment='table-2').start('efa')
        self.assertEqual(before, len(self.executor.actions()))
        state = self.make().store.read()
        state.phase = 'investigating'
        self.make().store.write(state)
        self.executor.maintenance_failures.pop('efa-rebind')
        result, refusal = self.attempt()
        self.assertIsNotNone(refusal)
        self.assert_no_recovery_mutation(before)


class FreshDecisionObservations(ControllerBench):
    def change_during_action(self, action, change):
        original = self.executor.run_maintenance

        def run(argv, timeout=None):
            result = original(argv, timeout=timeout)
            if argv[0] == action:
                change()
            return result

        return patch.object(self.executor, 'run_maintenance', side_effect=run)

    def test_pending_arriving_during_rebind_never_resumes(self):
        self.make().start('efa')
        with self.change_during_action('efa-rebind', lambda: setattr(
                self.executor, 'node_state', 'IDLE+DRAIN+REBOOT_REQUESTED')):
            result, refusal = self.attempt()
        self.assertIsNotNone(refusal)
        self.assertEqual([], self.executor.resumes())
        self.assertEqual([], self.executor.reboots())

    def test_pending_arriving_during_failed_rebind_never_reboots(self):
        self.make().start('efa')
        self.executor.maintenance_failures['efa-rebind'] = (1, 'Driver bind failed')
        with self.change_during_action('efa-rebind', lambda: setattr(
                self.executor, 'node_state', 'IDLE+DRAIN+REBOOT_ISSUED')):
            result, refusal = self.attempt()
        self.assertIsNotNone(refusal)
        self.assertEqual([], self.executor.reboots())
        self.assertEqual([], self.executor.resumes())

    def test_pending_arriving_during_checks_never_resumes(self):
        self.make().start('gpu')
        with self.change_during_action('collect', lambda: setattr(
                self.executor, 'node_state', 'IDLE+DRAIN+REBOOT_ISSUED')):
            result, refusal = self.attempt()
        self.assertIsNotNone(refusal)
        self.assertEqual([], self.executor.resumes())

    def test_unreadable_boot_during_checks_never_resumes(self):
        self.make().start('efa')
        with self.change_during_action('collect', lambda: setattr(self.executor, 'boot_id', 'bad')):
            result, refusal = self.attempt()
        self.assertIsNotNone(refusal)
        self.assertEqual([], self.executor.resumes())

    def test_changed_boot_during_checks_never_resumes(self):
        self.make().start('efa')
        with self.change_during_action('collect', lambda: setattr(
                self.executor, 'boot_id', 'cccccccc-0000-0000-0000-000000000003')):
            result, refusal = self.attempt()
        self.assertIsNotNone(refusal)
        self.assertEqual([], self.executor.resumes())

    def test_missing_state_at_entry_refuses_before_actions(self):
        self.make().start('gpu')
        before = len(self.executor.actions())
        original = self.executor.run_slurm

        def run(argv, timeout=None):
            result = original(argv, timeout=timeout)
            if argv[1:3] == ['show', 'node']:
                return session.Completed(0, 'NodeName=gpu-g7-1 Reason=aim344-device-recovery', '')
            return result

        with patch.object(self.executor, 'run_slurm', side_effect=run):
            result, refusal = self.attempt()
        self.assertIsNotNone(refusal)
        self.assert_no_recovery_mutation(before)

    def test_missing_state_during_checks_never_resumes(self):
        self.make().start('efa')
        with self.change_during_action('collect', lambda: setattr(self.executor, 'node_state', '')):
            result, refusal = self.attempt()
        self.assertIsNotNone(refusal)
        self.assertEqual([], self.executor.resumes())

    def test_pending_arrives_after_entry_before_gpu_reboot(self):
        self.make().start('gpu')
        original = self.executor.run_slurm
        queue_reads = 0

        def run(argv, timeout=None):
            nonlocal queue_reads
            result = original(argv, timeout=timeout)
            if argv[0].endswith('squeue'):
                queue_reads += 1
                if queue_reads == 2:  # second target validation, after entry reconciliation
                    self.executor.node_state = 'IDLE+DRAIN+REBOOT_REQUESTED'
            return result

        with patch.object(self.executor, 'run_slurm', side_effect=run):
            result, refusal = self.attempt()
        self.assertIsNotNone(refusal)
        self.assertEqual([], self.executor.reboots())
        self.assertEqual([], self.executor.resumes())

    def test_boot_unreadable_after_entry_before_gpu_reboot(self):
        self.make().start('gpu')
        original = self.executor.run_slurm
        queue_reads = 0

        def run(argv, timeout=None):
            nonlocal queue_reads
            result = original(argv, timeout=timeout)
            if argv[0].endswith('squeue'):
                queue_reads += 1
                if queue_reads == 2:
                    self.executor.boot_id = 'not-a-boot'
            return result

        with patch.object(self.executor, 'run_slurm', side_effect=run):
            result, refusal = self.attempt()
        self.assertIsNotNone(refusal)
        self.assertEqual([], self.executor.reboots())
        self.assertEqual([], self.executor.resumes())
