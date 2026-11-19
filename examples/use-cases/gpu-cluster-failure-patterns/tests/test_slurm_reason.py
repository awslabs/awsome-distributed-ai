# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Replay Slurm-shaped fixtures locally; no scheduler or hardware operations.

Identifiers are synthetic; see fixtures/README.md. These are parser regression
inputs, not raw observations or hardware evidence. Negative cases are mutations.
Serializer: SchedMD/slurm slurm-25-05-9-1 src/api/node_info.c:420-475.
"""
import ast
import copy
import inspect
import json
from pathlib import Path
import re
import unittest

import test_device_session as bench

LAB = bench.LAB
session = bench.session
PEER = 'i-2123456789abcdef0'
ROWS = (Path(__file__).parent / 'fixtures/slurm-25.05.9-admission-nodes.txt').read_text().splitlines()
STAMP = ' [root@2026-09-24T00:46:47]'

# Reason-body mutations, including serializer field names and fake annotations.
FOREIGN_SUFFIXES = tuple(' ' + key + '=administrator-maintenance' + STAMP for key in
                        ('Comment', 'Extra', 'InstanceType', 'ReservationName',
                         'TLSCertTokenSet', 'TLSCertLastRenewal')) + (
    ' Comment=foreign Extra=foreign InstanceType=foreign' + STAMP,
    STAMP + ' Comment=foreign' + STAMP,
    ' [root@bad] Comment=foreign' + STAMP,
    ' Comment=foreign [root@2026-09-24T00:46:47',
    ' Comment=foreign [root@2026-99-24T00:46:47]',
    STAMP + ' Comment=foreign [root@bad]',
    ' : reboot issued' + STAMP,
    ' : reboot requested' + STAMP,
)



class SlurmReason(unittest.TestCase):
    def setUp(self):
        self.row = ROWS[0]
        self.instance = re.search(r'InstanceId=(\S+)', self.row)[1]
        self.own = 'AIM344-admission-' + self.instance

    def test_field_shaped_reason_body_is_never_a_field_boundary(self):
        for row in ROWS:
            instance = re.search(r'InstanceId=(\S+)', row)[1]
            own = 'AIM344-admission-' + instance
            for suffix in FOREIGN_SUFFIXES:
                reason = own + suffix
                observed = row.replace(own + STAMP, reason)
                multiline = observed.replace(' Reason=', '\n   Reason=').replace(
                    ' InstanceId=' + instance, '\n   InstanceId=' + instance)
                for value in (observed, multiline):
                    with self.subTest(instance=instance, reason=reason, multiline='\n' in value):
                        body, raw = session.node_reason(value)
                        self.assertNotEqual(body, own)
                        self.assertIn(reason, raw)

    def test_multiline_fields_continuations_and_absent_annotation(self):
        for body in (self.own, self.own + STAMP):
            row = ('NodeName=gpu-g7-aim344-1 State=IDLE+CLOUD+DRAIN\n   Reason=' + body +
                   '\n   Comment=administrator maintenance\n   Extra=foreign=value' +
                   '\n   InstanceId=' + self.instance + ' InstanceType=g7.48xlarge' +
                   '\n   ReservationName=foreign\n   TLSCertTokenSet=No TLSCertLastRenewal=None\n\n')
            self.assertEqual(session.node_reason(row), (self.own, body))
            for continuation in ('Comment=administrator-maintenance', 'Reason=foreign',
                                 '   Extra=foreign [root@bad]'):
                value = row.replace('\n   Comment=', '\n          ' + continuation + '\n   Comment=')
                parsed, raw = session.node_reason(value)
                self.assertNotEqual(parsed, self.own)
                self.assertIn(continuation, raw)

    def test_standalone_parsers_stay_identical_and_preserve_foreign_text(self):
        # These scripts install independently; no new runtime import dependency.
        source = inspect.getsource(session.node_reason).strip()
        shell = (LAB / 'facilitator/device-fault.sh').read_text().split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
        tree = ast.parse(shell)
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'node_reason')
        self.assertEqual(ast.get_source_segment(shell, function).strip(), source)
        namespace = {'re': re}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'fault-reason', 'exec'), namespace)
        for reason in ('aim344-device-recovery' + STAMP,
                       'aim344-device-recovery extra text' + STAMP,
                       'aim344-device-recovery foreign=value' + STAMP,
                       'administrator maintenance' + STAMP):
            row = self.row.replace(self.own + STAMP, reason)
            expected = reason.removesuffix(STAMP)
            self.assertEqual(namespace['node_reason'](row), (expected, reason))
            self.assertEqual(session.DeviceSession._node_field(row, 'Reason'), expected)

class ReasonAudit(bench.Base):
    def test_foreign_field_tokens_remain_in_status_audit(self):
        for suffix in FOREIGN_SUFFIXES:
            with self.subTest(suffix=suffix):
                raw = 'aim344-device-recovery' + suffix
                self.executor.node_state = 'IDLE+DRAIN'
                self.executor.node_reason = raw
                self.assertEqual(self.make().status().node_reason, raw)
                with self.assertRaises(session.Refusal):
                    self.make().start('gpu')
                self.assertFalse(any('State=RESUME' in call for call in self.executor.slurm_calls))

    def test_full_annotation_persisted_and_reported_without_changing_ownership(self):
        raw = 'aim344-device-recovery' + STAMP
        self.executor.node_state = 'IDLE+DRAIN'
        self.executor.node_reason = raw
        result = self.make().start('gpu')
        self.assertEqual(result.state.original_node_reason, raw)
        self.executor.node_reason = raw
        self.assertEqual(self.make().status().node_reason, raw)



# Synthetic-identifier node block and durable dispatching record preserve
# the Slurm serializer grammar and the identity relationships under test.
REBOOT_ROW = (Path(__file__).parent / 'fixtures/slurm-25.05.9-reboot-issued-node.txt').read_text()
REBOOT_RECORD = json.loads((Path(__file__).parent / 'fixtures/slurm-reboot-replacement-dispatching.json').read_text())
REBOOT_REASON = 'aim344-device-recovery : reboot issued'


class RebootOwnership(bench.Base):
    def test_reboot_suffix_is_refused_by_both_fault_helper_checks(self):
        # Reuse the existing real embedded-helper bench; never touch live sysfs.
        import test_active_flr
        helper = test_active_flr.TargetHelper()
        for reason in (REBOOT_REASON, REBOOT_REASON + ' Comment=foreign',
                       'aim344-device-recovery : unknown'):
            for after in (False, True):
                with self.subTest(reason=reason, after=after):
                    writes, output, error = (helper.exercise(reason_after=reason) if after
                                             else helper.exercise(reason=reason))
                    self.assertEqual(writes, [])
                    self.assertIsNotNone(error)

    def test_foreign_suffix_after_our_reboot_cannot_resume(self):
        self.make().start('gpu')
        slurm = self.executor.run_slurm
        def changed(argv, timeout=None):
            result = slurm(argv, timeout)
            if argv[1:2] == ['reboot']:
                self.executor.node_reason = REBOOT_REASON + ' Comment=foreign'
            return result
        self.executor.run_slurm = changed
        with self.assertRaisesRegex(session.Refusal, 'drain reason changed'):
            self.make().recover()
        self.assertFalse(any('State=RESUME' in c for c in self.executor.slurm_calls))
        self.assertEqual(self.make().status().node_reason, REBOOT_REASON + ' Comment=foreign')

    def test_exact_recorded_reason_and_durable_identity(self):
        handler = self.make(now=REBOOT_RECORD['prepared_at'] + 60)
        # The fixture's round record predates the removal of the deadline sweep.
        evidence = {key: value for key, value in REBOOT_RECORD['evidence'].items()
                    if key not in ('deadline_at', 'deadline_attempts')}
        state = session.State.from_dict(evidence)
        handler.assignment.update(target_node=state.target_node,
                                  target_instance_id=state.target_instance_id)
        self.assertEqual(session.node_reason(REBOOT_ROW),
                         (REBOOT_REASON, REBOOT_REASON + ' [slurm@2026-09-24T02:13:00]'))
        self.assertTrue(handler._owns_recovery_reason(REBOOT_ROW, state))
        self.assertFalse(handler._owns_recovery_reason(REBOOT_ROW))
        for key, value in [('reboot_requests', 0), ('reboot_requested_at', None),
                           ('reboot_requested_at', REBOOT_RECORD['prepared_at'] + 999),
                           ('operation', 'table-2/1/' + 'a' * 32), ('round', 2),
                           ('prepared', []), ('target_instance_id', PEER),
                           ('target_node', 'foreign')]:
            altered = copy.deepcopy(state)
            setattr(altered, key, value)
            with self.subTest(key=key, value=value):
                self.assertFalse(handler._owns_recovery_reason(REBOOT_ROW, altered))
        for old, new in [('InstanceId=' + state.target_instance_id, 'InstanceId=' + PEER),
                         ('State=IDLE+CLOUD+DRAIN+RESERVED', 'State=IDLE+CLOUD'),
                         ('InstanceType=', 'InstanceId=' + state.target_instance_id + ' InstanceType='),
                         (REBOOT_REASON, REBOOT_REASON + ' Comment=foreign'),
                         (REBOOT_REASON, REBOOT_REASON + ' unknown'),
                         (REBOOT_REASON, 'aim344-device-recovery-foreign : reboot issued'),
                         (REBOOT_REASON, 'administrator : reboot issued')]:
            with self.subTest(new=new):
                self.assertFalse(handler._owns_recovery_reason(REBOOT_ROW.replace(old, new), state))

    def test_actual_reboot_suffix_progresses_to_resume_without_second_reboot(self):
        self.make().start('gpu')
        slurm = self.executor.run_slurm
        def issued(argv, timeout=None):
            result = slurm(argv, timeout)
            if argv[1:2] == ['reboot']:
                self.executor.node_reason = REBOOT_REASON + ' [slurm@2026-09-24T02:13:00]'
            return result
        self.executor.run_slurm = issued
        result = self.make().recover()
        self.assertEqual(result.state.phase, 'runtime-ready')
        self.assertTrue(result.state.reboot_completed)
        self.assertEqual(result.state.reboot_requests, 1)
        self.make().recover()
        self.assertEqual(sum(c[1:2] == ['reboot'] for c in self.executor.slurm_calls), 1)
        self.assertEqual(sum('State=RESUME' in c for c in self.executor.slurm_calls), 1)

    def test_outstanding_owned_reboot_suffix_reconciles_before_restore(self):
        handler = self.make()
        handler.start('gpu')
        state = handler.store.read()
        state.reboot_requests = 1
        state.reboot_requested_at = 999
        handler.store.write(state)
        self.executor.node_reason = REBOOT_REASON
        self.executor.node_state = 'IDLE+DRAIN+REBOOT_ISSUED'
        self.executor.boot_id = 'bbbbbbbb-0000-0000-0000-000000000002'
        before = len(self.executor.maintenance_calls)
        with self.assertRaisesRegex(session.Refusal, 'still holds'):
            self.make().recover()
        self.assertFalse(any(c[0] in ('restore-runtime', 'remount-staging')
                             for c in self.executor.maintenance_calls[before:]))
        self.executor.node_state = 'IDLE+DRAIN'
        self.assertEqual(self.make().recover().state.phase, 'runtime-ready')
        self.assertFalse(any(c[1:2] == ['reboot'] for c in self.executor.slurm_calls))

    def test_suffix_never_authorizes_successor_admission_or_unowned_resume(self):
        handler = self.make()
        handler.start('gpu')
        self.executor.node_reason = REBOOT_REASON
        with self.assertRaisesRegex(session.Refusal, 'different reason'):
            self.make().recover()
        with self.assertRaisesRegex(session.Refusal, 'drain reason changed'):
            handler._require_current_recovery_observations(self.executor.boot_id, 'replacement admission')
        self.assertFalse(any(c[1:2] == ['reboot'] or 'State=RESUME' in c
                             for c in self.executor.slurm_calls))

if __name__ == '__main__':
    unittest.main()
