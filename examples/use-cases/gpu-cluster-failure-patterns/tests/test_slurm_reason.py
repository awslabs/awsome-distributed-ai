# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Replay Slurm-shaped fixtures locally; no scheduler or hardware operations.

Identifiers are synthetic; see fixtures/README.md. These are parser regression
inputs, not raw observations or hardware evidence. Negative cases are mutations.
Serializer: SchedMD/slurm slurm-25-05-9-1 src/api/node_info.c:420-475.
"""
import ast
import copy
import base64
import gzip
import inspect
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import test_replacement as replacement

LAB = replacement.bench.LAB
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
        self.bootstrap = replacement.BootstrapFiles().load('replacement-bootstrap.py')
        self.row = ROWS[0]
        self.instance = re.search(r'InstanceId=(\S+)', self.row)[1]
        self.own = 'AIM344-admission-' + self.instance
        self.manifest = {'node': 'gpu-g7-aim344-1', 'slurm_bin': '/fixture/slurm',
                         'participants': [{'uid': 22001}]}

    def isolated(self, row, manifest=None):
        with mock.patch.object(self.bootstrap, 'run', side_effect=[row, '', '']) as run:
            self.bootstrap.isolated(manifest or self.manifest, self.instance)
        self.assertEqual(len(run.call_args_list), 3)
        self.assertTrue(all('update' not in c.args[0] for c in run.call_args_list))

    def test_exact_recorded_pair_and_annotation_audit(self):
        for row in ROWS:
            with self.subTest(row=row):
                instance = re.search(r'InstanceId=(\S+)', row)[1]
                manifest = dict(self.manifest, node=re.search(r'NodeName=(\S+)', row)[1])
                with mock.patch.object(self.bootstrap, 'run', side_effect=[row, '', '']):
                    self.bootstrap.isolated(manifest, instance)
                self.assertEqual(self.bootstrap.node_reason(row),
                                 ('AIM344-admission-' + instance,
                                  'AIM344-admission-' + instance + STAMP))
        self.isolated(self.row.replace(' Reason=', '\n   Reason=').replace(
            ' InstanceId=', '\n   InstanceId=').replace(STAMP, ''))  # reason_time may be zero

    def test_foreign_spoof_extra_and_malformed_annotation_refused(self):
        # Mutations, not additional captured observations.
        for reason in ('administrator maintenance' + STAMP,
                       'Prolog error' + STAMP, 'foreign ' + self.own + STAMP,
                       self.own + '-foreign' + STAMP, self.own + ' extra text' + STAMP,
                       self.own + ' foreign=value' + STAMP,
                       self.own + STAMP + ' extra text',
                       self.own + STAMP + STAMP, self.own + ' [not-metadata]',
                       self.own + ' [root@]', self.own + ' [@2026-09-24T00:46:47]',
                       self.own + ' [root@2026-99-24T00:46:47]',
                       self.own + ' [root@2026-09-24T00:46:47',
                       self.own + ' [root@2026-09-24T00:46:47Z]', '', '(null)'):
            with self.subTest(reason=reason):
                row = self.row.replace(self.own + STAMP, reason)
                with mock.patch.object(self.bootstrap, 'run', return_value=row) as run:
                    with self.assertRaisesRegex(RuntimeError, 'identity-bound') as error:
                        self.bootstrap.isolated(self.manifest, self.instance)
                self.assertIn(reason, str(error.exception))
                self.assertEqual(run.call_count, 1)  # before queue/process or any write


    def test_field_shaped_reason_body_is_never_a_field_boundary(self):
        for row in ROWS:
            instance = re.search(r'InstanceId=(\S+)', row)[1]
            own = 'AIM344-admission-' + instance
            manifest = dict(self.manifest, node=re.search(r'NodeName=(\S+)', row)[1])
            for suffix in FOREIGN_SUFFIXES:
                reason = own + suffix
                observed = row.replace(own + STAMP, reason)
                multiline = observed.replace(' Reason=', '\n   Reason=').replace(
                    ' InstanceId=' + instance, '\n   InstanceId=' + instance)
                for value in (observed, multiline):
                    with self.subTest(instance=instance, reason=reason, multiline='\n' in value):
                        body, raw = self.bootstrap.node_reason(value)
                        self.assertNotEqual(body, own)
                        self.assertIn(reason, raw)
                        with mock.patch.object(self.bootstrap, 'run', return_value=value) as run:
                            with self.assertRaisesRegex(RuntimeError, 'identity-bound'):
                                self.bootstrap.isolated(manifest, instance)
                        self.assertEqual(run.call_count, 1)

    def test_multiline_fields_continuations_and_absent_annotation(self):
        for body in (self.own, self.own + STAMP):
            row = ('NodeName=gpu-g7-aim344-1 State=IDLE+CLOUD+DRAIN\n   Reason=' + body +
                   '\n   Comment=administrator maintenance\n   Extra=foreign=value' +
                   '\n   InstanceId=' + self.instance + ' InstanceType=g7.48xlarge' +
                   '\n   ReservationName=foreign\n   TLSCertTokenSet=No TLSCertLastRenewal=None\n\n')
            self.assertEqual(self.bootstrap.node_reason(row), (self.own, body))
            self.isolated(row)
            for continuation in ('Comment=administrator-maintenance', 'Reason=foreign',
                                 '   Extra=foreign [root@bad]'):
                value = row.replace('\n   Comment=', '\n          ' + continuation + '\n   Comment=')
                parsed, raw = self.bootstrap.node_reason(value)
                self.assertNotEqual(parsed, self.own)
                self.assertIn(continuation, raw)
                with mock.patch.object(self.bootstrap, 'run', return_value=value):
                    with self.assertRaises(RuntimeError):
                        self.bootstrap.isolated(self.manifest, self.instance)

    def test_missing_duplicate_malformed_identity_and_state_refused(self):
        for old, new in [('NodeName=gpu-g7-aim344-1', 'NodeName=another-node'),
                         ('NodeName=gpu-g7-aim344-1', ''),
                         ('InstanceId=' + self.instance, 'InstanceId=i-bad'),
                         ('InstanceId=' + self.instance, ''),
                         ('State=IDLE+CLOUD+DRAIN', ''),
                         ('State=IDLE+CLOUD+DRAIN', 'State=ALLOCATED+DRAIN'),
                         ('Reason=' + self.own + STAMP, ''),
                         ('Reason=', 'MissingReason='),
                         ('InstanceType=', 'InstanceId=' + self.instance + ' InstanceType='),
                         ('Reason=', 'Reason=foreign Reason=')]:
            with self.subTest(old=old, new=new):
                with mock.patch.object(self.bootstrap, 'run', return_value=self.row.replace(old, new)):
                    with self.assertRaisesRegex(RuntimeError, 'identity-bound'):
                        self.bootstrap.isolated(self.manifest, self.instance)

    def test_standalone_parsers_stay_identical_and_preserve_foreign_text(self):
        # These scripts install independently; no new runtime import dependency.
        source = inspect.getsource(self.bootstrap.node_reason).strip()
        self.assertEqual(inspect.getsource(replacement.session.node_reason).strip(), source)
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
            self.assertEqual(replacement.session.DeviceSession._node_field(row, 'Reason'), expected)

    def test_generated_gate_uses_same_parser_and_never_overwrites_foreign_drain(self):
        manifest = dict(self.manifest, region='eu-south-2', protected_instance_ids=[replacement.PEER],
                        release_sha256='a' * 64, maintenance_public_key='ssh-ed25519 TestOnly',
                        verification_environment={'PARTITION': 'gpu-g7-aim344'}, objects={})
        site = {'account': '123456789012', 'coordinator_instance_id': replacement.PEER,
                'coordinator_role_id': 'AROA' + 'A' * 17, 'cluster_id': 'pcs_cluster',
                'group_id': 'pcs_group', 'provisioning_mode': 'ssm-install'}
        generator = replacement.BootstrapFiles().load('prepare-replacement.py')
        artifacts = generator.generate(manifest, site, (LAB / 'facilitator/replacement-bootstrap.py').read_text())
        command = artifacts['bootstrap-document.json']['mainSteps'][0]['inputs']['runCommand'][0]
        installer = command.split("<<'AIM344_FIXED_INSTALL'\n", 1)[1].rsplit('AIM344_FIXED_INSTALL', 1)[0]
        payload = next(n for n in ast.walk(ast.parse(installer)) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'payload' for t in n.targets))
        files = json.loads(gzip.decompress(base64.b64decode(ast.literal_eval(payload.value.args[0].args[0].args[0]))))
        gate = files['/usr/local/sbin/aim344-replacement-admission']['text']
        self.assertIn(inspect.getsource(self.bootstrap.node_reason), gate)
        with tempfile.TemporaryDirectory() as directory:
            def mapped(value):
                return Path(directory) / str(value).lstrip('/')
            for name, text in {'/etc/aim344-replacement.json': json.dumps(manifest),
                               '/sys/devices/virtual/dmi/id/board_asset_tag': self.instance}.items():
                path = mapped(name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            # Missing or malformed admission metadata must still fail closed.
            for ready_text in (None, '{malformed'):
                if ready_text is not None:
                    ready = mapped('/var/lib/aim344-replacement/ready.json')
                    ready.parent.mkdir(parents=True, exist_ok=True)
                    ready.write_text(ready_text)
                for row in (self.row, self.row.replace(self.own, 'administrator maintenance')):
                    with mock.patch.dict('sys.modules', {'pathlib': SimpleNamespace(Path=mapped)}), \
                         mock.patch('subprocess.run', return_value=SimpleNamespace(stdout=row)) as run:
                        with self.assertRaisesRegex(SystemExit, 'remains isolated'):
                            exec(compile(gate, 'generated-admission', 'exec'), {})
                    self.assertEqual(run.call_count, 1)
                row = self.row.replace('State=IDLE+CLOUD+DRAIN', 'State=IDLE+CLOUD')
                with mock.patch.dict('sys.modules', {'pathlib': SimpleNamespace(Path=mapped)}), \
                     mock.patch('subprocess.run', return_value=SimpleNamespace(stdout=row)) as run:
                    with self.assertRaisesRegex(SystemExit, 'not admitted'):
                        exec(compile(gate, 'generated-admission', 'exec'), {})
                self.assertEqual(run.call_count, 2)
                self.assertIn('Reason=' + self.own, run.call_args.args[0])
                for suffix in FOREIGN_SUFFIXES:
                    foreign = row.replace(self.own + STAMP, self.own + suffix)
                    multiline = foreign.replace(' Reason=', '\n   Reason=').replace(
                        ' InstanceId=' + self.instance, '\n   InstanceId=' + self.instance)
                    for observed in (foreign, multiline):
                        with mock.patch.dict('sys.modules', {'pathlib': SimpleNamespace(Path=mapped)}), \
                             mock.patch('subprocess.run', return_value=SimpleNamespace(stdout=observed)) as run:
                            with self.assertRaisesRegex(SystemExit, 'remains isolated'):
                                exec(compile(gate, 'generated-admission', 'exec'), {})
                        self.assertEqual(run.call_count, 1)



class ReasonAudit(replacement.bench.Base):
    def test_foreign_field_tokens_remain_in_status_audit(self):
        for suffix in FOREIGN_SUFFIXES:
            with self.subTest(suffix=suffix):
                raw = 'aim344-device-recovery' + suffix
                self.executor.node_state = 'IDLE+DRAIN'
                self.executor.node_reason = raw
                self.assertEqual(self.make().status().node_reason, raw)
                with self.assertRaises(replacement.session.Refusal):
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


class RebootOwnership(replacement.bench.Base):
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
        with self.assertRaisesRegex(replacement.session.Refusal, 'drain reason changed'):
            self.make().recover()
        self.assertFalse(any('State=RESUME' in c for c in self.executor.slurm_calls))
        self.assertEqual(self.make().status().node_reason, REBOOT_REASON + ' Comment=foreign')

    def test_exact_recorded_reason_and_durable_identity(self):
        handler = self.make(now=REBOOT_RECORD['prepared_at'] + 60)
        state = replacement.session.State.from_dict(REBOOT_RECORD['evidence'])
        handler.assignment.update(target_node=state.target_node,
                                  target_instance_id=state.target_instance_id)
        self.assertEqual(replacement.session.node_reason(REBOOT_ROW),
                         (REBOOT_REASON, REBOOT_REASON + ' [slurm@2026-09-24T02:13:00]'))
        self.assertTrue(handler._owns_recovery_reason(REBOOT_ROW, state))
        self.assertFalse(handler._owns_recovery_reason(REBOOT_ROW))
        for key, value in [('reboot_requests', 0), ('reboot_requested_at', None),
                           ('reboot_requested_at', REBOOT_RECORD['prepared_at'] + 999),
                           ('operation', 'table-2/1/' + 'a' * 32), ('round', 2),
                           ('prepared', []), ('target_instance_id', replacement.PEER),
                           ('target_node', 'foreign')]:
            altered = copy.deepcopy(state)
            setattr(altered, key, value)
            with self.subTest(key=key, value=value):
                self.assertFalse(handler._owns_recovery_reason(REBOOT_ROW, altered))
        for old, new in [('InstanceId=' + state.target_instance_id, 'InstanceId=' + replacement.PEER),
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
        with self.assertRaisesRegex(replacement.session.Refusal, 'still holds'):
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
        with self.assertRaisesRegex(replacement.session.Refusal, 'different reason'):
            self.make().recover()
        with self.assertRaisesRegex(replacement.session.Refusal, 'drain reason changed'):
            handler._require_current_recovery_observations(self.executor.boot_id, 'replacement admission')
        self.assertFalse(any(c[1:2] == ['reboot'] or 'State=RESUME' in c
                             for c in self.executor.slurm_calls))

if __name__ == '__main__':
    unittest.main()
