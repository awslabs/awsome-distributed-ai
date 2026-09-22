# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Local replacement decisions, real coordinator + existing recording bench.

AWS/SSM/Slurm/SSH answers here are fixtures, never live success evidence.
"""
import copy
import contextlib
import datetime
import importlib.util
import io
import json
import os
import stat
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from typing import Callable, Optional
from unittest import mock
import test_device_session as bench

session = bench.session
OLD = 'i-0123456789abcdef0'
NEW = 'i-1123456789abcdef0'
PEER = 'i-2123456789abcdef0'
THIRD = 'i-3123456789abcdef0'
COMMAND = 'aaaaaaaa-1111-4111-8111-111111111111'


def reservation_row(name, start=1000):
    """25.05.9 reservation_info.c serialization, NOT captured hardware output."""
    stamp = lambda seconds: datetime.datetime.fromtimestamp(
        seconds, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
    return (f'ReservationName={name} StartTime={stamp(start)} '
            f'EndTime={stamp(start + 365 * 24 * 3600)} Duration=365-00:00:00 '
            'Nodes=gpu-g7-1 NodeCnt=1 CoreCnt=192 Features=(null) PartitionName=(null) '
            'Flags=SPEC_NODES,STATIC TRES=cpu=192 Users=root Groups=(null) Accounts=(null) '
            'Licenses=(null) State=ACTIVE BurstBuffer=(null) MaxStartDelay=(null)')


class ImageDigest(unittest.TestCase):
    def test_ordered_batches_and_short_reads(self):
        import hashlib
        helper = BootstrapFiles().load('maintenance.py')
        # Cross a batch boundary, with distinct chunks and a partial last chunk.
        data = b''.join(bytes([i]) * (1024 * 1024) for i in range(33)) + b'end'
        with tempfile.TemporaryFile() as image:
            image.write(data)
            image.flush()
            real_pread = os.pread
            def short_read(fd, size, offset):
                return real_pread(fd, min(size, 131071), offset)
            with mock.patch.object(helper.os, 'pread', side_effect=short_read):
                self.assertEqual(helper._image_digest(image.fileno(), len(data)),
                                 hashlib.sha256(data).hexdigest())
            self.assertEqual(image.tell(), len(data))

    def test_premature_eof_and_io_error_refuse(self):
        helper = BootstrapFiles().load('maintenance.py')
        with tempfile.TemporaryFile() as image:
            image.write(b'short')
            image.flush()
            with self.assertRaisesRegex(helper.Refusal, 'changed size'):
                helper._image_digest(image.fileno(), 100)
            with mock.patch.object(helper.os, 'pread', side_effect=OSError('read failure')):
                with self.assertRaisesRegex(OSError, 'read failure'):
                    helper._image_digest(image.fileno(), 5)


class ReplacementExecutor(bench.FakeExecutor):
    def __init__(self):
        super().__init__()
        self.cloud_calls = []
        self.target_host = '10.0.0.1'
        self.known_hosts = ''
        self.node_id = OLD
        self.node_address = '10.0.0.1'
        self.states = {OLD: 'running', NEW: 'running'}
        self.addresses = {OLD: '10.0.0.1', NEW: '10.0.0.1'}
        self.successors = {OLD: NEW}
        self.terminate_ack_lost = False
        self.bootstrap_status = 'Success'
        self.bootstrap_report_bad = False
        self.failed_check = ''
        self.pending_after_checks = False
        self.group_bad = False
        self.member_bad = ''
        self.duplicate_identity = False
        self.after_cloud: Optional[Callable] = None
        self.after_evidence: Optional[Callable] = None
        self.queue_unreadable = False
        self.install_mode = False
        self.reservation = ''
        self.reservation_create_lost_ack = False
        self.reservation_delete_failure = False
        self.reservation_transform = lambda text: text
        self.topology = 'Sockets=192 CoresPerSocket=1 ThreadsPerCore=1 CPUTot=192'
        self.group = {'id': 'pcs_group', 'status': 'ACTIVE',
                      'scalingConfiguration': {'minInstanceCount': 2, 'maxInstanceCount': 2},
                      'customLaunchTemplate': {'id': 'lt-test', 'version': '1'}}

    def run_cloud(self, region, service, action, payload):
        self.cloud_calls.append((service, action, copy.deepcopy(payload)))
        answer = {}
        if action == 'get-cluster':
            answer = {'cluster': {'status': 'ACTIVE', 'slurmConfiguration': {
                'slurmCustomSettings': [{'parameterName': 'Prolog', 'parameterValue':
                    '/usr/local/sbin/aim344-replacement-prolog' if self.install_mode
                    else '/opt/aim344/.prolog/dispatch.sh'}]}}}
        elif action == 'get-compute-node-group':
            answer = {'computeNodeGroup': dict(self.group, status='UPDATING') if self.group_bad else self.group}
        elif action == 'describe-instances':
            instance = payload['InstanceIds'][0]
            row = {'InstanceId': instance, 'State': {'Name': self.states[instance]},
                   'CapacityReservationId': 'cr-test', 'InstanceType': 'g7.48xlarge',
                   'SubnetId': 'subnet-test', 'ImageId': 'ami-test',
                   'PrivateIpAddress': self.addresses[instance],
                   'Tags': [{'Key': 'aws:pcs:compute-node-group-id', 'Value': 'pcs_group'}]}
            if self.member_bad:
                row[self.member_bad] = 'foreign'
            answer = {'Reservations': [{'Instances': [row]}]}
        elif action == 'terminate-instances':
            old = payload['InstanceIds'][0]
            assert payload['InstanceIds'] == [self.node_id] and old in self.successors, 'only the failed predecessor may be terminated'
            self.states[old] = 'terminated'
            self.node_id = self.successors.pop(old)
            self.node_address = self.addresses[self.node_id]
            self.inspect['instance_id'] = self.node_id
            self.inspect['confirmation'] = {action: f'{self.node_id}/{action}/' +
                (self.inspect['gpu_uuid'] if action == 'gpu-remove' else self.inspect['efa_bdf'])
                for action in ('gpu-remove', 'efa-unbind', 'efa-rebind')}
            self.node_state = 'IDLE+CLOUD'
            self.node_reason = ''
            if self.terminate_ack_lost:
                self.terminate_ack_lost = False
                return session.Completed(124, '', 'fixture: acknowledgement lost')
            answer = {'TerminatingInstances': [{'InstanceId': old, 'CurrentState': {'Name': 'shutting-down'}}]}
        elif action == 'send-command':
            assert payload['InstanceIds'] == [self.node_id]
            answer = {'Command': {'CommandId': COMMAND, 'InstanceIds': [self.node_id]}}
        elif action == 'get-command-invocation':
            report = {'ready': True, 'instance_id': self.node_id, 'node': 'gpu-g7-1',
                      'release': 'a' * 64, 'host_key': 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITestOnly'}
            if self.bootstrap_report_bad:
                report['instance_id'] = OLD
            answer = {'CommandId': COMMAND, 'InstanceId': self.node_id, 'Status': self.bootstrap_status,
                      'ResponseCode': 0 if self.bootstrap_status == 'Success' else 1,
                      'StandardOutputContent': json.dumps(report)}
        if self.after_cloud:
            self.after_cloud(action)
        return session.Completed(0, json.dumps(answer), '')

    def run_slurm(self, argv, timeout=None):
        if Path(argv[0]).name == 'scontrol':
            if argv[1:3] == ['show', 'reservation']:
                self.slurm_calls.append(list(argv))
                return session.Completed(0, self.reservation, '')
            if argv[1:3] == ['create', 'reservation']:
                self.slurm_calls.append(list(argv))
                name = next(a.split('=', 1)[1] for a in argv if a.startswith('ReservationName='))
                self.reservation = self.reservation_transform(reservation_row(name))
                return session.Completed(1 if self.reservation_create_lost_ack else 0, '', '')
            if argv[1] == 'delete':
                self.slurm_calls.append(list(argv))
                if not self.reservation_delete_failure:
                    self.reservation = ''
                return session.Completed(1 if self.reservation_delete_failure else 0, '', '')
        result = super().run_slurm(argv, timeout)
        if Path(argv[0]).name == 'squeue' and self.queue_unreadable:
            return session.Completed(1, '', 'fixture unreadable queue')
        if Path(argv[0]).name == 'scontrol' and argv[1:3] == ['show', 'node']:
            extra = ' InstanceId=' + OLD if self.duplicate_identity else ''
            state = '' if self.node_state is None else f'State={self.node_state}'
            return session.Completed(0, f'NodeName={argv[3]} {state} '
                f'Reason={self.node_reason} InstanceId={self.node_id} NodeAddr={self.node_address}{extra} '
                f'{self.topology}\n', '')
        return result

    def run_maintenance(self, argv, timeout=None):
        if argv == ['replacement-evidence'] and self.after_evidence:
            self.after_evidence()
        if argv[0] == 'collect' and self.node_id in (NEW, THIRD):
            self.maintenance_calls.append(list(argv))
            check = argv[1]
            names = {'0': 'nvidia-smi', '2': 'efa-enumeration', '3': 'topology-check', '6': 'efa-loopback'}
            verdict = 'FAIL' if self.failed_check == check else 'PASS'
            if check == '6' and self.pending_after_checks:
                self.node_state += '+REBOOT_ISSUED'
            return session.Completed(0, f'[DEBUG] EFA statistics: no errors detected\n[{verdict}] {check}-{names[check]}: fixture complete\n', '')
        return super().run_maintenance(argv, timeout)


class Replacement(bench.Base):
    def setUp(self):
        super().setUp()
        self.executor = ReplacementExecutor()
        self.config_body['assignments']['table-1']['target_host'] = '10.0.0.1'
        self.config_body['assignments']['table-1']['replacement'] = {
            'enabled': True, 'region': 'eu-south-2', 'group_id': 'pcs_group', 'cluster_id': 'pcs_cluster',
            'capacity_reservation_id': 'cr-test', 'instance_type': 'g7.48xlarge',
            'subnet_ids': ['subnet-test'], 'image_id': 'ami-test', 'protected_instance_ids': [PEER],
            'scaling_configuration': self.executor.group['scalingConfiguration'],
            'launch_template': self.executor.group['customLaunchTemplate'],
            'bootstrap_document': 'AIM344Fixture', 'bootstrap_document_version': '1',
            'bootstrap_document_sha256': 'b' * 64, 'release_sha256': 'a' * 64}
        self.write_config()
        handler = self.make()
        handler.start('efa')
        state = handler.store.read()
        state.phase = 'replacement-required'
        state.failure = 'fixture: supported recovery could not qualify'
        handler.store.write(state)

    def retired(self):
        return [p['InstanceIds'] for _, action, p in self.executor.cloud_calls if action == 'terminate-instances']

    def no_roll(self):
        self.config_body['assignments']['table-1']['replacement']['provisioning_mode'] = 'ssm-install'
        self.write_config()
        self.executor.install_mode = True
        return self.make()

    def test_no_roll_barrier_precedes_retirement_and_survives_bootstrap_failure(self):
        handler = self.no_roll()
        self.executor.bootstrap_status = 'Failed'
        def guarded(action):
            if action in ('terminate-instances', 'send-command'):
                self.assertIn('Nodes=gpu-g7-1', self.executor.reservation)
                self.assertIn('Users=root', self.executor.reservation)
        self.executor.after_cloud = guarded
        with self.assertRaises(session.Refusal):
            handler.replace()
        self.assertTrue(self.executor.reservation)
        self.assertEqual(self.retired(), [[OLD]])
        self.executor.bootstrap_status = 'Success'
        handler.replace()
        self.assertFalse(self.executor.reservation)
        self.assertEqual(self.retired(), [[OLD]])

    def test_no_roll_lost_create_ack_and_release_retry_do_not_retire_successor(self):
        handler = self.no_roll()
        self.executor.reservation_create_lost_ack = True
        self.executor.reservation_delete_failure = True
        with self.assertRaises(session.Refusal):
            handler.replace()
        self.assertEqual(handler._replacement_record()['phase'], 'complete')
        self.assertEqual(self.retired(), [[OLD]])
        self.assertTrue(self.executor.reservation)
        with self.assertRaisesRegex(session.Refusal, 'release is pending'):
            self.make().start('gpu')
        self.executor.reservation_delete_failure = False
        self.make().replace()
        self.assertFalse(self.executor.reservation)
        self.assertEqual(self.retired(), [[OLD]])

    def test_no_roll_changed_or_missing_barrier_refuses(self):
        handler = self.no_roll()
        self.executor.bootstrap_status = 'Failed'
        with self.assertRaises(session.Refusal):
            handler.replace()
        original = self.executor.reservation
        for text in ('', original.replace('Users=root', 'Users=aim344-t1'),
                     original.replace('Nodes=gpu-g7-1', 'Nodes=gpu-g7-2'),
                     original.replace('STATIC', 'REPLACE'),
                     original.replace('365-00:00:00', '00:10:00'),
                     original + '\n' + original):
            with self.subTest(reservation=text):
                self.executor.reservation = text
                with self.assertRaises(session.Refusal):
                    self.make().replace()
                self.assertEqual(self.retired(), [[OLD]])
        self.executor.reservation = original
        self.executor.bootstrap_status = 'Success'
        self.make().replace()

    def test_no_roll_foreign_preexisting_reservation_not_adopted(self):
        handler = self.no_roll()
        self.executor.reservation = reservation_row('aim344-replacement-' + OLD)
        with self.assertRaisesRegex(session.Refusal, 'not owned'):
            handler.replace()
        self.assertEqual(self.retired(), [])

    def test_no_roll_source_shaped_lifetime_persisted_and_checked(self):
        handler = self.no_roll()
        self.executor.reservation_delete_failure = True
        with self.assertRaises(session.Refusal):
            handler.replace()
        record = handler._replacement_record()
        self.assertEqual(record['admission_lifetime'], [1000, 1000 + 365 * 24 * 3600])
        self.assertEqual(record['admission_topology'], {'cores': 192, 'cpus': 192})
        self.assertEqual(session.safe_environment('bin')['TZ'], 'UTC')
        self.assertEqual(session.safe_environment('bin')['SLURM_TIME_FORMAT'], 'standard')
        self.executor.reservation_delete_failure = False
        original = self.executor.reservation
        # Alter both times coherently: same duration/ACTIVE is still not our barrier.
        for start in (999, 1001):
            self.executor.reservation = reservation_row('aim344-replacement-' + OLD, start)
            with self.assertRaises(session.Refusal):
                self.make(now=1010).replace()
        self.executor.reservation = original
        for now in (999, record['admission_lifetime'][1] - 3599, record['admission_lifetime'][1] + 1):
            with self.assertRaises(session.Refusal):
                self.make(now=now).replace()
        self.assertTrue(self.executor.reservation)
        self.make(now=1010).replace()
        self.assertFalse(self.executor.reservation)
        self.assertEqual(self.retired(), [[OLD]])

    def test_no_roll_resource_corruption_create_retry_and_release(self):
        mutations = [
            ('CoreCnt=192', 'CoreCnt=1'), ('CoreCnt=192', ''),
            ('CoreCnt=192', 'CoreCnt=192 CoreCnt=192'),
            ('TRES=cpu=192', 'TRES=cpu=1'), ('TRES=cpu=192', ''),
            ('TRES=cpu=192', 'TRES=cpu=192,cpu=192'),
            ('TRES=cpu=192', 'TRES=cpu=192 TRES=cpu=192'),
            ('TRES=cpu=192', 'NodeName=gpu-g7-1 CoreIDs=0 TRES=cpu=192'),
            ('TRES=cpu=192', 'CoreIDs=0-191 TRES=cpu=192'),
            ('TRES=cpu=192', 'NodeName=gpu-g7-2 TRES=cpu=192'),
            ('Groups=(null)', 'Groups=users'),
            ('Flags=SPEC_NODES,STATIC', 'Flags=SPEC_NODES,STATIC_ALLOC'),
            ('Flags=SPEC_NODES,STATIC', 'Flags=SPEC_NODES,STATIC,STATIC'),
            ('StartTime=1970-01-01T00:16:40', 'StartTime=unknown'),
            ('EndTime=1971-01-01T00:16:40', 'EndTime=1971-01-02T00:16:40'),
            ('StartTime=1970-01-01T00:16:40', ''),
            ('EndTime=1971-01-01T00:16:40', 'EndTime=1971-01-01T00:16:40 EndTime=1971-01-01T00:16:40'),
        ]
        for boundary in ('create', 'retry', 'release'):
            for old, new in mutations:
                with self.subTest(boundary=boundary, old=old, new=new):
                    case = Replacement()
                    case.setUp()
                    try:
                        handler = case.no_roll()
                        if boundary == 'create':
                            case.executor.reservation_transform = lambda text: text.replace(old, new)
                        else:
                            case.executor.bootstrap_status = 'Failed' if boundary == 'retry' else 'Success'
                            case.executor.reservation_delete_failure = boundary == 'release'
                            with case.assertRaises(session.Refusal):
                                handler.replace()
                            case.executor.reservation = case.executor.reservation.replace(old, new)
                            case.executor.bootstrap_status = 'Success'
                            case.executor.reservation_delete_failure = False
                        before = len(case.executor.slurm_calls)
                        with case.assertRaises(session.Refusal):
                            case.make().replace()
                        case.assertFalse(any(c[1] == 'delete' for c in case.executor.slurm_calls[before:]))
                        case.assertEqual(case.retired(), [] if boundary == 'create' else [[OLD]])
                        case.assertTrue(case.executor.reservation)
                        print('NO_ROLL_CORRUPTION', json.dumps({'boundary': boundary, 'old': old,
                              'new': new, 'retired': case.retired(), 'slurm': case.executor.slurm_calls}))
                    finally:
                        case.tearDown()

    def test_no_roll_lost_create_record_rechecks_resources_and_intent(self):
        handler = self.no_roll()
        run = self.executor.run_slurm
        def lost(argv, timeout=None):
            result = run(argv, timeout)
            if argv[1:3] == ['create', 'reservation']:
                raise OSError('fixture crash after scheduler create before readback')
            return result
        with mock.patch.object(self.executor, 'run_slurm', side_effect=lost):
            with self.assertRaises(OSError):
                handler.replace()
        self.assertEqual(self.retired(), [])
        self.assertNotIn('admission_lifetime', handler._replacement_record())
        original = self.executor.reservation
        for row in (original.replace('CoreCnt=192', 'CoreCnt=1'),
                    reservation_row('aim344-replacement-' + OLD, 900),
                    reservation_row('aim344-replacement-' + OLD, 1400)):
            self.executor.reservation = row
            with self.assertRaises(session.Refusal):
                self.make(now=1500).replace()
            self.assertEqual(self.retired(), [])
        self.executor.reservation = original
        self.make(now=1500).replace()
        self.assertEqual(self.retired(), [[OLD]])

    def test_no_roll_topology_missing_inconsistent_or_changed_refuses(self):
        handler = self.no_roll()
        original = self.executor.topology
        for text in ('', original + ' Sockets=192', original.replace('CPUTot=192', 'CPUTot=96')):
            self.executor.topology = text
            with self.assertRaises(session.Refusal):
                handler.replace()
            self.assertEqual(self.retired(), [])
        self.executor.topology = original
        self.executor.bootstrap_status = 'Failed'
        with self.assertRaises(session.Refusal):
            handler.replace()
        self.executor.topology = 'Sockets=96 CoresPerSocket=1 ThreadsPerCore=2 CPUTot=192'
        with self.assertRaises(session.Refusal):
            self.make().replace()
        self.assertEqual(self.retired(), [[OLD]])

    def test_no_roll_final_binding_sync_failure_retries_before_delete(self):
        handler = self.no_roll()
        self.config_body['assignments']['alias'] = copy.deepcopy(self.config_body['assignments']['table-1'])
        self.write_config()
        path = handler._replacement_path()
        sync = session._sync_directory
        def fail_after_complete(directory):
            if path.exists() and json.loads(path.read_text()).get('phase') == 'complete':
                raise OSError('fixture final binding rename succeeded, directory fsync failed')
            return sync(directory)
        with mock.patch.object(session, '_sync_directory', side_effect=fail_after_complete):
            with self.assertRaises(OSError):
                handler.replace()
        self.assertEqual(handler._replacement_record()['phase'], 'complete')
        self.assertTrue(handler._replacement_record()['admission_release_pending'])
        self.assertFalse(any(c[1] == 'delete' for c in self.executor.slurm_calls))
        fsync = os.fsync
        for failure in ('file', 'directory'):
            def failing(fd):
                kind = 'directory' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'file'
                if kind == failure:
                    raise OSError('fixture persistent ' + kind + ' sync failure')
                return fsync(fd)
            for _ in range(2):
                with mock.patch.object(session.os, 'fsync', side_effect=failing):
                    with self.assertRaises(OSError if failure == 'file' else session.Refusal):
                        self.make().replace()
                self.assertFalse(any(c[1] == 'delete' for c in self.executor.slurm_calls))
                with self.assertRaises(session.Refusal):
                    self.make().start('gpu')
                with self.assertRaises(session.Refusal):
                    self.make('alias').replace()
                with self.assertRaises(session.Refusal):
                    self.make('alias').start('efa')
        # Observe actual successful synchronization order on a fresh final retry.
        events = []
        run = self.executor.run_slurm
        def synced(fd):
            events.append('directory' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'file')
            return fsync(fd)
        def slurm(argv, timeout=None):
            if argv[1] == 'delete':
                events.append('delete')
                self.assertEqual(events[-3:], ['file', 'directory', 'delete'])
            return run(argv, timeout)
        with mock.patch.object(session.os, 'fsync', side_effect=synced), \
             mock.patch.object(self.executor, 'run_slurm', side_effect=slurm):
            self.make().replace()
        self.assertFalse(self.executor.reservation)
        self.assertEqual(self.retired(), [[OLD]])
        print('NO_ROLL_DURABILITY', json.dumps({'events': events, 'retired': self.retired()}))

    def late_retirement_cases(self, boundary):
        changes = {'job': ('jobs', ['99|another-user|other|RUNNING']),
                   'queue-unreadable': ('queue_unreadable', True),
                   'missing-state': ('node_state', None),
                   'lost-isolation': ('node_state', 'IDLE+CLOUD'),
                   'busy-isolated': ('node_state', 'ALLOCATED+DRAIN'),
                   'changed-id': ('node_id', PEER),
                   'changed-address': ('node_address', '10.0.0.99'),
                   'foreign-reason': ('node_reason', 'administrator-maintenance')}
        for name, (field, value) in changes.items():
            with self.subTest(boundary=boundary, change=name):
                case = Replacement()
                case.setUp()
                try:
                    executor = case.executor
                    def mutate(*args):
                        if not args or args[0] == 'describe-instances':
                            setattr(executor, field, value)
                    if boundary == 'evidence':
                        executor.after_evidence = mutate
                    else:
                        executor.after_cloud = mutate
                    refusal = None
                    try:
                        case.make().replace()
                    except session.Refusal as error:
                        refusal = str(error)
                    print('LATE_RETIREMENT_TRACE', json.dumps({
                        'boundary': boundary, 'change': name, 'refusal': refusal,
                        'cloud': executor.cloud_calls, 'slurm': executor.slurm_calls,
                        'maintenance': executor.maintenance_calls}))
                    case.assertEqual(case.retired(), [])
                    case.assertIsNotNone(refusal)
                finally:
                    case.tearDown()

    def test_late_evidence_changes_refuse_retirement(self):
        self.late_retirement_cases('evidence')

    def test_late_ec2_changes_refuse_retirement(self):
        self.late_retirement_cases('ec2')

    def successor_gpu_case(self, damage=None):
        # Retirement/discovery precede the actual local bootstrap. The first SSM
        # poll waits; the next returns the sandbox's real authenticated-report shape.
        self.executor.bootstrap_status = 'InProgress'
        with self.assertRaisesRegex(session.Refusal, 'bootstrap is incomplete'):
            self.make().replace()
        self.executor.bootstrap_status = 'Success'
        def continue_on_successor(bootstrap, node_path, manifest, report):
            self.config_body['assignments']['table-1']['replacement']['release_sha256'] = report['release']
            self.write_config()
            maintenance = BootstrapFiles().load('maintenance.py')
            cfg = json.loads(node_path('/etc/aim344-maintenance.json').read_text())
            inventory = json.loads(node_path('/etc/aim344-device-fault.json').read_text())
            # Execute the shipped token function with fresh bootstrap inventory.
            import ast
            script = (bench.LAB / 'facilitator/device-fault.sh').read_text()
            tree = ast.parse(script.split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0])
            token = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'token')
            namespace = {'cfg': inventory, 'instance': NEW}
            exec(compile(ast.Module(body=[token], type_ignores=[]), 'device-fault-token', 'exec'), namespace)
            inventory['confirmation'] = {a: namespace['token'](a)
                for a in ('gpu-remove', 'efa-unbind', 'efa-rebind')}
            commands, outputs = [], []
            original_cloud = self.executor.run_cloud
            original_maintenance = self.executor.run_maintenance
            def cloud(region, service, action, payload):
                result = original_cloud(region, service, action, payload)
                if action == 'get-command-invocation':
                    body = json.loads(result.stdout)
                    body['StandardOutputContent'] = json.dumps(report)
                    return session.Completed(0, json.dumps(body), '')
                return result
            def command(argv, **kwargs):
                commands.append(list(argv))
                if argv[0] == '/usr/bin/findmnt':
                    overlay = damage == 'overlaid' and damaged[0]
                    return subprocess.CompletedProcess(argv, 0, '/opt/aim344\n' if overlay else '/\n', '')
                # Model only the privileged bind boundary; run real validation.
                if argv[:2] == ['/usr/bin/mount', '--bind']:
                    source, stage = map(Path, argv[2:])
                    for name in ('aim344.sqsh', 'nccl-baseline.sqsh'):
                        (stage / name).write_bytes((source / name).read_bytes())
                return subprocess.CompletedProcess(argv, 0, '', '')
            def runtime(argv, config, timeout=None):
                commands.append(list(argv))
                return 0  # privileged runtime shell boundary, not its decision
            def route(argv, timeout=None):
                if argv[0] not in ('remount-staging', 'restore-runtime'):
                    return original_maintenance(argv, timeout)
                self.executor.maintenance_calls.append(list(argv))
                out, err = io.StringIO(), io.StringIO()
                with mock.patch.object(maintenance, 'Path', side_effect=node_path), \
                     mock.patch.object(maintenance, 'load_config', return_value=cfg), \
                     mock.patch.object(maintenance.os, 'geteuid', return_value=0), \
                     mock.patch.object(maintenance, 'run', side_effect=runtime), \
                     mock.patch.object(maintenance.subprocess, 'run', side_effect=command), \
                     mock.patch.object(maintenance.sys, 'argv', ['helper', *argv]), \
                     mock.patch.object(maintenance, 'STAGING_UID', os.getuid(), create=True), \
                     contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    rc = maintenance.main()
                outputs.append({'argv': argv, 'rc': rc, 'stdout': out.getvalue(), 'stderr': err.getvalue()})
                return session.Completed(rc, out.getvalue(), err.getvalue())
            damaged = [False]
            with mock.patch.object(self.executor, 'run_cloud', side_effect=cloud), \
                 mock.patch.object(self.executor, 'run_maintenance', side_effect=route):
                self.assertEqual(self.make().replace().state.phase, 'runtime-ready')
                self.executor.inspect = inventory
                handler = self.make()
                started = handler.start('gpu')
                self.assertEqual(started.state.round, 2)
                self.assertEqual(started.state.target_instance_id, NEW)
                self.assertIn(inventory['confirmation']['gpu-remove'],
                              ' '.join(' '.join(c) for c in self.executor.maintenance_calls))
                stage = node_path('/opt/aim344')
                pristine = (stage / 'aim344.sqsh').read_bytes()
                damaged[0] = True
                if damage == 'missing':
                    (stage / 'aim344.sqsh').unlink()
                if damage == 'wrong-source':
                    (stage / 'aim344.sqsh').write_bytes(b'wrong root source')
                if damage == 'writable-source':
                    (stage / 'aim344.sqsh').chmod(0o666)
                source = node_path('/opt/dlami/nvme/aim344')
                if damage == 'stale-nvme':
                    source.mkdir(parents=True)
                    for image in ('aim344.sqsh', 'nccl-baseline.sqsh'):
                        (source / image).write_bytes(b'stale unrelated NVMe image')
                before = len(self.executor.maintenance_calls)
                command_before = len(commands)
                real_mount, real_stat = Path.is_mount, os.stat
                def mounted(path):
                    return str(path).endswith('/opt/dlami/nvme') if damage == 'stale-nvme' else real_mount(path)
                def observed_stat(path, *args, **kwargs):
                    if (damage == 'stale-nvme' and str(path) == str(stage)
                            and any(c[:2] == ['/usr/bin/mount', '--bind'] for c in commands)):
                        return real_stat(source, *args, **kwargs)
                    return real_stat(path, *args, **kwargs)
                refusal = None
                result = None
                try:
                    with mock.patch.object(Path, 'is_mount', mounted), \
                         mock.patch.object(os, 'stat', side_effect=observed_stat):
                        result = handler.recover()
                except session.Refusal as error:
                    refusal = str(error)
                finally:
                    print('SUCCESSOR_GPU_TRACE', json.dumps({'damage': damage,
                        'inventory': inventory, 'config': cfg, 'refusal': refusal,
                        'state': handler.store.read().to_dict(), 'cloud': self.executor.cloud_calls,
                        'slurm': self.executor.slurm_calls, 'maintenance': self.executor.maintenance_calls,
                        'helper_outputs': outputs, 'commands': commands}))
                actions = [c[0] for c in self.executor.maintenance_calls[before:]]
                self.assertIn('remount-staging', actions)
                self.assertTrue(any(c[1:2] == ['reboot'] for c in self.executor.slurm_calls))
                self.assertFalse(any(c[:2] == ['/usr/bin/mount', '--bind'] for c in commands[command_before:]))
                if damage in ('missing', 'wrong-source', 'overlaid', 'writable-source'):
                    self.assertIsNotNone(refusal)
                    self.assertNotIn('restore-runtime', actions)
                    self.assertIn('DRAIN', self.executor.node_state)
                else:
                    self.assertIsNone(refusal)
                    assert result is not None
                    self.assertEqual(result.state.phase, 'runtime-ready')
                    self.assertIn('restore-runtime', actions)
                    self.assertEqual((stage / 'aim344.sqsh').read_bytes(), pristine)
        fixture = BootstrapFiles()
        try:
            fixture.bootstrap_sandbox(continue_on_successor)
        finally:
            fixture.doCleanups()

    def test_successor_gpu_root_staging_recovery(self):
        self.successor_gpu_case()

    def test_successor_gpu_missing_source_refuses_runtime(self):
        self.successor_gpu_case('missing')

    def test_successor_gpu_wrong_source_refuses_runtime(self):
        self.successor_gpu_case('wrong-source')

    def test_successor_gpu_stale_nvme_never_overlays_root(self):
        self.successor_gpu_case('stale-nvme')

    def test_successor_gpu_overlaid_source_refuses_runtime(self):
        self.successor_gpu_case('overlaid')

    def test_successor_gpu_writable_source_refuses_runtime(self):
        self.successor_gpu_case('writable-source')

    def test_success_repeat_and_participant_new_round(self):
        stale_handler = self.make()
        result = self.make().replace()
        self.assertEqual(result.state.phase, 'runtime-ready')
        self.assertEqual(result.state.target_instance_id, NEW)
        self.assertEqual(self.retired(), [[OLD]])
        self.make().replace()
        self.assertEqual(self.retired(), [[OLD]])
        # A handler constructed on the old binding must refresh after locking.
        result = stale_handler.start('efa')
        self.assertEqual(result.state.round, 2)
        self.assertEqual(result.state.target_instance_id, NEW)
        record = stale_handler._replacement_record()
        self.assertEqual(record['evidence']['target_instance_id'], OLD)
        self.assertEqual(record['retired'], [OLD])
        self.assertEqual(set(record['checks']), {'0', '2', '3', '6'})
        print('LOCAL_REPLACEMENT_ARGV', json.dumps(self.executor.cloud_calls))

    def second_replacement_prepared(self, phase):
        self.executor.addresses.update({NEW: '10.0.0.2', THIRD: '10.0.0.3'})
        self.executor.states[THIRD] = 'running'
        self.config_body['assignments']['alias'] = copy.deepcopy(self.config_body['assignments']['table-1'])
        self.write_config()
        stale_alias = self.make('alias')
        self.assertEqual(self.make().replace().state.target_instance_id, NEW)
        first_record = self.make()._replacement_record()
        handler = self.make()
        started = handler.start('efa')
        self.assertEqual((started.state.round, started.state.target_instance_id), (2, NEW))
        self.executor.maintenance_failures['instance-id'] = (255, 'fixture: second round unreachable')
        with self.assertRaises(session.Refusal):
            handler.recover()
        self.executor.maintenance_failures.clear()
        self.assertEqual(handler.store.read().phase, 'replacement-required')
        # The bench permits NEW retirement only after its own new round failed.
        self.executor.successors[NEW] = THIRD
        original = handler._write_replacement
        def interrupt(record):
            original(record)
            if record['phase'] == phase:
                raise RuntimeError('fixture: durable second operation before send')
        with mock.patch.object(handler, '_write_replacement', side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, 'durable second operation'):
                handler.replace()
        self.assertEqual(self.retired(), [[OLD]])
        self.assertEqual(handler._replacement_record()['phase'], phase)
        archive = self.state_dir / f'replacement-history-{OLD}.json'
        self.assertEqual(json.loads(archive.read_text()), first_record)
        with self.assertRaisesRegex(session.Refusal, 'Another table owns'):
            self.make('alias').replace()
        with self.assertRaisesRegex(session.Refusal, 'being replaced'):
            stale_alias.start('efa')
        return first_record, stale_alias

    def fresh_replacement_handler(self):
        # Model a new process, not the previous connection's SSH destination/trust.
        self.executor.target_host = self.config_body['assignments']['table-1']['target_host']
        self.executor.known_hosts = ''
        return self.make()

    def second_retry_trace(self, label, refusal=None):
        print('SECOND_REPLACEMENT_TRACE', json.dumps({
            'case': label, 'refusal': refusal,
            'state': self.make().store.read().to_dict(),
            'record': self.make()._replacement_record(),
            'history': {p.name: json.loads(p.read_text()) for p in self.state_dir.glob('replacement-history-*.json')},
            'target_host': self.executor.target_host, 'known_hosts': self.executor.known_hosts,
            'node_id': self.executor.node_id, 'node_address': self.executor.node_address,
            'states': self.executor.states, 'cloud': self.executor.cloud_calls,
            'slurm': self.executor.slurm_calls, 'maintenance': self.executor.maintenance_calls}))

    def second_retry_success(self, phase, lost_ack=False):
        first, stale_alias = self.second_replacement_prepared(phase)
        observed_routes = []
        def observe(action):
            if action == 'describe-instances':
                observed_routes.append((self.executor.target_host, self.executor.known_hosts))
        self.executor.after_cloud = observe
        refusal = None
        try:
            if lost_ack:
                self.executor.terminate_ack_lost = True
                with self.assertRaisesRegex(session.Refusal, 'did not confirm completion'):
                    self.fresh_replacement_handler().replace()
                self.assertEqual(self.retired(), [[OLD], [NEW]])
                self.assertEqual(self.make()._replacement_record()['phase'], 'dispatching')
            result = self.fresh_replacement_handler().replace()
            self.assertEqual((result.state.phase, result.state.target_instance_id), ('runtime-ready', THIRD))
            self.assertEqual(observed_routes[0], ('10.0.0.2', str(self.state_dir / f'host-{NEW}')))
            record = self.make()._replacement_record()
            self.assertEqual(record['retired'], [OLD, NEW])
            self.assertEqual(record['evidence']['target_instance_id'], NEW)
            self.assertEqual(record['binding'], {'instance_id': THIRD, 'host': '10.0.0.3'})
            self.assertEqual(json.loads((self.state_dir / f'replacement-history-{OLD}.json').read_text()), first)
            self.assertEqual(set(record['checks']), {'0', '2', '3', '6'})
            self.assertEqual([p['InstanceIds'] for _, a, p in self.executor.cloud_calls if a == 'send-command'], [[NEW], [THIRD]])
            self.assertEqual(self.executor.maintenance_calls.count(['admit-replacement']), 2)
            self.fresh_replacement_handler().replace()
            self.assertEqual(self.retired(), [[OLD], [NEW]])
            self.assertEqual(stale_alias.start('efa').state.target_instance_id, THIRD)
            self.assertEqual(stale_alias.assignment['target_host'], '10.0.0.3')
            self.assertEqual(self.executor.known_hosts, str(self.state_dir / f'host-{THIRD}'))
        except Exception as error:
            refusal = str(error)
            raise
        finally:
            self.second_retry_trace(f'{phase}/lost_ack={lost_ack}', refusal)

    def test_second_replacement_prepared_retry_changed_ip(self):
        self.second_retry_success('prepared')

    def test_second_replacement_dispatching_retry_changed_ip(self):
        self.second_retry_success('dispatching')

    def test_second_replacement_lost_ack_never_retires_candidate(self):
        self.second_retry_success('dispatching', lost_ack=True)

    def test_second_replacement_retry_changed_observations_refuse(self):
        self.second_replacement_prepared('dispatching')
        changes = {'changed-address': ('node_address', '10.0.0.99'),
                   'changed-id': ('node_id', THIRD),
                   'foreign-reason': ('node_reason', 'administrator-maintenance'),
                   'busy-isolation': ('node_state', 'ALLOCATED+DRAIN'),
                   'lost-isolation': ('node_state', 'IDLE+CLOUD'),
                   'missing-state': ('node_state', None),
                   'unreadable-queue': ('queue_unreadable', True),
                   'nonempty-queue': ('jobs', ['99|another-user|other|RUNNING'])}
        for name, (field, value) in changes.items():
            with self.subTest(change=name):
                original = getattr(self.executor, field)
                setattr(self.executor, field, value)
                try:
                    with self.assertRaises(session.Refusal) as caught:
                        self.fresh_replacement_handler().replace()
                    self.second_retry_trace(name, str(caught.exception))
                    self.assertEqual(self.retired(), [[OLD]])
                finally:
                    setattr(self.executor, field, original)
        # Same operation must remain usable after those transient refusals clear.
        try:
            self.assertEqual(self.fresh_replacement_handler().replace().state.target_instance_id, THIRD)
            self.assertEqual(self.retired(), [[OLD], [NEW]])
        finally:
            self.second_retry_trace('unchanged-after-negative-controls')

    def test_second_replacement_predecessor_corruption_refuses(self):
        self.second_replacement_prepared('prepared')
        handler = self.make()
        original = handler._replacement_record()
        changes = {
            'candidate-as-predecessor': {'predecessor': {'instance_id': THIRD, 'host': '10.0.0.3'}},
            'wrong-predecessor-host': {'predecessor': {'instance_id': NEW, 'host': '10.0.0.99'}},
            'uncommitted-binding': {'binding': {'instance_id': NEW, 'host': '10.0.0.2'}},
            'missing-predecessor': {'predecessor': None}}
        for name, update in changes.items():
            with self.subTest(change=name):
                record = dict(original, **update)
                if name == 'missing-predecessor':
                    record.pop('predecessor')
                handler._write_replacement(record)
                before = len(self.executor.cloud_calls)
                try:
                    with self.assertRaises(session.Refusal) as caught:
                        self.fresh_replacement_handler().replace()
                    self.assertEqual(len(self.executor.cloud_calls), before)
                    self.assertEqual(self.retired(), [[OLD]])
                    print('PREDECESSOR_REFUSAL', json.dumps({'case': name, 'record': record,
                        'refusal': str(caught.exception), 'cloud': self.executor.cloud_calls}))
                finally:
                    handler._write_replacement(original)
        self.assertEqual(self.fresh_replacement_handler().replace().state.target_instance_id, THIRD)

    def test_legacy_completed_binding_can_prepare_new_operation(self):
        self.second_replacement_prepared('prepared')
        # A completed record from the preceding version has no predecessor field.
        # Its authenticated successor binding still authorizes the next operation.
        archive = self.state_dir / f'replacement-history-{OLD}.json'
        legacy = json.loads(archive.read_text())
        legacy.pop('predecessor')
        archive.unlink()
        self.make()._write_replacement(legacy)
        try:
            self.assertEqual(self.fresh_replacement_handler().replace().state.target_instance_id, THIRD)
            self.assertEqual(self.retired(), [[OLD], [NEW]])
            self.assertEqual(json.loads(archive.read_text()), legacy)
        finally:
            self.second_retry_trace('legacy-completed-binding')

    def test_no_argument_grammar(self):
        self.assertEqual(session.parse_participant_request('replace').verb, 'replace')
        for text in ('replace gpu-g7-2', 'replace --node gpu-g7-2', 'replace ' + NEW,
                     'replace; reboot', 'replace\nrecover'):
            self.assertIsNotNone(session.parse_participant_request(text).error)
        script = bench.LAB / '12.device-exercise.sh'
        run = subprocess.run(['bash', str(script), 'replace', NEW], capture_output=True)
        self.assertEqual(run.returncode, 2)

    def test_lost_ack_reconciles_exact_old_id(self):
        self.executor.terminate_ack_lost = True
        with self.assertRaises(session.Refusal):
            self.make().replace()
        self.assertEqual(self.make()._replacement_record()['phase'], 'dispatching')
        self.make().replace()
        self.assertEqual(self.retired(), [[OLD]])

    def test_crash_after_intent_before_dispatch(self):
        handler = self.make()
        original = handler._write_replacement
        def crash(record):
            original(record)
            if record['phase'] == 'dispatching':
                raise RuntimeError('fixture crash before API send')
        with mock.patch.object(handler, '_write_replacement', side_effect=crash):
            with self.assertRaises(RuntimeError):
                handler.replace()
        self.assertEqual(self.retired(), [])
        self.make().replace()
        self.assertEqual(self.retired(), [[OLD]])

    def test_bootstrap_failure_retry_does_not_retire_new_node(self):
        self.executor.bootstrap_status = 'Failed'
        with self.assertRaisesRegex(session.Refusal, 'bootstrap is incomplete'):
            self.make().replace()
        self.executor.bootstrap_status = 'Success'
        self.make().replace()
        self.assertEqual(self.retired(), [[OLD]])

    def test_health_failure_retry_does_not_retire_new_node(self):
        self.executor.failed_check = '2'
        with self.assertRaisesRegex(session.Refusal, 'check 2'):
            self.make().replace()
        self.assertFalse(any('State=RESUME' in call for call in self.executor.slurm_calls))
        self.executor.failed_check = ''
        self.make().replace()
        self.assertEqual(self.retired(), [[OLD]])

    def test_admission_drain_is_reconciled_but_foreign_prolog_error_is_not(self):
        self.executor.bootstrap_status = 'InProgress'
        with self.assertRaises(session.Refusal):
            self.make().replace()
        self.executor.bootstrap_status = 'Success'
        for reason in ('Prolog error', 'administrator maintenance', 'AIM344-admission-' + OLD):
            self.executor.node_state = 'IDLE+DRAIN+CLOUD'
            self.executor.node_reason = reason
            before = len(self.executor.slurm_calls)
            with self.subTest(reason=reason), self.assertRaisesRegex(session.Refusal, 'foreign isolation'):
                self.make().replace()
            self.assertFalse(any('State=RESUME' in c for c in self.executor.slurm_calls[before:]))
        self.executor.node_reason = 'AIM344-admission-' + NEW
        self.assertEqual(self.make().replace().state.phase, 'runtime-ready')
        self.assertEqual(self.retired(), [[OLD]])

    def test_pending_introduced_during_health_refuses_admission(self):
        self.executor.pending_after_checks = True
        with self.assertRaisesRegex(session.Refusal, 'reboot is pending'):
            self.make().replace()
        self.assertNotIn(['admit-replacement'], self.executor.maintenance_calls)
        self.assertFalse(any('State=RESUME' in call for call in self.executor.slurm_calls))

    def test_wrong_authenticated_bootstrap_identity_refuses(self):
        self.executor.bootstrap_report_bad = True
        with self.assertRaisesRegex(session.Refusal, 'Bootstrap identity'):
            self.make().replace()
        self.assertNotIn(['admit-replacement'], self.executor.maintenance_calls)

    def test_duplicate_scheduler_identity_refuses_before_retirement(self):
        self.executor.duplicate_identity = True
        with self.assertRaises(session.Refusal):
            self.make().replace()
        self.assertEqual(self.retired(), [])

    def test_foreign_jobs_and_capacity_refuse(self):
        self.executor.jobs = ['99|another-user|other|RUNNING']
        with self.assertRaises(session.Refusal):
            self.make().replace()
        self.executor.jobs = []
        self.executor.group_bad = True
        with self.assertRaises(session.Refusal):
            self.make().replace()
        self.assertEqual(self.retired(), [])

    def test_bad_membership_refuses(self):
        for field in ('CapacityReservationId', 'SubnetId', 'ImageId', 'InstanceType'):
            self.executor.member_bad = field
            with self.subTest(field=field), self.assertRaises(session.Refusal):
                self.make().replace()
        self.assertEqual(self.retired(), [])

    def test_coordinator_refuses(self):
        self.config_body['assignments']['table-1']['coordinator_node'] = 'gpu-g7-1'
        self.write_config()
        with self.assertRaisesRegex(session.Refusal, 'coordinator'):
            self.make().replace()
        self.assertEqual(self.retired(), [])

    def test_alias_cannot_steal_inflight_binding(self):
        other = copy.deepcopy(self.config_body['assignments']['table-1'])
        self.config_body['assignments']['alias'] = other
        self.write_config()
        self.executor.bootstrap_status = 'InProgress'
        with self.assertRaises(session.Refusal):
            self.make().replace()
        with self.assertRaises(session.Refusal):
            self.make('alias').replace()
        for verb in ('start', 'recover', 'expire'):
            with self.subTest(verb=verb), self.assertRaises(session.Refusal):
                method = getattr(self.make('alias'), verb)
                method('efa') if verb == 'start' else method()
        self.assertEqual(self.retired(), [[OLD]])

    def test_alias_refresh_and_negative_control_unchanged(self):
        self.config_body['assignments']['alias'] = copy.deepcopy(self.config_body['assignments']['table-1'])
        self.write_config()
        before = copy.deepcopy(self.config_body['assignments']['table-2'])
        self.make().replace()
        alias = self.make('alias')
        self.assertEqual(alias.start('efa').state.target_instance_id, NEW)
        self.assertEqual(self.config_body['assignments']['table-2'], before)

    def test_binding_commit_crash_reconciles_state(self):
        handler = self.make()
        original = handler.store.write
        def crash(state):
            if state.phase == 'runtime-ready':
                raise RuntimeError('fixture crash at assignment commit')
            return original(state)
        with mock.patch.object(handler.store, 'write', side_effect=crash):
            with self.assertRaises(RuntimeError):
                handler.replace()
        result = self.make().replace()
        self.assertEqual(result.state.phase, 'runtime-ready')
        self.assertEqual(result.state.target_instance_id, NEW)
        self.assertEqual(self.retired(), [[OLD]])

    def test_pair_lock_refuses_parallel_replace(self):
        handler = self.make()
        with handler.store.exclusive_pair():
            with self.assertRaises(session.Refusal):
                self.make().replace()
        self.assertEqual(self.retired(), [])

    def test_same_id_new_boot_not_replacement(self):
        self.executor.states[OLD] = 'terminated'
        self.executor.boot_id = 'bbbbbbbb-0000-0000-0000-000000000002'
        with self.assertRaisesRegex(session.Refusal, 'new assigned identity'):
            self.make().replace()
        self.assertEqual(self.retired(), [])

    def test_old_response_after_new_candidate_refuses(self):
        self.executor.bootstrap_status = 'InProgress'
        with self.assertRaises(session.Refusal):
            self.make().replace()
        self.executor.node_id = OLD
        with self.assertRaisesRegex(session.Refusal, 'identity changed'):
            self.make().replace()
        self.assertEqual(self.retired(), [[OLD]])


    def test_recovery_of_shared_held_node_refuses_before_mutation(self):
        self.config_body['assignments']['alias'] = copy.deepcopy(self.config_body['assignments']['table-1'])
        self.write_config()
        owner = self.make()
        alias = self.make('alias')
        alias.store.write(owner.store.read())
        before = len(self.executor.maintenance_calls)
        with self.assertRaisesRegex(session.Refusal, 'Another table'):
            alias.recover()
        self.assertEqual(len(self.executor.maintenance_calls), before)

    def test_unreachable_recovery_becomes_replacement_required(self):
        handler = self.make()
        state = handler.store.read()
        state.phase = 'fault-applied'
        handler.store.write(state)
        self.executor.maintenance_failures['instance-id'] = (255, 'fixture unreachable')
        with self.assertRaises(session.Refusal):
            handler.recover()
        self.assertEqual(handler.store.read().phase, 'replacement-required')
        self.executor.maintenance_failures.clear()
        self.assertEqual(handler.replace().state.target_instance_id, NEW)

    def test_pending_old_reboot_is_retired_without_second_slurm_reboot(self):
        self.executor.node_state += '+REBOOT_ISSUED'
        self.make().replace()
        self.assertEqual(self.retired(), [[OLD]])
        self.assertFalse(any(c[1:2] == ['reboot'] for c in self.executor.slurm_calls))

    def test_fsync_failure_blocks_dispatch(self):
        handler = self.make()
        with mock.patch.object(session, '_sync_directory', side_effect=OSError('fixture EIO')):
            with self.assertRaises(OSError):
                handler.replace()
        self.assertEqual(self.retired(), [])
        handler.replace()
        self.assertEqual(self.retired(), [[OLD]])

    def test_malformed_binding_does_not_retarget_retirement(self):
        self.executor.terminate_ack_lost = True
        handler = self.make()
        with self.assertRaises(session.Refusal):
            handler.replace()
        record = handler._replacement_record()
        record['old_id'] = NEW
        handler._write_replacement(record)
        with self.assertRaisesRegex(session.Refusal, 'Unusable replacement'):
            self.make().replace()
        self.assertEqual(self.retired(), [[OLD]])

    def test_missing_host_trust_refuses_before_ssh(self):
        original = self.executor.run_cloud
        def cloud(region, service, action, payload):
            result = original(region, service, action, payload)
            if action == 'get-command-invocation':
                body = json.loads(result.stdout)
                report = json.loads(body['StandardOutputContent'])
                report.pop('host_key')
                body['StandardOutputContent'] = json.dumps(report)
                return session.Completed(0, json.dumps(body), '')
            return result
        with mock.patch.object(self.executor, 'run_cloud', side_effect=cloud):
            with self.assertRaisesRegex(session.Refusal, 'host key'):
                self.make().replace()
        self.assertNotIn(['admit-replacement'], self.executor.maintenance_calls)

    def test_missing_prolog_blocks_before_retirement(self):
        original = self.executor.run_cloud
        def cloud(region, service, action, payload):
            if action == 'get-cluster':
                return session.Completed(0, '{"cluster":{"status":"ACTIVE"}}', '')
            return original(region, service, action, payload)
        with mock.patch.object(self.executor, 'run_cloud', side_effect=cloud):
            with self.assertRaisesRegex(session.Refusal, 'Prolog'):
                self.make().replace()
        self.assertEqual(self.retired(), [])

    def test_instance_change_during_admission_prevents_resume(self):
        original = self.executor.run_maintenance
        def maintenance(argv, timeout=None):
            result = original(argv, timeout)
            if argv == ['admit-replacement']:
                self.executor.node_id = PEER
            return result
        with mock.patch.object(self.executor, 'run_maintenance', side_effect=maintenance):
            with self.assertRaisesRegex(session.Refusal, 'identity changed before RESUME'):
                self.make().replace()
        self.assertFalse(any('State=RESUME' in c for c in self.executor.slurm_calls))

    def test_resume_registration_observation_and_fail_closed_retry(self):
        handler = self.no_roll()
        original = self.executor.run_slurm
        mode = ['stuck']
        resumed = [False]
        reads = [0]
        def slurm(argv, timeout=None):
            result = original(argv, timeout)
            if 'State=RESUME' in argv:
                resumed[0] = True
                reads[0] = 0
            elif 'State=DRAIN' in argv:
                resumed[0] = False
            elif argv[1:3] == ['show', 'node'] and resumed[0]:
                reads[0] += 1
                text = result.stdout
                if mode[0] == 'stuck' or reads[0] == 1:
                    text = text.replace('State=IDLE', 'State=IDLE+NOT_RESPONDING')
                elif mode[0] == 'address':
                    text = text.replace('NodeAddr=10.0.0.1', 'NodeAddr=10.0.0.99')
                elif mode[0] == 'reason':
                    text = text.replace('Reason=' + self.executor.node_reason,
                                        'Reason=administrator')
                elif mode[0] == 'reboot':
                    text = text.replace('State=IDLE', 'State=IDLE+REBOOT_ISSUED')
                elif mode[0] == 'boot':
                    self.executor.boot_id = 'bbbbbbbb-0000-0000-0000-000000000002'
                elif mode[0] == 'queue':
                    self.executor.jobs = ['99|other|unrelated|RUNNING']
                return session.Completed(result.returncode, text, result.stderr)
            return result
        boot = self.executor.boot_id
        with mock.patch.object(self.executor, 'run_slurm', side_effect=slurm), \
                mock.patch.object(session.time, 'sleep') as sleep:
            for failure in ('stuck', 'address', 'reason', 'reboot', 'boot', 'queue'):
                mode[0] = failure
                resumed[0] = False
                self.executor.boot_id = boot
                self.executor.jobs = []
                before = sum('State=RESUME' in c for c in self.executor.slurm_calls)
                with self.subTest(failure=failure), self.assertRaises(session.Refusal):
                    handler.replace()
                self.assertEqual(sum('State=RESUME' in c for c in self.executor.slurm_calls), before + 1)
                self.assertTrue(self.executor.reservation)
                self.assertNotEqual(handler._replacement_record()['phase'], 'complete')
                self.assertEqual(self.retired(), [[OLD]])
                if failure == 'stuck':
                    self.assertEqual(reads[0], 30)
                    self.assertEqual(sleep.call_count, 29)
            mode[0] = 'ready'
            resumed[0] = False
            self.executor.boot_id = boot
            self.executor.jobs = []
            self.assertEqual(handler.replace().state.phase, 'runtime-ready')
            self.assertFalse(self.executor.reservation)
            self.assertEqual(self.retired(), [[OLD]])

    def test_warn_and_wrong_producer_refuse(self):
        original = self.executor.run_maintenance
        for output in ('[WARN] 0-nvidia-smi: fixture warning\n',
                       '[PASS] 2-efa-enumeration: wrong producer\n'):
            def maintenance(argv, timeout=None):
                if argv == ['collect', '0'] and self.executor.node_id == NEW:
                    return session.Completed(0, output, '')
                return original(argv, timeout)
            with mock.patch.object(self.executor, 'run_maintenance', side_effect=maintenance):
                with self.assertRaisesRegex(session.Refusal, 'check 0'):
                    self.make().replace()
        self.assertEqual(self.retired(), [[OLD]])


class BootstrapFiles(unittest.TestCase):
    def test_bootstrap_rejects_overlaid_source_on_retry(self):
        def retry(bootstrap, node_path, manifest, report):
            original = bootstrap.run
            def command(argv):
                if argv[0] == '/usr/bin/findmnt':
                    return '/opt/aim344'
                return original(argv)
            with mock.patch.object(bootstrap, 'run', side_effect=command):
                with self.assertRaisesRegex(RuntimeError, 'root filesystem'):
                    bootstrap.bootstrap(manifest)
        self.bootstrap_sandbox(retry)

    def test_legacy_nvme_source_validation_preserved(self):
        maintenance = self.load('maintenance.py')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nvme = root / 'nvme'
            source = nvme / 'aim344'
            source.mkdir(parents=True)
            stage = root / 'aim344'
            stage.mkdir()
            cfg = {'stage_dir': str(stage), 'staging_source_root': str(nvme),
                   'staging_fs_uuid': 'expected-uuid', 'slurm_bin': '/slurm'}
            for name in ('aim344.sqsh', 'nccl-baseline.sqsh'):
                (source / name).write_bytes(b'legacy fixture image')
            real_stat = os.stat
            mounted = [False]
            uuid = ['expected-uuid']
            calls = []
            def command(argv, **kwargs):
                calls.append(argv)
                if argv[0] == '/usr/bin/mount':
                    mounted[0] = True
                return subprocess.CompletedProcess(argv, 0, uuid[0], '')
            def observed(path, *args, **kwargs):
                return real_stat(source if str(path) == str(stage) and mounted[0] else path,
                                 *args, **kwargs)
            with mock.patch.object(Path, 'is_mount', lambda p: p == nvme or (p == stage and mounted[0])), \
                 mock.patch.object(maintenance.subprocess, 'run', side_effect=command), \
                 mock.patch.object(os, 'stat', side_effect=observed):
                self.assertEqual(maintenance.remount_staging(cfg), 0)
                self.assertEqual(sum(c[0] == '/usr/bin/mount' for c in calls), 1)
                uuid[0] = 'wrong-source'
                with self.assertRaisesRegex(maintenance.Refusal, 'UUID'):
                    maintenance.remount_staging(cfg)
                uuid[0] = 'expected-uuid'
                (source / 'aim344.sqsh').unlink()
                with self.assertRaisesRegex(maintenance.Refusal, 'absent'):
                    maintenance.remount_staging(cfg)
                self.assertEqual(sum(c[0] == '/usr/bin/mount' for c in calls), 1)
            with mock.patch.object(Path, 'is_mount', return_value=False):
                with self.assertRaisesRegex(maintenance.Refusal, 'not mounted'):
                    maintenance.remount_staging(cfg)

    def test_maintenance_admission_and_evidence_actual_dispatch(self):
        import contextlib
        maintenance = self.load('maintenance.py')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = root / 'records'
            records.mkdir()
            receipt = root / 'ready.json'
            receipt.write_text(json.dumps({'ready': True, 'instance_id': NEW}))
            real_path = Path
            def path(value):
                return receipt if value == '/var/lib/aim344-replacement/ready.json' else real_path(value)
            with mock.patch.object(maintenance, 'Path', side_effect=path), \
                 mock.patch.object(maintenance, 'RECORD_DIR', records), \
                 mock.patch.object(maintenance.os, 'geteuid', return_value=0), \
                 mock.patch.object(maintenance, 'load_config', return_value={}), \
                 mock.patch.object(maintenance, 'instance_identity', return_value=NEW), \
                 mock.patch.object(maintenance.sys, 'argv', ['helper', 'admit-replacement']):
                self.assertEqual(maintenance.main(), 0)
                admitted = json.loads((records / 'replacement-admitted.json').read_text())
                self.assertEqual(admitted['value'], NEW)
                receipt.write_text(json.dumps({'ready': True, 'instance_id': OLD}))
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertNotEqual(maintenance.main(), 0)
                self.assertEqual(json.loads((records / 'replacement-admitted.json').read_text()), admitted)
                (records / 'selected-persistence-mode.txt').write_text('retained original')
                with mock.patch.object(maintenance.sys, 'argv', ['helper', 'replacement-evidence']), \
                     contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(maintenance.main(), 0)
                capture = json.loads(output.getvalue())
                self.assertEqual(capture['records']['selected-persistence-mode.txt'], 'retained original')
                self.assertIsNone(capture['records']['native-dcgm-state.txt'])
                for verb in ('replacement-evidence', 'admit-replacement'):
                    with self.assertRaises(maintenance.Refusal):
                        maintenance.parse([verb, NEW])

    def test_enroot_shared_parent_validation_and_retry(self):
        from types import SimpleNamespace
        bootstrap = self.load('replacement-bootstrap.py')
        config = SimpleNamespace(read_text=lambda: 'ENROOT_RUNTIME_PATH /tmp/enroot/user-$(id -u)\n'
                                 'ENROOT_CACHE_PATH /tmp/enroot/cache\n')
        real_path = Path
        def path(value):
            return config if value == '/etc/enroot/enroot.conf' else real_path(value)
        with mock.patch.object(bootstrap, 'Path', side_effect=path), \
             mock.patch.object(bootstrap.os, 'open', side_effect=range(10, 18)), \
             mock.patch.object(bootstrap.os, 'fstat', return_value=SimpleNamespace(st_uid=0, st_mode=0o41777)), \
             mock.patch.object(bootstrap.os, 'mkdir', side_effect=[None] * 3 + [FileExistsError] * 3), \
             mock.patch.object(bootstrap.os, 'fchmod') as chmod, \
             mock.patch.object(bootstrap.os, 'close'):
            bootstrap.prepare_enroot()
            bootstrap.prepare_enroot()
            self.assertEqual(chmod.call_args_list,
                             [mock.call(fd, 0o1777) for fd in (11, 12, 13, 15, 16, 17)])
        config.read_text = lambda: 'ENROOT_RUNTIME_PATH /home/user/cache\n'
        with mock.patch.object(bootstrap, 'Path', side_effect=path):
            with self.assertRaisesRegex(RuntimeError, 'Unsupported shared enroot'):
                bootstrap.prepare_enroot()

    def test_runtime_fixture_initializer_real_retry_and_corruption_refusal(self):
        import re
        script = (bench.LAB / 'facilitator/restore-runtime.sh').read_text()
        match = re.search(r"<<'PYFIXTURE'\n(.*?)\nPYFIXTURE", script, re.S)
        assert match
        code = match[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            from types import SimpleNamespace
            with mock.patch.dict('sys.modules', {'pathlib': SimpleNamespace(Path=lambda value: root)}):
                exec(compile(code, 'runtime-fixture', 'exec'), {})
                self.assertEqual(json.loads((root / 'last.json').read_text()), {'next_step': 0})
                (root / 'last.json').write_text('{"next_step":12}')
                exec(compile(code, 'runtime-fixture', 'exec'), {})
                self.assertEqual(json.loads((root / 'last.json').read_text()), {'next_step': 12})
                (root / '.aim344-fixture').write_text('foreign')
                with self.assertRaisesRegex(SystemExit, 'Unexpected existing'):
                    exec(compile(code, 'runtime-fixture', 'exec'), {})

    def test_account_creation_retry_collision_and_privilege_refusal(self):
        from types import SimpleNamespace
        bootstrap = self.load('replacement-bootstrap.py')
        users, groups, calls = {}, {}, []
        entry = {'name': 'aim344-t1', 'uid': 1001, 'gid': 1001}
        def command(argv):
            calls.append(argv)
            if argv[0].endswith('groupadd'):
                groups[1001] = SimpleNamespace(gr_name='aim344-t1', gr_mem=[])
            if argv[0].endswith('useradd'):
                users['aim344-t1'] = SimpleNamespace(pw_uid=1001, pw_gid=1001)
            return ''
        with mock.patch.object(bootstrap.grp, 'getgrgid', side_effect=lambda gid: groups[gid]), \
             mock.patch.object(bootstrap.pwd, 'getpwnam', side_effect=lambda name: users[name]), \
             mock.patch.object(bootstrap.pwd, 'getpwuid', side_effect=KeyError), \
             mock.patch.object(bootstrap.grp, 'getgrall', side_effect=lambda: list(groups.values())), \
             mock.patch.object(bootstrap, 'run', side_effect=command):
            bootstrap.account(entry)
            bootstrap.account(entry)
            self.assertEqual(len(calls), 2)
            users['aim344-t1'].pw_uid = 1002
            with self.assertRaisesRegex(RuntimeError, 'account mismatch'):
                bootstrap.account(entry)
            users['aim344-t1'].pw_uid = 1001
            groups[1002] = SimpleNamespace(gr_name='docker', gr_mem=['aim344-t1'])
            with self.assertRaisesRegex(RuntimeError, 'privileged group'):
                bootstrap.account(entry)
            groups[1001].gr_name = 'foreign'
            with self.assertRaisesRegex(RuntimeError, 'gid collision'):
                bootstrap.account(entry)

    def test_inventory_uses_new_sysfs_and_rejects_management_efa(self):
        bootstrap = self.load('replacement-bootstrap.py')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def mapped(value):
                return root / str(value).lstrip('/')
            pci = mapped('/sys/bus/pci/devices')
            for bdf, driver in (('0000:00:01.0', 'ena'), ('0000:00:02.0', 'nvidia'),
                                ('0000:00:03.0', 'efa')):
                device = pci / bdf
                device.mkdir(parents=True)
                (device / 'driver').symlink_to('/fixture/drivers/' + driver)
                (device / 'vendor').write_text('0x1234')
                (device / 'device').write_text('0x5678')
            interface = mapped('/sys/class/net/eth0')
            interface.mkdir(parents=True)
            (interface / 'device').symlink_to(pci / '0000:00:01.0')
            efa = pci / '0000:00:03.0'
            (efa / 'infiniband/rdma-new').mkdir(parents=True)
            def command(argv):
                return ('[{"dev":"eth0"}]' if argv[0].endswith('/ip') else
                        'GPU-new-identity, 00000000:00:02.0')
            with mock.patch.object(bootstrap, 'Path', side_effect=mapped), \
                 mock.patch.object(bootstrap, 'run', side_effect=command):
                cfg = bootstrap.inventory({'node': 'gpu-g7-1', 'slurm_bin': '/slurm',
                    'participants': [{'name': 'aim344-t1'}]}, NEW)
                self.assertEqual(cfg['instance_id'], NEW)
                self.assertEqual(cfg['gpu_bdf'], '0000:00:02.0')
                self.assertEqual(cfg['efa_rdma_device'], 'rdma-new')
                (efa / 'net/eth1').mkdir(parents=True)
                with self.assertRaisesRegex(RuntimeError, 'No separate EFA'):
                    bootstrap.inventory({'node': 'gpu-g7-1'}, NEW)

    def load(self, filename):
        spec = importlib.util.spec_from_file_location('replacement_test_module', bench.LAB / 'facilitator' / filename)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_partial_tree_retry_and_symlink_refusal(self):
        bootstrap = self.load('replacement-bootstrap.py')
        bootstrap.ROOT_UID = os.getuid()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, dest = root / 'source', root / 'dest'
            source.mkdir()
            (source / 'first').write_text('pinned')
            bootstrap.install_tree(source, dest)
            (source / 'second').write_text('more pinned')
            bootstrap.install_tree(source, dest)
            self.assertEqual((dest / 'second').read_text(), 'more pinned')
            (dest / 'first').unlink()
            (dest / 'first').symlink_to(root / 'outside')
            with self.assertRaises(RuntimeError):
                bootstrap.install_tree(source, dest)
            self.assertFalse((root / 'outside').exists())

    def test_generator_real_artifacts_and_syntax(self):
        generator = self.load('prepare-replacement.py')
        manifest = {'node': 'gpu-g7-1', 'region': 'eu-south-2',
                    'slurm_bin': '/opt/aws/pcs/scheduler/slurm-25.05/bin',
                    'participants': [{'name': 'aim344-t1', 'uid': 1001, 'gid': 1001}],
                    'protected_instance_ids': [PEER], 'release_sha256': 'a' * 64,
                    'maintenance_public_key': 'ssh-ed25519 AAAATestPublicOnly',
                    'objects': {name: {'bucket': 'fixture-bucket', 'key': name, 'sha256': 'a' * 64}
                                for name in ('companion.tgz', 'healthcheck-pinned.tgz', 'aim344.sqsh', 'nccl-baseline.sqsh')}}
        site = {'account': '123456789012', 'coordinator_instance_id': PEER,
                'coordinator_role_id': 'AROA' + 'A' * 17,
                'cluster_id': 'pcs_cluster', 'group_id': 'pcs_group'}
        bootstrap = (bench.LAB / 'facilitator/replacement-bootstrap.py').read_text()
        artifacts = generator.generate(manifest, site, bootstrap)
        self.assertLessEqual(len(artifacts['replacement-user-data.mime'].encode()), 16384)
        self.assertFalse(artifacts['assignment-replacement.json']['enabled'])
        self.assertEqual(artifacts['bootstrap-document.json']['parameters'], {})
        statements = artifacts['coordinator-policy.json']['Statement']
        group_read = [s for s in statements if s['Action'] == ['pcs:GetComputeNodeGroup']]
        self.assertEqual(len(group_read), 1)
        self.assertEqual(group_read[0]['Resource'], [
            'arn:aws:pcs:eu-south-2:123456789012:cluster/pcs_cluster',
            'arn:aws:pcs:eu-south-2:123456789012:cluster/pcs_cluster/computenodegroup/pcs_group'])
        self.assertEqual(group_read[0]['Condition'], {'ArnEquals': {
            'ec2:SourceInstanceARN': 'arn:aws:ec2:eu-south-2:123456789012:instance/' + PEER}})
        self.assertTrue(any(s['Effect'] == 'Deny' and PEER in str(s['Resource']) for s in statements))
        self.assertNotIn('iam:PassRole', json.dumps(statements))
        self.assertNotIn('pcs:Update', json.dumps(statements))
        forwarded = [s for s in statements if s.get('Condition', {}).get('Bool')]
        self.assertEqual(len(forwarded), 2)
        for statement in forwarded:
            self.assertEqual(statement['Action'], 'ssm:SendCommand')
            self.assertEqual(statement['Condition']['Bool'], {'aws:ViaAWSService': 'true'})
            self.assertEqual(statement['Condition']['StringEquals']['aws:userid'],
                             site['coordinator_role_id'] + ':' + PEER)
        self.assertIn('ssm:resourceTag/aws:pcs:compute-node-group-id',
                      forwarded[1]['Condition']['StringEquals'])
        with tempfile.TemporaryDirectory() as temporary:
            hook = Path(temporary) / 'hook.sh'
            hook.write_text(artifacts['replacement-boothook.sh'])
            result = subprocess.run(['bash', '-n', str(hook)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            # Actually decode every generated payload and compile embedded Python.
            import base64
            import gzip
            import re
            payloads = re.findall(r"printf '%s' '([^']+)'", hook.read_text())
            self.assertEqual(len(payloads), 6)
            decoded = [gzip.decompress(base64.b64decode(p)).decode() for p in payloads]
            self.assertIn('DefaultDependencies=no', decoded[5])
            self.assertIn('Before=sysinit.target network-pre.target', decoded[5])
            self.assertIn('RequiredBy=sysinit.target', decoded[5])
            for family in ('iptables', 'ip6tables'):
                for chain in ('OUTPUT', 'FORWARD'):
                    self.assertIn(f'{family} -C {chain}', decoded[4])
            # Execute only the generated shell preamble, with a failing command
            # instead of privileged writes. It must not return to cloud-init.
            preamble = '\n'.join(hook.read_text().splitlines()[:4]) + '\nfalse\nprintf UNREACHABLE\n'
            held = subprocess.Popen(['bash'], stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, start_new_session=True)
            try:
                with self.assertRaises(subprocess.TimeoutExpired):
                    held.communicate(preamble, timeout=0.5)
            finally:
                import signal
                os.killpg(held.pid, signal.SIGKILL)
                stdout, stderr = held.communicate()
            self.assertNotIn('UNREACHABLE', stdout)
            self.assertIn('boot held before user access', stderr)
            self.assertEqual(decoded[1], bootstrap)
            compile(decoded[1], 'generated-bootstrap', 'exec')
            compile(decoded[2], 'generated-admission', 'exec')
            from types import SimpleNamespace
            node = Path(temporary) / 'admission-node'
            node.mkdir()
            def mapped(value):
                return node / str(value).lstrip('/')
            inputs = {'/etc/aim344-replacement.json': json.dumps(manifest),
                      '/sys/devices/virtual/dmi/id/board_asset_tag': NEW}
            for name, text in inputs.items():
                path = mapped(name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            # Execute the generated gate, not just its syntax, with filesystem
            # reads redirected to this node fixture. No scheduler is involved.
            def scheduler(argv, **kwargs):
                return SimpleNamespace(stdout=f'NodeName=gpu-g7-1 InstanceId={NEW} State=IDLE+CLOUD')
            with mock.patch.dict('sys.modules', {'pathlib': SimpleNamespace(Path=mapped)}), \
                 mock.patch('subprocess.run', side_effect=scheduler) as commands:
                with self.assertRaises(SystemExit):
                    exec(compile(decoded[2], 'admission', 'exec'), {})
                ready = mapped('/var/lib/aim344-replacement/ready.json')
                ready.parent.mkdir(parents=True)
                ready.write_text(json.dumps({'instance_id': NEW, 'ready': True}))
                with self.assertRaises(SystemExit):
                    exec(compile(decoded[2], 'admission', 'exec'), {})
                admitted = mapped('/var/lib/aim344-device-recovery/replacement-admitted.json')
                admitted.parent.mkdir(parents=True)
                admitted.write_text(json.dumps({'value': OLD}))
                with self.assertRaises(SystemExit):
                    exec(compile(decoded[2], 'admission', 'exec'), {})
                admitted.write_text(json.dumps({'value': NEW}))
                exec(compile(decoded[2], 'admission', 'exec'), {})
                self.assertTrue(any('Reason=AIM344-admission-' + NEW in c.args[0]
                                    for c in commands.call_args_list))
                admitted.unlink()
                for observed in (f'NodeName=gpu-g7-1 InstanceId={NEW} State=IDLE+DRAIN Reason=foreign',
                                 f'NodeName=gpu-g7-1 InstanceId={OLD} State=IDLE+CLOUD'):
                    commands.reset_mock()
                    commands.side_effect = lambda *a, **k: SimpleNamespace(stdout=observed)
                    with self.assertRaises(SystemExit):
                        exec(compile(decoded[2], 'admission', 'exec'), {})
                    self.assertFalse(any('update' in c.args[0] for c in commands.call_args_list))
        print('LOCAL_GENERATED_MIME_BYTES', len(artifacts['replacement-user-data.mime'].encode()))
        # Execute the fixed installer with real local files and redirected
        # system/service calls; this is not a live SSM or root installation.
        site['provisioning_mode'] = 'ssm-install'
        sandbox_bootstrap = bootstrap.replace('ROOT_UID = 0', f'ROOT_UID = {os.getuid()}')
        installed = generator.generate(manifest, site, sandbox_bootstrap)
        self.assertNotIn('replacement-user-data.mime', installed)
        doc = installed['bootstrap-document.json']
        self.assertEqual(doc['parameters'], {})
        self.assertLess(len(json.dumps(doc).encode()), 65536)
        command = doc['mainSteps'][0]['inputs']['runCommand'][0]
        installer = command.split("<<'AIM344_FIXED_INSTALL'\n", 1)[1].rsplit('AIM344_FIXED_INSTALL', 1)[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            root.chmod(0o755)
            def mapped_install(value):
                p = Path(value)
                return p if p.is_relative_to(root) else root / str(p).lstrip('/')
            dmi = mapped_install('/sys/devices/virtual/dmi/id/board_asset_tag')
            dmi.parent.mkdir(parents=True)
            dmi.write_text(PEER)
            def commands(argv, **kwargs):
                return subprocess.CompletedProcess(argv, 0,
                    f'NodeName=gpu-g7-1 InstanceId={NEW} State=IDLE+DRAIN', '')
            with mock.patch.dict('sys.modules', {'pathlib': SimpleNamespace(Path=mapped_install)}), \
                 mock.patch('os.geteuid', return_value=0), \
                 mock.patch('subprocess.run', side_effect=commands) as calls, \
                 mock.patch('fcntl.flock'), \
                 mock.patch('os.execv') as execute:
                with self.assertRaisesRegex(RuntimeError, 'Protected'):
                    exec(compile(installer, 'fixed-installer', 'exec'), {})
                self.assertFalse(mapped_install('/etc/aim344-replacement.json').exists())
                calls.assert_not_called()
                dmi.write_text(NEW)
                exec(compile(installer, 'fixed-installer', 'exec'), {})
                self.assertEqual(mapped_install('/etc/aim344-replacement.json').read_text(),
                                 json.dumps(manifest, sort_keys=True))
                self.assertEqual(mapped_install('/usr/local/sbin/aim344-replacement-bootstrap').read_text(),
                                 sandbox_bootstrap)
                firewall = mapped_install('/usr/local/sbin/aim344-replacement-imds').read_text()
                for family in ('iptables', 'ip6tables'):
                    for chain in ('OUTPUT', 'FORWARD'):
                        self.assertIn(f'{family} -C {chain}', firewall)
                self.assertIn('RequiredBy=slurmd.service', mapped_install(
                    '/etc/systemd/system/aim344-replacement-imds.service').read_text())
                execute.assert_called_once()
                # Repeated lost-ack installation retains byte-identical payload.
                exec(compile(installer, 'fixed-installer', 'exec'), {})
                self.assertEqual(execute.call_count, 2)

    def test_full_bootstrap_download_failure_hash_failure_then_retry(self):
        self.bootstrap_sandbox()

    def bootstrap_sandbox(self, after_bootstrap=None):
        bootstrap = self.load('replacement-bootstrap.py')
        previous_umask = os.umask(0o022)
        self.addCleanup(os.umask, previous_umask)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sandbox = root / 'node'
            sandbox.mkdir()
            (sandbox / 'root').mkdir(mode=0o700)
            def node_path(value):
                path = Path(value)
                if path.is_relative_to(root):
                    return path
                return sandbox / str(path).lstrip('/') if path.is_absolute() else path
            for name, text in (('/sys/devices/virtual/dmi/id/board_asset_tag', NEW),
                               ('/etc/ssh/ssh_host_ed25519_key.pub', 'ssh-ed25519 AAAATestHost fixture')):
                path = node_path(name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            objects = root / 'objects'
            pci = node_path('/sys/bus/pci/devices')
            for bdf, driver in (('0000:00:01.0', 'ena'), ('0000:00:02.0', 'nvidia'),
                                ('0000:00:03.0', 'efa')):
                device = pci / bdf
                device.mkdir(parents=True)
                driver_path = node_path('/sys/bus/pci/drivers/' + driver)
                driver_path.mkdir(parents=True)
                (device / 'driver').symlink_to(driver_path)
                (device / 'vendor').write_text('0x1234')
                (device / 'device').write_text('0x5678')
            interface = node_path('/sys/class/net/eth0')
            interface.mkdir(parents=True)
            (interface / 'device').symlink_to(pci / '0000:00:01.0')
            (pci / '0000:00:03.0/infiniband/rdma-new').mkdir(parents=True)
            objects.mkdir()
            with tarfile.open(objects / 'companion.tgz', 'w:gz') as tar:
                for name in ('facilitator/maintenance.py', 'facilitator/device-fault.sh',
                             'facilitator/restore-runtime.sh', 'prejob-prolog.sh',
                             '13.verify-after-recovery.sbatch'):
                    tar.add(bench.LAB / name, arcname=name)
            with tarfile.open(objects / 'healthcheck-pinned.tgz', 'w:gz') as tar:
                info = tarfile.TarInfo('validation/gpu-cluster-healthcheck/gpu-healthcheck.sh')
                content = b'#!/bin/bash\n# Explicit local fixture, not a health implementation\nexit 0\n'
                info.size = len(content)
                info.mode = 0o755
                tar.addfile(info, io.BytesIO(content))
            for image in ('aim344.sqsh', 'nccl-baseline.sqsh'):
                (objects / image).write_bytes(b'local mock image, not executable hardware evidence')
            manifest = {'node': 'gpu-g7-1', 'region': 'eu-south-2',
                        'slurm_bin': '/opt/aws/pcs/scheduler/slurm-25.05/bin',
                        'participants': [{'name': 'aim344-t1', 'uid': 1001, 'gid': 1001}],
                        'protected_instance_ids': [PEER],
                        'maintenance_public_key': 'ssh-ed25519 AAAATestClient',
                        'objects': {p.name: {'bucket': 'local-fixture', 'key': p.name,
                                            'sha256': bootstrap.digest(p),
                                            'version_id': 'fixture version'} for p in objects.iterdir()}}
            manifest['release_sha256'] = manifest['objects']['companion.tgz']['sha256']
            bootstrap.ROOT = node_path('/var/lib/aim344-replacement')
            bootstrap.ROOT.mkdir(parents=True)
            bootstrap.ROOT_UID = os.getuid()
            calls = []
            failures = {'download': True, 'hash': False}
            def run(argv):
                calls.append(argv)
                if argv[0] == '/usr/bin/findmnt':
                    return '/'
                if argv[0] == '/usr/sbin/ip':
                    return '[{"dev":"eth0"}]'
                if argv[0] == '/usr/bin/nvidia-smi':
                    return 'GPU-aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee, 00000000:00:02.0'
                if argv[0].endswith('/scontrol'):
                    return f'NodeName=gpu-g7-1 InstanceId={NEW} State=IDLE+DRAIN'
                if argv[0] == '/usr/local/bin/aws':
                    if failures['download']:
                        raise subprocess.CalledProcessError(1, argv, stderr='fixture unavailable')
                    self.assertNotIn('--cli-input-json', argv)
                    self.assertEqual(argv[argv.index('--bucket') + 1], 'local-fixture')
                    self.assertEqual(argv[argv.index('--version-id') + 1], 'fixture version')
                    destination = Path(argv[-1])
                    data = (objects / argv[argv.index('--key') + 1]).read_bytes()
                    destination.write_bytes(b'corrupt fixture' if failures['hash'] else data)
                return ''
            with mock.patch.object(bootstrap, 'Path', side_effect=node_path), \
                 mock.patch.object(bootstrap, 'run', side_effect=run), \
                 mock.patch.object(bootstrap, 'account') as account, \
                 mock.patch.object(bootstrap, 'prepare_enroot'):
                with self.assertRaises(subprocess.CalledProcessError):
                    bootstrap.bootstrap(manifest)
                failures['download'] = False
                failures['hash'] = True
                with self.assertRaisesRegex(RuntimeError, 'hash mismatch'):
                    bootstrap.bootstrap(manifest)
                failures['hash'] = False
                report = bootstrap.bootstrap(manifest)
                self.assertTrue(report['ready'])
                self.assertEqual(report['instance_id'], NEW)
                self.assertTrue(account.called)
                self.assertTrue(any('/var/lib/aim344-device-recovery/restore-runtime.sh' in c for c in calls))
                self.assertTrue(node_path('/opt/aim344/device-recovery/13.verify-after-recovery.sbatch').exists())
                if after_bootstrap:
                    after_bootstrap(bootstrap, node_path, manifest, report)
                    return
                before = len(calls)
                self.assertEqual(bootstrap.bootstrap(manifest), report)
                self.assertEqual(len(calls), before + 2)  # scheduler and staging source reads
                node_path('/opt/aim344/device-recovery/companion/prejob-prolog.sh').write_text('changed')
                with self.assertRaisesRegex(RuntimeError, 'Installed release changed'):
                    bootstrap.bootstrap(manifest)

    def test_archive_rejects_traversal_and_links(self):
        bootstrap = self.load('replacement-bootstrap.py')
        for name, kind in (('../escape', tarfile.REGTYPE), ('link', tarfile.SYMTYPE),
                           ('/absolute', tarfile.REGTYPE), ('dev', tarfile.CHRTYPE)):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                archive = Path(temporary) / 'bad.tgz'
                with tarfile.open(archive, 'w:gz') as tar:
                    info = tarfile.TarInfo(name)
                    info.type = kind
                    tar.addfile(info)
                with self.assertRaises(RuntimeError):
                    bootstrap.extract(archive, Path(temporary) / 'out')

    def test_regular_archive_and_durable_receipt(self):
        bootstrap = self.load('replacement-bootstrap.py')
        bootstrap.ROOT_UID = os.getuid()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with tarfile.open(root / 'good.tgz', 'w:gz') as tar:
                info = tarfile.TarInfo('file')
                info.size = 4
                tar.addfile(info, io.BytesIO(b'test'))
            bootstrap.extract(root / 'good.tgz', root / 'out')
            self.assertEqual((root / 'out/file').read_bytes(), b'test')
            bootstrap.durable(root / 'receipt', 'verified-local')
            self.assertEqual((root / 'receipt').read_text(), 'verified-local')
