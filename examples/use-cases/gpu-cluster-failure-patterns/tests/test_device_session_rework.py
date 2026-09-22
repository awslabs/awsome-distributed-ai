# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Repairs for the privileged-controller findings F1, F2, F7, F8, F9, F10, F11.

Each test below was written to fail against the rejected implementation
(commit 2d97e5ed) and to pass only once the named defect is repaired. They use
the same recorded fake executor as test_device_session.py rather than a new
framework, and they assert what the helper decided: which privileged calls it
made, in what order, and what it left on disk and in the state record.

Nothing here relaxes a safety condition to make a test pass. Where a test needs
a real kernel behaviour (a directory replaced under a live request, two
simultaneous starts) it exercises that behaviour with real processes and real
filesystem operations rather than asserting a code path was taken.

Run: python3 -m unittest discover -s <lab>/tests -p 'test_*.py'
"""
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import unittest

LAB = Path(__file__).resolve().parents[1]
HELPER = LAB / 'facilitator' / 'device-session.py'
RESTORE_RUNTIME = LAB / 'facilitator' / 'restore-runtime.sh'

_spec = importlib.util.spec_from_file_location('aim344_device_session_rework', HELPER)
if _spec is None or _spec.loader is None:                    # pragma: no cover
    raise unittest.SkipTest(f'Helper not importable: {HELPER}')
session = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(session)

_suite_spec = importlib.util.spec_from_file_location(
    'aim344_test_device_session', Path(__file__).resolve().parent / 'test_device_session.py')
_suite = importlib.util.module_from_spec(_suite_spec)
_suite_spec.loader.exec_module(_suite)
FakeExecutor = _suite.FakeExecutor


# Verdict lines exactly as the pinned suite emits them with no terminal attached.
# Taken from the collected logs on the target (check-6.log, check-0.json), where
# lib/common.sh suppressed colour because stdout was not a tty.
PASS_0 = ('# suite_revision=a4ba07eb15e6f277063b4000346f9109c98de843\n'
          '[PASS] 0-nvidia-smi: nvidia-smi OK, 8 GPU(s) detected\n')
PASS_3 = '[PASS] 3-topology-check: Topology OK: 8 GPUs, connectivity validated\n'
FAIL_0 = ('[FAIL] 0-nvidia-smi: Expected 8 GPUs, found 7 '
          '(severity: ISOLATE)\n')
WARN_6 = ('[WARN] 6-efa-loopback: EFA loopback completed for 2 domain(s); '
          'cumulative EFA statistics contain drops or retransmission timeouts '
          '(see logs)\n[WARN] EFA retransmission timeouts detected (2)\n')
PASS_2 = ('[PASS] 2-efa-enumeration: EFA OK: 2 PCI devices, 2 RDMA devices, '
          '2 uverbs nodes\n')
# A warning the controller CAN qualify against an unchanged baseline, for tests whose
# subject is the qualification machinery rather than check 6's counters. Check 6's two
# counters can no longer be auto-qualified at all -- their accumulation epoch is the
# instance launch or the last EFA driver reset and an EFA round rebinds that driver,
# so nothing this controller observes establishes continuity (S2, review t_3a0b8119).
# This one is check 2's memory-lock limit, `checks/2-efa-enumeration.sh:189-191` at
# the pinned revision, which is a point-in-time limit and not an accumulation.
#
# It carries the check's advisory AGGREGATE as well as the detail line, because that is
# what a real check-2 WARN prints: `warn_efa` increments `advisory_count` for each
# advisory and the final branch prints the count whenever it is nonzero. A capture
# holding only the detail is an incomplete observation and no longer qualifies, so a
# fixture without the aggregate would make every test that uses it fail for that
# reason rather than for its own subject.
WARN_2 = ('[WARN] 2-efa-enumeration: Memory lock limit 8192 KB is below 16 GiB '
          '-- EFA performance may be degraded\n'
          '[WARN] 2-efa-enumeration: EFA devices enumerated with 1 advisories; '
          'inspect raw output\n')
# Check 6's other two verdicts, defined here rather than beside the R2 tests because
# ReworkExecutor's defaults need PASS_6. SKIP_6 is checks/6-efa-loopback.sh:87
# verbatim, reached when fi_pingpong is not on PATH; PASS_6 is :197. Both were read
# from the deployed suite on the target at revision
# a4ba07eb15e6f277063b4000346f9109c98de843
# (participant-revision/runs/rework4-read-suite-and-slurm/output.log:11-152).
SKIP_6 = ('[SKIP] 6-efa-loopback: fi_pingpong not available '
          '(install the AWS EFA installer)\n')
PASS_6 = '[PASS] 6-efa-loopback: EFA loopback OK: 2 domain(s) tested\n'


class ReworkExecutor(FakeExecutor):
    """FakeExecutor plus per-check collect output and observed drain history."""

    def __init__(self):
        super().__init__()
        # check identifier -> text the suite would print for it.
        #
        # Check 6 defaults to PASS_6 rather than to its WARN. A check-6 counter
        # warning can no longer be qualified against a baseline at all -- see WARN_2
        # above -- so leaving WARN_6 as the default would have made 'the ordinary EFA
        # round resumes' depend on a warning that must now keep the node drained, and
        # every test about something else would fail for the S2 reason. Tests whose
        # subject IS the counter refusal set check 6 to WARN_6 explicitly.
        self.check_output = {'0': PASS_0, '3': PASS_3, '2': PASS_2, '6': PASS_6}
        self.gpu_restore_output = 'restored persistence_mode=Enabled\n'
        self.node_state_history = []
        self.collect_calls = []

    def run_maintenance(self, argv, timeout=None):
        if argv and argv[0] == 'collect' and argv[0] not in self.maintenance_failures:
            self.maintenance_calls.append(list(argv))
            self.collect_calls.append(str(argv[1]))
            text = self.check_output.get(str(argv[1]), PASS_0)
            code = 1 if '[FAIL]' in text else 0
            return session.Completed(code, text, '')
        if argv and argv[0] == 'gpu-restore' and 'gpu-restore' not in self.maintenance_failures:
            self.maintenance_calls.append(list(argv))
            return session.Completed(0, self.gpu_restore_output, '')
        return super().run_maintenance(argv, timeout=timeout)

    def run_slurm(self, argv, timeout=None):
        result = super().run_slurm(argv, timeout=timeout)
        if Path(argv[0]).name == 'scontrol' and argv[1:2] == ['update']:
            self.node_state_history.append(list(argv[2:]))
        return result

    # -- helpers the tests read --
    def resumes(self):
        return [c for c in self.slurm_calls if 'State=RESUME' in c]

    def reboots(self):
        return [c for c in self.slurm_calls if c[1:2] == ['reboot']]

    def actions(self):
        return [c[0] for c in self.maintenance_calls]


class Base(unittest.TestCase):
    """Same shape as the existing suite's Base, with the new config keys."""

    # An operation identifier of the shape the controller generates
    # (`<assignment>/<round>/<32 hex>`), for tests that hand-build a mid-round state
    # record and therefore skip the `start` that would have generated one. The target
    # now requires it on both `gpu-prepare` and `gpu-restore`, so a hand-built record
    # without one is a round the helper refuses to restore -- which is asserted
    # separately rather than worked around here.
    OPERATION = 'table-1/1/' + '0123456789abcdef' * 2

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
                    'participant_gid': os.getgid(),
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
                # Same physical node as table-1: this is the pair the review's
                # F2 race is about.
                'table-2': {
                    'participant_user': 'aim344-t2',
                    'participant_uid': os.getuid(),
                    'participant_gid': os.getgid(),
                    'caller_uid': os.getuid(),
                    'answers_dir': str(root / 'answers-2'),
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
            },
        }
        (root / 'answers-2').mkdir()
        self.write_config()
        self.executor = ReworkExecutor()

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config_body))
        os.chmod(self.config_path, 0o600)

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, assignment='table-1', peer_uid=None, now=1000.0, executor=None):
        config = session.load_config(self.config_path, require_root_owned=False)
        return session.DeviceSession(
            config=config,
            assignment_id=assignment,
            peer_uid=os.getuid() if peer_uid is None else peer_uid,
            executor=executor or self.executor,
            clock=lambda: now,
        )

    def baseline_for(self, state, executor=None, **overrides):
        """The pre-fault verdicts a real `start` would have recorded for a round.

        Tests that hand-build a mid-round state record skip `start`, so they also
        skip the baseline it captures. Recovery may accept a WARN only against a
        recorded pre-fault observation, so without this those tests would be
        asserting behaviour no real round can reach. This fills in the same shape
        `_capture_baseline` writes, from the same check output the fake executor
        would have produced, so the WARN a test relies on is backed by an
        observation rather than by an exemption.
        """
        source = executor or self.executor
        handler = self.make(executor=source)
        records = {}
        for check in handler.allowed_checks(state):
            text = source.check_output.get(str(check), PASS_0)
            records[f'round-{state.round}/baseline/check-{check}'] = {
                'verdict': session.DeviceSession.check_verdict(text) or 'UNREADABLE',
                'returncode': 0,
                'log': f'baseline-check-{check}.log',
                # The same warning details `_capture_baseline` records, derived
                # from the same text, so a WARN a test relies on is backed by an
                # observation of the same condition rather than by a bare label.
                'warnings': session.DeviceSession.warning_details(text),
                # And the boot it was observed on, which `_capture_baseline` also
                # records. A cumulative counter recorded before a reboot cannot be
                # compared with one read after it, so a hand-built baseline that
                # omitted this would refuse every counter warning.
                'boot_id': state.boot_id,
            }
        records.update(overrides)
        state.baseline_check_results = records
        return state


# --------------------------------------------------------------------------
# F1: privileged filesystem operations through participant-controlled paths.
# --------------------------------------------------------------------------
class F1PrivilegedWrites(Base):
    def test_root_never_writes_or_chowns_as_root_into_participant_paths(self):
        """The strategy decision itself, since the drop cannot run unprivileged.

        As root with a participant uid the helper must drop privilege; when it is
        already the participant it writes directly; with no participant account it
        refuses rather than writing as root.
        """
        self.assertEqual('drop', session.DeviceSession.write_strategy(0, 1001))
        self.assertEqual('direct', session.DeviceSession.write_strategy(1001, 1001))
        self.assertEqual('refuse', session.DeviceSession.write_strategy(0, None))
        self.assertEqual('refuse', session.DeviceSession.write_strategy(1002, 1001))

    def test_no_chown_is_performed_anywhere_in_the_results_path(self):
        """A chown by root follows a replaced symlink; there must be none left."""
        source = HELPER.read_text()
        body = source.split('# ---------------- answers directory ----------------', 1)[1]
        self.assertNotIn('os.chown', body)
        self.assertNotIn('shutil.chown', body)

    def test_a_replaced_directory_component_cannot_redirect_the_write(self):
        """Real race, not a code path assertion.

        A writer thread replaces the 'collected' directory with a symlink to an
        outside directory while writes are in flight. Because every component is
        opened O_NOFOLLOW and descriptor relative, a write either lands under the
        real results tree or is refused; it must never land in the victim tree.
        """
        victim = Path(self.tmp.name) / 'victim'
        victim.mkdir()
        handler = self.make()
        handler.store.write(session.State(phase='fault-applied', kind='gpu',
                                          started_at=1.0, target_node='gpu-g7-1'))
        collected = self.answers / 'collected'
        stop = False
        refusals = []
        written = []

        def swap():
            while not stop:
                try:
                    if collected.is_dir() and not collected.is_symlink():
                        shutil.rmtree(collected)
                        collected.symlink_to(victim)
                    elif collected.is_symlink():
                        collected.unlink()
                except OSError:
                    pass
                time.sleep(0.001)

        import threading
        swapper = threading.Thread(target=swap)
        swapper.start()
        try:
            for attempt in range(200):
                try:
                    written.append(handler._save_output(
                        '0', f'capture {attempt}\n', phase='fault', round=1))
                except session.Refusal as refusal:
                    refusals.append(str(refusal))
        finally:
            stop = True
            swapper.join()
        # Whatever happened, nothing reached the victim directory.
        self.assertEqual([], sorted(p.name for p in victim.rglob('*')),
                         f'a write escaped into {victim}; refusals={refusals[:3]}')
        # And the run really did exercise both outcomes or at least one write.
        self.assertTrue(written or refusals)

    def test_a_symlinked_results_root_is_refused_not_followed(self):
        victim = Path(self.tmp.name) / 'victim-root'
        victim.mkdir()
        (victim / 'keep').write_text('original\n')
        shutil.rmtree(self.answers)
        self.answers.symlink_to(victim)
        handler = self.make()
        with self.assertRaises(session.Refusal) as caught:
            handler._save_output('0', 'text\n', phase='fault', round=1)
        self.assertIn('symlink', str(caught.exception).lower())
        self.assertEqual(['keep'], sorted(p.name for p in victim.iterdir()))

    def test_a_pre_existing_fifo_at_the_output_path_cannot_block_the_writer(self):
        """O_EXCL refuses an existing leaf of any type, so a FIFO cannot stall it."""
        target = self.answers / 'collected' / 'round-1'
        target.mkdir(parents=True)
        os.mkfifo(target / 'check-0-fault.log')
        handler = self.make()
        saved = handler._save_output('0', 'text\n', phase='fault', round=1)
        self.assertNotEqual('check-0-fault.log', saved.name)
        self.assertTrue(saved.is_file() and not saved.is_symlink())
        self.assertEqual('text\n', saved.read_text())
        # The FIFO is still a FIFO: nothing was written through it.
        import stat as stat_module
        self.assertTrue(stat_module.S_ISFIFO(
            (target / 'check-0-fault.log').lstat().st_mode))

    def test_a_results_component_owned_by_someone_else_is_refused(self):
        """Ownership is checked on the opened descriptor, not on a name.

        This is the check that runs inside the privilege-dropped child, so it is
        exercised directly: the root path calls exactly this function after
        setuid, and a component owned by another account must stop the write.
        """
        handler = self.make()
        stranger = os.getuid() + 4242
        with self.assertRaises(session.Refusal) as caught:
            session._write_under(self.answers, ('collected', 'round-1'),
                                 lambda attempt: 'check-0-fault.log',
                                 'text\n', stranger)
        self.assertIn('not owned by your account', str(caught.exception))
        self.assertEqual([], list((self.answers / 'collected').rglob('*'))
                         if (self.answers / 'collected').is_dir() else [])
        # The same call with the true owner succeeds, so the check is not simply
        # refusing everything.
        written = session._write_under(self.answers, ('collected', 'round-1'),
                                       lambda attempt: 'check-0-fault.log',
                                       'text\n', os.getuid())
        self.assertEqual('text\n', written.read_text())

    def test_a_missing_participant_account_refuses_rather_than_writing_as_root(self):
        handler = self.make()
        handler.assignment.pop('participant_uid')
        handler.assignment.pop('participant_gid', None)
        with self.assertRaises(session.Refusal) as caught:
            handler._save_output('0', 'text\n', phase='fault', round=1)
        self.assertIn('no participant account', str(caught.exception))


# --------------------------------------------------------------------------
# F2: exclusion must be keyed by the physical pair, before any prepare or scan.
# --------------------------------------------------------------------------
def _start_in_child(config_path, assignment, barrier, results):   # pragma: no cover
    """Run one start() in a separate process against the same state directory."""
    spec = importlib.util.spec_from_file_location(f'child_{assignment}', HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Recorder:
        def __init__(self):
            self.mutations = []
            self.node_state = 'IDLE+CLOUD'
            self.node_reason = ''

        def run_maintenance(self, argv, timeout=None):
            action = argv[0]
            if action in ('gpu-remove', 'efa-unbind', 'gpu-prepare'):
                self.mutations.append(action)
            if action == 'inspect':
                return module.Completed(0, json.dumps({
                    'action': 'inspect',
                    'instance_id': 'i-0123456789abcdef0',
                    'confirmation': {
                        'gpu-remove': 'i-0123456789abcdef0/gpu-remove/GPU-x',
                        'efa-unbind': 'i-0123456789abcdef0/efa-unbind/0000:b0:00.0',
                        'efa-rebind': 'i-0123456789abcdef0/efa-rebind/0000:b0:00.0',
                    }}) + '\n', '')
            if action == 'boot-id':
                return module.Completed(0, 'aaaaaaaa-0000-0000-0000-000000000001\n', '')
            return module.Completed(0, '{}\n', '')

        def run_slurm(self, argv, timeout=None):
            program = Path(argv[0]).name
            if program == 'scontrol' and argv[1:3] == ['show', 'node']:
                return module.Completed(
                    0, f'NodeName={argv[3]} State={self.node_state}\n', '')
            if program == 'squeue':
                return module.Completed(0, '', '')
            return module.Completed(0, '', '')

    recorder = Recorder()
    config = module.load_config(config_path, require_root_owned=False)
    handler = module.DeviceSession(config, assignment, peer_uid=os.getuid(),
                                   executor=recorder, clock=time.time)
    barrier.wait()
    outcome = {'assignment': assignment}
    try:
        result = handler.start('gpu')
        outcome['phase'] = result.state.phase
        outcome['refusal'] = None
    except module.Refusal as refusal:
        outcome['phase'] = None
        outcome['refusal'] = str(refusal)
    outcome['mutations'] = recorder.mutations
    results.append(outcome)


class F2PairExclusion(Base):
    def test_two_simultaneous_starts_on_one_free_node_produce_one_owner(self):
        """The real race the review asked for: two processes, same free target."""
        manager = multiprocessing.Manager()
        results = manager.list()
        barrier = manager.Barrier(2)
        children = [
            multiprocessing.Process(target=_start_in_child,
                                    args=(str(self.config_path), name, barrier, results))
            for name in ('table-1', 'table-2')]
        for child in children:
            child.start()
        for child in children:
            child.join(60)
            self.assertEqual(0, child.exitcode, 'child start() crashed')
        outcomes = list(results)
        self.assertEqual(2, len(outcomes))
        owners = [o for o in outcomes if o['phase'] == 'fault-applied']
        refused = [o for o in outcomes if o['refusal']]
        self.assertEqual(1, len(owners), f'expected exactly one owner, got {outcomes}')
        self.assertEqual(1, len(refused), f'expected exactly one refusal, got {outcomes}')
        # Only the winner touched the hardware.
        mutating = [o for o in outcomes if o['mutations']]
        self.assertEqual(1, len(mutating), f'more than one table mutated: {outcomes}')
        self.assertIn('gpu-remove', mutating[0]['mutations'])
        # The loser did not even prepare.
        loser = refused[0]
        self.assertEqual([], loser['mutations'])

    def test_the_pair_lock_is_taken_before_any_state_scan_or_prepare(self):
        """Holding the pair lock refuses the other table with nothing mutated."""
        first = self.make(assignment='table-1')
        second = self.make(assignment='table-2')
        with first.store.exclusive_pair():
            before = len(self.executor.maintenance_calls)
            with self.assertRaises(session.Refusal) as caught:
                second.start('gpu')
            self.assertIn('another table', str(caught.exception).lower())
            self.assertEqual(before, len(self.executor.maintenance_calls))
            self.assertEqual([], self.executor.node_state_history)

    def test_the_pair_lock_is_named_for_the_node_not_the_assignment(self):
        first = self.make(assignment='table-1')
        second = self.make(assignment='table-2')
        self.assertEqual(first.store.pair_lock_path, second.store.pair_lock_path)
        self.assertNotEqual(first.store.lock_path, second.store.lock_path)

    def test_a_different_node_is_not_blocked_by_the_pair_lock(self):
        self.config_body['assignments']['table-2']['target_node'] = 'gpu-g7-9'
        self.write_config()
        first = self.make(assignment='table-1')
        second = self.make(assignment='table-2')
        self.assertNotEqual(first.store.pair_lock_path, second.store.pair_lock_path)
        with first.store.exclusive_pair():
            with second.store.exclusive_pair():
                pass

    def test_recovery_also_holds_the_pair_lock(self):
        handler = self.make(assignment='table-1')
        handler.start('gpu')
        other = self.make(assignment='table-2')
        with other.store.exclusive_pair():
            with self.assertRaises(session.Refusal) as caught:
                self.make(assignment='table-1').recover()
            self.assertIn('exercise node', str(caught.exception).lower())


# --------------------------------------------------------------------------
# F7: a failed required check or restoration must never reach RESUME.
# --------------------------------------------------------------------------
class F7FailClosedRecovery(Base):
    def test_a_failed_check_after_recovery_keeps_the_node_drained(self):
        handler = self.make()
        handler.start('gpu')
        self.executor.check_output['0'] = FAIL_0
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        self.assertIn('not returned to service', str(caught.exception))
        self.assertEqual([], self.executor.resumes())
        state = self.make().store.read()
        self.assertEqual('recovery-failed', state.phase)
        self.assertIsNone(state.recovered_by)
        self.assertIn('Check 0 did not pass', state.notes)
        # The drain that start applied is still the node's state.
        self.assertIn('DRAIN', self.executor.node_state)

    def test_a_failed_gpu_restore_keeps_the_node_drained(self):
        handler = self.make()
        handler.start('gpu')
        self.executor.maintenance_failures['gpu-restore'] = (
            1, 'Persistence is Disabled, expected the recorded Enabled.\n')
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        self.assertIn('not fully restored', str(caught.exception))
        self.assertEqual([], self.executor.resumes())
        self.assertEqual('recovery-failed', self.make().store.read().phase)

    def test_telemetry_that_did_not_come_back_is_not_a_pass(self):
        """maintenance.gpu_restore prints this and used to still return 0."""
        handler = self.make()
        handler.start('gpu')
        self.executor.gpu_restore_output = (
            'restored persistence_mode=Enabled\nrestored nvidia-dcgm=failed\n'
            'nvidia-dcgm did not return to active; preserve this state for the '
            'facilitator.\n')
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        self.assertIn('telemetry', str(caught.exception).lower())
        self.assertEqual([], self.executor.resumes())
        self.assertEqual('recovery-failed', self.make().store.read().phase)

    def test_a_transport_timeout_during_checks_is_not_a_pass(self):
        handler = self.make()
        handler.start('gpu')
        self.executor.maintenance_failures['collect'] = (
            124, 'External observation deadline reached.\n')
        with self.assertRaises(session.Refusal):
            self.make().recover()
        self.assertEqual([], self.executor.resumes())
        self.assertEqual('recovery-failed', self.make().store.read().phase)

    def test_a_baseline_warn_is_preserved_distinctly_without_resume(self):
        """Final finding 5 disables unproven memlock acceptance, retaining evidence."""
        self.executor.check_output['2'] = WARN_2
        handler = self.make()
        handler.start('efa')
        with self.assertRaises(session.Refusal):
            self.make().recover()
        state = self.make().store.read()
        self.assertEqual('replacement-required', state.phase)
        self.assertEqual([], self.executor.resumes())
        self.assertIn('WARN', state.notes)
        recorded = state.check_results
        self.assertEqual('WARN', recorded['round-1/recovered/check-2']['verdict'])
        self.assertEqual('PASS', recorded['round-1/recovered/check-6']['verdict'])

    def test_a_qualified_pass_does_resume(self):
        handler = self.make()
        handler.start('gpu')
        result = self.make().recover()
        self.assertEqual('runtime-ready', result.state.phase)
        self.assertEqual(1, len(self.executor.resumes()))
        self.assertEqual('', result.state.notes)

    def test_an_unreadable_verdict_is_treated_as_not_passed(self):
        """A missing verdict line is not evidence of health."""
        handler = self.make()
        handler.start('gpu')
        self.executor.check_output['0'] = '# suite_revision=abc\nsome noise\n'
        with self.assertRaises(session.Refusal):
            self.make().recover()
        self.assertEqual([], self.executor.resumes())

    def test_the_verdict_parser_reads_the_real_shipped_lines(self):
        """Both outcomes, not only the failing one."""
        self.assertEqual('PASS', session.DeviceSession.check_verdict(PASS_0))
        self.assertEqual('PASS', session.DeviceSession.check_verdict(PASS_2))
        self.assertEqual('FAIL', session.DeviceSession.check_verdict(FAIL_0))
        self.assertEqual('WARN', session.DeviceSession.check_verdict(WARN_6))
        self.assertIsNone(session.DeviceSession.check_verdict('no verdict here\n'))
        # A FAIL anywhere wins over an earlier PASS on another line.
        self.assertEqual('FAIL', session.DeviceSession.check_verdict(PASS_2 + FAIL_0))


# --------------------------------------------------------------------------
# F8: identity, drain and job ownership before any reboot or runtime mutation.
# --------------------------------------------------------------------------
class F8ChecksBeforeReboot(Base):
    def test_a_foreign_job_stops_recovery_without_queueing_a_reboot(self):
        handler = self.make()
        handler.start('gpu')
        self.executor.jobs = ['777|someone-else|their-training|RUNNING']
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        self.assertIn('unapproved job', str(caught.exception).lower())
        self.assertEqual([], self.executor.reboots())
        self.assertEqual([], self.executor.resumes())

    def test_a_changed_instance_stops_recovery_before_any_reboot(self):
        handler = self.make()
        handler.start('gpu')
        self.executor.inspect['instance_id'] = 'i-0999999999999999f'
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        self.assertIn('instance', str(caught.exception).lower())
        self.assertEqual([], self.executor.reboots())
        self.assertNotIn('restore-runtime', self.executor.actions())

    def test_an_efa_safety_refusal_is_not_a_reboot_fallback(self):
        """device-fault.sh refuses rebind while a job remains. That means end the
        allocation, not reboot the node."""
        handler = self.make()
        handler.start('efa')
        self.executor.maintenance_failures['efa-rebind'] = (
            1, 'End the faulted allocation before EFA rebind; old communicators '
               'are not reusable.\n')
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        self.assertIn('stopped before changing anything', str(caught.exception))
        self.assertEqual([], self.executor.reboots())
        self.assertEqual([], self.executor.resumes())
        self.assertEqual('recovery-failed', self.make().store.read().phase)

    def test_a_device_rebind_failure_requires_replacement_not_reboot(self):
        """Final finding 2: no originals means no safe EFA reboot fallback."""
        handler = self.make()
        handler.start('efa')
        self.executor.maintenance_failures['efa-rebind'] = (
            1, 'Unexpected driver binding.\n')
        # 'Unexpected driver binding' is itself an identity refusal, so use a
        # device-level failure that is not in the safety set.
        self.executor.maintenance_failures['efa-rebind'] = (
            1, 'write error: No such device\n')
        # Even clean check output cannot authorize the unsafe fallback.
        self.executor.check_output['6'] = PASS_6
        with self.assertRaises(session.Refusal):
            self.make().recover()
        self.assertEqual([], self.executor.reboots())
        self.assertEqual([], self.executor.resumes())
        self.assertNotIn('restore-runtime', self.executor.actions())
        state = self.make().store.read()
        self.assertEqual('replacement-required', state.phase)
        self.assertIn('No such device', state.notes)

    def test_a_counter_warning_is_not_qualified_across_a_reboot(self):
        """A cumulative EFA counter warning is never auto-qualified.

        The narrowest form of the S2 rule, and it is narrower than a reboot. Check 6's
        two counters accumulate from the instance launch or the last driver reset (AWS
        EFA metric documentation; drivers/infiniband/hw/efa/efa_main.c resets on probe
        and on remove), and this exercise's own helper writes the driver's unbind and
        bind files (device-fault.sh:143-146). So the round rebinds the driver whether
        or not it also reboots, and nothing this controller observes distinguishes 'the
        same counter still at 2' from 'reset, then two new events'. The node keeps its
        drain, and the refusal says the warning cannot be auto-resumed rather than
        claiming the hardware is faulty.
        """
        handler = self.make()
        self.executor.check_output['6'] = WARN_6
        handler.start('efa')
        # Historical already-issued request, not authorization of a new fallback.
        # Final finding 2 forbids creating that request via failed rebind now.
        state = handler.store.read()
        state.reboot_requests = 1
        state.reboot_requested_at = 900.0
        handler.store.write(state)
        handler._scontrol('reboot', 'reason=aim344-device-recovery', 'gpu-g7-1')
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        message = str(caught.exception)
        self.assertIn('cumulative EFA counter', message)
        self.assertIn('last EFA driver reset', message)
        self.assertIn('not a finding that the hardware is faulty', message)
        self.assertEqual(1, len(self.executor.reboots()))
        self.assertEqual([], self.executor.resumes(),
                         'a counter warning was qualified after a driver reset')
        self.assertEqual('replacement-required', self.make().store.read().phase)

    def test_a_counter_warning_is_refused_without_any_reboot_too(self):
        """The rule is the driver reset, not the boot.

        The rejected predicate compared boot ids, so an EFA round that rebound the
        device successfully and never rebooted passed it. That is the case this
        asserts: no reboot at all, an unchanged counter warning, and the node still
        keeps its drain.
        """
        handler = self.make()
        self.executor.check_output['6'] = WARN_6
        handler.start('efa')
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        self.assertIn('cumulative EFA counter', str(caught.exception))
        self.assertEqual([], self.executor.reboots(),
                         'this case is only interesting without a reboot')
        self.assertEqual([], self.executor.resumes())

    def test_the_node_is_re_drained_if_it_is_in_service_before_recovery(self):
        handler = self.make()
        handler.start('gpu')
        # Somebody resumed the node between start and recover.
        self.executor.node_state = 'IDLE+CLOUD'
        self.executor.node_reason = ''
        self.make().recover()
        drains = [c for c in self.executor.node_state_history if 'State=DRAIN' in c]
        self.assertGreaterEqual(len(drains), 2)
        reboot_index = [i for i, c in enumerate(self.executor.slurm_calls)
                        if c[1:2] == ['reboot']][0]
        last_drain_index = max(i for i, c in enumerate(self.executor.slurm_calls)
                               if 'State=DRAIN' in c)
        self.assertLess(last_drain_index, reboot_index)

    def test_a_foreign_drain_reason_stops_recovery(self):
        handler = self.make()
        handler.start('gpu')
        self.executor.node_reason = 'hardware-under-investigation'
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        self.assertIn('hardware-under-investigation', str(caught.exception))
        self.assertEqual([], self.executor.reboots())


# --------------------------------------------------------------------------
# F9: a completed reboot is persisted and never repeated.
# --------------------------------------------------------------------------
class F9NoRepeatedReboot(Base):
    def test_a_failure_after_the_boot_retries_without_rebooting_again(self):
        handler = self.make()
        handler.start('gpu')
        self.executor.maintenance_failures['restore-runtime'] = (
            1, 'Missing staged image: /opt/aim344/aim344.sqsh\n')
        with self.assertRaises(session.Refusal):
            self.make().recover()
        first = len(self.executor.reboots())
        self.assertEqual(1, first)
        state = self.make().store.read()
        self.assertTrue(state.reboot_completed)
        self.assertEqual('bbbbbbbb-0000-0000-0000-000000000002',
                         state.boot_id_after_reboot)
        # Retry with the fault cleared: restoration resumes on the same boot.
        del self.executor.maintenance_failures['restore-runtime']
        result = self.make().recover()
        self.assertEqual(first, len(self.executor.reboots()))
        self.assertEqual('runtime-ready', result.state.phase)
        self.assertEqual(state.boot_id_after_reboot, result.state.boot_id)

    def test_a_failed_check_retry_does_not_reboot_again(self):
        handler = self.make()
        handler.start('gpu')
        self.executor.check_output['0'] = FAIL_0
        with self.assertRaises(session.Refusal):
            self.make().recover()
        self.assertEqual(1, len(self.executor.reboots()))
        self.executor.check_output['0'] = PASS_0
        result = self.make().recover()
        self.assertEqual(1, len(self.executor.reboots()))
        self.assertEqual('runtime-ready', result.state.phase)
        self.assertEqual(1, len(self.executor.resumes()))

    def test_the_completed_boot_is_recorded_before_restoration_runs(self):
        """Persisted before the fallible work, not after it."""
        handler = self.make()
        handler.start('gpu')
        observed = {}
        original = self.executor.run_maintenance

        def watch(argv, timeout=None):
            if argv and argv[0] == 'restore-runtime':
                observed['at_restore'] = self.make().store.read().reboot_completed
            return original(argv, timeout=timeout)

        self.executor.run_maintenance = watch
        self.make().recover()
        self.assertTrue(observed.get('at_restore'),
                        'the completed reboot was not persisted before restoration')


# --------------------------------------------------------------------------
# F10: ownership and intent recorded before the first mutation.
# --------------------------------------------------------------------------
class F10RecoverableRecord(Base):
    # What the target really prints when `gpu-prepare` fails at its persistence
    # verification. maintenance.gpu_prepare writes both original-state records and
    # then prints this line (maintenance.py:589-592) BEFORE it pauses telemetry or
    # touches persistence, so 'Persistence is still Enabled' can only be reached
    # after it. A fake that omitted the stdout would describe a refusal the target
    # cannot produce, and the controller reads that line to tell a round whose
    # originals exist from one that was refused before they were taken.
    PREPARE_FAILED_AFTER_RECORDING = (
        1,
        'recorded persistence_mode=Enabled nvidia-dcgm.service=active '
        'operation=table-1/1/' + '0123456789abcdef' * 2 + '\n'
        'paused nvidia-dcgm.service\n',
        'Persistence is still Enabled on the selected GPU.\n')

    def test_a_failed_gpu_prepare_leaves_a_recoverable_record(self):
        self.executor.maintenance_failures['gpu-prepare'] = \
            self.PREPARE_FAILED_AFTER_RECORDING
        handler = self.make()
        with self.assertRaises(session.Refusal):
            handler.start('gpu')
        state = self.make().store.read()
        self.assertNotEqual('ready', state.phase)
        self.assertEqual('gpu', state.kind)
        self.assertEqual('gpu-g7-1', state.target_node)
        self.assertIn('gpu-prepare-attempted', state.prepared)
        # The target recorded the originals before it failed, so this round has a
        # state to put back and recovery must act on it.
        self.assertIn('gpu-prepare-recorded', state.prepared)
        self.assertIn('Persistence is still Enabled', state.failure)
        # recover must accept this, not say there is nothing to recover.
        result = self.make().recover()
        self.assertEqual('runtime-ready', result.state.phase)
        self.assertIn('gpu-restore', self.executor.actions())

    def test_the_original_node_state_is_recorded_before_the_drain(self):
        handler = self.make()
        handler.start('gpu')
        state = self.make().store.read()
        self.assertEqual('IDLE+CLOUD', state.original_node_state)
        self.assertIn('drain', state.prepared)

    def test_a_failed_drain_is_recorded_and_nothing_is_mutated(self):
        class RefusingDrain(ReworkExecutor):
            def run_slurm(self, argv, timeout=None):
                if (Path(argv[0]).name == 'scontrol' and argv[1:2] == ['update']
                        and any('State=DRAIN' in a for a in argv)):
                    self.slurm_calls.append(list(argv))
                    return session.Completed(1, '', 'scontrol: error: Invalid node\n')
                return super().run_slurm(argv, timeout=timeout)

        executor = RefusingDrain()
        handler = self.make(executor=executor)
        with self.assertRaises(session.Refusal):
            handler.start('gpu')
        state = self.make(executor=executor).store.read()
        self.assertEqual('preparing', state.phase)
        self.assertNotIn('drain', state.prepared)
        self.assertNotIn('gpu-prepare', executor.actions())
        self.assertNotIn('gpu-remove', executor.actions())

    def test_a_partly_prepared_round_still_holds_the_node_against_other_tables(self):
        self.executor.maintenance_failures['gpu-prepare'] = \
            self.PREPARE_FAILED_AFTER_RECORDING
        with self.assertRaises(session.Refusal):
            self.make(assignment='table-1').start('gpu')
        with self.assertRaises(session.Refusal) as caught:
            self.make(assignment='table-2').start('gpu')
        self.assertIn('another table', str(caught.exception).lower())

    def test_a_recovery_failed_round_still_holds_the_node(self):
        handler = self.make(assignment='table-1')
        handler.start('gpu')
        self.executor.check_output['0'] = FAIL_0
        with self.assertRaises(session.Refusal):
            self.make(assignment='table-1').recover()
        with self.assertRaises(session.Refusal) as caught:
            self.make(assignment='table-2').start('gpu')
        self.assertIn('another table', str(caught.exception).lower())

    def test_a_recovery_failed_round_refuses_a_new_round_on_the_same_table(self):
        handler = self.make()
        handler.start('gpu')
        self.executor.check_output['0'] = FAIL_0
        with self.assertRaises(session.Refusal):
            self.make().recover()
        with self.assertRaises(session.Refusal) as caught:
            self.make().start('efa')
        self.assertIn('did not finish recovering', str(caught.exception))


# --------------------------------------------------------------------------
# R1: no privileged dereference of participant-controlled fixture entries.
# --------------------------------------------------------------------------
def _handover_program():
    """The shipped fixture-handover program, lifted out of restore-runtime.sh.

    The tests below run the real shipped text rather than a copy of it, with only
    the account lookup stubbed, because the exercise's accounts do not exist on a
    developer machine. Everything the finding is about -- which syscalls touch
    which inode -- is the shipped code's own.
    """
    text = RESTORE_RUNTIME.read_text()
    start = text.index("<<'PYHANDOVER'\n") + len("<<'PYHANDOVER'\n")
    end = text.index('\nPYHANDOVER', start)
    body = text[start:end]
    preamble = (
        'import grp, pwd, sys\n'
        'class _Entry:\n'
        '    pw_uid = int(sys.argv[3])\n'
        '    pw_gid = int(sys.argv[4])\n'
        'pwd.getpwnam = lambda name: _Entry()\n'
        'class _Group:\n'
        '    gr_name = "stub-group"\n'
        'grp.getgrgid = lambda gid: _Group()\n'
    )
    return preamble + body


class R1FixtureHandover(unittest.TestCase):
    """The privileged fixture handover must not follow participant-made names.

    restore-runtime runs as root on the fault target and hands the checkpoint
    fixture to the authenticated table. The fixture is deliberately participant
    writable, including from inside the distributed workload, so any entry in it
    can be a symlink to somewhere else on the node. A path-based chown
    dereferences a symlink argument by default, which is what the rejected
    implementation did.

    These tests do not run as root, so they cannot change an inode to an account
    they do not own. They instead use a group the running user already belongs
    to, which an unprivileged process may set: if the handover reaches an entry
    outside the fixture, that file's group changes, and if it does not, the group
    is untouched. test_the_group_signal_detects_a_real_escape is the negative
    control for that signal, run against the exact command shape the review
    rejected.
    """

    @classmethod
    def setUpClass(cls):
        alternatives = [g for g in os.getgroups() if g != os.getgid()]
        if not alternatives:
            raise unittest.SkipTest('no supplementary group to use as the signal')
        cls.other_gid = alternatives[0]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.fixture = root / 'fixture'
        self.fixture.mkdir()
        os.chmod(self.fixture, 0o755)
        self.outside = root / 'outside'
        self.outside.mkdir()
        self.victim = self.outside / 'root-owned-secret'
        self.victim.write_text('a file the participant must not come to own\n')
        os.chown(self.victim, -1, os.getgid())
        self.program = _handover_program()

    def tearDown(self):
        self.tmp.cleanup()

    def run_handover(self, expect_success=True):
        finished = subprocess.run(
            [sys.executable, '-c', self.program, 'stub-participant',
             str(self.fixture), str(os.getuid()), str(self.other_gid)],
            capture_output=True, text=True, check=False)
        if expect_success:
            self.assertEqual(0, finished.returncode,
                             f'handover failed: {finished.stderr}')
        return finished

    def victim_gid(self):
        return os.stat(self.victim).st_gid

    # -- the escape itself, on every shape a participant can create --
    def test_a_symlink_entry_does_not_change_ownership_outside_the_fixture(self):
        (self.fixture / 'last.json').symlink_to(self.victim)
        before = self.victim_gid()
        finished = self.run_handover()
        self.assertEqual(before, self.victim_gid(),
                         'the handover followed a participant symlink')
        self.assertIn('left_untouched=', finished.stdout)
        self.assertIn('last.json', finished.stdout)

    def test_a_symlink_to_a_directory_outside_the_fixture_is_refused(self):
        (self.fixture / 'checkpoints').symlink_to(self.outside)
        before = os.stat(self.outside).st_gid
        finished = self.run_handover()
        self.assertEqual(before, os.stat(self.outside).st_gid)
        self.assertEqual(os.stat(self.victim).st_gid, self.victim_gid())
        self.assertIn('checkpoints', finished.stdout)

    def test_an_absolute_and_a_relative_traversal_link_are_both_refused(self):
        (self.fixture / 'absolute').symlink_to(self.victim)
        (self.fixture / 'relative').symlink_to(
            Path('..') / 'outside' / 'root-owned-secret')
        before = self.victim_gid()
        finished = self.run_handover()
        self.assertEqual(before, self.victim_gid())
        for name in ('absolute', 'relative'):
            self.assertIn(name, finished.stdout)

    def test_a_dangling_link_replaced_by_a_real_file_is_still_not_followed(self):
        """The name is resolved once, through a descriptor, or not at all."""
        link = self.fixture / 'appears-later'
        link.symlink_to(self.outside / 'not-there-yet')
        (self.outside / 'not-there-yet').write_text('planted\n')
        os.chown(self.outside / 'not-there-yet', -1, os.getgid())
        before = os.stat(self.outside / 'not-there-yet').st_gid
        self.run_handover()
        self.assertEqual(before, os.stat(self.outside / 'not-there-yet').st_gid)

    def test_a_fifo_entry_is_left_alone_rather_than_opened_for_data(self):
        os.mkfifo(self.fixture / 'pipe')
        finished = self.run_handover()
        self.assertIn('pipe', finished.stdout)
        self.assertIn('not a regular file or directory', finished.stdout)

    # -- the negative control for the signal these tests rely on --
    def test_the_group_signal_detects_a_real_escape(self):
        """The rejected command shape must make this signal move.

        Without this, a passing test above could mean the escape did not happen
        or merely that the check cannot see it. This runs the exact rejected
        construction from the review, find -mindepth 1 -maxdepth 1 -exec chown,
        and requires the victim's group to change.
        """
        (self.fixture / 'last.json').symlink_to(self.victim)
        before = self.victim_gid()
        subprocess.run(
            ['find', str(self.fixture), '-mindepth', '1', '-maxdepth', '1',
             '-exec', 'chown', f'{os.getuid()}:{self.other_gid}', '{}', '+'],
            capture_output=True, text=True, check=False)
        self.assertNotEqual(
            before, self.victim_gid(),
            'the rejected command shape did not move the signal, so the tests '
            'above cannot be trusted to detect an escape')

    # -- and the legitimate restoration must still work --
    def test_the_previous_rounds_own_entries_are_handed_over(self):
        marker = self.fixture / '.aim344-fixture'
        marker.write_text('AIM344 isolated checkpoint fixture\n')
        checkpoint = self.fixture / 'last.json'
        checkpoint.write_text('{"next_step": 3}\n')
        nested = self.fixture / 'dist-checkpoint'
        nested.mkdir()
        for path in (marker, checkpoint, nested):
            os.chown(path, -1, os.getgid())
        finished = self.run_handover()
        for path in (marker, checkpoint, nested):
            self.assertEqual(self.other_gid, os.stat(path).st_gid,
                             f'{path.name} was not handed over')
        self.assertIn('handed_over=3', finished.stdout)
        self.assertNotIn('left_untouched=', finished.stdout)
        # The participant's saved progress is preserved, not reinitialised.
        self.assertEqual('{"next_step": 3}\n', checkpoint.read_text())

    def test_the_fixture_directory_itself_becomes_private_to_the_table(self):
        self.run_handover()
        info = os.stat(self.fixture)
        self.assertEqual(0o700, stat.S_IMODE(info.st_mode))
        self.assertEqual(self.other_gid, info.st_gid)

    def test_one_bad_entry_does_not_stop_the_good_ones(self):
        (self.fixture / 'escape').symlink_to(self.victim)
        keep = self.fixture / 'last.json'
        keep.write_text('{"next_step": 1}\n')
        os.chown(keep, -1, os.getgid())
        before = self.victim_gid()
        finished = self.run_handover()
        self.assertEqual(self.other_gid, os.stat(keep).st_gid)
        self.assertEqual(before, self.victim_gid())
        self.assertIn('handed_over=1', finished.stdout)

    def test_the_shipped_script_has_no_path_based_chown_left(self):
        """The repair must remove the construction, not add a check beside it."""
        text = RESTORE_RUNTIME.read_text()
        code = [line for line in text.splitlines()
                if not line.lstrip().startswith('#')]
        flat = ' '.join(code)
        self.assertNotIn('-exec chown', flat)
        self.assertNotIn('chown -R', flat)
        for line in code:
            stripped = line.strip()
            self.assertFalse(
                stripped.startswith('chown ') or stripped.startswith('chmod '),
                f'a path-based ownership command remains: {stripped}')


# --------------------------------------------------------------------------
# R2: qualification is fail closed, and a WARN needs a real baseline.
# --------------------------------------------------------------------------


class R2QualifiedResume(Base):
    """A node returns to service only on evidence that it is healthy.

    The exact verdict strings here are the pinned suite's own. SKIP_6 is
    checks/6-efa-loopback.sh:87 verbatim, reached when fi_pingpong is not on PATH;
    WARN_6 is :195, reached from cumulative rx_drops or retransmission timeouts;
    PASS_6 is :197. All three were read from the deployed suite on the target at
    revision a4ba07eb15e6f277063b4000346f9109c98de843
    (participant-revision/runs/rework4-read-suite-and-slurm/output.log:11-152).
    They are defined at the top of this module because ReworkExecutor's defaults
    need PASS_6.
    """

    def test_a_skipped_required_check_does_not_resume_the_node(self):
        """SKIP is an absent qualification, not a lenient pass.

        This is the reachable case the review named: without fi_pingpong the
        loopback check skips, so the round has no loopback evidence at all.
        """
        handler = self.make()
        handler.start('efa')
        self.executor.check_output['6'] = SKIP_6
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        self.assertIn('skipped', str(caught.exception).lower())
        self.assertEqual([], self.executor.resumes())
        state = self.make().store.read()
        self.assertEqual('recovery-failed', state.phase)
        self.assertIsNone(state.recovered_by)
        # The drain the round applied is still the node's state.
        self.assertIn('DRAIN', self.executor.node_state)
        # And the SKIP is preserved as what it was, not rewritten.
        self.assertEqual('SKIP', state.check_results[
            'round-1/recovered/check-6']['verdict'])

    def test_a_pass_carried_by_a_nonzero_return_code_does_not_resume(self):
        """The text said PASS but the route did not complete.

        124 is this helper's own external observation deadline
        (Executor._run), which is not evidence the remote check ended.
        """
        handler = self.make()
        handler.start('gpu')
        original = self.executor.run_maintenance

        def truncated(argv, timeout=None):
            result = original(argv, timeout=timeout)
            if argv and argv[0] == 'collect' and str(argv[1]) == '3':
                return session.Completed(124, PASS_3, 'deadline reached\n')
            return result

        self.executor.run_maintenance = truncated
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        self.assertIn('did not complete', str(caught.exception))
        self.assertIn('124', str(caught.exception))
        self.assertEqual([], self.executor.resumes())
        self.assertEqual('recovery-failed', self.make().store.read().phase)

    def test_a_warn_carried_by_a_nonzero_return_code_does_not_resume(self):
        handler = self.make()
        handler.start('efa')
        original = self.executor.run_maintenance

        def truncated(argv, timeout=None):
            result = original(argv, timeout=timeout)
            if argv and argv[0] == 'collect' and str(argv[1]) == '6':
                return session.Completed(2, WARN_6, '')
            return result

        self.executor.run_maintenance = truncated
        with self.assertRaises(session.Refusal):
            self.make().recover()
        self.assertEqual([], self.executor.resumes())

    def test_a_warn_with_no_recorded_baseline_does_not_resume(self):
        """'The baseline also warns' must be read, not asserted.

        A round whose pre-fault observation of that check is missing has nothing
        establishing the warning was already there. Uses check 2's limit warning, so
        the refusal is the missing baseline rather than S2's counter rule.
        """
        self.executor.check_output['2'] = WARN_2
        handler = self.make()
        handler.start('efa')
        state = self.make().store.read()
        state.baseline_check_results = {}
        self.make().store.write(state)
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        message = str(caught.exception)
        self.assertIn('no pre-fault result', message)
        self.assertEqual([], self.executor.resumes())
        self.assertEqual('replacement-required', self.make().store.read().phase)

    def test_a_warn_absent_from_the_baseline_is_treated_as_new(self):
        """The counters this WARN comes from can rise during the round.

        checks/6-efa-loopback.sh warns on cumulative rx_drops or retransmission
        timeouts, so a warning that appeared during this round looks exactly like
        a pre-existing one unless the baseline is compared.
        """
        self.executor.check_output['6'] = PASS_6      # clean before the fault
        handler = self.make()
        handler.start('efa')
        self.assertEqual('PASS', self.make().store.read().baseline_check_results[
            'round-1/baseline/check-6']['verdict'])
        self.executor.check_output['6'] = WARN_6      # anomaly appears after
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        self.assertIn('the warning is new', str(caught.exception))
        self.assertEqual([], self.executor.resumes())

    def test_a_warn_matching_the_recorded_baseline_is_refused_and_named(self):
        """Final finding 5: text equality is not supported memlock provenance."""
        self.executor.check_output['2'] = WARN_2
        handler = self.make()
        handler.start('efa')
        baseline = self.make().store.read().baseline_check_results
        self.assertEqual('WARN', baseline['round-1/baseline/check-2']['verdict'])
        with self.assertRaises(session.Refusal):
            self.make().recover()
        state = self.make().store.read()
        self.assertEqual('replacement-required', state.phase)
        self.assertEqual([], self.executor.resumes())
        self.assertIn('WARN', state.notes)
        self.assertIn('auto-qualification is disabled', state.notes)
        recorded = state.check_results
        self.assertEqual('WARN', recorded['round-1/recovered/check-2']['verdict'])
        self.assertEqual('PASS', recorded['round-1/recovered/check-6']['verdict'])

    def test_a_baseline_that_could_not_be_collected_is_not_a_baseline(self):
        """A recorded non-observation must not stand in for an observation."""
        self.executor.check_output['2'] = WARN_2
        handler = self.make()
        handler.start('efa')
        state = self.make().store.read()
        state.baseline_check_results['round-1/baseline/check-2'] = {
            'verdict': 'NOT-COLLECTED', 'returncode': None, 'log': None}
        self.make().store.write(state)
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        self.assertIn('not WARN', str(caught.exception))
        self.assertEqual([], self.executor.resumes())

    def test_a_baseline_warn_that_did_not_complete_is_not_a_baseline(self):
        self.executor.check_output['2'] = WARN_2
        handler = self.make()
        handler.start('efa')
        state = self.make().store.read()
        state.baseline_check_results['round-1/baseline/check-2'] = {
            'verdict': 'WARN', 'returncode': 124, 'log': 'x.log'}
        self.make().store.write(state)
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        self.assertIn('did not complete', str(caught.exception))
        self.assertEqual([], self.executor.resumes())

    def test_the_baseline_is_captured_before_the_device_is_touched(self):
        """Otherwise it is not a baseline; it is a post-fault observation."""
        handler = self.make()
        order = []
        original = self.executor.run_maintenance

        def watch(argv, timeout=None):
            if argv:
                order.append(str(argv[0]) if argv[0] != 'collect'
                             else f'collect-{argv[1]}')
            return original(argv, timeout=timeout)

        self.executor.run_maintenance = watch
        handler.start('efa')
        self.assertIn('collect-2', order)
        self.assertIn('efa-unbind', order)
        self.assertLess(order.index('collect-2'), order.index('efa-unbind'),
                        f'the baseline ran after the mutation: {order}')
        self.assertLess(order.index('collect-6'), order.index('efa-unbind'))

    def test_the_baseline_of_one_round_does_not_qualify_the_next(self):
        """Each round records its own; a stale record must not carry over."""
        handler = self.make()
        handler.start('efa')
        self.make().recover()
        second = self.make().start('efa')
        self.assertEqual(2, second.state.round)
        keys = sorted(self.make().store.read().baseline_check_results)
        self.assertEqual(['round-2/baseline/check-2', 'round-2/baseline/check-6'],
                         keys)


# --------------------------------------------------------------------------
# R3: every recovery attempt revalidates before it mutates.
# --------------------------------------------------------------------------
class R3RevalidateEveryAttempt(Base):
    """A retry after a completed reboot is still a mutation.

    The rejected code guarded the drain and job checks with
    `if not state.reboot_completed`, so a restoration-only retry went from an
    instance-id comparison straight to remount-staging and restore-runtime.
    """

    def reach_a_restoration_retry(self):
        """Leave the round with the reboot done and restoration outstanding."""
        handler = self.make()
        handler.start('gpu')
        self.executor.maintenance_failures['restore-runtime'] = (
            1, 'Missing staged image: /opt/aim344/aim344.sqsh\n')
        with self.assertRaises(session.Refusal):
            self.make().recover()
        state = self.make().store.read()
        self.assertTrue(state.reboot_completed)
        del self.executor.maintenance_failures['restore-runtime']
        return state

    def test_a_foreign_job_appearing_between_attempts_stops_the_retry(self):
        self.reach_a_restoration_retry()
        before = len(self.executor.maintenance_calls)
        self.executor.jobs = ['777|someone-else|their-training|RUNNING']
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        self.assertIn('unapproved job', str(caught.exception).lower())
        self.assertEqual([], self.executor.resumes())
        after = [call for call in self.executor.maintenance_calls[before:]
                 if call and call[0] in ('restore-runtime', 'remount-staging',
                                         'gpu-restore')]
        self.assertEqual([], after,
                         f'the retry mutated the node anyway: {after}')

    def test_a_foreign_drain_appearing_between_attempts_stops_the_retry(self):
        self.reach_a_restoration_retry()
        before = len(self.executor.maintenance_calls)
        self.executor.node_reason = 'hardware-under-investigation'
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        self.assertIn('hardware-under-investigation', str(caught.exception))
        self.assertEqual([], self.executor.resumes())
        after = [call for call in self.executor.maintenance_calls[before:]
                 if call and call[0] in ('restore-runtime', 'remount-staging',
                                         'gpu-restore')]
        self.assertEqual([], after)

    def test_a_job_outside_the_exercise_stops_the_retry(self):
        """Same account, different job name: still not this exercise's work."""
        self.reach_a_restoration_retry()
        self.executor.jobs = ['778|aim344-t1|someones-real-training|RUNNING']
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        self.assertIn('outside this exercise', str(caught.exception).lower())
        self.assertEqual([], self.executor.resumes())

    def test_a_node_returned_to_service_between_attempts_is_re_drained_first(self):
        self.reach_a_restoration_retry()
        self.executor.node_state = 'IDLE+CLOUD'
        self.executor.node_reason = ''
        before = len([c for c in self.executor.slurm_calls
                      if 'State=DRAIN' in c])
        result = self.make().recover()
        drains = [c for c in self.executor.slurm_calls if 'State=DRAIN' in c]
        self.assertGreater(len(drains), before,
                           'the retry mutated an in-service node')
        restore_index = max(
            i for i, c in enumerate(self.executor.maintenance_calls)
            if c and c[0] == 'restore-runtime')
        # The drain was reinstated before the restoration that follows it.
        self.assertEqual('runtime-ready', result.state.phase)
        self.assertGreater(restore_index, 0)

    def test_a_changed_instance_between_attempts_stops_the_retry(self):
        self.reach_a_restoration_retry()
        self.executor.inspect['instance_id'] = 'i-0999999999999999f'
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        self.assertIn('instance', str(caught.exception).lower())
        self.assertEqual([], self.executor.resumes())

    def test_a_clean_restoration_retry_still_succeeds(self):
        """The success path: revalidation must not break the ordinary retry."""
        self.reach_a_restoration_retry()
        reboots = len(self.executor.reboots())
        result = self.make().recover()
        self.assertEqual('runtime-ready', result.state.phase)
        self.assertEqual(1, len(self.executor.resumes()))
        self.assertEqual(reboots, len(self.executor.reboots()),
                         'the retry rebooted again')


# --------------------------------------------------------------------------
# R4: an unobserved reboot is reconciled, never reissued.
# --------------------------------------------------------------------------
class R4ReconcilePendingReboot(Base):
    """A lost observation is not evidence that no reboot happened.

    The rejected code left reboot_completed false when `_await_boot` timed out and
    told the participant to run recover again; that retry selected another reboot
    before comparing boot ids. The state flags this reconciliation reads,
    REBOOT_REQUESTED and REBOOT_ISSUED, are the tokens this Slurm build actually
    emits: they are the only REBOOT_* strings in the deployed libslurm.so.43.0.0
    and libslurmfull.so on the coordinator
    (participant-revision/runs/rework4-read-slurm-reboot-tokens/output.log:8-22).
    """

    def reach_a_timed_out_reboot(self, still_pending=True, came_back=False):
        """Request a reboot, then lose the observation of its completion."""
        handler = self.make()
        handler.start('gpu')
        # _await_boot polls boot-id; keep answering with the pre-reboot value so
        # the observation window closes without seeing the new boot.
        held = self.executor.boot_id

        def unchanged(argv, timeout=None):
            if argv and argv[0] == 'boot-id':
                return session.Completed(0, held + '\n', '')
            return self.original_maintenance(argv, timeout=timeout)

        self.original_maintenance = self.executor.run_maintenance
        self.executor.run_maintenance = unchanged
        # A short wait so the test does not sit through the real 900 seconds.
        self.config_body['assignments']['table-1']['reboot_wait_seconds'] = 0
        self.config_body['assignments']['table-1']['reboot_poll_seconds'] = 0
        self.write_config()
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        self.assertIn('has not returned yet', str(caught.exception))
        state = self.make().store.read()
        self.assertFalse(state.reboot_completed)
        self.assertEqual(1, state.reboot_requests)
        self.assertEqual(1, len(self.executor.reboots()))
        # Now describe what really happened on the node.
        self.executor.run_maintenance = self.original_maintenance
        if came_back:
            self.executor.boot_id = 'cccccccc-0000-0000-0000-000000000003'
            self.executor.node_state = 'IDLE+DRAIN'
        elif still_pending:
            # The node has not rebooted at all yet, so it still reports the boot
            # the round started on. The fake scheduler advances its own boot id
            # when it accepts `scontrol reboot`; put it back, because accepting a
            # request is not the same as completing one.
            self.executor.boot_id = held
            self.executor.node_state = 'IDLE+DRAIN+REBOOT_REQUESTED'
        return state

    def test_a_reboot_that_completed_after_the_deadline_is_not_repeated(self):
        """It came back just after we stopped watching."""
        self.reach_a_timed_out_reboot(came_back=True)
        result = self.make().recover()
        self.assertEqual(1, len(self.executor.reboots()),
                         'the node was rebooted twice')
        self.assertEqual('runtime-ready', result.state.phase)
        self.assertTrue(result.state.reboot_completed)
        self.assertEqual('cccccccc-0000-0000-0000-000000000003',
                         result.state.boot_id_after_reboot)
        self.assertIn('completed while this round was not watching',
                      ' '.join(result.state.notes.split()))
        self.assertEqual(1, len(self.executor.resumes()))

    def test_a_still_pending_reboot_is_not_requested_again(self):
        """The scheduler still holds the request; waiting is the whole remedy."""
        self.reach_a_timed_out_reboot(still_pending=True)
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        message = str(caught.exception)
        self.assertIn('already requested', message)
        self.assertIn('no second reboot', message)
        self.assertEqual(1, len(self.executor.reboots()),
                         'a second reboot was queued')
        self.assertEqual([], self.executor.resumes())
        self.assertEqual(1, self.make().store.read().reboot_requests)

    def test_the_issued_flag_is_recognised_as_pending_too(self):
        self.reach_a_timed_out_reboot(still_pending=True)
        self.executor.node_state = 'IDLE+DRAIN+REBOOT_ISSUED'
        with self.assertRaises(session.Refusal):
            self.make().recover()
        self.assertEqual(1, len(self.executor.reboots()))

    def test_the_request_is_persisted_before_it_is_issued(self):
        """A process that dies between the two must not look like a clean start."""
        handler = self.make()
        handler.start('gpu')
        observed = {}
        original = self.executor.run_slurm

        def watch(argv, timeout=None):
            if Path(argv[0]).name == 'scontrol' and argv[1:2] == ['reboot']:
                observed['requests'] = self.make().store.read().reboot_requests
            return original(argv, timeout=timeout)

        self.executor.run_slurm = watch
        self.make().recover()
        self.assertEqual(1, observed.get('requests'),
                         'the reboot request was not durable before it was made')

    def test_an_unreadable_node_is_assumed_to_have_a_pending_reboot(self):
        """Conservative: the failure to avoid is issuing a second reboot."""
        self.reach_a_timed_out_reboot(still_pending=True)

        def refuse(argv, timeout=None):
            if Path(argv[0]).name == 'scontrol' and argv[1:3] == ['show', 'node']:
                return session.Completed(1, '', 'scontrol: error: Invalid node\n')
            return session.Completed(0, '', '')

        self.executor.run_slurm = refuse
        with self.assertRaises(session.Refusal):
            self.make().recover()
        self.assertEqual(1, len(self.executor.reboots()))

    def test_a_first_reboot_is_still_requested_normally(self):
        """The positive control: reconciliation must not become 'never reboot'."""
        handler = self.make()
        handler.start('gpu')
        result = self.make().recover()
        self.assertEqual(1, len(self.executor.reboots()))
        self.assertEqual(1, result.state.reboot_requests)
        self.assertEqual('runtime-ready', result.state.phase)


# --------------------------------------------------------------------------
# R5: GPU preparation intent is durable before the side effect.
# --------------------------------------------------------------------------
class R5PreparationIntentIsDurable(Base):
    """Process death after telemetry is paused must stay recoverable.

    maintenance.gpu_prepare stops nvidia-dcgm first and disables persistence
    second (maintenance.py:214-229), so a coordinator that dies between those two
    steps left the durable record with no marker, and recovery skipped gpu-restore
    because it only tests those markers.
    """

    def test_the_intent_is_recorded_before_gpu_prepare_is_called(self):
        handler = self.make()
        observed = {}
        original = self.executor.run_maintenance

        def watch(argv, timeout=None):
            if argv and argv[0] == 'gpu-prepare':
                observed['prepared'] = list(
                    self.make().store.read().prepared)
            return original(argv, timeout=timeout)

        self.executor.run_maintenance = watch
        handler.start('gpu')
        self.assertIn('gpu-prepare-attempted', observed.get('prepared', []),
                      'the preparation marker was not durable before the call')

    def test_a_process_death_during_preparation_still_leaves_a_restorable_record(self):
        """The R5 window itself, with no exception to catch.

        The privileged call is killed the way a dead coordinator would leave it:
        it never returns and nothing raises inside start. The durable record must
        still name the preparation, so a later recovery restores what was paused.
        """
        handler = self.make()
        original = self.executor.run_maintenance

        class Died(BaseException):
            """Not a Refusal: nothing in start may catch this."""

        def die_mid_prepare(argv, timeout=None):
            if argv and argv[0] == 'gpu-prepare':
                raise Died()
            return original(argv, timeout=timeout)

        self.executor.run_maintenance = die_mid_prepare
        with self.assertRaises(Died):
            handler.start('gpu')
        state = self.make().store.read()
        self.assertIn('gpu-prepare-attempted', state.prepared)
        self.assertEqual('gpu', state.kind)
        self.assertEqual('preparing', state.phase)
        # And recovery acts on it: gpu-restore runs rather than being skipped.
        self.executor.run_maintenance = original
        result = self.make().recover()
        self.assertIn('gpu-restore', self.executor.actions())
        self.assertEqual('runtime-ready', result.state.phase)

    def test_a_missing_original_state_record_is_not_a_successful_restore(self):
        """gpu_restore refuses without a recorded persistence mode.

        maintenance.py:237-238 raises and returns nonzero, which must arrive here
        as a required-step failure rather than a round that reaches RESUME.
        """
        handler = self.make()
        handler.start('gpu')
        self.executor.maintenance_failures['gpu-restore'] = (
            3, 'No recorded persistence mode to restore.\n')
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        self.assertIn('not fully restored', str(caught.exception))
        self.assertEqual([], self.executor.resumes())
        self.assertEqual('recovery-failed', self.make().store.read().phase)

    def test_the_original_node_state_is_recorded_before_the_first_mutation(self):
        handler = self.make()
        observed = {}
        original = self.executor.run_slurm

        def watch(argv, timeout=None):
            if (Path(argv[0]).name == 'scontrol' and argv[1:2] == ['update']
                    and any('State=DRAIN' in a for a in argv)):
                stored = self.make().store.read()
                observed['original_state'] = stored.original_node_state
                observed['phase'] = stored.phase
            return original(argv, timeout=timeout)

        self.executor.run_slurm = watch
        handler.start('gpu')
        self.assertEqual('IDLE+CLOUD', observed.get('original_state'))
        self.assertEqual('preparing', observed.get('phase'))

    def test_an_efa_round_records_no_gpu_preparation(self):
        """The negative control: the marker must track what actually ran."""
        handler = self.make()
        handler.start('efa')
        state = self.make().store.read()
        self.assertNotIn('gpu-prepare-attempted', state.prepared)
        self.assertNotIn('gpu-prepare', state.prepared)
        self.make().recover()
        self.assertNotIn('gpu-restore', self.executor.actions())


# --------------------------------------------------------------------------
# F11: round and phase evidence, never overwritten.
# --------------------------------------------------------------------------
class F11EvidencePreserved(Base):
    def test_a_recheck_does_not_overwrite_the_fault_time_capture(self):
        handler = self.make()
        handler.start('gpu')
        self.executor.check_output['0'] = FAIL_0
        fault = self.make().collect('0')
        self.assertIn('[FAIL]', fault.saved_path.read_text())
        # Recovery restores the device, so the same check now passes.
        self.executor.check_output['0'] = PASS_0
        self.make().recover()
        self.assertIn('[FAIL]', fault.saved_path.read_text(),
                      'the fault-time capture was overwritten')
        recovered = self.make().store.read().check_results[
            'round-1/recovered/check-0']
        self.assertEqual('PASS', recovered['verdict'])
        self.assertNotEqual(str(fault.saved_path), recovered['log'])
        self.assertIn('[PASS]', Path(recovered['log']).read_text())

    def test_an_explicit_participant_recheck_keeps_both_captures(self):
        handler = self.make()
        handler.start('gpu')
        self.executor.check_output['0'] = FAIL_0
        fault = self.make().collect('0')
        self.executor.check_output['0'] = PASS_0
        self.make().recover()
        after = self.make().collect('0')
        self.assertNotEqual(fault.saved_path, after.saved_path)
        self.assertIn('[FAIL]', fault.saved_path.read_text())
        self.assertIn('[PASS]', after.saved_path.read_text())

    def test_a_second_round_does_not_overwrite_the_first_rounds_evidence(self):
        handler = self.make()
        handler.start('gpu')
        self.executor.check_output['0'] = FAIL_0
        first = self.make().collect('0')
        self.executor.check_output['0'] = PASS_0
        self.make().recover()
        second_round = self.make().start('gpu')
        self.assertEqual(2, second_round.state.round)
        self.executor.check_output['0'] = FAIL_0
        second = self.make().collect('0')
        self.assertNotEqual(first.saved_path, second.saved_path)
        self.assertIn('round-1', str(first.saved_path))
        self.assertIn('round-2', str(second.saved_path))
        self.assertTrue(first.saved_path.is_file())

    def test_repeated_collects_of_one_check_in_one_round_are_all_kept(self):
        handler = self.make()
        handler.start('gpu')
        first = self.make().collect('0')
        second = self.make().collect('0')
        self.assertNotEqual(first.saved_path, second.saved_path)
        self.assertTrue(first.saved_path.is_file())
        self.assertTrue(second.saved_path.is_file())

    def test_the_recorded_results_name_their_round_and_phase(self):
        handler = self.make()
        handler.start('efa')
        self.make().recover()
        keys = sorted(self.make().store.read().check_results)
        self.assertEqual(['round-1/recovered/check-2', 'round-1/recovered/check-6'],
                         keys)


# --------------------------------------------------------------------------
# R9: the active EFA path is qualified before injection, or refused.
# --------------------------------------------------------------------------
class R9ActiveEfaPreconditions(Base):
    """The plan's pre-injection conditions, enforced rather than assumed.

    DEVICE-RECOVERY.md:53 requires two advancing progress records and increasing
    counters on the SELECTED EFA immediately before an active unbind, and an abort
    if progress stopped during preparation. The rejected helper selected an
    own-name-prefixed job (device-session.py:800-801 @0fdf6204), forwarded its id
    to efa-unbind, and checked neither progress nor traffic; device-fault.sh:116-140
    checks ownership and identity only. A queued or idle job called aim344-anything
    therefore qualified the active path.

    Each test drives the observation the target reports, then asserts what the
    helper decided: whether efa-unbind was called at all, and what the round
    recorded about which path it took.
    """

    def active_job(self, job_id='95'):
        self.executor.jobs = [f'{job_id}|aim344-t1|aim344-device|RUNNING']
        return job_id

    def adopt(self, handler):
        """Enable the active path on one handler, as an adoption decision would.

        The participant route refuses the active path on this stack for a measured
        reason (see the class comment on DeviceSession). The qualification itself
        still has to work, and a facilitator-driven active trial needs it, so these
        tests enable it explicitly rather than deleting the coverage. A test that
        needs the refusal does not call this.
        """
        handler.ACTIVE_EFA_PARTICIPANT_PATH_ADOPTED = True
        return handler

    def test_the_participant_route_refuses_the_active_path_before_any_mutation(self):
        """The adoption decision, not silently testing the idle path instead.

        Measured on the pair: the unbind write stayed outstanding 171 s while the
        workload ran and returned 11 s after it was cancelled
        (runs/rework5-unbind-blocking-probe/output.log), so a
        one-command-per-connection route cannot complete this round. The refusal
        must come before the drain, so the participant is not left holding a
        reserved node.
        """
        self.active_job()
        with self.assertRaises(session.Refusal) as raised:
            self.make().start('efa')
        message = str(raised.exception)
        self.assertIn('runs while your node is idle', message)
        self.assertIn('does not return until', message)
        self.assertEqual([], [c for c in self.executor.slurm_calls
                              if 'State=DRAIN' in c],
                         'the node was reserved for a round that cannot proceed')
        for action in ('efa-unbind', 'efa-activity', 'collect'):
            self.assertNotIn(action, self.executor.actions())
        self.assertEqual('ready', self.make().store.read().phase)

    def test_an_idle_efa_round_is_unaffected_by_that_refusal(self):
        """The positive control: the round the exercise actually ships still runs."""
        self.executor.jobs = []
        self.make().start('efa')
        self.assertEqual(1, len([c for c in self.executor.maintenance_calls
                                 if c[0] == 'efa-unbind']))

    def test_a_qualified_active_workload_is_injected_and_recorded(self):
        """The success path first, so this class is not merely a rejector."""
        job = self.active_job()
        handler = self.adopt(self.make())
        handler.start('efa')
        unbinds = [c for c in self.executor.maintenance_calls
                   if c[0] == 'efa-unbind']
        self.assertEqual(1, len(unbinds), self.executor.maintenance_calls)
        self.assertIn('--job', unbinds[0])
        self.assertEqual(job, unbinds[0][unbinds[0].index('--job') + 1])
        evidence = self.make().store.read().active_efa_evidence
        self.assertIsNotNone(evidence)
        assert evidence is not None
        self.assertTrue(evidence['qualified'], evidence)
        self.assertEqual([], evidence['refusals'])
        self.assertEqual('rdmap176s0', evidence['device'])
        self.assertGreaterEqual(evidence['moved_bytes'],
                                session.DeviceSession.ACTIVE_EFA_MIN_BYTES)

    def test_the_observation_is_made_before_the_unbind_not_after(self):
        self.active_job()
        self.adopt(self.make()).start('efa')
        actions = self.executor.actions()
        self.assertIn('efa-activity', actions)
        self.assertLess(actions.index('efa-activity'), actions.index('efa-unbind'),
                        'the device was unbound before its activity was observed')

    def test_a_job_that_moved_no_bytes_on_the_selected_device_is_refused(self):
        """The central R9 case: a named job that is not using the device."""
        self.active_job()
        self.executor.efa_activity_moved_bytes = 0
        with self.assertRaises(session.Refusal) as raised:
            self.adopt(self.make()).start('efa')
        self.assertIn('moved 0 B', str(raised.exception))
        self.assertEqual([], [c for c in self.executor.maintenance_calls
                              if c[0] == 'efa-unbind'])
        evidence = self.make().store.read().active_efa_evidence
        assert evidence is not None
        self.assertFalse(evidence['qualified'])

    def test_traffic_just_below_the_threshold_is_refused(self):
        """A boundary case, so the threshold is a real condition."""
        self.active_job()
        self.executor.efa_activity_moved_bytes = (
            session.DeviceSession.ACTIVE_EFA_MIN_BYTES - 1)
        with self.assertRaises(session.Refusal):
            self.adopt(self.make()).start('efa')
        self.assertEqual([], [c for c in self.executor.maintenance_calls
                              if c[0] == 'efa-unbind'])

    def test_traffic_exactly_at_the_threshold_is_accepted(self):
        """The other side of the same boundary."""
        self.active_job()
        self.executor.efa_activity_moved_bytes = (
            session.DeviceSession.ACTIVE_EFA_MIN_BYTES)
        self.adopt(self.make()).start('efa')
        self.assertEqual(1, len([c for c in self.executor.maintenance_calls
                                 if c[0] == 'efa-unbind']))

    def test_absent_counters_are_not_read_as_absent_traffic(self):
        self.active_job()
        self.executor.efa_activity_no_counters = True
        with self.assertRaises(session.Refusal) as raised:
            self.adopt(self.make()).start('efa')
        self.assertIn('unestablished', str(raised.exception))
        self.assertEqual([], [c for c in self.executor.maintenance_calls
                              if c[0] == 'efa-unbind'])

    def test_a_job_that_has_barely_started_is_refused(self):
        """Advancing collectives take time; a one-second-old job has none."""
        self.active_job()
        self.executor.efa_activity_runtime = '0:03'
        with self.assertRaises(session.Refusal) as raised:
            self.adopt(self.make()).start('efa')
        self.assertIn('had run for 3 s', str(raised.exception))
        self.assertEqual([], [c for c in self.executor.maintenance_calls
                              if c[0] == 'efa-unbind'])

    def test_a_workload_that_stopped_during_preparation_aborts_the_injection(self):
        """The plan's explicit abort condition.

        The job is present and running when the window opens and gone when it
        closes, which is exactly 'progress stopped during preparation'.
        """
        self.active_job()
        self.executor.efa_activity_jobs_after = []
        with self.assertRaises(session.Refusal) as raised:
            self.adopt(self.make()).start('efa')
        self.assertIn('was not on the node when the observation after',
                      str(raised.exception))
        self.assertEqual([], [c for c in self.executor.maintenance_calls
                              if c[0] == 'efa-unbind'])

    def test_a_job_that_stopped_running_during_preparation_is_refused(self):
        self.active_job()
        self.executor.efa_activity_jobs_after = [
            {'id': '95', 'user': 'aim344-t1', 'name': 'aim344-device',
             'state': 'COMPLETING', 'runtime': '2:10'}]
        with self.assertRaises(session.Refusal) as raised:
            self.adopt(self.make()).start('efa')
        self.assertIn('COMPLETING', str(raised.exception))
        self.assertEqual([], [c for c in self.executor.maintenance_calls
                              if c[0] == 'efa-unbind'])

    def test_a_different_job_at_the_end_of_the_window_is_refused(self):
        """The workload about to be faulted must be the one still running."""
        self.active_job()
        self.executor.efa_activity_jobs_after = [
            {'id': '96', 'user': 'aim344-t1', 'name': 'aim344-device',
             'state': 'RUNNING', 'runtime': '0:20'}]
        with self.assertRaises(session.Refusal):
            self.adopt(self.make()).start('efa')
        self.assertEqual([], [c for c in self.executor.maintenance_calls
                              if c[0] == 'efa-unbind'])

    def test_an_unreadable_queue_is_not_a_qualification(self):
        self.active_job()
        self.executor.efa_activity_queue_readable = False
        with self.assertRaises(session.Refusal) as raised:
            self.adopt(self.make()).start('efa')
        self.assertIn('queue could not be read', str(raised.exception))

    def test_an_observation_that_did_not_complete_refuses_the_round(self):
        self.active_job()
        self.executor.maintenance_failures['efa-activity'] = (
            3, 'No byte counters are exposed for rdmap176s0\n')
        with self.assertRaises(session.Refusal) as raised:
            self.adopt(self.make()).start('efa')
        self.assertIn('did not complete', str(raised.exception))
        self.assertEqual([], [c for c in self.executor.maintenance_calls
                              if c[0] == 'efa-unbind'])

    def test_a_refused_active_round_leaves_the_node_reserved_and_untouched(self):
        """A refusal is not a half-applied round."""
        self.active_job()
        self.executor.efa_activity_moved_bytes = 0
        with self.assertRaises(session.Refusal):
            self.adopt(self.make()).start('efa')
        state = self.make().store.read()
        self.assertEqual('preparing', state.phase)
        self.assertIn('drain', state.prepared)
        self.assertEqual([], [c for c in self.executor.slurm_calls
                              if 'State=RESUME' in c])
        for action in ('efa-unbind', 'gpu-remove'):
            self.assertNotIn(action, self.executor.actions())

    def test_an_idle_efa_round_is_not_silently_treated_as_active(self):
        """An idle round is legitimate; it must simply not claim the active path.

        With no job on the node the unbind carries no --job and the round records
        no active-workload observation, so a reader can tell which path ran.
        """
        self.executor.jobs = []
        self.make().start('efa')
        unbinds = [c for c in self.executor.maintenance_calls
                   if c[0] == 'efa-unbind']
        self.assertEqual(1, len(unbinds))
        self.assertNotIn('--job', unbinds[0])
        self.assertNotIn('efa-activity', self.executor.actions())
        self.assertIsNone(self.make().store.read().active_efa_evidence)

    def test_status_names_the_path_the_round_took(self):
        self.executor.jobs = []
        idle = self.make().start('efa')
        rendered = session.render(self.make().status(), 'status')
        self.assertIn('EFA path: idle', rendered)
        self.assertEqual('efa', idle.state.kind)
        self.make().recover()
        self.active_job()
        self.adopt(self.make()).start('efa')
        rendered = session.render(self.make().status(), 'status')
        self.assertIn('EFA path: active, qualified=True', rendered)
        self.assertIn('rdmap176s0', rendered)

    def test_a_gpu_round_does_not_observe_efa_activity(self):
        """The condition belongs to the active EFA path only."""
        self.make().start('gpu')
        self.assertNotIn('efa-activity', self.executor.actions())
        self.assertIsNone(self.make().store.read().active_efa_evidence)

    def test_the_slurm_elapsed_parser_reads_every_shape_slurm_prints(self):
        """Exercised on both readable and unreadable input.

        A parser validated only on what it rejects, or only on what it accepts, is
        half verified. Slurm prints MM:SS, HH:MM:SS and D-HH:MM:SS, and prints
        INVALID for a job that has not started.
        """
        parse = session.DeviceSession._parse_slurm_runtime
        self.assertEqual(0, parse('0:00'))
        self.assertEqual(3, parse('0:03'))
        self.assertEqual(125, parse('2:05'))
        self.assertEqual(3725, parse('1:02:05'))
        self.assertEqual(90125, parse('1-01:02:05'))
        for unreadable in ('INVALID', '', None, 'N/A', 'abc', '1:2:3:4'):
            self.assertIsNone(parse(unreadable), repr(unreadable))

    def test_an_unreadable_elapsed_time_is_not_treated_as_long_enough(self):
        """The parser returning None must refuse, not pass by omission."""
        self.active_job()
        self.executor.efa_activity_runtime = 'INVALID'
        with self.assertRaises(session.Refusal) as raised:
            self.adopt(self.make()).start('efa')
        self.assertIn('unreadable elapsed time', str(raised.exception))
        self.assertEqual([], [c for c in self.executor.maintenance_calls
                              if c[0] == 'efa-unbind'])

    def test_the_observation_is_filed_as_participant_evidence(self):
        self.active_job()
        self.adopt(self.make()).start('efa')
        evidence = self.make().store.read().active_efa_evidence
        assert evidence is not None
        self.assertIsNotNone(evidence['log'])
        saved = Path(evidence['log'])
        self.assertTrue(saved.is_file(), saved)
        self.assertIn('active-precondition', saved.name)
        record = json.loads(saved.read_text())
        self.assertEqual('efa-activity', record['action'])
        self.assertEqual('rdmap176s0', record['device'])


if __name__ == '__main__':
    unittest.main()
