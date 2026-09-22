# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Repairs for the re-review's remaining safety findings on `3ce7e192`.

S4-D and S2 are decisions the shipped controller makes, so each test here drives
the real `start`/`recover` verbs. Where the decision depends on what the target's
maintenance helper actually does, the fake executor calls the REAL
`maintenance.gpu_prepare`/`gpu_restore` rather than answering for them: the finding
the S4-D tests are about was previously invisible because the bench used an
executor that returned a successful `gpu-restore` without reading any record
(test_device_session_rework.py:81-83). A stub that always succeeds cannot
distinguish a round whose originals exist from one that never took any.

The `systemctl`/`nvidia-smi` programs those real functions invoke are the same
recording stubs `test_safety_rework3.py` uses, so no test asserts a return code a
mock was told to produce.

Run: python3 -m unittest discover -s <lab>/tests -p 'test_*.py'
"""
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

LAB = Path(__file__).resolve().parents[1]

_rework_spec = importlib.util.spec_from_file_location(
    'aim344_rework_safety5',
    Path(__file__).resolve().parent / 'test_device_session_rework.py')
if _rework_spec is None or _rework_spec.loader is None:        # pragma: no cover
    raise unittest.SkipTest('test_device_session_rework.py not importable')
_rework = importlib.util.module_from_spec(_rework_spec)
_rework_spec.loader.exec_module(_rework)
session = _rework.session
Base = _rework.Base
ReworkExecutor = _rework.ReworkExecutor

_safety3_spec = importlib.util.spec_from_file_location(
    'aim344_safety3_for_rework5',
    Path(__file__).resolve().parent / 'test_safety_rework3.py')
if _safety3_spec is None or _safety3_spec.loader is None:      # pragma: no cover
    raise unittest.SkipTest('test_safety_rework3.py not importable')
_safety3 = importlib.util.module_from_spec(_safety3_spec)
_safety3_spec.loader.exec_module(_safety3)
maintenance = _safety3.maintenance
SYSTEMCTL_STUB = _safety3.SYSTEMCTL_STUB
NVIDIA_SMI_STUB = _safety3.NVIDIA_SMI_STUB

# The pristine functions, captured once at import. A test that calls `setUp` again
# inside its own body -- which the journal-vs-phase test does, deliberately, to build
# two rounds -- must not save an already-patched callable as "the original", or the
# restore in tearDown reinstates the patch and the next test recurses. Captured here
# rather than per-instance so that cannot happen.
REAL_READ_TEXT = Path.read_text
REAL_SUBPROCESS_RUN = maintenance.subprocess.run


class TargetHelperExecutor(ReworkExecutor):
    """A fake scheduler side, with a REAL maintenance helper on the target side.

    `gpu-prepare` and `gpu-restore` are executed by calling
    `maintenance.gpu_prepare`/`maintenance.gpu_restore` for real, against real record
    files under a temporary RECORD_DIR and stub `systemctl`/`nvidia-smi` programs.
    Their stdout, stderr and return code are handed back to the controller exactly as
    the ssh forced command would hand them back, including a Refusal's message on
    stderr with rc 3 (maintenance.main's own mapping).

    Everything else -- scontrol, squeue, inspect, boot-id, collect -- stays the
    recorded fake, because this class is about the controller's interaction with the
    target's refusal semantics, not about running a cluster.
    """

    def __init__(self, records, config, unit_state='active', load_state='loaded'):
        super().__init__()
        self.records = records
        self.target_config = config
        self.unit_state = unit_state
        self.load_state = load_state
        self.helper_calls = []

    def _run_helper(self, function, argv):
        # Exercise the actual forced-command grammar, including first-capture
        # authorization; a hand-written parser can silently discard the flag.
        _, options = maintenance.parse(argv)
        stream = io.StringIO()
        saved_stdout, saved_record_dir = sys.stdout, maintenance.RECORD_DIR
        sys.stdout = stream
        maintenance.RECORD_DIR = self.records
        try:
            code = function(self.target_config, options)
            return session.Completed(code, stream.getvalue(), '')
        except maintenance.Refusal as refusal:
            # maintenance.main writes a Refusal to stderr and returns 3.
            return session.Completed(3, stream.getvalue(), str(refusal) + '\n')
        finally:
            sys.stdout = saved_stdout
            maintenance.RECORD_DIR = saved_record_dir

    def run_maintenance(self, argv, timeout=None):
        action = argv[0] if argv else ''
        if action in ('gpu-prepare', 'gpu-restore'):
            # A scripted failure wins, so a test can still describe a transport
            # failure or an answer the target could not have produced. Only when
            # nothing is scripted does the real helper run.
            if action in self.maintenance_failures:
                return super().run_maintenance(argv, timeout=timeout)
            self.maintenance_calls.append(list(argv))
            self.helper_calls.append(list(argv))
            function = (maintenance.gpu_prepare if action == 'gpu-prepare'
                        else maintenance.gpu_restore)
            return self._run_helper(function, argv)
        return super().run_maintenance(argv, timeout=timeout)


class RealTargetHelper(Base):
    """Base plus the stub programs and record directory the real helper needs."""

    def setUp(self):
        super().setUp()
        root = Path(self.tmp.name)
        self.records = root / 'target-records'
        self.records.mkdir()
        self.stubs = root / 'stubs'
        self.stubs.mkdir()
        self.stub_state = root / 'stub-state'
        self.stub_state.mkdir()
        (self.stub_state / 'unit-state.txt').write_text('active\n')
        (self.stub_state / 'unit-load-state.txt').write_text('loaded\n')
        (self.stub_state / 'persistence-mode.txt').write_text('Enabled\n')
        for name, body in (('systemctl', SYSTEMCTL_STUB),
                           ('nvidia-smi', NVIDIA_SMI_STUB)):
            path = self.stubs / name
            path.write_text(body)
            path.chmod(0o755)
        self.fault_config = root / 'aim344-device-fault.json'
        self.fault_config.write_text(json.dumps({
            'gpu_uuid': 'GPU-11111111-2222-4333-8444-555555555555',
            'efa_rdma_device': 'rdmap176s0',
            'slurm_node': 'gpu-g7-1',
        }))
        self.target_config = {'slurm_bin': '/opt/aws/pcs/scheduler/slurm-25.05/bin',
                              'telemetry_settle_attempts': 2,
                              'telemetry_settle_pause_seconds': 0}

        stubs, stub_state, fault_config = self.stubs, self.stub_state, self.fault_config
        real_run = REAL_SUBPROCESS_RUN
        real_read_text = REAL_READ_TEXT

        def redirected(argv, **kwargs):
            argv = list(argv)
            if argv and str(argv[0]).startswith('/usr/bin/'):
                name = Path(argv[0]).name
                if (stubs / name).is_file():
                    argv[0] = str(stubs / name)
            env = dict(kwargs.pop('env', None) or os.environ)
            env['AIM344_STUB_DIR'] = str(stub_state)
            return real_run(argv, env=env, **kwargs)

        maintenance.subprocess.run = redirected

        def read_text(path, *args, **kwargs):
            if str(path) == '/etc/aim344-device-fault.json':
                return real_read_text(fault_config, *args, **kwargs)
            return real_read_text(path, *args, **kwargs)

        Path.read_text = read_text
        self.executor = TargetHelperExecutor(self.records, self.target_config)

    def tearDown(self):
        maintenance.subprocess.run = REAL_SUBPROCESS_RUN
        Path.read_text = REAL_READ_TEXT
        super().tearDown()
    # -- what the target's own state looks like, for the tests to arrange --
    def set_unit(self, active_state, load_state='loaded'):
        (self.stub_state / 'unit-state.txt').write_text(active_state + '\n')
        (self.stub_state / 'unit-load-state.txt').write_text(load_state + '\n')

    def unit_now(self):
        return (self.stub_state / 'unit-state.txt').read_text().strip()

    def persistence_now(self):
        return (self.stub_state / 'persistence-mode.txt').read_text().strip()

    def records_present(self):
        return sorted(p.name for p in self.records.iterdir())


class FreshCaptureCaller(RealTargetHelper):
    """First-start authorization travels through the real parser and helper."""

    def test_only_new_start_authorizes_capture_and_recovery_reuses_identity(self):
        started = self.make().start('gpu')
        token = started.state.operation
        calls = list(self.executor.helper_calls)
        print('FIRST_START_HELPER_ARGV', calls)
        self.assertEqual([['gpu-prepare', '--operation', token, '--fresh-capture']], calls)
        self.assertEqual('inactive', self.unit_now())
        self.assertEqual('Disabled', self.persistence_now())
        again = self.make().start('gpu')
        self.assertTrue(again.already_started)
        self.assertEqual(token, again.state.operation)
        self.assertEqual(calls, self.executor.helper_calls)
        self.make().recover()
        print('AFTER_RECOVERY_HELPER_ARGV', self.executor.helper_calls)
        self.assertEqual(['gpu-restore', '--operation', token],
                         self.executor.helper_calls[-1])
        self.assertEqual('active', self.unit_now())
        self.assertEqual('Enabled', self.persistence_now())
        next_round = self.make().start('gpu')
        self.assertNotEqual(token, next_round.state.operation)
        self.assertEqual(['gpu-prepare', '--operation', next_round.state.operation,
                          '--fresh-capture'], self.executor.helper_calls[-1])
        print('NEXT_ROUND_HELPER_ARGV', self.executor.helper_calls[-1])

    def test_failed_first_capture_is_not_resent_by_start_or_recover(self):
        self.set_unit('failed')
        with self.assertRaises(session.Refusal):
            self.make().start('gpu')
        calls = list(self.executor.helper_calls)
        self.assertIn('--fresh-capture', calls[0])
        self.make().start('gpu')
        with self.assertRaises(session.Refusal):
            self.make().recover()
        print('REFUSED_ROUND_HELPER_ARGV', self.executor.helper_calls)
        self.assertEqual(calls, self.executor.helper_calls)
        self.assertEqual([], self.records_present())
        self.assertEqual([], self.executor.resumes())

    def test_ready_label_does_not_authorize_recapture_of_an_existing_operation(self):
        self.make().start('gpu')
        store = self.make().store
        state = store.read()
        state.phase = 'ready'
        store.write(state)
        before = {p.name: p.read_text() for p in self.records.iterdir()}
        calls = list(self.executor.helper_calls)
        try:
            with self.assertRaises(session.Refusal):
                self.make().start('gpu')
        finally:
            print('READY_WITH_EXISTING_OPERATION_HELPER_ARGV', self.executor.helper_calls)
        self.assertEqual(calls, self.executor.helper_calls)
        self.assertEqual(before, {p.name: p.read_text() for p in self.records.iterdir()})


# ---------------------------------------------------------------------------
# S4-D: recovery must not reboot a round whose preparation was refused before
# any original state was captured and before any device was touched.
# ---------------------------------------------------------------------------
class S4DRefusedPreparationRecovery(RealTargetHelper):
    """The controller's decision, with the target's real refusal semantics.

    The review's counterexample, in the order the code takes it:
    `device-session.py` persists `preparing`, drains, captures the baseline and
    journals `gpu-prepare-attempted`; `maintenance.gpu_prepare` refuses a telemetry
    original it cannot restore, before writing either record; `start` keeps the
    failed preparation and tells the participant to recover; `recover` used to set
    `needs_reboot=True` unconditionally for a GPU round and issue `scontrol reboot`.
    The reboot changed a service whose state had just been judged unrestorable, and
    the `gpu-restore` that follows had no valid originals to undo it with.
    """

    def refused_before_capture(self, load_state='loaded', active_state='failed'):
        """Start a GPU round whose preparation the target refuses before recording.

        `failed` on a loaded unit and any state on a `not-found` unit are both
        refusals `maintenance.gpu_prepare` makes before its first `_write_record`.
        Returns the recorded state after `start` has raised.
        """
        self.set_unit(active_state, load_state)
        handler = self.make()
        with self.assertRaises(session.Refusal):
            handler.start('gpu')
        return self.make().store.read()

    def test_the_target_really_refused_before_taking_any_original(self):
        """The premise, established on the real helper rather than assumed.

        If the helper actually recorded something, the rest of this class would be
        testing a different situation. So this asserts the target's own side: no
        record file exists, telemetry was not paused and persistence was not changed.
        """
        state = self.refused_before_capture()
        self.assertEqual([], self.records_present(),
                         'the refused preparation still wrote an original-state record')
        self.assertEqual('failed', self.unit_now(),
                         'the refused preparation changed the unit')
        self.assertEqual('Enabled', self.persistence_now(),
                         'the refused preparation changed persistence')
        self.assertIn('gpu-prepare-attempted', state.prepared)
        self.assertIn('gpu-prepare-refused-before-capture', state.prepared)
        self.assertNotIn('gpu-prepare-recorded', state.prepared)
        self.assertNotIn('gpu-remove-attempted', state.prepared,
                         'a device mutation was issued after a refused preparation')

    def test_recovery_does_not_reboot_a_round_refused_before_capture(self):
        """The finding itself: no reboot, no remount, no restoration mutations.

        The behavioural assertions come first, deliberately. A message assertion that
        fires before them would make this test's failure on the pre-repair artifact a
        wording difference rather than a demonstration that the node was cycled.
        """
        self.refused_before_capture()
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        self.assertEqual([], self.executor.reboots(),
                         'a node was rebooted for a round that changed nothing')
        actions = self.executor.actions()
        for mutation in ('remount-staging', 'restore-runtime', 'gpu-restore',
                         'gpu-remove', 'efa-rebind'):
            self.assertNotIn(mutation, actions,
                             f'{mutation} ran for a round that changed nothing')
        self.assertEqual([], self.executor.resumes())
        self.assertIn('did not modify', str(caught.exception))

    def test_the_node_keeps_its_isolation_and_the_round_is_reported(self):
        """The permitted outcome: keep isolation, report, do not invent a rollback."""
        self.refused_before_capture()
        with self.assertRaises(session.Refusal):
            self.make().recover()
        state = self.make().store.read()
        self.assertEqual('recovery-failed', state.phase)
        self.assertIsNone(state.recovered_by)
        # The drain the round took is still in place: the node is isolated, not
        # returned to service and not cycled.
        self.assertIn('drain', state.prepared)
        self.assertEqual('IDLE+DRAIN', self.executor.node_state)
        self.assertEqual(self.config_body['assignments']['table-1']['drain_reason'],
                         self.executor.node_reason)
        self.assertEqual([], self.executor.reboots())
        self.assertIn('nothing to put back', state.failure)

    def test_a_repeated_recovery_attempt_still_does_not_reboot(self):
        """Retrying must not accumulate into a reboot."""
        self.refused_before_capture()
        for attempt in range(3):
            with self.assertRaises(session.Refusal):
                self.make().recover()
        self.assertEqual([], self.executor.reboots())
        self.assertEqual(0, self.make().store.read().reboot_requests)

    def test_a_missing_unit_refusal_is_the_same_failure_class(self):
        """Not only the named `failed` state.

        The S4 unit-validity repair adds a second way `gpu_prepare` refuses before
        recording: a unit that is not loaded. It must reach the same controller
        decision, because the property that matters is 'no original was taken', not
        which check produced the refusal.
        """
        state = self.refused_before_capture(load_state='not-found',
                                            active_state='inactive')
        self.assertIn('gpu-prepare-refused-before-capture', state.prepared)
        self.assertEqual([], self.records_present())
        with self.assertRaises(session.Refusal):
            self.make().recover()
        self.assertEqual([], self.executor.reboots())

    # -- the cases that must stay recoverable --
    def test_a_capture_that_succeeded_then_failed_still_reboots_and_restores(self):
        """Preparation recorded the originals and then failed: recovery proceeds.

        The review asked for this explicitly. The target writes both records, pauses
        telemetry, and then fails its persistence verification -- reproduced here by
        letting the real helper record and pause, then making the persistence write
        leave the wrong value. Telemetry is paused, so the round MUST reboot and
        restore.
        """
        self.set_unit('active', 'loaded')
        # nvidia-smi accepts --persistence-mode=0 but the query keeps answering
        # Enabled, which is the shape of the real 'Persistence is still Enabled'
        # refusal: it happens after both records are written.
        (self.stub_state / 'persistence-sticky').write_text('Enabled\n')
        handler = self.make()
        with self.assertRaises(session.Refusal) as caught:
            handler.start('gpu')
        self.assertIn('Persistence is still', str(caught.exception))
        state = self.make().store.read()
        self.assertIn('gpu-prepare-recorded', state.prepared,
                      'the originals were recorded, so the round is recoverable')
        self.assertNotIn('gpu-prepare-refused-before-capture', state.prepared)
        self.assertIn('native-dcgm-state.txt', self.records_present())
        self.assertEqual('inactive', self.unit_now(),
                         'this case is only interesting if telemetry was paused')
        # Recovery proceeds: reboot, remount, restore-runtime, gpu-restore.
        result = self.make().recover()
        self.assertEqual(1, len(self.executor.reboots()))
        self.assertIn('gpu-restore', self.executor.actions())
        self.assertEqual('runtime-ready', result.state.phase)
        # And the real gpu_restore put the recorded original back.
        self.assertEqual('active', self.unit_now())

    def test_a_round_that_removed_the_gpu_still_reboots(self):
        """An actual device mutation always stays recoverable."""
        self.set_unit('active', 'loaded')
        handler = self.make()
        handler.start('gpu')
        state = self.make().store.read()
        self.assertIn('gpu-remove-attempted', state.prepared)
        result = self.make().recover()
        self.assertEqual(1, len(self.executor.reboots()))
        self.assertEqual('runtime-ready', result.state.phase)
        self.assertEqual(1, len(self.executor.resumes()))

    def test_a_mutation_that_did_not_report_success_still_reboots(self):
        """The device operation failed, so the device state is unknown.

        The marker is journalled before the call, so this round is recoverable even
        though nothing established that the removal happened. That is the direction
        the gate must fail in: unknown means recover.
        """
        self.set_unit('active', 'loaded')
        self.executor.maintenance_failures['gpu-remove'] = (
            1, 'write error: Device or resource busy\n')
        with self.assertRaises(session.Refusal):
            self.make().start('gpu')
        state = self.make().store.read()
        self.assertIn('gpu-remove-attempted', state.prepared)
        result = self.make().recover()
        self.assertEqual(1, len(self.executor.reboots()))
        self.assertEqual('runtime-ready', result.state.phase)

    def test_an_unreachable_target_during_preparation_stays_recoverable(self):
        """A transport failure establishes nothing about what the target did.

        rc 255 is the ssh transport, not the helper, so this round's originals may
        exist and its telemetry may be paused. It must NOT be treated as a refusal
        before capture: recovery proceeds, and it is `gpu-restore` -- reading the real
        record files -- that decides whether there was anything to put back. Here
        there was not, so the round fails closed at restoration with its drain
        intact, which is a different outcome from refusing to recover at all.
        """
        self.set_unit('active', 'loaded')
        self.executor.maintenance_failures['gpu-prepare'] = (
            255, 'ssh: connect to host gpu-g7-1 port 22: Connection refused\n')
        with self.assertRaises(session.Refusal):
            self.make().start('gpu')
        state = self.make().store.read()
        self.assertNotIn('gpu-prepare-refused-before-capture', state.prepared)
        del self.executor.maintenance_failures['gpu-prepare']
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        # Recovery ran: the reboot happened and the restoration was attempted.
        self.assertEqual(1, len(self.executor.reboots()))
        self.assertIn('gpu-restore', self.executor.actions())
        # And the REAL gpu_restore refused, because no record for this operation
        # exists. The node keeps its drain rather than resuming.
        self.assertIn('not fully restored', str(caught.exception))
        self.assertIn('No recorded persistence mode', str(caught.exception))
        self.assertEqual([], self.executor.resumes())
        self.assertEqual('recovery-failed', self.make().store.read().phase)

    def test_an_efa_round_is_unaffected_by_the_gate(self):
        """No GPU preparation is involved, and the rebind path is unchanged."""
        self.set_unit('active', 'loaded')
        handler = self.make()
        handler.start('efa')
        state = self.make().store.read()
        self.assertIn('efa-unbind-attempted', state.prepared)
        self.executor.check_output['6'] = _rework.PASS_6
        result = self.make().recover()
        self.assertEqual([], self.executor.reboots())
        self.assertEqual('runtime-ready', result.state.phase)

    def test_the_mutation_marker_is_durable_before_the_call(self):
        """A marker written after the call cannot record one that never returned."""
        self.set_unit('active', 'loaded')
        handler = self.make()
        observed = {}
        original = self.executor.run_maintenance

        def watch(argv, timeout=None):
            if argv and argv[0] == 'gpu-remove':
                observed['prepared'] = list(self.make().store.read().prepared)
            return original(argv, timeout=timeout)

        self.executor.run_maintenance = watch
        handler.start('gpu')
        self.assertIn('gpu-remove-attempted', observed.get('prepared', []),
                      'the mutation marker was not durable before the call')

    def test_the_gate_reads_the_journal_not_the_phase(self):
        """Phase cannot answer the question, so the gate must not consult it.

        The two rounds that matter are both left at `preparing`: one whose device
        mutation was issued but never returned -- a coordinator that died mid-call,
        which is the window the preparation markers exist for -- and one whose
        preparation the target refused before recording anything. Phase is identical,
        the journal is not, and the decisions must differ.

        They are run in sequence on the same node because a round in progress holds
        it: the first is recovered to completion, which releases it, before the second
        starts.
        """
        # Round one: gpu-remove is issued and never returns.
        self.set_unit('active', 'loaded')

        class Died(BaseException):
            """Not a Refusal: nothing in start may catch this."""

        original = self.executor.run_maintenance

        def die_mid_mutation(argv, timeout=None):
            if argv and argv[0] == 'gpu-remove':
                raise Died()
            return original(argv, timeout=timeout)

        self.executor.run_maintenance = die_mid_mutation
        with self.assertRaises(Died):
            self.make().start('gpu')
        self.executor.run_maintenance = original
        mutated = self.make().store.read()
        self.assertEqual('preparing', mutated.phase)
        self.assertIn('gpu-remove-attempted', mutated.prepared)
        # It recovers: same phase, and the gate lets it through.
        result = self.make().recover()
        self.assertEqual(1, len(self.executor.reboots()))
        self.assertEqual('runtime-ready', result.state.phase)

        # Round two, on the same node, refused before capture.
        refused = self.refused_before_capture()
        self.assertEqual(mutated.phase, refused.phase,
                         'these two rounds are separated by phase, so this test has '
                         'no force')
        self.assertNotIn('gpu-remove-attempted', refused.prepared)
        with self.assertRaises(session.Refusal):
            self.make().recover()
        self.assertEqual(1, len(self.executor.reboots()),
                         'the round that changed nothing was rebooted anyway')


class S4DUnknownPreparationOutcome(RealTargetHelper):
    """The other side of the S4-D gate: an UNKNOWN outcome must still recover.

    The gate distinguishes three cases, and only one of them may skip recovery. The
    class above covers the established pre-capture refusal. This one covers the case
    that must NOT be treated as one, and it was a real defect: the classifier accepted
    any `returncode not in (0, 255)` with non-empty output as an established refusal.

    An observation deadline produces exactly that shape. `Executor._run` returns
    `Completed(124, partial_stdout, '...remote command termination is not
    established.')` (device-session.py:187-197), and `maintenance.main` independently
    returns 124 for its own `subprocess.TimeoutExpired`. Neither establishes that the
    helper stopped -- and worse, a killed run never flushes its block-buffered stdout,
    so the `recorded persistence_mode=` marker is ABSENT even when both records were
    written and telemetry was already paused. The round is then refused restoration on
    the grounds that it "did not modify" a node it may well have modified.

    Only `maintenance.main`'s own refusal status (3) establishes a decision.
    """

    def timed_out_preparation(self, returncode=124):
        """Start a GPU round whose gpu-prepare answers without establishing a stop.

        The unit is loaded/active and persistence Enabled, i.e. a state the real
        helper WOULD record and then mutate -- so "nothing was changed" is not
        available as an explanation for the missing marker.
        """
        self.set_unit('active', 'loaded')
        self.executor.maintenance_failures['gpu-prepare'] = (
            returncode,
            'External observation deadline reached; remote command termination is '
            'not established.\n')
        with self.assertRaises(session.Refusal):
            self.make().start('gpu')
        return self.make().store.read()

    def test_a_deadline_is_not_an_established_refusal(self):
        state = self.timed_out_preparation()
        self.assertIn('gpu-prepare-attempted', state.prepared)
        self.assertNotIn(
            'gpu-prepare-refused-before-capture', state.prepared,
            'an observation deadline was journalled as an established pre-capture '
            'refusal, so recovery would decline to restore this round')

    def test_recovery_proceeds_for_an_unknown_preparation_outcome(self):
        """A message assertion: no phrase claiming the node was untouched.

        This checks what recovery SAYS, not which calls it made, so it is classified as
        a message/state check rather than as behavioural evidence. The tests that assert
        the reboot and restoration calls themselves are elsewhere in this class.
        """
        self.timed_out_preparation()
        try:
            self.make().recover()
        except session.Refusal as refusal:
            message = str(refusal)
        else:
            message = ''
        # It may still fail closed further along -- with no records on the target,
        # gpu-restore refuses, which is correct. What it must not do is claim the node
        # was untouched and skip recovery.
        self.assertNotIn(
            'did not modify', message,
            'recovery claimed the node was not modified after a preparation whose '
            'outcome was never established')
        state = self.make().store.read()
        self.assertNotIn('nothing to put back', state.failure or '')

    def test_a_non_root_invocation_is_not_an_established_refusal(self):
        """rc 1 is `main`'s non-root path, not a Refusal. Same class of defect."""
        state = self.timed_out_preparation(returncode=1)
        self.assertNotIn('gpu-prepare-refused-before-capture', state.prepared)

    def test_the_helpers_own_refusal_status_is_still_honoured(self):
        """The positive control: rc 3 with no marker still stops the reboot.

        Without this, the fix above could have been "never classify a refusal", which
        would silently undo the S4-D repair.
        """
        self.set_unit('failed', 'loaded')
        with self.assertRaises(session.Refusal):
            self.make().start('gpu')
        state = self.make().store.read()
        self.assertIn('gpu-prepare-refused-before-capture', state.prepared)
        with self.assertRaises(session.Refusal):
            self.make().recover()
        self.assertEqual([], self.executor.reboots(),
                         'an established pre-capture refusal reached a reboot')


class S4DNeverPreparedRecovery(RealTargetHelper):
    """A round that stopped BEFORE it called anything on the node.

    The safety review's third finding. `start` persists `preparing` before the drain
    and journals `gpu-prepare-attempted` before it calls the target, so a round whose
    drain failed -- or whose baseline collection was interrupted -- holds neither
    preparation marker and neither mutation marker. Nothing on the exercise node was
    asked to change: no telemetry was paused, no persistence was altered, no device
    was removed, and no original state was captured.

    The pre-repair gate reads only the ESTABLISHED PRE-CAPTURE REFUSAL marker, so it
    returns for this round as well, and recovery then selects a GPU reboot, remounts
    staging, runs the runtime restoration, and skips `gpu-restore` because neither
    preparation marker exists. A node that provably was not modified is therefore
    cycled, with no captured originals to put anything back from.

    Both entry points are covered because they share `_recover`: the participant's
    `recover` verb and the facilitator's deadline `expire`.
    """

    class RefusingDrain(TargetHelperExecutor):
        """The scheduler refuses the exercise's own drain request ONCE.

        Same shape as the existing failed-drain bench
        (test_device_session_rework.py:836-853), on the executor that runs the REAL
        maintenance helper, so `gpu-restore` here is the shipped function reading real
        record files rather than a stub that returns success.

        Only the FIRST drain request fails. A bench that refused every drain would make
        recovery stop at `_require_recoverable_target` -- 'is in service and could not be
        reserved' -- and that refusal is a different guard, so the test would pass on the
        unrepaired controller for a reason unrelated to the finding. Measured: with a
        permanently refusing drain the pre-repair controller does refuse, on that
        message. A transient scheduler failure is also the realistic case: the round
        stopped before it prepared anything, and recovery can reinstate the drain.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.drain_requests = 0

        def run_slurm(self, argv, timeout=None):
            if (Path(argv[0]).name == 'scontrol' and argv[1:2] == ['update']
                    and any('State=DRAIN' in str(a) for a in argv)):
                self.drain_requests += 1
                if self.drain_requests == 1:
                    self.slurm_calls.append(list(argv))
                    return session.Completed(1, '', 'scontrol: error: Invalid node\n')
            return super().run_slurm(argv, timeout=timeout)

    def setUp(self):
        super().setUp()
        self.executor = self.RefusingDrain(self.records, self.target_config)

    def never_prepared(self):
        """Start a GPU round whose drain fails, and return the recorded state.

        The premise is asserted by the caller rather than assumed: no preparation
        marker, no mutation marker, no record file on the target, and the node's
        telemetry and persistence exactly as they were found.
        """
        self.set_unit('active', 'loaded')
        with self.assertRaises(session.Refusal):
            self.make().start('gpu')
        return self.make().store.read()

    @staticmethod
    def attempt(call):
        """Run one recovery entry point, returning its refusal message or ''.

        The refusal is captured rather than required, so the behavioural assertions
        below are what fails on the unrepaired controller. `assertRaises` there would
        report 'Refusal not raised' -- true, but it names the missing refusal instead of
        the reboot and the runtime mutations that actually ran, which are the harm.
        """
        try:
            call()
        except session.Refusal as refusal:
            return str(refusal)
        return ''

    def test_the_round_provably_called_nothing_on_the_node(self):
        state = self.never_prepared()
        self.assertEqual('preparing', state.phase)
        self.assertNotIn('drain', state.prepared)
        self.assertNotIn('gpu-prepare-attempted', state.prepared)
        self.assertNotIn('gpu-prepare-recorded', state.prepared)
        self.assertNotIn('gpu-prepare-refused-before-capture', state.prepared)
        self.assertNotIn('gpu-remove-attempted', state.prepared)
        self.assertEqual([], self.records_present(),
                         'a round that never called gpu-prepare has an original-state '
                         'record')
        self.assertEqual('active', self.unit_now())
        self.assertEqual('Enabled', self.persistence_now())

    def test_participant_recovery_does_not_reboot_a_never_prepared_round(self):
        """The behavioural assertions first, before any message.

        A message assertion that fired first would make this test's failure on the
        pre-repair artifact a wording difference rather than a demonstration that the
        node was cycled.
        """
        self.never_prepared()
        message = self.attempt(lambda: self.make().recover())
        self.assertEqual([], self.executor.reboots(),
                         'a node that was never prepared was rebooted')
        actions = self.executor.actions()
        for mutation in ('remount-staging', 'restore-runtime', 'gpu-restore',
                         'gpu-prepare', 'gpu-remove', 'efa-rebind'):
            self.assertNotIn(mutation, actions,
                             f'{mutation} ran for a round that called nothing')
        self.assertEqual([], self.executor.resumes(),
                         'a node that was never prepared was returned to service')
        self.assertEqual([], self.records_present(),
                         'recovery wrote an original-state record of its own')
        self.assertEqual('active', self.unit_now(),
                         'recovery changed the telemetry unit of an unprepared round')
        self.assertEqual('Enabled', self.persistence_now())
        self.assertIn('did not modify', message)

    def test_the_node_keeps_its_isolation_and_the_round_is_reported(self):
        """The permitted outcome: isolate, report, invent no rollback.

        The exercise's own drain is reinstated -- that is what keeps other work off the
        node while a facilitator looks -- and nothing else is changed.
        """
        self.never_prepared()
        self.attempt(lambda: self.make().recover())
        state = self.make().store.read()
        self.assertEqual([], self.executor.reboots())
        self.assertEqual([], self.executor.resumes())
        self.assertEqual('recovery-failed', state.phase)
        self.assertIsNone(state.recovered_by)
        self.assertEqual('IDLE+DRAIN', self.executor.node_state)
        self.assertEqual(self.config_body['assignments']['table-1']['drain_reason'],
                         self.executor.node_reason)
        self.assertIn('nothing to put back', state.failure or '')

    def test_repeated_attempts_do_not_accumulate_into_a_reboot(self):
        self.never_prepared()
        for attempt in range(3):
            self.attempt(lambda: self.make().recover())
        self.assertEqual([], self.executor.reboots())
        self.assertEqual([], self.executor.resumes())
        self.assertEqual(0, self.make().store.read().reboot_requests)

    def test_the_deadline_path_does_not_reboot_it_either(self):
        """The facilitator's unattended entry point, not only the participant's.

        `expire` shares `_recover`, so the gate has to hold there too -- and this is the
        path a participant who walks away actually reaches.
        """
        self.never_prepared()
        # The round started at clock 1000 with a 1800 s deadline, so this is past it.
        self.attempt(lambda: self.make(now=3000.0).expire())
        self.assertEqual([], self.executor.reboots(),
                         'the deadline path rebooted a round that called nothing')
        actions = self.executor.actions()
        for mutation in ('remount-staging', 'restore-runtime', 'gpu-restore'):
            self.assertNotIn(mutation, actions,
                             f'{mutation} ran on the deadline path')
        self.assertEqual([], self.executor.resumes())
        self.assertEqual('active', self.unit_now())
        self.assertEqual(1, self.make().store.read().deadline_attempts,
                         'the deadline attempt was not counted')

    def test_an_interrupted_baseline_reaches_the_same_decision(self):
        """The other way to hold neither marker, with the drain in place.

        The baseline collection runs after the drain and before
        `gpu-prepare-attempted`. A coordinator that dies there leaves a round that
        drained the node and called nothing else on it. The drain is this exercise's
        own and is not a device mutation, so the decision must be the same.
        """
        self.set_unit('active', 'loaded')
        executor = TargetHelperExecutor(self.records, self.target_config)
        self.executor = executor

        class Died(BaseException):
            """Not a Refusal: nothing in start may catch this."""

        original = executor.run_maintenance

        def die_in_baseline(argv, timeout=None):
            if argv and argv[0] == 'collect':
                raise Died()
            return original(argv, timeout=timeout)

        executor.run_maintenance = die_in_baseline
        with self.assertRaises(Died):
            self.make().start('gpu')
        executor.run_maintenance = original
        state = self.make().store.read()
        self.assertIn('drain', state.prepared)
        self.assertNotIn('gpu-prepare-attempted', state.prepared)
        self.assertNotIn('gpu-remove-attempted', state.prepared)
        self.assertEqual([], self.records_present())

        message = self.attempt(lambda: self.make().recover())
        self.assertEqual([], executor.reboots(),
                         'an interrupted baseline was recovered by rebooting')
        for mutation in ('remount-staging', 'restore-runtime', 'gpu-restore'):
            self.assertNotIn(mutation, executor.actions())
        self.assertEqual([], executor.resumes())
        self.assertEqual('active', self.unit_now())
        self.assertIn('did not modify', message)

    # -- the rounds that must stay recoverable, and the ambiguous one that must not --
    def test_a_legacy_record_in_a_post_mutation_phase_is_refused_not_rebooted(self):
        """An ambiguous record is not upgraded into a mutated round.

        A record written before the markers existed can hold `investigating` with only
        the drain on its journal. An earlier revision read that phase as proof that the
        device operation had returned, journalled a marker saying so, and recovered the
        round. That inference was wrong at its root: `investigating` is written by
        `collect` and ALSO by both recovery entry points' refusal handlers, so the phase
        does not identify its author, and a never-prepared round that met a transient
        target-validation failure acquired the same phase.

        A display phase therefore authorizes nothing. This state establishes neither
        that a device call happened nor that none did, so it keeps its isolation and is
        reported for the facilitator. The cost is stated in the refusal: such a round
        captured no original state either, so rebooting it would leave the reboot's own
        effects in place with nothing to put them back.
        """
        executor = TargetHelperExecutor(self.records, self.target_config)
        self.executor = executor
        legacy = _rework.Base.baseline_for(self, session.State(
            phase='investigating', kind='efa', started_at=1.0,
            target_node='gpu-g7-1', target_instance_id='i-0123456789abcdef0',
            boot_id='dddddddd-0000-0000-0000-00000000000b', prepared=['drain']))
        self.make().store.write(legacy)
        executor.check_output['6'] = _rework.PASS_6
        message = self.attempt(lambda: self.make().recover())
        self.assertEqual([], executor.reboots(),
                         'an ambiguous legacy record was recovered by rebooting')
        for mutation in ('remount-staging', 'restore-runtime', 'gpu-restore',
                         'efa-rebind'):
            self.assertNotIn(mutation, executor.actions(),
                             f'{mutation} ran for a record that establishes no call')
        self.assertEqual([], executor.resumes())
        self.assertEqual('recovery-failed', self.make().store.read().phase)
        self.assertIn('does not establish', message)

    def test_the_ambiguous_legacy_refusal_holds_on_every_retry(self):
        """Repeated participant recovery and the deadline sweep reach it too.

        The journal does not change, so the decision must not either. This also covers
        the deadline entry point, which shares `_recover` and is the one an abandoned
        table actually reaches.
        """
        executor = TargetHelperExecutor(self.records, self.target_config)
        self.executor = executor
        legacy = _rework.Base.baseline_for(self, session.State(
            phase='investigating', kind='efa', started_at=1.0,
            target_node='gpu-g7-1', target_instance_id='i-0123456789abcdef0',
            boot_id='dddddddd-0000-0000-0000-00000000000b', prepared=['drain'],
            deadline_at=1800.0))
        self.make().store.write(legacy)
        executor.check_output['6'] = _rework.PASS_6
        for _ in range(2):
            self.attempt(lambda: self.make().recover())
        self.attempt(lambda: self.make(now=3000.0).expire())
        self.assertEqual([], executor.reboots(),
                         'a retry or the deadline path rebooted an ambiguous round')
        for mutation in ('remount-staging', 'restore-runtime', 'gpu-restore'):
            self.assertNotIn(mutation, executor.actions())
        self.assertEqual([], executor.resumes())
        state = self.make().store.read()
        self.assertEqual(0, state.reboot_requests)
        self.assertEqual(1, state.deadline_attempts)

    def test_a_recorded_device_call_with_an_unknown_outcome_still_recovers(self):
        """The valid path this refusal must not swallow.

        A journalled `efa-unbind-attempted` records the REQUEST, not its completion.
        Its outcome may be unknown, and that is precisely the round recovery exists
        for: the device is in an unknown state and only the rebind or the reboot can
        settle it. This is the same legacy-looking phase as the refused case above,
        distinguished by evidence rather than by a label.
        """
        executor = TargetHelperExecutor(self.records, self.target_config)
        self.executor = executor
        state = _rework.Base.baseline_for(self, session.State(
            phase='investigating', kind='efa', started_at=1.0,
            target_node='gpu-g7-1', target_instance_id='i-0123456789abcdef0',
            boot_id='dddddddd-0000-0000-0000-00000000000b',
            prepared=['drain', 'efa-unbind-attempted']))
        self.make().store.write(state)
        executor.check_output['6'] = _rework.PASS_6
        result = self.make().recover()
        self.assertEqual('runtime-ready', result.state.phase)
        self.assertIn('efa-rebind', executor.actions())
        self.assertIn('restore-runtime', executor.actions())
        self.assertEqual(1, len(executor.resumes()))

    def test_the_phase_reconciliation_does_not_excuse_a_preparing_round(self):
        """`preparing` is the phase every never-prepared round holds.

        Kept as its own case because it is the state a real interrupted start leaves,
        and it must reach the no-call refusal rather than the ambiguous-record one.
        """
        state = self.never_prepared()
        self.assertEqual('preparing', state.phase)
        message = self.attempt(lambda: self.make().recover())
        self.assertEqual([], self.executor.reboots())
        self.assertIn('never asked the node to pause telemetry', message)

    def test_a_transient_validation_refusal_does_not_license_the_next_attempt(self):
        """The continuation the review named: refuse, then refuse again.

        The first recovery of a never-prepared round meets a transient obstacle BEFORE
        the preparation gate -- here the target's instance-id request fails, which is
        what `_require_recoverable_target` reads first -- and the entry point's own
        handler writes `investigating`. On the rejected controller the next attempt read
        that phase as proof of a device call, journalled a marker saying so, bypassed
        the no-call refusal and rebooted a node whose originals were never captured.

        Once the obstacle clears, the second and third attempts must still refuse, and
        the deadline path with them. Nothing about the round changed while it waited.
        """
        self.never_prepared()
        original = self.executor.run_maintenance
        failing = {'now': True}

        def answering(argv, timeout=None):
            if failing['now'] and argv and argv[0] == 'instance-id':
                return session.Completed(255, '', 'Connection timed out\n')
            return original(argv, timeout=timeout)

        self.executor.run_maintenance = answering
        first = self.attempt(lambda: self.make().recover())
        self.assertIn('not reachable', first)
        after_first = self.make().store.read()
        self.assertEqual('investigating', after_first.phase,
                         'this continuation needs the phase recovery itself assigns')
        self.assertEqual([], self.executor.reboots())

        # The obstacle clears. The journal has not changed, so neither may the decision.
        failing['now'] = False
        for attempt in range(2):
            message = self.attempt(lambda: self.make().recover())
            self.assertIn('never asked the node to pause telemetry', message,
                          f'attempt {attempt + 2} did not reach the no-call refusal')
        self.attempt(lambda: self.make(now=3000.0).expire())

        self.assertEqual([], self.executor.reboots(),
                         'a round that called nothing was rebooted after a transient '
                         'validation failure')
        actions = self.executor.actions()
        for mutation in ('remount-staging', 'restore-runtime', 'gpu-restore',
                         'gpu-prepare', 'gpu-remove'):
            self.assertNotIn(mutation, actions, f'{mutation} ran on a never-prepared '
                                                f'round')
        self.assertEqual([], self.executor.resumes())
        self.assertEqual('active', self.unit_now())
        self.assertEqual([], self.records_present())
        final = self.make().store.read()
        self.assertEqual(0, final.reboot_requests)
        self.assertEqual(1, final.deadline_attempts)

    def test_a_prepared_round_still_reboots_and_restores(self):
        """The positive control: a preparation call was made, so recovery proceeds.

        Without this, the repair could be 'never reboot', which would strand every
        real round. Here `gpu-prepare` and `gpu-remove` both ran, so the round reboots,
        remounts, restores the runtime and puts the recorded originals back.
        """
        self.set_unit('active', 'loaded')
        executor = TargetHelperExecutor(self.records, self.target_config)
        self.executor = executor
        self.make().start('gpu')
        state = self.make().store.read()
        self.assertIn('gpu-prepare', state.prepared)
        self.assertIn('gpu-remove-attempted', state.prepared)
        result = self.make().recover()
        self.assertEqual(1, len(executor.reboots()))
        self.assertIn('gpu-restore', executor.actions())
        self.assertEqual('runtime-ready', result.state.phase)
        self.assertEqual('active', self.unit_now(),
                         'the recorded telemetry original was not put back')

    def test_an_attempted_preparation_with_an_unknown_outcome_still_recovers(self):
        """Unknown is not 'nothing happened'.

        A transport failure on `gpu-prepare` leaves `gpu-prepare-attempted` on the
        journal and establishes nothing about what the target did, so recovery must
        proceed -- and it is `gpu-restore`, reading the real record files, that decides
        whether there was anything to put back.
        """
        self.set_unit('active', 'loaded')
        executor = TargetHelperExecutor(self.records, self.target_config)
        self.executor = executor
        executor.maintenance_failures['gpu-prepare'] = (
            255, 'ssh: connect to host gpu-g7-1 port 22: Connection refused\n')
        with self.assertRaises(session.Refusal):
            self.make().start('gpu')
        state = self.make().store.read()
        self.assertIn('gpu-prepare-attempted', state.prepared)
        del executor.maintenance_failures['gpu-prepare']
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        self.assertEqual(1, len(executor.reboots()),
                         'a preparation whose outcome is unknown was treated as '
                         'nothing having happened')
        self.assertIn('gpu-restore', executor.actions())
        self.assertNotIn('did not modify', str(caught.exception))
        self.assertIn('No recorded persistence mode', str(caught.exception))


if __name__ == '__main__':                                    # pragma: no cover
    unittest.main()
