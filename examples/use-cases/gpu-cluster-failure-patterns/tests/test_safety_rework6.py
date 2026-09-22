# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Recovery decisions read per-operation facts, not display phases (rework 12).

Two findings from the safety review of `13d57063`, each exercised on the real
`start`/`recover`/`expire` verbs through the recorded fake executor:

  D2  every reboot-completion write requires a usable recorded baseline, a usable
      current observation, and no request still held by the scheduler. The retry
      branch of `_reconcile_outstanding_reboot` previously accepted any non-empty
      recorded value and wrote the completion before the pending check, so the
      refusal the first wait had just raised was undone by the next attempt.

  D5  warning qualification is deny-by-default and requires a complete observation
      of the pinned check: the check-2 advisory aggregate on both sides, its count
      matching the advisories beside it, and a condition whose producer's own success
      the sentence establishes.

The D1 cases (a phase never authorizes a mutation) live beside their siblings in
test_safety_rework5.py, where the never-prepared bench already is.

Every test asserts what was DONE -- which privileged calls were issued, what the
state record holds -- rather than the wording of a message. Each one fails on the
pre-repair tree; the discriminating run is
participant-revision/rework12-negative-control.py.

Run: python3 -m unittest discover -s <lab>/tests -p 'test_*.py'
"""
import importlib.util
import json
from pathlib import Path
import unittest

_rework_spec = importlib.util.spec_from_file_location(
    'aim344_rework_safety6',
    Path(__file__).resolve().parent / 'test_device_session_rework.py')
if _rework_spec is None or _rework_spec.loader is None:        # pragma: no cover
    raise unittest.SkipTest('test_device_session_rework.py not importable')
_rework = importlib.util.module_from_spec(_rework_spec)
_rework_spec.loader.exec_module(_rework)
session = _rework.session
Base = _rework.Base

_safety3_spec = importlib.util.spec_from_file_location(
    'aim344_safety3_for_rework6',
    Path(__file__).resolve().parent / 'test_safety_rework3.py')
if _safety3_spec is None or _safety3_spec.loader is None:      # pragma: no cover
    raise unittest.SkipTest('test_safety_rework3.py not importable')
_safety3 = importlib.util.module_from_spec(_safety3_spec)
_safety3_spec.loader.exec_module(_safety3)
maintenance = _safety3.maintenance
# The real-helper bench: real `gpu_prepare`/`gpu_restore` against real record files,
# with recording stub `systemctl`/`nvidia-smi` programs. Bound to a private name and
# deleted once the classes below have taken their methods: a TestCase subclass left at
# module level is COLLECTED by discovery, which ran that whole 54-test class a second
# time under this module's name (measured: 536 tests instead of 482).
_RealHelper = _safety3.S4TelemetryRestoration
TABLE1_R1 = _safety3.TABLE1_R1
TABLE2_R1 = _safety3.TABLE2_R1

STARTED_ON = 'aaaaaaaa-0000-0000-0000-000000000001'
CAME_BACK_ON = 'cccccccc-0000-0000-0000-000000000003'


class D2RebootCompletionOnRetry(Base):
    """The reconciliation branch a retry takes, with the first wait's conditions.

    `_await_boot` and its caller already required a boot-id-shaped baseline and an
    unheld scheduler request. `_reconcile_outstanding_reboot` is the branch every
    LATER attempt goes through, and it applied neither, so a round refused once could
    record a completed reboot on its next attempt without anything new being observed.
    """

    def outstanding_request(self, recorded_boot, current_boot, node_state):
        """A GPU round with a recorded, unreconciled reboot request.

        `recorded_boot` is what the round's state holds as the boot it started on;
        `current_boot` is what the target answers now; `node_state` is what the
        scheduler reports, which is where the REBOOT_ISSUED flag lives. The round is
        started for real first, so the journal, the operation token and the baseline
        are the ones a real round has.
        """
        self.make().start('gpu')
        state = self.make().store.read()
        state.boot_id = recorded_boot
        state.reboot_requests = 1
        state.reboot_requested_at = 900.0
        self.make().store.write(state)
        self.executor.boot_id = current_boot
        self.executor.node_state = node_state
        # Zero window: the repaired code refuses in the reconciliation before any
        # wait, and the pre-repair code recorded a completion there too, so no test
        # here needs `_await_boot` to poll.
        self.config_body['assignments']['table-1']['reboot_wait_seconds'] = 0
        self.config_body['assignments']['table-1']['reboot_poll_seconds'] = 0
        self.write_config()
        return state

    def attempt(self, handler=None):
        """One recovery, returning its refusal message or ''."""
        try:
            (handler or self.make()).recover()
        except session.Refusal as refusal:
            return str(refusal)
        return ''

    def assert_nothing_was_restored(self, reboots=0):
        actions = self.executor.actions()
        for mutation in ('remount-staging', 'restore-runtime', 'gpu-restore'):
            self.assertNotIn(mutation, actions,
                             f'{mutation} ran on an unestablished reboot completion')
        self.assertEqual([], self.executor.resumes(),
                         'a node was returned to service on an unestablished reboot')
        self.assertEqual(reboots, len(self.executor.reboots()),
                         'the reboot count is not what this case allows')

    # -- a baseline the first wait refuses must not qualify on the retry --
    def test_a_malformed_recorded_boot_is_not_a_comparable_baseline(self):
        """The carried case: `boot-before` is non-empty and is not a boot id.

        `_await_boot` refuses it, so the first attempt stops. On the retry the
        reconciliation compared that value with a real UUID, found them different, and
        recorded a completed reboot -- a completion manufactured from a baseline no
        observation established. The node had not restarted at all here: it answers
        the same boot throughout and the scheduler holds no flag.
        """
        self.outstanding_request('boot-before', STARTED_ON, 'IDLE+DRAIN')
        first = self.attempt()
        second = self.attempt()
        after = self.make().store.read()
        self.assertFalse(after.reboot_completed,
                         'a completion was recorded against a malformed baseline')
        self.assertIsNone(after.boot_id_after_reboot)
        self.assertEqual('boot-before', after.boot_id,
                         'the malformed baseline was overwritten by this attempt')
        self.assertNotEqual('runtime-ready', after.phase)
        self.assert_nothing_was_restored()
        self.assertEqual(1, after.reboot_requests,
                         'the outstanding request was erased or duplicated')
        for message in (first, second):
            self.assertIn('not a boot id', message)

    def test_the_malformed_baseline_refusal_holds_on_a_third_attempt(self):
        """Two consecutive failures, then one more: the record does not drift."""
        self.outstanding_request('boot-before', STARTED_ON, 'IDLE+DRAIN')
        for _ in range(3):
            self.attempt()
        after = self.make().store.read()
        self.assertFalse(after.reboot_completed)
        self.assertEqual(1, after.reboot_requests)
        self.assert_nothing_was_restored()

    def test_the_deadline_path_reaches_the_same_refusal(self):
        """`expire` shares `_recover`, and it is what an abandoned table meets."""
        self.outstanding_request('boot-before', STARTED_ON, 'IDLE+DRAIN')
        try:
            self.make(now=9999.0).expire()
        except session.Refusal as refusal:
            self.assertIn('not a boot id', str(refusal))
        else:
            self.fail('the deadline path accepted a malformed baseline')
        after = self.make().store.read()
        self.assertFalse(after.reboot_completed)
        self.assert_nothing_was_restored()
        self.assertEqual(1, after.deadline_attempts)

    # -- a changed boot with the request still held is not a completed reboot --
    def test_a_changed_boot_with_the_request_still_pending_is_not_a_completion(self):
        """The first wait's own condition, applied to the retry branch.

        The node really did restart, and the scheduler still holds REBOOT_ISSUED: it
        has not retired the request, so another restart is due and a round that
        recorded completion here could restore, qualify and resume under it. The first
        attempt refuses on exactly this condition after `_await_boot`; before this
        repair the very next attempt took the reconciliation branch and recorded the
        completion regardless.
        """
        self.outstanding_request(STARTED_ON, CAME_BACK_ON,
                                'IDLE+DRAIN+REBOOT_ISSUED')
        message = self.attempt()
        after = self.make().store.read()
        self.assertFalse(after.reboot_completed,
                         'a completion was recorded while the request was pending')
        self.assertNotIn('reboot', after.prepared)
        self.assertEqual(STARTED_ON, after.boot_id,
                         'the recorded boot was advanced on a pending request')
        self.assert_nothing_was_restored()
        self.assertIn('still holds', message)

    def test_the_pending_refusal_survives_a_second_and_third_attempt(self):
        """Retrying must not accumulate into a completion or a second reboot."""
        self.outstanding_request(STARTED_ON, CAME_BACK_ON,
                                'IDLE+DRAIN+REBOOT_ISSUED')
        for _ in range(3):
            self.attempt()
        after = self.make().store.read()
        self.assertFalse(after.reboot_completed)
        self.assertEqual(1, after.reboot_requests)
        self.assert_nothing_was_restored()

    def test_a_reboot_requested_flag_is_treated_the_same_way(self):
        """The other flag this Slurm build emits, not only the one measured first."""
        self.outstanding_request(STARTED_ON, CAME_BACK_ON,
                                'IDLE+DRAIN+REBOOT_REQUESTED')
        self.attempt()
        self.assertFalse(self.make().store.read().reboot_completed)
        self.assert_nothing_was_restored()

    # -- the positive controls: these paths must keep working --
    def test_a_changed_boot_with_no_pending_request_still_reconciles(self):
        """The valid reconciliation. Without this the repair could be 'never accept'.

        The node came back while the round was not watching and the scheduler has
        retired the request, so the completion is recorded, the staging mount is
        re-established and the round finishes.
        """
        self.outstanding_request(STARTED_ON, CAME_BACK_ON, 'IDLE+DRAIN')
        result = self.make().recover()
        self.assertEqual('runtime-ready', result.state.phase)
        self.assertTrue(result.state.reboot_completed)
        self.assertEqual(CAME_BACK_ON, result.state.boot_id_after_reboot)
        self.assertEqual(STARTED_ON, result.state.boot_id_before_reboot)
        self.assertIn('remount-staging', self.executor.actions())
        self.assertIn('gpu-restore', self.executor.actions())
        self.assertEqual([], self.executor.reboots(),
                         'the reconciled reboot was requested again')
        self.assertEqual(1, len(self.executor.resumes()))

    def test_the_pending_case_completes_once_the_scheduler_retires_the_request(self):
        """The refusal is retryable, not terminal: the same round finishes later."""
        self.outstanding_request(STARTED_ON, CAME_BACK_ON,
                                'IDLE+DRAIN+REBOOT_ISSUED')
        self.attempt()
        self.executor.node_state = 'IDLE+DRAIN'
        result = self.make().recover()
        self.assertEqual('runtime-ready', result.state.phase)
        self.assertTrue(result.state.reboot_completed)
        self.assertEqual(CAME_BACK_ON, result.state.boot_id_after_reboot)
        self.assertEqual([], self.executor.reboots())
        self.assertEqual(1, len(self.executor.resumes()))

    def test_an_ordinary_gpu_round_is_unaffected(self):
        """No recorded request at all: the reconciliation must not enter."""
        self.make().start('gpu')
        result = self.make().recover()
        self.assertEqual('runtime-ready', result.state.phase)
        self.assertEqual(1, len(self.executor.reboots()))
        self.assertTrue(result.state.reboot_completed)
        self.assertEqual(1, len(self.executor.resumes()))


class D2RetryAfterARecordedCompletion(Base):
    """The other retry shape: a completion is already on the record.

    `_reconcile_outstanding_reboot` returned True immediately for
    `state.reboot_completed`, before reading the current boot or the scheduler's flags.
    That is correct about one thing -- the round never requests a second reboot -- and
    wrong about another: the record says the node came back once, not that it is still
    on that boot or that no request is now held. A restoration-only retry is reached
    whenever the first attempt's `gpu-restore` or a required check failed, and between
    the two attempts the node can restart again under a request this exercise did not
    make, which removes the staging bind mount the restoration needs.

    Each test drives a real `start('gpu')` and a real first `recover` that reaches the
    completion, then makes the second attempt meet the condition under test.
    """

    def completed_reboot_then_a_failure(self):
        """A round whose reboot completed and whose restoration then failed.

        The failure is the shipped `gpu-restore` returning non-zero, which is a
        required-step failure: the node keeps its drain and the round stays retryable
        with `reboot_completed=True` on the record. That is the real route to a
        restoration-only retry.
        """
        self.make().start('gpu')
        self.executor.maintenance_failures['gpu-restore'] = (
            1, 'The recorded GPU state was not fully restored.\n')
        try:
            self.make().recover()
        except session.Refusal:
            pass
        else:                                                  # pragma: no cover
            self.fail('the first attempt did not fail its restoration')
        state = self.make().store.read()
        self.assertTrue(state.reboot_completed,
                        'this bench needs a recorded completion')
        self.assertEqual(1, len(self.executor.reboots()))
        # The restoration is allowed to succeed from now on, so anything the second
        # attempt refuses is refused for the reason under test.
        self.executor.maintenance_failures.pop('gpu-restore', None)
        self.executor.node_state = 'IDLE+DRAIN'
        return state

    def second_attempt(self):
        try:
            self.make().recover()
        except session.Refusal as refusal:
            return str(refusal)
        return ''

    def assert_no_second_cycle_and_no_resume(self):
        self.assertEqual(1, len(self.executor.reboots()),
                         'a second reboot was requested for a completed one')
        self.assertEqual([], self.executor.resumes(),
                         'a node was resumed while its current state was unestablished')

    def test_a_reboot_pending_again_stops_the_restoration_only_retry(self):
        """A request held now is not answered by a completion recorded earlier."""
        self.completed_reboot_then_a_failure()
        before = len(self.executor.maintenance_calls)
        self.executor.node_state = 'IDLE+DRAIN+REBOOT_ISSUED'
        message = self.second_attempt()
        during = [call[0] for call in self.executor.maintenance_calls[before:]]
        for mutation in ('remount-staging', 'restore-runtime', 'gpu-restore'):
            self.assertNotIn(mutation, during,
                             f'{mutation} ran with a reboot pending again')
        self.assert_no_second_cycle_and_no_resume()
        self.assertNotEqual('runtime-ready', self.make().store.read().phase)
        self.assertIn('pending', message)

    def test_a_changed_current_boot_stops_the_restoration_only_retry(self):
        """The node restarted again, so the round's observations are stale."""
        state = self.completed_reboot_then_a_failure()
        before = len(self.executor.maintenance_calls)
        self.executor.boot_id = 'eeeeeeee-0000-0000-0000-00000000000e'
        self.assertNotEqual(state.boot_id, self.executor.boot_id)
        message = self.second_attempt()
        during = [call[0] for call in self.executor.maintenance_calls[before:]]
        for mutation in ('remount-staging', 'restore-runtime', 'gpu-restore'):
            self.assertNotIn(mutation, during,
                             f'{mutation} ran against a boot the round never observed')
        self.assert_no_second_cycle_and_no_resume()
        after = self.make().store.read()
        self.assertEqual(state.boot_id, after.boot_id,
                         'the recorded boot was silently advanced')
        self.assertIn('restarted again', message)

    def test_an_unreadable_current_boot_stops_it_too(self):
        """An unread boot is not evidence that the node is where the round left it."""
        self.completed_reboot_then_a_failure()
        before = len(self.executor.maintenance_calls)
        original = self.executor.run_maintenance

        def answering(argv, timeout=None):
            if argv and argv[0] == 'boot-id':
                return session.Completed(255, '', '')
            return original(argv, timeout=timeout)

        self.executor.run_maintenance = answering
        message = self.second_attempt()
        during = [call[0] for call in self.executor.maintenance_calls[before:]]
        for mutation in ('remount-staging', 'restore-runtime', 'gpu-restore'):
            self.assertNotIn(mutation, during)
        self.assert_no_second_cycle_and_no_resume()
        self.assertIn('current boot', message)

    def test_two_consecutive_refusals_do_not_drift_the_record(self):
        """The same condition twice: no completion is unwound, no reboot is added."""
        state = self.completed_reboot_then_a_failure()
        self.executor.node_state = 'IDLE+DRAIN+REBOOT_REQUESTED'
        self.second_attempt()
        self.second_attempt()
        after = self.make().store.read()
        self.assertTrue(after.reboot_completed,
                        'the recorded completion was unwound by a refusal')
        self.assertEqual(state.reboot_requests, after.reboot_requests)
        self.assert_no_second_cycle_and_no_resume()

    def test_the_deadline_path_applies_the_same_conditions(self):
        """`expire` shares `_recover`, so the unattended path must refuse too."""
        self.completed_reboot_then_a_failure()
        self.executor.node_state = 'IDLE+DRAIN+REBOOT_ISSUED'
        try:
            self.make(now=9999.0).expire()
        except session.Refusal as refusal:
            self.assertIn('pending', str(refusal))
        else:                                                  # pragma: no cover
            self.fail('the deadline path restored a node with a reboot pending')
        self.assert_no_second_cycle_and_no_resume()

    # -- the positive controls --
    def test_the_ordinary_restoration_only_retry_still_finishes(self):
        """Nothing pending, boot unchanged: the retry restores and resumes.

        This is the case the existing repair exists for, and it must not become a
        refusal. Without it, 're-check everything' could strand every round whose
        first restoration failed.
        """
        self.completed_reboot_then_a_failure()
        before = len(self.executor.maintenance_calls)
        result = self.make().recover()
        during = [call[0] for call in self.executor.maintenance_calls[before:]]
        self.assertEqual('runtime-ready', result.state.phase)
        self.assertIn('gpu-restore', during)
        self.assertIn('restore-runtime', during)
        self.assertEqual(1, len(self.executor.reboots()),
                         'the retry rebooted a node that had already come back')
        self.assertEqual(1, len(self.executor.resumes()))

    def test_the_refusal_clears_when_the_scheduler_retires_the_request(self):
        """Retryable, not terminal: the same round finishes once the flag is gone."""
        self.completed_reboot_then_a_failure()
        self.executor.node_state = 'IDLE+DRAIN+REBOOT_ISSUED'
        self.second_attempt()
        self.executor.node_state = 'IDLE+DRAIN'
        result = self.make().recover()
        self.assertEqual('runtime-ready', result.state.phase)
        self.assertEqual(1, len(self.executor.reboots()))
        self.assertEqual(1, len(self.executor.resumes()))


class RealHelperBench(unittest.TestCase):
    """The rework-3 real-helper bench, without inheriting its test methods.

    The fixture is taken by reference rather than by subclassing, because subclassing
    `S4TelemetryRestoration` would make discovery run its whole suite a second time
    under this module's name. The bench itself is unchanged: real
    `maintenance.gpu_prepare`/`gpu_restore`, real record files under a temporary
    RECORD_DIR, and the same recording `systemctl`/`nvidia-smi` stubs.
    """

    setUp = _RealHelper.setUp
    tearDown = _RealHelper.tearDown
    capture = _RealHelper.capture
    unit_state = _RealHelper.unit_state
    persistence = _RealHelper.persistence
    set_load_state = _RealHelper.set_load_state
    script_is_active = _RealHelper.script_is_active
    systemctl_calls = _RealHelper.systemctl_calls

    def read_record(self, name):
        return json.loads((self.records / name).read_text())

    def strip_operation(self, name):
        """Remove a record's `operation` field, keeping everything else.

        The shape a truncated write, a partial restore from a backup or a hand-edit
        leaves: a readable record that names no round.
        """
        record = self.read_record(name)
        record.pop('operation', None)
        (self.records / name).write_text(json.dumps(record, sort_keys=True) + '\n')

    def markers(self):
        return sorted(path.name for path in self.records.iterdir()
                      if path.name.startswith(maintenance.CAPTURE_MARKER_PREFIX))


class D3InheritedOriginalsWithoutAMarker(RealHelperBench):
    """A capture from before the marker existed is still a capture.

    `_has_captured_before` consulted only the marker file, so an intact
    operation-tagged pair written by the previous version answered 'this operation has
    never captured'. A repeat then treated a later loss of one record's token as a
    file it was free to record over, and rewrote BOTH originals from the node it had
    already paused and disabled. `gpu_restore` accepted the newly tagged values and
    reported success for restoring `inactive`/`Disabled`.

    A record naming this operation is itself evidence that this operation captured;
    nothing else could have written that token. Both kinds of evidence now answer the
    question, so the protection does not depend on which version wrote the pair.
    """

    def intact_pair_without_a_marker(self):
        """A prepared round whose capture marker does not exist.

        Built by running the real first `gpu_prepare` and then deleting the marker,
        which is exactly the on-disk state the previous version left: two
        operation-tagged records, no marker, telemetry paused, persistence disabled.
        """
        self.set_load_state('loaded')
        code, output = self.capture(maintenance.gpu_prepare, self.config,
                                    {'operation': TABLE1_R1})
        self.assertEqual(0, code, output)
        for name in self.markers():
            (self.records / name).unlink()
        self.assertEqual([], self.markers(), 'this bench needs a markerless pair')
        self.assertEqual('inactive', self.unit_state())
        self.assertEqual('Disabled', self.persistence())
        return {name: self.read_record(name)
                for name in ('native-dcgm-state.txt',
                             'selected-persistence-mode.txt')}

    def test_a_surviving_original_is_capture_evidence_without_a_marker(self):
        """The question itself, on the markerless pair."""
        self.intact_pair_without_a_marker()
        self.assertTrue(
            maintenance._has_captured_before(TABLE1_R1, (('usable', 'active'),
                                                         ('absent', None))),
            'a surviving record for this operation was not read as a capture')
        self.assertFalse(
            maintenance._has_captured_before(TABLE1_R1, (('absent', None),
                                                         ('absent', None))),
            'no evidence at all was read as a capture')

    def test_a_markerless_pair_that_loses_one_token_is_not_overwritten(self):
        """The finding: one record loses its token, the other survives.

        The repeat must refuse with the drain in place. Recording again would write the
        state this round produced as the state it found, permanently.
        """
        before = self.intact_pair_without_a_marker()
        self.strip_operation('selected-persistence-mode.txt')
        outcome, output = self.capture(maintenance.gpu_prepare, self.config,
                                      {'operation': TABLE1_R1})
        self.assertIsInstance(outcome, maintenance.Refusal, output)
        surviving = self.read_record('native-dcgm-state.txt')
        self.assertEqual(before['native-dcgm-state.txt'], surviving,
                         'the surviving original was overwritten')
        self.assertEqual('active', surviving['value'],
                         'the original state this round found was lost')

    def test_the_other_half_is_protected_the_same_way(self):
        """The symmetric case: the telemetry record is the one that loses its token."""
        before = self.intact_pair_without_a_marker()
        self.strip_operation('native-dcgm-state.txt')
        outcome, output = self.capture(maintenance.gpu_prepare, self.config,
                                      {'operation': TABLE1_R1})
        self.assertIsInstance(outcome, maintenance.Refusal, output)
        surviving = self.read_record('selected-persistence-mode.txt')
        self.assertEqual(before['selected-persistence-mode.txt'], surviving)
        self.assertEqual('Enabled', surviving['value'])

    def test_the_refusal_holds_on_a_second_and_third_repeat(self):
        """Nothing accumulates: the records do not drift across attempts."""
        before = self.intact_pair_without_a_marker()
        self.strip_operation('selected-persistence-mode.txt')
        for _ in range(3):
            outcome, _ = self.capture(maintenance.gpu_prepare, self.config,
                                      {'operation': TABLE1_R1})
            self.assertIsInstance(outcome, maintenance.Refusal)
        self.assertEqual(before['native-dcgm-state.txt'],
                         self.read_record('native-dcgm-state.txt'))

    def test_restoration_still_puts_back_what_the_markerless_round_found(self):
        """The point of preserving it: the original is still restorable.

        A refusal that preserved the record but left it unusable would be no better
        than the overwrite. The surviving half plus a repaired partner restores to
        `active`/`Enabled`, which is what the round actually found.
        """
        before = self.intact_pair_without_a_marker()
        self.strip_operation('selected-persistence-mode.txt')
        self.capture(maintenance.gpu_prepare, self.config, {'operation': TABLE1_R1})
        # Fixture reset only, not a supported recovery remedy: put the known
        # original back to test restoration independently of record loss.
        (self.records / 'selected-persistence-mode.txt').write_text(
            json.dumps(before['selected-persistence-mode.txt'], sort_keys=True) + '\n')
        code, output = self.capture(maintenance.gpu_restore, self.config,
                                    {'operation': TABLE1_R1})
        self.assertEqual(0, code, output)
        self.assertEqual('active', self.unit_state())
        self.assertEqual('Enabled', self.persistence())

    # -- the positive controls --
    def test_a_markerless_intact_pair_is_still_a_usable_repeat(self):
        """A repeat that finds both records usable proceeds and keeps them.

        Without this, 'refuse whenever the marker is missing' would strand every round
        started by the previous version.
        """
        before = self.intact_pair_without_a_marker()
        code, output = self.capture(maintenance.gpu_prepare, self.config,
                                   {'operation': TABLE1_R1})
        self.assertEqual(0, code, output)
        self.assertIn('kept from this operation', output)
        self.assertEqual(before, {name: self.read_record(name) for name in before})

    def test_a_new_operation_cannot_overwrite_an_untagged_partner(self):
        """PI contract: a new token does not establish whose untagged record this is.

        The old expectation required overwrite of this ambiguous pair. Refuse
        instead; the adjacent tagged-foreign pair remains a positive control.
        """
        self.intact_pair_without_a_marker()
        self.strip_operation('selected-persistence-mode.txt')
        before = {p.name: p.read_text() for p in self.records.iterdir()}
        code, output = self.capture(maintenance.gpu_prepare, self.config,
                                   {'operation': TABLE2_R1, 'fresh_capture': True})
        self.assertIsInstance(code, maintenance.Refusal, output)
        self.assertEqual(before, {p.name: p.read_text() for p in self.records.iterdir()})

    def test_a_new_operation_can_capture_over_a_tagged_foreign_pair(self):
        """A pair that identifies its prior operation is not untagged ambiguity."""
        self.intact_pair_without_a_marker()
        code, output = self.capture(maintenance.gpu_prepare, self.config,
                                   {'operation': TABLE2_R1, 'fresh_capture': True})
        self.assertEqual(0, code, output)
        self.assertEqual(TABLE2_R1,
                         self.read_record('native-dcgm-state.txt')['operation'],
                         'a new operation could not capture at all')


class D4RepeatObservesThePreparedState(RealHelperBench):
    """A preparation that reports success has observed the prepared state.

    The settle, restorable and readable checks lived inside the first-capture branch,
    so a repeat with two valid saved originals skipped all three. A unit left
    `activating` by a slow restart, or `failed`, or unreadable, then matched neither
    the `active` stop-and-confirm branch nor the persistence change, and the helper
    returned 0 while telemetry still held the GPU. Those checks are now applied to the
    observation itself, on a first call and on a repeat alike.
    """

    def prepared_round(self):
        """A completed first preparation for TABLE1_R1, with its originals recorded."""
        self.set_load_state('loaded')
        code, output = self.capture(maintenance.gpu_prepare, self.config,
                                    {'operation': TABLE1_R1})
        self.assertEqual(0, code, output)
        return {name: self.read_record(name)
                for name in ('native-dcgm-state.txt',
                             'selected-persistence-mode.txt')}

    def test_a_repeat_whose_unit_never_settles_does_not_report_success(self):
        """`activating` for the whole bounded wait is not a prepared node."""
        before = self.prepared_round()
        (self.stub_state / 'unit-state.txt').write_text('activating\n')
        outcome, output = self.capture(maintenance.gpu_prepare, self.config,
                                      {'operation': TABLE1_R1})
        self.assertIsInstance(outcome, maintenance.Refusal, output)
        self.assertEqual('activating', self.unit_state(),
                         'a refused repeat changed the unit')
        self.assertEqual(before, {name: self.read_record(name) for name in before},
                         'a refused repeat rewrote the originals')

    def test_a_repeat_whose_unit_is_failed_does_not_report_success(self):
        """`failed` is settled, loaded, and not evidence the GPU was released."""
        before = self.prepared_round()
        (self.stub_state / 'unit-state.txt').write_text('failed\n')
        outcome, output = self.capture(maintenance.gpu_prepare, self.config,
                                      {'operation': TABLE1_R1})
        self.assertIsInstance(outcome, maintenance.Refusal, output)
        self.assertIn('released the GPU', str(outcome))
        self.assertEqual(before, {name: self.read_record(name) for name in before})

    def test_a_repeat_whose_unit_is_unreadable_does_not_report_success(self):
        """An empty answer is a failed invocation, not a stopped service.

        This one already refused before this repair, on the neither-property check
        `gpu_prepare` applies to every call: the stub exits 3 printing nothing, so
        `_read_unit` returns ('', ''). It is kept as a regression guard for that path,
        not as a discriminating case, and the negative-control log shows it passing on
        both trees.
        """
        before = self.prepared_round()
        (self.stub_state / 'unit-state.txt').write_text('EMPTY\n')
        outcome, output = self.capture(maintenance.gpu_prepare, self.config,
                                      {'operation': TABLE1_R1})
        self.assertIsInstance(outcome, maintenance.Refusal, output)
        self.assertEqual(before, {name: self.read_record(name) for name in before})

    def test_two_consecutive_refused_repeats_leave_the_originals_intact(self):
        before = self.prepared_round()
        (self.stub_state / 'unit-state.txt').write_text('failed\n')
        for _ in range(2):
            outcome, _ = self.capture(maintenance.gpu_prepare, self.config,
                                      {'operation': TABLE1_R1})
            self.assertIsInstance(outcome, maintenance.Refusal)
        self.assertEqual(before, {name: self.read_record(name) for name in before})

    # -- the positive controls --
    def test_an_ordinary_repeat_of_a_settled_node_still_succeeds(self):
        """Telemetry already inactive, persistence already disabled: nothing to do."""
        before = self.prepared_round()
        code, output = self.capture(maintenance.gpu_prepare, self.config,
                                   {'operation': TABLE1_R1})
        self.assertEqual(0, code, output)
        self.assertEqual('inactive', self.unit_state())
        self.assertEqual(before, {name: self.read_record(name) for name in before})

    def test_a_repeat_still_re_applies_a_pause_that_was_undone(self):
        """The drifted-repeat control: the service came back, so stop it again."""
        before = self.prepared_round()
        (self.stub_state / 'unit-state.txt').write_text('active\n')
        code, output = self.capture(maintenance.gpu_prepare, self.config,
                                   {'operation': TABLE1_R1})
        self.assertEqual(0, code, output)
        self.assertEqual('inactive', self.unit_state(),
                         'the repeat did not re-apply the pause')
        self.assertEqual(before, {name: self.read_record(name) for name in before},
                         'the repeat rewrote the originals it should keep')

    def test_a_first_call_on_a_failed_unit_still_refuses_before_recording(self):
        """The first-capture rule is unchanged: no round starts on `failed`."""
        self.set_load_state('loaded')
        (self.stub_state / 'unit-state.txt').write_text('failed\n')
        outcome, output = self.capture(maintenance.gpu_prepare, self.config,
                                      {'operation': TABLE1_R1})
        self.assertIsInstance(outcome, maintenance.Refusal, output)
        self.assertIn('no defined restoration', str(outcome))
        self.assertFalse((self.records / 'native-dcgm-state.txt').exists())


class D5QualificationRequiresACompleteTrustedObservation(Base):
    """Deny by default: an allowed condition must also be a complete observation.

    Two residuals, both reached through the real `start`/`recover` verbs.

    The aggregate was required to AGREE when present and not required to be present, so
    a detail-only capture qualified on exactly the evidence the aggregate exists to
    complete. The pinned check prints it whenever `warn_efa` counted anything
    (checks/2-efa-enumeration.sh@a4ba07eb, the `advisory_count -gt 0` branch), so a
    check-2 WARN implies both lines and a capture with one of them is partial.

    The module advisories were on the allowlist because the sentence names a live
    condition, but the check does not establish that its own reading succeeded: step 5
    is `if ! lsmod | grep -qw "${mod}"`, whose warning branch is taken both when lsmod
    succeeds and the module is absent, and when lsmod fails or is absent itself. The
    same applies to `Memory lock limit 0 KB`, because the pin's own fallback is
    `memlock=$(ulimit -l 2>/dev/null || echo "0")`.

    Scope of this verification: the capture strings below are fixtures. Their format is
    the deployed `check_warn` emitter's, and the two sentences are copied from
    `checks/2-efa-enumeration.sh@a4ba07eb` (:190-191 for the advisory, :212-213 for the
    aggregate). No recorded run of the pinned check contains the advisory sentence --
    the exercise pair's limit was above the threshold every time, so it printed
    `[DEBUG] Memory lock limit OK` -- which means neither the accepting nor the refusing
    path here has been exercised against real advisory output. These tests establish
    what the parser does with the sentences the source can print, not that a live
    capture reaches it.
    """

    MEMLOCK = ('[WARN] 2-efa-enumeration: Memory lock limit 8192 KB is below 16 GiB '
               '-- EFA performance may be degraded\n')
    AGGREGATE_1 = ('[WARN] 2-efa-enumeration: EFA devices enumerated with 1 '
                   'advisories; inspect raw output\n')
    AGGREGATE_2 = ('[WARN] 2-efa-enumeration: EFA devices enumerated with 2 '
                   'advisories; inspect raw output\n')
    MODULE = '[WARN] 2-efa-enumeration: EFA kernel module efa not loaded\n'
    ZERO_MEMLOCK = ('[WARN] 2-efa-enumeration: Memory lock limit 0 KB is below 16 GiB '
                    '-- EFA performance may be degraded\n')

    def qualify(self, baseline_text, recovered_text, check='2', kind='efa'):
        """One real round: baseline warns, recovery warns, the controller decides."""
        self.executor.check_output[check] = baseline_text
        self.make().start(kind)
        self.executor.check_output[check] = recovered_text
        try:
            return self.make().recover(), None
        except session.Refusal as refusal:
            return None, refusal

    def assert_kept_drained(self, result, refusal):
        self.assertIsNone(result, 'the node was returned to service')
        assert refusal is not None
        self.assertEqual([], self.executor.resumes())
        # Final finding 5 routes unsupported WARN to isolated replacement.
        self.assertEqual('replacement-required', self.make().store.read().phase)
        return str(refusal)

    # -- the aggregate must be present, not merely consistent when it survives --
    def test_a_detail_only_capture_does_not_qualify(self):
        """Both sides identical, both incomplete: the count is missing from each."""
        result, refusal = self.qualify(self.MEMLOCK, self.MEMLOCK)
        message = self.assert_kept_drained(result, refusal)
        self.assertIn('without the count', message)

    def test_a_detail_only_recovery_against_a_complete_baseline_does_not_qualify(self):
        """The asymmetric case: the baseline is complete and this capture is not.

        This one already refused before this repair, and for a different reason: the two
        captures do not hold the same condition set, so the aggregate counts as a
        condition that vanished. Kept as a regression guard for that path; the
        negative-control log shows it passing on both trees. The symmetric case above is
        the discriminating one.
        """
        result, refusal = self.qualify(self.MEMLOCK + self.AGGREGATE_1, self.MEMLOCK)
        self.assert_kept_drained(result, refusal)

    def test_a_complete_recovery_against_a_detail_only_baseline_does_not_qualify(self):
        """And the other way: an incomplete baseline cannot qualify anything.

        Also already refused before the repair, as a condition that appeared. Regression
        guard, not a discriminating case.
        """
        result, refusal = self.qualify(self.MEMLOCK, self.MEMLOCK + self.AGGREGATE_1)
        self.assert_kept_drained(result, refusal)

    def test_a_count_that_disagrees_with_its_advisories_still_does_not_qualify(self):
        """The previously repaired case, preserved: present but inconsistent."""
        capture = self.MEMLOCK + self.AGGREGATE_2
        result, refusal = self.qualify(capture, capture)
        message = self.assert_kept_drained(result, refusal)
        self.assertIn('does not match', message)

    # -- a condition whose producer's success is not established --
    def test_a_module_advisory_does_not_qualify_even_when_complete(self):
        """Complete, consistent, identical, and still not two observations."""
        capture = self.MODULE + self.AGGREGATE_1
        result, refusal = self.qualify(capture, capture)
        message = self.assert_kept_drained(result, refusal)
        self.assertIn('cannot confirm succeeded', message)

    def test_a_gdrdrv_advisory_does_not_qualify_either(self):
        """The pin's own gdrdrv sentence, which is a different one from step 5's."""
        capture = ('[WARN] 2-efa-enumeration: gdrdrv module not loaded -- GPUDirect '
                   'RDMA may fall back to slower paths\n') + self.AGGREGATE_1
        result, refusal = self.qualify(capture, capture)
        self.assert_kept_drained(result, refusal)

    def test_a_zero_memory_lock_limit_does_not_qualify(self):
        """`0 KB` is what the pin prints when `ulimit -l` itself failed."""
        capture = self.ZERO_MEMLOCK + self.AGGREGATE_1
        result, refusal = self.qualify(capture, capture)
        message = self.assert_kept_drained(result, refusal)
        self.assertIn('will not qualify automatically', message)

    def test_a_module_advisory_beside_a_qualifiable_one_refuses_the_whole_capture(self):
        """One unqualifiable condition is enough: the capture is judged as a whole."""
        capture = self.MEMLOCK + self.MODULE + self.AGGREGATE_2
        result, refusal = self.qualify(capture, capture)
        self.assert_kept_drained(result, refusal)

    # -- the positive paths, which must keep working --
    def test_a_complete_unchanged_memory_lock_fixture_is_unsupported(self):
        """Final finding 5: source sentences do not establish a supported capture."""
        capture = self.MEMLOCK + self.AGGREGATE_1
        result, refusal = self.qualify(capture, capture)
        message = self.assert_kept_drained(result, refusal)
        self.assertIn('auto-qualification is disabled', message)

    def test_a_changed_limit_in_a_complete_capture_still_keeps_the_node_drained(self):
        """Equality is still required of an allowed condition."""
        before = self.MEMLOCK + self.AGGREGATE_1
        after = before.replace('8192', '1024')
        result, refusal = self.qualify(before, after)
        message = self.assert_kept_drained(result, refusal)
        self.assertIn('1024 KB', message)

    def test_a_clean_pass_after_recovery_still_resumes(self):
        """Nothing to qualify: a healthy round is unaffected by the allowlist."""
        self.executor.check_output['2'] = self.MODULE + self.AGGREGATE_1
        self.make().start('efa')
        self.executor.check_output['2'] = _rework.PASS_2
        result = self.make().recover()
        self.assertEqual('runtime-ready', result.state.phase)
        self.assertEqual(1, len(self.executor.resumes()))

    def test_an_ordinary_gpu_round_with_clean_checks_still_resumes(self):
        """The other fault kind, end to end, with no warning anywhere."""
        self.make().start('gpu')
        result = self.make().recover()
        self.assertEqual('runtime-ready', result.state.phase)
        self.assertEqual(1, len(self.executor.resumes()))


# The borrowed bench class must not stay in this module's namespace: unittest's loader
# collects every TestCase subclass it finds by name, whatever the name looks like, so
# leaving it here would run the rework-3 S4 suite twice. Its methods are already bound
# into the classes above, which is all this module needs.
del _RealHelper


if __name__ == '__main__':                                     # pragma: no cover
    unittest.main()
