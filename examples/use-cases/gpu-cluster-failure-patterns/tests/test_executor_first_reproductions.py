# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Executor-first reproductions of the six findings in t_3b88abe0's REJECT verdict
on t_6aba1d69 (rework 12, commit f28f0daf94a4de825c72ed41f801bdd827e41b9f).

Each class below is one card-body finding, exercised through the real
controller/helper functions this repo already tests with (no new framework, no
scanner). Every negative case has a positive control in the same class that
exercises the same caller shape on the accepted condition, so an import/API
mismatch cannot masquerade as a discriminating failure. Baseline commit under
test is the SAME f28f0daf the parent/companion cards used; this file's own
existence and every test in it are new, added on top of that tree for this
review round, not a change to the site under test.

Full narrative, citations and outcomes are recorded in
participant-revision/executor-first/REPRODUCTIONS.md as each unit lands.
Findings whose source-derived concern is not executable within this bench are
recorded there as NOT REPRODUCED with the exact missing observation -- none is
fabricated here.
"""
import importlib.util
import json
import os
from pathlib import Path
import unittest

_rework_spec = importlib.util.spec_from_file_location(
    'aim344_executor_first_rework',
    Path(__file__).resolve().parent / 'test_device_session_rework.py')
if _rework_spec is None or _rework_spec.loader is None:      # pragma: no cover
    raise unittest.SkipTest('test_device_session_rework.py not importable')
_rework = importlib.util.module_from_spec(_rework_spec)
_rework_spec.loader.exec_module(_rework)
session = _rework.session
Base = _rework.Base

_safety3_spec = importlib.util.spec_from_file_location(
    'aim344_safety3_for_executor_first',
    Path(__file__).resolve().parent / 'test_safety_rework3.py')
if _safety3_spec is None or _safety3_spec.loader is None:    # pragma: no cover
    raise unittest.SkipTest('test_safety_rework3.py not importable')
_safety3 = importlib.util.module_from_spec(_safety3_spec)
_safety3_spec.loader.exec_module(_safety3)
maintenance = _safety3.maintenance
_RealHelper = _safety3.S4TelemetryRestoration
TABLE1_R1 = _safety3.TABLE1_R1
TABLE2_R1 = _safety3.TABLE2_R1

_safety6_spec = importlib.util.spec_from_file_location(
    'aim344_safety6_for_executor_first',
    Path(__file__).resolve().parent / 'test_safety_rework6.py')
if _safety6_spec is None or _safety6_spec.loader is None:    # pragma: no cover
    raise unittest.SkipTest('test_safety_rework6.py not importable')
_safety6 = importlib.util.module_from_spec(_safety6_spec)
_safety6_spec.loader.exec_module(_safety6)
D2Base = _safety6.D2RebootCompletionOnRetry
STARTED_ON = _safety6.STARTED_ON
CAME_BACK_ON = _safety6.CAME_BACK_ON


class RealHelperBench(unittest.TestCase):
    """Same real-helper fixture the D3/D4 classes in test_safety_rework6.py use.

    Taken by reference, not by subclassing, for the reason test_safety_rework6.py
    documents beside its own copy: a TestCase subclass sitting in this module's
    namespace is collected a second time by `unittest discover`.
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
        record = self.read_record(name)
        record.pop('operation', None)
        (self.records / name).write_text(json.dumps(record, sort_keys=True) + '\n')

    def markers(self):
        return sorted(path.name for path in self.records.iterdir()
                      if path.name.startswith(maintenance.CAPTURE_MARKER_PREFIX))


# ---------------------------------------------------------------------------
# Finding 1 (card body): real pinned Check6 statistics-skipped PASS accepted
# after WARN. Referee's D5 full-caller finding, 2026-09-18 06:46 UTC comment on
# t_3b88abe0; pinned source checks/6-efa-loopback.sh@a4ba07eb, lines 182,185
# ("rdma statistic show failed -- EFA statistics skipped",
# "rdma tool not found -- EFA statistics skipped"), which leave stats_warning=0
# and can still reach check_pass at :197. That PASS text carries no counter
# detail and no aggregate at all -- it is what the pinned check prints when the
# statistics collection step never ran, distinct from a genuinely clean read.
# ---------------------------------------------------------------------------
class Finding1StatisticsSkippedPassAfterWarn(Base):
    """A baseline WARN about EFA counters must not be waved through by a PASS
    that never actually re-observed those counters.

    `_baseline_allows_warning` (device-session.py:1141) is never consulted on
    this path because the recovered check's own verdict is PASS, not WARN
    (device-session.py:2545-2575): the branch that calls the qualification
    guard is `elif verdict == 'WARN':`, so a PASS bypasses it entirely,
    whatever the PASS text says about which counters were actually read.
    """

    # The exact pinned sentences from checks/6-efa-loopback.sh@a4ba07eb:182,185,
    # in the shape check_pass (lib/common.sh:171-175) emits with no counter or
    # aggregate at all -- these are the two "collection never ran" branches,
    # both of which still fall through to `check_pass` at :193-197 when the
    # loopback test itself passed.
    WARN_COUNTER = ('[WARN] 6-efa-loopback: EFA loopback completed for 2 '
                    'domain(s); cumulative EFA statistics contain drops or '
                    'retransmission timeouts (see logs)\n'
                    '[WARN] EFA retransmission timeouts detected (2)\n')
    PASS_STATS_SKIPPED_NO_RDMA = (
        '[PASS] 6-efa-loopback: EFA loopback OK: 2 domain(s) tested\n')
    # (The pinned check emits the identical PASS sentence whether statistics
    # were skipped or collected clean: verdict text alone cannot distinguish
    # them, which is exactly the referee's point -- see REPRODUCTIONS.md.)

    def qualify(self, baseline_text, recovered_text, check='6', kind='efa'):
        self.executor.check_output[check] = baseline_text
        self.make().start(kind)
        self.executor.check_output[check] = recovered_text
        try:
            return self.make().recover(), None
        except session.Refusal as refusal:
            return None, refusal

    def test_a_statistics_skipped_pass_after_a_counter_warn_still_resumes(self):
        """The defect: baseline WARNs about counters, recovery's PASS never
        re-observed them (the pinned check's own "skipped" branch), and the
        controller resumes anyway because it never runs the WARN guard on a
        PASS verdict.

        EXPECTED (post-repair): refused, node stays drained, no resume issued.
        ACTUAL on f28f0daf (this reproduction): resumes. This is the red test
        for finding 1.
        """
        result, refusal = self.qualify(self.WARN_COUNTER,
                                       self.PASS_STATS_SKIPPED_NO_RDMA)
        self.assertIsNone(
            result,
            'a check-6 PASS that never re-observed the counters a baseline '
            'WARN reported was accepted, and the node was returned to '
            'service without establishing that the counters are still what '
            'the baseline saw')
        assert refusal is not None

    def test_a_genuinely_clean_pass_still_resumes(self):
        """Positive control, same caller shape: no baseline WARN at all, so
        there is nothing this guard needs to re-observe. Must keep resuming.
        """
        result, refusal = self.qualify(self.PASS_STATS_SKIPPED_NO_RDMA,
                                       self.PASS_STATS_SKIPPED_NO_RDMA)
        self.assertIsNone(refusal, f'a clean round with no baseline warning '
                          f'was refused: {refusal}')
        assert result is not None
        self.assertEqual('runtime-ready', result.state.phase)
        self.assertEqual(1, len(self.executor.resumes()))


# ---------------------------------------------------------------------------
# Finding 2 (card body): no local reboot request, plus an external
# pending/unread boot, through first/retry/deadline recovery.  Referee's
# "MAJOR 2" in the final ranked review on t_3b88abe0, source
# device-session.py:2072-2073 (`if not state.reboot_requests: return False`),
# which returns from `_reconcile_outstanding_reboot` before reading either the
# current boot id or the scheduler's pending-reboot flags -- so a controller
# that never issued a reboot itself never checks whether one is nonetheless
# outstanding for the same node.
# ---------------------------------------------------------------------------
class Finding2NoLocalRequestIgnoresExternalPendingReboot(Base):
    """An EFA round that never asked for a reboot must still notice one that
    is pending or already happened for an unrelated reason before it restores
    and resumes.
    """

    def start_efa_and_fail_once(self):
        """An ordinary EFA round whose rebind needs no reboot, and whose first
        restore-runtime attempt fails transiently (so a retry is reached).
        """
        handler = self.make()
        handler.start('efa')
        self.executor.maintenance_failures['restore-runtime'] = (
            1, 'Missing staged image: /opt/aim344/aim344.sqsh\n')
        with self.assertRaises(session.Refusal):
            self.make().recover()
        self.assertEqual('investigating', self.make().store.read().phase)
        del self.executor.maintenance_failures['restore-runtime']

    def test_a_retry_ignores_an_external_reboot_still_pending_with_the_scheduler(self):
        """The defect: between attempts, something outside this round issues a
        reboot (an operator, a health-check remediation, anything). The
        scheduler now holds REBOOT_ISSUED for the node and its boot id has
        already changed. This round never asked for that reboot, so
        `state.reboot_requests` is 0, and `_reconcile_outstanding_reboot`
        returns False at its very first line without reading either fact.

        EXPECTED (post-repair): the retry refuses before any mutating call --
        `_reconcile_outstanding_reboot` runs before the rebind decision
        (device-session.py:2357-2359), so a correct refusal here must precede
        `efa-rebind`/`remount-staging`/`restore-runtime` too, not merely
        precede RESUME.
        ACTUAL on f28f0daf (this reproduction): no exception at all -- the
        retry proceeds through rebind and restoration and resumes the node
        while the scheduler still holds a reboot request for it.
        """
        self.start_efa_and_fail_once()
        # Simulate the external reboot: the scheduler now holds the flag, and
        # the node has already come back on a different boot -- both facts an
        # attentive retry would have to observe before restoring anything.
        self.executor.node_state = 'IDLE+DRAIN+REBOOT_ISSUED'
        self.executor.boot_id = 'deadbeef-0000-0000-0000-000000000009'
        before_actions = list(self.executor.actions())
        try:
            result = self.make().recover()
        except session.Refusal as refusal:
            # Expected once repaired. The refusal must come BEFORE any
            # mutating call this attempt makes, not merely before RESUME.
            new_actions = self.executor.actions()[len(before_actions):]
            for mutation in ('efa-rebind', 'remount-staging', 'restore-runtime'):
                self.assertNotIn(
                    mutation, new_actions,
                    f'{mutation} ran before the pending-reboot refusal: {refusal}')
            self.assertEqual([], self.executor.resumes())
            return
        self.fail(
            f'recovery restored and resumed the node while the scheduler '
            f'still held a reboot request this round never issued, and '
            f'without reading the boot id that had already changed: '
            f'phase={result.state.phase!r}')

    def test_an_ordinary_retry_with_no_external_reboot_still_succeeds(self):
        """Positive control, same caller shape: nothing external happened
        between attempts, so the retry must recover normally.
        """
        self.start_efa_and_fail_once()
        result = self.make().recover()
        self.assertEqual('runtime-ready', result.state.phase)
        self.assertEqual(1, len(self.executor.resumes()))

    def test_the_deadline_path_has_the_same_gap(self):
        """`expire` shares `_recover`, so an abandoned table meets it too."""
        self.start_efa_and_fail_once()
        self.executor.node_state = 'IDLE+DRAIN+REBOOT_ISSUED'
        self.executor.boot_id = 'deadbeef-0000-0000-0000-000000000009'
        before_actions = list(self.executor.actions())
        try:
            result = self.make(now=9999.0).expire()
        except session.Refusal as refusal:
            new_actions = self.executor.actions()[len(before_actions):]
            for mutation in ('efa-rebind', 'remount-staging', 'restore-runtime'):
                self.assertNotIn(
                    mutation, new_actions,
                    f'{mutation} ran before the deadline-path refusal: {refusal}')
            self.assertEqual([], self.executor.resumes())
            return
        self.fail(
            f'the deadline path restored and resumed the node while the '
            f'scheduler still held a reboot request this round never issued: '
            f'phase={result.state.phase!r}')


# ---------------------------------------------------------------------------
# Finding 3 (card body): intact markerless originals repeat, then both
# token/file loss leads to recapture. Referee's D3/D4 unit and "MAJOR 3" on
# t_3b88abe0, source maintenance.py:785-822. The existing D3 tests in
# test_safety_rework6.py cover exactly ONE record losing its operation token
# while its partner survives, which the surviving named record protects. What
# they do not cover is the combined case the card body and the referee both
# name: an intact-but-markerless pair survives one repeat (confirmed by the
# existing positive control `test_a_markerless_intact_pair_is_still_a_usable_
# repeat`), and the repeat itself does not write the marker either (referee's
# citation: maintenance.py:785-802 only prints; the sole marker write is in
# the opposite first-capture `else` branch at :816). If BOTH tokens are then
# lost, `_has_captured_before` has neither kind of evidence and returns
# False, so the already-mutated node's current (paused/disabled) reading is
# recorded as if it were the untouched original.
# ---------------------------------------------------------------------------
class Finding3MarkerlessRepeatThenBothTokensLostRecaptures(RealHelperBench):
    """A repeat of an intact markerless pair must not become recapturable
    once both records subsequently lose their operation token.
    """

    def intact_pair_survives_one_markerless_repeat(self):
        """Build the exact on-disk state the card names: a first capture, its
        marker deleted (the pre-marker-version shape), then one accepted
        repeat -- confirmed by the existing positive control to proceed
        without writing a marker of its own.

        Whether the repeat itself writes a NEW marker is an observation of
        this baseline (f28f0daf), not an acceptance contract: a valid repair
        could choose to make the repeat write a marker (closing the gap that
        way instead of by consulting surviving records), and doing so would
        not violate the safety property this reproduction actually tests --
        that the true original values survive a subsequent loss of both
        tokens. So the marker state is recorded for the log rather than
        asserted; only the pre/post record VALUES are the assertion.
        """
        self.set_load_state('loaded')
        code, output = self.capture(maintenance.gpu_prepare, self.config,
                                    {'operation': TABLE1_R1})
        self.assertEqual(0, code, output)
        for name in self.markers():
            (self.records / name).unlink()
        self.assertEqual([], self.markers(), 'this bench needs a markerless pair')
        before = {name: self.read_record(name)
                 for name in ('native-dcgm-state.txt', 'selected-persistence-mode.txt')}
        # The repeat the card names: intact markerless pair, accepted.
        code, output = self.capture(maintenance.gpu_prepare, self.config,
                                    {'operation': TABLE1_R1})
        self.assertEqual(0, code, output)
        # Baseline observation (f28f0daf): recorded, not asserted as contract
        # -- see the docstring above.
        print('[finding3 baseline observation] markers after the accepted '
              'markerless repeat: {!r} (f28f0daf writes none; a repair MAY '
              'write one here instead of closing the gap via '
              'surviving-record evidence, and that would still satisfy this '
              "reproduction's actual assertion below)".format(self.markers()))
        self.assertEqual(before, {name: self.read_record(name) for name in before},
                         'the repeat itself already altered the originals')
        return before

    def test_losing_both_tokens_after_the_repeat_lets_the_node_recapture(self):
        """The defect: with no marker and no surviving named record, a THIRD
        call treats the node as never captured and records its current
        (already-paused/disabled) state as the original.

        EXPECTED (post-repair): refused, original active/Enabled evidence
        preserved untouched.
        ACTUAL on f28f0daf (this reproduction): both records are silently
        overwritten with the current inactive/Disabled reading, permanently
        losing the true original the first capture observed.
        """
        before = self.intact_pair_survives_one_markerless_repeat()
        self.strip_operation('native-dcgm-state.txt')
        self.strip_operation('selected-persistence-mode.txt')
        code, output = self.capture(maintenance.gpu_prepare, self.config,
                                    {'operation': TABLE1_R1})
        self.assertIsInstance(
            code, maintenance.Refusal,
            f'a repeat with both original-state tokens lost was accepted '
            f'(rc={code!r}), overwriting the true original with the node\'s '
            f'current state: {output!r}')
        after = {name: self.read_record(name) for name in before}
        self.assertEqual(
            before['native-dcgm-state.txt']['value'], after['native-dcgm-state.txt']['value'],
            'the true original telemetry state (active) was overwritten '
            'with the current paused state')
        self.assertEqual(
            before['selected-persistence-mode.txt']['value'],
            after['selected-persistence-mode.txt']['value'],
            'the true original persistence mode (Enabled) was overwritten '
            'with the current disabled state')

    def test_losing_only_one_token_after_the_repeat_still_refuses(self):
        """Positive control, same caller shape and same repeat history: the
        SINGLE-token-loss case the existing D3 tests already cover must keep
        refusing even after this extra intervening repeat.
        """
        before = self.intact_pair_survives_one_markerless_repeat()
        self.strip_operation('selected-persistence-mode.txt')
        outcome, output = self.capture(maintenance.gpu_prepare, self.config,
                                       {'operation': TABLE1_R1})
        self.assertIsInstance(outcome, maintenance.Refusal, output)
        surviving = self.read_record('native-dcgm-state.txt')
        self.assertEqual(before['native-dcgm-state.txt'], surviving)


# ---------------------------------------------------------------------------
# Finding 4 (card body): legacy operation token/restoration obligation
# missing before mutations. Referee's cross-unit finding on t_3b88abe0
# ("Cross-unit recovery prerequisite residual", 2026-09-18 06:40 UTC) and
# "MAJOR 4" in the final ranked review, source
# device-session.py:2285-2326 (`_require_recoverable_preparation`), which
# returns as soon as ANY mutation marker is present (:2319-2320,
# `MUTATION_MARKERS = ('gpu-remove-attempted', 'efa-unbind-attempted')`)
# without checking whether the round also carries a valid operation token or
# a matching preparation marker. `gpu-restore` is only called later, gated on
# `'gpu-prepare' in state.prepared or 'gpu-prepare-attempted' in
# state.prepared` (:2478) -- a DIFFERENT marker subset than the one that let
# recovery proceed. A legacy record naming only `gpu-remove-attempted` (no
# `gpu-prepare*` marker, no operation token) is therefore accepted at
# :2319-2320, rebooted, and its `gpu-restore` skipped entirely at :2478 --
# reaching RESUME with clean checks despite never restoring any original
# telemetry/persistence state, and without ever validating the operation
# token _require_operation would have refused it on had gpu-restore been
# reached at all.
# ---------------------------------------------------------------------------
class Finding4LegacyMutationMarkerSkipsRestorationObligation(Base):
    """A round whose only journalled evidence is a device-mutation-attempt
    marker (no preparation marker, no operation token) must not reboot and
    resume without any check for the restoration obligation that marker
    subset does not cover.
    """

    def legacy_record_with_mutation_marker_only(self, prepared_extra):
        return session.State(
            phase='investigating', kind='gpu', started_at=1.0,
            target_node='gpu-g7-1', target_instance_id='i-0123456789abcdef0',
            boot_id='dddddddd-0000-0000-0000-00000000000b',
            prepared=['drain'] + prepared_extra, operation=None)

    def test_a_mutation_marker_alone_reboots_and_resumes_without_restoring(self):
        """The defect: `gpu-remove-attempted` alone satisfies
        `_mutation_was_attempted` at :2319-2320 and the round proceeds to
        reboot and restore-runtime. `gpu-restore` is then skipped because
        neither `gpu-prepare` nor `gpu-prepare-attempted` is in `prepared`
        (:2478), so no telemetry/persistence restoration is even attempted,
        and the missing operation token that `_require_operation` would have
        caught inside `gpu-restore` is never reached to catch it.

        EXPECTED (post-repair): refused before any reboot -- a `Refusal` here
        is only a pass if it precedes the reboot/restore-runtime calls; a
        refusal raised AFTER those calls already ran (e.g. inside
        gpu-restore, too late to undo the reboot) is not the fix the card
        asks for, so this asserts on the executor's action history rather
        than on Refusal-or-not alone.
        ACTUAL on f28f0daf (this reproduction): reboots, skips gpu-restore
        entirely, and resumes on clean checks -- no Refusal at all.
        """
        legacy = self.legacy_record_with_mutation_marker_only(
            ['gpu-remove-attempted'])
        self.make().store.write(legacy)
        try:
            result = self.make().recover()
        except session.Refusal as refusal:
            self.assertEqual(
                [], self.executor.reboots(),
                f'a reboot was issued before the refusal caught the missing '
                f'restoration obligation: {refusal}')
            self.assertEqual([], self.executor.resumes())
            return
        self.assertNotEqual(
            'runtime-ready', result.state.phase,
            'a legacy record with only a mutation-attempt marker (no '
            'operation token, no preparation marker) was rebooted and '
            'resumed without gpu-restore ever running')
        self.assertNotIn('gpu-restore', self.executor.actions(),
                         'gpu-restore ran without a usable operation token')

    def test_a_preparation_marker_alongside_it_still_recovers_normally(self):
        """Positive control, same caller shape: the SAME legacy record shape
        but WITH a preparation marker present (so `gpu-restore` is reached and
        can itself refuse on the missing token, or restore if a token
        exists) must not be blocked by whatever repair closes the gap above.
        """
        legacy = self.legacy_record_with_mutation_marker_only(
            ['gpu-remove-attempted', 'gpu-prepare-attempted'])
        legacy.operation = 'table-1/1/' + '0' * 32
        self.make().store.write(legacy)
        result = self.make().recover()
        self.assertEqual('runtime-ready', result.state.phase)
        self.assertIn('gpu-restore', self.executor.actions())


# ---------------------------------------------------------------------------
# Finding 5 (card body): intent/original-write ordering lacks file+directory
# fsync before dependent side effect; failure injection must assert the side
# effect is not issued. Referee's Evidence/privilege unit and "MAJOR 5" on
# t_3b88abe0, source device-session.py:351-361 (`StateStore.write`) and
# maintenance.py:304-316 (`_write_record`), both of which `os.replace()` a
# temp file into place with no `os.fsync` on either the file descriptor or
# the containing directory. Per fsync(2): "Calling fsync() does not
# necessarily ensure that the entry in the directory containing the file has
# also reached disk. For that an explicit fsync() on a file descriptor for
# the directory is also needed."
# ---------------------------------------------------------------------------
class Finding5WritesLackFsyncBeforeDependentSideEffect(unittest.TestCase):
    """Every write this controller/helper treats as durable evidence must
    actually be flushed to storage -- file AND containing directory -- before
    anything that depends on that evidence being crash-durable proceeds.

    Scope note (see REPRODUCTIONS.md for the full disclosure): the tests
    below in this class establish two DIFFERENT things, and neither is the
    full "inject an fsync failure and assert the dependent side effect is not
    issued" contract the card asks for on its own:

      1. presence -- `test_*_fsyncs_the_file_and_its_directory` shows zero
         `os.fsync` calls happen at all on f28f0daf. This is a real
         reproduction of the missing durability primitive, but it is a
         weaker claim than "file synced, then directory synced, in that
         order": with zero calls there is no order to inspect.
      2. failure ordering on the WRITE call itself (not on fsync, which does
         not exist yet to fail) -- `test_a_failed_original_state_write_does_
         not_reach_the_dependent_mutation` injects a failure into the
         existing write call (`Path.write_text`) and confirms the ALREADY-
         CORRECT ordering property: gpu_prepare does not proceed to pause
         telemetry or change persistence when the record write raises.

    What is NOT reproduced here, and is recorded as such in REPRODUCTIONS.md:
    "does adding fsync, and having fsync itself fail, correctly block the
    dependent side effect" cannot be exercised against f28f0daf, because
    there is no fsync call in the baseline to inject a failure into -- the
    call site the card asks about does not exist yet. That half of the
    contract is only checkable once the repair adds it, on the repaired tree.
    """

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.real_fsync = os.fsync
        self.fsync_calls = []

    def spy_fsync(self):
        def spy(fd):
            self.fsync_calls.append(fd)
            return self.real_fsync(fd)
        return spy

    def test_statestore_write_calls_fsync_at_least_once(self):
        """Presence check only (see class docstring): does `StateStore.write`
        (device-session.py:351-361) call `os.fsync` at all, on either the
        file descriptor or a directory descriptor. It does
        `os.open(...); json.dump(...); os.chmod(...); os.replace(...)` with
        no `os.fsync` call anywhere in that sequence.

        EXPECTED (post-repair): at least one `os.fsync` call.
        ACTUAL on f28f0daf (this reproduction): zero.
        """
        directory = Path(self.tmp.name) / 'state'
        store = session.StateStore(directory, 'table-1')
        os.fsync = self.spy_fsync()
        try:
            store.write(session.State())
        finally:
            os.fsync = self.real_fsync
        self.assertGreater(
            len(self.fsync_calls), 0,
            'StateStore.write persisted a state record with zero fsync '
            'calls, so a host crash immediately after can lose an already-'
            'issued operation\'s intent record without trace')

    def test_write_record_calls_fsync_at_least_once(self):
        """Presence check only (see class docstring), for the helper's
        original-state records (maintenance.py:304-316, `_write_record`):
        `temporary.write_text(...); os.replace(temporary, target)`, no fsync
        at all.

        EXPECTED (post-repair): at least one `os.fsync` call.
        ACTUAL on f28f0daf (this reproduction): zero.
        """
        records = Path(self.tmp.name) / 'records'
        records.mkdir()
        original_dir = maintenance.RECORD_DIR
        maintenance.RECORD_DIR = records
        os.fsync = self.spy_fsync()
        try:
            maintenance._write_record('native-dcgm-state.txt', 'active',
                                      TABLE1_R1)
        finally:
            os.fsync = self.real_fsync
            maintenance.RECORD_DIR = original_dir
        self.assertGreater(
            len(self.fsync_calls), 0,
            '_write_record persisted an original-state record with zero '
            'fsync calls, so a crash right after gpu_prepare\'s intent/'
            'original write can lose it while the dependent pause/'
            'persistence-mode mutation that follows still lands')

    def test_a_read_of_a_flushed_file_still_succeeds(self):
        """Positive control, same caller shape: an ordinary write/read round
        trip (no injected crash, no fsync spy) must keep working regardless
        of whatever repair adds the missing fsync calls.
        """
        directory = Path(self.tmp.name) / 'state-control'
        store = session.StateStore(directory, 'table-1')
        written = session.State(phase='ready')
        store.write(written)
        reread = store.read()
        self.assertEqual('ready', reread.phase)


class Finding5FailureInjectionOnTheExistingWriteCall(RealHelperBench):
    """The failure-injection half of the card's finding 5, on the write call
    that DOES exist in f28f0daf: `_write_record`'s underlying
    `Path.write_text`. This is a real ordering property already present in
    the baseline, exercised through the actual `gpu_prepare` caller rather
    than by calling `_write_record` directly, so it demonstrates the
    controller-level consequence: a failed original-state write must not be
    followed by the mutation it was meant to make recoverable.

    This does NOT stand in for "an fsync failure blocks the side effect" --
    see Finding5WritesLackFsyncBeforeDependentSideEffect's class docstring
    and REPRODUCTIONS.md for why that half is NOT REPRODUCED on this tree.
    """

    def test_a_failed_original_state_write_does_not_reach_the_dependent_mutation(self):
        """Inject a failure into the SECOND original-state write inside
        `gpu_prepare` (`selected-persistence-mode.txt`, maintenance.py:822)
        so the first (`native-dcgm-state.txt`, :821) has already landed. If
        the write itself fails, `gpu_prepare` must not proceed to
        `systemctl stop` (the dependent mutation the write was supposed to
        make recoverable).

        This already passes on f28f0daf: it establishes the ordering
        property this bench CAN check without adding fsync, as a baseline
        fact for the implementer, not as evidence the durability gap is
        closed.
        """
        self.set_load_state('loaded')
        original_write_text = Path.write_text
        calls = {'n': 0}

        def failing_write_text(self_path, *args, **kwargs):
            calls['n'] += 1
            if self_path.name.startswith('selected-persistence-mode.txt'):
                raise OSError('simulated write failure (disk full, etc.)')
            return original_write_text(self_path, *args, **kwargs)

        Path.write_text = failing_write_text
        try:
            with self.assertRaises(OSError):
                self.capture(maintenance.gpu_prepare, self.config,
                            {'operation': TABLE1_R1})
        finally:
            Path.write_text = original_write_text
        self.assertEqual(
            'active', self.unit_state(),
            'systemctl stop ran even though the original-state write for '
            'this operation failed partway through')
        self.assertEqual('Enabled', self.persistence())
        for call in self.systemctl_calls():
            self.assertNotIn('stop', call,
                             f'a stop was issued despite the failed write: {call}')

    def test_a_successful_write_still_reaches_the_dependent_mutation(self):
        """Positive control, same caller shape: an ordinary successful
        capture (no injected failure) must still pause telemetry and disable
        persistence as before.
        """
        self.set_load_state('loaded')
        code, output = self.capture(maintenance.gpu_prepare, self.config,
                                    {'operation': TABLE1_R1})
        self.assertEqual(0, code, output)
        self.assertEqual('inactive', self.unit_state())
        self.assertEqual('Disabled', self.persistence())


# ---------------------------------------------------------------------------
# Finding 6 (card body): accepted memlock WARN lacks pinned producer/check
# identity/actual supported capture. Referee's D5 primary-source unit and
# "MAJOR 6" on t_3b88abe0, source device-session.py:1141
# (`_baseline_allows_warning`), which never checks the `check` argument it
# was called with against the check-name PREFIX embedded in the warning text
# itself (device-session.py:924-925, `warning_details`: `name, body =
# match.group(1), match.group(2)` -- captured, then only used to build the
# comparable `condition` string, never compared against the caller's own
# `check` id). maintenance.collect (maintenance.py:200-203) prints
# `# suite_revision=...` only when the REVISION file exists, and
# `_capture_baseline` (device-session.py:1636-1681) stores no suite identity
# at all in the baseline record. So a WARN whose own embedded prefix names a
# DIFFERENT check than the one the controller asked for -- exactly what a
# producer-identity mismatch would look like on the wire -- is accepted as
# long as the condition TEXT after the prefix matches.
# ---------------------------------------------------------------------------
class Finding6MemlockWarnQualifiesWithoutProducerIdentity(Base):
    """A memlock WARN must not auto-qualify unless it actually came from the
    check the controller asked for and the pinned producer it trusts.
    """

    def test_a_warn_whose_embedded_check_name_does_not_match_still_qualifies(self):
        """The defect: both the baseline and the recovered capture for check
        '2' (memlock/EFA enumeration) actually carry warning lines whose
        embedded check-name prefix says '6-efa-loopback' -- a different
        check entirely. `_baseline_allows_warning` is called with
        `check='2'` but never reads that embedded name at all; it only
        compares condition TEXT (after stripping whatever prefix is
        present) between the two captures. Since both captures carry the
        same wrong-named text, they compare equal and qualify.

        This is exactly what a suite-identity mismatch, a pin drift, or a
        maintenance-route bug that returned the wrong check's output would
        produce on the wire: a WARN whose own text does not agree with what
        was asked for, silently accepted anyway.

        EXPECTED (post-repair): refused -- the embedded check name '6-efa-
        loopback' does not match the requested check '2'.
        ACTUAL on f28f0daf (this reproduction): resumes.
        """
        capture_wrong_name = (
            '[WARN] 6-efa-loopback: Memory lock limit 8192 KB is below 16 '
            'GiB -- EFA performance may be degraded\n'
            '[WARN] 6-efa-loopback: EFA devices enumerated with 1 '
            'advisories; inspect raw output\n')
        self.executor.check_output['2'] = capture_wrong_name
        self.make().start('efa')
        self.executor.check_output['2'] = capture_wrong_name
        try:
            result = self.make().recover()
        except session.Refusal:
            return
        self.fail(
            f'a memlock WARN whose own embedded check-name prefix names a '
            f'different check than the one requested (2) was accepted and '
            f'the node was resumed: phase={result.state.phase!r}')

    def test_a_warn_whose_embedded_check_name_matches_is_still_unsupported(self):
        """Final finding 5 supersedes this frozen positive expectation.

        This unchanged source-shaped fixture is not a supported real capture.
        Correct names alone must not authorize RESUME. Original test bytes and
        passing baseline execution remain in rework-final qualification evidence.
        """
        capture_correct_name = (
            '[WARN] 2-efa-enumeration: Memory lock limit 8192 KB is below '
            '16 GiB -- EFA performance may be degraded\n'
            '[WARN] 2-efa-enumeration: EFA devices enumerated with 1 '
            'advisories; inspect raw output\n')
        self.executor.check_output['2'] = capture_correct_name
        self.make().start('efa')
        self.executor.check_output['2'] = capture_correct_name
        with self.assertRaises(session.Refusal) as caught:
            self.make().recover()
        self.assertIn('auto-qualification is disabled', str(caught.exception))
        self.assertEqual('replacement-required', self.make().store.read().phase)
        self.assertEqual([], self.executor.resumes())
