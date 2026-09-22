# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Additional verification for the durable-write repair (card finding 5).

The independent reproductions in `test_executor_first_reproductions.py` establish
that `StateStore.write` and `maintenance._write_record` call `os.fsync` at all, which
is a PRESENCE check: their own docstrings say so, and they explicitly do not check
the synchronization ORDER or what happens when a synchronization fails. That second
half is what the card requires -- "durable file and directory synchronization before
side effects" -- and it only became checkable once the repair added the call sites, so
it is verified here rather than there.

What each test establishes, and what none of them claims:

  the ORDER, on the real call path
      file fsync, then rename, then directory fsync, then the dependent operation. The
      order is recorded by spying on the real primitives during a real `gpu_prepare`
      and a real `StateStore.write`, not by reading the source.

  a FAILURE of either synchronization blocks the dependent side effect
      including a directory fsync that fails with EINVAL, which is the case a
      mid-run review specifically named: an earlier version of this repair let EINVAL
      return normally on the reasoning that some filesystems cannot synchronize a
      directory descriptor, and that would let the round proceed when the durability
      the next step relies on was NOT established. No EINVAL was observed on the
      exercise filesystem; it is injected here because the code must not depend on the
      platform to be safe.

  a SUCCESS control on the same call path
      the same round with no injected failure still prepares the node and still writes
      its record, so 'refuse whenever anything is synchronized' would fail these.

Not established here: that any real filesystem in this exercise ever fails these
calls, and that a host crash was survived. Nothing here powers off a machine; the
subject is the ordering and the refusal, which is what the controller can decide.
"""
import errno
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import unittest

_safety3_spec = importlib.util.spec_from_file_location(
    'aim344_safety3_for_durability',
    Path(__file__).resolve().parent / 'test_safety_rework3.py')
if _safety3_spec is None or _safety3_spec.loader is None:    # pragma: no cover
    raise unittest.SkipTest('test_safety_rework3.py not importable')
_safety3 = importlib.util.module_from_spec(_safety3_spec)
_safety3_spec.loader.exec_module(_safety3)
maintenance = _safety3.maintenance
session = _safety3.session
TABLE1_R1 = _safety3.TABLE1_R1
_RealHelper = _safety3.S4TelemetryRestoration
# The controller bench, taken from the same module the S4 fixture was built against
# so `session` here is the SAME module object the handler raises its Refusal from.
# Loading device-session.py again would create a second Refusal class, and
# assertRaises would then report an error instead of a pass or a failure.
_rework = _safety3._rework


class DurabilityOrderingOnTheRealHelper(unittest.TestCase):
    """`gpu_prepare`'s record writes, with the real record files on disk.

    The fixture is the same real-helper bench the S4 tests use, borrowed by reference
    rather than by subclassing for the reason that file documents beside its own copy:
    a TestCase subclass in this module's namespace would be collected twice.
    """

    setUp = _RealHelper.setUp
    tearDown = _RealHelper.tearDown
    capture = _RealHelper.capture
    unit_state = _RealHelper.unit_state
    persistence = _RealHelper.persistence
    set_load_state = _RealHelper.set_load_state
    systemctl_calls = _RealHelper.systemctl_calls

    def trace_durability(self, fail_on=None, error=None):
        """Record the real fsync/rename/mutation order, optionally failing one sync.

        `fail_on` is 'file', 'directory' or None. The spy wraps `os.fsync`,
        `os.replace` AND `subprocess.run` for the duration of one call, so the
        recorded sequence carries the dependent mutation in the same list as the
        synchronizations. That is the point: a sequence holding only fsync and rename
        events shows the order of the writes among themselves, not their order
        relative to the side effect they are supposed to precede.
        """
        events = []
        real_fsync = os.fsync
        real_replace = os.replace
        redirected = self.redirected

        def spy_fsync(fd):
            try:
                is_dir = os.fstat(fd).st_mode & 0o170000 == 0o040000
            except OSError:                                   # pragma: no cover
                is_dir = False
            kind = 'directory' if is_dir else 'file'
            events.append(f'fsync-{kind}')
            if fail_on == kind:
                raise OSError(error or errno.EIO, 'injected synchronization failure')
            return real_fsync(fd)

        def spy_replace(src, dst):
            events.append('rename')
            return real_replace(src, dst)

        def spy_run(argv, **kwargs):
            argv = list(argv)
            name = Path(str(argv[0])).name
            # Only the calls that CHANGE the node are recorded as mutations. The
            # helper's own observations (`systemctl show`, the persistence-mode query)
            # read state and are not what has to follow the synchronization.
            if name == 'systemctl' and 'stop' in argv:
                events.append('systemctl-stop')
            elif name == 'nvidia-smi' and any(
                    str(a).startswith('--persistence-mode=') for a in argv):
                events.append('persistence-mode-write')
            return redirected(argv, **kwargs)

        os.fsync = spy_fsync
        os.replace = spy_replace
        maintenance.subprocess.run = spy_run
        self.addCleanup(setattr, os, 'fsync', real_fsync)
        self.addCleanup(setattr, os, 'replace', real_replace)
        # `subprocess.run` is deliberately NOT restored by a cleanup here: the borrowed
        # tearDown already restores it, and cleanups run AFTER tearDown, so a cleanup
        # of our own would put this spy back in place after the fixture had removed it
        # and leak it into the next test. Measured, not reasoned about: doing that made
        # the following test's fixture capture the spy as its own 'real' run and
        # recurse.
        return events

    def test_the_record_is_synchronized_and_renamed_before_the_mutation(self):
        """The ordering the card asks for, on the call path that has the side effect.

        `gpu_prepare` writes the telemetry and persistence originals and then pauses
        the telemetry unit and disables persistence. The recorded sequence therefore
        has to show, in one list: every record synchronized and renamed and its
        directory entry synchronized, and only THEN the calls that change the node.
        """
        self.set_load_state('loaded')
        events = self.trace_durability()
        code, output = self.capture(maintenance.gpu_prepare, self.config,
                                    {'operation': TABLE1_R1})
        self.assertEqual(0, code, output)
        # Three records are written before the mutation: the capture marker and the two
        # original-state records. Each contributes the same three events in order, and
        # the two mutating calls follow all nine.
        self.assertEqual(
            ['fsync-file', 'rename', 'fsync-directory'] * 3
            + ['systemctl-stop', 'persistence-mode-write'],
            events,
            f'unexpected durability/mutation sequence: {events}')
        # The mutations really happened, so this is the ordering of a round that
        # proceeded rather than one that stopped early.
        self.assertEqual('inactive', self.unit_state())
        self.assertEqual('Disabled', self.persistence())

    def test_a_failed_file_synchronization_blocks_the_mutation(self):
        self.set_load_state('loaded')
        self.trace_durability(fail_on='file')
        outcome, output = self.capture(maintenance.gpu_prepare, self.config,
                                       {'operation': TABLE1_R1})
        self.assertIsInstance(outcome, maintenance.Refusal, output)
        self.assertIn('synchronized to storage', str(outcome))
        self.assertEqual('active', self.unit_state(),
                         'telemetry was paused after a record write that was not '
                         'established to survive a restart')
        self.assertEqual('Enabled', self.persistence())
        for call in self.systemctl_calls():
            self.assertNotIn('stop', call, f'a stop was issued: {call}')

    def test_a_failed_directory_synchronization_blocks_the_mutation(self):
        """A renamed file whose directory entry was not synchronized is not durable.

        fsync(2) is explicit that the two are separate, so this failure has the same
        consequence as the file's own: the record may not name anything after a restart,
        while the mutation it was supposed to explain would already have happened.
        """
        self.set_load_state('loaded')
        self.trace_durability(fail_on='directory')
        outcome, output = self.capture(maintenance.gpu_prepare, self.config,
                                       {'operation': TABLE1_R1})
        self.assertIsInstance(outcome, maintenance.Refusal, output)
        self.assertIn('synchronized to storage', str(outcome))
        self.assertEqual('active', self.unit_state())
        self.assertEqual('Enabled', self.persistence())
        for call in self.systemctl_calls():
            self.assertNotIn('stop', call, f'a stop was issued: {call}')

    def test_an_einval_directory_synchronization_also_blocks_the_mutation(self):
        """EINVAL is not permission to continue, and this is why it is asserted.

        An earlier version of this repair returned normally on EINVAL, reasoning that
        some filesystems cannot synchronize a directory descriptor at all. That turns
        'the platform cannot establish durability' into 'durability is established',
        which is the inference this whole finding is about. No EINVAL was observed on
        the exercise filesystem; it is injected here so the refusal does not depend on
        which filesystem the exercise happens to run on.
        """
        self.set_load_state('loaded')
        self.trace_durability(fail_on='directory', error=errno.EINVAL)
        outcome, output = self.capture(maintenance.gpu_prepare, self.config,
                                       {'operation': TABLE1_R1})
        self.assertIsInstance(outcome, maintenance.Refusal, output)
        self.assertEqual('active', self.unit_state())
        self.assertEqual('Enabled', self.persistence())

    def test_an_ordinary_capture_with_working_synchronization_still_prepares(self):
        """Positive control: nothing injected, so nothing refused.

        Without this, 'refuse whenever a synchronization is attempted' would pass every
        test above while making the exercise unrunnable.
        """
        self.set_load_state('loaded')
        code, output = self.capture(maintenance.gpu_prepare, self.config,
                                    {'operation': TABLE1_R1})
        self.assertEqual(0, code, output)
        record = json.loads((self.records / 'native-dcgm-state.txt').read_text())
        self.assertEqual('active', record['value'])
        self.assertEqual(TABLE1_R1, record['operation'])
        self.assertEqual('inactive', self.unit_state())
        self.assertEqual('Disabled', self.persistence())


class HelperCaptureRetry(unittest.TestCase):
    """Final-review findings 6/7 on the existing real-helper executable bench.

    No TestCase subclass is imported for these cases. The borrowed fixture methods
    do not cause inherited regression cases to be counted as newly added tests.
    """

    setUp = _safety3.S4TelemetryRestoration.setUp
    tearDown = _safety3.S4TelemetryRestoration.tearDown
    capture = _safety3.S4TelemetryRestoration.capture
    unit_state = _safety3.S4TelemetryRestoration.unit_state
    persistence = _safety3.S4TelemetryRestoration.persistence

    def prepare(self, fresh=False):
        return self.capture(maintenance.gpu_prepare, self.config,
                            {'operation': TABLE1_R1, 'fresh_capture': fresh})

    def records_now(self):
        return {p.name: p.read_text() for p in sorted(self.records.iterdir())}

    def observe(self, label, function):
        """Print actual before/after files, helper output and issued argv, even red."""
        before = self.records_now()
        calls = []
        redirected = self.redirected

        def recording(argv, **kwargs):
            calls.append(list(argv))
            return redirected(argv, **kwargs)

        maintenance.subprocess.run = recording
        try:
            result, output = function()
        finally:
            maintenance.subprocess.run = redirected
        print(json.dumps({'case': label, 'before': before,
                          'after': self.records_now(), 'result': str(result),
                          'helper_output': output, 'issued_argv': calls,
                          'telemetry': self.unit_state(),
                          'persistence': self.persistence()}, sort_keys=True))
        return result, output, calls, before

    def legacy_pair(self):
        result, output = self.prepare(fresh=True)
        self.assertEqual(0, result, output)
        (self.records / maintenance._capture_marker_name(TABLE1_R1)).unlink()

    def late_loss(self, remove):
        self.legacy_pair()
        for path in tuple(self.records.iterdir()):
            if remove:
                path.unlink()
            else:
                record = json.loads(path.read_text())
                record.pop('operation')
                path.write_text(json.dumps(record))
        result, output, calls, before = self.observe(
            'markerless-pair-' + ('files-lost' if remove else 'tokens-lost'),
            self.prepare)
        self.assertEqual(before, self.records_now(), 'lost originals were recaptured')
        self.assertIsInstance(result, maintenance.Refusal, output)
        self.assertEqual([], calls, 'ambiguous repeat reached node operations')

    def test_markerless_tokens_lost_before_repeat_refuses(self):
        self.late_loss(remove=False)

    def test_markerless_files_lost_before_repeat_refuses(self):
        self.late_loss(remove=True)

    def test_empty_legacy_request_is_not_fresh_authorization(self):
        result, output, calls, before = self.observe('empty-legacy', self.prepare)
        self.assertIsInstance(result, maintenance.Refusal, output)
        self.assertEqual(before, self.records_now())
        self.assertEqual([], calls)

    def test_explicit_first_capture_prepares(self):
        result, output, calls, _ = self.observe(
            'authorized-first', lambda: self.prepare(fresh=True))
        self.assertEqual(0, result, output)
        self.assertEqual('inactive', self.unit_state())
        self.assertEqual('Disabled', self.persistence())
        self.assertTrue(any('stop' in c for c in calls))
        self.assertTrue(any('--persistence-mode=0' in c for c in calls))
        for name, value in [('native-dcgm-state.txt', 'active'),
                            ('selected-persistence-mode.txt', 'Enabled')]:
            self.assertEqual(value, json.loads((self.records / name).read_text())['value'])

    def test_fresh_flag_does_not_authorize_untagged_legacy_records(self):
        self.legacy_pair()
        for path in self.records.iterdir():
            record = json.loads(path.read_text())
            record.pop('operation')
            path.write_text(json.dumps(record))
        result, output, calls, before = self.observe(
            'fresh-flag-with-untagged-records', lambda: self.prepare(fresh=True))
        self.assertIsInstance(result, maintenance.Refusal, output)
        self.assertEqual(before, self.records_now())
        self.assertEqual([], calls)

    def test_fresh_flag_does_not_override_surviving_capture_marker(self):
        self.assertEqual(0, self.prepare(fresh=True)[0])
        for name in ('native-dcgm-state.txt', 'selected-persistence-mode.txt'):
            (self.records / name).unlink()
        result, output, calls, before = self.observe(
            'fresh-flag-after-record-loss', lambda: self.prepare(fresh=True))
        self.assertIsInstance(result, maintenance.Refusal, output)
        self.assertEqual(before, self.records_now())
        self.assertEqual([], calls)

    def test_fresh_flag_on_intact_retry_keeps_originals(self):
        self.assertEqual(0, self.prepare(fresh=True)[0])
        result, output, _, before = self.observe(
            'fresh-flag-intact-retry', lambda: self.prepare(fresh=True))
        self.assertEqual(0, result, output)
        self.assertEqual(before, self.records_now())

    def test_fresh_capture_grammar_is_bounded(self):
        for argv in (['gpu-prepare', '--fresh-capture', '--operation', TABLE1_R1],
                     ['gpu-prepare', '--operation', TABLE1_R1, '--fresh-capture']):
            self.assertEqual(('gpu-prepare', {'operation': TABLE1_R1,
                                             'fresh_capture': True}),
                             maintenance.parse(argv))
        for argv in (['gpu-prepare', '--fresh-capture'],
                     ['gpu-restore', '--operation', TABLE1_R1, '--fresh-capture'],
                     ['gpu-prepare', '--operation', TABLE1_R1,
                      '--fresh-capture', '--fresh-capture'],
                     ['gpu-prepare', '--operation', TABLE1_R1, '--fresh-capture=true']):
            with self.subTest(argv=argv), self.assertRaises(maintenance.Refusal):
                maintenance.parse(argv)

    def test_intact_markerless_retry_preserves_originals(self):
        self.legacy_pair()
        originals = self.records_now()
        # Same retry branch, with work to do rather than an inert paused node.
        (self.stub_state / 'unit-state.txt').write_text('active\n')
        (self.stub_state / 'persistence-mode.txt').write_text('Enabled\n')
        result, output, calls, _ = self.observe('intact-markerless', self.prepare)
        self.assertEqual(0, result, output)
        for name, text in originals.items():
            self.assertEqual(text, (self.records / name).read_text())
        self.assertTrue(any('stop' in c for c in calls))
        self.assertTrue(any('--persistence-mode=0' in c for c in calls))
        restored, text = self.capture(maintenance.gpu_restore, self.config,
                                     {'operation': TABLE1_R1})
        self.assertEqual(0, restored, text)
        self.assertEqual('active', self.unit_state())
        self.assertEqual('Enabled', self.persistence())

    def last_sync_retry(self, retry_failure=None, error=errno.EIO):
        real_sync = os.fsync
        directory_calls = 0

        def last_directory_failure(fd):
            nonlocal directory_calls
            if os.path.isdir(f'/proc/self/fd/{fd}'):
                directory_calls += 1
                if directory_calls == 3:
                    raise OSError(error, 'injected last-original directory fsync')
            return real_sync(fd)

        os.fsync = last_directory_failure
        try:
            first, output, calls, _ = self.observe(
                'last-record-sync-failure', lambda: self.prepare(fresh=True))
        finally:
            os.fsync = real_sync
        self.assertIsInstance(first, maintenance.Refusal, output)
        self.assertEqual(3, directory_calls)
        self.assertFalse(any('stop' in c or '--persistence-mode=0' in c for c in calls))
        self.assertEqual('active', self.unit_state())
        self.assertEqual('Enabled', self.persistence())
        originals = self.records_now()
        self.assertIn('selected-persistence-mode.txt', originals)
        events = []
        redirected = self.redirected

        def retry_sync(fd):
            path = Path(os.readlink(f'/proc/self/fd/{fd}'))
            kind = 'directory' if path.is_dir() else 'file'
            events.append(['fsync', kind, path.name])
            if kind == retry_failure:
                raise OSError(error, 'injected retained-original fsync')
            return real_sync(fd)

        def retry_run(argv, **kwargs):
            if 'stop' in argv or '--persistence-mode=0' in argv:
                events.append(['mutation', list(argv)])
            return redirected(argv, **kwargs)

        os.fsync = retry_sync
        self.redirected = retry_run
        try:
            result, output, calls, _ = self.observe('retry-after-last-sync-failure', self.prepare)
        finally:
            os.fsync = real_sync
            self.redirected = redirected
            maintenance.subprocess.run = redirected
        print('DURABILITY_EVENTS ' + json.dumps(events))
        self.assertEqual(originals, self.records_now(), 'retained originals were rewritten')
        if retry_failure:
            self.assertIsInstance(result, maintenance.Refusal, output)
            self.assertFalse(any(e[0] == 'mutation' for e in events))
            self.assertEqual('active', self.unit_state())
            self.assertEqual('Enabled', self.persistence())
        else:
            self.assertEqual(0, result, output)
            mutations = [i for i, e in enumerate(events) if e[0] == 'mutation']
            self.assertTrue(mutations, events)
            prior = events[:mutations[0]]
            for name in originals:
                self.assertIn(['fsync', 'file', name], prior)
            self.assertIn(['fsync', 'directory', self.records.name], prior)
            directory_index = prior.index(['fsync', 'directory', self.records.name])
            self.assertTrue(all(prior.index(['fsync', 'file', n]) < directory_index
                                for n in originals), prior)
            self.assertEqual('inactive', self.unit_state())
            self.assertEqual('Disabled', self.persistence())

    def test_last_record_directory_failure_retry_resynchronizes_before_mutation(self):
        self.last_sync_retry()

    def test_last_record_directory_failure_retry_file_failure_refuses(self):
        self.last_sync_retry(retry_failure='file')

    def test_last_record_directory_failure_retry_directory_failure_refuses(self):
        self.last_sync_retry(retry_failure='directory')

    def test_last_record_einval_retry_einval_refuses(self):
        self.last_sync_retry(retry_failure='directory', error=errno.EINVAL)


class StateStoreDurabilityOrdering(unittest.TestCase):
    """The controller's own state record, which every side effect is written before."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name) / 'state'

    def trace(self, fail_on=None, error=None):
        events = []
        real_fsync = os.fsync
        real_replace = os.replace

        def spy_fsync(fd):
            try:
                is_dir = os.fstat(fd).st_mode & 0o170000 == 0o040000
            except OSError:                                   # pragma: no cover
                is_dir = False
            kind = 'directory' if is_dir else 'file'
            events.append(f'fsync-{kind}')
            if fail_on == kind:
                raise OSError(error or errno.EIO, 'injected synchronization failure')
            return real_fsync(fd)

        def spy_replace(src, dst):
            events.append('rename')
            return real_replace(src, dst)

        os.fsync = spy_fsync
        os.replace = spy_replace
        self.addCleanup(setattr, os, 'fsync', real_fsync)
        self.addCleanup(setattr, os, 'replace', real_replace)
        return events

    def test_the_state_record_is_synchronized_then_renamed_then_the_directory(self):
        store = session.StateStore(self.directory, 'table-1')
        events = self.trace()
        store.write(session.State(phase='ready'))
        self.assertEqual(['fsync-file', 'rename', 'fsync-directory'], events)
        self.assertEqual('ready', store.read().phase)

    def test_a_failed_file_synchronization_refuses_and_leaves_no_record(self):
        """The caller's next step is the side effect this record explains.

        So the write raises rather than returning, and what a reader sees afterwards is
        the record that was there before -- here, none at all, which `read` reports as
        a fresh `ready` state rather than as the 'preparing' the failed write carried.
        """
        store = session.StateStore(self.directory, 'table-1')
        self.trace(fail_on='file')
        with self.assertRaises(session.Refusal) as caught:
            store.write(session.State(phase='preparing'))
        self.assertIn('survives a restart', str(caught.exception))
        self.assertFalse(store.path.exists(),
                         'a record that was not synchronized was renamed into place')
        self.assertEqual('ready', store.read().phase)

    def test_a_failed_directory_synchronization_refuses(self):
        store = session.StateStore(self.directory, 'table-1')
        self.trace(fail_on='directory')
        with self.assertRaises(session.Refusal) as caught:
            store.write(session.State(phase='preparing'))
        self.assertIn('synchronized to storage', str(caught.exception))

    def test_an_einval_directory_synchronization_refuses_too(self):
        store = session.StateStore(self.directory, 'table-1')
        self.trace(fail_on='directory', error=errno.EINVAL)
        with self.assertRaises(session.Refusal):
            store.write(session.State(phase='preparing'))

    def test_an_ordinary_write_and_read_round_trip_still_works(self):
        """Positive control on the same call path."""
        store = session.StateStore(self.directory, 'table-1')
        store.write(session.State(phase='investigating', kind='efa', started_at=1.0,
                                  target_node='gpu-g7-1'))
        reread = store.read()
        self.assertEqual('investigating', reread.phase)
        self.assertEqual('efa', reread.kind)


class ControllerStopsWhenItsRecordIsNotDurable(_rework.Base):
    """The caller's side of the same repair, through the real controller.

    `StateStoreDurabilityOrdering` above establishes that `write` refuses; it does
    NOT establish that a caller stops. That distinction matters here because the
    whole premise is 'the record is durable BEFORE the operation it explains', and
    only the caller can satisfy it. So these drive the real `DeviceSession.start`
    with the synchronization failing, and assert on the privileged calls the fake
    executor recorded.
    """

    def fail_directory_sync(self, error=None):
        """Make the state record's directory synchronization fail.

        Only the directory half is injected, because that is the half a mid-run
        review found being swallowed, and because it fails AFTER the rename -- the
        hardest case for the caller, since the record is already visible to a reader
        when the failure arrives.
        """
        real_fsync = os.fsync

        def spy(fd):
            try:
                is_dir = os.fstat(fd).st_mode & 0o170000 == 0o040000
            except OSError:                                   # pragma: no cover
                is_dir = False
            if is_dir:
                raise OSError(error or errno.EIO, 'injected directory sync failure')
            return real_fsync(fd)

        os.fsync = spy
        self.addCleanup(setattr, os, 'fsync', real_fsync)

    def test_start_issues_no_privileged_operation_when_its_record_is_not_durable(self):
        """`start` writes its intent record before it drains or touches a device.

        If that write cannot be established as durable, none of what it explains may
        happen: no drain, no `gpu-prepare`, no device call. Otherwise a node can be
        changed while the record naming the change does not survive the machine,
        which is exactly the state the absence-based recovery gate misreads.
        """
        self.fail_directory_sync()
        with self.assertRaises(session.Refusal) as caught:
            self.make().start('gpu')
        self.assertIn('synchronized to storage', str(caught.exception))
        actions = self.executor.actions()
        for mutation in ('gpu-prepare', 'gpu-remove', 'efa-unbind'):
            self.assertNotIn(mutation, actions,
                             f'{mutation} ran for a round whose record was not '
                             f'established to be durable')
        drains = [call for call in self.executor.slurm_calls
                  if 'State=DRAIN' in call]
        self.assertEqual([], drains, 'the node was drained anyway')
        self.assertEqual([], self.executor.reboots())
        self.assertEqual([], self.executor.resumes())

    def test_an_einval_directory_sync_stops_the_caller_too(self):
        """EINVAL is the case the earlier version of this repair let through."""
        self.fail_directory_sync(errno.EINVAL)
        with self.assertRaises(session.Refusal):
            self.make().start('gpu')
        for mutation in ('gpu-prepare', 'gpu-remove', 'efa-unbind'):
            self.assertNotIn(mutation, self.executor.actions())
        self.assertEqual([], [call for call in self.executor.slurm_calls
                              if 'State=DRAIN' in call])

    def test_recovery_issues_no_mutation_when_its_record_is_not_durable(self):
        """The same on the recovery side, which writes before every mutation.

        The round is started normally first, so the failure is injected into a
        recovery attempt rather than into the start, and the assertion is that this
        attempt reboots and restores nothing.
        """
        self.make().start('gpu')
        before = list(self.executor.actions())
        self.fail_directory_sync()
        with self.assertRaises(session.Refusal):
            self.make().recover()
        new_actions = self.executor.actions()[len(before):]
        for mutation in ('remount-staging', 'restore-runtime', 'gpu-restore'):
            self.assertNotIn(mutation, new_actions,
                             f'{mutation} ran during an attempt whose record was '
                             f'not established to be durable')
        self.assertEqual([], self.executor.reboots())
        self.assertEqual([], self.executor.resumes())

    def test_an_ordinary_round_with_working_synchronization_still_starts(self):
        """Positive control: nothing injected, so the ordinary round proceeds."""
        result = self.make().start('gpu')
        self.assertEqual('fault-applied', result.state.phase)
        self.assertIn('gpu-prepare', self.executor.actions())
        self.assertIn('gpu-remove', self.executor.actions())


# The borrowed bench must not stay in this module's namespace: unittest's loader
# collects every TestCase subclass it finds, so leaving it here would run the whole
# rework-3 S4 suite a second time.
del _RealHelper


if __name__ == '__main__':                                     # pragma: no cover
    unittest.main()
