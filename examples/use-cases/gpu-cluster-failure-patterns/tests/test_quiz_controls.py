# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Quiz-style lab controls: the seat schedule, `next`, hidden kinds, split tables.

CPU-only, through the same recorded fake executor as the other controller suites.
A test here asserts what the controller decided, wrote and printed, never what a
device did; the hardware behaviour of each exercise is a separate measurement.
"""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import test_active_flr as active_fixture
import test_device_session as fixture

session = fixture.session
LAB = Path(__file__).resolve().parents[1]
HELPER = LAB / 'facilitator' / 'device-session.py'
MAINTENANCE = LAB / 'facilitator' / 'maintenance.py'
INSTALLER = LAB / 'facilitator' / 'install-participant-control.sh'
ENTRY = LAB / '12.device-exercise.sh'

SCHEDULE = [
    {'exercise': 'storage', 'settings': {'writer_rank': 9}},
    {'exercise': 'dataloader'},
    {'exercise': 'silent-corruption',
     'settings': {'corrupt_rank': 13, 'corrupt_step': 37, 'corrupt_bit': 22,
                  'corrupt_recurring': False}},
    {'exercise': 'efa-reset'},
]
# Words that name a hidden round's kind or the operation applied to it. Node names
# such as gpu-g7-1 are not kind words, so the pattern requires a word boundary
# that is not a hyphen.
KIND_WORDS = re.compile(r'(?i)(?<![\w-])(?:efa|gpu|flr|efa-reset|efa-flr)(?![\w-])')
REPLACE_INSTRUCTION = re.compile(
    r'(?i)\b(?:use|run|retry|repeat)(?:\s+participant)?\s+replace\b'
    r'|\bthen\s+replace\s+if\s+required\b|\bowner\s+must\s+repeat\s+replace\b'
    r'|\bor\s+use\s+replace\b')


class QuizBase(fixture.Base):
    """A seat with split check tables and a four-incident schedule."""

    # The active-FLR fake from the existing suite, reused as is: it plays the
    # target's authorization, mutation-start and mutation-returned records.
    active = active_fixture.ActiveSession.active
    progress = staticmethod(active_fixture.ActiveSession.progress)

    def setUp(self):
        super().setUp()
        table = self.config_body['assignments']['table-1']
        del table['allowed_checks']
        table['collect_checks'] = [0, 2, 3, 6]
        table['qualify_checks'] = {'gpu': [0, 3], 'efa': [2, 6]}
        table['exercise_schedule'] = session.validate_schedule(SCHEDULE)
        self.write_config()
        self.active_calls = []
        self.bad_identity = False
        self.rc = 0
        for kind in ('gpu', 'efa'):
            old = 'gpu-remove' if kind == 'gpu' else 'efa-unbind'
            self.executor.inspect['confirmation'][kind + '-flr'] = (
                self.executor.inspect['confirmation'][old].replace(old, kind + '-flr'))
        self.executor.run_active_fault = self.active

    def set_schedule(self, entries):
        self.config_body['assignments']['table-1']['exercise_schedule'] = entries
        self.write_config()

    def running_job_with_progress(self, now=1000.0):
        self.executor.jobs = ['226|aim344-t1|aim344-device|RUNNING']
        path = self.answers / 'device-226.log'
        path.write_text(self.progress(now))

    def advance_to(self, number):
        """Issue incidents until `number` is the current one."""
        result = None
        for _ in range(number):
            result = self.make().next()
        return result

    def run_main(self, command, now=1000.0, argv=None):
        """Drive main() as the forced command would, with the fake executor."""
        real_load, real_session = session.load_config, session.DeviceSession
        out, err = io.StringIO(), io.StringIO()
        executor = self.executor
        with mock.patch.object(session, 'load_config',
                               lambda path, **_: real_load(path, require_root_owned=False)), \
                mock.patch.object(session, 'DeviceSession',
                                  lambda config, assignment_id, peer_uid, **given: real_session(
                                      config, assignment_id, peer_uid,
                                      **dict({'executor': executor, 'clock': lambda: now},
                                             **given))), \
                mock.patch.dict(os.environ, {'SSH_ORIGINAL_COMMAND': command,
                                             'SUDO_UID': str(os.getuid())}), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = session.main(argv or ['--assignment', 'table-1',
                                         '--config', str(self.config_path)])
        return code, out.getvalue(), err.getvalue()

    def assert_no_kind(self, text):
        self.assertIsNone(KIND_WORDS.search(text), text)

    def node_calls(self):
        """Every scheduler mutation and every maintenance call made so far."""
        updates = [c for c in self.executor.slurm_calls
                   if Path(c[0]).name == 'scontrol' and c[1] in ('update', 'reboot')]
        return updates + self.executor.maintenance_calls


class Grammar(unittest.TestCase):
    def test_next_takes_no_argument(self):
        self.assertEqual('next', session.parse_participant_request('next').verb)
        for text in ('next 2', 'next efa-reset', 'next --incident 3', 'next;id'):
            with self.subTest(text=text):
                self.assertIsNotNone(session.parse_participant_request(text).error)

    def test_replace_still_parses_so_it_can_be_refused_with_a_reason(self):
        self.assertEqual('replace', session.parse_participant_request('replace').verb)

    def test_entry_point_passes_next_and_lists_no_replace(self):
        text = ENTRY.read_text()
        self.assertIn('next|status|recover|replace)', text)
        usage = text[text.index("<<'USAGE'"):text.index('\nUSAGE\n')]
        self.assertIn('next ', usage)
        self.assertNotRegex(usage, r'\n  replace ')


class ScheduleValidation(unittest.TestCase):
    def test_the_recommended_schedule_is_accepted(self):
        checked = session.validate_schedule(SCHEDULE)
        self.assertEqual(['storage', 'dataloader', 'silent-corruption', 'efa-reset'],
                         [entry['exercise'] for entry in checked])
        self.assertEqual({}, checked[1]['settings'])

    def test_every_listed_exercise_is_known(self):
        for name in session.JOB_EXERCISES + tuple(session.DEVICE_EXERCISES):
            with self.subTest(name=name):
                session.validate_schedule([{'exercise': name}])

    def test_unusable_schedules_are_refused(self):
        cases = {
            'empty': [],
            'not a list': {'exercise': 'storage'},
            'unknown exercise': [{'exercise': 'gpu-remove'}],
            'exercise not text': [{'exercise': ['storage']}],
            'extra field': [{'exercise': 'storage', 'answer': 'x'}],
            'settings not an object': [{'exercise': 'storage', 'settings': [1]}],
            'bad key': [{'exercise': 'storage', 'settings': {'Writer': 1}}],
            'shell text': [{'exercise': 'storage', 'settings': {'writer_rank': '9; id'}}],
            'nested value': [{'exercise': 'storage', 'settings': {'writer_rank': [9]}}],
            'not finite': [{'exercise': 'storage', 'settings': {'x': float('nan')}}],
            'too long': [{'exercise': 'storage'}] * (session.SCHEDULE_LIMIT + 1),
        }
        for name, entries in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(session.Refusal):
                    session.validate_schedule(entries)


class SplitCheckTables(QuizBase):
    def held_round(self, kind):
        self.make().store.write(session.State(phase='fault-applied', kind=kind,
                                              started_at=1.0, target_node='gpu-g7-1'))

    def test_collect_accepts_every_read_only_check_whatever_the_kind(self):
        for kind in ('gpu', 'efa'):
            with self.subTest(kind=kind):
                self.held_round(kind)
                for check in ('0', '2', '3', '6'):
                    self.make().collect(check)
        collected = [c[1] for c in self.executor.maintenance_calls if c[0] == 'collect']
        self.assertEqual(['0', '2', '3', '6'] * 2, collected)

    def test_a_refused_check_names_the_same_list_for_every_kind(self):
        messages = []
        for kind in ('gpu', 'efa'):
            self.held_round(kind)
            for check in ('1', '4', '5'):
                with self.assertRaises(session.Refusal) as caught:
                    self.make().collect(check)
                messages.append(str(caught.exception).replace(check, 'N', 1))
        self.assertEqual(1, len(set(messages)), messages)
        self.assertIn('Available: 0, 2, 3, 6.', messages[0])
        self.assertEqual([], self.executor.maintenance_calls)

    def test_recovery_still_qualifies_only_the_kinds_own_checks(self):
        """Widening collect must not make a GPU round depend on Check 6."""
        for kind, expected in (('gpu', ['0', '3']), ('efa', ['2', '6'])):
            with self.subTest(kind=kind):
                handler = self.make()
                state = session.State(phase='fault-applied', kind=kind, started_at=1.0,
                                      target_node='gpu-g7-1')
                self.assertEqual(expected, handler.qualify_checks(state))
                self.assertEqual(expected, handler.allowed_checks(state))
                self.assertEqual(['0', '2', '3', '6'], handler.collect_checks(state))

    def test_idle_rounds_record_a_baseline_for_every_collectable_check(self):
        """The pre-job round: the same baseline files whatever the kind."""
        names = {}
        for kind in ('efa', 'gpu'):
            self.executor.maintenance_calls.clear()
            self.make().start(kind)
            state = self.make().store.read()
            names[kind] = sorted(Path(record['log']).name for record in
                                 state.baseline_check_results.values())
            collected = [c[1] for c in self.executor.maintenance_calls if c[0] == 'collect']
            self.assertEqual(['0', '2', '3', '6'], collected)
            self.executor.jobs = []
            self.make().recover()
            recovered = [c[1] for c in self.executor.maintenance_calls
                         if c[0] == 'collect'][4:]
            self.assertEqual(self.make().qualify_checks(state), recovered)
        self.assertEqual(names['efa'], names['gpu'])
        self.assertEqual(['check-0-baseline.log', 'check-2-baseline.log',
                          'check-3-baseline.log', 'check-6-baseline.log'], names['efa'])

    def test_idle_efa_round_still_reaches_runtime_ready(self):
        """The pre-job check exercise: start efa, collect 2, recover, node back."""
        started = self.make().start('efa')
        self.assertEqual('fault-applied', started.state.phase)
        self.assertIn('DRAIN', self.executor.node_state)
        self.make().collect('2')
        result = self.make().recover()
        self.assertEqual('runtime-ready', result.state.phase)
        self.assertNotIn('DRAIN', self.executor.node_state)
        rendered = session.render(self.make().status(), 'status')
        self.assertIn('Exercise: efa', rendered)
        self.assertIn('EFA path: idle', rendered)

    def test_a_configuration_written_before_the_split_keeps_its_meaning(self):
        table = self.config_body['assignments']['table-1']
        for key in ('collect_checks', 'qualify_checks', 'exercise_schedule'):
            del table[key]
        table['allowed_checks'] = {'gpu': [0, 3], 'efa': [2, 6]}
        self.write_config()
        self.held_round('efa')
        with self.assertRaises(session.Refusal) as caught:
            self.make().collect('0')
        self.assertIn('Available: 2, 6.', str(caught.exception))

    def test_a_schedule_needs_the_split_tables(self):
        table = self.config_body['assignments']['table-1']
        del table['collect_checks']
        table['allowed_checks'] = table.pop('qualify_checks')
        self.write_config()
        with self.assertRaisesRegex(session.Refusal, 'collect and qualify'):
            self.make().next()

    def test_status_lists_the_collectable_checks_during_a_round(self):
        self.held_round('gpu')
        rendered = session.render(self.make().status(), 'status')
        self.assertIn('Checks you can collect: 0, 2, 3, 6, kernel', rendered)

    def test_installer_writes_the_split_tables_and_no_single_table(self):
        text = INSTALLER.read_text()
        self.assertIn("'collect_checks': [0, 2, 3, 6],", text)
        self.assertIn("'qualify_checks': {'gpu': [0, 3], 'efa': [2, 6]},", text)
        self.assertNotIn("'allowed_checks':", text)


class JobIncidents(QuizBase):
    def test_next_hands_the_launcher_the_first_incident_and_touches_nothing(self):
        result = self.make().next()
        self.assertEqual(1, result.incident)
        self.assertFalse(result.under_way)
        profile = json.loads(Path(result.saved_path).read_text())
        self.assertEqual({'incident': 1, 'incidents': 4, 'exercise': 'storage',
                          'settings': {'writer_rank': 9}, 'controller_step': 'none',
                          'issued_at': 1000.0}, profile)
        self.assertEqual(self.answers / 'incidents' / 'incident-1.json',
                         Path(result.saved_path))
        self.assertEqual([], self.node_calls())
        rendered = session.render(result, 'next')
        self.assertIn('Incident 1 is ready.', rendered)
        self.assertIn(f'Profile: {result.saved_path}', rendered)
        self.assertNotIn('storage', rendered)

    def test_the_schedule_and_later_profiles_stay_in_root_only_records(self):
        self.make().next()
        written = sorted(p.name for p in (self.answers / 'incidents').iterdir())
        self.assertEqual(['incident-1.json'], written)
        progress = self.state_dir / 'schedule-table-1.json'
        self.assertEqual(0, stat.S_IMODE(progress.stat().st_mode) & 0o077)
        self.assertEqual([1], [e['number'] for e in
                               json.loads(progress.read_text())['issued']])
        for path in self.answers.rglob('*'):
            if path.is_file():
                self.assertNotIn('silent-corruption', path.read_text())
                self.assertNotIn('efa-reset', path.read_text())

    def test_each_next_moves_one_incident_on_in_the_seats_order(self):
        exercises = []
        for _ in range(3):
            result = self.make().next()
            exercises.append(json.loads(Path(result.saved_path).read_text())['exercise'])
        self.assertEqual(['storage', 'dataloader', 'silent-corruption'], exercises)
        third = json.loads((self.answers / 'incidents' / 'incident-3.json').read_text())
        self.assertEqual({'corrupt_rank': 13, 'corrupt_step': 37, 'corrupt_bit': 22,
                          'corrupt_recurring': False}, third['settings'])
        self.assertEqual([], self.node_calls())

    def test_the_answers_differ_by_seat_through_settings_not_order(self):
        self.set_schedule(session.validate_schedule(
            [{'exercise': 'storage', 'settings': {'writer_rank': 0}}]))
        result = self.make().next()
        self.assertEqual({'writer_rank': 0},
                         json.loads(Path(result.saved_path).read_text())['settings'])

    def test_an_exhausted_schedule_says_so(self):
        self.set_schedule(session.validate_schedule([{'exercise': 'slow-job'}]))
        self.make().next()
        with self.assertRaisesRegex(session.Refusal, 'no next incident'):
            self.make().next()

    def test_a_seat_without_a_schedule_is_told_to_ask(self):
        del self.config_body['assignments']['table-1']['exercise_schedule']
        self.write_config()
        with self.assertRaisesRegex(session.Refusal, 'no incident schedule'):
            self.make().next()

    def test_a_schedule_changed_after_incidents_were_issued_is_refused(self):
        self.make().next()
        self.set_schedule(session.validate_schedule([{'exercise': 'dataloader'},
                                                     {'exercise': 'storage'}]))
        with self.assertRaisesRegex(session.Refusal, 'does not match its schedule'):
            self.make().next()

    def test_a_failed_profile_write_is_finished_rather_than_skipped(self):
        os.chmod(self.answers, 0o500)
        try:
            with self.assertRaises(session.Refusal):
                self.make().next()
        finally:
            os.chmod(self.answers, 0o700)
        progress = json.loads((self.state_dir / 'schedule-table-1.json').read_text())
        self.assertEqual([None], [e['profile'] for e in progress['issued']])
        result = self.make().next()
        self.assertEqual(1, result.incident)
        self.assertEqual('storage', json.loads(Path(result.saved_path).read_text())['exercise'])
        self.assertEqual(2, self.make().next().incident)

    def test_status_counts_incidents_without_naming_them(self):
        self.advance_to(2)
        rendered = session.render(self.make().status(), 'status')
        self.assertIn('Incidents issued: 2 of 4', rendered)
        self.assertIn('Start the next incident the way your page shows.', rendered)
        for name in ('storage', 'dataloader', 'start gpu'):
            self.assertNotIn(name, rendered)

    def test_collect_outside_a_controller_round_is_still_refused(self):
        self.make().next()
        with self.assertRaisesRegex(session.Refusal, 'no exercise running'):
            self.make().collect('0')
        self.assertEqual([], self.executor.maintenance_calls)


class DeviceIncident(QuizBase):
    def to_device_incident(self):
        result = self.advance_to(4)
        self.assertEqual(4, result.incident)
        return result

    def test_its_profile_looks_like_any_other_until_the_job_reports_progress(self):
        issued = self.to_device_incident()
        profile = json.loads(Path(issued.saved_path).read_text())
        self.assertEqual('next-after-progress', profile['controller_step'])
        # Asking again before the job runs repeats the answer and changes nothing.
        self.assertEqual(session.render(issued, 'next'),
                         session.render(self.make().next(), 'next'))
        # A running allocation alone is not enough: no progress log yet.
        self.executor.jobs = ['226|aim344-t1|aim344-device|RUNNING']
        repeat = self.make().next()
        self.assertFalse(repeat.under_way)
        self.assertEqual(issued.saved_path, repeat.saved_path)
        self.assertEqual([], self.active_calls)
        self.assertEqual([], self.node_calls())
        self.assertEqual(['incident-1.json', 'incident-2.json', 'incident-3.json',
                          'incident-4.json'],
                         sorted(p.name for p in (self.answers / 'incidents').iterdir()))

    def test_stale_progress_does_not_start_it(self):
        self.to_device_incident()
        self.running_job_with_progress(now=900.0)
        self.assertFalse(self.make().next().under_way)
        self.assertEqual([], self.active_calls)

    def test_the_rendered_reply_is_the_same_for_a_job_and_the_device_incident(self):
        first = session.render(self.make().next(), 'next')
        fourth = session.render(self.advance_to(3), 'next')
        self.assertEqual(first.replace('incident-1', 'incident-X').replace('Incident 1', 'Incident X'),
                         fourth.replace('incident-4', 'incident-X').replace('Incident 4', 'Incident X'))

    def test_progress_starts_it_through_start_with_every_guard(self):
        self.to_device_incident()
        self.running_job_with_progress()
        result = self.make().next()
        self.assertTrue(result.under_way)
        self.assertEqual(['efa-flr'], [call[0] for call in self.active_calls])
        state = self.make().store.read()
        self.assertEqual({'number': 4, 'exercise': 'efa-reset'}, state.incident)
        self.assertEqual('efa', state.kind)
        self.assertEqual('fault-applied', state.phase)
        self.assertIn('drain', state.prepared)
        self.assertIn('efa-flr-attempted', state.prepared)
        drains = [c for c in self.executor.slurm_calls if 'State=DRAIN' in c]
        self.assertEqual(1, len(drains))
        rendered = session.render(result, 'next')
        self.assertEqual('Incident 4 is under way.', rendered.splitlines()[0])
        self.assert_no_kind(rendered)

    def test_its_transcript_is_kept_for_the_operator_not_the_participant(self):
        self.to_device_incident()
        self.running_job_with_progress()
        self.make().next()
        participant_files = [p.name for p in self.answers.rglob('*') if p.is_file()]
        self.assertFalse(any('flr' in name or 'efa' in name for name in participant_files),
                         participant_files)
        state = self.make().store.read()
        record = Path(state.active_fault['operator_log'])
        self.assertEqual(self.state_dir, record.parent)
        self.assertIn('efa-flr', record.read_text())
        self.assertEqual(0, stat.S_IMODE(record.stat().st_mode) & 0o077)

    def test_status_and_collect_do_not_name_the_kind_while_it_is_unresolved(self):
        self.to_device_incident()
        self.running_job_with_progress()
        self.make().next()
        rendered = session.render(self.make().status(), 'status')
        self.assertIn('Incident: 4', rendered)
        self.assertIn('Checks you can collect: 0, 2, 3, 6, kernel', rendered)
        for absent in ('Exercise:', 'Active FLR', 'EFA path', 'Answer for'):
            self.assertNotIn(absent, rendered)
        self.assert_no_kind(rendered)
        for check in ('0', '3', 'kernel'):
            collected = self.make().collect(check)
            tail = session.render(collected, 'collect').split(collected.output.rstrip('\n'))[-1]
            self.assert_no_kind(tail)
            self.assertNotIn('flr', str(collected.saved_path))

    def test_a_guess_with_a_legacy_verb_gets_the_same_neutral_refusal(self):
        self.to_device_incident()
        self.running_job_with_progress()
        self.make().next()
        replies = set()
        for kind in ('gpu', 'efa'):
            for active in (False, True):
                with self.assertRaises(session.Refusal) as caught:
                    self.make().start(kind, active=active)
                replies.add(str(caught.exception))
        self.assertEqual({'Incident 4 is still in progress on your node. Finish it with '
                          'recover before starting anything else.'}, replies)
        self.assertEqual(1, len(self.active_calls))

    def test_next_during_the_held_round_changes_nothing(self):
        self.to_device_incident()
        self.running_job_with_progress()
        self.make().next()
        before = len(self.node_calls())
        with self.assertRaisesRegex(session.Refusal, 'Incident 4 still holds your node'):
            self.make().next()
        self.assertEqual(before, len(self.node_calls()))

    def test_recovery_reveals_the_answer_and_the_schedule_moves_on(self):
        self.set_schedule(session.validate_schedule([{'exercise': 'efa-reset'},
                                                     {'exercise': 'storage'}]))
        self.make().next()
        self.running_job_with_progress()
        self.assertTrue(self.make().next().under_way)
        self.executor.jobs = []
        recovered = self.make().recover()
        self.assertEqual('runtime-ready', recovered.state.phase)
        self.assertTrue(any(c[1] == 'reboot' for c in self.executor.slurm_calls))
        rendered = session.render(self.make().status(), 'status')
        self.assertIn('Answer for incident 1: the lab controller reset one EFA device on '
                      'your exercise node (a PCIe function-level reset of 0000:b0:00.0)',
                      rendered)
        following = self.make().next()
        self.assertEqual(2, following.incident)
        self.assertFalse(following.under_way)

    def test_a_refused_start_is_concealed_and_kept_for_the_operator(self):
        self.to_device_incident()
        self.running_job_with_progress()
        del self.executor.inspect['confirmation']['efa-flr']
        with self.assertRaises(session.Refusal) as caught:
            self.make().next()
        self.assert_no_kind(str(caught.exception))
        progress = json.loads((self.state_dir / 'schedule-table-1.json').read_text())
        self.assertIn('confirmation', progress['issued'][3]['last_refusal']['text'])
        self.assertEqual('ready', self.make().store.read().phase)
        view = session.operator_view(self.make())
        self.assertIn('last refusal:', view)

    def test_a_foreign_job_on_the_target_stops_next_before_anything_changes(self):
        self.to_device_incident()
        self.executor.jobs = ['301|someone-else|aim344-x|RUNNING']
        with self.assertRaisesRegex(session.Refusal, 'unapproved job'):
            self.make().next()
        self.assertEqual([], self.active_calls)
        self.assertFalse(any('State=DRAIN' in c for c in self.executor.slurm_calls))


class WarnRefusalAndOperatorReplacement(QuizBase):
    """Decisions 3 and 4: any WARN keeps the node out, an operator replaces it."""

    def fire_and_warn(self):
        self.set_schedule(session.validate_schedule([{'exercise': 'efa-reset'},
                                                     {'exercise': 'storage'}]))
        self.make().next()
        self.running_job_with_progress()
        self.make().next()
        self.executor.jobs = []
        outputs = {'2': '[PASS] 2-efa-enumeration: EFA OK\n',
                   '6': '[WARN] EFA retransmission timeouts detected (2)\n'
                        '[WARN] 6-efa-loopback: EFA statistics show drops or '
                        'retransmission timeouts (see logs)\n'}
        original = self.executor.run_maintenance

        def per_check(argv, timeout=None):
            if argv[0] == 'collect':
                self.executor.maintenance_calls.append(list(argv))
                return session.Completed(0, outputs[argv[1]], '')
            return original(argv, timeout)
        self.executor.run_maintenance = per_check

    def test_a_warn_after_recovery_keeps_the_node_out_and_says_who_replaces_it(self):
        self.fire_and_warn()
        code, out, err = self.run_main('recover')
        self.assertEqual(3, code)
        self.assertIn(session.OPERATOR_REPLACEMENT, err)
        self.assertIsNone(REPLACE_INSTRUCTION.search(err), err)
        state = self.make().store.read()
        self.assertEqual('replacement-required', state.phase)
        self.assertEqual([], [c for c in self.executor.slurm_calls if 'State=RESUME' in c])
        self.assertIn('DRAIN', self.executor.node_state)

    def test_status_then_reveals_the_answer_and_the_operator_step(self):
        self.fire_and_warn()
        self.run_main('recover')
        code, out, _ = self.run_main('status')
        self.assertEqual(0, code)
        self.assertIn('Answer for incident 1:', out)
        self.assertIn(f'Next: {session.OPERATOR_REPLACEMENT}', out)
        self.assertIsNone(REPLACE_INSTRUCTION.search(out), out)

    def test_next_says_the_rest_cannot_run_live_on_that_node(self):
        self.fire_and_warn()
        self.run_main('recover')
        code, _, err = self.run_main('next')
        self.assertEqual(3, code)
        self.assertIn('an operator replaces it after the session', err)
        self.assertIn('expected output', err)

    def test_the_participant_replace_verb_changes_nothing(self):
        self.fire_and_warn()
        self.run_main('recover')
        before = (len(self.executor.slurm_calls), len(self.executor.maintenance_calls))
        code, out, err = self.run_main('replace')
        self.assertEqual(3, code)
        self.assertEqual(session.PARTICIPANT_REPLACE_REFUSAL + '\n', err)
        self.assertEqual('', out)
        self.assertEqual(before, (len(self.executor.slurm_calls),
                                  len(self.executor.maintenance_calls)))
        self.assertEqual('replacement-required', self.make().store.read().phase)


class ParticipantOutputBoundary(QuizBase):
    def test_next_and_status_through_main_name_no_kind(self):
        for _ in range(3):
            code, out, err = self.run_main('next')
            self.assertEqual(0, code, err)
            self.assertNotIn('storage', out)
            self.assertNotIn('dataloader', out)
            self.assertNotIn('silent', out)
        code, out, _ = self.run_main('next')
        self.assertIn('Incident 4 is ready.', out)
        self.running_job_with_progress()
        code, out, err = self.run_main('next')
        self.assertEqual(0, code, err)
        self.assertIn('Incident 4 is under way.', out)
        self.assert_no_kind(out)
        code, out, _ = self.run_main('status')
        self.assert_no_kind(out)

    def fire(self):
        self.advance_to(4)
        self.running_job_with_progress()
        self.assertTrue(self.make().next().under_way)

    def test_recover_refusals_name_no_kind_while_the_round_is_unresolved(self):
        self.fire()
        # The faulted allocation is still running: recovery refuses and says why.
        code, _, err = self.run_main('recover')
        self.assertEqual(3, code)
        self.assertIn('still active', err)
        self.assert_no_kind(err)

    def test_a_refusal_that_names_the_operation_is_concealed(self):
        self.fire()
        handler = self.make()
        state = handler.store.read()
        state.job_id = None
        handler.store.write(state)
        self.executor.jobs = []
        code, _, err = self.run_main('recover')
        self.assertEqual(3, code)
        self.assertIn('The incident restoration provenance is incomplete.', err)
        self.assert_no_kind(err)
        self.assertIsNone(REPLACE_INSTRUCTION.search(err), err)
        self.assertIn('Failure: Active FLR restoration provenance is incomplete',
                      session.operator_view(self.make()))

    def test_operator_flags_are_root_only_and_exclusive(self):
        for argv in (['--operator', 'status', '--assignment', 'table-1',
                      '--config', str(self.config_path)],
                     ['--operator', 'replace', '--assignment', 'table-1',
                      '--config', str(self.config_path)]):
            with self.subTest(argv=argv):
                code, _, err = self.run_main('', argv=argv)
                self.assertEqual(3, code)
                self.assertIn('root', err)
        code, _, err = self.run_main('', argv=['--operator', 'status', '--sweep',
                                               '--config', str(self.config_path)])
        self.assertEqual(3, code)
        self.assertIn('Use one of', err)


class OperatorView(QuizBase):
    def test_the_operator_sees_the_kind_schedule_and_settings(self):
        self.advance_to(4)
        self.running_job_with_progress()
        self.make().next()
        view = session.operator_view(self.make())
        self.assertIn('Exercise: efa', view)
        self.assertIn('Scheduled exercise: efa-reset', view)
        self.assertIn('Active FLR: efa-flr', view)
        self.assertIn('transcript: ', view)
        self.assertIn('Schedule (4 of 4 issued):', view)
        self.assertIn('1. storage {"writer_rank": 9} issued 1970-01-01T00:16:40Z', view)
        self.assertIn('3. silent-corruption {"corrupt_bit": 22', view)
        self.assertIn('4. efa-reset {} issued', view)
        self.assertIn('round=1 phase=fault-applied', view)

    def test_unissued_incidents_are_listed_for_the_operator_only(self):
        self.make().next()
        view = session.operator_view(self.make())
        self.assertIn('2. dataloader {} not issued', view)
        self.assertIn('4. efa-reset {} not issued', view)
        participant = session.render(self.make().status(), 'status')
        self.assertNotIn('dataloader', participant)

    def test_a_seat_without_a_schedule_is_shown_as_such(self):
        del self.config_body['assignments']['table-1']['exercise_schedule']
        self.write_config()
        self.assertIn('Schedule: none', session.operator_view(self.make()))


class ParticipantText(unittest.TestCase):
    @staticmethod
    def refusal_texts():
        """Every Refusal literal in the controller and the target helper."""
        found = []
        for path in (HELPER, MAINTENANCE):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not node.args:
                    continue
                name = getattr(node.func, 'id', None) or getattr(node.func, 'attr', None)
                if name != 'Refusal':
                    continue
                pieces = [n.value for n in ast.walk(node.args[0])
                          if isinstance(n, ast.Constant) and isinstance(n.value, str)]
                found.append((path.name, node.lineno, ''.join(pieces)))
        return found

    def test_no_participant_text_tells_them_to_run_replace(self):
        cases = [item for item in self.refusal_texts()
                 if REPLACE_INSTRUCTION.search(item[2])]
        self.assertGreater(len(cases), 20)
        for name, line, text in cases:
            with self.subTest(source=f'{name}:{line}'):
                translated = session.participant_replacement_text(text)
                self.assertIsNone(REPLACE_INSTRUCTION.search(translated), translated)
                self.assertIn('operator', translated)

    def test_retryable_messages_keep_their_recover_step(self):
        text = ('Your node was not fully restored: x It stays out of service, with your '
                'evidence kept. Run recover again, or use replace to continue on a fresh '
                'assigned instance.')
        translated = session.participant_replacement_text(text)
        self.assertIn('Run recover again.', translated)
        self.assertIn(session.OPERATOR_REPLACEMENT_NOTE, translated)
        self.assertNotIn('rest of this session', translated)

    def test_terminal_messages_state_the_operator_replacement(self):
        text = ('reason The node stays isolated. Replacement is required; no fallback '
                'reboot or instructor restoration is allowed. Run replace to continue on '
                'a fresh assigned instance.')
        translated = session.participant_replacement_text(text)
        self.assertTrue(translated.endswith(session.OPERATOR_REPLACEMENT), translated)

    def test_text_without_replace_is_unchanged(self):
        text = 'Your exercise node has not returned yet. It stays reserved; run recover again.'
        self.assertEqual(text, session.participant_replacement_text(text))

    def test_concealment_keeps_node_names_and_check_numbers(self):
        self.assertEqual('The incident did not confirm completion on gpu-g7-1; Check 2 FAIL.',
                         session.conceal_kind('Active FLR did not confirm completion on '
                                              'gpu-g7-1; Check 2 FAIL.'))
        self.assertEqual('a cumulative device counter and device operation',
                         session.conceal_kind('a cumulative EFA counter and efa-flr-attempted'))


class InstallerSchedule(unittest.TestCase):
    """The installer's configuration step, run on its own as an ordinary user."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = self.root / 'device-session.json'
        self.state_dir = self.root / 'state'
        self.state_dir.mkdir(mode=0o700)
        text = INSTALLER.read_text()
        start = text.index('python3 - "$session_config"')
        body = text.index("<<'PY'\n", start) + len("<<'PY'\n")
        self.block = text[body:text.index('\nPY\n', body)]
        self.user = subprocess.run(['id', '-un'], capture_output=True, text=True,
                                   check=True).stdout.strip()

    def tearDown(self):
        self.tmp.cleanup()

    def install(self, schedule_file=None):
        env = dict(os.environ, AIM344_PARTITION='queue', AIM344_CONTROL_USER=self.user)
        env.pop('AIM344_SCHEDULE_FILE', None)
        if schedule_file:
            env['AIM344_SCHEDULE_FILE'] = str(schedule_file)
        return subprocess.run(
            [sys.executable, '-', str(self.config), 'table-1', self.user, 'gpu-g7-1',
             'i-0123456789abcdef0', '10.0.0.1', str(self.state_dir), '/opt/slurm/bin',
             '/opt/suite/gpu-healthcheck.sh', str(HELPER)],
            input=self.block, capture_output=True, text=True, env=env)

    def schedule_file(self, entries, mode=0o600):
        path = self.root / 'seat-1.json'
        path.write_text(json.dumps(entries))
        os.chmod(path, mode)
        return path

    def assignment(self):
        return json.loads(self.config.read_text())['assignments']['table-1']

    def test_the_schedule_is_checked_and_copied_and_the_record_reset(self):
        progress = self.state_dir / 'schedule-table-1.json'
        progress.write_text('{"issued": []}')
        done = self.install(self.schedule_file(SCHEDULE))
        self.assertEqual(0, done.returncode, done.stderr)
        table = self.assignment()
        self.assertEqual(session.validate_schedule(SCHEDULE), table['exercise_schedule'])
        self.assertEqual([0, 2, 3, 6], table['collect_checks'])
        self.assertEqual({'gpu': [0, 3], 'efa': [2, 6]}, table['qualify_checks'])
        self.assertNotIn('allowed_checks', table)
        self.assertFalse(progress.exists())
        self.assertIn('4-incident schedule', done.stdout)
        self.assertEqual(0o600, stat.S_IMODE(self.config.stat().st_mode))

    def test_a_rerun_without_a_file_keeps_the_schedule_and_its_record(self):
        self.install(self.schedule_file(SCHEDULE))
        progress = self.state_dir / 'schedule-table-1.json'
        progress.write_text('{"issued": []}')
        done = self.install()
        self.assertEqual(0, done.returncode, done.stderr)
        self.assertEqual(4, len(self.assignment()['exercise_schedule']))
        self.assertTrue(progress.exists())

    def test_a_readable_or_unknown_schedule_is_refused(self):
        done = self.install(self.schedule_file(SCHEDULE, mode=0o644))
        self.assertNotEqual(0, done.returncode)
        self.assertIn('chmod 600', done.stderr)
        done = self.install(self.schedule_file([{'exercise': 'nvlink'}]))
        self.assertNotEqual(0, done.returncode)
        self.assertIn('unknown exercise', done.stderr)
        link = self.root / 'seat-link.json'
        link.symlink_to(self.schedule_file(SCHEDULE))
        done = self.install(link)
        self.assertNotEqual(0, done.returncode)
        self.assertFalse(self.config.exists())


if __name__ == '__main__':
    unittest.main()
