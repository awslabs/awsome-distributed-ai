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
# that is not a hyphen. The idle round unbinds and rebinds, the active round is a
# function-level reset, and only EFA is an RDMA device.
KIND_WORDS = re.compile(
    r'(?i)(?<![\w-])(?:efa|gpu|flr|efa-reset|efa-flr|unbind\w*|rebind\w*|rdma'
    r'|function-level|(?:efa|gpu)-(?:flr|remove|unbind|rebind)[\w-]*)(?![\w-])')
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

    def progress_record(self):
        return json.loads((self.state_dir / 'schedule-table-1.json').read_text())

    def issue_id(self, number):
        """The issue id the controller recorded for incident `number`."""
        return self.progress_record()['issued'][number - 1]['issue_id']

    def advance_to(self, number):
        """Issue incidents until `number` is the current one."""
        result = None
        for _ in range(number):
            result = self.make().next()
        return result

    def run_main(self, command, now=1000.0, argv=None):
        """Drive main() as 12.device-exercise.sh does through sudo, with the fake
        executor."""
        real_session = session.DeviceSession
        out, err = io.StringIO(), io.StringIO()
        executor = self.executor
        with mock.patch.object(session, 'DeviceSession',
                                  lambda config, assignment_id, **given: real_session(
                                      config, assignment_id,
                                      **dict({'executor': executor, 'clock': lambda: now},
                                             **given))), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = session.main(argv or ['--config', str(self.config_path), *command.split()])
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

    def test_ended_takes_one_issue_id(self):
        issue = 'a' * 32
        request = session.parse_participant_request(f'ended {issue}')
        self.assertEqual(('ended', issue), (request.verb, request.argument))
        for text in ('ended', 'ended 2', 'ended --incident 3', f'ended {issue} {issue}',
                     'ended ' + 'A' * 32, 'ended ' + 'a' * 31, 'ended ' + 'g' * 32):
            with self.subTest(text=text):
                self.assertIsNotNone(session.parse_participant_request(text).error)

    def test_replace_still_parses_so_it_can_be_refused_with_a_reason(self):
        self.assertEqual('replace', session.parse_participant_request('replace').verb)

    def test_entry_point_passes_next_and_lists_no_replace(self):
        text = ENTRY.read_text()
        self.assertIn('next|status|recover|replace)', text)
        self.assertIn('[[ $2 =~ ^[0-9a-f]{32}$ ]] || usage\n        request=(ended "$2")', text)
        usage = text[text.index("<<'USAGE'"):text.index('\nUSAGE\n')]
        self.assertIn('next ', usage)
        self.assertNotRegex(usage, r'\n  replace ')

    def test_entry_point_runs_the_controller_through_sudo_with_the_request_words(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / 'sudo'
            fake.write_text('#!/bin/sh\nprintf "%s|" "$@"\n')
            fake.chmod(0o755)
            env = dict(os.environ, PATH=f'{tmp}:{os.environ["PATH"]}')
            issue = 'a' * 32
            for words, expected in ((['collect', '0'], 'collect|0|'),
                                    (['collect', 'kernel'], 'collect|kernel|'),
                                    (['ended', issue], f'ended|{issue}|'),
                                    (['active', 'efa'], 'active|efa|'),
                                    (['status'], 'status|')):
                with self.subTest(words=words):
                    run = subprocess.run(['bash', str(ENTRY), *words], capture_output=True,
                                         text=True, env=env, timeout=20)
                    self.assertEqual(0, run.returncode, run.stderr)
                    self.assertEqual('-n|/usr/local/sbin/aim344-device-session|' + expected,
                                     run.stdout)
            refused = subprocess.run(['bash', str(ENTRY), 'collect', '0;id'], capture_output=True,
                                     text=True, env=env, timeout=20)
            self.assertEqual(2, refused.returncode)
            self.assertEqual('', refused.stdout)


class ScheduleValidation(unittest.TestCase):
    def test_application_step_must_follow_checkpoint_200(self):
        for step in (1, 200):
            with self.subTest(step=step), self.assertRaises(session.Refusal):
                session.validate_schedule([{'exercise': 'cuda-assert',
                    'settings': {'assert_rank': 0, 'assert_step': step}}])
        self.assertEqual(201, session.validate_schedule([{'exercise': 'cuda-assert',
            'settings': {'assert_rank': 0, 'assert_step': 201}}])[0]['settings']['assert_step'])


    def test_the_recommended_schedule_is_accepted(self):
        checked = session.validate_schedule(SCHEDULE)
        self.assertEqual(['storage', 'dataloader', 'silent-corruption', 'efa-reset'],
                         [entry['exercise'] for entry in checked])
        self.assertEqual({}, checked[1]['settings'])

    def test_every_listed_exercise_is_known(self):
        self.assertEqual(set(session.JOB_EXERCISES) | set(session.DEVICE_EXERCISES),
                         set(session.EXERCISE_SETTINGS))
        for name in session.JOB_EXERCISES + tuple(session.DEVICE_EXERCISES):
            with self.subTest(name=name):
                # The launcher needs the rank whose answer differs.
                settings = {'corrupt_rank': 0} if name == 'silent-corruption' else {}
                if name == 'cuda-assert':
                    settings = {'assert_rank': 0}
                session.validate_schedule([{'exercise': name, 'settings': settings}])

    def test_the_application_incident_takes_a_rank_and_a_step(self):
        self.assertIn('cuda-assert', session.JOB_EXERCISES)
        self.assertIn('cuda-assert', session.APPLICATION_EXERCISES)
        self.assertNotIn('cuda-assert', session.DEVICE_EXERCISES)
        accepted = [{'exercise': 'cuda-assert', 'settings': {'assert_rank': 15, 'assert_step': 201}},
                    {'exercise': 'cuda-assert', 'settings': {'assert_rank': 0}}]
        self.assertEqual(accepted, session.validate_schedule(accepted, world_size=16))
        refused = {
            'no rank': ({'exercise': 'cuda-assert', 'settings': {'assert_step': 201}},
                        'needs the setting assert_rank'),
            'rank past the job': ({'exercise': 'cuda-assert', 'settings': {'assert_rank': 16}},
                                  'from 0 to 15'),
            'rank as text': ({'exercise': 'cuda-assert', 'settings': {'assert_rank': '3'}}, 'a rank'),
            'step 0': ({'exercise': 'cuda-assert', 'settings': {'assert_rank': 1, 'assert_step': 0}},
                       'from 1 to 4095'),
            'another setting': ({'exercise': 'cuda-assert',
                                 'settings': {'assert_rank': 1, 'corrupt_bit': 3}},
                                'it takes assert_rank, assert_step'),
        }
        for name, (entry, expected) in refused.items():
            with self.subTest(case=name):
                with self.assertRaisesRegex(session.Refusal, re.escape(expected)):
                    session.validate_schedule([entry], world_size=16)

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
            'an option as text': [{'exercise': 'storage', 'settings': {'mode': '--resume'}}],
            'a leading dash': [{'exercise': 'storage', 'settings': {'mode': '-x'}}],
            'text too long': [{'exercise': 'storage', 'settings': {'mode': 'a' * 129}}],
            'nested value': [{'exercise': 'storage', 'settings': {'writer_rank': [9]}}],
            'not finite': [{'exercise': 'storage', 'settings': {'x': float('nan')}}],
            'too long': [{'exercise': 'storage'}] * (session.SCHEDULE_LIMIT + 1),
        }
        for name, entries in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(session.Refusal):
                    session.validate_schedule(entries)

    def test_each_exercise_takes_only_the_settings_its_launcher_applies(self):
        accepted = [
            {'exercise': 'storage', 'settings': {'writer_rank': 15}},
            {'exercise': 'dataloader', 'settings': {'dataset_cache': 'worker-thread',
                                                     'affected_rank': 0, 'worker_cuda': True}},
            {'exercise': 'silent-corruption',
             'settings': {'corrupt_rank': 15, 'corrupt_step': 1, 'corrupt_bit': 31,
                          'corrupt_recurring': True}},
        ]
        self.assertEqual(accepted, session.validate_schedule(accepted, world_size=16))
        refused = {
            'a typo': ({'exercise': 'storage', 'settings': {'writer-rank': 9}},
                       'unusable setting name'),
            'another exercise\'s setting': ({'exercise': 'storage', 'settings': {'corrupt_rank': 1}},
                                            'has no setting corrupt_rank; it takes writer_rank'),
            'a setting where none applies': ({'exercise': 'slow-job', 'settings': {'mode': 'x'}},
                                             'it takes no settings'),
            'writer past the last rank': ({'exercise': 'storage', 'settings': {'writer_rank': 16}},
                                          'a rank of the 16-rank job, from 0 to 15'),
            'flip past the last rank': ({'exercise': 'silent-corruption',
                                         'settings': {'corrupt_rank': 16}}, 'from 0 to 15'),
            'loader rank past the last rank': ({'exercise': 'dataloader',
                                                'settings': {'affected_rank': 20}}, 'from 0 to 15'),
            'a rank as text': ({'exercise': 'storage', 'settings': {'writer_rank': '9'}}, 'a rank'),
            'a flag as a number': ({'exercise': 'dataloader', 'settings': {'worker_cuda': 1}},
                                   'true or false'),
            'an unknown cache': ({'exercise': 'dataloader', 'settings': {'dataset_cache': 'a-b'}},
                                 'one of none, rank-thread, worker-thread'),
            'step 0': ({'exercise': 'silent-corruption',
                        'settings': {'corrupt_rank': 1, 'corrupt_step': 0}}, 'from 1 to 4095'),
            # job.sh read_profile accepts a step below 4096 only.
            'step 4096': ({'exercise': 'silent-corruption',
                           'settings': {'corrupt_rank': 1, 'corrupt_step': 4096}}, 'from 1 to 4095'),
            # job.sh read_profile refuses a rank or worker CUDA without a cached dataset.
            'no cache with a rank': ({'exercise': 'dataloader',
                                      'settings': {'dataset_cache': 'none', 'affected_rank': 3}},
                                     'takes neither affected_rank nor worker_cuda true'),
            'no cache with worker CUDA': ({'exercise': 'dataloader',
                                           'settings': {'dataset_cache': 'none', 'worker_cuda': True}},
                                          'takes neither affected_rank nor worker_cuda true'),
            'bit 32': ({'exercise': 'silent-corruption',
                        'settings': {'corrupt_rank': 1, 'corrupt_bit': 32}}, 'from 0 to 31'),
            'no flip rank': ({'exercise': 'silent-corruption', 'settings': {'corrupt_step': 5}},
                             'needs the setting corrupt_rank'),
        }
        for name, (entry, expected) in refused.items():
            with self.subTest(case=name):
                with self.assertRaisesRegex(session.Refusal, re.escape(expected)):
                    session.validate_schedule([entry], world_size=16)

    def test_the_launchers_accepted_edges_are_accepted(self):
        for entry in ({'exercise': 'silent-corruption',
                       'settings': {'corrupt_rank': 1, 'corrupt_step': 4095}},
                      {'exercise': 'dataloader',
                       'settings': {'dataset_cache': 'none', 'worker_cuda': False}}):
            with self.subTest(entry=entry):
                self.assertEqual([entry], session.validate_schedule([entry], world_size=16))

    def test_the_rank_bound_is_the_installers_world_size(self):
        entry = [{'exercise': 'storage', 'settings': {'writer_rank': 20}}]
        self.assertEqual(entry, session.validate_schedule(entry, world_size=32))
        # The controller checks again without a world size; the installer bounded it.
        self.assertEqual(entry, session.validate_schedule(entry))
        with self.assertRaises(session.Refusal):
            session.validate_schedule([{'exercise': 'storage', 'settings': {'writer_rank': -1}}])


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

    def test_a_seat_with_no_qualification_checks_starts_nothing(self):
        """F3: no qualify table, or an empty list, is a refusal, not zero checks."""
        table = self.config_body['assignments']['table-1']
        for qualify in (None, {'gpu': [0, 3], 'efa': []}):
            with self.subTest(qualify=qualify):
                if qualify is None:
                    table.pop('qualify_checks', None)
                else:
                    table['qualify_checks'] = qualify
                self.write_config()
                with self.assertRaisesRegex(session.Refusal, 'names no checks'):
                    self.make().start('efa')
                self.assertEqual([], self.node_calls())
                self.assertEqual('ready', self.make().store.read().phase)

    def test_recovery_with_no_qualification_checks_keeps_the_node_drained(self):
        self.make().start('efa')
        self.assertIn('DRAIN', self.executor.node_state)
        del self.config_body['assignments']['table-1']['qualify_checks']
        self.write_config()
        before = len(self.node_calls())
        with self.assertRaisesRegex(session.Refusal, 'names no checks'):
            self.make().recover()
        self.assertEqual(before, len(self.node_calls()))
        self.assertEqual('fault-applied', self.make().store.read().phase)
        self.assertIn('DRAIN', self.executor.node_state)

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
                          'issued_at': 1000.0, 'issue_id': self.issue_id(1)}, profile)
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

    def test_the_issued_incident_record_is_read_whatever_its_mode(self):
        self.make().next()
        os.chmod(self.state_dir / 'schedule-table-1.json', 0o644)
        self.assertEqual('Incidents issued: 1 of 4', self.make().status().schedule_line)

    def test_a_seat_without_a_schedule_is_told_to_ask(self):
        del self.config_body['assignments']['table-1']['exercise_schedule']
        self.write_config()
        with self.assertRaisesRegex(session.Refusal, 'no incident schedule'):
            self.make().next()

    def test_a_stored_schedule_the_controller_refuses_names_no_exercise(self):
        # A seat installed by an earlier installer, or a hand-edited file.
        self.set_schedule([{'exercise': 'storage', 'settings': {'writer_rank': 9}},
                           {'exercise': 'silent-corruption',
                            'settings': {'corrupt_rank': 13, 'corrupt_step': 5000}}])
        with self.assertRaises(session.Refusal) as caught:
            self.make().next()
        text = str(caught.exception)
        self.assertEqual("This seat's incident schedule is unusable; raise your hand.", text)
        rendered = session.render(self.make().status(), 'status')
        for name in ('silent-corruption', 'corrupt', 'storage', 'writer_rank', '5000'):
            self.assertNotIn(name, text + rendered)
        # The operator reads which entry and which setting.
        view = session.operator_view(self.make())
        self.assertIn('Schedule: unusable (Schedule entry 2 setting corrupt_step must be '
                      'a whole number from 1 to 4095.)', view)

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
        self.assertIn('Next: Your node is in service. Start the next incident the way your '
                      'page shows.', session.render(self.make().status(), 'status'))
        self.advance_to(2)
        rendered = session.render(self.make().status(), 'status')
        self.assertIn('Incidents issued: 2 of 4', rendered)
        self.assertIn('Next: Incident 2 is under way. Follow your page for its next step.',
                      rendered)
        self.assertNotIn('Start the next incident', rendered)
        for name in ('storage', 'dataloader', 'start gpu'):
            self.assertNotIn(name, rendered)

    def test_after_resume_ends_an_incident_status_says_it_has_ended(self):
        """Code review batch 4, Nit 2, and Spain batch 4, finding 2."""
        self.advance_to(2)
        code, out, err = self.run_main(f'ended {self.issue_id(2)}', now=2000.0)
        self.assertEqual((0, ''), (code, err))
        self.assertEqual('Incident 2 has ended.\n', out)
        rendered = session.render(self.make().status(), 'status')
        self.assertIn('Incidents issued: 2 of 4', rendered)
        self.assertIn('Next: Incident 2 has ended. Start the next incident the way your page '
                      'shows.', rendered)
        self.assertNotIn('under way', rendered)
        for name in ('storage', 'dataloader', 'start gpu'):
            self.assertNotIn(name, rendered)
        self.assert_no_kind(rendered)
        # A second end keeps the first time, and the next incident is under way again.
        self.run_main(f'ended {self.issue_id(2)}', now=3000.0)
        progress = self.progress_record()
        self.assertEqual(2000.0, progress['issued'][1]['ended_at'])
        self.assertNotIn('ended_at', progress['issued'][0])
        self.assertEqual(3, self.make().next().incident)
        self.assertIn('Next: Incident 3 is under way.',
                      session.render(self.make().status(), 'status'))
        self.assertEqual([], self.node_calls())

    def test_the_last_incident_ended_says_there_is_no_next(self):
        self.set_schedule(session.validate_schedule([{'exercise': 'storage'},
                                                     {'exercise': 'dataloader'}]))
        self.advance_to(2)
        self.make().ended(self.issue_id(2))
        rendered = session.render(self.make().status(), 'status')
        self.assertIn('Next: Incident 2 has ended. It was the last incident for this seat.',
                      rendered)
        self.assertNotIn('Start the next incident', rendered)

    def test_ended_before_any_incident_or_without_a_schedule_is_refused(self):
        with self.assertRaisesRegex(session.Refusal, 'No incident with that issue id has been '
                                                     'issued on your seat'):
            self.make().ended('a' * 32)
        del self.config_body['assignments']['table-1']['exercise_schedule']
        self.write_config()
        with self.assertRaisesRegex(session.Refusal, 'no incident schedule; raise your hand'):
            self.make().ended('a' * 32)

    def test_ended_marks_only_the_incident_it_names(self):
        """Codex review of batch 5, finding 1: a late end must not end the next incident."""
        self.advance_to(1)
        first = self.issue_id(1)
        self.make().next()
        # Incident 1's resume finishes after terminal B issued Incident 2.
        code, out, err = self.run_main(f'ended {first}', now=2000.0)
        self.assertEqual((0, 'Incident 1 has ended.\n', ''), (code, out, err))
        progress = self.progress_record()
        self.assertEqual(2000.0, progress['issued'][0]['ended_at'])
        self.assertNotIn('ended_at', progress['issued'][1])
        self.assertIn('Next: Incident 2 is under way.',
                      session.render(self.make().status(), 'status'))
        # An issue id from another seat or an earlier pass ends nothing.
        code, out, err = self.run_main('ended ' + 'b' * 32)
        self.assertEqual((3, ''), (code, out))
        self.assertEqual('No incident with that issue id has been issued on your seat.\n', err)
        self.assertEqual(progress, self.progress_record())
        self.assertEqual([], self.node_calls())

    def test_ended_refuses_an_incident_whose_profile_was_not_handed_over(self):
        """Code review batch 5, Nit 3."""
        self.make().next()
        os.chmod(self.answers / 'incidents', 0o500)
        try:
            with self.assertRaises(session.Refusal):
                self.make().next()
        finally:
            os.chmod(self.answers / 'incidents', 0o700)
        code, out, err = self.run_main(f'ended {self.issue_id(2)}')
        self.assertEqual((3, ''), (code, out))
        self.assertEqual('Incident 2 has not started yet; follow your page.\n', err)
        self.assertNotIn('ended_at', self.progress_record()['issued'][1])
        # Once next finishes the hand-over, the incident can end.
        self.assertEqual(2, self.make().next().incident)
        self.assertEqual(2, self.make().ended(self.issue_id(2)).incident)
        self.assertIn('ended_at', self.progress_record()['issued'][1])

    def test_the_operator_view_shows_when_an_incident_ended(self):
        """Code review batch 5, Nit 6."""
        self.advance_to(2)
        self.make(now=1793699949.0).ended(self.issue_id(1))
        view = session.operator_view(self.make())
        [first] = [line for line in view.splitlines() if line.startswith('  1. ')]
        self.assertTrue(first.endswith(' ended 2026-11-03T09:59:09Z'), first)
        [second] = [line for line in view.splitlines() if line.startswith('  2. ')]
        self.assertNotIn(' ended ', second)

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
        issued = json.loads((self.state_dir / 'schedule-table-1.json').read_text())['issued']
        self.assertEqual({'number': 4, 'exercise': 'efa-reset',
                          'issue_id': issued[3]['issue_id']}, state.incident)
        self.assertRegex(issued[3]['issue_id'], r'\A[0-9a-f]{32}\Z')
        self.assertEqual(4, len({entry['issue_id'] for entry in issued}))
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

    def device_round_that_did_not_return(self):
        """Incident 1 is the device incident; its round ends, recovered, without a
        confirmed reset. Returns the participant's status after recovery."""
        self.set_schedule(session.validate_schedule([{'exercise': 'efa-reset'},
                                                     {'exercise': 'storage'}]))
        self.make().next()
        self.running_job_with_progress()
        with self.assertRaises(session.Refusal) as caught:
            self.make().next()
        self.assert_no_kind(str(caught.exception))
        self.executor.jobs = []
        self.assertEqual('runtime-ready', self.make().recover().state.phase)
        return session.render(self.make().status(), 'status')

    def test_a_round_that_never_issued_its_operation_says_so_and_starts_again(self):
        def not_dispatched(argv, record):
            self.active_calls.append(argv)
            return session.Completed(255, '', 'maintenance route unavailable\n', dispatched=False)
        self.executor.run_active_fault = not_dispatched
        rendered = self.device_round_that_did_not_return()
        self.assertEqual('not-issued', self.make().store.read().active_fault['outcome'])
        self.assertIn('Answer for incident 1: the lab did not apply this incident, so nothing '
                      'you saw came from it. Start it again with bash job.sh incident.', rendered)
        # The whole status stays concealed: the incident is started again.
        self.assert_no_kind(rendered)
        self.assertNotIn('reset', rendered)
        self.assertNotIn('not-issued', rendered)
        operator = session.render(self.make().status(), 'status', operator=True)
        self.assertIn('Active FLR: efa-flr', operator)
        self.assertIn('outcome=not-issued', operator)
        # The same incident again, not the next one: its profile until the job runs.
        again = self.make().next()
        self.assertEqual(1, again.incident)
        self.assertFalse(again.under_way)
        retry = session.render(self.make().status(), 'status')
        self.assert_no_kind(retry)
        self.assertNotIn('not-issued', retry)
        self.executor.run_active_fault = self.active
        self.running_job_with_progress()
        started = self.make().next()
        self.assertTrue(started.under_way)
        state = self.make().store.read()
        self.assertEqual((2, 'returned'), (state.round, state.active_fault['outcome']))
        self.assertEqual(1, state.incident['number'])
        # Once this round is recovered, the reveal and the move to incident 2 are
        # those of test_recovery_reveals_the_answer_and_the_schedule_moves_on.

    def test_a_reset_without_confirmation_is_not_revealed_as_done_or_repeated(self):
        self.rc = 124
        rendered = self.device_round_that_did_not_return()
        self.assertEqual('unknown-or-refused', self.make().store.read().active_fault['outcome'])
        self.assertIn('Answer for incident 1: the lab controller was to reset one EFA device on '
                      'your exercise node (0000:b0:00.0) while your job was running, and it has '
                      'no confirmation that the reset completed', rendered)
        self.assertNotIn('the lab controller reset one', rendered)
        # The device may have been reset, so the incident is not started a second time.
        calls = len(self.active_calls)
        self.assertEqual(2, self.make().next().incident)
        self.assertEqual(calls, len(self.active_calls))

    def test_before_its_job_reports_progress_status_says_it_is_under_way(self):
        """Spain batch 3: 'Start the next incident' printed while the job ran."""
        self.advance_to(3)
        job = session.render(self.make().status(), 'status')
        self.assertEqual(4, self.make().next().incident)
        self.executor.jobs = ['226|aim344-t1|aim344-device|RUNNING']
        device = session.render(self.make().status(), 'status')
        self.assertIn('Next: Incident 4 is under way. Follow your page for its next step.',
                      device)
        self.assertNotIn('Start the next incident', device)
        self.assert_no_kind(device)
        # The same lines as a job incident's status, apart from the numbers.
        self.assertEqual(re.sub(r'\d+', 'N', job), re.sub(r'\d+', 'N', device))
        self.assertEqual([], self.node_calls())

    def test_ended_does_not_end_the_device_incident_or_name_it(self):
        """Codex review of batch 5, finding 3: the reply is a job incident's."""
        self.advance_to(3)
        job = self.run_main(f'ended {self.issue_id(3)}')
        self.make().next()
        device = self.run_main(f'ended {self.issue_id(4)}')
        self.assertEqual((0, 'Incident 4 has ended.\n', ''), device)
        # The same exit status and text as a job incident's, apart from the number.
        self.assertEqual(job, (device[0], device[1].replace('4', '3'), device[2]))
        progress = self.progress_record()
        self.assertNotIn('ended_at', progress['issued'][3])
        self.assertIn('Next: Incident 4 is under way.',
                      session.render(self.make().status(), 'status'))
        self.assertEqual([], self.node_calls())

    def test_a_failed_preparation_shows_no_answer_before_recovery(self):
        """Batch-3 review finding 1: a refused drain leaves the round preparing."""
        self.set_schedule(session.validate_schedule([{'exercise': 'efa-reset'},
                                                     {'exercise': 'storage'}]))
        self.make().next()
        self.running_job_with_progress()
        real = self.executor.run_slurm

        def refuse_drain(argv, timeout=None):
            if argv[1:2] == ['update'] and 'State=DRAIN' in argv:
                self.executor.slurm_calls.append(list(argv))
                return session.Completed(1, '', 'drain refused\n')
            return real(argv, timeout)
        self.executor.run_slurm = refuse_drain
        with self.assertRaises(session.Refusal):
            self.make().next()
        state = self.make().store.read()
        self.assertEqual(('preparing', 'not-issued'),
                         (state.phase, state.active_fault['outcome']))
        before = session.render(self.make().status(), 'status')
        self.assertNotIn('Answer for incident', before)
        self.assertNotIn('Start it again', before)
        self.assertIn('Next: Preparation did not finish; run recover to restore your node.',
                      before)
        self.assertNotIn('Preparation is running', before)
        self.assert_no_kind(before)
        # This round changed nothing on the node, so recovery refuses it and the
        # round stays unrevealed. The answer after a recovered not-issued round is
        # test_a_round_that_never_issued_its_operation_says_so_and_starts_again.
        self.executor.run_slurm = real
        self.executor.jobs = []
        with self.assertRaises(session.Refusal):
            self.make().recover()
        phase = self.make().store.read().phase
        self.assertNotIn(phase, session.REVEAL_PHASES)
        after = session.render(self.make().status(), 'status')
        self.assertNotIn('Answer for incident', after)
        self.assertNotIn('Start it again', after)
        self.assert_no_kind(after)

    def test_status_during_a_running_preparation_asks_for_status_again(self):
        """Spain single-user pass, finding 1: status read 'preparing' while next
        was still preparing the fault, and told the participant to recover."""
        self.to_device_incident()
        self.running_job_with_progress()
        seen = []
        real = self.executor.run_slurm

        def status_during_drain(argv, timeout=None):
            result = real(argv, timeout)
            if argv[1:2] == ['update'] and 'State=DRAIN' in argv:
                # The preparing record is written and next still holds the lock.
                seen.append(self.run_main('status'))
            return result
        self.executor.run_slurm = status_during_drain
        self.assertTrue(self.make().next().under_way)
        self.assertEqual(1, len(seen))
        code, out, err = seen[0]
        self.assertEqual(0, code, err)
        self.assertIn('Exercise phase: preparing', out)
        self.assertIn('Next: Preparation is running; run status again.', out)
        self.assertNotIn('run recover', out)
        self.assert_no_kind(out)
        # The status did not make next refuse: the fault was applied once.
        self.assertEqual('fault-applied', self.make().store.read().phase)
        self.assertEqual(1, len(self.active_calls))

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


class ActiveRoundNextStep(QuizBase):
    def test_an_active_round_points_to_the_recorded_healthy_values(self):
        """F7: the active round records no baseline files, so it cites none."""
        self.advance_to(4)
        self.running_job_with_progress()
        self.make().next()
        self.assertEqual(session.ACTIVE_ROUND_COMPARE, self.make().status().next_step)
        self.assertEqual([], list(self.answers.rglob('*baseline*')))

    def test_an_idle_round_still_compares_with_its_own_baseline(self):
        self.make().start('efa')
        self.assertIn('compare it with your baseline', self.make().status().next_step)
        self.assertTrue(list(self.answers.rglob('check-2-baseline.log')))


class ScheduleReissue(QuizBase):
    """F2: a round left by an earlier pass of the schedule is not this pass's."""

    def rehearse_device_incident(self):
        self.advance_to(4)
        self.running_job_with_progress()
        self.assertTrue(self.make().next().under_way)
        self.executor.jobs = []
        self.assertEqual('runtime-ready', self.make().recover().state.phase)

    def test_status_does_not_answer_an_incident_of_an_earlier_pass(self):
        self.rehearse_device_incident()
        self.assertIn('Answer for incident 4', session.render(self.make().status(), 'status'))
        (self.state_dir / 'schedule-table-1.json').unlink()
        rendered = session.render(self.make().status(), 'status')
        self.assertNotIn('Answer for incident', rendered)
        self.assertNotIn('Incident: 4', rendered)
        self.assertIn('Incident: 4', session.operator_view(self.make()))

    def test_the_device_incident_runs_again_in_the_next_pass(self):
        self.rehearse_device_incident()
        (self.state_dir / 'schedule-table-1.json').unlink()
        calls = len(self.active_calls)
        self.advance_to(4)
        self.running_job_with_progress()
        result = self.make().next()
        self.assertTrue(result.under_way)
        self.assertEqual(calls + 1, len(self.active_calls))
        self.assertEqual(2, self.make().store.read().round)

    def test_an_issued_entry_without_an_identifier_is_refused(self):
        self.make().next()
        path = self.state_dir / 'schedule-table-1.json'
        progress = json.loads(path.read_text())
        del progress['issued'][0]['issue_id']
        path.write_text(json.dumps(progress))
        with self.assertRaisesRegex(session.Refusal, 'does not match its schedule'):
            self.make().next()

    def test_a_round_record_without_an_identifier_is_read_but_never_current(self):
        state = session.State.from_dict({
            'phase': 'runtime-ready', 'kind': 'efa', 'started_at': 1.0,
            'target_node': 'gpu-g7-1', 'incident': {'number': 1, 'exercise': 'efa-reset'}})
        self.assertFalse(session.same_issue(state.incident, {'number': 1, 'issue_id': None}))
        with self.assertRaisesRegex(session.Refusal, 'malformed incident'):
            session.State.from_dict({
                'phase': 'runtime-ready', 'kind': 'efa', 'started_at': 1.0,
                'target_node': 'gpu-g7-1',
                'incident': {'number': 1, 'exercise': 'efa-reset', 'issue_id': 'abc'}})


class CollectAfterRecovery(QuizBase):
    """F4: a recovered node is in service, so collect narrows, then stops."""

    def recover_device_incident(self):
        self.set_schedule(session.validate_schedule([
            {'exercise': 'efa-reset'},
            {'exercise': 'silent-corruption', 'settings': {'corrupt_rank': 13}}]))
        self.make().next()
        self.running_job_with_progress()
        self.assertTrue(self.make().next().under_way)
        self.executor.jobs = []
        self.assertEqual('runtime-ready', self.make().recover().state.phase)
        self.executor.maintenance_calls.clear()

    def test_collect_is_refused_once_a_later_incident_is_issued(self):
        self.recover_device_incident()
        self.assertEqual(2, self.make().next().incident)
        self.executor.jobs = ['230|aim344-t1|aim344-job|RUNNING']
        for check in ('0', '2', '6', 'kernel'):
            with self.subTest(check=check):
                with self.assertRaisesRegex(session.Refusal, 'no exercise running'):
                    self.make().collect(check)
        self.assertEqual([], self.executor.maintenance_calls)
        self.assertNotIn('Checks you can collect',
                         session.render(self.make().status(), 'status'))

    def test_before_that_only_the_rounds_own_checks_run_and_never_beside_a_job(self):
        self.recover_device_incident()
        self.assertIn('Checks you can collect: 2, 6, kernel',
                      session.render(self.make().status(), 'status'))
        with self.assertRaisesRegex(session.Refusal, r'Available: 2, 6\.'):
            self.make().collect('0')
        self.executor.jobs = ['230|aim344-t1|aim344-job|RUNNING']
        with self.assertRaisesRegex(session.Refusal, 'a job is running'):
            self.make().collect('6')
        self.assertEqual([], self.executor.maintenance_calls)
        self.executor.jobs = []
        saved = self.make().collect('2').saved_path
        self.assertEqual([['collect', '2']],
                         [list(c[:2]) for c in self.executor.maintenance_calls])
        self.assertIn('recovered', Path(saved).name)

    def test_a_round_started_with_start_closes_when_next_issues_an_incident(self):
        self.make(now=1000.0).start('efa')
        self.make(now=1010.0).recover()
        self.make(now=1020.0).collect('2')
        self.make(now=2000.0).next()
        with self.assertRaisesRegex(session.Refusal, 'no exercise running'):
            self.make(now=2010.0).collect('2')


class WarnRefusalAndOperatorReplacement(QuizBase):
    """Decisions 3 and 4: a WARN other than the Check 6 counters keeps the node out,
    and an operator replaces it."""

    def fire_and_warn(self, outputs=None):
        self.set_schedule(session.validate_schedule([{'exercise': 'efa-reset'},
                                                     {'exercise': 'storage'}]))
        self.make().next()
        self.running_job_with_progress()
        self.make().next()
        self.executor.jobs = []
        outputs = outputs or {'2': '[PASS] 2-efa-enumeration: EFA OK\n',
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


class CounterWarningRecordedAsMonitor(WarnRefusalAndOperatorReplacement):
    """PI, 2026-11-03: recovery needs Checks 2 and 6 to pass; a Check 6 warning about
    the cumulative EFA counters is recorded as MONITOR and does not refuse recovery."""

    PASS_2 = '[PASS] 2-efa-enumeration: EFA OK\n'
    PASS_6 = '[PASS] 6-efa-loopback: EFA loopback OK: 2 domain(s) tested\n'
    RETRANS = '[WARN] 6-efa-loopback: EFA retransmission timeouts detected (2)\n'
    DROPS = '[WARN] 6-efa-loopback: EFA rx_drops detected (5) -- possible network issues\n'

    def recover_with(self, check_6, check_2=PASS_2):
        self.fire_and_warn({'2': check_2, '6': check_6})
        return self.run_main('recover')

    def test_a_counter_warning_is_recorded_as_monitor_and_the_node_resumes(self):
        code, out, err = self.recover_with(self.RETRANS + self.PASS_6)
        self.assertEqual(0, code, err)
        state = self.make().store.read()
        self.assertEqual('runtime-ready', state.phase)
        self.assertEqual(1, len([c for c in self.executor.slurm_calls if 'State=RESUME' in c]))
        record = state.check_results['round-1/recovered/check-6']
        self.assertEqual(('WARN', 'MONITOR'), (record['verdict'], record['severity']))
        self.assertNotIn('severity', state.check_results['round-1/recovered/check-2'])
        self.assertIn('retrans_timeout_events=2', state.notes)
        self.assertIn('recorded as MONITOR', out)
        # Only what was observed: running totals read after the loopback. A
        # same-instance reboot did not reset them, so no starting point is named.
        self.assertIn("reads once, after its loopback test; they are running totals, "
                      "not a count of this round's events.", out)
        for claim in ('instance launch', 'driver last reset', 'since the'):
            self.assertNotIn(claim, out)
        code, status, _ = self.run_main('status')
        self.assertEqual(0, code)
        self.assertRegex(status, r'round-1/recovered/check-6: WARN, recorded as MONITOR \(')
        self.assertRegex(status, r'round-1/recovered/check-2: PASS \(')
        self.assertIn('Answer for incident 1:', status)

    def test_both_counters_and_the_unprefixed_form_are_monitored(self):
        unprefixed = '[WARN] EFA retransmission timeouts detected (7)\n'
        code, _, err = self.recover_with(self.DROPS + unprefixed + self.PASS_6)
        self.assertEqual(0, code, err)
        state = self.make().store.read()
        self.assertEqual('runtime-ready', state.phase)
        self.assertIn('retrans_timeout_events=7, rx_drops=5', state.notes)

    def test_any_other_check_6_warning_keeps_its_refusal(self):
        incomplete = ('[WARN] 6-efa-loopback: EFA statistics incomplete or malformed: '
                      'rx_drops=unknown, retrans_timeout_events=unknown; absolute totals '
                      'unknown, not zero\n')
        cases = (self.RETRANS,                                          # no loopback PASS
                 self.RETRANS + incomplete + self.PASS_6,               # another warning
                 '[WARN] 2-efa-enumeration: EFA rx_drops detected (5) -- possible '
                 'network issues\n' + self.PASS_6,                      # another check's name
                 self.RETRANS + '[WARN]\n' + self.PASS_6,               # unreadable line
                 self.RETRANS + 'EFA statistics skipped\n' + self.PASS_6)
        for index, check_6 in enumerate(cases):
            with self.subTest(check_6=check_6):
                if index:
                    # A fresh seat for each case; the framework's tearDown ends the last.
                    self.tearDown()
                    self.setUp()
                code, _, err = self.recover_with(check_6)
                self.assertEqual(3, code)
                state = self.make().store.read()
                self.assertEqual('replacement-required', state.phase, err)
                self.assertNotIn('severity', state.check_results['round-1/recovered/check-6'])
                self.assertEqual([], [c for c in self.executor.slurm_calls if 'State=RESUME' in c])

    def test_check_2_must_still_pass(self):
        warn_2 = ('[WARN] 2-efa-enumeration: Memory lock limit 8192 KB is below 16 GiB\n'
                  '[WARN] 2-efa-enumeration: 1 EFA advisory(ies)\n')
        code, _, err = self.recover_with(self.RETRANS + self.PASS_6, check_2=warn_2)
        self.assertEqual(3, code)
        state = self.make().store.read()
        self.assertEqual('replacement-required', state.phase)
        self.assertEqual('MONITOR', state.check_results['round-1/recovered/check-6']['severity'])
        self.assertIn('Check 2 returned WARN', err)
        self.assertEqual([], [c for c in self.executor.slurm_calls if 'State=RESUME' in c])

    def test_the_rule_is_limited_to_check_6(self):
        handler = self.make()
        for check in ('0', '2', '3'):
            self.assertIsNone(handler._monitored_counters(check, self.RETRANS + self.PASS_6))
        self.assertEqual(['retrans_timeout_events=2'],
                         handler._monitored_counters('6', self.RETRANS + self.PASS_6))


class InstallerWorldSize(unittest.TestCase):
    def test_config_writer_preserves_explicit_world_size(self):
        import pwd
        source = INSTALLER.read_text().split("<<'PY'", 2)[2].split('\nPY\n', 1)[0]
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / 'config.json'
            who = pwd.getpwuid(os.getuid()).pw_name
            argv = ['-', str(config), 'seat-test', who, 'node-a', 'i-0123456789abcdef0',
                    'node-a', temp, '/opt/slurm/bin', '/opt/suite/check.sh', str(HELPER)]
            env = dict(os.environ, AIM344_PARTITION='test', AIM344_WORLD_SIZE='8')
            env.pop('AIM344_CHECKED_SCHEDULE', None)
            result = subprocess.run([sys.executable, '-B', '-c', source, *argv[1:]],
                                    env=env, text=True, capture_output=True)
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual(8, json.loads(config.read_text())['assignments']['seat-test']['world_size'])


class FiveIncidentSeat(QuizBase):
    """The event schedule: slow job, storage, DataLoader, silent corruption, device."""

    EVENT = [{'exercise': 'slow-job'}] + SCHEDULE

    def setUp(self):
        super().setUp()
        self.set_schedule(session.validate_schedule(self.EVENT))

    def test_the_participant_counts_five_and_the_device_incident_is_number_5(self):
        self.assertIn('Incidents issued: 0 of 5', session.render(self.make().status(), 'status'))
        for number, exercise in enumerate(('slow-job', 'storage', 'dataloader',
                                           'silent-corruption'), start=1):
            result = self.make().next()
            self.assertEqual(number, result.incident)
            profile = json.loads(Path(result.saved_path).read_text())
            self.assertEqual((number, exercise), (profile['incident'], profile['exercise']))
        self.assertIn('Incidents issued: 4 of 5', session.render(self.make().status(), 'status'))
        issued = self.make().next()
        self.assertEqual(5, issued.incident)
        self.assertEqual('Incident 5 is ready.', session.render(issued, 'next').splitlines()[0])
        self.running_job_with_progress()
        under_way = self.make().next()
        self.assertEqual('Incident 5 is under way.',
                         session.render(under_way, 'next').splitlines()[0])
        status = session.render(self.make().status(), 'status')
        self.assertIn('Incident: 5', status)
        self.assertIn('Incidents issued: 5 of 5', status)
        self.executor.jobs = []
        self.assertEqual('runtime-ready', self.make().recover().state.phase)
        self.assertIn('Answer for incident 5: the lab controller reset one EFA device',
                      session.render(self.make().status(), 'status'))

    def test_the_installer_and_readme_examples_list_six_in_the_pages_order(self):
        def example(text, start):
            block = text[text.index(start):]
            return re.findall(r'"exercise": "([a-z-]+)"', block[:block.index(']') + 1])
        order = ['slow-job', 'storage', 'dataloader', 'silent-corruption', 'cuda-assert', 'efa-reset']
        installer = INSTALLER.read_text()
        self.assertEqual(order, example(installer, '#   [{"exercise"'))
        readme = (LAB / 'README.md').read_text()
        self.assertEqual(order, example(readme, '[{"exercise"'))


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

    def test_the_configurations_one_assignment_is_used(self):
        code, out, err = self.run_main('status')
        self.assertEqual(0, code, err)
        self.assertIn('Exercise phase: ready', out)
        self.assertIn('Incidents issued: 0 of 4', out)
        self.config_body['assignments']['table-2'] = dict(
            self.config_body['assignments']['table-1'])
        self.write_config()
        code, _, err = self.run_main('status')
        self.assertEqual(3, code)
        self.assertIn('must hold one assignment', err)

    def test_status_answers_while_another_request_holds_the_lock(self):
        handler = self.make()
        with handler.store.exclusive():
            code, out, err = self.run_main('status')
            self.assertEqual(0, code, err)
            self.assertIn('Exercise phase: ready', out)
            code, _, err = self.run_main('next')
            self.assertEqual(3, code)
            self.assertIn('Another request for this assignment is in progress.', err)

    def test_the_operator_view_is_root_only_and_has_no_replace_route(self):
        code, _, err = self.run_main('', argv=['--operator', 'status',
                                               '--config', str(self.config_path)])
        self.assertEqual(3, code)
        self.assertIn('root', err)
        for argv in (['--operator', 'replace'], ['--sweep'], ['--expire'],
                     ['--assignment', 'table-1']):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                session.main(argv + ['--config', str(self.config_path), 'status'])


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

    def test_every_refusal_is_concealed_against_the_wider_pattern(self):
        """F8: no refusal still names an operation once it is concealed."""
        texts = self.refusal_texts()
        self.assertGreater(len(texts), 100)
        for name, line, text in texts:
            with self.subTest(source=f'{name}:{line}'):
                shown = session.conceal_kind(session.participant_replacement_text(text))
                self.assertIsNone(KIND_WORDS.search(shown), shown)
        for word in ('the unbind', 'for the rebind', 'The provisioned RDMA device name',
                     'a PCIe function-level reset of 0000:b0:00.0'):
            with self.subTest(word=word):
                self.assertIsNone(KIND_WORDS.search(session.conceal_kind(word)))

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
    """The installer's schedule check and configuration step, each run on its own as
    an ordinary user, in the order the installer runs them."""

    @staticmethod
    def block(text, opening):
        start = text.index(opening)
        body = text.index("<<'PY'\n", start) + len("<<'PY'\n")
        return text[body:text.index('\nPY\n', body)]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = self.root / 'device-session.json'
        self.state_dir = self.root / 'state'
        self.state_dir.mkdir(mode=0o700)
        text = INSTALLER.read_text()
        self.check_block = self.block(text, 'python3 - "$AIM344_SCHEDULE_FILE"')
        self.config_block = self.block(text, 'python3 - "$session_config"')
        self.user = subprocess.run(['id', '-un'], capture_output=True, text=True,
                                   check=True).stdout.strip()

    def tearDown(self):
        self.tmp.cleanup()

    def check(self, schedule_file):
        return subprocess.run(
            [sys.executable, '-', str(schedule_file), str(HELPER), str(self.config),
             str(self.state_dir), 'table-1'],
            input=self.check_block, capture_output=True, text=True)

    def configure(self, checked=None):
        env = dict(os.environ, AIM344_PARTITION='queue')
        env.pop('AIM344_SCHEDULE_FILE', None)
        env.pop('AIM344_CHECKED_SCHEDULE', None)
        if checked:
            env['AIM344_CHECKED_SCHEDULE'] = checked
        return subprocess.run(
            [sys.executable, '-', str(self.config), 'table-1', self.user, 'gpu-g7-1',
             'i-0123456789abcdef0', '10.0.0.1', str(self.state_dir), '/opt/slurm/bin',
             '/opt/suite/gpu-healthcheck.sh', str(HELPER)],
            input=self.config_block, capture_output=True, text=True, env=env)

    def install(self, schedule_file=None):
        if schedule_file is None:
            return self.configure()
        checked = self.check(schedule_file)
        if checked.returncode != 0:
            return checked
        return self.configure(checked.stdout.strip())

    def write_state(self, **fields):
        session.StateStore(self.state_dir, 'table-1').write(session.State(
            kind='efa', started_at=1.0, target_node='gpu-g7-1', **fields))

    def read_state(self):
        return session.StateStore(self.state_dir, 'table-1').read()

    def schedule_file(self, entries, mode=0o600):
        path = self.root / 'seat-1.json'
        path.write_text(json.dumps(entries))
        os.chmod(path, mode)
        return path

    def assignment(self):
        return json.loads(self.config.read_text())['assignments']['table-1']

    def test_existing_world_size_validates_new_and_inherited_schedules(self):
        for world_size, rank, valid in [(8, 3, True), (8, 9, False), (32, 20, True)]:
            with self.subTest(world_size=world_size, rank=rank):
                initial = {'state_dir': str(self.state_dir), 'assignments': {
                    'table-1': {'world_size': world_size}}}
                self.config.write_text(json.dumps(initial))
                entries = [{'exercise': 'cuda-assert', 'settings': {'assert_rank': rank}}]
                with mock.patch.dict(os.environ):
                    os.environ.pop('AIM344_WORLD_SIZE', None)
                    checked = self.check(self.schedule_file(entries))
                    saved = self.configure(json.dumps(entries))
                self.assertEqual(valid, checked.returncode == 0, checked.stderr)
                self.assertEqual(valid, saved.returncode == 0, saved.stderr)
                if valid:
                    self.assertEqual(world_size, self.assignment()['world_size'])
        # Changing only world size must also validate the inherited rank settings.
        self.config.write_text(json.dumps({'state_dir': str(self.state_dir), 'assignments': {
            'table-1': {'world_size': 16, 'exercise_schedule': [
                {'exercise': 'storage', 'settings': {'writer_rank': 9}}]}}}))
        before = self.config.read_bytes()
        with mock.patch.dict(os.environ, AIM344_WORLD_SIZE='8'):
            checked = self.check('')
            saved = self.configure()
        self.assertNotEqual(0, checked.returncode)
        self.assertNotEqual(0, saved.returncode)
        self.assertEqual(before, self.config.read_bytes())

    def test_supplied_null_refuses_without_resetting_existing_records(self):
        self.assertEqual(0, self.install(self.schedule_file(SCHEDULE)).returncode)
        self.write_state(phase='runtime-ready', round=3)
        progress = self.state_dir / 'schedule-table-1.json'
        progress.write_text('{"issued": [{"issue_id": "keep-this-record"}]}')
        paths = [self.config, progress, self.state_dir / 'table-1.json']
        before = {p.name: p.read_bytes() for p in paths}
        checked = self.check(self.schedule_file(None))
        configured = self.configure('null')
        for stage, result in [('precheck', checked), ('configure', configured)]:
            with self.subTest(stage=stage):
                self.assertNotEqual(0, result.returncode, result.stdout)
        self.assertEqual(before, {p.name: p.read_bytes() for p in paths if p.exists()})
        self.assertFalse(list(self.state_dir.glob('*.before-reset.json')))

    def test_refused_world_size_preserves_config_progress_and_round(self):
        self.assertEqual(0, self.configure().returncode)
        self.write_state(phase='runtime-ready', round=3)
        progress = self.state_dir / 'schedule-table-1.json'
        progress.write_text('{"issued": [{"issue_id": "keep-this-record"}]}')
        before = {p.name: p.read_bytes() for p in [self.config, progress,
                                                   self.state_dir / 'table-1.json']}
        entries = [{'exercise': 'cuda-assert', 'settings': {'assert_rank': 0}}]
        with mock.patch.dict(os.environ, AIM344_WORLD_SIZE='3'):
            for input_file in (str(self.schedule_file(entries)), ''):
                self.assertNotEqual(0, self.check(input_file).returncode)
            self.assertNotEqual(0, self.configure(json.dumps(entries)).returncode)
        after = {p.name: p.read_bytes() for p in [self.config, progress,
                                                  self.state_dir / 'table-1.json'] if p.exists()}
        self.assertEqual(before, after)
        self.assertFalse(list(self.state_dir.glob('*.before-reset.json')))

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

    def test_an_unknown_schedule_is_refused_whatever_the_file_mode(self):
        done = self.install(self.schedule_file([{'exercise': 'nvlink'}]))
        self.assertNotEqual(0, done.returncode)
        self.assertIn('unknown exercise', done.stderr)
        self.assertFalse(self.config.exists())
        done = self.install(self.schedule_file(SCHEDULE, mode=0o644))
        self.assertEqual(0, done.returncode, done.stderr)
        self.assertEqual(4, len(self.assignment()['exercise_schedule']))

    def test_a_rank_outside_the_job_is_refused_when_the_seat_is_installed(self):
        entries = [{'exercise': 'silent-corruption', 'settings': {'corrupt_rank': 16}}]
        done = self.check(self.schedule_file(entries))
        self.assertNotEqual(0, done.returncode)
        self.assertIn('setting corrupt_rank must be a rank of the 16-rank job, from 0 to 15',
                      done.stderr)
        larger = subprocess.run(
            [sys.executable, '-', str(self.schedule_file(entries)), str(HELPER), str(self.config),
             str(self.state_dir), 'table-1'], input=self.check_block, capture_output=True,
            text=True, env=dict(os.environ, AIM344_WORLD_SIZE='32'))
        self.assertEqual(0, larger.returncode, larger.stderr)
        self.assertEqual(entries, json.loads(larger.stdout))
        bad = subprocess.run(
            [sys.executable, '-', str(self.schedule_file(SCHEDULE)), str(HELPER), str(self.config),
             str(self.state_dir), 'table-1'], input=self.check_block, capture_output=True,
            text=True, env=dict(os.environ, AIM344_WORLD_SIZE='sixteen'))
        self.assertNotEqual(0, bad.returncode)
        self.assertIn('AIM344_WORLD_SIZE', bad.stderr)

    def test_the_schedule_is_checked_before_the_installer_changes_anything(self):
        """F6: a refused file stops the installer before its first change."""
        text = INSTALLER.read_text()
        check = text.index('python3 - "$AIM344_SCHEDULE_FILE"')
        for first_change in ('install -o root -g root -m 0755', 'ssh-keygen',
                             'install -d -o root -g root -m 0700 "$state_dir"',
                             'install -d -o "$participant"'):
            with self.subTest(change=first_change):
                self.assertLess(check, text.index(first_change))
        self.assertIn('AIM344_CHECKED_SCHEDULE=$checked_schedule \\\npython3 - '
                      '"$session_config"', text)
        before = sorted(p.name for p in self.root.rglob('*'))
        for name, entries, mode, expected in (
                ('unknown', [{'exercise': 'nvlink'}], 0o600, 'unknown exercise'),
                ('not JSON', None, 0o600, 'not JSON')):
            with self.subTest(case=name):
                path = self.schedule_file(entries or [], mode=mode)
                if entries is None:
                    path.write_text('[{')
                done = self.check(path)
                self.assertNotEqual(0, done.returncode)
                self.assertIn(expected, done.stderr)
                self.assertEqual('', done.stdout)
        after = sorted(p.name for p in self.root.rglob('*') if p.name != 'seat-1.json')
        self.assertEqual([n for n in before if n != 'seat-1.json'], after)

    def test_a_reset_is_refused_while_a_round_holds_the_node(self):
        """F2: the installer does not restart a schedule under a held round."""
        self.install(self.schedule_file(SCHEDULE))
        progress = self.state_dir / 'schedule-table-1.json'
        progress.write_text('{"issued": []}')
        os.chmod(progress, 0o600)
        self.write_state(phase='fault-applied')
        done = self.check(self.schedule_file(SCHEDULE))
        self.assertNotEqual(0, done.returncode)
        self.assertIn('still holds the exercise node', done.stderr)
        # The configuration step refuses too, should the round begin in between.
        done = self.configure(json.dumps(session.validate_schedule(SCHEDULE)))
        self.assertNotEqual(0, done.returncode)
        self.assertIn('still holds the exercise node', done.stderr)
        self.assertTrue(progress.exists())
        self.assertEqual('fault-applied', self.read_state().phase)

    def test_a_reset_keeps_the_earlier_round_and_continues_from_a_fresh_record(self):
        """F2: status after a reset shows nothing of the earlier pass."""
        incident = {'number': 4, 'exercise': 'efa-reset', 'issue_id': 'a' * 32}
        self.write_state(phase='runtime-ready', round=2, incident=incident,
                         check_results={'round-2/recovered/check-6': {'verdict': 'PASS'}})
        done = self.install(self.schedule_file(SCHEDULE))
        self.assertEqual(0, done.returncode, done.stderr)
        state = self.read_state()
        self.assertEqual(('ready', 3, None, {}),
                         (state.phase, state.round, state.incident, state.check_results))
        kept = self.state_dir / 'table-1.round-2.before-reset.json'
        self.assertEqual(incident, json.loads(kept.read_text())['incident'])
        self.assertEqual(0, stat.S_IMODE(kept.stat().st_mode) & 0o077)
        self.assertIn('continues from round 3', done.stdout)


class ApplicationIncidentBase(QuizBase):
    """A seat whose first incident is the application assertion, then storage.

    The target node gpu-g7-1 is the first of the allocation's two hosts, so with
    the default 16-rank job its ranks are 0-7. Nothing here applies a fault: the
    controller only records the running allocation, hands over the profile, and
    reads evidence.
    """

    APPLICATION = [{'exercise': 'cuda-assert', 'settings': {'assert_rank': 3, 'assert_step': 201}},
                   {'exercise': 'storage', 'settings': {'writer_rank': 9}}]
    GPU_HEADER = 'index, uuid, pci.bus_id, gpu_recovery_action'

    def setUp(self):
        super().setUp()
        self.set_schedule(session.validate_schedule(self.APPLICATION))
        self.hosts = ['gpu-g7-1', 'gpu-g7-2']
        self.job_start = '2026-11-05T12:10:00'
        self.job_state = 'RUNNING'
        self.allocated_gpus = 8
        self.live_steps = []
        # The read-only recovery query fixture, varied per test.
        self.rows = self.gpu_rows()
        self.route_rc = 0            # the maintenance route's own exit status
        self.query_rc = 0            # nvidia-smi's exit status inside the record
        self.process_rc = 0
        self.process_stdout = 'gpu_uuid, pid, process_name\n'
        self.record_instance = None  # None: the fake target's identity
        self.record_boot = None      # None: the fake target's current boot
        self.record_stdout = None    # None: the rows above as CSV
        self.record_text = None      # None: a JSON record; else raw stdout
        real_slurm = self.executor.run_slurm
        real_maintenance = self.executor.run_maintenance

        def run_slurm(argv, timeout=None):
            if Path(argv[0]).name == 'squeue' and '--steps' in argv:
                self.executor.slurm_calls.append(list(argv))
                return session.Completed(0, '\n'.join(self.live_steps), '')
            if Path(argv[0]).name == 'scontrol' and argv[1:3] == ['show', 'job']:
                self.executor.slurm_calls.append(list(argv))
                nodelist = ','.join(self.hosts)
                return session.Completed(0, (
                    f'JobId={argv[-1]} JobName=aim344-device UserId=aim344-t1(1001) '
                    f'JobState={self.job_state} StartTime={self.job_start} '
                    f'NodeList={nodelist} NumNodes={len(self.hosts)} '
                    f'TresPerNode=gres/gpu:{self.allocated_gpus}\n'), '')
            if Path(argv[0]).name == 'scontrol' and argv[1:3] == ['show', 'hostnames']:
                self.executor.slurm_calls.append(list(argv))
                return session.Completed(0, ''.join(h + '\n' for h in self.hosts), '')
            return real_slurm(argv, timeout)

        def run_maintenance(argv, timeout=None):
            if argv[0] == 'gpu-recovery':
                self.executor.maintenance_calls.append(list(argv))
                return session.Completed(self.route_rc, self.recovery_text(), '')
            return real_maintenance(argv, timeout)
        self.executor.run_slurm = run_slurm
        self.executor.run_maintenance = run_maintenance

    def gpu_rows(self, count=8, action='None'):
        rows = []
        for index in range(count):
            uuid = (self.executor.inspect['gpu_uuid'] if index == 0
                    else f'GPU-{index:08d}-0000-0000-0000-000000000000')
            bus = '00000000:BA:00.0' if index == 0 else f'00000000:5{index}:00.0'
            rows.append((str(index), uuid, bus, action))
        return rows

    def recovery_text(self):
        if self.record_text is not None:
            return self.record_text
        stdout = self.record_stdout
        if stdout is None:
            stdout = self.GPU_HEADER + '\n' + ''.join(', '.join(row) + '\n' for row in self.rows)
        return json.dumps({
            'action': 'gpu-recovery',
            'instance_id': self.record_instance or self.executor.inspect['instance_id'],
            'boot_id': self.record_boot or self.executor.boot_id,
            'observed_at': 1000.5,
            'query': 'nvidia-smi --query-gpu=index,uuid,pci.bus_id,gpu_recovery_action --format=csv',
            'returncode': self.query_rc,
            'process_returncode': self.process_rc,
            'process_stdout': self.process_stdout,
            'process_stderr': '',
            'stdout': stdout,
            'stderr': '' if self.query_rc == 0 else 'NVIDIA-SMI has failed\n',
        }, sort_keys=True) + '\n'

    def running_allocation(self, job='226'):
        self.executor.jobs = [f'{job}|aim344-t1|aim344-device|RUNNING']

    def issue(self):
        self.running_allocation()
        return self.make().next()

    def incident_dir(self, number=1):
        return self.answers / 'collected' / f'incident-{number}-{self.issue_id(number)}'

    def mutations(self):
        """Every call that could change the node: scheduler updates and reboots,
        and every maintenance action other than the read-only ones."""
        updates = [c for c in self.executor.slurm_calls
                   if Path(c[0]).name == 'scontrol' and c[1] in ('update', 'reboot')]
        read_only = {'boot-id', 'instance-id', 'inspect', 'collect', 'kernel-evidence',
                     'gpu-recovery'}
        return updates + [c for c in self.executor.maintenance_calls if c[0] not in read_only]


class ApplicationIncidentGrammar(unittest.TestCase):
    def test_collect_recovery_and_resume_parse(self):
        request = session.parse_participant_request('collect recovery')
        self.assertEqual(('collect', 'recovery'), (request.verb, request.argument))
        issue = 'a' * 32
        request = session.parse_participant_request(f'resume {issue}')
        self.assertEqual(('resume', issue), (request.verb, request.argument))
        for text in ('resume', 'resume 2', f'resume {issue} {issue}', 'resume ' + 'A' * 32,
                     'collect recovery 0', 'collect Recovery', 'collect recover'):
            with self.subTest(text=text):
                self.assertIsNotNone(session.parse_participant_request(text).error)


class ApplicationIncidentNext(ApplicationIncidentBase):
    def test_allocated_gpu_count_must_match_configured_world_size(self):
        self.running_allocation()
        self.allocated_gpus = 4
        with self.assertRaises(session.Refusal):
            self.make().next()
        self.config_body['assignments']['table-1']['world_size'] = 8
        self.write_config()
        result = self.make().next()
        self.assertEqual(4, self.progress_record()['issued'][0]['job']['gpus_per_node'])

    def test_undelivered_profile_does_not_cross_allocation(self):
        self.running_allocation()
        handler = self.make()
        with mock.patch.object(handler, '_write_participant_file',
                               side_effect=session.Refusal('write failed')):
            with self.assertRaises(session.Refusal):
                handler.next()
        self.running_allocation('227')
        with self.assertRaisesRegex(session.Refusal, 'allocation'):
            self.make().next()
        self.assertIsNone(self.progress_record()['issued'][0]['profile'])


    def test_next_needs_the_running_allocation_and_records_it_before_the_profile(self):
        with self.assertRaisesRegex(session.Refusal, 'allocation'):
            self.make().next()
        self.assertEqual([], self.progress_record().get('issued', [])
                         if (self.state_dir / 'schedule-table-1.json').exists() else [])
        self.assertFalse(list(self.answers.rglob('incident-*.json')))
        result = self.issue()
        self.assertEqual(1, result.incident)
        self.assertFalse(result.under_way)
        profile = json.loads(Path(result.saved_path).read_text())
        self.assertEqual({'incident': 1, 'incidents': 2, 'exercise': 'cuda-assert',
                          'settings': {'assert_rank': 3, 'assert_step': 201},
                          'controller_step': 'none', 'issued_at': 1000.0,
                          'issue_id': self.issue_id(1)}, profile)
        entry = self.progress_record()['issued'][0]
        self.assertEqual({'id': '226', 'target_node': 'gpu-g7-1',
                          'target_instance_id': 'i-0123456789abcdef0',
                          'boot_id': 'aaaaaaaa-0000-0000-0000-000000000001',
                          'job_started_at': 1793880600, 'hosts': ['gpu-g7-1', 'gpu-g7-2'],
                          'gpus_per_node': 8, 'ranks': [0, 7]}, entry['job'])
        self.assertEqual(1000.0, entry['issued_at'])
        actions = [c[0] for c in self.executor.maintenance_calls]
        self.assertEqual(sorted(['boot-id', 'instance-id']), sorted(actions))
        self.assertEqual([], self.mutations())
        rendered = session.render(result, 'next')
        self.assertIn('Incident 1 is ready.', rendered)
        self.assert_no_kind(rendered)
        self.assertNotIn('assert', rendered)

    def test_a_pending_or_foreign_allocation_is_refused(self):
        for jobs in (['226|aim344-t1|aim344-device|PENDING'],
                     ['301|someone-else|aim344-x|RUNNING'],
                     ['226|aim344-t1|other-job|RUNNING']):
            with self.subTest(jobs=jobs):
                self.executor.jobs = jobs
                with self.assertRaises(session.Refusal):
                    self.make().next()
        self.assertFalse((self.state_dir / 'schedule-table-1.json').exists())

    def test_the_rank_must_run_on_the_target_node(self):
        """Rank r runs on hosts[r // (world_size / hosts)], hosts in Slurm's order."""
        self.set_schedule(session.validate_schedule(
            [{'exercise': 'cuda-assert', 'settings': {'assert_rank': 9}}]))
        self.running_allocation()
        with self.assertRaisesRegex(session.Refusal, 'not on your exercise node'):
            self.make().next()
        self.assertFalse((self.state_dir / 'schedule-table-1.json').exists())
        # The target second in the allocation holds ranks 8-15.
        self.hosts = ['gpu-g7-2', 'gpu-g7-1']
        result = self.make().next()
        self.assertEqual(1, result.incident)
        self.assertEqual([8, 15], self.progress_record()['issued'][0]['job']['ranks'])
        # A larger configured world size widens each node's share.
        (self.state_dir / 'schedule-table-1.json').unlink()
        self.hosts = ['gpu-g7-1', 'gpu-g7-2']
        self.config_body['assignments']['table-1']['world_size'] = 32
        self.allocated_gpus = 16
        self.write_config()
        self.make().next()
        self.assertEqual([0, 15], self.progress_record()['issued'][0]['job']['ranks'])

    def test_an_allocation_the_mapping_cannot_place_is_refused(self):
        self.running_allocation()
        for hosts in (['gpu-g7-2', 'gpu-g7-3'], ['gpu-g7-1'], ['gpu-g7-1', 'gpu-g7-2', 'gpu-g7-3']):
            with self.subTest(hosts=hosts):
                self.hosts = hosts
                with self.assertRaises(session.Refusal):
                    self.make().next()
        self.config_body['assignments']['table-1']['world_size'] = 15
        self.write_config()
        self.hosts = ['gpu-g7-1', 'gpu-g7-2']
        with self.assertRaises(session.Refusal):
            self.make().next()
        self.assertFalse((self.state_dir / 'schedule-table-1.json').exists())

    def test_a_changed_identity_or_unusable_boot_is_refused(self):
        self.running_allocation()
        self.executor.inspect['instance_id'] = 'i-0fedcba9876543210'
        with self.assertRaisesRegex(session.Refusal, 'identity'):
            self.make().next()
        self.executor.inspect['instance_id'] = 'i-0123456789abcdef0'
        self.executor.boot_id = ''
        with self.assertRaises(session.Refusal):
            self.make().next()
        self.assertFalse((self.state_dir / 'schedule-table-1.json').exists())

    def test_next_waits_until_the_incident_has_ended(self):
        self.issue()
        with self.assertRaisesRegex(session.Refusal, 'Incident 1 is under way'):
            self.make().next()
        self.assertEqual(1, len(self.progress_record()['issued']))
        self.make().ended(self.issue_id(1))
        following = self.make().next()
        self.assertEqual(2, following.incident)
        self.assertEqual('storage', json.loads(Path(following.saved_path).read_text())['exercise'])
        self.assertEqual([], self.mutations())

    def test_status_tells_collect_then_resume_and_later_next(self):
        self.assertIn('Next: Your node is in service. Start the next incident',
                      session.render(self.make().status(), 'status'))
        self.issue()
        rendered = session.render(self.make().status(), 'status')
        self.assertIn('Exercise phase: ready', rendered)
        self.assertIn('Incident: 1', rendered)
        self.assertIn('Incidents issued: 1 of 2', rendered)
        self.assertIn('Checks you can collect: 0, 2, 3, 6, kernel, recovery', rendered)
        self.assertIn('Next: Incident 1 is under way. Collect your evidence, then run the '
                      'resume step your page shows; next continues after that.', rendered)
        self.assertNotIn('Answer for', rendered)
        self.assertNotIn('Start the next incident', rendered)
        self.assert_no_kind(rendered)
        for word in ('assert', 'cuda', 'Xid', 'application', 'storage'):
            self.assertNotIn(word, rendered)
        self.make().ended(self.issue_id(1))
        after = session.render(self.make().status(), 'status')
        self.assertIn('Next: Incident 1 has ended. Start the next incident', after)
        self.assertNotIn('Checks you can collect', after)

    def test_other_job_incidents_do_not_read_the_node(self):
        self.set_schedule(session.validate_schedule([{'exercise': 'storage'}]))
        result = self.make().next()
        self.assertEqual(1, result.incident)
        self.assertEqual([], self.executor.maintenance_calls)
        self.assertNotIn('job', self.progress_record()['issued'][0])
        with self.assertRaisesRegex(session.Refusal, 'no exercise running'):
            self.make().collect('0')
        with self.assertRaisesRegex(session.Refusal, 'pre-resume check'):
            self.make().resume(self.issue_id(1))
        self.assertNotIn('ended_at', self.progress_record()['issued'][0])


class ApplicationIncidentCollect(ApplicationIncidentBase):
    def test_read_only_checks_run_beside_the_job_into_the_incident_directory(self):
        self.issue()
        self.executor.maintenance_calls.clear()
        first = self.make().collect('0')
        self.assertEqual(self.incident_dir() / 'check-0-incident.log', Path(first.saved_path))
        self.assertEqual('# maintenance collect returncode=0\n' + self.executor.collect_output, first.output)
        second = self.make().collect('0')
        self.assertEqual(self.incident_dir() / 'check-0-incident-attempt-2.log',
                         Path(second.saved_path))
        self.assertEqual('# maintenance collect returncode=0\n' + self.executor.collect_output, Path(first.saved_path).read_text())
        self.make().collect('3')
        # Checks 2 and 6 load the devices, so they wait for the allocation to end.
        for check in ('2', '6'):
            with self.subTest(check=check):
                with self.assertRaisesRegex(session.Refusal, 'after your allocation ends'):
                    self.make().collect(check)
        with self.assertRaisesRegex(session.Refusal, r'Available: 0, 2, 3, 6\.'):
            self.make().collect('5')
        self.assertEqual([['collect', '0'], ['collect', '0'], ['collect', '3']],
                         self.executor.maintenance_calls)
        self.executor.jobs = []
        self.make().collect('2')
        state = self.make().store.read()
        self.assertEqual(('ready', [], None), (state.phase, state.collected, state.incident))
        self.assertEqual([], self.mutations())
        rendered = session.render(first, 'collect')
        self.assertIn(f'Saved: {first.saved_path}', rendered)
        self.assertIn('Next: Incident 1 is under way.', rendered)

    def test_kernel_evidence_uses_the_recorded_boot_and_job_start(self):
        self.issue()
        self.executor.maintenance_calls.clear()
        result = self.make().collect('kernel')
        self.assertEqual([['kernel-evidence', '--boot', 'aaaaaaaa-0000-0000-0000-000000000001',
                           '--since', '1793880600']], self.executor.maintenance_calls)
        self.assertEqual(self.incident_dir() / 'check-kernel-incident.log', Path(result.saved_path))
        self.assertIn('# bounded kernel attempt returncode=0; not full capture', result.output)
        self.assertIn('"action": "kernel-evidence"', result.output)
        # A target that cannot answer is evidence unavailable, saved as such.
        self.executor.maintenance_failures['kernel-evidence'] = (255, 'Connection closed\n')
        failed = self.make().collect('kernel')
        self.assertIn('returncode=255', failed.output)
        self.assertEqual(self.incident_dir() / 'check-kernel-incident-attempt-2.log',
                         Path(failed.saved_path))

    def test_collect_recovery_saves_the_query_and_its_reading(self):
        self.issue()
        self.executor.maintenance_calls.clear()
        result = self.make().collect('recovery')
        self.assertEqual([['gpu-recovery']], self.executor.maintenance_calls)
        self.assertEqual(self.incident_dir() / 'check-recovery-incident.log', Path(result.saved_path))
        saved = Path(result.saved_path).read_text()
        self.assertEqual(saved, result.output)
        self.assertIn('# gpu-recovery route returncode=0', saved)
        self.assertIn('# reading: none', saved)
        self.assertIn('# gpus: 8 of 8 expected; every assigned device reports None', saved)
        self.assertIn('# boot: aaaaaaaa-0000-0000-0000-000000000001 matches the boot recorded '
                      'when incident 1 was issued', saved)
        self.assertIn(self.GPU_HEADER, saved)
        self.assertIn(self.recovery_text().strip(), saved)
        self.assertEqual([], self.mutations())
        self.assertEqual('ready', self.make().store.read().phase)

    def readings(self, cases):
        results = {}
        for name, prepare in cases.items():
            self.tearDown()
            self.setUp()
            self.issue()
            prepare()
            text = self.make().collect('recovery').output
            [reading] = re.findall(r'^# reading: (\S+)', text, re.M)
            results[name] = (reading, text)
        return results

    def test_every_unclear_or_unreadable_answer_is_evidence_unavailable_not_none(self):
        bus = '00000000:5F:00.0'
        uuid = 'GPU-0000000f-0000-0000-0000-000000000000'

        def rows(**changes):
            def apply():
                for index, action in changes.items():
                    row = list(self.rows[int(index)])
                    row[3] = action
                    self.rows[int(index)] = tuple(row)
            return apply

        def setter(name, value):
            def apply():
                setattr(self, name, value)
            return apply

        def duplicate():
            self.rows[5] = (self.rows[5][0], self.rows[2][1], self.rows[5][2], 'None')

        def missing():
            del self.rows[7]

        def extra():
            self.rows.append(('8', uuid, bus, 'None'))

        results = self.readings({
            'none': lambda: None,
            'reset': rows(**{'0': 'GPU Reset Required'}),
            'reboot': rows(**{'3': 'Node Reboot Required'}),
            'drain': rows(**{'1': 'Drain P2P'}),
            'n/a': rows(**{'0': 'N/A'}),
            'bracketed n/a': rows(**{'2': '[N/A]'}),
            'blank': rows(**{'4': ''}),
            'empty output': setter('record_stdout', ''),
            'header only': setter('record_stdout', self.GPU_HEADER + '\n'),
            'wrong header': setter('record_stdout', 'index, uuid\n0, GPU-1\n'),
            'nonzero query': setter('query_rc', 9),
            'nonzero route': setter('route_rc', 3),
            'no record': setter('record_text', 'nvidia-smi: not found\n'),
            'duplicate gpu': duplicate,
            'missing gpu': missing,
            'extra gpu': extra,
            'wrong instance': setter('record_instance', 'i-0fedcba9876543210'),
            'wrong boot': setter('record_boot', 'bbbbbbbb-0000-0000-0000-000000000002'),
        })
        expected = {'none': 'none', 'reset': 'action-required', 'reboot': 'action-required',
                    'drain': 'action-required'}
        for name, (reading, text) in results.items():
            with self.subTest(case=name):
                self.assertEqual(expected.get(name, 'unavailable'), reading, text)
                self.assertIn('# gpu-recovery route returncode=', text)
                if name not in ('no record',):
                    self.assertIn('"action": "gpu-recovery"', text)
        self.assertIn('GPU Reset Required', results['reset'][1])
        self.assertIn('does not match the boot recorded', results['wrong boot'][1])
        self.assertIn('identity', results['wrong instance'][1])
        self.assertIn('7 of 8 expected', results['missing gpu'][1])
        self.assertIn('9 of 8 expected', results['extra gpu'][1])

    def test_an_earlier_device_round_keeps_its_files_and_its_answer(self):
        self.set_schedule(session.validate_schedule(
            [{'exercise': 'efa-reset'}] + self.APPLICATION[:1]))
        self.make().next()
        self.running_job_with_progress()
        self.assertTrue(self.make().next().under_way)
        self.executor.jobs = []
        self.assertEqual('runtime-ready', self.make().recover().state.phase)
        self.make().collect('2')
        round_dir = self.answers / 'collected' / 'round-1'
        before = {p.name: p.read_bytes() for p in round_dir.iterdir()}
        self.assertIn('check-2-recovered.log', before)
        # The application incident is issued on the recovered node.
        self.running_allocation()
        self.assertEqual(2, self.make().next().incident)
        self.make().collect('0')
        self.make().collect('kernel')
        self.make().collect('recovery')
        self.assertEqual(before, {p.name: p.read_bytes() for p in round_dir.iterdir()})
        self.assertEqual(['check-0-incident.log', 'check-kernel-incident.log',
                          'check-recovery-incident.log'],
                         sorted(p.name for p in self.incident_dir(2).iterdir()))
        state = self.make().store.read()
        self.assertEqual(('runtime-ready', 1), (state.phase, state.incident['number']))
        rendered = session.render(self.make().status(), 'status')
        self.assertIn('Answer for incident 1', rendered)
        self.assertIn('Next: Incident 2 is under way. Collect your evidence', rendered)
        self.assertNotIn('Checks collected: 0', rendered)
        self.assertIn('Checks you can collect: 0, 2, 3, 6, kernel, recovery', rendered)
        # The device-round kernel capture still reads its own round's boot.
        kernel = [c for c in self.executor.maintenance_calls if c[0] == 'kernel-evidence'][-1]
        self.assertEqual('bbbbbbbb-0000-0000-0000-000000000002', kernel[2])

    def test_collect_is_refused_after_the_incident_ends(self):
        self.issue()
        self.make().ended(self.issue_id(1))
        for check in ('0', 'kernel', 'recovery'):
            with self.subTest(check=check):
                with self.assertRaisesRegex(session.Refusal, 'no exercise running'):
                    self.make().collect(check)

    def test_collect_recovery_needs_a_current_incident(self):
        with self.assertRaisesRegex(session.Refusal, 'no exercise running'):
            self.make().collect('recovery')
        self.make().store.write(session.State(phase='fault-applied', kind='gpu',
                                              started_at=1.0, target_node='gpu-g7-1'))
        with self.assertRaisesRegex(session.Refusal, 'listed by status'):
            self.make().collect('recovery')
        self.assertEqual([], self.executor.maintenance_calls)


class ApplicationIncidentResume(ApplicationIncidentBase):
    def test_missing_gpu_identity_or_unfinished_process_refuses_resume(self):
        def missing():
            self.executor.inspect.pop('gpu_uuid')
        cases = {
            'missing identity': missing,
            'compute process': lambda: setattr(self, 'process_stdout',
                'gpu_uuid, pid, process_name\n' + self.rows[0][1] + ', 123, python3\n'),
            'unreadable processes': lambda: setattr(self, 'process_rc', 9),
            'missing process table': lambda: setattr(self, 'process_stdout', ''),
            'old active step': lambda: setattr(self, 'live_steps', ['226.3']),
        }
        for name, (code, out, err) in self.refusals(cases).items():
            with self.subTest(case=name):
                self.assertEqual(3, code, out + err)
                self.assertIn('Observation:', err)

    def test_a_clear_node_may_resume_and_the_observation_is_saved(self):
        self.issue()
        self.executor.maintenance_calls.clear()
        code, out, err = self.run_main(f'resume {self.issue_id(1)}', now=1200.0)
        self.assertEqual((0, ''), (code, err))
        self.assertIn('Incident 1 may resume.', out)
        saved = self.incident_dir() / 'resume-check.log'
        self.assertIn(f'Observation: {saved}', out)
        text = saved.read_text()
        self.assertIn('# reading: none', text)
        self.assertIn('# resume: allowed', text)
        self.assertIn('"action": "gpu-recovery"', text)
        actions = [c[0] for c in self.executor.maintenance_calls]
        self.assertEqual(sorted(['instance-id', 'boot-id', 'inspect', 'gpu-recovery']),
                         sorted(actions))
        self.assertEqual([], self.mutations())
        entry = self.progress_record()['issued'][0]
        self.assertNotIn('ended_at', entry)
        self.assertEqual({'at': 1200.0, 'allowed': True, 'log': str(saved)}, entry['resume_check'])
        self.assert_no_kind(out)
        # ended still comes from the resumed job, and next then continues.
        self.make().ended(self.issue_id(1))
        self.assertIn('ended_at', self.progress_record()['issued'][0])
        self.assertEqual(2, self.make().next().incident)

    def refusals(self, cases):
        results = {}
        for name, prepare in cases.items():
            self.tearDown()
            self.setUp()
            self.issue()
            prepare()
            self.executor.maintenance_calls.clear()
            code, out, err = self.run_main(f'resume {self.issue_id(1)}', now=1200.0)
            results[name] = (code, out, err)
            self.assertEqual([], self.mutations())
            self.assertNotIn('ended_at', self.progress_record()['issued'][0])
        return results

    def test_every_unclear_reading_refuses_and_changes_nothing(self):
        def rows(index, action):
            def apply():
                row = list(self.rows[index])
                row[3] = action
                self.rows[index] = tuple(row)
            return apply

        def setter(name, value):
            def apply():
                setattr(self, name, value)
            return apply

        def rebooted():
            self.executor.boot_id = 'bbbbbbbb-0000-0000-0000-000000000002'

        def other_gpu():
            self.rows[0] = ('0', 'GPU-ffffffff-0000-0000-0000-000000000000', self.rows[0][2], 'None')

        results = self.refusals({
            'reset': rows(0, 'GPU Reset Required'),
            'reboot': rows(7, 'Node Reboot Required'),
            'n/a': rows(0, 'N/A'),
            'blank': rows(3, ''),
            'empty output': setter('record_stdout', ''),
            'nonzero query': setter('query_rc', 9),
            'nonzero route': setter('route_rc', 3),
            'duplicate gpu': lambda: self.rows.__setitem__(
                5, (self.rows[5][0], self.rows[2][1], self.rows[5][2], 'None')),
            'missing gpu': lambda: self.rows.__delitem__(7),
            'assigned gpu absent': other_gpu,
            'wrong instance in record': setter('record_instance', 'i-0fedcba9876543210'),
            'wrong boot in record': setter('record_boot', 'bbbbbbbb-0000-0000-0000-000000000002'),
            'node rebooted': rebooted,
            'allocation ended': lambda: setattr(self.executor, 'jobs', []),
            'another allocation': lambda: self.running_allocation('227'),
            'foreign job': lambda: setattr(self.executor, 'jobs',
                                           ['301|someone-else|aim344-x|RUNNING']),
        })
        for name, (code, out, err) in results.items():
            with self.subTest(case=name):
                self.assertEqual(3, code, (out, err))
                self.assertEqual('', out)
                self.assertIn('not ready to resume', err)
                self.assertIsNone(KIND_WORDS.search(err), err)
                self.assertNotIn('resume_check_allowed', err)
        self.assertIn('did not read None', results['reset'][2])
        self.assertIn('Observation:', results['reset'][2])
        entry = self.progress_record()['issued'][0]
        self.assertFalse(entry['resume_check']['allowed'])
        saved = Path(entry['resume_check']['log']).read_text()
        self.assertIn('# resume: refused', saved)

    def test_only_the_current_unended_application_incident_may_be_checked(self):
        self.issue()
        first = self.issue_id(1)
        code, _, err = self.run_main('resume ' + 'b' * 32)
        self.assertEqual(3, code)
        self.assertIn('No incident with that issue id', err)
        self.make().ended(first)
        code, _, err = self.run_main(f'resume {first}')
        self.assertEqual(3, code)
        self.assertIn('Incident 1 has ended', err)
        self.make().next()
        code, _, err = self.run_main(f'resume {first}')
        self.assertEqual(3, code)
        self.assertIn('Incident 1 has ended', err)
        # The storage incident now current has no pre-resume check of its own.
        code, _, err = self.run_main(f'resume {self.issue_id(2)}')
        self.assertEqual(3, code)
        self.assertIn('no pre-resume check', err)
        self.assertEqual([], [c for c in self.executor.maintenance_calls
                              if c[0] == 'gpu-recovery'])

    def test_a_held_device_round_refuses_the_check(self):
        self.issue()
        self.make().store.write(session.State(phase='fault-applied', kind='efa',
                                              started_at=1.0, target_node='gpu-g7-1'))
        with self.assertRaisesRegex(session.Refusal, 'holds your node'):
            self.make().resume(self.issue_id(1))
        self.assertEqual([], [c for c in self.executor.maintenance_calls
                              if c[0] == 'gpu-recovery'])

    def test_the_operator_view_shows_the_recorded_allocation_and_check(self):
        self.issue()
        self.make().resume(self.issue_id(1))
        view = session.operator_view(self.make())
        self.assertIn('1. cuda-assert {"assert_rank": 3, "assert_step": 201} issued', view)
        self.assertIn('job=226 ranks=0-7 boot=aaaaaaaa-0000-0000-0000-000000000001', view)
        self.assertIn('resume check: allowed', view)


if __name__ == '__main__':
    unittest.main()
