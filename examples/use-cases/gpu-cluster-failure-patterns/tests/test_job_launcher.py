# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""job.sh, the run-id and lab-log handling in common.sh, the numbered launchers'
argument checks, and the login profile, all run on the CPU.

job.sh runs with a recording common.sh, a scripted 12.device-exercise.sh and a
fake scontrol, so nothing reaches Slurm, a container, a GPU or the lab controls.
The tests check which commands the launcher issues and what the participant sees;
they are not evidence of any run on hardware.
"""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

LAB = Path(__file__).resolve().parents[1]
JOB = LAB / 'job.sh'
USER = subprocess.run(['id', '-un'], capture_output=True, text=True, check=True).stdout.strip()

# Words that would tell a participant the cause of an incident before the answer.
CAUSE_WORDS = re.compile(r'storage|writer|inject|filler|plugin|slow|efa|reset|flr|'
                         r'dataloader|corrupt|exercise|profile', re.IGNORECASE)

FAKE_COMMON = r'''
note() { printf '%s\n' "$*" >> "$TEST_CALLS"; }
prepare_slurm() { note "prepare_slurm RESULTS_DIR=$RESULTS_DIR"; }
prepare_torch() { note "prepare_torch"; }
node_command() {
    note "node_command $*"
    case "$*" in
        *plugin-state.sh*) printf 'host=fake plugin_action=%s\n' "${*: -1}" ;;
        *recover-checkpoint.py*) echo 'Removed lab filler and partial write; retained last.json.' ;;
    esac
}
run_torch() {
    local phase=$1
    shift
    note "run_torch phase=$phase run_id=$AIM344_RUN_ID copy=$AIM344_PROGRESS_COPY lab=$AIM344_LAB_LOG iface=${AIM344_EFA_IFACE:-unset} args=$*"
    note "job_env settings=$(printenv AIM344_INCIDENT_SETTINGS || echo unset) limit=${AIM344_JOB_LIMIT_SECONDS:-default}"
    if [[ -n ${TEST_PROGRESS:-} ]]; then
        for n in 20 40 60; do
            printf 'AIM344 collective_progress=%s collectives; tensor_bytes=134217728 B; timestamp=%s s since epoch; 0 mismatches; job_id=%s; world_size=16; rank=0.\n' \
                "$n" "$(date +%s)" "$SLURM_JOB_ID" >> "$HOME/aim344-results/device-$SLURM_JOB_ID.log"
        done
    fi
    local until=$((SECONDS + ${TEST_RUN_SECONDS:-0}))
    while ((SECONDS < until)); do
        if [[ -n ${TEST_UNTIL:-} ]] && grep -q -- "$TEST_UNTIL" "$TEST_NEXT_LOG" 2>/dev/null; then break; fi
        sleep 0.2
    done
    printf 'launcher_exit_code=%s (dimensionless)\n' "${TEST_RC:-0}"
    return "${TEST_RC:-0}"
}
run_sweep() { note "run_sweep $1 run_id=$AIM344_RUN_ID lab=$AIM344_LAB_LOG iface=${AIM344_EFA_IFACE:-unset}"; }
'''

# Replies are read from $TEST_NEXT_DIR/<call>.out and <call>.rc, else default.*.
FAKE_CONTROLLER = r'''#!/bin/bash
n=$(( $(cat "$TEST_NEXT_LOG" 2>/dev/null | wc -l) + 1 ))
reply=$TEST_NEXT_DIR/$n.out; [[ -f $reply ]] || reply=$TEST_NEXT_DIR/default.out
rc=$TEST_NEXT_DIR/$n.rc; [[ -f $rc ]] || rc=$TEST_NEXT_DIR/default.rc
printf '%s %s: %s\n' "$n" "$*" "$(head -n 1 "$reply")" >> "$TEST_NEXT_LOG"
if [[ $(cat "$rc" 2>/dev/null || echo 0) == 0 ]]; then cat "$reply"; else cat "$reply" >&2; fi
exit "$(cat "$rc" 2>/dev/null || echo 0)"
'''

FAKE_SCONTROL = r'''#!/bin/bash
[[ $1 == show && $2 == job ]] || exit 1
printf 'JobId=%s JobName=%s UserId=x(1) JobState=RUNNING TimeLimit=%s NumNodes=2\n' \
    "$3" "${TEST_JOB_NAME:-aim344-job}" "${TEST_TIME_LIMIT:-00:09:00}"
'''


class Launcher(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.lab = root / 'lab'
        self.lab.mkdir()
        shutil.copy(JOB, self.lab / 'job.sh')
        (self.lab / 'common.sh').write_text(FAKE_COMMON)
        (self.lab / '12.device-exercise.sh').write_text(FAKE_CONTROLLER)
        bindir = root / 'bin'
        bindir.mkdir()
        (bindir / 'scontrol').write_text(FAKE_SCONTROL)
        (bindir / 'scontrol').chmod(0o755)
        self.home = root / 'home'
        self.incidents = self.home / 'aim344-results' / 'incidents'
        self.incidents.mkdir(parents=True)
        self.replies = root / 'replies'
        self.replies.mkdir()
        self.calls = root / 'calls.log'
        self.next_log = root / 'next.log'
        self.results = root / 'results'
        self.env = {
            'PATH': f'{bindir}:{os.environ["PATH"]}', 'HOME': str(self.home),
            'SLURM_JOB_ID': '4242', 'AIM344_EFA_DEVICE': 'rdmap83s0',
            'RESULTS_DIR': str(self.results), 'TEST_CALLS': str(self.calls),
            'TEST_NEXT_LOG': str(self.next_log), 'TEST_NEXT_DIR': str(self.replies),
            'LANG': 'C.UTF-8',
        }
        self.reply('default', 'Incident 9 is ready.\nNext: unused\n')

    def tearDown(self):
        self.tmp.cleanup()

    def reply(self, call, text, rc=0):
        (self.replies / f'{call}.out').write_text(text)
        (self.replies / f'{call}.rc').write_text(str(rc))

    def profile(self, number, exercise, settings=None, issued_at=100.0, name=None):
        path = self.incidents / (name or f'incident-{number}.json')
        path.write_text(json.dumps({
            'incident': number, 'incidents': 4, 'exercise': exercise,
            'settings': settings or {}, 'issued_at': issued_at,
            'controller_step': 'next-after-progress' if exercise == 'efa-reset' else 'none'}))
        return path

    def issue(self, number, exercise, settings=None):
        path = self.profile(number, exercise, settings)
        self.reply(1, f'Incident {number} is ready.\nProfile: {path}\nNext: Your launcher runs '
                      'this incident from the profile above.\n')
        return path

    def run_job(self, *argv, timeout=60, **env):
        started = time.monotonic()
        result = subprocess.run(['bash', str(self.lab / 'job.sh'), *argv], capture_output=True,
                                text=True, timeout=timeout, env={**self.env, **env})
        result.seconds = time.monotonic() - started
        return result

    def lines(self, path):
        return path.read_text().splitlines() if path.exists() else []

    def torch_calls(self):
        return [line for line in self.lines(self.calls) if line.startswith('run_torch')]

    def job_env(self):
        return [line for line in self.lines(self.calls) if line.startswith('job_env')]

    def next_calls(self):
        return self.lines(self.next_log)

    def assert_no_cause(self, result):
        # The temporary directory's random name is not participant output.
        seen = (result.stdout + result.stderr).replace(self.tmp.name, '<tmp>')
        self.assertIsNone(CAUSE_WORDS.search(seen), seen)

    # ---------------------------------------------------------------- grammar
    def test_usage_and_unknown_arguments_fail(self):
        for argv in ((), ('incident', 'extra'), ('fix-storage',), ('run', '--writer-rank', '9'),
                     ('--help',), ('notes', 'x'), ('resume', '--reinit-in-worker'),
                     ('run', '--dataset-cache', 'worker-thread'), ('rerun', '--writer-rank=9'),
                     ('rerun', '--dataset-cache', 'lock'), ('rerun', '--dataset-cache'),
                     ('rerun', '--dataloader-timeout', 'x'), ('rerun', '--dataloader-timeout=-1'),
                     ('rerun', '--worker-cuda'), ('rerun', '--reinit-in-worker', 'extra')):
            with self.subTest(argv=argv):
                result = self.run_job(*argv)
                self.assertEqual(2, result.returncode)
                self.assertIn('Usage: bash job.sh <command>', result.stderr)
                self.assertEqual([], self.next_calls())
        usage = JOB.read_text()
        usage = usage[usage.index("<<'USAGE'"):usage.index('\nUSAGE\n')]
        self.assertIsNone(re.search(r'storage|inject|plugin|reset|corrupt|dataloader', usage, re.I))

    # ---------------------------------------------------------------- notes
    def test_notes_creates_the_template_once(self):
        env = {'SLURM_JOB_ID': ''}
        result = self.run_job('notes', **env)
        notes = self.home / 'aim344-results' / 'notes.md'
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(f'Created your notes file: {notes}\n', result.stdout)
        text = notes.read_text()
        self.assertTrue(text.startswith('# Your cluster\n'))
        for heading in ('# Triage table', '# Healthy values', '# Incident 1', '# Incident 2',
                        '# Incident 3', '# Incident 4', '# Incident 5', '# Loss table',
                        '## Card A', '## Card B'):
            self.assertIn('\n' + heading + '\n', '\n' + text)
        for heading in ('# Card answers', '## Card C', '# Incident 6'):
            self.assertNotIn('\n' + heading + '\n', '\n' + text)
        # A case of three questions after each incident page and the cost page.
        self.assertEqual(6, text.count('\n## Case\n\n1.\n\n2.\n\n3.\n'))
        # The device incident's record table sits under Incident 5.
        self.assertLess(text.index('# Incident 5'), text.index('| Check 6 | | |'))
        self.assertLess(text.index('| Check 6 | | |'), text.index('# Loss table'))
        self.assertIn('| What you see | Who reported | First record to read | Check that separates '
                      'the causes | Kind of recovery | Your healthy value |', text)
        for row in ('Nobody times out, but the job is slow', 'One rank does not report',
                    'Every rank reports at once', 'Progress stops and nothing prints',
                    'Everything passes, but an answer differs'):
            self.assertIn(f'| {row} | | | | | |', text)
        for row in ('Last progress line', 'Watchdog lines', 'Kernel log', 'Check 0', 'Check 6'):
            self.assertIn(f'| {row} | | |', text)
        for row in ('| t0 | Last progress line | |', '| Detection | t0 | t1 | |',
                    '| Rework | The last checkpoint | The stop | |', '| Incident 3 | | | |',
                    '| Incident 4 | | | |'):
            self.assertIn(row, text)
        self.assertNotIn('| Incident 2 | | | |', text)
        self.assertEqual(4, text.count('Transfer question:'))
        notes.write_text('mine\n')
        again = self.run_job('notes', **env)
        self.assertEqual(0, again.returncode)
        self.assertIn('already exists and was left as it is', again.stdout)
        self.assertEqual('mine\n', notes.read_text())
        self.assertEqual([], self.lines(self.calls))

    # ---------------------------------------------------------------- allocation rules
    def test_job_commands_need_the_pages_allocation_and_the_login_profile(self):
        cases = (({'SLURM_JOB_ID': ''}, 'inside your allocation'),
                 ({'TEST_JOB_NAME': 'interactive'}, 'job name starts with aim344-'),
                 ({'TEST_TIME_LIMIT': '00:20:00'}, '1 to 10 minutes'),
                 ({'TEST_TIME_LIMIT': '00:00:30'}, '1 to 10 minutes'),
                 ({'AIM344_EFA_DEVICE': ''}, 'AIM344_EFA_DEVICE is not set'),
                 ({'AIM344_EFA_DEVICE': 'x;y'}, 'AIM344_EFA_DEVICE is not set'))
        for verb in ('run', 'incident', 'rerun', 'resume'):
            for env, message in cases:
                with self.subTest(verb=verb, env=env):
                    result = self.run_job(verb, **env)
                    self.assertEqual(3, result.returncode)
                    self.assertIn(message, result.stderr)
                    self.assertEqual([], self.next_calls())
                    self.assertEqual([], self.lines(self.calls))

    # ---------------------------------------------------------------- run
    def test_run_starts_the_healthy_job_and_never_asks_for_an_incident(self):
        result = self.run_job('run')
        self.assertEqual(0, result.returncode, result.stderr)
        [call] = self.torch_calls()
        self.assertRegex(call, r'^run_torch phase=job run_id=job-\d{8}T\d{6}Z copy=1 lab= '
                               r'iface=rdmap83s0 args=device --checkpoint-every=200 --replica-check '
                               r'--writer-rank=0 --duration-seconds=60$')
        self.assertEqual(['job_env settings=unset limit=default'], self.job_env())
        self.assertEqual([], self.next_calls())
        run_id = call.split('run_id=')[1].split()[0]
        self.assertEqual(2, result.stdout.count(f'Job log: {self.results}/{run_id}.log\n'))
        self.assertRegex(result.stdout, r'job_end_utc=\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ\n')
        self.assert_no_cause(result)

    def test_results_default_to_node_local_storage(self):
        env = dict(self.env)
        del env['RESULTS_DIR']
        result = subprocess.run(['bash', str(self.lab / 'job.sh'), 'run'], capture_output=True,
                                text=True, timeout=60, env=env)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn(f'prepare_slurm RESULTS_DIR=/var/tmp/aim344-{USER}-4242', self.lines(self.calls))

    # ---------------------------------------------------------------- storage incident
    def test_storage_incident_asks_once_and_passes_its_settings_silently(self):
        profile = self.issue(1, 'storage', {'writer_rank': 9})
        result = self.run_job('incident', TEST_RC='1')
        self.assertEqual(1, result.returncode)
        self.assertEqual(1, len(self.next_calls()))
        [call] = self.torch_calls()
        lab = self.incidents / 'incident-1-lab.log'
        self.assertRegex(call, rf'^run_torch phase=job run_id=incident-1-\d{{8}}T\d{{6}}Z copy=1 '
                               rf'lab={re.escape(str(lab))} iface=rdmap83s0 args=device '
                               r'--checkpoint-every=200 --replica-check --mark-lab-lines --writer-rank=9 '
                               r'--duration-seconds=60 --inject$')
        self.assertEqual(['job_env settings=unset limit=default'], self.job_env())
        # The controller's reply, which names the profile, stays in the lab log.
        self.assertIn(f'Profile: {profile}', lab.read_text())
        self.assertIn('Starting incident 1.', result.stdout)
        self.assertIn('launcher_exit_code=1 (dimensionless)', result.stdout)
        self.assert_no_cause(result)

    def test_a_refused_incident_prints_the_refusal_and_runs_nothing(self):
        self.reply(1, 'All 4 incidents for this seat have been issued; there is no next incident.\n', rc=3)
        result = self.run_job('incident')
        self.assertEqual(3, result.returncode)
        self.assertIn('there is no next incident', result.stderr)
        self.assertEqual([], self.torch_calls())

    def test_a_reply_without_a_profile_runs_nothing(self):
        self.reply(1, 'Incident 1 is ready.\n')
        result = self.run_job('incident')
        self.assertEqual(3, result.returncode)
        self.assertIn('handed over no incident profile', result.stderr)
        self.assertEqual([], self.torch_calls())

    def test_storage_rerun_resumes_on_the_full_directory_without_a_new_fill(self):
        self.profile(1, 'storage', {'writer_rank': 9})
        result = self.run_job('rerun')
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual([], self.next_calls())
        [call] = self.torch_calls()
        self.assertIn('run_id=incident-1-rerun-', call)
        self.assertTrue(call.endswith('--writer-rank=9 --duration-seconds=60 --resume'), call)
        self.assertNotIn('--inject', call)

    def test_storage_resume_frees_the_space_in_the_lab_log_then_resumes(self):
        self.profile(1, 'storage', {'writer_rank': 9})
        result = self.run_job('resume')
        self.assertEqual(0, result.returncode, result.stderr)
        calls = self.lines(self.calls)
        recovery = calls.index('node_command python3 /opt/aim344/recover-checkpoint.py')
        self.assertEqual('prepare_torch', calls[recovery - 1])
        [call] = self.torch_calls()
        self.assertGreater(calls.index(call), recovery)
        self.assertIn('run_id=incident-1-resume-', call)
        self.assertTrue(call.endswith('--writer-rank=9 --duration-seconds=10 --resume'), call)
        self.assertIn('Removed lab filler', (self.incidents / 'incident-1-lab.log').read_text())
        self.assert_no_cause(result)
        self.assertEqual([], self.next_calls())

    # ---------------------------------------------------------------- slow job incident
    def test_slow_job_incident_keeps_the_lab_step_out_of_the_participant_output(self):
        self.issue(2, 'slow-job')
        result = self.run_job('incident')
        self.assertEqual(0, result.returncode, result.stderr)
        calls = self.lines(self.calls)
        lab = self.incidents / 'incident-2-lab.log'
        self.assertIn('node_command bash /opt/aim344/plugin-state.sh hide', calls)
        sweep = [line for line in calls if line.startswith('run_sweep')]
        self.assertEqual(1, len(sweep))
        self.assertRegex(sweep[0], rf'^run_sweep job run_id=incident-2-\d{{8}}T\d{{6}}Z '
                                   rf'lab={re.escape(str(lab))} iface=unset$')
        self.assertIn('plugin_action=hide', lab.read_text())
        self.assert_no_cause(result)
        self.assertEqual(1, len(self.next_calls()))

    def test_slow_job_rerun_keeps_the_fault_and_resume_restores_first(self):
        self.profile(2, 'slow-job')
        self.assertEqual(0, self.run_job('rerun').returncode)
        self.assertNotIn('plugin-state', ' '.join(self.lines(self.calls)))
        self.calls.unlink()
        result = self.run_job('resume')
        self.assertEqual(0, result.returncode, result.stderr)
        calls = self.lines(self.calls)
        self.assertLess(calls.index('node_command bash /opt/aim344/plugin-state.sh restore'),
                        [i for i, line in enumerate(calls) if line.startswith('run_sweep')][0])
        self.assertIn('plugin_action=restore', (self.incidents / 'incident-2-lab.log').read_text())
        self.assert_no_cause(result)
        self.assertEqual([], self.next_calls())

    def test_a_failed_lab_step_stops_with_a_neutral_message(self):
        self.issue(2, 'slow-job')
        (self.lab / 'common.sh').write_text(FAKE_COMMON + '\nnode_command() { echo "Unexpected plugin state." >&2; return 1; }\n')
        result = self.run_job('incident')
        self.assertEqual(3, result.returncode)
        self.assertIn('could not prepare this step', result.stderr)
        self.assertIn('Unexpected plugin state.', (self.incidents / 'incident-2-lab.log').read_text())
        self.assert_no_cause(result)

    # ---------------------------------------------------------------- controller incident
    def test_controller_incident_asks_again_after_two_progress_records_until_under_way(self):
        self.issue(4, 'efa-reset')
        self.reply(2, 'Incident 4 is ready.\nProfile: x\n')
        self.reply(3, 'Incident 4 is under way.\nNext: Watch your job.\n')
        result = self.run_job('incident', TEST_PROGRESS='1', TEST_RUN_SECONDS='20',
                              TEST_UNTIL='under way')
        self.assertEqual(0, result.returncode, result.stderr)
        calls = self.next_calls()
        self.assertEqual(3, len(calls), calls)
        self.assertTrue(all(' next: ' in line for line in calls), calls)
        [call] = self.torch_calls()
        self.assertRegex(call, r'^run_torch phase=device run_id=incident-4-\d{8}T\d{6}Z copy=1 .* '
                               r'args=device --checkpoint-every=200 --replica-check --mark-lab-lines '
                               r'--writer-rank=0 --duration-seconds=120$')
        lab = (self.incidents / 'incident-4-lab.log').read_text()
        self.assertIn('next, attempt 2, exit 0: Incident 4 is under way.', lab)
        self.assert_no_cause(result)
        self.assertNotIn('under way', result.stdout + result.stderr)
        self.assertEqual([], list(self.incidents.glob('.job-ended-*')))

    def test_controller_step_stops_at_the_first_refusal(self):
        self.issue(4, 'efa-reset')
        self.reply(2, 'Your job is not running on your node; nothing was changed.\n', rc=3)
        result = self.run_job('incident', TEST_PROGRESS='1', TEST_RUN_SECONDS='6')
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(2, len(self.next_calls()))
        self.assertIn('Your job is not running on your node; nothing was changed.', result.stderr)

    def test_controller_step_waits_for_progress_and_ends_with_the_job(self):
        self.issue(4, 'efa-reset')
        result = self.run_job('incident')
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(1, len(self.next_calls()))
        self.assertIn('The job ended before the lab controls could act', result.stderr)

    def test_controller_step_is_bounded_by_the_run(self):
        self.issue(4, 'efa-reset')
        self.reply('default', 'Incident 4 is ready.\nProfile: x\n')
        result = self.run_job('incident', TEST_PROGRESS='1', TEST_RUN_SECONDS='4')
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertLessEqual(len(self.next_calls()), 3)
        self.assertIn('The lab controls did not act during this run', result.stderr)
        self.assertLess(result.seconds, 20)

    def test_an_earlier_run_in_the_allocation_does_not_count_as_progress(self):
        self.issue(4, 'efa-reset')
        copy = self.home / 'aim344-results' / 'device-4242.log'
        copy.write_text(''.join(
            f'AIM344 collective_progress={n} collectives; tensor_bytes=1 B; timestamp=1 s since '
            f'epoch; 0 mismatches; job_id=4242; world_size=16; rank=0.\n' for n in (20, 40)))
        result = self.run_job('incident')
        self.assertEqual(1, len(self.next_calls()))
        self.assertIn('The job ended before the lab controls could act', result.stderr)

    def test_controller_rerun_and_resume_never_ask_the_lab_controls(self):
        self.profile(1, 'storage', {'writer_rank': 9}, issued_at=100.0)
        self.profile(4, 'efa-reset', issued_at=400.0)
        result = self.run_job('rerun', TEST_PROGRESS='1')
        self.assertEqual(0, result.returncode, result.stderr)
        [call] = self.torch_calls()
        self.assertIn('phase=device run_id=incident-4-rerun-', call)
        resume = self.run_job('resume')
        self.assertEqual(3, resume.returncode)
        self.assertIn('Incident 4 is recovered with the command on your page', resume.stderr)
        self.assertEqual([], self.next_calls())
        self.assert_no_cause(resume)

    def test_a_refusal_starts_no_container_and_incident_prepares_before_asking(self):
        for number, exercise, settings, argv in (
                (5, 'efa-reset', {}, ('resume',)),
                (1, 'slow-job', {}, ('rerun', '--dataset-cache', 'worker-thread')),
                (2, 'storage', {'writer_rank': 9}, ('rerun', '--reinit-in-worker')),
                (4, 'silent-corruption', {'corrupt_rank': 13}, ('rerun', '--dataloader-timeout', '60')),
                (5, 'efa-reset', {}, ('rerun', '--dataset-cache=worker-thread'))):
            with self.subTest(exercise=exercise, argv=argv):
                for path in self.incidents.glob('incident-*.json'):
                    path.unlink()
                self.profile(number, exercise, settings)
                result = self.run_job(*argv)
                self.assertEqual(3, result.returncode)
                self.assertEqual([], self.lines(self.calls))
                if argv[0] == 'rerun':
                    self.assertIn(f'Incident {number} takes no options. Run it again with: '
                                  'bash job.sh rerun', result.stderr)
                self.assert_no_cause(result)
        self.issue(1, 'storage', {'writer_rank': 9})
        self.reply(1, 'Incident 4 still holds your node. Collect your evidence and finish it '
                      'with recover; next continues after that.\n', rc=3)
        result = self.run_job('incident')
        self.assertEqual(3, result.returncode)
        self.assertEqual([f'prepare_slurm RESULTS_DIR={self.results}'], self.lines(self.calls))
        self.assertIn('still holds your node', result.stderr)

    # ---------------------------------------------------------------- DataLoader incident
    LOADER = ('args=dataloader --start-method fork --epochs 2 --stack-dump-seconds=60 '
              '--mark-lab-lines')

    def test_dataloader_incident_runs_the_rank_thread_job_without_naming_it(self):
        self.issue(3, 'dataloader')
        result = self.run_job('incident', TEST_RC='124')
        self.assertEqual(124, result.returncode)
        self.assertEqual(1, len(self.next_calls()))
        [call] = self.torch_calls()
        lab = self.incidents / 'incident-3-lab.log'
        self.assertRegex(call, rf'^run_torch phase=job run_id=incident-3-\d{{8}}T\d{{6}}Z copy=1 '
                               rf'lab={re.escape(str(lab))} iface=rdmap83s0 ')
        self.assertTrue(call.endswith(self.LOADER), call)
        # The setting reaches the job's environment, and the launcher ends it at 120 s.
        self.assertEqual(['job_env settings={"dataset_cache":"rank-thread"} limit=120'],
                         self.job_env())
        self.assertIn('Starting incident 3.', result.stdout)
        self.assertNotIn('rank-thread', result.stdout + result.stderr)
        self.assert_no_cause(result)

    def test_dataloader_rerun_passes_each_fix_and_keeps_the_dataset(self):
        self.profile(3, 'dataloader')
        for argv, forwarded in ((('--dataset-cache', 'worker-thread'), '--dataset-cache=worker-thread'),
                                (('--dataset-cache=worker-thread',), '--dataset-cache=worker-thread'),
                                (('--reinit-in-worker',), '--reinit-in-worker'),
                                (('--dataloader-timeout', '60'), '--dataloader-timeout=60'),
                                (('--dataloader-timeout=2.5',), '--dataloader-timeout=2.5'),
                                ((), '')):
            with self.subTest(argv=argv):
                if self.calls.exists():
                    self.calls.unlink()
                result = self.run_job('rerun', *argv)
                self.assertEqual(0, result.returncode, result.stderr)
                [call] = self.torch_calls()
                self.assertIn('run_id=incident-3-rerun-', call)
                self.assertTrue(call.endswith((self.LOADER + ' ' + forwarded).rstrip()), call)
                self.assertEqual(['job_env settings={"dataset_cache":"rank-thread"} limit=120'],
                                 self.job_env())
        self.assertEqual([], self.next_calls())

    def test_dataloader_resume_runs_the_job_without_the_lab_setting(self):
        self.profile(3, 'dataloader', {'affected_rank': 5})
        result = self.run_job('resume')
        self.assertEqual(0, result.returncode, result.stderr)
        [call] = self.torch_calls()
        self.assertIn('run_id=incident-3-resume-', call)
        self.assertTrue(call.endswith(self.LOADER), call)
        self.assertEqual(['job_env settings=unset limit=120'], self.job_env())
        self.assertEqual([], self.next_calls())
        self.assert_no_cause(result)

    def test_dataloader_settings_from_the_profile_are_passed_as_given(self):
        self.issue(3, 'dataloader', {'affected_rank': 5, 'worker_cuda': True})
        self.assertEqual(0, self.run_job('incident').returncode)
        self.assertEqual(['job_env settings={"affected_rank":5,"dataset_cache":"rank-thread",'
                          '"worker_cuda":true} limit=120'], self.job_env())

    # ---------------------------------------------------------------- silent-corruption incident
    FLIP = {'corrupt_rank': 13, 'corrupt_step': 37, 'corrupt_bit': 22, 'corrupt_recurring': False}

    def test_silent_corruption_incident_passes_the_seats_flip_in_the_environment(self):
        self.issue(4, 'silent-corruption', self.FLIP)
        result = self.run_job('incident')
        self.assertEqual(0, result.returncode, result.stderr)
        [call] = self.torch_calls()
        self.assertRegex(call, r'^run_torch phase=job run_id=incident-4-\d{8}T\d{6}Z copy=1 .* '
                               r'args=device --checkpoint-every=200 --replica-check --mark-lab-lines '
                               r'--writer-rank=0 --duration-seconds=20$')
        self.assertEqual(['job_env settings={"corrupt_bit":22,"corrupt_rank":13,'
                          '"corrupt_recurring":false,"corrupt_step":37} limit=default'],
                         self.job_env())
        self.assert_no_cause(result)

    def test_silent_corruption_rerun_repeats_only_a_recurring_flip(self):
        for recurring, expected in ((False, 'unset'),
                                    (True, '{"corrupt_bit":22,"corrupt_rank":13,'
                                           '"corrupt_recurring":true,"corrupt_step":37}')):
            with self.subTest(recurring=recurring):
                if self.calls.exists():
                    self.calls.unlink()
                self.profile(4, 'silent-corruption', {**self.FLIP, 'corrupt_recurring': recurring})
                result = self.run_job('rerun')
                self.assertEqual(0, result.returncode, result.stderr)
                [call] = self.torch_calls()
                self.assertIn('run_id=incident-4-rerun-', call)
                self.assertTrue(call.endswith('--writer-rank=0 --duration-seconds=20'), call)
                self.assertEqual([f'job_env settings={expected} limit=default'], self.job_env())

    def test_silent_corruption_resume_resumes_the_stand_in_without_the_flip(self):
        self.profile(4, 'silent-corruption', {**self.FLIP, 'corrupt_recurring': True})
        result = self.run_job('resume')
        self.assertEqual(0, result.returncode, result.stderr)
        [call] = self.torch_calls()
        self.assertIn('run_id=incident-4-resume-', call)
        self.assertTrue(call.endswith('args=device --checkpoint-every=200 --replica-check '
                                      '--mark-lab-lines --writer-rank=0 --duration-seconds=10 '
                                      '--resume'), call)
        self.assertEqual(['job_env settings=unset limit=default'], self.job_env())
        self.assertEqual([], self.next_calls())
        self.assert_no_cause(result)

    def test_a_settings_value_from_the_shell_never_reaches_another_job(self):
        stray = {'AIM344_INCIDENT_SETTINGS': '{"corrupt_rank": 1}'}
        self.assertEqual(0, self.run_job('run', **stray).returncode)
        self.profile(2, 'storage', {'writer_rank': 9})
        self.assertEqual(0, self.run_job('rerun', **stray).returncode)
        self.profile(3, 'dataloader', issued_at=200.0)
        self.assertEqual(0, self.run_job('resume', **stray).returncode)
        self.assertEqual(['job_env settings=unset limit=default'] * 2
                         + ['job_env settings=unset limit=120'], self.job_env())

    def test_an_unknown_exercise_is_answered_from_the_page(self):
        self.issue(6, 'future-exercise')
        result = self.run_job('incident')
        self.assertEqual(3, result.returncode)
        self.assertIn('Incident 6 cannot run live with this version of the lab', result.stderr)
        self.assertEqual([], self.torch_calls())

    # ---------------------------------------------------------------- bad profiles

    def test_a_profile_the_launcher_cannot_apply_is_refused(self):
        for exercise, settings in (('storage', {'writer_rank': 9, 'extra': 1}),
                                   ('storage', {'writer_rank': True}),
                                   ('storage', {'writer_rank': -1}),
                                   ('storage', {'writer_rank': '9'}),
                                   ('silent-corruption', {}),
                                   ('silent-corruption', {'corrupt_step': 37}),
                                   ('silent-corruption', {'corrupt_rank': True}),
                                   ('silent-corruption', {'corrupt_rank': 13, 'corrupt_bit': 32}),
                                   ('silent-corruption', {'corrupt_rank': 13, 'corrupt_step': 0}),
                                   ('silent-corruption', {'corrupt_rank': 13, 'corrupt_recurring': 1}),
                                   ('silent-corruption', {'corrupt_rank': 13, 'writer_rank': 1}),
                                   ('dataloader', {'dataset_cache': 'lock'}),
                                   ('dataloader', {'worker_cuda': 1}),
                                   ('dataloader', {'corrupt_rank': 1}),
                                   ('dataloader', {'dataloader_timeout': 60}),
                                   ('slow-job', {'writer_rank': 9})):
            with self.subTest(exercise=exercise, settings=settings):
                self.issue(1, exercise, settings)
                if self.next_log.exists():
                    self.next_log.unlink()
                result = self.run_job('incident')
                self.assertEqual(3, result.returncode)
                self.assertIn('cannot be used by this launcher', result.stderr)
                self.assertNotIn('Traceback', result.stderr)
                self.assertEqual([], self.torch_calls())

    def test_a_linked_profile_is_refused(self):
        real = self.profile(1, 'storage', {'writer_rank': 9}, name='elsewhere.json')
        link = self.incidents / 'incident-1.json'
        link.symlink_to(real)
        self.reply(1, f'Incident 1 is ready.\nProfile: {link}\n')
        result = self.run_job('incident')
        self.assertEqual(3, result.returncode)
        self.assertIn('cannot be used by this launcher', result.stderr)

    def test_rerun_needs_an_incident(self):
        for verb in ('rerun', 'resume'):
            result = self.run_job(verb)
            self.assertEqual(3, result.returncode)
            self.assertIn('No incident has started on your seat yet', result.stderr)
            self.assertEqual([], self.torch_calls())

    def test_rerun_uses_the_incident_issued_last(self):
        self.profile(1, 'storage', {'writer_rank': 9}, issued_at=100.0)
        self.profile(2, 'slow-job', issued_at=200.0, name='incident-2-attempt-2.json')
        self.profile(2, 'slow-job', issued_at=150.0)
        result = self.run_job('rerun')
        self.assertEqual(0, result.returncode, result.stderr)
        sweep = [line for line in self.lines(self.calls) if line.startswith('run_sweep')]
        self.assertEqual(1, len(sweep))
        self.assertIn(f'lab={self.incidents}/incident-2-attempt-2-lab.log', sweep[0])


class CommonRunIds(unittest.TestCase):
    """run_torch and run_sweep from the real common.sh, with a fake srun."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'pins.env').write_text('')
        shutil.copy(LAB / 'lab-lines.py', self.root / 'lab-lines.py')
        (self.root / 'summarize.py').write_text('print("summary")\n')
        srun = self.root / 'srun'
        srun.write_text('#!/bin/bash\n'
                        'case "$*" in\n'
                        '  *inventory.sh*) echo "OFI plugin library directory is absent from its normal search path." ;;\n'
                        '  *efa-counters.py*) echo "counters $*" ;;\n'
                        '  *sweep-rank.sh*) echo "NCCL INFO Using network Socket" ;;\n'
                        '  *torch-node.sh*) echo "AIM344 collective_progress=20 collectives";'
                        ' echo "AIM344 lab: checkpoint injection: full"; echo "checkpoint write retry: full" ;;\n'
                        'esac\n'
                        'exit "${TEST_RC:-0}"\n')
        srun.chmod(0o755)
        self.home = self.root / 'home'
        self.home.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def run_common(self, body, **env):
        script = ('set -euo pipefail\nLAB_DIR=$TEST_DIR\nsource "$TEST_COMMON"\n'
                  'prepare_torch() { :; }\nRESULTS_DIR=$TEST_DIR\nAIM344_CPUS_PER_NODE=1\n'
                  'GPUS_PER_NODE=1\nCONTAINER_ARGS=()\n' + body)
        return subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=30,
                              env={**os.environ, 'PATH': f'{self.root}:{os.environ["PATH"]}',
                                   'TEST_DIR': str(self.root), 'TEST_COMMON': str(LAB / 'common.sh'),
                                   'HOME': str(self.home), 'SLURM_JOB_ID': '4242', **env})

    def test_job_sh_run_ids_lab_lines_and_progress_copy(self):
        lab = self.root / 'incident-1-lab.log'
        for rc in (0, 1):
            with self.subTest(rc=rc):
                result = self.run_common(
                    f'AIM344_RUN_ID=incident-1-T AIM344_LAB_LOG={lab} AIM344_PROGRESS_COPY=1\n'
                    'run_torch job device --inject\n', TEST_RC=str(rc))
                self.assertEqual(rc, result.returncode, result.stderr)
                job_log = (self.root / 'incident-1-T.log').read_text()
                copy = (self.home / 'aim344-results' / 'device-4242.log').read_text()
                for text in (result.stdout, job_log, copy):
                    self.assertIn('AIM344 collective_progress=20 collectives\n', text)
                    self.assertIn('checkpoint write retry: full\n', text)
                    self.assertNotIn('injection', text)
                self.assertIn('AIM344 lab: checkpoint injection: full\n', lab.read_text())
                self.assertEqual(f'launcher_exit_code={rc} (dimensionless)\n',
                                 (self.root / 'incident-1-T.status.log').read_text())

    def test_numbered_launchers_keep_their_names_and_print_lab_lines(self):
        result = self.run_common('run_torch storage-injected storage --inject\n')
        self.assertEqual(0, result.returncode, result.stderr)
        [log] = list(self.root.glob('storage-injected-*Z.log'))
        self.assertIn('AIM344 lab: checkpoint injection: full', log.read_text())
        self.assertFalse((self.home / 'aim344-results').exists())

    def test_the_job_limit_sets_the_launchers_timeout(self):
        timeout = self.root / 'timeout'
        timeout.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$TEST_DIR/timeout.args"\n'
                           'shift 3\nexec "$@"\n')
        timeout.chmod(0o755)
        for limit, expected in ((None, '180s'), ('120', '120s')):
            with self.subTest(limit=limit):
                body = (f'AIM344_JOB_LIMIT_SECONDS={limit}\n' if limit else '') \
                       + 'AIM344_RUN_ID=T run_torch job dataloader\n'
                result = self.run_common(body)
                self.assertEqual(0, result.returncode, result.stderr)
                last = (self.root / 'timeout.args').read_text().splitlines()[-1]
                self.assertTrue(last.startswith(f'--signal=TERM --kill-after=30s {expected} srun '), last)
        refused = self.run_common('AIM344_JOB_LIMIT_SECONDS=2m\nrun_torch job dataloader\n')
        self.assertEqual(2, refused.returncode)
        self.assertIn('AIM344_JOB_LIMIT_SECONDS must be a positive number of seconds', refused.stderr)
        # The device phase keeps no launcher deadline at all.
        lines = len((self.root / 'timeout.args').read_text().splitlines())
        self.assertEqual(0, self.run_common('AIM344_JOB_LIMIT_SECONDS=120\nrun_torch device\n').returncode)
        self.assertEqual(lines, len((self.root / 'timeout.args').read_text().splitlines()))

    def test_sweep_inventory_goes_to_the_lab_log(self):
        lab = self.root / 'incident-2-lab.log'
        result = self.run_common(f'AIM344_RUN_ID=incident-2-T AIM344_LAB_LOG={lab}\nrun_sweep job\n')
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertNotIn('plugin library', result.stdout)
        self.assertIn('plugin library directory is absent', lab.read_text())
        self.assertFalse(list(self.root.glob('*.inventory.log')))
        self.assertTrue((self.root / 'incident-2-T.nccl.log').exists())
        plain = self.run_common('run_sweep fallback\n')
        self.assertIn('plugin library directory is absent', plain.stdout)
        self.assertTrue(list(self.root.glob('fallback-*Z.inventory.log')))


class NumberedLaunchers(unittest.TestCase):
    """Unknown arguments fail before a launcher uses the allocation."""

    RECORDING = ('note() { printf "%s\\n" "$*" >> "$TEST_CALLS"; }\n'
                 'prepare_slurm() { note prepare_slurm; }\nprepare_torch() { note prepare_torch; }\n'
                 'node_command() { note "node_command $*"; }\nrun_sweep() { note "run_sweep $*"; }\n'
                 'run_torch() { note "run_torch $*"; }\nset_allocation_cpus() { note cpus; }\n')

    def run_script(self, name, *argv):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / 'common.sh').write_text(self.RECORDING)
            shutil.copy(LAB / name, directory / name)
            calls = directory / 'calls'
            result = subprocess.run(['bash', str(directory / name), *argv], capture_output=True,
                                    text=True, timeout=20,
                                    env={**os.environ, 'TEST_CALLS': str(calls), 'SLURM_JOB_ID': '',
                                         'SLURM_JOB_CPUS_PER_NODE': ''})
            result.calls = calls.read_text().splitlines() if calls.exists() else []
            return result

    def test_launchers_that_take_no_argument_refuse_one(self):
        for name in ('0.baseline.sh', '2.recover-prejob.sh', '3.inject-fallback.sh',
                     '4.recover-fallback.sh', '9.cleanup.sh', '10.healthcheck-nccl.sh',
                     '11.device-workload.sh', '13.verify-after-recovery.sbatch'):
            with self.subTest(name=name):
                result = self.run_script(name, '--dataset-cache', 'rank-thread')
                self.assertEqual(2, result.returncode, result.stderr)
                self.assertIn('takes no arguments', result.stderr)
                self.assertEqual([], result.calls)

    def test_storage_launchers_accept_only_a_writer_rank(self):
        for name, phase in (('5.inject-storage.sh', 'storage-injected storage --inject'),
                            ('6.recover-storage.sh', 'storage-recovered storage --resume')):
            with self.subTest(name=name):
                plain = self.run_script(name)
                self.assertEqual(0, plain.returncode, plain.stderr)
                self.assertIn(f'run_torch {phase}', plain.calls)
                ranked = self.run_script(name, '--writer-rank', '9')
                self.assertEqual(0, ranked.returncode, ranked.stderr)
                self.assertIn(f'run_torch {phase} --writer-rank 9', ranked.calls)
                for argv in (('--writer-rank',), ('--writer-rank', 'x'), ('--no-such-flag',),
                             ('--writer-rank', '9', '--inject')):
                    bad = self.run_script(name, *argv)
                    self.assertEqual(2, bad.returncode, argv)
                    self.assertIn('[--writer-rank N]', bad.stderr)
                    self.assertEqual([], bad.calls)

    def test_the_device_exercise_usage_names_no_cause(self):
        text = (LAB / '12.device-exercise.sh').read_text()
        usage = text[text.index("<<'USAGE'"):text.index('\nUSAGE\n')]
        header = text[:text.index('set -euo pipefail')]
        for part in (usage, header):
            self.assertIsNone(re.search(r'\b(gpu|efa|flr|unbind|removal|pci)\b', part, re.I), part)
        result = subprocess.run(['bash', str(LAB / '12.device-exercise.sh'), 'nonsense'],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(2, result.returncode)
        self.assertIsNone(re.search(r'\b(gpu|efa|flr)\b', result.stderr, re.I), result.stderr)
        # The grammar is unchanged.
        self.assertIn('start|active)', text)
        self.assertIn('[[ $2 == gpu || $2 == efa ]] || usage', text)


class LoginProfile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.lab = root / 'lab'
        (self.lab / 'facilitator').mkdir(parents=True)
        shutil.copy(LAB / 'facilitator' / 'participant-profile.sh', self.lab / 'facilitator')
        (self.lab / 'lab.env').write_text('export PARTITION=${PARTITION:-gpu}\n'
                                          'export GPUS_PER_NODE=${GPUS_PER_NODE:-8}\n')
        (self.lab / 'facilitator' / 'allocation-env.py').write_text(
            'import sys\nassert sys.argv[1:] == ["--partition", "gpu"], sys.argv\n'
            'print("export PARTITION=gpu\\nexport GPUS_PER_NODE=8\\nexport CPUS_PER_RANK=12\\n'
            'export MEMORY_PER_NODE=0")\n')
        self.seats = root / 'seats'
        self.seats.mkdir()
        (self.seats / f'{USER}.env').write_text(
            "export AIM344_TARGET_NODE=gpu-g7-1\nexport AIM344_PEER_NODE=gpu-g7-2\n"
            "export AIM344_EFA_DEVICE=rdmap83s0\nexport AIM344_SLURM_BIN=/opt/slurm-25.11/bin\n")
        self.home = root / 'home'
        self.home.mkdir()
        self.profile = self.lab / 'facilitator' / 'participant-profile.sh'

    def tearDown(self):
        self.tmp.cleanup()

    def shell(self, commands, **env):
        script = f'cd "$HOME"; source {self.profile}; {commands}'
        base = {'HOME': str(self.home), 'PATH': '/usr/bin:/bin', 'AIM344_SEAT_DIR': str(self.seats)}
        return subprocess.run(['bash', '--norc', '--noprofile', '-i', '-c', script],
                              capture_output=True, text=True, timeout=30, env={**base, **env})

    def test_a_fresh_shell_gets_the_lab_values(self):
        # printenv reads the environment a launcher such as job.sh inherits.
        result = self.shell('pwd; printf "%s\\n" "$(printenv AIM344_TARGET_NODE) '
                            '$(printenv AIM344_PEER_NODE) $(printenv AIM344_EFA_DEVICE)" '
                            '"$(printenv PATH)" "$(printenv PARTITION) $(printenv GPUS_PER_NODE) '
                            '$(printenv CPUS_PER_RANK) $(printenv MEMORY_PER_NODE)" '
                            '"$(printenv RESULTS_DIR || echo none)"')
        lines = result.stdout.splitlines()
        self.assertEqual([str(self.lab), 'gpu-g7-1 gpu-g7-2 rdmap83s0',
                          '/opt/amazon/efa/bin:/opt/slurm-25.11/bin:/usr/bin:/bin',
                          'gpu 8 12 0', 'none'], lines, result.stderr)

    def test_an_allocation_shell_gets_results_dir_and_no_second_path_entry(self):
        result = self.shell(f'SLURM_JOB_ID=77 bash --norc --noprofile -i -c '
                            f'\'source {self.profile}; printf "%s\\n" "$(printenv RESULTS_DIR)" '
                            f'"$(printenv PATH)" "$PWD"\'')
        self.assertEqual([f'/var/tmp/aim344-{USER}-77',
                          '/opt/amazon/efa/bin:/opt/slurm-25.11/bin:/usr/bin:/bin', str(self.lab)],
                         result.stdout.splitlines(), result.stderr)

    def test_without_a_seat_file_the_profile_changes_nothing(self):
        (self.seats / f'{USER}.env').unlink()
        result = self.shell('pwd; echo "${AIM344_TARGET_NODE:-unset} $PATH"', SLURM_JOB_ID='77')
        self.assertEqual([str(self.home), 'unset /usr/bin:/bin'], result.stdout.splitlines())

    def test_a_script_does_not_run_the_profile(self):
        result = subprocess.run(['bash', '--norc', '-c', f'cd "$HOME"; source {self.profile}; pwd'],
                                capture_output=True, text=True, timeout=30,
                                env={'HOME': str(self.home), 'PATH': '/usr/bin:/bin',
                                     'AIM344_SEAT_DIR': str(self.seats)})
        self.assertEqual(str(self.home), result.stdout.strip())


class InstallerProfileSteps(unittest.TestCase):
    """The installer's login-profile steps, each run on its own as an ordinary user."""

    TEXT = (LAB / 'facilitator' / 'install-participant-control.sh').read_text()

    @classmethod
    def block(cls, opening):
        start = cls.TEXT.index(opening)
        body = cls.TEXT.index("<<'PY'\n", start) + len("<<'PY'\n")
        return cls.TEXT[body:cls.TEXT.index('\nPY\n', body)]

    def test_seat_values_are_checked_before_any_change(self):
        start = self.TEXT.index('peer_node=${AIM344_PEER_NODE:-}')
        end = self.TEXT.index('\nfi\n', start) + len('\nfi\n')
        # The check precedes the first change the installer makes.
        self.assertLess(end, self.TEXT.index('echo "== installing the session helper"'))
        snippet = 'set -euo pipefail\ntarget_node=gpu-g7-1\nslurm_bin=${TEST_SLURM_BIN:-/opt/slurm/bin}\n' \
                  + self.TEXT[start:end] + 'echo checked\n'
        for env, ok in (({}, True),
                        ({'AIM344_PEER_NODE': 'gpu-g7-2', 'AIM344_EFA_DEVICE': 'rdmap83s0'}, True),
                        ({'AIM344_PEER_NODE': 'gpu-g7-2'}, False),
                        ({'AIM344_EFA_DEVICE': 'rdmap83s0'}, False),
                        ({'AIM344_PEER_NODE': 'gpu-g7-1', 'AIM344_EFA_DEVICE': 'rdmap83s0'}, False),
                        ({'AIM344_PEER_NODE': 'gpu g7', 'AIM344_EFA_DEVICE': 'rdmap83s0'}, False),
                        ({'AIM344_PEER_NODE': 'gpu-g7-2', 'AIM344_EFA_DEVICE': 'rdma;id'}, False),
                        ({'AIM344_PEER_NODE': 'gpu-g7-2', 'AIM344_EFA_DEVICE': 'rdmap83s0',
                          'TEST_SLURM_BIN': '/opt/slurm bin'}, False)):
            with self.subTest(env=env):
                base = {k: v for k, v in os.environ.items()
                        if k not in ('AIM344_PEER_NODE', 'AIM344_EFA_DEVICE')}
                result = subprocess.run(['bash', '-c', snippet], capture_output=True, text=True,
                                        timeout=20, env={**base, **env})
                self.assertEqual(ok, result.returncode == 0, result.stderr)
                self.assertEqual(ok, result.stdout == 'checked\n')

    def test_the_seat_file_holds_the_handout_values(self):
        block = self.block('python3 - "$seat_dir"')
        with tempfile.TemporaryDirectory() as tmp:
            seats = Path(tmp) / 'seats'
            result = subprocess.run([sys.executable, '-', str(seats), 'aim344-t1', 'gpu-g7-1',
                                     'gpu-g7-2', 'rdmap83s0', '/opt/aws/pcs/scheduler/slurm-25.11/bin'],
                                    input=block, capture_output=True, text=True, timeout=20)
            self.assertEqual(0, result.returncode, result.stderr)
            seat = seats / 'aim344-t1.env'
            self.assertEqual('export AIM344_TARGET_NODE=gpu-g7-1\nexport AIM344_PEER_NODE=gpu-g7-2\n'
                             'export AIM344_EFA_DEVICE=rdmap83s0\n'
                             'export AIM344_SLURM_BIN=/opt/aws/pcs/scheduler/slurm-25.11/bin\n',
                             seat.read_text())
            self.assertEqual(0o644, seat.stat().st_mode & 0o777)
            seats.chmod(0o777)
            refused = subprocess.run([sys.executable, '-', str(seats), 'aim344-t1', 'a', 'b', 'c', '/d'],
                                     input=block, capture_output=True, text=True, timeout=20)
            self.assertNotEqual(0, refused.returncode)
            self.assertIn('writable by others', refused.stderr)

    def test_the_bashrc_line_is_added_once_and_never_through_a_link(self):
        block = self.block('python3 - "$participant_home"')
        profile = '/opt/aim344/device-recovery/companion/facilitator/participant-profile.sh'
        line = f'[ -r {profile} ] && . {profile}  # AIM344 login profile'
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / '.bashrc').write_text('# skel, no final newline')
            for _ in range(2):
                result = subprocess.run([sys.executable, '-', str(home), profile], input=block,
                                        capture_output=True, text=True, timeout=20)
                self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual(f'# skel, no final newline\n{line}\n', (home / '.bashrc').read_text())
            target = home / 'elsewhere'
            target.write_text('untouched\n')
            (home / '.bashrc').unlink()
            (home / '.bashrc').symlink_to(target)
            linked = subprocess.run([sys.executable, '-', str(home), profile], input=block,
                                    capture_output=True, text=True, timeout=20)
            self.assertNotEqual(0, linked.returncode)
            self.assertEqual('untouched\n', target.read_text())


if __name__ == '__main__':
    unittest.main()
