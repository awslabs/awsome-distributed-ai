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
import signal
import subprocess
import sys
import tempfile
import time
import unittest

LAB = Path(__file__).resolve().parents[1]
# The collection steps take only the files of the host they run on, named as
# workload.py names them. The fake nodes of CommonRunIds both run here.
HOST = Path('/proc/sys/kernel/hostname').read_text().strip()
JOB = LAB / 'job.sh'
USER = subprocess.run(['id', '-un'], capture_output=True, text=True, check=True).stdout.strip()

# Words that would tell a participant the cause of an incident before the answer.
CAUSE_WORDS = re.compile(r'storage|writer|inject|filler|plugin|slow|efa|reset|flr|'
                         r'dataloader|corrupt|exercise|profile', re.IGNORECASE)

FAKE_COMMON = r'''
note() { printf '%s\n' "$*" >> "$TEST_CALLS"; }
prepare_allocation() { note "prepare_allocation RESULTS_DIR=$RESULTS_DIR"; }
prepare_slurm() { note "prepare_slurm, which starts a container"; }
use_nccl_container() { note "use_nccl_container"; }
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
    # What the job would read, kept apart from the call order the other tests check.
    printf 'job_env settings=%s limit=%s\n' "$(printenv AIM344_INCIDENT_SETTINGS || echo unset)" \
        "${AIM344_JOB_LIMIT_SECONDS:-default}" >> "$TEST_CALLS.env"
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

# Records the container removal; $TEST_CLEANUP_RC makes it fail.
FAKE_CLEANUP = r'''#!/bin/bash
printf 'cleanup\n' >> "$TEST_CALLS"
echo 'current-job container removed: pyxis_aim344_torch_4242'
exit "${TEST_CLEANUP_RC:-0}"
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
        (self.lab / '9.cleanup.sh').write_text(FAKE_CLEANUP)
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

    @staticmethod
    def issue_id(number):
        return f'{number:032x}'

    def profile(self, number, exercise, settings=None, issued_at=100.0, name=None):
        path = self.incidents / (name or f'incident-{number}.json')
        path.write_text(json.dumps({
            'incident': number, 'incidents': 4, 'exercise': exercise,
            'settings': settings or {}, 'issued_at': issued_at,
            'controller_step': 'next-after-progress' if exercise == 'efa-reset' else 'none',
            'issue_id': self.issue_id(number)}))
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
        return self.lines(Path(f'{self.calls}.env'))

    def reset_calls(self):
        for path in (self.calls, Path(f'{self.calls}.env')):
            if path.exists():
                path.unlink()

    def controller_verbs(self):
        # Each logged call reads '<n> <verb> [<argument>]: <first reply line>'.
        return [line.split()[1].rstrip(':') for line in self.lines(self.next_log)]

    def controller_requests(self):
        # The verb and its argument, as the launcher sent them.
        return [line.split(': ', 1)[0].split(' ', 1)[1] for line in self.lines(self.next_log)]

    def next_calls(self):
        return [line for line in self.lines(self.next_log) if line.split()[1] == 'next:']

    def job_processes(self):
        # The controller step is a subshell of job.sh, with the same command line.
        found = subprocess.run(['pgrep', '-f', str(self.lab / 'job.sh')], capture_output=True,
                               text=True).stdout
        return found.split()

    def later_progress(self):
        """A later run in the same allocation writes two fresh progress records."""
        with (self.home / 'aim344-results' / 'device-4242.log').open('a') as copy:
            for n in (20, 40):
                copy.write(f'AIM344 collective_progress={n} collectives; tensor_bytes=1 B; '
                           f'timestamp={time.time()} s since epoch; 0 mismatches; job_id=4242; '
                           'world_size=16; rank=0.\n')

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
                     ('rerun', '--worker-cuda'), ('rerun', '--reinit-in-worker', 'extra'),
                     # No page gives the plain dataset as a fix; with a seat's
                     # dataset settings the job would refuse it and name them.
                     ('rerun', '--dataset-cache', 'none'), ('rerun', '--dataset-cache=none')):
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
        headings = [line for line in text.splitlines() if line.startswith('#')]
        # The pages' five incidents, in their order, each followed by its case; the
        # recovery-wait cards under Incident 5; the cost page's case after the loss table.
        self.assertEqual(['# Your cluster', '# Triage table', '# Healthy values',
                          '# Incident 1', '## Case', '# Incident 2', '## Case',
                          '# Incident 3', '## Case', '# Incident 4', '## Case',
                          '# Incident 5', '## Case', '## Card A', '## Card B',
                          '# Loss table', '## Case'], headings)
        # Each case block holds the three answers its page asks for.
        self.assertEqual(6, text.count('\n## Case\n\n1.\n\n2.\n\n3.\n'))
        self.assertIn('| What you see | Who reported | First record to read | Check that separates '
                      'the causes | Kind of recovery | Your healthy value |', text)
        for row in ('Nobody times out, but the job is slow', 'One rank does not report',
                    'Every rank reports at once', 'Progress stops and nothing prints',
                    'Everything passes, but an answer differs'):
            self.assertIn(f'| {row} | | | | | |', text)
        # The device incident's record table is in the Incident 5 section.
        incident_5 = text[text.index('\n# Incident 5\n'):text.index('\n## Card A\n')]
        # The rows of page 090's step 11, in its order.
        rows = re.findall(r'^\| ([^|]+?) \| \| \|$', incident_5, re.M)
        self.assertEqual(['Last progress line', 'Watchdog lines', 'NCCL WARN lines', 'Kernel log',
                          'Check 0', 'Check 2', 'Check 3', 'Check 6'], rows)
        self.assertEqual(1, text.count('| Kernel log | | |'))
        # The healthy sweep's values that Incident 1 compares against.
        healthy = text[text.index('\n# Healthy values\n'):text.index('\n# Incident 1\n')]
        # Page 030's fields, in the order of its table of healthy values.
        fields = [line for line in healthy.splitlines() if line.endswith(':')]
        self.assertEqual(['Transport:', 'Ranks that finished NCCL setup:', 'Checksums:',
                          'Checkpoint line:', 'Longest gap between progress lines:',
                          'End of epoch 1:', 'Sweep, transport:', 'Sweep, 2 GiB line:',
                          'Sweep, EFA byte changes:', 'Bus bandwidth at 128 MiB:',
                          'Bus bandwidth at the smaller size:', 'Wrong elements:',
                          'EFA devices that moved bytes:', 'Storage:', 'Warnings:',
                          'Pre-session Check 6 record:'], fields)
        loss = text[text.index('\n# Loss table\n'):]
        for row in ('| t0 | Last progress line | |', '| Detection | t0 | t1 | |',
                    '| Rework | The last checkpoint | The stop | |', '| Incident 3 | | | |',
                    '| Incident 4 | | | |'):
            self.assertIn(row, loss)
        self.assertNotIn('| Incident 2 |', loss)
        # Incidents 1 to 4 each have a transfer question; Incident 5 has none.
        self.assertEqual(4, text.count('Transfer question:'))
        self.assertNotIn('Transfer question:', incident_5)
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
                 ({'TEST_JOB_NAME': 'interactive', 'TEST_TIME_LIMIT': 'UNLIMITED'},
                  'job name starts with aim344-'),
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

    def test_any_time_limit_of_the_allocation_is_accepted(self):
        # A seat holds its own two nodes, so the allocation may carry the
        # partition's default limit, UNLIMITED included, or any long limit.
        for limit in ('UNLIMITED', '2-00:00:00', '08:00:00', '00:30:00', '00:00:30'):
            with self.subTest(limit=limit):
                self.reset_calls()
                result = self.run_job('run', TEST_TIME_LIMIT=limit)
                self.assertEqual(0, result.returncode, result.stderr)
                self.assertNotIn('minutes', result.stderr)
                self.assertEqual(2, len(self.torch_calls()), self.torch_calls())

    def test_the_launcher_reads_no_time_limit(self):
        text = JOB.read_text()
        self.assertNotIn('TimeLimit', text)
        self.assertNotIn('1 to 10 minutes', text)

    # ---------------------------------------------------------------- run
    def test_run_starts_the_healthy_job_the_epoch_job_then_the_sweep_and_never_asks_for_an_incident(self):
        # Before any incident the controller has not created the incidents directory.
        shutil.rmtree(self.incidents)
        result = self.run_job('run')
        self.assertEqual(0, result.returncode, result.stderr)
        lab = self.incidents / 'job-lab.log'
        [call, epochs] = self.torch_calls()
        self.assertRegex(call, rf'^run_torch phase=job run_id=job-\d{{8}}T\d{{6}}Z copy=1 '
                               rf'lab={re.escape(str(lab))} iface=rdmap83s0 args=device '
                               r'--checkpoint-every=200 --replica-check --stack-dump-seconds=60 --writer-rank=0 '
                               r'--duration-seconds=60$')
        # Incident 3's job as its resume runs it: no setting, the same 120-second limit.
        self.assertRegex(epochs, rf'^run_torch phase=job run_id=job-epochs-\d{{8}}T\d{{6}}Z copy=1 '
                                 rf'lab={re.escape(str(lab))} iface=rdmap83s0 ')
        self.assertTrue(epochs.endswith(self.LOADER), epochs)
        self.assertEqual(['job_env settings=unset limit=default', 'job_env settings=unset limit=120'],
                         self.job_env())
        self.assertEqual([], self.next_calls())
        calls = self.lines(self.calls)
        # The allocation is checked once, no container starts before a step needs
        # it, and the sweep runs in the NCCL container after the torch jobs.
        self.assertEqual([f'prepare_allocation RESULTS_DIR={self.results}', call, epochs,
                          'use_nccl_container'], calls[:4])
        self.assertRegex(calls[4], rf'^run_sweep job run_id=job-\d{{8}}T\d{{6}}Z '
                                   rf'lab={re.escape(str(lab))} iface=unset$')
        self.assertEqual('cleanup', calls[5])
        self.assertEqual(6, len(calls), calls)
        run_id = call.split('run_id=')[1].split()[0]
        epochs_id = epochs.split('run_id=')[1].split()[0]
        sweep_id = calls[4].split('run_id=')[1].split()[0]
        self.assertEqual(2, result.stdout.count(f'Job log: {self.results}/{run_id}.log\n'))
        self.assertEqual(2, result.stdout.count(f'Job log: {self.results}/{epochs_id}.log\n'))
        self.assertEqual(2, result.stdout.count(f'Job log: {self.results}/{sweep_id}.nccl.log\n'))
        self.assertEqual(3, len(re.findall(r'job_end_utc=\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ\n',
                                           result.stdout)))
        # The container removal is a lab line.
        self.assertNotIn('container removed', result.stdout + result.stderr)
        self.assertIn('current-job container removed', lab.read_text())
        self.assertEqual(0o700, self.incidents.stat().st_mode & 0o777)
        self.assert_no_cause(result)

    def test_a_failed_healthy_job_skips_the_sweep_and_still_removes_the_containers(self):
        result = self.run_job('run', TEST_RC='1')
        self.assertEqual(1, result.returncode)
        calls = self.lines(self.calls)
        # A failed stand-in also skips the epoch job.
        self.assertEqual(1, len(self.torch_calls()), calls)
        self.assertEqual([], [line for line in calls if line.startswith('run_sweep')])
        self.assertEqual('cleanup', calls[-1])

    def test_a_failed_container_removal_keeps_the_job_status_and_says_so_neutrally(self):
        result = self.run_job('run', TEST_CLEANUP_RC='1')
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn('The lab could not remove the job containers on your nodes; raise your hand.',
                      result.stderr)
        self.assertNotIn('table expert', result.stderr)
        self.assert_no_cause(result)

    def test_results_default_to_node_local_storage(self):
        env = dict(self.env)
        del env['RESULTS_DIR']
        result = subprocess.run(['bash', str(self.lab / 'job.sh'), 'run'], capture_output=True,
                                text=True, timeout=60, env=env)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn(f'prepare_allocation RESULTS_DIR=/var/tmp/aim344-{USER}-4242', self.lines(self.calls))

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
                               r'--checkpoint-every=200 --replica-check --stack-dump-seconds=60 --writer-rank=9 '
                               r'--duration-seconds=60 --inject$')
        self.assertEqual(['job_env settings=unset limit=default'], self.job_env())
        # The controller's reply, which names the profile, stays in the lab log.
        self.assertIn(f'Profile: {profile}', lab.read_text())
        self.assertIn('Starting incident 1.', result.stdout)
        self.assertIn('launcher_exit_code=1 (dimensionless)', result.stdout)
        self.assert_no_cause(result)
        # The incident keeps its containers, and starts no NCCL container.
        self.assertNotIn('cleanup', self.lines(self.calls))
        self.assertNotIn('prepare_slurm, which starts a container', self.lines(self.calls))

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
        self.assertNotIn('cleanup', self.lines(self.calls))

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
        # The resumed job ends after its first save past the resume point, within 60 s.
        self.assertTrue(call.endswith('--writer-rank=9 --duration-seconds=60 --resume --end-after-save'), call)
        self.assertEqual('cleanup', calls[-1])
        self.assertGreater(calls.index('cleanup'), calls.index(call))
        lab = (self.incidents / 'incident-1-lab.log').read_text()
        self.assertIn('Removed lab filler', lab)
        self.assertIn('current-job container removed', lab)
        self.assert_no_cause(result)
        self.assertEqual([], self.next_calls())
        # The completed resume ends the incident, after its job and the cleanup, and
        # names it by its issue id.
        self.assertEqual([f'ended {self.issue_id(1)}'], self.controller_requests())
        self.assertIn('ended, exit 0: Incident 9 is ready.', lab)
        self.assertNotIn('ended', result.stdout + result.stderr)

    def test_a_failed_resume_job_does_not_end_the_incident(self):
        self.profile(1, 'storage', {'writer_rank': 9})
        result = self.run_job('resume', TEST_RC='1')
        self.assertEqual(1, result.returncode)
        self.assertEqual([], self.controller_verbs())

    def test_a_refused_end_keeps_the_resume_status_and_asks_for_a_hand(self):
        self.profile(2, 'slow-job')
        self.reply(1, 'This seat has no incident schedule; raise your hand.\n', rc=3)
        result = self.run_job('resume')
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(['ended'], self.controller_verbs())
        self.assertIn('The lab controls did not record that incident 2 has ended; raise your hand.',
                      result.stderr)
        self.assert_no_cause(result)

    def test_rerun_never_ends_the_incident(self):
        self.profile(2, 'slow-job')
        result = self.run_job('rerun')
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual([], self.controller_verbs())

    def test_a_profile_without_a_valid_issue_id_is_refused(self):
        for value in (None, '', 'ABC', '0' * 31, 'g' * 32, 7):
            with self.subTest(value=value):
                path = self.profile(1, 'storage', {'writer_rank': 9})
                profile = json.loads(path.read_text())
                if value is None:
                    del profile['issue_id']
                else:
                    profile['issue_id'] = value
                path.write_text(json.dumps(profile))
                result = self.run_job('resume')
                self.assertEqual(3, result.returncode)
                self.assertIn('The incident profile cannot be used by this launcher; raise your hand.',
                              result.stderr)
                self.assertEqual([], self.lines(self.calls))
                self.assertEqual([], self.controller_verbs())

    # ---------------------------------------------------------------- a busy controller
    BUSY = ('Another request for this assignment is in progress. Wait for it to finish, '
            'then run status.\n')

    def test_a_busy_reply_to_ended_is_sent_again(self):
        """Code review batch 5, Nit 4: a status in terminal B during the resume."""
        self.profile(2, 'slow-job')
        self.reply(1, self.BUSY, rc=3)
        self.reply(2, self.BUSY, rc=3)
        self.reply(3, 'Incident 2 has ended.\n')
        result = self.run_job('resume')
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(['ended'] * 3, self.controller_verbs())
        self.assertEqual('', result.stderr)
        lab = (self.incidents / 'incident-2-lab.log').read_text()
        self.assertIn('ended: sent again after 2 busy replies', lab)
        self.assertIn('ended, exit 0: Incident 2 has ended.', lab)

    def test_a_controller_that_stays_busy_is_asked_for_a_bounded_time(self):
        self.assertIn('\nBUSY_SECONDS=30\n', JOB.read_text())
        # The launcher's bound, shortened in this copy only.
        launcher = self.lab / 'job.sh'
        launcher.write_text(launcher.read_text().replace('\nBUSY_SECONDS=30\n',
                                                         '\nBUSY_SECONDS=3\n'))
        self.profile(2, 'slow-job')
        self.reply('default', self.BUSY, rc=3)
        result = self.run_job('resume')
        self.assertEqual(0, result.returncode, result.stderr)
        # The first request and one more each second for BUSY_SECONDS.
        self.assertEqual(4, len(self.controller_verbs()))
        self.assertIn('The lab controls did not record that incident 2 has ended; raise your hand.',
                      result.stderr)
        self.assertNotIn('Another request', result.stderr)

    def test_a_refusal_other_than_busy_is_not_sent_again(self):
        self.profile(2, 'slow-job')
        self.reply(1, 'Another table is using this exercise node right now. Wait until they '
                      'finish recovering, then start again.\n', rc=3)
        result = self.run_job('resume')
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(['ended'], self.controller_verbs())

    def test_a_busy_reply_to_the_incident_request_is_sent_again(self):
        path = self.profile(1, 'storage', {'writer_rank': 9})
        self.reply(1, self.BUSY, rc=3)
        self.reply(2, f'Incident 1 is ready.\nProfile: {path}\nNext: Your launcher runs '
                      'this incident from the profile above.\n')
        result = self.run_job('incident')
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(['next', 'next'], self.controller_verbs())
        self.assertIn('Starting incident 1.', result.stdout)
        self.assertNotIn('Another request', result.stdout + result.stderr)
        self.assertEqual(1, len(self.torch_calls()))
        lab = (self.incidents / 'incident-1-lab.log').read_text()
        self.assertIn('next: sent again after 1 busy replies', lab)

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
        # The hidden plugin lives in this container, so the incident keeps it.
        self.assertNotIn('cleanup', calls)
        self.assertLess(calls.index('node_command bash /opt/aim344/plugin-state.sh hide'),
                        calls.index(sweep[0]))

    def test_slow_job_rerun_keeps_the_fault_and_resume_restores_first(self):
        self.profile(2, 'slow-job')
        self.assertEqual(0, self.run_job('rerun').returncode)
        self.assertNotIn('plugin-state', ' '.join(self.lines(self.calls)))
        self.assertEqual(1, len([line for line in self.lines(self.calls) if line.startswith('run_sweep')]))
        self.assertNotIn('cleanup', self.lines(self.calls))
        self.calls.unlink()
        result = self.run_job('resume')
        self.assertEqual(0, result.returncode, result.stderr)
        calls = self.lines(self.calls)
        sweep = [i for i, line in enumerate(calls) if line.startswith('run_sweep')]
        self.assertEqual(1, len(sweep))
        self.assertLess(calls.index('node_command bash /opt/aim344/plugin-state.sh restore'), sweep[0])
        self.assertEqual(len(calls) - 1, calls.index('cleanup'))
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
                               r'args=device --checkpoint-every=200 --replica-check --stack-dump-seconds=60 '
                               r'--writer-rank=0 --duration-seconds=120$')
        lab = (self.incidents / 'incident-4-lab.log').read_text()
        self.assertIn('next, attempt 2, exit 0: Incident 4 is under way.', lab)
        self.assert_no_cause(result)
        self.assertNotIn('under way', result.stdout + result.stderr)
        self.assertEqual([], list(self.incidents.glob('.job-ended-*')))
        self.assertNotIn('cleanup', self.lines(self.calls))
        self.assertEqual([], self.job_processes())

    def test_controller_step_stops_at_the_first_refusal(self):
        self.issue(4, 'efa-reset')
        self.reply(2, 'Your job is not running on your node; nothing was changed.\n', rc=3)
        result = self.run_job('incident', TEST_PROGRESS='1', TEST_RUN_SECONDS='6')
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(2, len(self.next_calls()))
        self.assertIn('Your job is not running on your node; nothing was changed.', result.stderr)

    def test_a_busy_reply_in_the_controller_step_is_not_final(self):
        """Spain e2e at 3f9a95e2, finding 1: a status in terminal B lost the fault."""
        self.issue(4, 'efa-reset')
        self.reply(2, self.BUSY, rc=3)
        self.reply(3, 'Incident 4 is under way.\nNext: Watch your job.\n')
        result = self.run_job('incident', TEST_PROGRESS='1', TEST_RUN_SECONDS='20',
                              TEST_UNTIL='under way')
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(3, len(self.next_calls()))
        self.assertNotIn('Another request', result.stdout + result.stderr)
        lab = (self.incidents / 'incident-4-lab.log').read_text()
        self.assertIn('next, busy reply 1, exit 3: Another request for this assignment is in '
                      'progress.', lab)
        # The busy reply did not use up an attempt.
        self.assertIn('next, attempt 1, exit 0: Incident 4 is under way.', lab)
        self.assertEqual([], self.job_processes())

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

    def test_an_exit_inside_the_job_ends_the_controller_step(self):
        # prepare_torch and run_torch exit the launcher instead of returning when
        # TORCH_IMAGE or CHECKPOINT_DIR is unset or Slurm returns no node address.
        for body in ('exit 1', ': "${TORCH_IMAGE:?Set TORCH_IMAGE to the image}"'):
            with self.subTest(body=body):
                for path in self.incidents.glob('*'):
                    path.unlink()
                if self.next_log.exists():
                    self.next_log.unlink()
                self.issue(5, 'efa-reset')
                (self.lab / 'common.sh').write_text(
                    FAKE_COMMON + f'\nrun_torch() {{ note run_torch; {body}; }}\n')
                result = self.run_job('incident', timeout=30)
                self.assertEqual(1, result.returncode, result.stderr)
                self.assertEqual([], self.job_processes())
                self.later_progress()
                time.sleep(3)
                self.assertEqual(1, len(self.next_calls()), self.next_calls())
                self.assertEqual([], list(self.incidents.glob('.job-ended-*')))

    def test_a_killed_launcher_leaves_no_controller_step_behind(self):
        for sig in (signal.SIGTERM, signal.SIGKILL):
            with self.subTest(signal=sig.name):
                if self.next_log.exists():
                    self.next_log.unlink()
                self.issue(5, 'efa-reset')
                launcher = subprocess.Popen(['bash', str(self.lab / 'job.sh'), 'incident'],
                                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                            env={**self.env, 'TEST_RUN_SECONDS': '60'})
                try:
                    deadline = time.monotonic() + 20
                    while len(self.job_processes()) < 2 and time.monotonic() < deadline:
                        time.sleep(0.1)
                    self.assertEqual(2, len(self.job_processes()))
                    os.kill(launcher.pid, sig)
                    launcher.wait(timeout=20)
                    deadline = time.monotonic() + 10
                    while self.job_processes() and time.monotonic() < deadline:
                        time.sleep(0.1)
                    self.assertEqual([], self.job_processes())
                    self.later_progress()
                    time.sleep(3)
                    self.assertEqual(1, len(self.next_calls()), self.next_calls())
                finally:
                    for pid in self.job_processes():
                        os.kill(int(pid), signal.SIGKILL)

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
        # Neither next nor ended.
        self.assertEqual([], self.controller_verbs())
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
        self.assertEqual([f'prepare_allocation RESULTS_DIR={self.results}'], self.lines(self.calls))
        self.assertIn('still holds your node', result.stderr)

    # ---------------------------------------------------------------- DataLoader incident
    LOADER = 'args=dataloader --start-method fork --epochs 2 --stack-dump-seconds=60'

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
                self.reset_calls()
                # A plain rerun and the timeout alone fail on the cluster; here they
                # fail as the fake job.
                rc = '0' if forwarded in ('--dataset-cache=worker-thread',
                                          '--reinit-in-worker') else '1'
                result = self.run_job('rerun', *argv, TEST_RC=rc)
                self.assertEqual(int(rc), result.returncode, result.stderr)
                [call] = self.torch_calls()
                self.assertIn('run_id=incident-3-rerun-', call)
                self.assertTrue(call.endswith((self.LOADER + ' ' + forwarded).rstrip()), call)
                self.assertEqual(['job_env settings={"dataset_cache":"rank-thread"} limit=120'],
                                 self.job_env())
        self.assertEqual([], self.next_calls())

    def test_a_completed_dataloader_fix_rerun_ends_the_incident(self):
        """Code review batch 5, finding 1: page 070 ends Incident 3 with a fix rerun."""
        for argv in (('--dataset-cache', 'worker-thread'), ('--dataset-cache=worker-thread',),
                     ('--reinit-in-worker',), ('--dataloader-timeout', '60', '--reinit-in-worker')):
            with self.subTest(argv=argv):
                if self.next_log.exists():
                    self.next_log.unlink()
                self.profile(3, 'dataloader')
                result = self.run_job('rerun', *argv)
                self.assertEqual(0, result.returncode, result.stderr)
                self.assertEqual([f'ended {self.issue_id(3)}'], self.controller_requests())
                self.assertNotIn('ended', result.stdout + result.stderr)
                self.assert_no_cause(result)
                self.assertNotIn('cleanup', self.lines(self.calls))

    def test_a_dataloader_rerun_without_a_fix_or_that_fails_does_not_end_it(self):
        for argv, rc in (((), '0'), (('--dataset-cache', 'rank-thread'), '0'),
                         (('--dataloader-timeout', '60'), '0'), (('--dataloader-timeout', '60'), '1'),
                         ((), '124'), (('--reinit-in-worker',), '1'),
                         (('--dataset-cache', 'worker-thread'), '124')):
            with self.subTest(argv=argv, rc=rc):
                self.profile(3, 'dataloader')
                result = self.run_job('rerun', *argv, TEST_RC=rc)
                self.assertEqual(int(rc), result.returncode, result.stderr)
                self.assertEqual([], self.controller_verbs())

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
                               r'args=device --checkpoint-every=200 --replica-check --stack-dump-seconds=60 '
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
                self.reset_calls()
                self.profile(4, 'silent-corruption', {**self.FLIP, 'corrupt_recurring': recurring})
                result = self.run_job('rerun')
                self.assertEqual(0, result.returncode, result.stderr)
                [call] = self.torch_calls()
                self.assertIn('run_id=incident-4-rerun-', call)
                self.assertTrue(call.endswith('--writer-rank=0 --duration-seconds=20'), call)
                self.assertEqual([f'job_env settings={expected} limit=default'], self.job_env())

    def test_silent_corruption_resume_rolls_back_before_the_changed_step_without_the_flip(self):
        for recurring in (False, True):
            with self.subTest(recurring=recurring):
                self.reset_calls()
                self.profile(4, 'silent-corruption', {**self.FLIP, 'corrupt_recurring': recurring})
                result = self.run_job('resume')
                self.assertEqual(0, result.returncode, result.stderr)
                [call] = self.torch_calls()
                self.assertIn('run_id=incident-4-resume-', call)
                self.assertTrue(call.endswith('args=device --checkpoint-every=200 --replica-check --stack-dump-seconds=60 '
                                              '--writer-rank=0 --duration-seconds=60 '
                                              '--resume --end-after-save'), call)
                # Only the rollback point reaches the job, in its environment: no flip.
                self.assertEqual(['job_env settings={"resume_before":37} limit=default'],
                                 self.job_env())
                self.assertEqual([], self.next_calls())
                self.assert_no_cause(result)
        # Without a step in the profile the flip is at the job's default, step 1.
        self.reset_calls()
        self.profile(4, 'silent-corruption', {'corrupt_rank': 13})
        self.assertEqual(0, self.run_job('resume').returncode)
        self.assertEqual(['job_env settings={"resume_before":1} limit=default'], self.job_env())

    def test_a_settings_value_from_the_shell_never_reaches_another_job(self):
        stray = {'AIM344_INCIDENT_SETTINGS': '{"corrupt_rank": 1}'}
        self.assertEqual(0, self.run_job('run', **stray).returncode)
        self.profile(2, 'storage', {'writer_rank': 9})
        self.assertEqual(0, self.run_job('rerun', **stray).returncode)
        self.profile(3, 'dataloader', issued_at=200.0)
        self.assertEqual(0, self.run_job('resume', **stray).returncode)
        # run: the stand-in and the epoch job; then the storage rerun; then the resume.
        self.assertEqual(['job_env settings=unset limit=default', 'job_env settings=unset limit=120',
                          'job_env settings=unset limit=default', 'job_env settings=unset limit=120'],
                         self.job_env())

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
                                   # These act on the cached dataset, and the job
                                   # would refuse them without one.
                                   ('dataloader', {'dataset_cache': 'none', 'worker_cuda': True}),
                                   ('dataloader', {'dataset_cache': 'none', 'affected_rank': 0}),
                                   ('silent-corruption', {'corrupt_rank': 13, 'resume_before': 1}),
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


class HelpLines(unittest.TestCase):
    """Participant pages name no table expert or facilitator; the lines a
    participant reads ask them to raise a hand instead."""

    PARTICIPANT_SOURCES = ('job.sh', '12.device-exercise.sh', '14.case-exercise.sh',
                           'facilitator/participant-profile.sh', 'workload.py',
                           'facilitator/device-session.py')
    SENDS_TO_A_PERSON = re.compile(
        r'(?i)table expert|\b(?:ask|tell) (?:your |the )?facilitator|'
        r'your facilitator|facilitator must')

    def test_no_participant_line_sends_them_to_a_table_expert_or_facilitator(self):
        for name in self.PARTICIPANT_SOURCES:
            with self.subTest(name=name):
                self.assertIsNone(self.SENDS_TO_A_PERSON.search((LAB / name).read_text()))


class CommonRunIds(unittest.TestCase):
    """run_torch and run_sweep from the real common.sh, with a fake srun and scontrol.

    The fake torch job writes its lab lines as workload.py --lab-dir does, into
    one directory per fake node, and the fake srun runs common.sh's collection
    command against that node's directory, as RESULTS_DIR is node-local.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'pins.env').write_text('')
        (self.root / 'summarize.py').write_text('print("summary")\n')
        srun = self.root / 'srun'
        srun.write_text('#!/bin/bash\n'
                        'printf "%s\\n" "$*" >> "$TEST_DIR/srun.calls"\n'
                        'case "$*" in\n'
                        '  *inventory.sh*) echo "OFI plugin library directory is absent from its normal search path." ;;\n'
                        '  *efa-counters.py*) echo "counters $*" ;;\n'
                        '  *sweep-rank.sh*) echo "NCCL INFO Using network Socket" ;;\n'
                        '  *torch-node.sh*)\n'
                        '    echo "AIM344 collective_progress=20 collectives"\n'
                        '    lab=; for a in "$@"; do [[ $a == --lab-dir=* ]] && lab=${a#--lab-dir=}; done\n'
                        '    if [[ $lab == /results/.lab-records ]]; then\n'
                        '      for n in nodeA nodeB; do mkdir -p "$TEST_DIR/nodes/$n/.lab-records"; done\n'
                        '      printf "2026-11-03T01:00:02.000Z checkpoint injection: full\\n" \\\n'
                        '        >> "$TEST_DIR/nodes/nodeB/.lab-records/$(cat /proc/sys/kernel/hostname)-pid999999907.log"\n'
                        '      printf "2026-11-03T01:00:01.000Z DataLoader lifecycle: {\\"rank\\": 0}\\n'
                        '2026-11-03T01:00:03.000Z DataLoader lifecycle: {\\"rank\\": 0, \\"late\\": 1}\\n" \\\n'
                        '        >> "$TEST_DIR/nodes/nodeA/.lab-records/$(cat /proc/sys/kernel/hostname)-pid999999905.log"\n'
                        '    else\n'
                        '      echo "checkpoint injection: full"\n'
                        '    fi\n'
                        '    echo "checkpoint write retry: full" ;;\n'
                        '  *" collect "*|*" acknowledge "*)\n'
                        '    node=; for a in "$@"; do [[ $a == --nodelist=* ]] && node=${a#--nodelist=}; done\n'
                        '    args=("$@"); while [[ ${args[0]} != bash ]]; do args=("${args[@]:1}"); done\n'
                        '    [[ -z ${TEST_COLLECT_STDERR:-} ]] || echo "srun: $TEST_COLLECT_STDERR" >&2\n'
                        '    [[ -z ${TEST_COLLECT_RC:-} ]] || exit "$TEST_COLLECT_RC"\n'
                        '    "${args[@]:0:4}" "$TEST_DIR/nodes/$node/.lab-records" "${args[@]:5}"\n'
                        '    exit ;;\n'
                        '  *"--container-remap-root true") exit 0 ;;\n'
                        'esac\n'
                        'exit "${TEST_RC:-0}"\n')
        srun.chmod(0o755)
        scontrol = self.root / 'scontrol'
        scontrol.write_text('#!/bin/bash\n[[ $1 == show && $2 == hostnames ]] || exit 1\n'
                            'printf "nodeA\\nnodeB\\n"\n')
        scontrol.chmod(0o755)
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
                                   'HOME': str(self.home), 'SLURM_JOB_ID': '4242',
                                   'SLURM_JOB_NODELIST': 'node[A-B]', **env})

    def srun_calls(self):
        path = self.root / 'srun.calls'
        return path.read_text().splitlines() if path.exists() else []

    def records(self, node):
        return sorted((self.root / 'nodes' / node / '.lab-records').glob('*.log*'))

    def test_job_sh_lab_lines_reach_the_lab_log_and_no_output(self):
        lab = self.root / 'incident-1-lab.log'
        for rc in (0, 1):
            with self.subTest(rc=rc):
                lab.write_text('earlier\n')
                (self.root / 'srun.calls').unlink(missing_ok=True)
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
                    self.assertNotIn('lifecycle', text)
                    self.assertNotIn('\n\n', text)
                self.assertEqual('', result.stderr)
                create, torch, *collection = self.srun_calls()
                # The container is created before the timed step.
                self.assertTrue(create.endswith(' --container-remap-root true'), create)
                self.assertTrue(torch.endswith('torch-node.sh device --inject --lab-dir=/results/.lab-records'),
                                torch)
                # Per node, after the job and outside the container: a collect step,
                # then, once the lab log holds the records, an acknowledge step.
                self.assertEqual(4, len(collection), collection)
                for (node, name), call in zip((('nodeA', 'collect'), ('nodeB', 'collect'),
                                               ('nodeA', 'acknowledge'), ('nodeB', 'acknowledge')),
                                              collection):
                    self.assertIn(f'--nodes=1 --ntasks=1 --nodelist={node} ', call)
                    self.assertIn(' --overlap bash -c ', call)
                    self.assertIn(f' {name} {self.root}/.lab-records', call)
                    self.assertNotIn('--container', call)
                # Both nodes' lines, whole and in time order, after what the log held.
                self.assertEqual('earlier\n'
                                 '2026-11-03T01:00:01.000Z DataLoader lifecycle: {"rank": 0}\n'
                                 '2026-11-03T01:00:02.000Z checkpoint injection: full\n'
                                 '2026-11-03T01:00:03.000Z DataLoader lifecycle: {"rank": 0, "late": 1}\n',
                                 lab.read_text())
                self.assertEqual([], self.records('nodeA') + self.records('nodeB'))
                self.assertEqual(f'launcher_exit_code={rc} (dimensionless)\n',
                                 (self.root / 'incident-1-T.status.log').read_text())

    def test_an_interrupted_runs_lines_are_collected_by_the_next_run(self):
        lab = self.root / 'incident-3-lab.log'
        left = self.root / 'nodes' / 'nodeB' / '.lab-records'
        left.mkdir(parents=True)
        (left / f'{HOST}-pid999999903.log').write_text('2026-11-03T00:59:00.000Z DataLoader lifecycle: {"left": 1}\n')
        result = self.run_common(f'AIM344_RUN_ID=incident-3-T AIM344_LAB_LOG={lab}\nrun_torch job dataloader\n')
        self.assertEqual(0, result.returncode, result.stderr)
        lines = lab.read_text().splitlines()
        self.assertEqual('2026-11-03T00:59:00.000Z DataLoader lifecycle: {"left": 1}', lines[0])
        self.assertEqual(4, len(lines), lines)
        self.assertEqual([], self.records('nodeB'))

    def test_a_file_is_collected_only_after_its_writer_has_exited(self):
        lab = self.root / 'incident-3-lab.log'
        left = self.root / 'nodes' / 'nodeB' / '.lab-records'
        left.mkdir(parents=True)
        running = subprocess.Popen(['sleep', '60'])
        # Started from a shell that exits at once, so that init reaps it when it ends.
        exiting = int(subprocess.run(['bash', '-c', 'sleep 2 > /dev/null 2>&1 & echo $!'],
                                     capture_output=True, text=True, check=True).stdout)
        self.addCleanup(running.kill)
        (left / f'{HOST}-pid{running.pid}.log').write_text('2026-11-03T00:58:00.000Z still writing\n')
        (left / f'{HOST}-pid{exiting}.log').write_text('2026-11-03T00:59:00.000Z exiting\n')
        result = self.run_common(f'AIM344_RUN_ID=incident-3-T AIM344_LAB_LOG={lab}\nrun_torch job dataloader\n',
                                 AIM344_LAB_RECORDS_WAIT_SECONDS='10')
        self.assertEqual(0, result.returncode, result.stderr)
        text = lab.read_text()
        # The process that exited during the wait is collected; the one still
        # running keeps its file on the node, unread and not removed.
        self.assertIn('exiting', text)
        self.assertNotIn('still writing', text)
        self.assertEqual([left / f'{HOST}-pid{running.pid}.log'], self.records('nodeB'))
        running.kill()
        running.wait()
        self.run_common(f'AIM344_RUN_ID=incident-3-U AIM344_LAB_LOG={lab}\nrun_torch job dataloader\n')
        self.assertIn('still writing', lab.read_text())
        self.assertEqual([], self.records('nodeB'))

    def test_the_files_are_removed_only_after_the_lab_log_stored_them(self):
        lab = self.root / 'incident-1-lab.log'
        sort = self.root / 'sort'
        sort.write_text('#!/bin/bash\nexit 1\n')
        sort.chmod(0o755)
        result = self.run_common(f'AIM344_RUN_ID=incident-1-T AIM344_LAB_LOG={lab}\nrun_torch job dataloader\n')
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual('', result.stderr)
        # Nothing reached the lab log, so nothing was acknowledged or removed.
        self.assertNotIn('lifecycle', lab.read_text())
        self.assertNotIn(' acknowledge ', '\n'.join(self.srun_calls()))
        for node, pid in (('nodeA', 999999905), ('nodeB', 999999907)):
            self.assertRegex(' '.join(p.name for p in self.records(node)),
                             rf'\A{re.escape(HOST)}-pid{pid}\.log\.sent-4242-[0-9T]+\Z')
        sort.unlink()
        self.run_common(f'AIM344_RUN_ID=incident-1-U AIM344_LAB_LOG={lab}\nrun_torch job dataloader\n')
        # The next job's collection stores the kept records with its own and removes them.
        text = lab.read_text()
        self.assertEqual(2, text.count('checkpoint injection: full'))
        self.assertEqual([], self.records('nodeA') + self.records('nodeB'))

    def test_a_failed_read_on_the_node_stores_and_removes_nothing(self):
        """Codex batch-3 finding 1: a read that failed inside the step was acknowledged."""
        lab = self.root / 'incident-1-lab.log'
        cat = self.root / 'cat'
        cat.write_text('#!/bin/bash\n'
                       'for a in "$@"; do [[ $a == *.sent-* ]] && { echo "cat: $a: Input/output error" >&2; exit 1; }; done\n'
                       'exec /bin/cat "$@"\n')
        cat.chmod(0o755)
        result = self.run_common(f'AIM344_RUN_ID=incident-1-T AIM344_LAB_LOG={lab}\nrun_torch job dataloader\n')
        self.assertEqual(0, result.returncode, result.stderr)
        text = lab.read_text()
        self.assertNotIn('lifecycle', text)
        self.assertNotIn('injection', text)
        for node in ('nodeA', 'nodeB'):
            self.assertRegex(text, rf'collect_lab_records: collect on {node} exit 1; files left in ')
            self.assertEqual(1, len(self.records(node)), self.records(node))
        self.assertNotIn(' acknowledge ', '\n'.join(self.srun_calls()))
        cat.unlink()
        self.run_common(f'AIM344_RUN_ID=incident-1-U AIM344_LAB_LOG={lab}\nrun_torch job dataloader\n')
        # The next collection sends the kept files with its own, once each.
        text = lab.read_text()
        self.assertEqual(2, text.count('checkpoint injection: full'))
        self.assertEqual(2, text.count('DataLoader lifecycle: {"rank": 0}\n'))
        self.assertEqual([], self.records('nodeA') + self.records('nodeB'))

    def test_each_node_takes_only_its_own_hosts_files(self):
        """Codex batch-3 finding 2: on shared storage a node took another host's live file."""
        lab = self.root / 'incident-1-lab.log'
        shared = self.root / 'nodes' / 'nodeB' / '.lab-records'
        shared.mkdir(parents=True)
        # A writer on another host: its PID means nothing on this one.
        other = shared / 'other-host-pid999999911.log'
        other.write_text('2026-11-03T00:59:00.000Z still writing elsewhere\n')
        sent = shared / 'other-host-pid999999912.log.sent-4242-20261103T000000000000000'
        sent.write_text('2026-11-03T00:58:00.000Z sent by the other host\n')
        result = self.run_common(f'AIM344_RUN_ID=incident-1-T AIM344_LAB_LOG={lab}\nrun_torch job dataloader\n')
        self.assertEqual(0, result.returncode, result.stderr)
        text = lab.read_text()
        self.assertIn('checkpoint injection: full', text)
        self.assertNotIn('elsewhere', text)
        self.assertNotIn('other host', text)
        self.assertEqual([other, sent], self.records('nodeB'))
        self.assertEqual('2026-11-03T00:59:00.000Z still writing elsewhere\n', other.read_text())

    def test_a_file_left_as_collected_by_an_earlier_version_is_collected(self):
        """Code review batch 4, Nit 1: ed4c0284 renamed a taken file to .log.collected,
        and a failed acknowledge left it there for no later collection to read."""
        lab = self.root / 'incident-1-lab.log'
        directory = self.root / 'nodes' / 'nodeB' / '.lab-records'
        directory.mkdir(parents=True)
        host = Path('/proc/sys/kernel/hostname').read_text().strip()
        legacy = directory / f'{host}-pid999999901.log.collected'
        legacy.write_text('2026-11-03T00:57:00.000Z left by an earlier collection\n')
        other = directory / 'other-host-pid999999913.log.collected'
        other.write_text('2026-11-03T00:57:30.000Z collected on the other host\n')
        result = self.run_common(f'AIM344_RUN_ID=incident-1-T AIM344_LAB_LOG={lab}\nrun_torch job dataloader\n')
        self.assertEqual(0, result.returncode, result.stderr)
        text = lab.read_text()
        self.assertEqual(1, text.count('left by an earlier collection'))
        self.assertIn('checkpoint injection: full', text)
        self.assertNotIn('other host', text)
        # Read, stored and acknowledged: only the other host's file is left.
        self.assertEqual([other], self.records('nodeB'))

    def test_a_failed_collection_keeps_the_files_and_the_jobs_status(self):
        lab = self.root / 'incident-1-lab.log'
        result = self.run_common(f'AIM344_RUN_ID=incident-1-T AIM344_LAB_LOG={lab}\nrun_torch job device\n',
                                 TEST_COLLECT_RC='1', TEST_COLLECT_STDERR='Unable to create step')
        self.assertEqual(0, result.returncode, result.stderr)
        # Nothing of the collection reaches the participant: srun's own error and
        # the note go to the lab log.
        self.assertEqual('', result.stderr)
        self.assertNotIn('collect_lab_records', result.stdout)
        self.assertNotIn('Unable', result.stdout)
        text = lab.read_text()
        self.assertEqual(2, text.count('srun: Unable to create step'))
        for node in ('nodeA', 'nodeB'):
            self.assertRegex(text, rf'collect_lab_records: collect on {node} exit 1; files left in '
                                   + re.escape(f'{self.root}/.lab-records'))
        self.assertNotIn(' acknowledge ', '\n'.join(self.srun_calls()))
        self.assertEqual(1, len(self.records('nodeA')))
        self.assertEqual(1, len(self.records('nodeB')))

    def test_without_a_lab_log_or_in_the_device_phase_lab_lines_print_and_nothing_is_collected(self):
        lab = self.root / 'incident-5-lab.log'
        for body in ('run_torch storage-injected storage --inject\n',
                     f'AIM344_RUN_ID=incident-5-T AIM344_LAB_LOG={lab}\nrun_torch device device\n'):
            with self.subTest(body=body):
                (self.root / 'srun.calls').unlink(missing_ok=True)
                result = self.run_common(body)
                self.assertEqual(0, result.returncode, result.stderr)
                # Every torch job, the device phase included, has its container
                # created first and prints the same line for it.
                *create, call = self.srun_calls()
                self.assertEqual(['--container-remap-root true'],
                                 [line[-len('--container-remap-root true'):] for line in create])
                self.assertIn('Preparing the job container on both nodes.\n', result.stdout)
                self.assertNotIn('--lab-dir', call)
                self.assertIn('checkpoint injection: full', result.stdout)
        # The device phase's container creation may append to the lab log; no lab line does.
        self.assertEqual('', lab.read_text() if lab.exists() else '')

    def test_numbered_launchers_keep_their_names_and_print_lab_lines(self):
        result = self.run_common('run_torch storage-injected storage --inject\n')
        self.assertEqual(0, result.returncode, result.stderr)
        [log] = list(self.root.glob('storage-injected-*Z.log'))
        self.assertIn('checkpoint injection: full', log.read_text())
        self.assertFalse((self.home / 'aim344-results').exists())

    def test_the_container_is_created_before_the_timed_step_and_outside_it(self):
        timeout = self.root / 'timeout'
        timeout.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$TEST_DIR/timeout.args"\n'
                           'shift 3\nexec "$@"\n')
        timeout.chmod(0o755)
        lab = self.root / 'incident-3-lab.log'
        result = self.run_common(f'AIM344_JOB_LIMIT_SECONDS=120 AIM344_RUN_ID=T AIM344_LAB_LOG={lab}\n'
                                 'run_torch job dataloader\n')
        self.assertEqual(0, result.returncode, result.stderr)
        create, torch = self.srun_calls()[:2]
        self.assertTrue(create.endswith(' --container-remap-root true'), create)
        self.assertIn('torch-node.sh dataloader', torch)
        # The job's own step runs under the job's limit, the creation step under
        # a bound of its own (the collection steps have their own 60-second bound).
        timed = (self.root / 'timeout.args').read_text().splitlines()
        [prepare] = [line for line in timed if line.endswith('remap-root true')]
        self.assertTrue(prepare.startswith('--signal=TERM --kill-after=30s 600s srun '), prepare)
        [job] = [line for line in timed if 'torch-node.sh' in line]
        self.assertTrue(job.startswith('--signal=TERM --kill-after=30s 120s srun '), job)
        self.assertTrue(result.stdout.startswith('Preparing the job container on both nodes.\n'),
                        result.stdout)

    def test_a_failed_container_preparation_stops_the_job_with_a_neutral_line(self):
        lab = self.root / 'incident-3-lab.log'
        srun = self.root / 'srun'
        # The creation step prints what a failed container start would print.
        srun.write_text(srun.read_text().replace(
            'case "$*" in\n', 'case "$*" in\n  *"--container-remap-root true") echo "pyxis: container start failed"; exit 42 ;;\n', 1))
        for lab_log in (f'AIM344_LAB_LOG={lab} ', ''):
            with self.subTest(lab_log=bool(lab_log)):
                (self.root / 'srun.calls').unlink(missing_ok=True)
                result = self.run_common(f'AIM344_RUN_ID=T {lab_log}run_torch job dataloader\n'
                                         'echo "unreached"\n')
                self.assertEqual(42, result.returncode, result.stdout + result.stderr)
                self.assertEqual('The job container could not be prepared on both nodes. '
                                 'The job did not start.\n', result.stderr)
                self.assertIn('launcher_exit_code=42 (dimensionless)', result.stdout)
                self.assertEqual('launcher_exit_code=42 (dimensionless)\n',
                                 (self.root / 'T.status.log').read_text())
                self.assertFalse(any('torch-node.sh' in call for call in self.srun_calls()))
                self.assertIsNone(CAUSE_WORDS.search(result.stderr), result.stderr)
                if lab_log:
                    # Pyxis's own output goes to the lab log, not to the participant.
                    self.assertIn('pyxis: container start failed', lab.read_text())
                    self.assertNotIn('pyxis', result.stdout + result.stderr)

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
        # The device phase keeps no launcher deadline on its step; only its
        # container creation is bounded.
        lines = len((self.root / 'timeout.args').read_text().splitlines())
        self.assertEqual(0, self.run_common('AIM344_JOB_LIMIT_SECONDS=120\nrun_torch device\n').returncode)
        added = (self.root / 'timeout.args').read_text().splitlines()[lines:]
        self.assertEqual(1, len(added), added)
        self.assertTrue(added[0].startswith('--signal=TERM --kill-after=30s 600s srun '), added)
        self.assertTrue(added[0].endswith(' --container-remap-root true'), added)

    def test_the_device_step_carries_its_own_slurm_time_limit(self):
        # The allocation may have no time limit, so Slurm ends the device step at
        # a step limit of six minutes; the srun client and its log stay attached.
        timeout = self.root / 'timeout'
        timeout.write_text('#!/bin/bash\nshift 3\nexec "$@"\n')
        timeout.chmod(0o755)
        for phase, expected in (('device', ['--time=6']), ('job', [])):
            with self.subTest(phase=phase):
                (self.root / 'srun.calls').unlink(missing_ok=True)
                self.assertEqual(0, self.run_common(f'run_torch {phase} device\n').returncode)
                [step] = [call for call in self.srun_calls() if 'torch-node.sh' in call]
                self.assertEqual(expected, [word for word in step.split() if word.startswith('--time')])
                if expected:
                    self.assertTrue(step.startswith('--time=6 --nodes=2 '), step)

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


    def test_every_srun_reads_dev_null_instead_of_the_launchers_input(self):
        srun = self.root / 'srun'
        srun.write_text('#!/bin/bash\nprintf "%s %s\\n" "$(readlink /proc/$$/fd/0)" "${*: -1}" >> "$TEST_SRUN_LOG"\n'
                        'case "$*" in *SLURM_GPUS_ON_NODE*) printf "8\\n8\\n" ;; esac\n')
        srun.chmod(0o755)
        suite = self.root / 'suite.sh'
        # The suite launches its own steps, as Check 5's suite does.
        suite.write_text('srun --mpi=pmix all_reduce_perf\n')
        log = self.root / 'srun.log'
        env = {'TEST_SRUN_LOG': str(log), 'SLURM_JOB_CPUS_PER_NODE': '192(x2)',
               'SLURM_JOB_NUM_NODES': '2', 'NCCL_ENROOT_IMAGE': '/images/nccl.sqsh',
               'SLURM_JOB_NODELIST': 'node[A-B]'}
        body = ('prepare_allocation\nnode_command true\nAIM344_RUN_ID=T run_sweep job\n'
                f'AIM344_RUN_ID=T AIM344_LAB_LOG={self.root}/lab.log run_torch job dataloader\n'
                f'run_check5 {suite} {self.root}/check5\n')
        script = ('set -euo pipefail\nLAB_DIR=$TEST_DIR\nsource "$TEST_COMMON"\n'
                  'prepare_torch() { :; }\n' + body)
        result = subprocess.run(['bash', '-c', script], input='typed while the job ran\n' * 4,
                                capture_output=True, text=True, timeout=30,
                                env={**os.environ, 'PATH': f'{self.root}:{os.environ["PATH"]}',
                                     'TEST_DIR': str(self.root), 'TEST_COMMON': str(LAB / 'common.sh'),
                                     'HOME': str(self.home), 'SLURM_JOB_ID': '4242', **env})
        self.assertEqual(0, result.returncode, result.stderr)
        calls = log.read_text().splitlines()
        # The GPU count, the results directory, node_command (also the sweep's
        # inventory and counters), the sweep, the torch job, the collection of its
        # lab records on each node, Check 5's preparation and the suite's own step.
        self.assertGreaterEqual(len(calls), 14, calls)
        # Each collection step's last argument is the collection's token.
        collections = [line for line in calls if re.fullmatch(r'/dev/null 4242-[0-9T]+', line)]
        self.assertEqual(4, len(collections), calls)
        self.assertIn('/dev/null all_reduce_perf', calls)
        self.assertEqual([], [line for line in calls if not line.startswith('/dev/null ')], calls)

    def test_the_check5_entry_point_and_its_suite_read_dev_null(self):
        srun = self.root / 'srun'
        srun.write_text('#!/bin/bash\nprintf "%s %s\\n" "$(readlink /proc/$$/fd/0)" "${*: -1}" >> "$TEST_SRUN_LOG"\n')
        srun.chmod(0o755)
        suite = self.root / 'suite.sh'
        suite.write_text('srun --mpi=pmix all_reduce_perf\n')
        log = self.root / 'srun.log'
        result = subprocess.run(['bash', str(LAB / '10.healthcheck-nccl.sh')],
                                input='typed while the check ran\n' * 4,
                                capture_output=True, text=True, timeout=60,
                                env={**os.environ, 'PATH': f'{self.root}:{os.environ["PATH"]}',
                                     'TEST_SRUN_LOG': str(log), 'SLURM_JOB_ID': '4242',
                                     'SLURM_JOB_CPUS_PER_NODE': '192(x2)', 'SLURM_JOB_NUM_NODES': '2',
                                     'NCCL_ENROOT_IMAGE': '/images/nccl.sqsh', 'NCCL_PROTO': 'Simple',
                                     'AIM344_SUITE_ENTRY': str(suite),
                                     'RESULTS_DIR': str(self.root / 'results')})
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        calls = log.read_text().splitlines()
        # The counters before, Check 5's preparation, the suite's own step, the counters after.
        self.assertEqual(4, len(calls), calls)
        self.assertEqual([], [line for line in calls if not line.startswith('/dev/null ')], calls)
        self.assertIn('Check 5 protocol: NCCL_PROTO=Simple OFI_NCCL_PROTOCOL=RDMA\n', result.stdout)

    def test_a_key_pressed_during_a_timed_job_does_not_stop_it(self):
        # Under timeout, srun runs in a process group of its own. In a terminal, a
        # process of that group that reads the terminal is stopped (SIGTTIN), so a
        # job whose srun read the participant's input would wait until the limit.
        import pty
        srun = self.root / 'srun'
        srun.write_text('#!/bin/bash\nstatus=0\nIFS= read -r line || status=$?\n'
                        'echo "srun read status=$status line=[$line]"\n')
        srun.chmod(0o755)
        script = ('set -euo pipefail\nLAB_DIR=$TEST_DIR\nsource "$TEST_COMMON"\n'
                  'prepare_torch() { :; }\nRESULTS_DIR=$TEST_DIR\nAIM344_CPUS_PER_NODE=1\n'
                  'CONTAINER_ARGS=()\nAIM344_JOB_LIMIT_SECONDS=5\n'
                  'AIM344_RUN_ID=T run_torch job dataloader || echo "run_torch status=$?"\n')
        env = {**os.environ, 'PATH': f'{self.root}:{os.environ["PATH"]}', 'TEST_DIR': str(self.root),
               'TEST_COMMON': str(LAB / 'common.sh'), 'HOME': str(self.home), 'SLURM_JOB_ID': '4242'}
        started = time.monotonic()
        pid, terminal = pty.fork()
        if pid == 0:  # pragma: no cover - the child execs at once
            os.execvpe('bash', ['bash', '-c', script], env)
        output = b''
        try:
            time.sleep(0.5)
            os.write(terminal, b'\r\r')
            while time.monotonic() - started < 30:
                try:
                    chunk = os.read(terminal, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                output += chunk
        finally:
            os.close(terminal)
            _, status = os.waitpid(pid, 0)
        text = output.decode(errors='replace')
        self.assertEqual(0, os.waitstatus_to_exitcode(status), text)
        self.assertIn('srun read status=1 line=[]', text)
        self.assertIn('launcher_exit_code=0 (dimensionless)', text)
        self.assertNotIn('run_torch status=', text)
        self.assertLess(time.monotonic() - started, 5, text)

    def test_allocation_checks_start_no_container_and_prepare_slurm_still_does(self):
        srun = self.root / 'srun'
        srun.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$TEST_SRUN_LOG"\n'
                        'case "$*" in *SLURM_GPUS_ON_NODE*) printf "8\\n8\\n" ;; esac\n')
        srun.chmod(0o755)
        env = {'TEST_SRUN_LOG': str(self.root / 'srun.log'), 'SLURM_JOB_CPUS_PER_NODE': '192(x2)',
               'SLURM_JOB_NUM_NODES': '2', 'NCCL_ENROOT_IMAGE': '/images/nccl.sqsh'}
        show = 'printf "%s\\n" "${CONTAINER_ARGS[@]}"\n'
        result = self.run_common('prepare_allocation\n' + show, **env)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn('allocation: 2 nodes, 8 GPUs per node, 16 GPU ranks\n', result.stdout)
        self.assertIn('--container-image=/images/nccl.sqsh\n--container-name=aim344_4242\n', result.stdout)
        srun_calls = (self.root / 'srun.log').read_text().splitlines()
        self.assertEqual([], [line for line in srun_calls if '--container' in line])
        # A torch job points node_command at the torch container; a sweep after it
        # points it back at the NCCL container.
        result = self.run_common('prepare_allocation\nCONTAINER_ARGS=(--container-image=/images/torch.sqsh)\n'
                                 'use_nccl_container\n' + show, **env)
        self.assertIn('--container-image=/images/nccl.sqsh\n', result.stdout)
        self.assertNotIn('torch', result.stdout)
        (self.root / 'srun.log').unlink()
        result = self.run_common('prepare_slurm\n', **env)
        self.assertEqual(0, result.returncode, result.stderr)
        started = [line for line in (self.root / 'srun.log').read_text().splitlines()
                   if '--container-image=/images/nccl.sqsh' in line]
        self.assertEqual(1, len(started))
        self.assertTrue(started[0].endswith(' --container-remap-root true'), started[0])


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

    def seat_check(self, env, lab, slurm_bin):
        start = self.TEXT.index('peer_node=${AIM344_PEER_NODE:-}')
        end = self.TEXT.index('\nfi\n', start) + len('\nfi\n')
        # The check precedes the first change the installer makes.
        self.assertLess(end, self.TEXT.index('echo "== installing the session helper"'))
        snippet = 'set -euo pipefail\ntarget_node=gpu-g7-1\n' + self.TEXT[start:end] + 'echo checked\n'
        base = {k: v for k, v in os.environ.items() if k not in ('AIM344_PEER_NODE', 'AIM344_EFA_DEVICE')}
        return subprocess.run(['bash', '-c', snippet], capture_output=True, text=True, timeout=20,
                              env={**base, 'lab': str(lab), 'slurm_bin': str(slurm_bin), **env})

    @staticmethod
    def lab_tree(root):
        """A lab directory and a Slurm client directory that pass the checks."""
        lab, slurm = root / 'lab', root / 'slurm-25.11' / 'bin'
        (lab / 'facilitator').mkdir(parents=True)
        slurm.mkdir(parents=True)
        for name in ('facilitator/participant-profile.sh', 'facilitator/allocation-env.py', 'lab.env'):
            (lab / name).write_text('# lab file\n')
            (lab / name).chmod(0o644)
        for path in (root, lab, lab / 'facilitator'):
            path.chmod(0o755)
        (slurm / 'salloc').write_text('#!/bin/sh\n')
        (slurm / 'salloc').chmod(0o755)
        return lab, slurm

    def test_seat_values_are_checked_before_any_change(self):
        profile = {'AIM344_PEER_NODE': 'gpu-g7-2', 'AIM344_EFA_DEVICE': 'rdmap83s0'}
        with tempfile.TemporaryDirectory() as tmp:
            lab, slurm = self.lab_tree(Path(tmp))
            for env, slurm_bin, ok in (
                    ({}, '/opt/slurm/bin', True),
                    (profile, slurm, True),
                    ({'AIM344_PEER_NODE': 'gpu-g7-2'}, slurm, False),
                    ({'AIM344_EFA_DEVICE': 'rdmap83s0'}, slurm, False),
                    ({'AIM344_PEER_NODE': 'gpu-g7-1', 'AIM344_EFA_DEVICE': 'rdmap83s0'}, slurm, False),
                    ({'AIM344_PEER_NODE': 'gpu g7', 'AIM344_EFA_DEVICE': 'rdmap83s0'}, slurm, False),
                    ({'AIM344_PEER_NODE': 'gpu-g7-2', 'AIM344_EFA_DEVICE': 'rdma;id'}, slurm, False),
                    (profile, '/opt/slurm bin', False),
                    # A Slurm client directory of another version, without salloc.
                    (profile, Path(tmp) / 'slurm-25.05' / 'bin', False)):
                with self.subTest(env=env, slurm_bin=slurm_bin):
                    result = self.seat_check(env, lab, slurm_bin)
                    self.assertEqual(ok, result.returncode == 0, result.stderr)
                    self.assertEqual(ok, result.stdout == 'checked\n')
            missing = self.seat_check(profile, lab, Path(tmp) / 'slurm-25.05' / 'bin')
            self.assertIn('No salloc in', missing.stderr)

    def test_the_profile_is_refused_from_a_lab_directory_others_can_change(self):
        profile = {'AIM344_PEER_NODE': 'gpu-g7-2', 'AIM344_EFA_DEVICE': 'rdmap83s0'}
        changes = {
            'the lab directory writable by its group': lambda lab: lab.chmod(0o775),
            'facilitator/ writable by others': lambda lab: (lab / 'facilitator').chmod(0o757),
            'the profile writable by its group': lambda lab: (lab / 'facilitator/participant-profile.sh').chmod(0o664),
            'allocation-env.py writable by others': lambda lab: (lab / 'facilitator/allocation-env.py').chmod(0o646),
            'lab.env writable by its group': lambda lab: (lab / 'lab.env').chmod(0o664),
            'the profile a symbolic link': lambda lab: (
                (lab / 'facilitator/participant-profile.sh').unlink(),
                (lab / 'facilitator/participant-profile.sh').symlink_to(lab / 'lab.env')),
            'the profile missing': lambda lab: (lab / 'facilitator/participant-profile.sh').unlink(),
            'a parent writable by others, not sticky': lambda lab: lab.parent.chmod(0o777),
        }
        for name, change in changes.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                lab, slurm = self.lab_tree(Path(tmp))
                change(lab)
                result = self.seat_check(profile, lab, slurm)
                self.assertNotEqual(0, result.returncode)
                self.assertIn('refusing to install the login profile', result.stderr)
                self.assertEqual('', result.stdout)
        with tempfile.TemporaryDirectory() as tmp:
            lab, slurm = self.lab_tree(Path(tmp))
            # lab.env is generated on site and may come after the installer.
            (lab / 'lab.env').unlink()
            self.assertEqual(0, self.seat_check(profile, lab, slurm).returncode)
            # A sticky parent such as /tmp lets others add entries, not replace these.
            lab.parent.chmod(0o1777)
            self.assertEqual(0, self.seat_check(profile, lab, slurm).returncode)
            # Through a symbolic link, the resolved directory is checked.
            link = Path(tmp) / 'link'
            link.symlink_to(lab)
            self.assertEqual(0, self.seat_check(profile, link, slurm).returncode)
            lab.chmod(0o775)
            self.assertNotEqual(0, self.seat_check(profile, link, slurm).returncode)

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
