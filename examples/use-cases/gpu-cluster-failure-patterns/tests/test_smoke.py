# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Smoke tests that need no GPU: syntax, and job.sh's six incidents against stub Slurm commands."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

LAB = Path(__file__).resolve().parent.parent
# What each incident's job must pass to its ranks (plugin-state.sh for Incident 1, workload.py otherwise).
INCIDENT_ARGUMENTS = {
    1: ['plugin-state.sh hide', 'sweep-rank.sh'],
    2: ['--writer-rank=9', '--inject'],
    3: ['dataloader', '--dataset-cache=rank-thread'],
    4: ['--corrupt-rank=13', '--corrupt-step=37', '--corrupt-bit=22'],
    5: ['--assert-rank=0', '--assert-step=201'],
    6: ['device', '--duration-seconds=120'],
}
STUB_SRUN = '''#!/bin/bash
printf '%s\\n' "$*" >> "$SRUN_LOG"
case "$*" in *SLURM_GPUS_ON_NODE*) printf '8\\n8\\n' ;; esac
'''
STUB_SCONTROL = '''#!/bin/bash
case "$*" in
  "show hostnames"*) printf 'gpu-1\\ngpu-2\\n' ;;
  "show node"*) echo 'NodeName=gpu-1 NodeAddr=10.0.0.1' ;;
esac
'''


class Smoke(unittest.TestCase):
    def test_scripts_parse(self):
        for script in LAB.glob('*.sh'):
            subprocess.run(['bash', '-n', str(script)], check=True)
        for module in LAB.glob('*.py'):
            compile(module.read_text(), str(module), 'exec')

    def test_job_options_exist_in_workload(self):
        workload = (LAB / 'workload.py').read_text()
        for option in set(re.findall(r'--[a-z][a-z-]+(?==|\b)', (LAB / 'job.sh').read_text())):
            if option.startswith(('--dataset', '--dataloader', '--reinit', '--writer', '--duration', '--checkpoint',
                                  '--replica', '--stack', '--start', '--epochs', '--inject', '--resume', '--end',
                                  '--corrupt', '--assert')):
                self.assertIn(f"'{option}'", workload, option)

    def run_job(self, home, *arguments):
        bin_dir = Path(home, 'bin')
        bin_dir.mkdir(exist_ok=True)
        for name, text in (('srun', STUB_SRUN), ('scontrol', STUB_SCONTROL)):
            (bin_dir / name).write_text(text)
            (bin_dir / name).chmod(0o755)
        log = Path(home, 'srun.log')
        log.write_text('')
        env = dict(os.environ, HOME=home, PATH=f'{bin_dir}:{os.environ["PATH"]}', SRUN_LOG=str(log),
                   SLURM_JOB_ID='7', SLURM_JOB_NUM_NODES='2', SLURM_JOB_CPUS_PER_NODE='192(x2)',
                   SLURM_JOB_NODELIST='gpu-[1-2]', AIM344_EFA_DEVICE='rdmap1s0', RESULTS_DIR=f'{home}/results',
                   TORCH_IMAGE='/opt/aim344/aim344.sqsh', NCCL_ENROOT_IMAGE='/opt/aim344/nccl-baseline.sqsh',
                   CHECKPOINT_DIR='/run/aim344-checkpoints')
        result = subprocess.run(['bash', str(LAB / 'job.sh'), *arguments], env=env, capture_output=True, text=True)
        return result, log.read_text()

    def test_each_incident_starts_its_job(self):
        with tempfile.TemporaryDirectory() as home:
            for number, expected in INCIDENT_ARGUMENTS.items():
                # The pages give the number; without one, job.sh starts the next incident.
                arguments = ['incident'] if number == 2 else ['incident', str(number)]
                result, calls = self.run_job(home, *arguments)
                self.assertIn(f'Starting incident {number}.', result.stdout, result.stderr)
                for text in expected:
                    self.assertIn(text, calls, f'incident {number}')
                if number != 1:  # the stub sweep prints no table for summarize.py
                    self.assertEqual(result.returncode, 0, result.stderr)
            result, calls = self.run_job(home, 'rerun', '--dataset-cache', 'worker-thread')
            self.assertEqual(result.returncode, 2)  # the current incident is 6
            Path(home, 'aim344-results', 'current-incident').write_text('3\n')
            result, calls = self.run_job(home, 'rerun', '--dataset-cache', 'worker-thread')
            self.assertIn('--dataset-cache=worker-thread', calls)
            result, calls = self.run_job(home, 'resume')
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_lab_collect_saves_the_kernel_log_that_the_page_greps(self):
        with tempfile.TemporaryDirectory() as home:
            lab = Path(home, 'lab')
            shutil.copytree(LAB, lab, ignore=shutil.ignore_patterns('tests', '__pycache__'))
            (lab / 'lab.env').write_text('export PARTITION=gpu GPUS_PER_NODE=8 AIM344_TARGET_NODE=gpu-1\n')
            bin_dir = Path(home, 'bin')
            bin_dir.mkdir()
            kernel_line = 'kernel: NVRM: Xid (PCI:0000:53:00): 43, pid=4242, name=python3'
            for name, text in (('squeue', '#!/bin/bash\n'), ('srun', f'#!/bin/bash\necho "{kernel_line}"\n')):
                (bin_dir / name).write_text(text)
                (bin_dir / name).chmod(0o755)
            env = dict(os.environ, HOME=home, PATH=f'{bin_dir}:{os.environ["PATH"]}')
            result = subprocess.run(['bash', str(lab / 'lab.sh'), 'collect', 'kernel'], env=env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            record = result.stdout.splitlines()[-1].removeprefix('Saved: ')
            # Incident 5's page reads the record with: grep -E 'NVRM|Xid' "$record"
            found = subprocess.run(['grep', '-E', 'NVRM|Xid', record], capture_output=True, text=True)
            self.assertEqual(found.stdout.strip(), kernel_line)

    def test_notes(self):
        with tempfile.TemporaryDirectory() as home:
            result, _ = self.run_job(home, 'notes')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('# Incident 6', Path(home, 'aim344-results', 'notes.md').read_text())


if __name__ == '__main__':
    unittest.main()
