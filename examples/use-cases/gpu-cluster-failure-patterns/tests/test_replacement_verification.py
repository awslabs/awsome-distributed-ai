# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Execute participant entry with explicit fake cluster programs, no hardware claims."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

LAB = Path(__file__).resolve().parents[1]


class VerificationEntry(unittest.TestCase):
    def test_success_and_storage_failure_flow(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binaries = root / 'bin'
            binaries.mkdir()
            script = root / 'verify.sbatch'
            # Only relocate outputs. The shipped control flow is executed by Bash.
            script.write_text((LAB / '13.verify-after-recovery.sbatch').read_text().replace('/var/tmp', str(root)))
            control = binaries / 'scontrol'
            control.write_text('#!/bin/bash\nif [ "$2" = hostnames ]; then printf "gpu-g7-1\\ngpu-g7-2\\n"; '
                               'else printf "NodeHostName=%s NodeAddr=10.0.0.1\\n" "$3"; fi\n')
            runner = binaries / 'srun'
            runner.write_text('''#!/usr/bin/env python3
import hashlib, json, os, pathlib, sys
args = ' '.join(sys.argv[1:])
with open(os.environ['LOCAL_ARGV'], 'a') as log:
    log.write(json.dumps(sys.argv[1:]) + '\\n')
if 'efa-counters.py delta' in args:
    for node in ('gpu-g7-1', 'gpu-g7-2'):
        print(json.dumps({'host': node, 'deltas_bytes': {
            '/sys/class/infiniband/rdma0/ports/1/hw_counters/tx_bytes': 123,
            '/sys/class/infiniband/rdma1/ports/1/hw_counters/tx_bytes': 456}}))
elif 'retrieved on' in args:
    bundle = pathlib.Path(os.environ['LOCAL_ROOT']) / ('aim344-verify-' + os.environ['SLURM_JOB_ID'] + '-results.tar')
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    for node in ('gpu-g7-1', 'gpu-g7-2'):
        print('   retrieved on ' + node + ': ' + digest)
elif 'torch-node.sh storage' in args:
    print('LOCAL MOCK storage process completed; not hardware evidence')
    sys.exit(int(os.environ.get('LOCAL_STORAGE_EXIT', '0')))
''')
            suite = root / 'suite.sh'
            suite.write_text('''#!/bin/bash
while [ "$#" -gt 0 ]; do
  if [ "$1" = --results-dir ]; then shift; out=$1; fi
  shift
done
mkdir -p "$out"
printf '  8 2 float sum 0 1 1 1 0 1 1 1 0\\nOut of bounds values : 0\\n' > "$out/nccl-allreduce-raw.txt"
printf '[PASS] 5-nccl-allreduce: LOCAL MOCK\\n'
''')
            for path in (control, runner, suite):
                path.chmod(0o755)
            env = dict(os.environ, PATH=str(binaries) + ':' + os.environ['PATH'],
                       AIM344_BASE=str(root), AIM344_SUITE_ENTRY=str(suite),
                       AIM344_RESULTS_DIR=str(root / 'results'), SLURM_JOB_ID='123',
                       SLURM_JOB_NODELIST='gpu-g7-[1-2]', SLURM_JOB_NUM_NODES='2',
                       LOCAL_ROOT=str(root), LOCAL_ARGV=str(root / 'argv.log'))
            for storage_exit in ('0', '7'):
                env['LOCAL_STORAGE_EXIT'] = storage_exit
                result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0 if storage_exit == '0' else 1, result.stdout + result.stderr)
                self.assertIn('every expected node reported 2 device(s) with traffic', result.stdout)
                self.assertIn('nodes_with_matching_bundle=2 of 2', result.stdout)
                self.assertIn('storage_exit_code=' + storage_exit, result.stdout)
                self.assertIn('srun --partition=gpu-g7 --nodelist=gpu-g7-1 --nodes=1', result.stdout)
                self.assertIn('> ~/aim344-verify-123-results.tar', result.stdout)
                self.assertIn('~/aim344-verify-123-results.tar | sha256sum -c -', result.stdout)
                print('LOCAL_VERIFICATION_ENTRY', result.returncode, result.stdout)
            calls = [json.loads(line) for line in (root / 'argv.log').read_text().splitlines()]
            self.assertTrue(any(c[-2:] == ['/opt/aim344/torch-node.sh', 'storage'] for c in calls))
