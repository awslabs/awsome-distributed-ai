# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Execute participant entry with explicit fake cluster programs, no hardware claims."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import importlib.util
import unittest

LAB = Path(__file__).resolve().parents[1]


def pmi_hook_environment(environment):
    """Run the exact deployed/provider hook, not a reimplementation of its gate.

    Fixture: NVIDIA/enroot v3.5.0 conf/hooks/extra/50-slurm-pmi.sh, byte-identical
    to both nodes' recorded /etc/enroot/hooks.d/50-slurm-pmi.sh in jobs269/270
    diagnosis. Only scontrol and common::checkcmd/err are local stand-ins.
    No mount, GPU or Slurm service is executed by this test.
    """
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        control = root / 'scontrol'
        control.write_text('#!/bin/bash\nprintf "SlurmdSpoolDir = /var/spool/slurmd\\nTmpFS = /tmp\\n"\n')
        control.chmod(0o755)
        (root / 'common.sh').write_text('common::checkcmd() { :; }\ncommon::err() { exit 1; }\n')
        mounts, environ = root / 'mounts', root / 'environment'
        mounts.touch()
        # Image PMIX settings are data in ENROOT_ENVIRON, NOT hook process env.
        environ.write_text('PMIX_MCA_gds=image-value\n')
        env = dict(environment, PATH=str(root) + ':' + os.defpath,
                   ENROOT_LIBRARY_PATH=str(root), ENROOT_MOUNTS=str(mounts),
                   ENROOT_ENVIRON=str(environ), SLURM_JOB_ID='270',
                   SLURM_STEP_ID='2', SLURM_JOB_UID='22001')
        result = subprocess.run(['bash', str(LAB / 'tests/fixtures/enroot-3.5.0-50-slurm-pmi.sh')],
                                env=env, capture_output=True, text=True, timeout=5)
        if result.returncode:
            raise AssertionError(result.stderr)
        return mounts.read_text(), environ.read_text()


class VerificationEntry(unittest.TestCase):
    def test_recorded_pmi_false_positive_and_explicit_step_contract(self):
        bad = {'PMIX_MCA_gds': 'hash'}  # exact exported lab.env trigger
        mounts, _ = pmi_hook_environment(bad)
        required = '/var/spool/slurmd/pmix.270.2'
        self.assertIn(required + ' x-create=dir,bind,rw,nosuid,noexec,nodev,private\n', mounts)
        self.assertFalse(Path(required).exists())
        print('LOCAL_RECORDED_BAD_ENV', bad, 'missing_source=', required, 'mounts=', mounts)
        # This is an explicit Enroot hook input, not an invented ENROOT_* switch.
        for mpi_type in ('none', 'pmi2'):
            self.assertEqual(pmi_hook_environment(dict(bad, SLURM_MPI_TYPE=mpi_type))[0], '')
        for mpi_type in ('pmix', 'pmix_v4'):
            mounts, env = pmi_hook_environment(dict(bad, SLURM_MPI_TYPE=mpi_type,
                                                   PMIX_SERVER_URI2='step-server', PMIX_RANK='0'))
            self.assertIn(required, mounts)
            self.assertIn('PMIX_SERVER_URI2=step-server', env)
            self.assertIn('PMIX_RANK=0', env)
        # An image-only PMIX setting does not activate the host-process gate.
        self.assertEqual(pmi_hook_environment({})[0], '')

    def test_device_storage_and_sweep_share_clean_allocation_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binaries = root / 'bin'
            binaries.mkdir()
            runner = binaries / 'srun'
            runner.write_text('''#!/usr/bin/env python3
import json, os, sys
from test_replacement_verification import pmi_hook_environment
args = sys.argv[1:]
env = dict(os.environ)
mounts = ''
if any(a.startswith('--container-image=') for a in args):
    # Slurm supplies fresh server state only for the actual PMIx step.
    if '--mpi=pmix' in args:
        env.update(PMIX_SERVER_URI2='fresh-step-server', PMIX_RANK='0')
    mounts, _ = pmi_hook_environment(env)
with open(os.environ['LOCAL_ARGV'], 'a') as f:
    f.write(json.dumps(dict(argv=args, env=env, mounts=mounts)) + '\\n')
if '--mpi=none' in args and '/var/spool/slurmd/pmix.' in mounts:
    sys.exit(77)
if 'SLURM_GPUS_ON_NODE' in ' '.join(args):
    print('8\\n8')
if '--mpi=pmix' in args:
    print('# LOCAL MOCK sweep rows; not hardware evidence')
    for power in range(3, 32):
        print(str(2 ** power) + ' 2 float sum 0 1 1 1 0 1 1 1 0')
    print('Out of bounds values : 0 OK')
''')
            control = binaries / 'scontrol'
            control.write_text('#!/bin/bash\nif [ "$2" = hostnames ]; then printf "node1\\nnode2\\n"; '
                               'else printf "NodeAddr=10.0.0.1\\n"; fi\n')
            for binary in (runner, control):
                binary.chmod(0o755)
            env = dict(os.environ, PATH=str(binaries) + ':' + os.environ['PATH'],
                       PYTHONPATH=str(LAB / 'tests'), LAB_DIR=str(LAB),
                       SLURM_JOB_ID='269', SLURM_JOB_NUM_NODES='2', SLURM_JOB_NODELIST='node[1-2]',
                       SLURM_CPUS_ON_NODE='192', NCCL_ENROOT_IMAGE='/opt/aim344/aim344.sqsh',
                       TORCH_IMAGE='/opt/aim344/aim344.sqsh', CHECKPOINT_DIR=str(root / 'checkpoints'),
                       RESULTS_DIR=str(root / 'results'), HOME=str(root), LOCAL_ARGV=str(root / 'argv'),
                       PMIX_MCA_gds='hash', PMIX_SERVER_URI2='stale', PMI_RANK='15',
                       SLURM_MPI_TYPE='pmix')
            result = subprocess.run(['bash', '-euc', '''source "$LAB_DIR/common.sh"
prepare_slurm
run_torch device device --duration-seconds=120
run_torch storage storage
run_sweep baseline
'''], env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            calls = [json.loads(line) for line in (root / 'argv').read_text().splitlines()]
            torch = [c for c in calls if '/opt/aim344/torch-node.sh' in c['argv']]
            self.assertEqual(len(torch), 2)
            for call in calls:
                if '--mpi=none' in call['argv']:
                    self.assertEqual(call['mounts'], '')
                    self.assertFalse(any(k.startswith(('PMI_', 'PMIX_')) for k in call['env']))
                if any(a.startswith('--container-image=') for a in call['argv']):
                    self.assertEqual(call['env']['SLURM_MPI_TYPE'],
                                     'pmix' if '--mpi=pmix' in call['argv'] else 'none')
            main = [c for c in calls if '--mpi=pmix' in c['argv']]
            self.assertEqual(len(main), 1)
            self.assertIn('--ntasks=16', main[0]['argv'])
            self.assertIn('--ntasks-per-node=8', main[0]['argv'])
            self.assertIn('--cpus-per-task=24', main[0]['argv'])
            self.assertIn('/var/spool/slurmd/pmix.270.2', main[0]['mounts'])
            self.assertEqual(main[0]['env']['PMIX_SERVER_URI2'], 'fresh-step-server')
            print('LOCAL_DEVICE_STORAGE_SWEEP_CONTRACT', json.dumps([
                dict(argv=c['argv'], mpi=c['env'].get('SLURM_MPI_TYPE'), mounts=c['mounts'])
                for c in calls]))

    def test_success_and_storage_failure_flow(self, launch_environment=None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            companion = root / 'companion'
            companion.mkdir()
            for name in ('common.sh', 'pins.env', '9.cleanup.sh'):
                # Relocate only the data directory for real local tiny roots.
                (companion / name).write_text((LAB / name).read_text().replace(
                    '/tmp/enroot/data/user-$EUID', str(root / 'data')))
            data = root / 'data'
            data.mkdir(mode=0o700)
            enroot = root / 'enroot'
            enroot.write_text("#!/usr/bin/env python3\nimport os,pathlib,sys\n"
                              "p=pathlib.Path(os.environ['ENROOT_DATA_PATH'])/sys.argv[-1]\n"
                              "assert sys.argv[1:3]==['remove','-f']\n"
                              "assert p.parent==pathlib.Path(os.environ['LOCAL_ROOT'])/'data'\n"
                              "p.rmdir()\n")
            enroot.chmod(0o755)
            binaries = root / 'bin'
            binaries.mkdir()
            script = root / 'verify.sbatch'
            # Only relocate outputs. The shipped control flow is executed by Bash.
            script.write_text((LAB / '13.verify-after-recovery.sbatch').read_text().replace('/var/tmp', str(root)))
            control = binaries / 'scontrol'
            control.write_text('#!/bin/bash\nif [ "$2" = hostnames ]; then printf "gpu-g7-1\\ngpu-g7-2\\n"; '
                               'exit "${LOCAL_NODELIST_EXIT:-0}"; '
                               'elif [ "$2" = job ]; then printf "JobId=%s UserId=%s(%s) JobState=RUNNING NumNodes=2\\n" "$3" "$(id -un)" "$(id -u)"; '
                               'else printf "NodeHostName=%s NodeAddr=10.0.0.1\\n" "$3"; exit "${LOCAL_NODE_MAP_EXIT:-0}"; fi\n')
            runner = binaries / 'srun'
            runner.write_text('''#!/usr/bin/env python3
import hashlib, json, os, pathlib, subprocess, sys
from test_replacement_verification import pmi_hook_environment
args = ' '.join(sys.argv[1:])
mounts = ''
if any(a.startswith('--container-image=') for a in sys.argv[1:]):
    mounts, _ = pmi_hook_environment(dict(os.environ))
with open(os.environ['LOCAL_ARGV'], 'a') as log:
    log.write(json.dumps({'argv': sys.argv[1:], 'env': dict(os.environ), 'mounts': mounts}) + '\\n')
if '--mpi=none' in sys.argv and '/var/spool/slurmd/pmix.' in mounts:
    print('LOCAL hook emitted missing required PMIx source for non-MPI step', file=sys.stderr)
    sys.exit(77)
if '--exclusive' in sys.argv:
    import subprocess
    assert '--cpus-per-task=192' in sys.argv and '--immediate=10' in sys.argv
    assert '--overlap' not in sys.argv
    assert not any(k.startswith('SLURM_SPANK__') for k in os.environ)
    if os.environ.get('LOCAL_CLEANUP_BUSY') == '1':
        sys.exit(1)
    sys.exit(subprocess.run(sys.argv[sys.argv.index('bash'):]).returncode)
if sys.argv[-1] == 'true':
    sys.exit(int(os.environ.get('LOCAL_PREP_EXIT', '0')))
if 'efa-counters.py delta' in args:
    for node in ('gpu-g7-1', 'gpu-g7-2'):
        print(json.dumps({'host': node, 'deltas_bytes': {
            '/sys/class/infiniband/rdma0/ports/1/hw_counters/tx_bytes': 123,
            '/sys/class/infiniband/rdma1/ports/1/hw_counters/tx_bytes': 456}}))
elif 'retrieved on' in args:
    bundle = pathlib.Path(os.environ['LOCAL_ROOT']) / ('aim344-verify-' + os.environ['SLURM_JOB_ID'] + '-results.tar')
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    if os.environ.get('LOCAL_RETRIEVAL_FAIL') == '1':
        digest = 'bad-digest'
    mode = os.environ.get('LOCAL_RETRIEVAL_ROWS', '')
    nodes = ['gpu-g7-1', 'gpu-g7-2']
    if mode == 'duplicate':
        nodes = ['gpu-g7-1', 'gpu-g7-1']
    elif mode == 'missing':
        nodes = nodes[:1]
    elif mode == 'unknown':
        nodes[1] = 'gpu-g7-3'
    elif mode == 'reverse':
        nodes.reverse()
    if mode or os.environ.get('LOCAL_RETRIEVAL_FAIL') == '1':
        for node in nodes:
            host = 'wrong-host' if mode == 'wrong-host' else node
            print('   retrieved on ' + host + ' (' + node + '): ' + digest)
        if mode == 'extra':
            print('unexpected output')
    else:
        # Execute the shipped remote Bash checksum command, not just matching
        # canned rows, so a failed sha256sum cannot hide inside printf.
        for node in nodes:
            remote = subprocess.run(sys.argv[sys.argv.index('bash'):],
                                    env=dict(os.environ, SLURMD_NODENAME=node,
                                             LOCAL_REMOTE_NODE=node))
            if remote.returncode:
                sys.exit(remote.returncode)
    sys.exit(int(os.environ.get('LOCAL_RETRIEVAL_EXIT', '0')))
elif '.incoming' in args:
    sys.exit(int(os.environ.get('LOCAL_DISTRIBUTION_EXIT', '0')))
elif 'test -r /run/aim344-checkpoints' in args:
    sys.exit(int(os.environ.get('LOCAL_FIXTURE_EXIT', '0')))
elif 'torch-node.sh storage' in args:
    # The Check 5 root must still exist when storage reuses it.
    if os.environ['SLURM_JOB_ID'] == '123':
        assert (pathlib.Path(os.environ['LOCAL_ROOT']) / 'data/pyxis_aim344_check5_123').is_dir()
    print('LOCAL MOCK storage process completed; not hardware evidence')
    sys.exit(int(os.environ.get('LOCAL_STORAGE_EXIT', '0')))
''')
            hostname = binaries / 'hostname'
            hostname.write_text('#!/bin/bash\nprintf "%s\\n" "${LOCAL_REMOTE_NODE:-batch-host}"\n')
            checksum = binaries / 'sha256sum'
            checksum.write_text('''#!/bin/bash
/usr/bin/sha256sum "$@" || exit $?
if [[ ${LOCAL_DIGEST_FAILURE:-} == local && -z ${LOCAL_REMOTE_NODE:-} && $1 == *-results.tar ]] ||
   [[ ${LOCAL_DIGEST_FAILURE:-} == remote && -n ${LOCAL_REMOTE_NODE:-} ]]; then
    exit 7
fi
''')
            hostname.chmod(0o755)
            checksum.chmod(0o755)
            deadline = binaries / 'timeout'
            deadline.write_text("#!/bin/bash\nprintf '%s\\n' \"$3\" >> \"$LOCAL_TIMEOUT_LOG\"\nshift 3\nexec \"$@\"\n")
            deadline.chmod(0o755)
            suite = root / 'suite.sh'
            suite.write_text('''#!/bin/bash
python3 -c 'import json,os,sys; json.dump(dict(env=dict(os.environ),argv=sys.argv[1:]),open(os.environ["LOCAL_SUITE_LOG"],"w"))' "$@"
while [ "$#" -gt 0 ]; do
  if [ "$1" = --results-dir ]; then shift; out=$1; fi
  shift
done
mkdir -p "$out"
printf '  8 2 float sum 0 1 1 1 0 1 1 1 0\\nOut of bounds values : 0\\n' > "$out/nccl-allreduce-raw.txt"
printf '[PASS] 5-nccl-allreduce: LOCAL MOCK\\n'
exit "${LOCAL_SUITE_EXIT:-0}"
''')
            for path in (control, runner, suite):
                path.chmod(0o755)
            env = dict(os.environ, PATH=str(binaries) + ':' + str(root) + ':' + os.environ['PATH'],
                       SLURM_JOB_CPUS_PER_NODE='192(x2)',
                       AIM344_BASE=str(root), AIM344_SUITE_ENTRY=str(suite),
                       AIM344_RESULTS_DIR=str(root / 'results'), SLURM_JOB_ID='123',
                       SLURM_JOB_NODELIST='gpu-g7-[1-2]', SLURM_JOB_NUM_NODES='2',
                       SLURM_JOB_PARTITION='assigned-queue', LOCAL_SUITE_LOG=str(root / 'suite.json'),
                       LOCAL_ROOT=str(root), LOCAL_ARGV=str(root / 'argv.log'),
                       PYTHONPATH=str(LAB / 'tests'),
                       LOCAL_TIMEOUT_LOG=str(root / 'timeouts.log'))
            spec = importlib.util.spec_from_file_location('bootstrap_launch', LAB / 'facilitator/replacement-bootstrap.py')
            bootstrap = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(bootstrap)
            manifest = {'verification_environment': {'PARTITION': 'assigned-queue',
                        'NCCL_CONTAINER': '/opt/aim344/aim344.sqsh',
                        'NCCL_SOCKET_IFNAME': '=eth0', 'PMIX_MCA_gds': 'hash'}}
            (companion / 'lab.env').write_text(launch_environment or bootstrap.verification_environment(manifest))
            env.update(LAB_DIR=str(companion), OMPI_MCA_btl_tcp_if_include='bad-include',
                       NCCL_TESTS_SPLIT='bad-split', NCCL_NET='bad-net',
                       PMIX_MCA_gds='hash', PMIX_SERVER_URI2='stale-server',
                       PMIX_RANK='15', PMI_RANK='15', SLURM_MPI_TYPE='pmix')
            for storage_exit in ('0', '7'):

                env['LOCAL_STORAGE_EXIT'] = storage_exit
                current = data / 'pyxis_aim344_check5_123'
                current.mkdir()
                historical = data / 'pyxis_aim344_check5_122'
                historical.mkdir(exist_ok=True)
                result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0 if storage_exit == '0' else 1, result.stdout + result.stderr)
                self.assertIn('every expected node reported 2 device(s) with traffic', result.stdout)
                self.assertIn('nodes_with_matching_bundle=2 of 2', result.stdout)
                self.assertIn('storage_exit_code=' + storage_exit, result.stdout)
                self.assertIn('srun --partition=assigned-queue --nodelist=gpu-g7-1 --nodes=1', result.stdout)
                self.assertIn('> ~/aim344-verify-123-results.tar', result.stdout)
                self.assertIn('~/aim344-verify-123-results.tar | sha256sum -c -', result.stdout)
                self.assertFalse(current.exists(), result.stdout + result.stderr)
                self.assertTrue(historical.is_dir())
                self.assertLess(result.stdout.index('storage_exit_code='),
                                result.stdout.index('current-job container removed:'))
                self.assertLess(result.stdout.index('nodes_with_matching_bundle=2 of 2'),
                                result.stdout.index('current-job container removed:'))
                print('LOCAL_VERIFICATION_ENTRY', result.returncode, result.stdout)
            calls = [json.loads(line) for line in (root / 'argv.log').read_text().splitlines()]
            storage_calls = [c for c in calls if c['argv'][-2:] == ['/opt/aim344/torch-node.sh', 'storage']]
            self.assertTrue(storage_calls)
            for call in storage_calls:
                self.assertIn('--container-name=aim344_check5_123', call['argv'])
                self.assertEqual(call['env']['SLURM_MPI_TYPE'], 'none')
                self.assertEqual(call['mounts'], '')
            launch = json.loads((root / 'suite.json').read_text())
            expected = {'NCCL_CONTAINER': '/opt/aim344/aim344.sqsh', 'NCCL_TIMEOUT': '180',
                        'NCCL_MPI': 'pmix', 'NCCL_TESTS_BIN': '/opt/nccl-tests/build/all_reduce_perf',
                        'NCCL_ISOLATION_TESTS': '1', 'NCCL_ISOLATION_TIMEOUT': '120',
                        'NCCL_SOCKET_IFNAME': '=eth0', 'OMPI_MCA_pml': 'ob1',
                        'OMPI_MCA_btl': 'tcp,self', 'PMIX_MCA_gds': 'hash', 'OFI_NCCL_PROTOCOL': 'RDMA'}
            for key, value in expected.items():
                self.assertEqual(launch['env'][key], value)
            self.assertEqual(launch['env']['SLURM_MPI_TYPE'], 'pmix')
            self.assertNotIn('PMIX_SERVER_URI2', launch['env'])
            self.assertNotIn('PMI_RANK', launch['env'])
            main_mounts, _ = pmi_hook_environment(launch['env'])
            self.assertIn('/var/spool/slurmd/pmix.270.2 ', main_mounts)
            for key in ('OMPI_MCA_btl_tcp_if_include', 'NCCL_TESTS_SPLIT', 'NCCL_NET'):
                self.assertNotIn(key, launch['env'])
            spank = 'SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_'
            self.assertEqual(launch['env'][spank + 'name'], 'aim344_check5_123')
            self.assertIn('OMPI_MCA_btl_tcp_if_include', launch['env'][spank + 'env'].split(','))
            self.assertIn('OMPI_MCA_btl_tcp_if_exclude', launch['env'][spank + 'env'].split(','))
            self.assertEqual(launch['argv'][:5], ['--check', '5', '--verbose', '--timeout', '360'])
            preps = [c for c in calls if c['argv'][-1] == 'true']
            self.assertTrue(preps)
            self.assertEqual((root / 'timeouts.log').read_text().splitlines(), ['600s', '600s'])
            for call in preps:
                self.assertEqual(call['env']['SLURM_MPI_TYPE'], 'none')
                self.assertEqual(call['mounts'], '')
                self.assertIn('--container-image=/opt/aim344/aim344.sqsh', call['argv'])
                self.assertIn('--ntasks=2', call['argv'])
                self.assertEqual(call['env'][spank + 'name'], 'aim344_check5_123')
            for call in calls:
                self.assertFalse(any(k.startswith(('PMI_', 'PMIX_')) for k in call['env']))
                if call not in preps:
                    self.assertNotIn(spank + 'name', call['env'])
                    self.assertNotIn(spank + 'env', call['env'])
            # Explicit caller image/timeout survive generated site defaults.
            env.update(NCCL_CONTAINER='/caller/native.sqsh', NCCL_TIMEOUT='240', LOCAL_STORAGE_EXIT='0',
                       SLURM_JOB_ID='456', AIM344_CONTAINER_PREP_TIMEOUT='720')
            result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            override = json.loads((root / 'suite.json').read_text())
            self.assertEqual(override['env']['NCCL_CONTAINER'], '/caller/native.sqsh')
            self.assertEqual(override['env']['NCCL_TIMEOUT'], '240')
            self.assertEqual(override['env'][spank + 'name'], 'aim344_check5_456')
            self.assertEqual((root / 'timeouts.log').read_text().splitlines()[-1], '720s')
            # Both supported naming forms and the separate storage image root.
            for name in ('pyxis_aim344_check5_456', 'pyxis_456_aim344_check5_456',
                         'pyxis_aim344_verify_456', 'pyxis_456_aim344_verify_456'):
                (data / name).mkdir()
            result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual([p.name for p in data.iterdir()], ['pyxis_aim344_check5_122'])
            # Retrieval failure must retain the current root; a busy Slurm step
            # must refuse cleanup, while a fixture failure retrieves before it.
            current = data / 'pyxis_aim344_check5_456'
            # Reproduced unsafe contracts plus adjacent malformed/failed receipts.
            # Keep all current naming forms until the entire retrieval succeeds;
            # no partial teardown on a good first row followed by a bad peer.
            retained = [data / name for name in (
                'pyxis_aim344_check5_456', 'pyxis_456_aim344_check5_456',
                'pyxis_aim344_verify_456', 'pyxis_456_aim344_verify_456')]
            failures = [
                {'LOCAL_RETRIEVAL_EXIT': '7'},
                {'LOCAL_RETRIEVAL_ROWS': 'duplicate'},
                {'LOCAL_RETRIEVAL_ROWS': 'missing'},
                {'LOCAL_RETRIEVAL_ROWS': 'unknown'},
                {'LOCAL_RETRIEVAL_ROWS': 'wrong-host'},
                {'LOCAL_RETRIEVAL_ROWS': 'extra'},
                {'LOCAL_RETRIEVAL_FAIL': '1'},
                {'LOCAL_DISTRIBUTION_EXIT': '7'},
                {'LOCAL_DIGEST_FAILURE': 'local'},
                {'LOCAL_DIGEST_FAILURE': 'remote'},
                {'LOCAL_NODELIST_EXIT': '7'},
                {'LOCAL_NODE_MAP_EXIT': '7'},
                {'SLURM_JOB_NUM_NODES': '3'},
                {'LOCAL_FIXTURE_EXIT': '1', 'LOCAL_RETRIEVAL_EXIT': '7'},
            ]
            for injection in failures:
                with self.subTest(retrieval=injection):
                    for path in retained:
                        path.mkdir(exist_ok=True)
                    prior = len((root / 'argv.log').read_text().splitlines())
                    result = subprocess.run(['bash', str(script)], env=dict(env, **injection),
                                            capture_output=True, text=True, timeout=20)
                    new_calls = [json.loads(line) for line in
                                 (root / 'argv.log').read_text().splitlines()[prior:]]
                    print('LOCAL_RETRIEVAL_CONTRACT', json.dumps(injection),
                          'verify_rc=', result.returncode,
                          'current_roots_retained=', [p.exists() for p in retained],
                          'historical_retained=', historical.is_dir(), result.stdout)
                    self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                    self.assertTrue(all(p.is_dir() for p in retained), result.stdout)
                    self.assertTrue(historical.is_dir())
                    self.assertFalse(any('--exclusive' in c['argv'] for c in new_calls))
                    self.assertIn('RESULT RETRIEVAL FAILED', result.stdout)
                    self.assertNotIn('To read this result', result.stdout)
                    self.assertTrue((root / 'aim344-verify-456-results.tar').is_file())
            # Retry after failed retrieval and reversed arrival order still work.
            result = subprocess.run(['bash', str(script)],
                                    env=dict(env, LOCAL_RETRIEVAL_ROWS='reverse'),
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse(any(p.exists() for p in retained))
            self.assertTrue(historical.is_dir())
            for variable in ('LOCAL_RETRIEVAL_FAIL', 'LOCAL_CLEANUP_BUSY', 'LOCAL_FIXTURE_EXIT'):
                current.mkdir(exist_ok=True)
                env[variable] = '1'
                result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertEqual(current.exists(), variable != 'LOCAL_FIXTURE_EXIT')
                self.assertTrue(historical.is_dir())
                if variable == 'LOCAL_FIXTURE_EXIT':
                    self.assertNotIn('storage_exit_code=', result.stdout)
                    self.assertLess(result.stdout.index('nodes_with_matching_bundle=2 of 2'),
                                    result.stdout.index('current-job container removed:'))
                env.pop(variable)
            # Missing rendezvous address follows the other early retrieval path.
            control_body = control.read_text()
            control.write_text(control_body.replace('NodeAddr=10.0.0.1', 'NodeAddr='))
            current.mkdir()
            result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn('could not resolve a rendezvous address', result.stdout)
            self.assertFalse(current.exists())
            self.assertTrue(historical.is_dir())
            self.assertLess(result.stdout.index('nodes_with_matching_bundle=2 of 2'),
                            result.stdout.index('current-job container removed:'))
            control.write_text(control_body)
            # A symlink with an exact current name never authorizes its target.
            current.symlink_to(historical, target_is_directory=True)
            result = subprocess.run(['bash', str(companion / '9.cleanup.sh')],
                                    env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(current.is_symlink())
            self.assertTrue(historical.is_dir())
            current.unlink()
            # A foreign or completed allocation cannot reach a cleanup step.
            original_control = control.read_text()
            for altered in (original_control.replace('JobState=RUNNING', 'JobState=COMPLETED'),
                            original_control.replace('UserId=%s(%s)', 'OtherId=%s(%s)')):
                control.write_text(altered)
                current.mkdir()
                result = subprocess.run(['bash', str(companion / '9.cleanup.sh')],
                                        env=env, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(current.is_dir())
                current.rmdir()
            control.write_text(original_control)
            # Literal validation precedes scheduler lookup; no historical ID repair.
            for job_id in ('', '456/../122', '456*', '0'):
                result = subprocess.run(['bash', str(companion / '9.cleanup.sh')],
                                        env=dict(env, SLURM_JOB_ID=job_id),
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(historical.is_dir())
            # An apparently successful remove that leaves the root is a failure.
            enroot_body = enroot.read_text()
            enroot.write_text('#!/bin/bash\nexit 0\n')
            current.mkdir()
            result = subprocess.run(['bash', str(companion / '9.cleanup.sh')],
                                    env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(current.is_dir())
            enroot.write_text(enroot_body)
            # Foreign-owner metadata is a local stub, not a privileged chown.
            stat_stub = binaries / 'stat'
            stat_stub.write_text('#!/bin/bash\nif [[ "$2" == %u && "${@: -1}" == */pyxis_aim344_check5_456 ]]; then printf "65534\\n"; else exec /usr/bin/stat "$@"; fi\n')
            stat_stub.chmod(0o755)
            result = subprocess.run(['bash', str(companion / '9.cleanup.sh')],
                                    env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(current.is_dir())
            stat_stub.unlink()
            for _ in range(2):
                result = subprocess.run(['bash', str(companion / '9.cleanup.sh')],
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(historical.is_dir())
            # Historical interactive entry uses the same exact selected suite and env.
            interactive = companion / '10.healthcheck-nccl.sh'
            interactive.write_text((LAB / '10.healthcheck-nccl.sh').read_text())
            env.update(SLURM_CPUS_ON_NODE='192', RESULTS_DIR=str(root / 'interactive'))
            result = subprocess.run(['bash', str(interactive)], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            interactive_launch = json.loads((root / 'suite.json').read_text())
            self.assertEqual(interactive_launch['env']['NCCL_CONTAINER'], '/caller/native.sqsh')
            self.assertEqual(interactive_launch['env'][spank + 'name'], 'aim344_check5_456')
            self.assertNotIn('OMPI_MCA_btl_tcp_if_include', interactive_launch['env'])
            # Cold preparation failure cannot be reported as collective success.
            (root / 'suite.json').unlink()
            env['LOCAL_PREP_EXIT'] = '124'
            current.mkdir()
            result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn('check5_exit_code=124', result.stdout)
            self.assertFalse(current.exists())
            self.assertTrue(historical.is_dir())
            self.assertFalse((root / 'suite.json').exists())
            env.update(LOCAL_PREP_EXIT='0', LOCAL_SUITE_EXIT='17')
            current.mkdir()
            result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn('check5_exit_code=17', result.stdout)
            self.assertFalse(current.exists())
            self.assertTrue(historical.is_dir())
            env['SLURM_JOB_PARTITION'] = 'wrong-queue'
            result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn('does not match assigned PARTITION', result.stderr)
