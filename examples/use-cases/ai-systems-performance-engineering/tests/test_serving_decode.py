"""CPU tests of Optional B (tools/serve.sh, lib/serving_decode.py), the --config option of
lib/serve_llm.py and prepare-event.sh --with-serving. Standard library only.

No server runs: vLLM, Slurm and the servers' counters are replaced by fakes. Step times
in these fixtures are synthetic, not measurements. The memory-read ceiling of 663 GB/s
is the median recorded by hold-and-measure.sh on two g7.48xlarge on 2026-11-03.
"""
from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'lib'))
import serving_decode  # noqa: E402

CEILING = 663e9
METRICS = ('vllm:generation_tokens_total', 'vllm:request_success_total',
           'vllm:inter_token_latency_seconds_count', 'vllm:inter_token_latency_seconds_sum')
SMOL_PIN = {'model_id': 'HuggingFaceTB/SmolLM2-1.7B', 'model_revision': 'effd688a12921b4cc83e3312b6feb579f70f9c71'}


def stub(directory, name, script):
    path = Path(directory) / name
    path.write_text('#!/usr/bin/env bash\n' + script)
    path.chmod(0o755)


class FakeServers:
    """Counters of vLLM servers. A request adds its tokens and its decode intervals.

    step[endpoint] is the synthetic time of one decode step. stale_reads makes the first
    reads after a level return the counters from before it, as a server that has not
    updated them yet. The highest number of requests in flight is kept per endpoint.
    """

    def __init__(self, endpoints, step, stale_reads=0, extra_tokens=0):
        self.counters = {endpoint: dict.fromkeys(METRICS, 0.0) for endpoint in endpoints}
        self.step, self.stale_reads, self.extra_tokens = step, stale_reads, extra_tokens
        self.in_flight = dict.fromkeys(endpoints, 0)
        self.most_in_flight = dict.fromkeys(endpoints, 0)
        self.requests = dict.fromkeys(endpoints, 0)
        self.last = {}
        self.lock = threading.Lock()

    def request(self, endpoint, tokens=serving_decode.OUTPUT_TOKENS):
        with self.lock:
            self.in_flight[endpoint] += 1
            self.most_in_flight[endpoint] = max(self.most_in_flight[endpoint], self.in_flight[endpoint])
        time.sleep(0.005)
        with self.lock:
            self.in_flight[endpoint] -= 1
            self.requests[endpoint] += 1
            counters = self.counters[endpoint]
            counters['vllm:generation_tokens_total'] += tokens + self.extra_tokens
            counters['vllm:request_success_total'] += 1
            counters['vllm:inter_token_latency_seconds_count'] += tokens - 1
            counters['vllm:inter_token_latency_seconds_sum'] += (tokens - 1) * self.step[endpoint]
        return dict(endpoint=endpoint, output_tokens=tokens, prompt_tokens=13, elapsed_seconds=0.005)

    def snapshot(self, endpoint):
        with self.lock:
            if self.stale_reads and endpoint in self.last:
                self.stale_reads -= 1
                return dict(self.last[endpoint])
            self.last[endpoint] = dict(self.counters[endpoint])
            return dict(self.counters[endpoint])


class DecodeMeasurementTests(unittest.TestCase):
    def test_weight_bytes_and_the_bound_at_the_recorded_ceiling(self):
        self.assertEqual(serving_decode.WEIGHT_BYTES, 3_422_752_768)
        lines = serving_decode.bound_lines(CEILING)
        self.assertEqual(lines, [
            'Reading the weights (3.42 GB) at your memory-read ceiling (663 GB/s per GPU) takes 5.16 ms.',
            'At one request per GPU, decode runs at most 194 tokens/s per GPU. MBU is 5.16 ms / ms per step.'])
        self.assertIn('run bash tools/hold-and-measure.sh', serving_decode.bound_lines(None)[0])

    def test_step_is_found_by_host_and_srun_pid(self):
        text = ('StepId=42.0 UserId=1000 State=COMPLETED SrunHost:Pid=login-1:4242 Name=check\n'
                'StepId=42.1 UserId=1000 State=RUNNING SrunHost:Pid=gpu-node-1:4242 Name=bash\n'
                'StepId=42.2 UserId=1000 State=RUNNING SrunHost:Pid=login-1:4243 Name=bash\n'
                'StepId=42.3 UserId=1000 State=RUNNING SrunHost:Pid=login-1:4242 Name=bash\n')
        self.assertEqual(serving_decode.find_step(text, '42', 4242, 'login-1.ec2.internal'), '42.3')
        self.assertIsNone(serving_decode.find_step(text, '42', 4244, 'login-1'))
        self.assertIsNone(serving_decode.find_step(text, '43', 4242, 'login-1'))
        self.assertIsNone(serving_decode.find_step('No job steps exist for this job.\n', '42', 4242, 'login-1'))

    def test_image_must_exist_with_its_hash_and_be_checked_on_both_nodes(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / 'images' / serving_decode.IMAGE_NAME
            self.assertEqual(serving_decode.image_path(dict(AIM347_HOME=directory)), image)
            self.assertEqual(serving_decode.image_path(dict(VLLM_IMAGE='/elsewhere.sqsh')), Path('/elsewhere.sqsh'))
            self.assertIn('--with-serving', serving_decode.image_problem(image, 2))
            image.parent.mkdir()
            image.write_text('image')
            Path(f'{image}.sha256').write_text(f'hash  {image}\n')
            self.assertIn('has not been checked on both GPU nodes', serving_decode.image_problem(image, 2))
            Path(f'{image}.verified').write_text('gpu-node-1 OK 2026-11-04T00:00:00Z\n'
                                                 'gpu-node-1 OK 2026-11-04T00:10:00Z\n')
            self.assertIn('has not been checked on both GPU nodes', serving_decode.image_problem(image, 2))
            with Path(f'{image}.verified').open('a') as stream:
                stream.write('gpu-node-2 OK 2026-11-04T00:10:00Z\n')
            self.assertIsNone(serving_decode.image_problem(image, 2))

    def test_level_sends_each_server_its_requests_and_reads_its_counters(self):
        endpoints = [f'http://10.0.0.{node}:{8100 + gpu}' for node in (1, 2) for gpu in range(8)]
        step = {endpoint: 0.008 if n % 2 else 0.010 for n, endpoint in enumerate(endpoints)}
        servers = FakeServers(endpoints, step, stale_reads=3)
        sleeps = []
        level = serving_decode.measure_level(endpoints, 4, request=servers.request, snap=servers.snapshot,
                                             sleep=sleeps.append)
        self.assertEqual(level['requests'], 16 * 4 * serving_decode.ROUNDS)
        self.assertEqual(set(servers.requests.values()), {4 * serving_decode.ROUNDS})
        self.assertTrue(all(1 <= count <= 4 for count in servers.most_in_flight.values()))
        self.assertEqual(sleeps, [0.1] * 3)
        record = level['servers'][endpoints[0]]
        self.assertEqual(record['decode_steps'], 4 * serving_decode.ROUNDS * (serving_decode.OUTPUT_TOKENS - 1))
        self.assertAlmostEqual(record['server_mean_itl_seconds'], 0.010)

        summary = serving_decode.level_summary(level, CEILING)
        self.assertAlmostEqual(summary['seconds_per_step'], 0.009)
        self.assertAlmostEqual(summary['tokens_per_second_per_gpu'], 4 / 0.009)
        self.assertAlmostEqual(summary['mbu'], 3_422_752_768 / CEILING / 0.009)
        self.assertIsNone(serving_decode.level_summary(level)['mbu'])

    def test_level_refuses_counters_that_other_traffic_changed(self):
        endpoints = ['http://10.0.0.1:8100', 'http://10.0.0.1:8101']
        servers = FakeServers(endpoints, dict.fromkeys(endpoints, 0.009), extra_tokens=1)
        with self.assertRaisesRegex(ValueError, 'Server counters do not match'):
            serving_decode.measure_level(endpoints, 1, request=servers.request, snap=servers.snapshot,
                                         sleep=lambda seconds: None)

    def test_rows_and_chart(self):
        summaries = [dict(requests_per_gpu=1, seconds_per_step=0.009, tokens_per_second_per_gpu=1 / 0.009,
                          mbu=0.5736),
                     dict(requests_per_gpu=16, seconds_per_step=0.0125, tokens_per_second_per_gpu=16 / 0.0125,
                          mbu=None)]
        self.assertEqual(serving_decode.row_line(summaries[0]),
                         '               1          9.00                       111   57 %')
        self.assertEqual(serving_decode.row_line(summaries[1]),
                         '              16         12.50                     1,280   n/a')
        lines = serving_decode.chart_lines(summaries)
        self.assertEqual(lines[0], 'Decode tokens/s per GPU by requests per GPU (higher is better)')
        self.assertTrue(lines[1].startswith('  1 request per GPU  '))
        self.assertTrue(lines[1].endswith('111  (MBU 57 %)'))
        self.assertTrue(lines[2].startswith('  16 requests per GPU  '))
        self.assertTrue(lines[2].endswith('1,280  (11.5x one request, MBU n/a)'))
        self.assertIn('tokens/s per GPU', lines[-1])


class FakeProcess:
    def __init__(self, pid=4242, ends_with=None):
        self.pid, self.returncode = pid, ends_with

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired('srun', timeout)
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


class ServeRunTests(unittest.TestCase):
    """serving_decode.run with fake srun, scontrol, scancel, servers and clock."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.results, self.state, home = root / 'results', root / 'state', root / 'home'
        (self.results / 'ceilings').mkdir(parents=True)
        (self.results / 'ceilings' / 'ceilings.json').write_text(json.dumps(
            dict(gemm_tflops_per_gpu=167.9e0, memory_read_bytes_per_second_per_gpu=CEILING)))
        self.state.mkdir()
        self.image = home / 'images' / serving_decode.IMAGE_NAME
        self.image.parent.mkdir(parents=True)
        self.image.write_text('image')
        Path(f'{self.image}.sha256').write_text(f'hash  {self.image}\n')
        Path(f'{self.image}.verified').write_text('gpu-node-1 OK t\ngpu-node-2 OK t\n')
        self.env = dict(LLM_RESULTS_DIR=str(self.results), AIM347_HOME=str(home), GPUS_PER_NODE='8',
                        SLURM_JOB_ID='42', SLURM_JOB_NUM_NODES='2', AIM347_ALLOCATION_ID='42',
                        SLURM_JOB_NODELIST='gpu-node-[1-2]', SLURM_JOB_CPUS_PER_NODE='192(x2)')
        self.endpoints = [f'http://10.0.0.{node + 1}:{8100 + gpu}' for node in range(2) for gpu in range(8)]
        self.servers = FakeServers(self.endpoints, dict.fromkeys(self.endpoints, 0.009))
        self.calls, self.launches, self.lines = [], [], []
        self.now = 0.0

    def tearDown(self):
        self.directory.cleanup()

    def popen(self, command, **kwargs):
        self.launches.append((command, kwargs))
        run_dir = self.results / kwargs['env']['SERVING_RUN']
        for node in range(2):
            for replica in range(8):
                path = run_dir / f'node-{node}' / f'replica-{replica}.json'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{}')
        self.process = FakeProcess()
        return self.process

    def runner(self, command, **kwargs):
        self.calls.append(command)
        if '--collect-manifest' in command:
            output = Path(command[command.index('--output') + 1])
            (output / 'endpoints.txt').write_text('\n'.join(self.endpoints) + '\n')
            return SimpleNamespace(returncode=0, stdout='{}', stderr='')
        if command[:3] == ['scontrol', 'show', 'step']:
            return SimpleNamespace(returncode=0, stderr='', stdout=(
                'StepId=42.0 UserId=1000 State=RUNNING SrunHost:Pid=login-1:999 Name=hold\n'
                'StepId=42.5 UserId=1000 State=RUNNING SrunHost:Pid=login-1:4242 Name=bash\n'))
        if command[:2] == ['scancel', '--signal=TERM']:
            self.process.returncode = 0
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    def clock(self):
        self.now += 1.0
        return self.now

    def out(self, *values, **kwargs):
        self.lines.append(' '.join(str(value) for value in values))

    def run_tool(self, **overrides):
        options = dict(popen=self.popen, runner=self.runner, health=lambda endpoint: True,
                       request=self.servers.request, snap=self.servers.snapshot, clock=self.clock,
                       sleep=lambda seconds: None, out=self.out, hostname='login-1')
        options.update(overrides)
        return serving_decode.run(self.env, self.state, **options)

    def test_serves_measures_every_level_and_stops_the_step(self):
        self.assertEqual(self.run_tool(), 0)
        command, kwargs = self.launches[0]
        self.assertEqual(command, ['bash', str(LAB / '8.serve-llm.sh'), '--tensor-parallel-size', '1',
                                   '--config', 'configs/smol-2048.json'])
        self.assertEqual(kwargs['env']['VLLM_IMAGE'], str(self.image))
        self.assertTrue(kwargs['start_new_session'])
        run_dir = self.results / kwargs['env']['SERVING_RUN']
        self.assertTrue(run_dir.name.startswith('serving-'))
        self.assertIn(['scancel', '--signal=TERM', '42.5'], self.calls)
        record = json.loads((run_dir / 'serving.json').read_text())
        self.assertEqual([level['requests_per_gpu'] for level in record['levels']], [1, 4, 16, 64])
        self.assertEqual(record['exit_status'], 0)
        self.assertEqual(record['serving_step_exit'], 0)
        self.assertEqual(record['endpoints'], self.endpoints)
        self.assertEqual(record['prompt_tokens'], 13)
        text = '\n'.join(self.lines)
        self.assertIn('[aim347] vLLM 0.20.2 serving HuggingFaceTB/SmolLM2-1.7B at effd688a: one server per GPU, '
                      '16 servers on 2 nodes', text)
        self.assertIn('[aim347] all 16 servers answer after', text)
        self.assertIn(serving_decode.TABLE_HEADER, text)
        self.assertIn('               1          9.00                       111   57 %', text)
        self.assertIn('              64          9.00                     7,111   57 %', text)
        self.assertIn('[aim347] the servers have stopped and the GPUs are free', text)
        self.assertEqual(self.lines[-1].splitlines()[0],
                         'Decode tokens/s per GPU by requests per GPU (higher is better)')
        # The progression table reads only stage directories with run.json.
        self.assertFalse((run_dir / 'run.json').exists())

    def test_a_serving_step_that_ends_early_is_reported_with_its_log(self):
        def popen(command, **kwargs):
            self.popen(command, **kwargs)
            (self.results / kwargs['env']['SERVING_RUN'] / 'launch.log').write_text('ValueError: fixture failure\n')
            self.process.returncode = 1
            return self.process

        self.assertEqual(self.run_tool(popen=popen), 4)
        text = '\n'.join(self.lines)
        self.assertIn('the serving step ended with exit status 1 before every server answered', text)
        self.assertIn('ValueError: fixture failure', text)
        self.assertNotIn(['scancel', '--signal=TERM', '42.5'], self.calls)

    def test_servers_that_never_answer_time_out_and_are_stopped(self):
        self.assertEqual(self.run_tool(health=lambda endpoint: False), 4)
        text = '\n'.join(self.lines)
        self.assertIn(f'only 0 of 16 servers answered within {serving_decode.STARTUP_SECONDS} s', text)
        self.assertIn('[aim347] 15 s: 0 of 16 servers answer', text)
        self.assertIn(['scancel', '--signal=TERM', '42.5'], self.calls)

    def test_counters_that_do_not_match_exit_5_and_stop_the_servers(self):
        self.servers.extra_tokens = 1
        self.assertEqual(self.run_tool(), 5)
        self.assertIn('the server counters at 1 requests per GPU cannot be used', '\n'.join(self.lines))
        self.assertIn(['scancel', '--signal=TERM', '42.5'], self.calls)

    def test_ctrl_c_stops_the_servers(self):
        def request(endpoint, tokens=serving_decode.OUTPUT_TOKENS):
            raise KeyboardInterrupt

        self.assertEqual(self.run_tool(request=request), 130)
        self.assertIn(['scancel', '--signal=TERM', '42.5'], self.calls)
        self.assertIn('[aim347] the servers have stopped and the GPUs are free', self.lines)

    def test_a_closed_terminal_still_stops_the_servers(self):
        # A hang-up raises Stopped; every later write to the closed terminal fails with EIO.
        hung_up = []

        def request(endpoint, tokens=serving_decode.OUTPUT_TOKENS):
            hung_up.append(True)
            raise serving_decode.Stopped(1)

        def out(*values, **kwargs):
            if hung_up:
                raise OSError(5, 'Input/output error')
            self.out(*values, **kwargs)

        self.assertEqual(self.run_tool(request=request, out=out), 130)
        self.assertIn(['scancel', '--signal=TERM', '42.5'], self.calls)
        self.assertFalse((self.state / 'serving').exists())

    def test_keeps_a_marker_while_the_servers_run(self):
        seen = []

        def request(endpoint, tokens=serving_decode.OUTPUT_TOKENS):
            seen.append(serving_decode.workshop.read_serving(self.state))
            return self.servers.request(endpoint, tokens)

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            self.assertEqual(self.run_tool(request=request), 0)
        self.assertEqual(seen[0]['pid'], os.getpid())
        self.assertFalse((self.state / 'serving').exists())
        # A second serve.sh refuses while the first one's marker names a live process.
        (self.state / 'serving').write_text(json.dumps(dict(pid=os.getpid(), run='serving-x')))
        with redirect_stderr(stderr):
            self.assertEqual(self.run_tool(), 2)
        self.assertIn('AIM347: tools/serve.sh is already running in another shell.', stderr.getvalue())
        self.assertEqual(len(self.launches), 1)

    def test_refuses_without_the_image_and_while_a_stage_runs(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            self.assertEqual(self.run_tool(), 0)
            (self.state / 'running').write_text(json.dumps(dict(pid=os.getpid(), stages=['efa'], invocation='x')))
            self.assertEqual(self.run_tool(), 2)
            self.image.unlink()
            self.assertEqual(self.run_tool(), 2)
        self.assertIn('a stage is still running', stderr.getvalue())
        self.assertIn('Provisioning imports it only when facilitator/prepare-event.sh runs with --with-serving',
                      stderr.getvalue())
        self.assertEqual(len(self.launches), 1)


class ServeLlmConfigTests(unittest.TestCase):
    """lib/serve_llm.py checks the snapshot pin against the config that --config names."""

    def launch(self, root, extra):
        import serve_llm
        model = root / 'model'
        model.mkdir(exist_ok=True)
        (model / 'aim347-pin.json').write_text(json.dumps(SMOL_PIN))
        cuda = SimpleNamespace(device_count=lambda: 8, get_device_properties=lambda i: SimpleNamespace(uuid=f'GPU-{i}'))
        argv = ['serve_llm.py', '--tensor-parallel-size', '1', '--gpus-per-node', '8', '--model', str(model),
                '--output', str(root / 'node-0'), *extra]
        read_text = Path.read_text

        def read(path, *args, **kwargs):
            if str(path) == '/proc/sys/net/ipv4/ip_local_port_range':
                return '32768 60999'
            return read_text(path, *args, **kwargs)

        environment = dict(SLURM_JOB_ID='42', SLURMD_NODENAME='gpu-node-1',
                           CUDA_VISIBLE_DEVICES=','.join(str(i) for i in range(8)))
        with patch.object(sys, 'argv', argv), patch.dict(os.environ, environment, clear=True), \
                patch.dict(sys.modules, vllm=SimpleNamespace(__version__='0.20.2'), torch=SimpleNamespace(cuda=cuda)), \
                patch.object(Path, 'read_text', read), \
                patch('serve_llm.subprocess.Popen', side_effect=lambda *a, **k: SimpleNamespace(
                    pid=100, poll=lambda: None)) as spawn, \
                patch('serve_llm.signal.signal'), patch('serve_llm.stop_replicas'), \
                patch('serve_llm.time.sleep', side_effect=KeyboardInterrupt):
            try:
                serve_llm.main()
            except KeyboardInterrupt:
                pass
        return spawn.call_args_list

    def test_smol_config_serves_the_prepared_snapshot(self):
        import serve_llm
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = self.launch(root, ['--config', 'configs/smol-2048.json'])
            self.assertEqual(len(calls), 8)
            self.assertEqual([call.kwargs['env']['CUDA_VISIBLE_DEVICES'] for call in calls], [str(i) for i in range(8)])
            self.assertEqual(json.loads((root / 'node-0' / 'replica-0.json').read_text())['model_pin'], SMOL_PIN)
            argv = ['serve_llm.py', '--collect-manifest', '--output', str(root), '--gpus-per-node', '8',
                    '--tensor-parallel-size', '1', '--model', str(root / 'model'), '--config', 'configs/smol-2048.json']
            env = dict(SLURM_JOB_ID='42', SLURM_JOB_NODELIST='gpu-node-1', SLURM_JOB_CPUS_PER_NODE='192')
            with patch.object(sys, 'argv', argv), patch.dict(os.environ, env, clear=True), \
                    patch('serve_llm.subprocess.check_output', side_effect=['gpu-node-1\n', 'NodeAddr=10.0.0.1']), \
                    patch('sys.stdout', new_callable=io.StringIO):
                serve_llm.main()
            manifest = json.loads((root / 'manifest.json').read_text())
            self.assertEqual(manifest['workload']['model_id'], 'HuggingFaceTB/SmolLM2-1.7B')
            self.assertEqual(manifest['placement'], dict(tensor_parallel_size=1, replicas=8))
            self.assertEqual((root / 'endpoints.txt').read_text().split(),
                             [f'http://10.0.0.1:{8100 + i}' for i in range(8)])

    def test_default_config_refuses_the_smol_snapshot(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, 'pin mismatch'):
            self.launch(Path(directory), [])


class PrepareServingImageTests(unittest.TestCase):
    """prepare-event.sh with stubs; the training image exists and verifies, data preparation fails."""

    def prepare(self, directory, *extra):
        root = Path(directory)
        bin_dir, stage_dir, calls = root / 'bin', root / 'stage', root / 'calls'
        bin_dir.mkdir(exist_ok=True)
        (stage_dir / 'images').mkdir(parents=True, exist_ok=True)
        image = stage_dir / 'images' / 'aim347-lab.sqsh'
        if not image.exists():
            image.write_text('training image')
            digest = subprocess.run(['sha256sum', str(image)], capture_output=True, text=True, check=True).stdout.split()[0]
            Path(f'{image}.sha256').write_text(f'{digest}  {image}\n')
        stub(bin_dir, 'id', '[[ $1 == -u ]] && echo 0\nexit 0\n')
        stub(bin_dir, 'scontrol', 'case "$2" in hostnames) echo gpu-node-1; echo gpu-node-2 ;; '
                                  'node) echo "NodeName=$4 NodeAddr=10.0.0.1 CPUEfctv=192" ;; esac\n')
        stub(bin_dir, 'findmnt', 'echo lustre\n')
        stub(bin_dir, 'srun', '''nodes=1 container=0
while [[ $1 == -* ]]; do
  [[ $1 == -N2 ]] && nodes=2
  [[ $1 == --container-image=* ]] && container=1
  [[ $1 == -w ]] && shift
  shift
done
[[ $1 == ip ]] && { echo '[{"dev": "enp71s0"}]'; exit 0; }
((container)) && exit 1
status=0
for node in $(seq "$nodes"); do SLURMD_NODENAME=gpu-node-$node "$@" || status=1; done
exit $status
''')
        stub(bin_dir, 'curl', 'exit 0\n')
        stub(bin_dir, 'docker', f'echo "docker $*" >> {calls}\n')
        stub(bin_dir, 'enroot', f'echo "enroot $*" >> {calls}\n[[ $1 == import && $2 == -o ]] && echo vllm > "$3"\n')
        stub(bin_dir, 'unsquashfs', 'exit 0\n')
        result = subprocess.run(['bash', str(LAB / 'facilitator/prepare-event.sh'), '--stage-dir', str(stage_dir),
                                 '--slurm-version', '0.0', '--instance-type', 'g7.48xlarge',
                                 '--grafana-url', 'https://203.0.113.10/grafana/', *extra],
                                env=dict(os.environ, PATH=f'{bin_dir}:{os.environ["PATH"]}', SLURM_JOB_ID='42',
                                         SLURM_JOB_NODELIST='gpu-node-[1-2]'),
                                capture_output=True, text=True, timeout=120)
        return result, (calls.read_text().splitlines() if calls.exists() else []), stage_dir / 'images'

    def test_with_serving_imports_and_verifies_the_vllm_image(self):
        source = 'vllm/vllm-openai@sha256:68b773151407ca28c05479c02c0c08a573285ba197c8ff35d2103acd97aff78c'
        with tempfile.TemporaryDirectory() as directory:
            result, calls, images = self.prepare(directory, '--with-serving')
            self.assertEqual(result.returncode, 4, result.stderr)
            self.assertIn('prepare-event: keeping', result.stdout)
            self.assertIn('prepare-event: serving image ok (', result.stdout)
            self.assertIn('prepare-event: serving image on both nodes ok (', result.stdout)
            self.assertEqual(calls[-3:], [f'docker pull {source}', f'docker tag {source} aim347-vllm:0.20.2',
                                          f'enroot import -o {images}/vllm-v0.20.2.sqsh.tmp dockerd://aim347-vllm:0.20.2'])
            serving = images / 'vllm-v0.20.2.sqsh'
            self.assertEqual(serving.read_text(), 'vllm\n')
            self.assertFalse(Path(f'{serving}.tmp').exists())
            subprocess.run(['sha256sum', '--check', '--strict', '--status', f'{serving}.sha256'], check=True)
            verified = Path(f'{serving}.verified').read_text().splitlines()
            self.assertEqual([line.split()[:2] for line in verified], [['gpu-node-1', 'OK'], ['gpu-node-2', 'OK']])

            # A rerun keeps an image whose hash checks.
            result, calls, _ = self.prepare(directory, '--with-serving')
            self.assertIn(f'prepare-event: keeping {serving} (its SHA-256 checks)', result.stdout)
            self.assertEqual(sum(call.startswith('docker pull') for call in calls), 1)

    def test_without_the_option_no_serving_image_is_prepared(self):
        with tempfile.TemporaryDirectory() as directory:
            result, calls, images = self.prepare(directory)
            self.assertEqual(result.returncode, 4, result.stderr)
            self.assertNotIn('serving image', result.stdout)
            self.assertFalse(any(call.startswith(('docker pull', 'enroot')) for call in calls))
            self.assertFalse((images / 'vllm-v0.20.2.sqsh').exists())


if __name__ == '__main__':
    unittest.main()
