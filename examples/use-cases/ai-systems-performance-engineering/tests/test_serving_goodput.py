"""Synthetic serving accounting fixtures; no actual inference performance."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch, call
import io
import itertools
import json
import os
from types import SimpleNamespace
import urllib.error
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
from serving_goodput import paired_serving_report, quality_valid, serving_goodput, stream_request


class ServingGoodputTests(unittest.TestCase):
    def launch_replicas(self, root, *, tp=2, gpus=8, inherited=None, extra=(),
                        ephemeral='32768 60999'):
        """Exercise main's actual Popen boundary, without vLLM, GPUs or Slurm."""
        import serve_llm
        cfg = json.loads((Path(__file__).resolve().parents[1] / 'configs/llm.json').read_text())
        model = root / 'model'
        model.mkdir(exist_ok=True)
        (model / 'aim347-pin.json').write_text(json.dumps(
            {key: cfg[key] for key in ('model_id', 'model_revision')}))
        environment = dict(SLURM_JOB_ID='fixture-job', SLURMD_NODENAME='node-a',
                           CUDA_VISIBLE_DEVICES=','.join(f'GPU-{i}' for i in range(gpus)),
                           VLLM_HOST_IP='192.0.2.1')
        if inherited is not None:
            environment['VLLM_PORT'] = inherited
        cuda = SimpleNamespace(device_count=lambda: gpus,
                               get_device_properties=lambda i: SimpleNamespace(uuid=f'GPU-{i}'))
        argv = ['serve_llm.py', '--tensor-parallel-size', str(tp), '--gpus-per-node', str(gpus),
                '--model', str(model), '--output', str(root / 'node-0'), *extra]
        read_text = Path.read_text

        def read(path, *args, **kwargs):
            if str(path) == '/proc/sys/net/ipv4/ip_local_port_range':
                return ephemeral
            return read_text(path, *args, **kwargs)

        children = [Mock(pid=100 + i, poll=Mock(return_value=None)) for i in range(gpus // tp)]
        with patch.object(sys, 'argv', argv), patch.dict(os.environ, environment, clear=True), \
                patch.dict(sys.modules, vllm=SimpleNamespace(__version__='0.20.2'),
                           torch=SimpleNamespace(cuda=cuda)), patch.object(Path, 'read_text', read), \
                patch('serve_llm.subprocess.Popen', side_effect=children) as spawn, \
                patch('serve_llm.signal.signal'), patch('serve_llm.stop_replicas') as stop, \
                patch('serve_llm.time.sleep', side_effect=KeyboardInterrupt):
            try:
                serve_llm.main()
            except KeyboardInterrupt:
                pass
            except SystemExit:
                spawn.assert_not_called()
                raise
            finally:
                self.assertEqual(os.environ.get('VLLM_PORT'), inherited)
            stop.assert_called_once_with(children)
        return spawn.call_args_list, cfg

    def test_simultaneous_replicas_get_distinct_internal_port_environments(self):
        for inherited in (None, '50103', 'not-a-port'):
            with self.subTest(inherited=inherited), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                calls, _ = self.launch_replicas(root, inherited=inherited)
                self.assertEqual(len(calls), 4)
                self.assertEqual([c.kwargs['env'].get('VLLM_PORT') for c in calls],
                                 ['20000', '21000', '22000', '23000'])
                for replica, child in enumerate(calls):
                    command, env = child.args[0], child.kwargs['env']
                    self.assertEqual(command[command.index('--port') + 1], str(8100 + replica))
                    self.assertEqual(env['CUDA_VISIBLE_DEVICES'], f'GPU-{2 * replica},GPU-{2 * replica + 1}')
                    self.assertEqual(env['VLLM_HOST_IP'], '192.0.2.1')
                    self.assertTrue(child.kwargs['start_new_session'])
                    record = json.loads((root / 'node-0' / f'replica-{replica}.json').read_text())
                    self.assertEqual(record['environment'], {'VLLM_PORT': env['VLLM_PORT']})

    def test_internal_port_validation_and_eight_replica_bound(self):
        from serve_llm import replica_port_starts
        for replicas in (1, 4, 8):
            starts = list(replica_port_starts(replicas, 8100, 20000, ephemeral=(32768, 60999)))
            self.assertEqual(starts, [20000 + 1000 * i for i in range(replicas)])
            spans = [set(range(start, start + 1000)) for start in starts]
            self.assertEqual(len(set.union(*spans)), replicas * 1000)
            self.assertTrue(set.union(*spans).isdisjoint(range(8100, 8100 + replicas)))
        self.assertEqual(list(replica_port_starts(1, 8100, 64536, ephemeral=(32768, 60999))), [64536])
        for replicas, api, internal, ephemeral in (
                (0, 8100, 20000, (32768, 60999)), (9, 8100, 20000, (32768, 60999)),
                (4, 0, 20000, (32768, 60999)), (4, 65533, 20000, (32768, 60999)),
                (4, 8100, 1023, (32768, 60999)), (1, 8100, 64537, (32768, 60999)),
                (4, 8100, 8000, (32768, 60999)), (4, 19998, 20000, (32768, 60999)),
                (4, 8100, 32768, (32768, 60999)), (4, 8100, 29000, (32768, 60999)),
                (4, 8100, 20000, (20000, 60999)), (4, 8100, 20000, (60999, 32768))):
            with self.subTest(replicas=replicas, api=api, internal=internal, ephemeral=ephemeral), \
                    self.assertRaises(ValueError):
                replica_port_starts(replicas, api, internal, ephemeral=ephemeral)
        for tp, gpus, extra, ephemeral in (
                (1, 9, (), '32768 60999'),
                (2, 8, ('--internal-port-base', '65535'), '32768 60999'),
                (2, 8, (), '20000 60999'), (2, 8, (), 'unreadable'),
                (2, 8, (), '32768'), (2, 8, ('--port-base', '21000'), '32768 60999')):
            with self.subTest(tp=tp, gpus=gpus, extra=extra, ephemeral=ephemeral), \
                    tempfile.TemporaryDirectory() as directory, \
                    patch('sys.stderr', new_callable=io.StringIO), \
                    patch('serve_llm.subprocess.Popen') as spawn:
                with self.assertRaises(SystemExit) as error:
                    self.launch_replicas(Path(directory), tp=tp, gpus=gpus, extra=extra, ephemeral=ephemeral)
                self.assertEqual(error.exception.code, 2)
                spawn.assert_not_called()
                self.assertFalse((Path(directory) / 'node-0').exists())

    def test_launch_and_manifest_preserve_placement_ports_and_internal_environment(self):
        from serve_llm import collect_manifest
        for tp, base, legacy in ((8, 20000, True), (8, 20000, False), (2, 24000, False), (1, 24000, False)):
            with self.subTest(tp=tp, base=base, legacy=legacy), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                calls, cfg = self.launch_replicas(root, tp=tp, extra=['--internal-port-base', str(base)])
                self.assertEqual(len(calls), 8 // tp)
                for replica, child in enumerate(calls):
                    env = child.kwargs['env']
                    self.assertEqual(env['CUDA_VISIBLE_DEVICES'], ','.join(
                        f'GPU-{i}' for i in range(replica * tp, (replica + 1) * tp)))
                    self.assertEqual(env['VLLM_PORT'], str(base + 1000 * replica))
                path = root / 'node-0' / 'replica-0.json'
                original = json.loads(path.read_text())
                if legacy:
                    del original['environment']
                    path.write_text(json.dumps(original))
                snapshots = {p: p.read_bytes() for p in root.glob('node-*/*.json')}
                kwargs = dict(addresses=['192.0.2.10'], gpus_per_node=8, tensor_parallel_size=tp,
                              max_num_seqs=256, max_num_batched_tokens=8192, port_base=8100,
                              cpus_per_node=4, cfg=cfg, model_path=str(root / 'model'),
                              expected_job='fixture-job', expected_nodes=['node-a'], internal_port_base=base)
                if not legacy:
                    for bad in (None, {}, {'VLLM_PORT': str(base + 1000)}, {'VLLM_PORT': base}):
                        with self.subTest(bad=bad):
                            path.write_text(json.dumps(original | {'environment': bad}))
                            with self.assertRaisesRegex(ValueError, 'internal port environment'):
                                collect_manifest(root, **kwargs)
                            self.assertFalse((root / 'manifest.json').exists())
                    if tp < 8:
                        record = dict(original)
                        del record['environment']
                        path.write_text(json.dumps(record))
                        with self.assertRaisesRegex(ValueError, 'internal port environment'):
                            collect_manifest(root, **kwargs)
                        path.write_bytes(snapshots[path])
                        second = root / 'node-0' / 'replica-1.json'
                        record = json.loads(second.read_text())
                        second.write_text(json.dumps(record | {'environment': original['environment']}))
                        with self.assertRaisesRegex(ValueError, 'internal port environment'):
                            collect_manifest(root, **kwargs)
                        second.write_bytes(snapshots[second])
                    path.write_bytes(snapshots[path])
                result = collect_manifest(root, **kwargs)
                self.assertEqual(result, json.loads((root / 'manifest.json').read_text()))
                self.assertEqual(result['gpu_budget'], 8)
                self.assertEqual(result['placement'], dict(tensor_parallel_size=tp, replicas=8 // tp))
                self.assertEqual(result['allocation']['gpu_uuids'], [f'GPU-{i}' for i in range(8)])
                self.assertEqual(result['replica_environments'], [
                    dict(node=0, replica=i, environment=None if legacy else {'VLLM_PORT': str(base + 1000 * i)})
                    for i in range(8 // tp)])
                self.assertEqual((root / 'endpoints.txt').read_text().splitlines(),
                                 [f'http://192.0.2.10:{8100 + i}' for i in range(8 // tp)])
                self.assertEqual(snapshots, {p: p.read_bytes() for p in snapshots})

    def test_wrapper_forwards_internal_base_and_collector_checks_it(self):
        import serve_llm
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Stub the scheduler, never start a container or a real serving process.
            stub = root / 'srun'
            stub.write_text(f'#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n')
            stub.chmod(0o755)
            env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH'],
                       SLURM_JOB_ID='fixture-job', SLURM_JOB_NUM_NODES='1', SLURM_JOB_CPUS_PER_NODE='4',
                       VLLM_IMAGE='/fixture.sqsh', LLM_DATA_DIR='/fixture-data', LLM_RESULTS_DIR='/fixture-results',
                       GPUS_PER_NODE='8', SERVING_RUN='fixture', NCCL_SOCKET_IFNAME='fixture')
            options = ['--tensor-parallel-size', '2', '--internal-port-base', '24000']
            wrapper = Path(__file__).resolve().parents[1] / '8.serve-llm.sh'
            forwarded = json.loads(subprocess.check_output(['bash', str(wrapper), *options], env=env, text=True))
            self.assertEqual(forwarded[-len(options):], options)
            self.assertIn('"$@"', forwarded[forwarded.index('-c') + 1])
            self.launch_replicas(root, extra=['--internal-port-base', '24000'])
            argv = ['serve_llm.py', '--collect-manifest', '--output', str(root), '--gpus-per-node', '8',
                    '--model', str(root / 'model'), *options]
            env.update(SLURM_JOB_NODELIST='node-a')
            with patch.object(sys, 'argv', argv), patch.dict(os.environ, env, clear=True), \
                    patch('serve_llm.subprocess.check_output', side_effect=['node-a\n', 'NodeAddr=192.0.2.10']), \
                    patch('sys.stdout', new_callable=io.StringIO):
                serve_llm.main()
            manifest = json.loads((root / 'manifest.json').read_text())
            self.assertEqual(manifest['replica_environments'][0]['environment'], {'VLLM_PORT': '24000'})

    def test_replica_cleanup_includes_exited_leaders_and_disappeared_groups(self):
        from serve_llm import stop_replicas
        import signal
        import subprocess
        processes = [Mock(pid=101), Mock(pid=102), Mock(pid=103)]
        processes[0].poll.return_value = 1
        processes[1].wait.side_effect = [subprocess.TimeoutExpired('fixture', 30), 0]
        with patch('serve_llm.os.killpg', side_effect=[None, ProcessLookupError(), None,
                                                    ProcessLookupError(), None, None, None]) as kill:
            stop_replicas(processes)
        for process in processes:
            self.assertIn(call(process.pid, signal.SIGTERM), kill.call_args_list)
            self.assertIn(call(process.pid, signal.SIGKILL), kill.call_args_list)
            process.wait.assert_any_call(timeout=30)

    def test_salloc_cpu_count_expands_homogeneous_allocation(self):
        from serve_llm import allocation_cpus
        self.assertEqual(allocation_cpus('192(x2)', 2), 192)
        self.assertEqual(allocation_cpus('192,192', 2), 192)
        for value in ('', '0(x2)', '192', '192,96', '192(x3)', '192(x2)junk'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                allocation_cpus(value, 2)

    def test_manifest_uses_exact_replica_commands_and_gpu_records(self):
        from serve_llm import collect_manifest
        cfg = dict(model_id='fixture', model_revision='fixed', tokenizer_revision='fixed')
        kwargs = dict(addresses=['10.0.0.1', '10.0.0.2'], gpus_per_node=2,
                      tensor_parallel_size=2, max_num_seqs=256, max_num_batched_tokens=8192,
                      port_base=8100, cpus_per_node=4, cfg=cfg, expected_job='fixture-job',
                      expected_nodes=['node-a', 'node-b'])
        command = ['vllm', 'serve', '/data/model', '--tensor-parallel-size', '2',
                   '--max-num-seqs', '256', '--max-num-batched-tokens', '8192',
                   '--port', '8100', '--dtype', 'bfloat16', '--max-model-len', '4096',
                   '--gpu-memory-utilization', '0.85', '--generation-config', 'vllm',
                   '--served-model-name', 'aim347', '--host', '0.0.0.0', '--no-enable-prefix-caching']
        for fault in (None, 'missing', 'duplicate_gpu', 'batch', 'truncated', 'model', 'extra',
                      'write', 'stale', 'wrong_node', 'missing_node', 'override'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                case_kwargs = dict(kwargs)
                if fault == 'override':
                    case_kwargs['model_path'] = '/data/pinned-copy'
                for node in range(2):
                    if fault == 'missing' and node == 1:
                        continue
                    record = dict(command=list(command), gpu_uuids=[f'{node}-a', f'{node}-b'],
                                  slurm_job_id='fixture-job', slurm_node_name=['node-a', 'node-b'][node],
                                  model_pin={k: cfg[k] for k in ('model_id', 'model_revision')},
                                  vllm_version='0.20.2')
                    if fault == 'override':
                        record['command'][2] = '/data/pinned-copy'
                    if node == 1:
                        if fault == 'duplicate_gpu':
                            record['gpu_uuids'][0] = '0-a'
                        elif fault == 'batch':
                            record['command'][record['command'].index('--max-num-seqs') + 1] = '128'
                        elif fault == 'truncated':
                            record['command'] = ['--tensor-parallel-size']
                        elif fault == 'model':
                            record['command'][2] = '/other/model'
                        elif fault == 'extra':
                            record['command'] += ['--quantization', 'fp8']
                        elif fault == 'stale':
                            record['slurm_job_id'] = 'previous-job'
                        elif fault == 'wrong_node':
                            record['slurm_node_name'] = 'previous-node'
                        elif fault == 'missing_node':
                            del record['slurm_node_name']
                    path = root / f'node-{node}' / 'replica-0.json'
                    path.parent.mkdir()
                    path.write_text(json.dumps(record))
                if fault == 'write':
                    original_open = Path.open

                    def fail_endpoint(path, *args, **kwargs):
                        if path.name == 'endpoints.txt':
                            raise OSError('injected write failure')
                        return original_open(path, *args, **kwargs)

                    with patch.object(Path, 'open', fail_endpoint), self.assertRaises(OSError):
                        collect_manifest(root, **kwargs)
                    self.assertFalse((root / 'manifest.json').exists())
                    self.assertFalse((root / 'endpoints.txt').exists())
                    self.assertEqual(collect_manifest(root, **kwargs)['gpu_budget'], 4)
                elif fault and fault != 'override':
                    with self.assertRaises(ValueError):
                        collect_manifest(root, **kwargs)
                    self.assertFalse((root / 'manifest.json').exists())
                else:
                    result = collect_manifest(root, **case_kwargs)
                    self.assertEqual(result['gpu_budget'], 4)
                    self.assertEqual(result['placement']['replicas'], 2)
                    self.assertEqual((root / 'endpoints.txt').read_text().splitlines(),
                                     ['http://10.0.0.1:8100', 'http://10.0.0.2:8100'])
                    with self.assertRaises(ValueError):
                        collect_manifest(root, **case_kwargs)

    def test_stream_uses_usage_not_chunk_count_and_keeps_failures(self):
        events = [
            {'choices': [{'delta': {'content': '{"answer":'}, 'finish_reason': None}]},
            {'choices': [{'delta': {'content': '42}'}, 'finish_reason': 'stop'}]},
            {'choices': [], 'usage': {'completion_tokens': 8, 'prompt_tokens': 20}},
        ]
        raw = b''.join(('data: ' + json.dumps(event) + '\n').encode() for event in events) + b'data: [DONE]\n'
        with patch('urllib.request.urlopen', return_value=io.BytesIO(raw)), patch('time.perf_counter', side_effect=itertools.count()):
            row = stream_request(['http://fixture'], 0, {'prompt': 'fixture', 'expected_json': {'answer': 42}},
                                 model='fixture', max_tokens=8, timeout_seconds=120)
        self.assertEqual(row['output_tokens'], 8)
        self.assertEqual(row['content_chunks'], 2)
        self.assertAlmostEqual(row['tpot_seconds'], 2 / 7)
        self.assertTrue(row['quality_valid'])
        with patch('urllib.request.urlopen', side_effect=urllib.error.URLError('fixture failure')):
            row = stream_request(['http://fixture'], 0, {'prompt': 'fixture', 'expected_json': {}},
                                 model='fixture', max_tokens=8, timeout_seconds=120)
        self.assertEqual(row['status'], 'failed')
        self.assertIsNone(row['tpot_seconds'])

    def test_malformed_stream_events_are_retained_as_failed_requests(self):
        events = [None, [], {'choices': None}, {'choices': [None]},
                  {'choices': [{'delta': None}]},
                  {'choices': [{'delta': {'content': 42}}]},
                  {'choices': [{'delta': {}, 'finish_reason': []}]},
                  {'usage': []}, {'usage': {}},
                  {'usage': {'completion_tokens': True, 'prompt_tokens': 20}},
                  {'usage': {'completion_tokens': -1, 'prompt_tokens': 20}}]
        rows = []
        for index, event in enumerate(events):
            with self.subTest(event=event):
                raw = ('data: ' + json.dumps(event) + '\n').encode()
                with patch('urllib.request.urlopen', return_value=io.BytesIO(raw)):
                    row = stream_request(['http://fixture'], index,
                                         {'prompt': 'fixture', 'expected_json': {}},
                                         model='fixture', max_tokens=8, timeout_seconds=120)
                self.assertEqual(row['status'], 'failed')
                self.assertEqual(row['request_index'], index)
                self.assertFalse(row['quality_valid'])
                self.assertIn('ValueError', row['error'])
                rows.append(row)
        result = serving_goodput(rows, 2, ttft_slo_seconds=0.5, tpot_slo_seconds=0.1)
        self.assertEqual(result['offered_requests'], len(events))
        self.assertEqual(result['failed_requests'], len(events))
        self.assertEqual(result['qualified_requests'], 0)

    def test_both_latency_limits_quality_and_failures(self):
        good = dict(status='completed', quality_valid=True, output_tokens=8,
                    ttft_seconds=0.2, tpot_seconds=0.05)
        rows = [good, good | {'ttft_seconds': 0.9}, good | {'tpot_seconds': 0.2},
                good | {'quality_valid': False}, good | {'status': 'timeout'},
                good | {'output_tokens': 1, 'tpot_seconds': None}]
        result = serving_goodput(rows, 2, ttft_slo_seconds=0.5, tpot_slo_seconds=0.1)
        self.assertEqual(result['offered_requests'], 6)
        self.assertEqual(result['qualified_requests'], 1)
        self.assertEqual(result['serving_goodput_requests_per_second'], 0.5)
        self.assertEqual(result['failed_requests'], 1)
        self.assertEqual(result['one_token_tpot_undefined_requests'], 1)
        self.assertEqual(result['qualified_fraction'], 1 / 6)

    def test_quality_is_task_correctness(self):
        self.assertTrue(quality_valid('{"answer": 42}', {'answer': 42}))
        self.assertFalse(quality_valid('{"answer": 41}', {'answer': 42}))
        self.assertFalse(quality_valid('some fluent answer', {'answer': 42}))
        self.assertFalse(quality_valid('{"answer": true}', {'answer': 1}))
        self.assertFalse(quality_valid('{"answer": 1.0}', {'answer': 1}))
        self.assertTrue(quality_valid('{"b": 2, "a": 1}', {'a': 1, 'b': 2}))

    def test_comparison_separates_placement_batch_and_load(self):
        before = dict(workload={'prompts': 'fixed'}, gpu_budget=8,
                      placement={'replicas': 1, 'tensor_parallel_size': 8},
                      server_batch={'max_num_seqs': 64}, client_load={'concurrency': 16},
                      metrics={'serving_goodput_requests_per_second': 1})
        after = copy.deepcopy(before)
        after['placement'] = {'replicas': 4, 'tensor_parallel_size': 2}
        paired_serving_report(before, after, experiment='placement')
        before['allocation'] = after['allocation'] = {'gpu_uuids': ['fixture-gpu']}
        paired_serving_report(before, after, experiment='placement')
        after['allocation'] = {'gpu_uuids': ['different-fixture-gpu']}
        with self.assertRaisesRegex(ValueError, 'actual allocation'):
            paired_serving_report(before, after, experiment='placement')
        after['allocation'] = before['allocation']
        after['client_load']['concurrency'] = 32
        with self.assertRaises(ValueError):
            paired_serving_report(before, after, experiment='placement')
        with self.assertRaises(ValueError):
            paired_serving_report(before, after, experiment='server_batch')


if __name__ == '__main__':
    unittest.main()
