"""Synthetic serving accounting fixtures; no actual inference performance."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch, call
import io
import itertools
import json
import urllib.error
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
from serving_goodput import paired_serving_report, quality_valid, serving_goodput, stream_request


class ServingGoodputTests(unittest.TestCase):
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
        command = ['vllm', 'serve', '/media/model', '--tensor-parallel-size', '2',
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
                    self.assertEqual(result['endpoints'], ['http://10.0.0.1:8100', 'http://10.0.0.2:8100'])
                    self.assertEqual((root / 'endpoints.txt').read_text().splitlines(),
                                     ['http://10.0.0.1:8100', 'http://10.0.0.2:8100'])
                    with self.assertRaises(ValueError):
                        collect_manifest(root, **case_kwargs)

    def test_client_rejects_substituted_and_duplicate_endpoints(self):
        import importlib.util
        module_path = Path(__file__).resolve().parents[1] / '9.load-llm.py'
        spec = importlib.util.spec_from_file_location('load_llm_fixture', module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'requests.jsonl').write_text(json.dumps({'prompt': 'fixture', 'expected_json': {}}) + '\n')
            (root / 'manifest.json').write_text(json.dumps({'endpoints': ['http://a', 'http://b']}))
            for endpoints in (['http://a', 'http://a'], ['http://a', 'http://other']):
                argv = ['client', *endpoints, '--requests-file', str(root / 'requests.jsonl'),
                        '--server-manifest', str(root / 'manifest.json'), '--output', str(root / 'out.json'),
                        '--concurrency', '2', '--ttft-slo-seconds', '2', '--tpot-slo-seconds', '0.05']
                with self.subTest(endpoints=endpoints), patch.object(sys, 'argv', argv), \
                        patch.object(module, 'stream_request') as request, self.assertRaises(SystemExit):
                    module.main()
                request.assert_not_called()

    def test_client_records_stops_source_hashes_and_pilot_only(self):
        import importlib.util
        import hashlib
        module_path = Path(__file__).resolve().parents[1] / '9.load-llm.py'
        loader = importlib.util.spec_from_file_location('stop_client_fixture', module_path)
        module = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specs = [{'id': 'case', 'prompt': 'fixture', 'expected_json': {}, 'stop': ['\n\nInput:']}]
            raw = (json.dumps(specs[0]) + '\n').encode()
            (root / 'requests.jsonl').write_bytes(raw)
            manifest = dict(endpoints=['http://a'], workload={}, gpu_budget=1,
                            placement={'replicas': 1, 'tensor_parallel_size': 1})
            (root / 'manifest.json').write_text(json.dumps(manifest))
            argv = ['client', 'http://a', '--requests-file', str(root / 'requests.jsonl'),
                    '--server-manifest', str(root / 'manifest.json'), '--output', str(root / 'out.json'),
                    '--concurrency', '1', '--ttft-slo-seconds', '2', '--tpot-slo-seconds', '0.05', '--pilot-only']
            row = dict(status='completed', quality_valid=True, output_tokens=8,
                       ttft_seconds=0.1, tpot_seconds=0.01)
            with patch.object(sys, 'argv', argv), patch.object(module, 'stream_request', return_value=row) as request, \
                    patch('urllib.request.urlopen', side_effect=lambda *a, **k: io.BytesIO(b'fixture metrics')):
                module.main()
            self.assertEqual(request.call_count, 3)  # two warmups, one measured
            self.assertTrue(all(c.args[2] == specs[0] for c in request.call_args_list))
            result = json.loads((root / 'out.json').read_text())
            self.assertFalse(result['performance_eligible'])
            self.assertEqual(result['workload']['requests_sha256'], hashlib.sha256(raw).hexdigest())
            self.assertIn('stop_policy', result['workload'])
            for name, digest in result['workload']['client_source_sha256'].items():
                self.assertEqual(digest, hashlib.sha256((module_path.parent / name).read_bytes()).hexdigest())

    def test_participant_frozen_fixture_normal_client_path(self):
        import importlib.util
        import hashlib
        module_path = Path(__file__).resolve().parents[1] / '9.load-llm.py'
        loader = importlib.util.spec_from_file_location('stop_client_fixture', module_path)
        module = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = (module_path.parent / 'configs/range64-fewshot-boundary.jsonl').read_bytes()
            specs = [json.loads(line) for line in raw.splitlines()]
            self.assertEqual(len(specs), 128)
            self.assertTrue(all(spec == specs[0] for spec in specs))
            self.assertEqual(specs[0]['stop'], ['\n\nInput:'])
            self.assertEqual(specs[0]['expected_json'], {'values': list(range(1, 65))})
            self.assertEqual(hashlib.sha256(raw).hexdigest(), '60dbda657eaac76d3192c4fe800b3a65a6aeea5bf733b635085bd9b540723ebb')
            (root / 'requests.jsonl').write_bytes(raw)
            manifest = dict(endpoints=['http://a'], workload={}, gpu_budget=1,
                            placement={'replicas': 1, 'tensor_parallel_size': 1})
            (root / 'manifest.json').write_text(json.dumps(manifest))
            argv = ['client', 'http://a', '--requests-file', str(root / 'requests.jsonl'),
                    '--server-manifest', str(root / 'manifest.json'), '--output', str(root / 'out.json'),
                    '--concurrency', '1', '--ttft-slo-seconds', '2', '--tpot-slo-seconds', '0.05']
            row = dict(status='completed', quality_valid=True, output_tokens=8,
                       ttft_seconds=0.1, tpot_seconds=0.01)
            with patch.object(sys, 'argv', argv), patch.object(module, 'stream_request', return_value=row) as request, \
                    patch('urllib.request.urlopen', side_effect=lambda *a, **k: io.BytesIO(b'fixture metrics')):
                module.main()
            self.assertEqual(request.call_count, 130)  # two warmups, 128 measured
            self.assertTrue(all(c.args[2] == specs[0] for c in request.call_args_list))
            result = json.loads((root / 'out.json').read_text())
            self.assertIsNot(result.get('performance_eligible'), False)
            self.assertEqual(result['workload']['requests_sha256'], hashlib.sha256(raw).hexdigest())
            self.assertIn('stop_policy', result['workload'])
            for name, digest in result['workload']['client_source_sha256'].items():
                self.assertEqual(digest, hashlib.sha256((module_path.parent / name).read_bytes()).hexdigest())

    def test_declared_stops_forwarded_without_output_clipping(self):
        text = '{"answer":42}\n\nInput: continued'
        events = [{'choices': [{'text': text, 'finish_reason': 'length'}]},
                  {'choices': [], 'usage': {'completion_tokens': 8, 'prompt_tokens': 20}}]
        raw = b''.join(('data: ' + json.dumps(e) + '\n').encode() for e in events) + b'data: [DONE]\n'
        for stops in (None, [], ['\n\nInput:'], ['END', ' STOP ']):
            spec = {'prompt': 'fixture', 'expected_json': {'answer': 42}}
            if stops is not None:
                spec.update(stop=stops, id='case-a')
            with self.subTest(stops=stops), patch('urllib.request.urlopen', return_value=io.BytesIO(raw)) as request:
                row = stream_request(['http://fixture'], 0, spec,
                                     model='fixture', max_tokens=8, timeout_seconds=120)
            payload = json.loads(request.call_args.args[0].data)
            self.assertEqual('stop' in payload, stops is not None)
            self.assertEqual('stop' in row, stops is not None)
            if stops is not None:
                self.assertEqual(payload['stop'], stops)
                self.assertEqual(row['stop'], stops)
                self.assertEqual(row['specification_id'], 'case-a')
            self.assertEqual(row['text'], text)
            self.assertFalse(row['quality_valid'])

    def test_invalid_stop_specs_rejected_before_network(self):
        from serving_goodput import validate_specification
        base = {'prompt': 'fixture', 'expected_json': {}}
        for bad in (None, '', 'END', 1, True, {}, [''], ['END', None], [1]):
            with self.subTest(stop=bad), patch('urllib.request.urlopen') as request, self.assertRaises(ValueError):
                stream_request(['http://fixture'], 0, base | {'stop': bad},
                               model='fixture', max_tokens=8, timeout_seconds=120)
            request.assert_not_called()
        for spec in ([], base | {'prompt': ''}, base | {'id': 1}, base | {'typo': []}):
            with self.assertRaises(ValueError):
                validate_specification(spec)

    def test_stop_workload_provenance_and_pilot_comparison_gates(self):
        import hashlib
        base = {'prompt': 'fixture', 'expected_json': {}}
        hashes = [hashlib.sha256(json.dumps(spec).encode()).hexdigest()
                  for spec in (base, base | {'stop': []}, base | {'stop': ['END']})]
        self.assertEqual(len(set(hashes)), 3)
        before = dict(workload={'requests_sha256': hashes[0]}, gpu_budget=16,
                      placement={'tensor_parallel_size': 8}, server_batch={}, client_load={},
                      metrics={'serving_goodput_requests_per_second': 0})
        for digest in hashes[1:]:
            after = copy.deepcopy(before)
            after['workload']['requests_sha256'] = digest
            with self.assertRaisesRegex(ValueError, 'workload'):
                paired_serving_report(before, after, experiment='placement')
        with self.assertRaisesRegex(ValueError, 'pilot-only'):
            paired_serving_report(before, before | {'performance_eligible': False}, experiment='placement')

    def test_stream_uses_usage_not_chunk_count_and_keeps_failures(self):
        events = [
            {'choices': [{'text': '{"answer":', 'finish_reason': None}]},
            {'choices': [{'text': '42}', 'finish_reason': 'stop'}]},
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
                  {'choices': [{'text': None}]},
                  {'choices': [{'text': 42}]},
                  {'choices': [{'text': '', 'finish_reason': []}]},
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
