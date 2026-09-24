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
