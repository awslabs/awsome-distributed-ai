"""What benchmark.py and 4.ramp.py record and refuse."""
import argparse
import asyncio
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
import httpx
import benchmark
from benchmark import run_rate, stream_request
from scoring import gap_data_available
from test_lab import StreamHandler


class BenchmarkTests(unittest.TestCase):
    def test_each_chunk_that_adds_tokens_is_recorded_with_its_time_and_token_count(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), StreamHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        async def call():
            async with httpx.AsyncClient(trust_env=False) as client:
                return await stream_request(client, f'http://127.0.0.1:{server.server_port}', [1], 4, 2)
        try:
            result = asyncio.run(call())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        self.assertTrue(result['ok'])
        self.assertEqual([c['n'] for c in result['chunks']], [1, 3])
        self.assertLess(result['chunks'][0]['t_ms'], result['chunks'][1]['t_ms'])

    def test_calls_skipped_after_a_failed_first_turn_keep_the_rate_judgeable_under_the_gap_rules(self):
        async def failed(*_):
            return {'ok': False, 'text': '', 'error': 'HTTPStatusError', 'chunks': []}
        data = {'model': 'm', 'revision': 'r', 'tasks': [{'task_id': 't', 'calls': 3}],
                'shape': {'name': 'A-agentic', 'output_tokens': 4}}
        args = argparse.Namespace(
            seed=1, duration_s=0.001, max_tasks=10, url='http://unused', timeout_s=1,
            context_length_tokens=100, gpus=1, ttft_slo_ms=2000, tpot_slo_ms=100,
            attainment_fraction=0.9, max_client_lag_ms=1000, architecture='unified',
            instance_type='test', region='test', evidence_scope='local-mock')
        original = (benchmark.stream_request, benchmark.initial_prompt)
        benchmark.stream_request, benchmark.initial_prompt = failed, lambda *_: [1]
        try:
            records = asyncio.run(run_rate(data, args, 1, 'measured', None))['requests']
        finally:
            benchmark.stream_request, benchmark.initial_prompt = original
        first_task = [r for r in records if r['task_id'] == 't-0']
        self.assertEqual([r.get('error') for r in first_task], ['HTTPStatusError'] + ['Skipped after failed predecessor'] * 2)
        self.assertTrue(gap_data_available(records))

    def test_calls_refused_for_the_context_budget_keep_the_rate_judgeable_under_the_gap_rules(self):
        data = {'model': 'm', 'revision': 'r', 'tasks': [{'task_id': 't', 'calls': 2}],
                'shape': {'name': 'A-agentic', 'output_tokens': 4}}
        args = argparse.Namespace(
            seed=1, duration_s=0.001, max_tasks=10, url='http://unused', timeout_s=1,
            context_length_tokens=3, gpus=1, ttft_slo_ms=2000, tpot_slo_ms=100,
            attainment_fraction=0.9, max_client_lag_ms=1000, architecture='unified',
            instance_type='test', region='test', evidence_scope='local-mock')
        original = benchmark.initial_prompt
        benchmark.initial_prompt = lambda *_: [1]
        try:
            records = asyncio.run(run_rate(data, args, 1, 'measured', None))['requests']
        finally:
            benchmark.initial_prompt = original
        first_task = [r for r in records if r['task_id'] == 't-0']
        self.assertEqual([r.get('error') for r in first_task], ['Context budget exceeded', 'Skipped after failed predecessor'])
        self.assertTrue(gap_data_available(records))

    def test_a_repeated_rate_is_rejected_before_any_traffic(self):
        with tempfile.TemporaryDirectory() as d:
            done = subprocess.run([sys.executable, '4.ramp.py', '--traffic', str(Path(d) / 'missing.json'), '--url', 'http://127.0.0.1:9',
                                   '--output', d, '--architecture', 'unified', '--instance-type', 'test', '--region', 'test',
                                   '--evidence-scope', 'local-mock', '--gpus', '1', '--rates', '1', '2', '1.0000001'],
                                  capture_output=True, text=True)
            self.assertEqual(done.returncode, 2)
            self.assertIn('--rates names one rate twice', done.stderr)
            self.assertEqual(list(Path(d).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
