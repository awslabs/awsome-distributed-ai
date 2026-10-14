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
from benchmark import ramp, stream_request
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
        self.assertEqual(result['max_gap_ms'], result['chunks'][1]['t_ms'] - result['chunks'][0]['t_ms'])

    def test_a_ramp_into_a_directory_with_ramp_files_stops_before_reading_traffic(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / '1-measured.json').write_text('{}')
            args = argparse.Namespace(output=d, traffic=str(Path(d) / 'missing.json'))
            with self.assertRaises(SystemExit) as cm:
                asyncio.run(ramp(args))
            self.assertIn('already holds ramp files', str(cm.exception))

    def test_a_repeated_rate_is_rejected_before_any_traffic(self):
        with tempfile.TemporaryDirectory() as d:
            done = subprocess.run([sys.executable, '4.ramp.py', '--traffic', str(Path(d) / 'missing.json'), '--url', 'http://127.0.0.1:9',
                                   '--output', d, '--architecture', 'unified', '--instance-type', 'test', '--region', 'test',
                                   '--evidence-scope', 'local-mock', '--gpus', '1', '--rates', '1', '1'],
                                  capture_output=True, text=True)
            self.assertEqual(done.returncode, 2)
            self.assertIn('--rates has a repeated value', done.stderr)
            self.assertEqual(list(Path(d).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
