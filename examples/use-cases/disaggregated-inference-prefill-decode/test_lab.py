"""CPU checks for traffic semantics, failure accounting, and streaming parsing."""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
import unittest
import httpx
from benchmark import stream_request, summarize
from deployment import render
from traffic import continuation, common_prefix, initial_prompt, tokenizer


class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        for count in (0, 1, 4):
            message = {'text': 'ready' if count else '', 'meta_info': {'completion_tokens': count, 'cached_tokens': 3}}
            self.wfile.write(('data: ' + json.dumps(message) + '\n\n').encode())
            self.wfile.flush()
            time.sleep(.01)
        if body['input_ids'] != [999]:
            self.wfile.write(b'data: [DONE]\n\n')


class LabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = tokenizer()

    def test_agentic_chains_preserve_actual_answer_and_overlap(self):
        shape = json.loads(Path('shape-a.json').read_text())
        prompt = initial_prompt(self.tok, shape, 'task-test')
        self.assertEqual(len(prompt), 1024)
        for turn in range(1, 15):
            previous = prompt
            answer = 'The warehouse is queued. Check the dispatch log. ' * 24
            prompt, overlap = continuation(self.tok, shape, previous, answer, 'task-test', turn)
            self.assertEqual(common_prefix(previous, prompt), len(previous))
            self.assertGreaterEqual(overlap, .5)
            self.assertLessEqual(overlap, .9)
            self.assertIn(answer.strip(), self.tok.decode(prompt[len(previous):]))
        self.assertLess(len(prompt) + 256, 32768)

    def test_long_context_warmup_cannot_seed_measured_prefix(self):
        shape = json.loads(Path('shape-b.json').read_text())
        a = initial_prompt(self.tok, shape, 'same-task', 'warmup')
        b = initial_prompt(self.tok, shape, 'same-task', 'measured')
        self.assertEqual(len(a), 8192)
        self.assertEqual(len(b), 8192)
        self.assertLess(common_prefix(a, b) / len(a), .05)

    def test_joint_slo_counts_errors_and_skipped_followups(self):
        rows = [{'ok': True, 'sent': True, 'ttft_ms': 100, 'tpot_ms': 10},
                {'ok': True, 'sent': True, 'ttft_ms': 100, 'tpot_ms': 200},
                {'ok': False, 'sent': True}, {'ok': False, 'sent': False}]
        r = summarize(rows, 10, 2, 200, 20, .9, 1, .3)
        self.assertEqual(r['slo_attainment_fraction'], .25)
        self.assertEqual(r['useful_calls_per_s'], .1)
        self.assertEqual(r['useful_calls_per_s_per_gpu'], .05)
        self.assertFalse(r['meets_joint_slo'])
        self.assertEqual(r['failed_or_skipped_calls'], 2)

    def test_stream_ignores_empty_initial_event_and_rejects_truncation(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), StreamHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        async def check():
            async with httpx.AsyncClient(trust_env=False) as client:
                url = f'http://127.0.0.1:{server.server_port}'
                good = await stream_request(client, url, [1], 4, 2)
                bad = await stream_request(client, url, [999], 4, 2)
                self.assertTrue(good['ok'])
                self.assertEqual(good['first_chunk_tokens'], 1)
                self.assertGreater(good['tpot_ms'], 0)
                self.assertFalse(bad['ok'])
        try:
            asyncio.run(check())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_deployments_hold_gpu_count_and_keep_both_selectors(self):
        c = json.loads(Path('config.example.json').read_text())
        c['nodes'] = ['node-a', 'node-b', 'node-c']
        def workers(doc):
            return [x for x in doc['items'] if x['kind'] == 'Deployment' and x['metadata']['labels'].get('component') == 'engine']
        unified = workers(render(c, 'unified'))
        split = workers(render(c, 'disaggregated'))
        scaled = workers(render(c, 'disaggregated', 2))
        self.assertEqual(len(unified), 2)
        self.assertEqual(len(split), 2)
        self.assertEqual(len(scaled), 3)
        for worker in split:
            pod = worker['spec']['template']['spec']
            engine = pod['containers'][0]
            self.assertIn('--disaggregation-transfer-backend', engine['args'])
            self.assertNotIn('--disaggregation-ib-device', engine['args'])
            self.assertEqual({x['name']: x['value'] for x in engine['env']}['SGLANG_DISAGGREGATION_NIXL_BACKEND'], 'LIBFABRIC')
            self.assertEqual(pod['preemptionPolicy'], 'Never')
        self.assertEqual(split[1], scaled[1])

if __name__ == '__main__':
    unittest.main()
