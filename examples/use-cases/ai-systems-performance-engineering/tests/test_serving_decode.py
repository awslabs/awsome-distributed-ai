"""CPU tests of the measurement of Optional B (tools/serve.sh, lib/serving_decode.py). Standard library only.

No server runs: the servers' counters are replaced by fakes. Step times
in these fixtures are synthetic, not measurements. The memory-read ceiling of 663 GB/s
is the median recorded by measure-ceilings.sh on two g7.48xlarge on 2026-11-03.
"""
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'lib'))
import serving_decode  # noqa: E402

CEILING = 663e9
METRICS = ('vllm:generation_tokens_total', 'vllm:request_success_total',
           'vllm:inter_token_latency_seconds_count', 'vllm:inter_token_latency_seconds_sum')


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
            'At 5.16 ms per step, a GPU with one request generates 194 tokens/s. MBU is 5.16 ms / ms per step.'])
        self.assertIn('run bash tools/measure-ceilings.sh', serving_decode.bound_lines(None)[0])

    def test_servers_are_addressed_by_node_address_and_gpu_port(self):
        # 8.serve-llm.sh starts the server of GPU g on port 8100 + g.
        answers = ['gpu-node-1\ngpu-node-2\n', 'NodeName=gpu-node-1 NodeAddr=10.0.0.1 State=ALLOCATED\n',
                   'NodeName=gpu-node-2 NodeAddr=10.0.0.2 State=ALLOCATED\n']
        with patch('serving_decode.subprocess.check_output', side_effect=answers):
            endpoints = serving_decode.server_endpoints('gpu-node-[1-2]', 2)
        self.assertEqual(endpoints, ['http://10.0.0.1:8100', 'http://10.0.0.1:8101',
                                     'http://10.0.0.2:8100', 'http://10.0.0.2:8101'])

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

    def test_rows(self):
        summaries = [dict(requests_per_gpu=1, seconds_per_step=0.009, tokens_per_second_per_gpu=1 / 0.009,
                          mbu=0.5736),
                     dict(requests_per_gpu=16, seconds_per_step=0.0125, tokens_per_second_per_gpu=16 / 0.0125,
                          mbu=None)]
        self.assertEqual(serving_decode.row_line(summaries[0]),
                         '               1          9.00                       111   57 %')
        self.assertEqual(serving_decode.row_line(summaries[1]),
                         '              16         12.50                     1,280   n/a')


if __name__ == '__main__':
    unittest.main()
