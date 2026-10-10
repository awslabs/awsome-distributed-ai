"""Regression checks for the workshop's arithmetic and real counter wire formats."""
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from metrics import push
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lib'))
from metrics import calculate
from counters import lustre_metrics, efa_metrics


class MeasurementTests(unittest.TestCase):
    def test_dense_denominator_and_gpu_count(self):
        # Synthetic arithmetic fixture, not a measured training result.
        result=calculate(10000,3_000_000_000,4,429.1)
        self.assertEqual(result['tflops_per_gpu'],45)
        self.assertAlmostEqual(result['mfu_dense_gemm_ratio'],45/429.1)
        sparse=calculate(10000,3_000_000_000,4,1000)
        self.assertLess(sparse['mfu_dense_gemm_ratio'],result['mfu_dense_gemm_ratio'])
        self.assertAlmostEqual(result['mfu_dense_gemm_ratio']/sparse['mfu_dense_gemm_ratio'],1000/429.1)
        self.assertEqual(calculate(20000,3_000_000_000,8,429.1),dict(result,tokens_per_second=20000))

    def test_pushgateway_keeps_each_configuration(self):
        with patch('urllib.request.urlopen') as request:
            push('http://localhost:9091','run-a','v0','test-fixture',{'tokens_per_second':10})
            baseline=request.call_args.args[0].full_url
            push('http://localhost:9091','run-a','v1','test-fixture',{'tokens_per_second':20})
            fixed=request.call_args.args[0].full_url
        self.assertNotEqual(baseline,fixed)
        self.assertTrue(baseline.endswith('/config/v0'))
        self.assertTrue(fixed.endswith('/config/v1'))

    def test_bad_inputs_fail(self):
        for value in [0,-1,math.inf,math.nan]:
            with self.assertRaises(ValueError): calculate(value,3_000_000_000,4,429.1)

    def test_lustre_sum_not_samples_and_multiple_mounts(self):
        raw='''llite.fs-a.stats=
snapshot_time 12345.000000 secs.nsecs
read_bytes 7 samples [bytes] 4 4096 8192
write_bytes 2 samples [bytes] 16 32 48
open 17 samples [reqs]
getattr 19 samples [reqs]
llite.fs-b.stats=
read_bytes 3 samples [bytes] 8 16 32
open 4 samples [reqs]
'''
        text=lustre_metrics(raw)
        self.assertIn('lustre_read_bytes_total{mount="fs-a"} 8192',text)
        self.assertIn('lustre_read_bytes_total{mount="fs-b"} 32',text)
        self.assertIn('lustre_open_operations_total{mount="fs-a"} 17',text)
        self.assertEqual(lustre_metrics(''), '\n')

    def test_missing_efa_counter_is_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            hw=Path(directory)/'rdmap0s0/ports/1/hw_counters'; hw.mkdir(parents=True)
            (hw/'tx_bytes').write_text('1234\n')
            text=efa_metrics(directory)
            self.assertIn('node_amazonefa_tx_bytes{device="rdmap0s0",port="1"} 1234',text)
            self.assertNotIn('node_amazonefa_rx_bytes',text)
            self.assertIn('aim347_efa_ports_visible 1',text)


if __name__=='__main__': unittest.main()
