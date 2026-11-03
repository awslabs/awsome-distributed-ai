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
            push('http://localhost:9091','run-a','sync','test-fixture',{'tokens_per_second':10}, measurement_kind='campaign')
            campaign=request.call_args.args[0]
            push('http://localhost:9091','run-a','sync','test-fixture',{'tokens_per_second':20}, measurement_kind='training_segment')
            segment=request.call_args.args[0]
            self.assertEqual(campaign.method, 'PUT')
            self.assertEqual(segment.method, 'PUT')
            self.assertNotEqual(campaign.full_url, segment.full_url)
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


    def test_lustre_2_15_client_stats_block(self):
        # Recorded llite.*.stats block from a Lustre 2.15 client (September 2026
        # rehearsal), including the start_time and elapsed_time lines.
        raw = '''llite.aobzzbmv-ffff8b70510cb000.stats=
snapshot_time             1790056030.227815959 secs.nsecs
start_time                1790051060.344947630 secs.nsecs
elapsed_time              4969.882868329 secs.nsecs
read_bytes                48966 samples [bytes] 3 93045440 236801406594 2532124456008791348
write_bytes               1024454 samples [bytes] 15 97241472 214123407410 3364767048228424446
read                      48430 samples [usec] 0 1948101 97293743 28680002073447
write                     1117207 samples [usec] 0 3009763 5949955261 1453763156155711
ioctl                     11115 samples [reqs]
open                      12941 samples [usec] 0 7109 267856 414565866
close                     12917 samples [usec] 0 6356 337843 236612759
mmap                      2768 samples [usec] 1 601 38896 13076930
page_fault                387326134 samples [usec] 0 6772 4233530243 68351681937
seek                      850141 samples [usec] 0 530 76246 1506096
fsync                     177 samples [usec] 11982 3012749 117829755 242275452575547
readdir                   5718 samples [usec] 0 1146 40631 16560001
setattr                   2 samples [usec] 135 347 482 138634
getattr                   270400 samples [usec] 0 10819 2468598 796843714
mkdir                     11 samples [usec] 282 585 4650 2051578
mknod                     518 samples [usec] 151 1363 170815 65827275
rename                    97 samples [usec] 135 535 29823 10389437
statfs                    278 samples [usec] 0 795 57495 20544449
inode_permission          339182 samples [usec] 0 322 3465 337517
opencount                 12941 samples [reqs] 1 383 519578 73846634
openclosetime             11098 samples [usec] 0 1104750833 89760736183 8644339553849271573
'''
        metrics = dict(line.rsplit(' ', 1) for line in lustre_metrics(raw).splitlines())
        mount = 'mount="aobzzbmv-ffff8b70510cb000"'
        self.assertEqual(metrics['lustre_read_bytes_total{' + mount + '}'], '236801406594')
        self.assertEqual(metrics['lustre_write_bytes_total{' + mount + '}'], '214123407410')
        self.assertEqual(metrics['lustre_open_operations_total{' + mount + '}'], '12941')
        self.assertEqual(metrics['lustre_getattr_operations_total{' + mount + '}'], '270400')
        self.assertEqual(len(metrics), 4)


if __name__=='__main__': unittest.main()
