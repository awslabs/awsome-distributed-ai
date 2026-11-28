"""Synthetic accounting checks; these fixtures are not hardware measurements."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lib'))
from serving_metrics import METRICS, decode_record


class ServingMetricTests(unittest.TestCase):
    def test_decode_counts_exclude_prefill_and_reject_interleaved_work(self):
        before = dict.fromkeys(METRICS,0)
        after = dict(zip(METRICS,[8,2,6,.3]))
        row = decode_record(before,after,[{'output_tokens':4},{'output_tokens':4}])
        self.assertEqual(row['decode_steps'],6)
        self.assertAlmostEqual(row['server_mean_itl_seconds'],.05)
        after[METRICS[0]]=9
        with self.assertRaises(ValueError): decode_record(before,after,[{'output_tokens':4},{'output_tokens':4}])


if __name__=='__main__': unittest.main()
