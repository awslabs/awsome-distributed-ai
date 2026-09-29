"""Synthetic arithmetic and clock fixtures, not training performance evidence."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
from llm_metrics import (ContinuousWindow, accumulation_steps,
                         paired_training_report, training_goodput, observability_values)


class LLMMetricTests(unittest.TestCase):
    def test_logging_prefetch_and_completion_remain_in_one_interval(self):
        events = []
        now = [0.0]

        def barrier():
            events.append('barrier')
            now[0] += 0.5

        def synchronize():
            events.append('synchronize')
            now[0] += 0.25

        window = ContinuousWindow(barrier, synchronize, lambda: now[0])
        window.start()
        now[0] += 2  # training
        now[0] += 3  # rank-zero logging/HTTP while input prefetch continues
        now[0] += 4  # training
        self.assertEqual(window.stop(), 9.75)
        self.assertEqual(events, ['synchronize', 'barrier', 'synchronize', 'barrier'])

    def test_batch_change_preserves_global_batch(self):
        self.assertEqual(accumulation_steps(32, 1, 8), 4)
        self.assertEqual(accumulation_steps(32, 2, 8), 2)
        with self.assertRaises(ValueError):
            accumulation_steps(30, 2, 8)

    def test_goodput_counts_retained_progress_only(self):
        # 400 replayed tokens and a pending save never enter the numerator.
        row = training_goodput(1000, 2000, 20, 160, scope='synthetic full interval')
        self.assertEqual(row['retained_progress_tokens'], 1000)
        self.assertEqual(row['training_goodput_tokens_per_second'], 50)
        self.assertEqual(row['tokens_per_allocated_gpu_hour'], 22500)
        with self.assertRaises(ValueError):
            training_goodput(2000, 1000, 20, 160, scope='invalid')

    def test_dashboard_never_promotes_segment_to_campaign_goodput(self):
        row = dict(status='completed', evidence_kind='LLM hardware run', profiled=False, numerical_verification=False,
                   training_throughput_tokens_per_second=100, measured_useful_tokens=1000,
                   measured_seconds=10, training_goodput={'training_goodput_tokens_per_second': 80})
        values = observability_values(row)
        self.assertEqual(values['llm_training_goodput_segment_tokens_per_second'], 80)
        self.assertNotIn('llm_training_goodput_campaign_tokens_per_second', values)
        row['evidence_kind'] = 'CPU correctness fixture'
        with self.assertRaises(ValueError):
            observability_values(row)
        campaign = training_goodput(0, 1000, 20, 160, scope='synthetic test')
        campaign.update(status='completed', evidence_kind='LLM hardware run', source='fresh controlled interruption and resume')
        self.assertEqual(observability_values(campaign)['llm_training_goodput_campaign_tokens_per_second'], 50)
        for flag in ('profiled', 'numerical_verification'):
            campaign[flag] = True
            with self.assertRaises(ValueError):
                observability_values(campaign)
            campaign[flag] = False
        campaign['status'] = 'failed'
        with self.assertRaises(ValueError):
            observability_values(campaign)

    def test_incomplete_or_unidentified_serving_is_not_published(self):
        row = dict(status='running', evidence_kind='LLM serving hardware run',
                   metrics={'metric': 'Serving goodput'})
        with self.assertRaises(ValueError):
            observability_values(row)
        row['status'] = 'failed'
        with self.assertRaises(ValueError):
            observability_values(row)
        row['status'] = 'completed'
        row.pop('evidence_kind')
        with self.assertRaises(ValueError):
            observability_values(row)

    def test_pair_rejects_work_changes_and_profiles(self):
        before = dict(workload={'global_batch_samples': 32}, allocation={'gpus': 8},
                      measurement_scope='steady', measured_useful_tokens=100,
                      measured_seconds=2, status='completed', profiled=False,
                      configuration={'microbatch': 1})
        after = copy.deepcopy(before)
        after.update(measured_seconds=1, configuration={'microbatch': 2})
        self.assertEqual(paired_training_report(before, after)['speedup_ratio'], 2)
        after['configuration']['deterministic'] = True
        with self.assertRaises(ValueError):
            paired_training_report(before, after)
        after['configuration']['deterministic'] = False
        after['workload']['global_batch_samples'] = 64
        with self.assertRaises(ValueError):
            paired_training_report(before, after)
        after['workload'] = before['workload']
        after['profiled'] = True
        with self.assertRaises(ValueError):
            paired_training_report(before, after)


if __name__ == '__main__':
    unittest.main()
