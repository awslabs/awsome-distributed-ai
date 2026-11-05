"""Synthetic arithmetic and clock fixtures, not training performance evidence."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
from llm_metrics import (ContinuousWindow, accumulation_steps, observability_values,
                         paired_training_report, retained_progress, training_goodput, update_work)

# Synthetic run of 5 updates of 32 x 2,048 positions without saves, the first one warmup.
WORKLOAD = dict(model_id='m', sequence_length=2048, global_batch_samples=32, data_parallel_size=16,
                updates=5, warmup_updates=1)
FAULT_FREE = dict(status='completed', profiled=False, numerical_verification=False,
                  measurement_scope='continuous post-warmup training', workload=WORKLOAD,
                  measured_useful_tokens=4 * 65504, training_throughput_tokens_per_second=80000.0)


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

    def test_retained_progress_counts_retained_progress_only(self):
        # 400 replayed tokens and a pending save never enter the numerator.
        row = retained_progress(1000, 2000, 20, scope='synthetic full interval', initial_updates=5, durable_updates=10)
        self.assertEqual(row, {'metric': 'Retained progress', 'retained_progress_tokens': 1000,
                               'retained_tokens_per_second': 50, 'elapsed_seconds': 20,
                               'initial_completed_updates': 5, 'durable_completed_updates': 10,
                               'scope': 'synthetic full interval'})
        with self.assertRaises(ValueError):
            retained_progress(2000, 1000, 20, scope='invalid', initial_updates=0, durable_updates=1)
        with self.assertRaises(ValueError):
            retained_progress(0, 1000, 0, scope='no interval', initial_updates=0, durable_updates=1)
        with self.assertRaises(ValueError):
            retained_progress(0, 1000, 20, scope='durable update before the start', initial_updates=5, durable_updates=4)

    def test_goodput_is_the_retained_rate_over_the_fault_free_rate(self):
        # 40 updates of 65,504 targets retained at 20,000 tokens/s against 80,000 tokens/s without saves.
        work = update_work(dict(WORKLOAD, updates=40, warmup_updates=2), 40 * 65504, 40)
        self.assertEqual(work['targets_per_update'], 65504)
        self.assertEqual(training_goodput(20000.0, work, FAULT_FREE), {
            'metric': 'Training goodput', 'training_goodput': 0.25,
            'retained_tokens_per_second': 20000.0, 'fault_free_tokens_per_second': 80000.0})

    def test_goodput_refuses_different_work_per_update(self):
        work = update_work(WORKLOAD, 40 * 32752, 40)
        with self.assertRaisesRegex(ValueError, '^different work per update: targets_per_update$'):
            training_goodput(20000.0, work, FAULT_FREE)
        work = update_work(dict(WORKLOAD, sequence_length=1024), 40 * 65504, 40)
        with self.assertRaisesRegex(ValueError, '^different work per update: sequence_length$'):
            training_goodput(20000.0, work, FAULT_FREE)

    def test_goodput_refuses_a_missing_saving_or_unfinished_reference(self):
        work = update_work(WORKLOAD, 40 * 65504, 40)
        for reference, reason in (
                (None, 'no fault-free run'), ({}, 'no fault-free run'),
                (dict(FAULT_FREE, status='failed'), 'the fault-free run is not completed'),
                (dict(FAULT_FREE, profiled=True), 'the fault-free run was profiled'),
                (dict(FAULT_FREE, numerical_verification=True), 'the fault-free run ran numerical verification'),
                (dict(FAULT_FREE, performance_eligible=False), 'the fault-free run is marked not performance-eligible'),
                (dict(FAULT_FREE, measurement_scope='training/save/final durable completion'), 'the fault-free run saves')):
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, f'^{reason}$'):
                training_goodput(20000.0, work, reference)
        with self.assertRaises(ValueError):
            training_goodput(20000.0, work, dict(FAULT_FREE, training_throughput_tokens_per_second=0.0))

    def test_dashboard_never_promotes_segment_to_campaign_goodput(self):
        row = dict(status='completed', evidence_kind='LLM hardware run', profiled=False, numerical_verification=False,
                   training_throughput_tokens_per_second=100, measured_useful_tokens=1000,
                   measured_seconds=10, retained_progress={'retained_tokens_per_second': 80})
        values = observability_values(row)
        self.assertEqual(values['llm_training_segment_retained_tokens_per_second'], 80)
        self.assertNotIn('llm_campaign_retained_tokens_per_second', values)
        row['evidence_kind'] = 'CPU correctness fixture'
        with self.assertRaises(ValueError):
            observability_values(row)
        campaign = retained_progress(0, 1000, 20, scope='synthetic test', initial_updates=0, durable_updates=6)
        campaign.update(status='completed', evidence_kind='LLM hardware run', source='fresh controlled interruption and resume')
        self.assertEqual(observability_values(campaign), {'llm_campaign_retained_tokens_per_second': 50,
                                                          'llm_campaign_retained_tokens': 1000,
                                                          'llm_campaign_elapsed_seconds': 20})
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
