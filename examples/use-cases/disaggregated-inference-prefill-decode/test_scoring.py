"""scoring.py on hand-built cases."""
import unittest
from scoring import highest_qualifying, percentile, pooled_itls_ms, summarize, worst_itl_ms


def ok(ttft=100, tpot=10, chunks=None):
    row = {'ok': True, 'sent': True, 'ttft_ms': ttft, 'tpot_ms': tpot, 'client_launch_lag_ms': 0}
    if chunks is not None:
        row['chunks'] = chunks
    return row


FAILED = {'ok': False, 'sent': True, 'client_launch_lag_ms': 0, 'chunks': []}


class ScoringTests(unittest.TestCase):
    def test_p99_is_the_value_at_nearest_rank(self):
        # ceil(100 * 0.99) - 1 = 98 (0-indexed), the 99th smallest value; ceil(3 * 0.99) - 1 = 2.
        self.assertEqual(percentile(list(range(1, 101)), .99), 99)
        self.assertEqual(percentile([10, 20, 30], .99), 30)

    def test_worst_itl_is_the_longest_raw_gap_not_divided_by_token_count(self):
        chunks = [{'t_ms': 0, 'n': 1}, {'t_ms': 40, 'n': 1}, {'t_ms': 440, 'n': 4}, {'t_ms': 480, 'n': 1}]
        self.assertEqual(worst_itl_ms({'chunks': chunks}), 400)

    def test_itl_p99_pools_each_gap_divided_by_its_tokens_once_per_token(self):
        chunks = [{'t_ms': 0, 'n': 1}, {'t_ms': 20, 'n': 1}, {'t_ms': 220, 'n': 4}]
        self.assertEqual(pooled_itls_ms([{'chunks': chunks}]), [20.0, 50.0, 50.0, 50.0, 50.0])

    def test_a_rate_meets_the_objective_at_exactly_ninety_percent_of_planned_calls(self):
        self.assertTrue(summarize([ok()] * 9 + [FAILED], 10, 1, 2000, 100, .9, 1, 1000)['meets_joint_slo'])
        self.assertFalse(summarize([ok()] * 8 + [FAILED] * 2, 10, 1, 2000, 100, .9, 1, 1000)['meets_joint_slo'])

    def test_the_walk_stops_at_the_first_failing_rate_and_takes_the_smallest_goodput_across_seeds(self):
        def score(row, ttft_ms, tpot_ms, attainment):
            return summarize(row['requests'], 10, 1, ttft_ms, tpot_ms, attainment, row['offered'], 1000)
        rows = [{'run': 's1', 'offered': 1, 'rates': [1, 2, 3], 'requests': [ok()] * 10},
                {'run': 's1', 'offered': 2, 'rates': [1, 2, 3], 'requests': [ok()] * 10},
                {'run': 's1', 'offered': 3, 'rates': [1, 2, 3], 'requests': [ok()] * 10},
                {'run': 's2', 'offered': 1, 'rates': [1, 2, 3], 'requests': [ok()] * 9 + [FAILED]},
                {'run': 's2', 'offered': 2, 'rates': [1, 2, 3], 'requests': [ok()] * 8 + [FAILED] * 2},
                {'run': 's2', 'offered': 3, 'rates': [1, 2, 3], 'requests': [ok()] * 10}]
        result = highest_qualifying(rows, 2000, 100, .9, score)
        self.assertEqual(result['offered'], 1)
        self.assertEqual(result['useful_per_gpu'], 0.9)

    def test_goodput_is_good_calls_over_the_measured_time_over_the_gpus_under_each_rule(self):
        stall = [{'t_ms': 0, 'n': 1}, {'t_ms': 10, 'n': 1}, {'t_ms': 410, 'n': 1}]
        steady = [{'t_ms': 0, 'n': 1}, {'t_ms': 10, 'n': 1}, {'t_ms': 20, 'n': 1}]
        records = [ok(tpot=205, chunks=stall)] * 2 + [ok(tpot=10, chunks=steady)] * 6 + [FAILED] * 2
        mean = summarize(records, 4, 2, 2000, 300, .9, 1, 1000, tpot_metric='mean')
        worst = summarize(records, 4, 2, 2000, 300, .9, 1, 1000, tpot_metric='worst-itl')
        pooled = summarize(records, 4, 2, 2000, 300, .9, 1, 1000, tpot_metric='itl-p99')
        self.assertEqual(mean['useful_calls_per_s_per_gpu'], 8 / 4 / 2)
        self.assertEqual(worst['useful_calls_per_s_per_gpu'], 6 / 4 / 2)
        # The pooled p99 (400 ms > 300 ms) gates the rate, not which calls count toward goodput.
        self.assertFalse(pooled['meets_joint_slo'])
        self.assertEqual(pooled['useful_calls_per_s_per_gpu'], 8 / 4 / 2)


if __name__ == '__main__':
    unittest.main()
