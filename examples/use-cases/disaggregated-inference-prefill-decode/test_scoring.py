"""scoring.py on hand-built cases."""
import unittest
from scoring import client_check, highest_qualifying, percentile, pooled_itls_ms, summarize, worst_itl_ms


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

    def test_the_qualifying_rate_is_the_highest_rate_any_file_meets_with_its_largest_goodput_there(self):
        def score(row, ttft_ms, tpot_ms, attainment):
            return summarize(row['requests'], 10, 1, ttft_ms, tpot_ms, attainment, row['offered'], 1)
        rows = [{'offered': 1, 'requests': [ok()] * 10}, {'offered': 2, 'requests': [ok()] * 8 + [FAILED] * 2},
                {'offered': 1, 'requests': [ok()] * 9 + [FAILED]}, {'offered': 2, 'requests': [ok()] * 9 + [FAILED]},
                {'offered': 4, 'requests': [ok()] * 5 + [FAILED] * 5}]
        result = highest_qualifying(rows, 2000, 100, .9, score)
        # As in the original lab: 2 qualifies because one file met it there, although the other file at 2 did not.
        self.assertEqual((result['offered'], result['useful_per_gpu'], result['files_met'], result['files_measured']), (2, 0.9, 1, 2))
        self.assertIsNone(highest_qualifying(rows[-1:], 2000, 100, .9, score))

    def test_a_call_that_could_not_connect_is_a_failed_call_and_only_launch_lag_invalidates_a_rate(self):
        unreachable = {'ok': False, 'sent': True, 'client_launch_lag_ms': 0, 'chunks': [], 'error': 'ConnectError: refused'}
        s = client_check(summarize([ok()] * 19 + [unreachable], 10, 1, 2000, 100, .9, 1, 2), max_client_lag_ms=100)
        self.assertEqual((s['meets_joint_slo'], s['client_valid'], s['unreachable_calls'], s['slo_attainment_fraction']), (True, True, 1, .95))
        lagged = [dict(ok(), client_launch_lag_ms=150)] + [ok()] * 9
        self.assertFalse(client_check(summarize(lagged, 10, 1, 2000, 100, .9, 1, 1), max_client_lag_ms=100)['meets_joint_slo'])

    def test_goodput_is_good_calls_over_the_measured_time_over_the_gpus_under_each_rule(self):
        stall = [{'t_ms': 0, 'n': 1}, {'t_ms': 10, 'n': 1}, {'t_ms': 410, 'n': 1}]
        steady = [{'t_ms': 0, 'n': 1}, {'t_ms': 10, 'n': 1}, {'t_ms': 20, 'n': 1}]
        records = [ok(tpot=205, chunks=stall)] * 2 + [ok(tpot=10, chunks=steady)] * 6 + [FAILED] * 2
        mean = summarize(records, 4, 2, 2000, 300, .9, 1, 1000, tpot_metric='mean')
        worst = summarize(records, 4, 2, 2000, 300, .9, 1, 1000, tpot_metric='worst-itl')
        pooled = summarize(records, 4, 2, 2000, 300, .9, 1, 1000, tpot_metric='itl-p99')
        self.assertEqual(mean['useful_calls_per_s_per_gpu'], 8 / 4 / 2)
        self.assertEqual(worst['useful_calls_per_s_per_gpu'], 6 / 4 / 2)
        self.assertEqual(pooled['useful_calls_per_s_per_gpu'], 8 / 4 / 2)
        # No failed calls, so attainment passes and only the pooled p99 (400 ms) decides the rate.
        gated = [ok(tpot=205, chunks=stall)] * 2 + [ok(tpot=10, chunks=steady)] * 8
        self.assertFalse(summarize(gated, 4, 2, 2000, 300, .9, 1, 1000, tpot_metric='itl-p99')['meets_joint_slo'])
        self.assertTrue(summarize(gated, 4, 2, 2000, 500, .9, 1, 1000, tpot_metric='itl-p99')['meets_joint_slo'])


if __name__ == '__main__':
    unittest.main()
