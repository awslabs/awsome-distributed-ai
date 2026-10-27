"""5.collect.py through its main()."""
import contextlib
import csv
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scoring import UNAVAILABLE, client_check, summarize

spec = importlib.util.spec_from_file_location('collect', '5.collect.py')
collect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect)


def _run(directory, architecture, rate, rows, seed=345, rates=None, duration_s=10, name=None):
    """One measured-rate file as benchmark.py writes it."""
    shape = {'name': 'B-long-context', 'input_tokens': 8192, 'output_tokens': 256}
    s = client_check(summarize(rows, 10, 1, 2000, 100, .9, rate, sum(r['sent'] for r in rows) / 10), 100)
    arguments = {'max_client_lag_ms': 100, 'duration_s': duration_s, 'timeout_s': 310, 'context_length_tokens': 32768,
                 'gpus': 1, 'seed': seed, 'skip_warmup': False, 'stop_failure_fraction': .25, 'rates': rates or [rate]}
    meta = {'architecture': architecture, 'instance_type': 'g7.48xlarge', 'region': 'test', 'evidence_scope': 'local-mock',
            'model': 'test-model', 'model_revision': 'r', 'shape': shape, 'traffic_sha256': '0' * 64, 'seed': seed, 'phase': 'measured',
            'arguments': arguments}
    path = Path(directory) / (name or f'{architecture}-{seed}')
    path.mkdir(exist_ok=True)
    (path / f'{rate:g}-measured.json').write_text(json.dumps({'metadata': meta, 'summary': s, 'requests': rows}))
    return path


def _collect(out, *runs, report=False, extra=()):
    argv = ['5.collect.py', '--runs', *map(str, runs), '--output', str(out), *extra] + ([] if report else ['--no-report'])
    printed = io.StringIO()
    with patch('sys.argv', argv), contextlib.redirect_stdout(printed):
        collect.main()
    with (Path(out) / 'summary.csv').open() as f:
        rows = list(csv.DictReader(f))
    return rows, json.loads((Path(out) / 'qualified-rates.json').read_text()), printed.getvalue()


def _call(gap_ms):
    chunks = [{'t_ms': 0, 'n': 1}, {'t_ms': 10, 'n': 1}, {'t_ms': 10 + gap_ms, 'n': 1}]
    return {'ok': True, 'sent': True, 'ttft_ms': 100, 'tpot_ms': (10 + gap_ms) / 2, 'client_launch_lag_ms': 0, 'chunks': chunks}


FAILED = {'ok': False, 'sent': True, 'client_launch_lag_ms': 0, 'chunks': []}


def _group(qualified, arch):
    return next(g for g in qualified if g['group'][0] == arch)


class CollectTests(unittest.TestCase):
    def test_summary_csv_keeps_the_original_row_per_file_and_adds_the_gap_rules(self):
        records = [_call(10)] * 8 + [_call(150)] * 2
        with tempfile.TemporaryDirectory() as d:
            run = _run(d, 'unified', 1, records)
            stored = json.loads((run / '1-measured.json').read_text())['summary']
            rows, _, _ = _collect(Path(d) / 'out', run)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        for key, value in stored.items():
            self.assertEqual(row[key], '' if value is None else str(value), key)
        for d, prefix in (('itl-p99', 'itl_p99'), ('worst-itl', 'worst_itl')):
            expected = summarize(records, 10, 1, 2000, 100, .9, 1, 1, tpot_metric=d)
            self.assertEqual(row[f'{prefix}_meets_joint_slo'], str(expected['meets_joint_slo']), d)
            self.assertEqual(float(row[f'{prefix}_useful_calls_per_s_per_gpu']), expected['useful_calls_per_s_per_gpu'], d)
        self.assertEqual(row['objective_given_on_command_line'], 'False')

    def test_qualified_rates_keep_the_original_fields_and_add_each_rules_rate_and_goodput_at_that_rate(self):
        # Unified passes 1 and, under mean only, 2 (two calls with a 150 ms gap average 80 ms); it fails 4 with 13 good calls
        # of 20, which still gives it the largest observed goodput. Disaggregated passes 2 with 9 good calls of 10.
        steady, stalled = _call(10), _call(150)
        with tempfile.TemporaryDirectory() as d:
            runs = sorted({_run(d, 'unified', 1, [steady] * 10, seed=1, rates=[1, 2, 4]),
                           _run(d, 'unified', 2, [steady] * 8 + [stalled] * 2, seed=1, rates=[1, 2, 4]),
                           _run(d, 'unified', 4, [steady] * 13 + [FAILED] * 7, seed=1, rates=[1, 2, 4]),
                           _run(d, 'disaggregated', 2, [steady] * 9 + [FAILED], seed=1, rates=[2])})
            _, qualified, _ = _collect(Path(d) / 'out', *runs)
        u, dis = _group(qualified, 'unified'), _group(qualified, 'disaggregated')
        self.assertEqual((u['highest_tested_offered_tasks_per_s_meeting_slo'], u['max_observed_useful_calls_per_s_per_gpu']), (2, 1.3))
        self.assertEqual(u['by_rule']['mean'], {'highest_tested_offered_tasks_per_s_meeting_slo': 2, 'useful_calls_per_s_per_gpu_at_that_rate': 1.0})
        self.assertEqual(u['by_rule']['worst-itl'], {'highest_tested_offered_tasks_per_s_meeting_slo': 1, 'useful_calls_per_s_per_gpu_at_that_rate': 1.0})
        self.assertEqual(dis['by_rule']['itl-p99'], {'highest_tested_offered_tasks_per_s_meeting_slo': 2, 'useful_calls_per_s_per_gpu_at_that_rate': 0.9})

    def test_a_file_without_chunks_is_unavailable_under_the_gap_rules_and_named(self):
        records = [{'ok': True, 'sent': True, 'ttft_ms': 100, 'tpot_ms': 10, 'client_launch_lag_ms': 0}] * 10
        with tempfile.TemporaryDirectory() as d:
            rows, qualified, printed = _collect(Path(d) / 'out', _run(d, 'unified', 1, records))
        self.assertEqual(rows[0]['meets_joint_slo'], 'True')
        self.assertEqual((rows[0]['itl_p99_meets_joint_slo'], rows[0]['worst_itl_useful_calls_per_s_per_gpu']), (UNAVAILABLE, UNAVAILABLE))
        self.assertEqual(_group(qualified, 'unified')['by_rule']['worst-itl'], UNAVAILABLE)
        self.assertIn('has no per-chunk timing', printed)

    def test_files_from_the_original_lab_and_from_this_version_give_the_same_columns_in_either_order(self):
        old = [{'ok': True, 'sent': True, 'ttft_ms': 100, 'tpot_ms': 10, 'client_launch_lag_ms': 0}] * 10
        with tempfile.TemporaryDirectory() as d:
            a = _run(d, 'unified', 1, old, name='a-old')
            b = _run(d, 'unified', 2, [_call(10)] * 10, name='b-new')
            first, _, _ = _collect(Path(d) / 'one', a, b)
            second, _, _ = _collect(Path(d) / 'two', b, a)
        self.assertEqual(list(first[0]), list(second[0]))
        self.assertEqual(len(first), 2)

    def test_a_report_rendering_error_keeps_the_other_outputs_and_exits_non_zero(self):
        records = [_call(10)] * 10
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'out'
            with patch.object(collect.report, 'render_sections', side_effect=KeyError('section')), self.assertRaises(SystemExit) as refusal:
                _collect(out, _run(d, 'unified', 1, records, seed=1), _run(d, 'disaggregated', 1, records, seed=1), report=True)
            self.assertIn('report.html rendering failed (KeyError', str(refusal.exception))
            self.assertEqual(sorted(p.name for p in out.iterdir()), ['qualified-rates.json', 'summary.csv'])

    def test_a_rate_that_fails_only_on_client_launch_lag_names_it_in_the_report(self):
        lagged = [dict(_call(10), client_launch_lag_ms=250)] + [_call(10)] * 9
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'out'
            rows, _, _ = _collect(out, _run(d, 'unified', 1, lagged, seed=1), _run(d, 'disaggregated', 1, [_call(10)] * 10, seed=1), report=True)
            html = (out / 'report.html').read_text()
        self.assertEqual({r['meets_joint_slo'] for r in rows if r['architecture'] == 'unified'}, {'False'})
        self.assertIn('<span class="pf fail">invalid</span> <small class="over">launch lag 250 ms over 100 ms</small>', html)

    def test_calls_that_could_not_connect_are_failed_calls_counted_in_the_report_without_invalidating_the_rate(self):
        unreachable = [{'ok': False, 'sent': True, 'error': 'ConnectError', 'client_launch_lag_ms': 0, 'chunks': []}] + [_call(10)] * 9
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'out'
            rows, _, printed = _collect(out, _run(d, 'unified', 1, unreachable, seed=1), _run(d, 'disaggregated', 1, [_call(10)] * 10, seed=1), report=True)
            html = (out / 'report.html').read_text()
        u = next(r for r in rows if r['architecture'] == 'unified')
        self.assertEqual((u['meets_joint_slo'], u['client_valid'], u['unreachable_calls']), ('True', 'True', '1'))
        self.assertIn('<span class="pf pass">ok</span><small>1 calls could not connect</small>', html)
        self.assertIn('1 of 10 calls could not connect; if they cluster or are many, rerun the ramp from this rate into the same --output', html)

    def test_pooled_ttft_modes_relabel_the_itl_p99_attainment_and_the_percentile_figure(self):
        records = [_call(10)] * 10
        for metric, position in (('p95', 'is 19, so p95'), ('p99', 'is 20, so p99')):
            with tempfile.TemporaryDirectory() as d:
                out = Path(d) / 'out'
                _collect(out, _run(d, 'unified', 1, records, seed=1), _run(d, 'disaggregated', 1, records, seed=1), report=True,
                         extra=('--ttft-metric', metric))
                html = (out / 'report.html').read_text()
            self.assertIn('share of planned calls that completed, pooled p99', html, metric)
            self.assertIn(f'TTFT is judged as {metric} of the completed calls', html, metric)
            self.assertIn(position, html, metric)

    def test_another_objective_on_the_command_line_recomputes_the_rows_and_marks_them(self):
        ok = lambda ttft: {'ok': True, 'sent': True, 'ttft_ms': ttft, 'tpot_ms': 10, 'client_launch_lag_ms': 0, 'chunks': []}
        with tempfile.TemporaryDirectory() as d:
            runs = (_run(d, 'unified', 1, [ok(1500)] * 10, seed=1), _run(d, 'disaggregated', 1, [ok(500)] * 10, seed=1))
            _, recorded, _ = _collect(Path(d) / 'recorded', *runs)
            tighter_rows, tighter, printed = _collect(Path(d) / 'tighter', *runs, extra=('--ttft-slo-ms', '1000'))
            stored = json.loads((runs[0] / '1-measured.json').read_text())['summary']
        self.assertEqual([_group(recorded, a)['highest_tested_offered_tasks_per_s_meeting_slo'] for a in ('unified', 'disaggregated')], [1, 1])
        self.assertEqual([_group(tighter, a)['highest_tested_offered_tasks_per_s_meeting_slo'] for a in ('unified', 'disaggregated')], [None, 1])
        self.assertEqual({float(r['slo_ttft_ms']) for r in tighter_rows}, {1000.0})
        self.assertEqual({r['objective_given_on_command_line'] for r in tighter_rows}, {'True'})
        self.assertIn('judged at ttft-slo-ms given on the command line', printed)
        self.assertEqual(stored['slo_ttft_ms'], 2000)

    def test_a_file_whose_records_and_summary_disagree_is_left_out_listed_and_exits_non_zero(self):
        records = [_call(10)] * 10
        with tempfile.TemporaryDirectory() as d:
            good = _run(d, 'unified', 1, records, seed=1)
            tampered = _run(d, 'disaggregated', 1, records, seed=1) / '1-measured.json'
            doc = json.loads(tampered.read_text())
            doc['summary']['meets_joint_slo'] = not doc['summary']['meets_joint_slo']
            tampered.write_text(json.dumps(doc))
            out = Path(d) / 'out'
            with self.assertRaises(SystemExit) as refusal:
                _collect(out, good, tampered.parent)
            self.assertIn(str(tampered), str(refusal.exception))
            with (out / 'summary.csv').open() as f:
                self.assertEqual({r['architecture'] for r in csv.DictReader(f)}, {'unified'})

    def test_a_reversed_ramp_that_stopped_at_its_first_rate_is_collected_with_the_ascending_one(self):
        with tempfile.TemporaryDirectory() as d:
            runs = sorted({_run(d, 'unified', 1, [_call(10)] * 10, seed=1, rates=[1, 2]),
                           _run(d, 'unified', 2, [_call(10)] * 10, seed=1, rates=[1, 2]),
                           _run(d, 'unified', 2, [_call(10)] * 5 + [FAILED] * 5, seed=2, rates=[2, 1])})
            _, qualified, _ = _collect(Path(d) / 'out', *runs)
        self.assertEqual(_group(qualified, 'unified')['highest_tested_offered_tasks_per_s_meeting_slo'], 2)

    def test_a_pair_measured_with_different_settings_still_gets_a_map_with_the_difference_marked(self):
        records = [_call(10)] * 10
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'out'
            _, _, _ = _collect(out, _run(d, 'unified', 1, records, seed=1, duration_s=20), _run(d, 'disaggregated', 1, records, seed=2), report=True)
            html = (out / 'report.html').read_text()
        self.assertIn('duration_s differs (disaggregated 10; unified 20); measure again with them aligned', html)
        self.assertIn('seeds differ (disaggregated 2; unified 1)', html)
        self.assertIn('class="pmap"', html)

    def test_one_stack_only_draws_no_report_removes_the_earlier_one_and_says_why(self):
        records = [_call(10)] * 10
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'out'
            _collect(out, _run(d, 'unified', 1, records, seed=1), _run(d, 'disaggregated', 1, records, seed=1), report=True)
            self.assertTrue((out / 'report.html').exists())
            _, _, printed = _collect(out, _run(d, 'unified', 1, records, seed=1), report=True)
            self.assertFalse((out / 'report.html').exists())
        self.assertIn('Removed the earlier', printed)
        self.assertIn('has no unified and disaggregated pair', printed)


    def test_unreadable_and_malformed_files_are_left_out_and_listed_while_the_others_are_collected(self):
        with tempfile.TemporaryDirectory() as d:
            good = _run(d, 'unified', 1, [_call(10)] * 10, seed=1)
            (good / '2-measured.json').symlink_to(Path(d) / 'missing.json')
            (good / '4-measured.json').write_text(json.dumps({'metadata': [], 'summary': {}, 'requests': []}))
            out = Path(d) / 'out'
            with self.assertRaises(SystemExit) as refusal:
                _collect(out, good)
            with (out / 'summary.csv').open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 1)
        self.assertIn('2-measured.json', str(refusal.exception))
        self.assertIn('4-measured.json', str(refusal.exception))

    def test_a_chunk_without_its_token_count_makes_the_gap_rules_unavailable_without_stopping(self):
        broken = [dict(_call(10), chunks=[{'t_ms': 0, 'n': 1}, {'t_ms': 10}])] + [_call(10)] * 9
        with tempfile.TemporaryDirectory() as d:
            rows, _, _ = _collect(Path(d) / 'out', _run(d, 'unified', 1, broken, seed=1))
        self.assertEqual((rows[0]['meets_joint_slo'], rows[0]['itl_p99_meets_joint_slo']), ('True', UNAVAILABLE))

    def test_a_value_equal_to_the_recorded_one_is_not_marked_as_given_and_summary_csv_names_the_seed(self):
        with tempfile.TemporaryDirectory() as d:
            rows, _, printed = _collect(Path(d) / 'out', _run(d, 'unified', 1, [_call(10)] * 10, seed=7), extra=('--tpot-slo-ms', '100', '--ttft-metric', 'per-call'))
        self.assertEqual((rows[0]['objective_given_on_command_line'], rows[0]['seed']), ('False', '7'))
        self.assertNotIn('given on the command line', printed)

    def test_marks_are_printed_when_no_report_is_drawn_and_a_missing_traffic_hash_is_marked(self):
        with tempfile.TemporaryDirectory() as d:
            u = _run(d, 'unified', 1, [_call(10)] * 10, seed=1, duration_s=20)
            dis = _run(d, 'disaggregated', 1, [_call(10)] * 10, seed=1)
            doc = json.loads((dis / '1-measured.json').read_text())
            del doc['metadata']['traffic_sha256']
            (dis / '1-measured.json').write_text(json.dumps(doc))
            _, _, printed = _collect(Path(d) / 'out', u, dis)
        self.assertIn('B-long-context: duration_s differs', printed)
        self.assertIn('traffic_sha256 unknown for disaggregated', printed)

    def test_a_run_with_no_readable_file_removes_the_earlier_report(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'out'
            _collect(out, _run(d, 'unified', 1, [_call(10)] * 10, seed=1), _run(d, 'disaggregated', 1, [_call(10)] * 10, seed=1), report=True)
            empty = Path(d) / 'empty'
            empty.mkdir()
            with self.assertRaises(SystemExit), contextlib.redirect_stdout(io.StringIO()):
                with patch('sys.argv', ['5.collect.py', '--runs', str(empty), '--output', str(out)]):
                    collect.main()
            self.assertFalse((out / 'report.html').exists())


if __name__ == '__main__':
    unittest.main()
