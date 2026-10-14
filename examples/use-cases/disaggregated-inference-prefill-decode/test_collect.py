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
from scoring import UNAVAILABLE, summarize

spec = importlib.util.spec_from_file_location('collect', '5.collect.py')
collect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect)


def _run(directory, architecture, rate, rows, seed=345, rates=None):
    """One measured-rate file with the fields scoring.load_rows requires."""
    shape = {'name': 'B-long-context', 'input_tokens': 8192, 'output_tokens': 256}
    s = summarize(rows, 10, 1, 2000, 100, .9, rate, 100)
    arguments = {'max_client_lag_ms': 100, 'duration_s': 10, 'timeout_s': 310, 'context_length_tokens': 32768,
                 'gpus': 1, 'seed': seed, 'skip_warmup': False, 'stop_failure_fraction': .25,
                 'rates': rates or [rate], 'ramp_id': None}
    meta = {'architecture': architecture, 'instance_type': 'g7.48xlarge', 'region': 'test', 'evidence_scope': 'local-mock',
            'model': 'test-model', 'model_revision': 'r', 'shape': shape, 'traffic_sha256': '0' * 64, 'seed': seed, 'phase': 'measured',
            'arguments': arguments}
    path = Path(directory) / f'{architecture}-{seed}'
    path.mkdir(exist_ok=True)
    (path / f'{rate:g}-measured.json').write_text(json.dumps({'metadata': meta, 'summary': s, 'requests': rows}))
    return path


def _collect(out, *runs, report=False, extra=()):
    argv = ['5.collect.py', '--runs', *map(str, runs), '--output', str(out), *extra] + ([] if report else ['--no-report'])
    with patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()):
        collect.main()
    with (Path(out) / 'summary.csv').open() as f:
        rows = list(csv.DictReader(f))
    return rows, json.loads((Path(out) / 'qualified-rates.json').read_text())


def _call(gap_ms):
    chunks = [{'t_ms': 0, 'n': 1}, {'t_ms': 10, 'n': 1}, {'t_ms': 10 + gap_ms, 'n': 1}]
    return {'ok': True, 'sent': True, 'ttft_ms': 100, 'tpot_ms': (10 + gap_ms) / 2, 'client_launch_lag_ms': 0, 'chunks': chunks}


class CollectTests(unittest.TestCase):
    def test_summary_csv_has_one_row_per_file_and_rule_with_scorings_verdict(self):
        records = [_call(10)] * 8 + [_call(150)] * 2
        with tempfile.TemporaryDirectory() as d:
            rows, _ = _collect(Path(d) / 'out', _run(d, 'unified', 1, records))
        self.assertEqual(list(rows[0])[:9], list(collect.SUMMARY_LEADING_FIELDS))
        self.assertEqual([r['definition'] for r in rows], ['mean', 'itl-p99', 'worst-itl'])
        for r in rows:
            expected = summarize(records, 10, 1, 2000, 100, .9, 1, 100, tpot_metric=r['definition'])
            self.assertEqual(r['meets_joint_slo'], str(expected['meets_joint_slo']), r['definition'])
            self.assertEqual(float(r['useful_calls_per_s_per_gpu']), expected['useful_calls_per_s_per_gpu'], r['definition'])

    def test_qualified_rates_hold_each_stacks_rate_and_goodput_under_each_rule(self):
        # Unified at 2 tasks/s has two calls with a 150 ms gap: mean still passes (80 ms average),
        # worst-itl and itl-p99 fail there, so both walks stop at 1. Disaggregated passes both rates;
        # one failed call at 2 tasks/s leaves 9 good calls over 10 s on 1 GPU.
        steady, stalled, failed = _call(10), _call(150), {'ok': False, 'sent': True, 'client_launch_lag_ms': 0, 'chunks': []}
        with tempfile.TemporaryDirectory() as d:
            runs = sorted({_run(d, 'unified', 1, [steady] * 10, seed=1, rates=[1, 2]),
                           _run(d, 'unified', 2, [steady] * 8 + [stalled] * 2, seed=1, rates=[1, 2]),
                           _run(d, 'disaggregated', 1, [steady] * 10, seed=1, rates=[1, 2]),
                           _run(d, 'disaggregated', 2, [steady] * 9 + [failed], seed=1, rates=[1, 2])})
            _, by_definition = _collect(Path(d) / 'out', *runs)
        got = {(definition, arch): (v['highest_qualifying_offered_tasks_per_s'], v['useful_calls_per_s_per_gpu'])
               for definition, shapes in by_definition['by_definition'].items() for arch, v in shapes['B-long-context'].items()}
        self.assertEqual(got, {('mean', 'unified'): (2, 1.0), ('mean', 'disaggregated'): (2, 0.9),
                               ('itl-p99', 'unified'): (1, 1.0), ('itl-p99', 'disaggregated'): (2, 0.9),
                               ('worst-itl', 'unified'): (1, 1.0), ('worst-itl', 'disaggregated'): (2, 0.9)})

    def test_a_file_without_chunks_is_unavailable_under_the_gap_rules(self):
        records = [{'ok': True, 'sent': True, 'ttft_ms': 100, 'tpot_ms': 10, 'client_launch_lag_ms': 0}] * 10
        with tempfile.TemporaryDirectory() as d:
            rows, by_definition = _collect(Path(d) / 'out', _run(d, 'unified', 1, records))
        for r in rows:
            expected = 'True' if r['definition'] == 'mean' else UNAVAILABLE
            self.assertEqual(r['meets_joint_slo'], expected, r['definition'])
            if r['definition'] != 'mean':
                self.assertEqual((r['slo_attainment_fraction'], r['useful_calls_per_s_per_gpu']), (UNAVAILABLE, UNAVAILABLE))
        self.assertEqual(by_definition['by_definition']['worst-itl']['B-long-context']['unified']['highest_qualifying_offered_tasks_per_s'], UNAVAILABLE)

    def test_collect_writes_summary_qualified_rates_and_report_only(self):
        records = [_call(10)] * 10
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'out'
            _collect(out, _run(d, 'unified', 1, records, seed=1), _run(d, 'disaggregated', 1, records, seed=1), report=True)
            self.assertEqual(sorted(p.name for p in out.iterdir()), ['qualified-rates.json', 'report.html', 'summary.csv'])


    def test_another_objective_on_the_command_line_re_judges_the_stored_files(self):
        ok = lambda ttft: {'ok': True, 'sent': True, 'ttft_ms': ttft, 'tpot_ms': 10, 'client_launch_lag_ms': 0, 'chunks': []}
        with tempfile.TemporaryDirectory() as d:
            runs = (_run(d, 'unified', 1, [ok(1500)] * 10, seed=1), _run(d, 'disaggregated', 1, [ok(500)] * 10, seed=1))
            _, recorded = _collect(Path(d) / 'recorded', *runs)
            tighter_rows, tighter = _collect(Path(d) / 'tighter', *runs, extra=('--ttft-slo-ms', '1000'))
        rate = lambda q, arch: q['by_definition']['mean']['B-long-context'][arch]['highest_qualifying_offered_tasks_per_s']
        self.assertEqual((rate(recorded, 'unified'), rate(recorded, 'disaggregated')), (1, 1))
        self.assertEqual((rate(tighter, 'unified'), rate(tighter, 'disaggregated')), (None, 1))
        self.assertEqual(tighter['objectives_by_shape']['B-long-context']['ttft_bound_ms'], 1000)
        self.assertEqual({float(r['slo_ttft_ms']) for r in tighter_rows}, {1000.0})

    def test_a_file_that_does_not_reproduce_is_named_and_left_out(self):
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
                self.assertEqual({r['stack'] for r in csv.DictReader(f)}, {'unified'})


if __name__ == '__main__':
    unittest.main()
