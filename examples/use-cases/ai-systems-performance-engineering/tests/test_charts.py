"""CPU tests of the terminal charts (lib/charts.py and their callers). Standard library only.

Recorded values come from the run of 2026-11-03 on two g7.48xlarge (tests/fixtures/spain-20261103):
baseline 13.19 s, efa 5.76 s and sdpa 3.39 s for the same work on allocation 637, GEMM peak
167.9 TFLOP/s per GPU. Values marked synthetic are software fixtures, not measurements.
"""
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'lib'))
import charts  # noqa: E402
import workshop  # noqa: E402

SPAIN = Path(__file__).resolve().parent / 'fixtures' / 'spain-20261103'
# tools/profile-table.py output of the same session, rank and update chosen by default.
PROFILES = {
    'profile-socket-eager': dict(update=3.519, compute=1.316, nccl=3.489, exposed=2.193, idle=0.010,
                                 input_wait=0.001, attention=0.751, top=[]),
    'profile-efa-eager': dict(update=1.451, compute=1.373, nccl=1.093, exposed=0.067, idle=0.011,
                              input_wait=0.001, attention=0.770, top=[]),
    'profile-efa-sdpa': dict(update=0.851, compute=0.626, nccl=0.802, exposed=0.218, idle=0.008,
                             input_wait=0.001, attention=0.032, top=[]),
}
CUMULATIVE_TITLE = ('How much faster than baseline? Each bar adds one step '
                    '(your runs, same work on the same allocation)')


BAR = re.compile('[' + ''.join(charts.FILLS) + charts.LOST + charts.EIGHTHS[1:] + ']+')


def bar(line):
    """(the cells of the bar in a chart row, the text after it)."""
    match = BAR.search(line)
    return match.group(0), line[match.end():].strip()


def columns(lines):
    """The columns where the bars and the texts of these rows start."""
    return {(BAR.search(line).start(), len(line) - len(bar(line)[1])) for line in lines}


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), LAB / 'tools' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SpainResults(unittest.TestCase):
    """A results directory that holds the named Spain stages and the ceilings."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.results = Path(self.directory.name)
        shutil.copytree(SPAIN / 'ceilings', self.results / 'ceilings')

    def tearDown(self):
        self.directory.cleanup()

    def add(self, *names):
        for name in names:
            shutil.copytree(SPAIN / name, self.results / name)

    def edit_summary(self, name, **values):
        path = self.results / name / 'train' / 'summary.json'
        summary = json.loads(path.read_text())
        summary.update(values)
        path.write_text(json.dumps(summary))

    def assert_narrow(self, lines):
        self.assertEqual([line for line in lines if len(line) > charts.WIDTH], [])


class PrimitiveTests(unittest.TestCase):
    def test_solid_bars_use_eighth_blocks(self):
        self.assertEqual(charts.solid(2.5, 1.0), '██▌')
        self.assertEqual(charts.solid(0.26, 0.756), '▍')
        self.assertEqual(charts.solid(0.0, 1.0), '')

    def test_stacked_cells_keep_the_rounded_total_and_give_spare_cells_to_the_largest_remainders(self):
        # 0.770, 0.603, 0.067 and 0.011 s at 0.1 s per cell: 15 cells, the fourth part gets none.
        bar = charts.stacked([(0.770, 'a'), (0.603, 'b'), (0.067, 'c'), (0.011, 'd')], 0.1)
        self.assertEqual(bar, 'a' * 8 + 'b' * 6 + 'c')
        self.assertEqual(charts.stacked([(-1.0, 'a'), (0.5, 'b')], 0.1), 'b' * 5)

    def test_scales_are_one_two_two_and_a_half_or_five_times_a_power_of_ten(self):
        self.assertEqual(charts.nice_scale(3.519, 67), 0.1)
        self.assertEqual(charts.nice_scale(1.44, 52), 0.05)
        self.assertEqual(charts.nice_scale(1.44, 60), 0.025)


class CumulativeChartTests(SpainResults):
    def test_the_chart_grows_with_the_main_path(self):
        self.add('baseline')
        lines = workshop.cumulative_chart_lines(self.results)
        self.assertEqual(lines[0], CUMULATIVE_TITLE)
        self.assertEqual(bar(lines[1]), ('█' * 10, '1.00x'))
        self.assertEqual(lines[2:], ['  efa (efa, eager)          not run yet',
                                     '  sdpa (efa, sdpa)          not run yet',
                                     '                            █ baseline 1.00x   one cell = 0.1x'])
        self.add('efa')
        lines = workshop.cumulative_chart_lines(self.results)
        self.assertEqual(bar(lines[1]), ('█' * 10, '1.00x'))
        self.assertEqual(bar(lines[2]), ('█' * 10 + '▓' * 13, '2.29x  (EFA +1.29x)'))
        self.add('sdpa')
        lines = workshop.cumulative_chart_lines(self.results)
        self.assertEqual([bar(line) for line in lines[1:4]], [
            ('█' * 10, '1.00x'),
            ('█' * 10 + '▓' * 13, '2.29x  (EFA +1.29x)'),
            ('█' * 10 + '▓' * 13 + '▒' * 16, '3.89x  (SDPA +1.60x)')])
        self.assertEqual([line[:28] for line in lines[1:4]], ['  baseline (socket, eager)  ',
                                                             '  efa (efa, eager)          ',
                                                             '  sdpa (efa, sdpa)          '])
        self.assertEqual(len(columns(lines[1:4])), 1)
        self.assertEqual(lines[4], '                            █ baseline 1.00x   ▓ EFA +1.29x   ▒ SDPA +1.60x   '
                                   'one cell = 0.1x')
        self.assert_narrow(lines)

    def test_unpaired_failed_and_slower_runs_are_shown_as_such(self):
        self.add('baseline', 'efa', 'sdpa')
        self.edit_summary('sdpa', measured_seconds=6.6)  # synthetic: slower than efa
        lines = workshop.cumulative_chart_lines(self.results)
        # 13.19 / 6.6 = 2.00x: the bar keeps 20 cells of the efa bar and marks its 3 lost cells.
        self.assertEqual(bar(lines[3]), ('█' * 10 + '▓' * 10 + '·' * 3, '2.00x  (SDPA -0.29x, slower)'))
        self.assertEqual(lines[4:], ['                            █ baseline 1.00x   ▓ EFA +1.29x',
                                     '                            · SDPA -0.29x, lost against the bar before   '
                                     'one cell = 0.1x'])
        self.edit_summary('efa', allocation={'label': 'another allocation'})  # synthetic
        lines = workshop.cumulative_chart_lines(self.results)
        self.assertEqual(lines[2], '  efa (efa, eager)          not compared: different allocation')
        # Without an efa bar, the sdpa bar holds both steps; nothing is split or estimated.
        self.assertEqual(bar(lines[3]), ('█' * 10 + '▓' * 10, '2.00x  (EFA + SDPA +1.00x)'))
        run = json.loads((self.results / 'efa' / 'run.json').read_text())
        (self.results / 'efa' / 'run.json').write_text(json.dumps(dict(run, exit_status=4)))
        self.assertEqual(workshop.cumulative_chart_lines(self.results)[2],
                         '  efa (efa, eager)          failed, so it has no bar; run it again')
        shutil.rmtree(self.results / 'baseline')
        lines = workshop.cumulative_chart_lines(self.results)
        self.assertEqual(lines[1:], ['  baseline (socket, eager)  not run yet',
                                     '  efa (efa, eager)          failed, so it has no bar; run it again',
                                     '  sdpa (efa, sdpa)          no baseline to compare with'])


class GoodputChartTests(SpainResults):
    def test_reference_bar_from_the_spain_sdpa_run_and_missing_saving_runs(self):
        self.add('sdpa')
        lines = workshop.goodput_chart_lines(self.results)
        self.assertEqual(lines[:3], ['Training goodput, M tokens per GPU-hour (higher is better)',
                                     '  ckpt-sync, synchronous saves            not run yet',
                                     '  ckpt-process, process-mode saves        not run yet'])
        # 77,275 tokens/s x 3,600 / 16 GPUs.
        self.assertTrue(lines[3].startswith('  sdpa, no saves (throughput, reference)  █'))
        self.assertTrue(lines[3].endswith('  17.39'))
        self.assert_narrow(lines)

    def test_saving_runs_beside_the_reference(self):
        self.add('sdpa')
        for name, mode, value in (('ckpt-sync', 'sync', 12.5e6), ('ckpt-process', 'process', 15.0e6)):  # synthetic
            stage = workshop.plan('checkpoint', {'LLM_TRANSPORT': 'efa', 'ATTENTION': 'sdpa',
                                                 'CHECKPOINT_MODE': mode})[0 if mode == 'sync' else 1]
            (self.results / name / 'train').mkdir(parents=True)
            (self.results / name / 'run.json').write_text(json.dumps(dict(plan=stage, status='finished', exit_status=0)))
            (self.results / name / 'train' / 'summary.json').write_text(json.dumps(dict(
                training_goodput={'tokens_per_allocated_gpu_hour': value})))
        lines = workshop.goodput_chart_lines(self.results)
        rows = [bar(line) for line in lines[1:4]]
        self.assertEqual([text for _, text in rows], ['12.50', '15.00', '17.39'])
        # One scale for the three bars: 17.39 fills the bar width, and the others are in proportion.
        widths = [len(cells) - 1 + charts.EIGHTHS.index(cells[-1]) / 8 if cells[-1] in charts.EIGHTHS
                  else len(cells) for cells, _ in rows]
        self.assertAlmostEqual(widths[0] / widths[2], 12.50 / 17.39, delta=0.01)
        self.assertAlmostEqual(widths[1] / widths[2], 15.00 / 17.39, delta=0.01)
        self.assertEqual(len(columns(lines[1:4])), 1)
        self.assertEqual(lines[4], '  The reference is the throughput of the timed updates of stage sdpa, '
                                   'a run that does not save.')


class UpdateTimeChartTests(SpainResults):
    def test_baseline_and_probe_against_the_gemm_peak(self):
        self.add('baseline', 'probe-socket-eager')
        ceilings = workshop.read_ceilings(self.results)
        lines = workshop.update_time_chart_lines(self.results, ['baseline'], ceilings, 16)
        # 13.19 s over 4 updates; 65,504 targets x 11.48 GFLOP / (16 x 167.9 TFLOP/s) = 0.28 s.
        self.assertEqual(lines[0], "One update: the model's arithmetic at your GEMM peak, and the rest "
                                   '(mean of the timed updates)')
        self.assertEqual(lines[1][:29], '  baseline, 2,048 positions  ')
        self.assertEqual(bar(lines[1]), ('█' * 3 + '▒' * 30, '3.30 s'))
        self.assertEqual(lines[2:], ['                             █ arithmetic at your GEMM peak 0.28 s   '
                                     '▒ the rest 3.02 s',
                                     '                             one cell = 0.1 s'])
        stage = json.loads((self.results / 'probe-socket-eager' / 'run.json').read_text())['plan']
        self.assertEqual(workshop.update_time_names(stage), ['baseline', 'probe-socket-eager'])
        lines = workshop.update_time_chart_lines(self.results, workshop.update_time_names(stage), ceilings, 16)
        self.assertEqual(lines[4], '                                       '
                                   '█ arithmetic at your GEMM peak 0.13 s   ▒ the rest 3.10 s')
        lines = workshop.update_time_chart_lines(self.results, ['baseline', 'efa'], None, 16)
        self.assertEqual(lines[2], '                             arithmetic n/a (run bash tools/hold-and-measure.sh)')
        self.assertEqual(lines[3], '  efa                        not run yet')

    def test_result_block_of_sdpa_is_followed_by_its_three_charts(self):
        self.add('baseline', 'efa', 'sdpa')
        run = json.loads((self.results / 'sdpa' / 'run.json').read_text())
        lines = workshop.result_lines(run['plan'], self.results / 'sdpa', run, self.results, 16)
        titles = [index for index, line in enumerate(lines) if not line.startswith(' ') and line]
        self.assertEqual([lines[index] for index in titles], [
            'sdpa (efa, sdpa attention, no saving)',
            "One update: the model's arithmetic at your GEMM peak, and the rest (mean of the timed updates)",
            'Peak memory of the largest rank', CUMULATIVE_TITLE])
        update = lines[titles[1] + 1:titles[1] + 6]
        self.assertEqual([bar(update[0]), bar(update[2])], [('█' * 11 + '▒' * 47, '1.44 s'),
                                                            ('█' * 11 + '▒' * 23, '0.85 s')])
        self.assertEqual([update[1].strip(), update[3].strip(), update[4].strip()], [
            '█ arithmetic at your GEMM peak 0.28 s   ▒ the rest 1.16 s',
            '█ arithmetic at your GEMM peak 0.28 s   ▒ the rest 0.57 s', 'one cell = 0.025 s'])
        memory = lines[titles[2] + 1:titles[2] + 3]
        self.assertEqual([bar(line) for line in memory], [('█' * 64, '26.68 GiB'), ('█' * 20, '8.34 GiB')])
        self.assertEqual(bar(lines[titles[3] + 3])[1], '3.89x  (SDPA +1.60x)')
        self.assert_narrow(lines)

    def test_charts_follow_only_the_stages_they_belong_to(self):
        self.add('baseline', 'sdpa-socket', 'probe-efa-eager', 'efa')
        names = {}
        for name in ('baseline', 'sdpa-socket', 'efa', 'probe-efa-eager'):
            stage = json.loads((self.results / name / 'run.json').read_text())['plan']
            names[name] = [line for line in workshop.stage_chart_lines(stage, self.results, 16)
                           if line and not line.startswith(' ')]
        update, cumulative = ("One update: the model's arithmetic at your GEMM peak, and the rest "
                              '(mean of the timed updates)'), CUMULATIVE_TITLE
        self.assertEqual(names, {'baseline': [update, cumulative], 'sdpa-socket': [update],
                                 'efa': [update, cumulative], 'probe-efa-eager': [update]})
        profile = workshop.plan('profile', {'LLM_TRANSPORT': 'efa', 'ATTENTION': 'eager', 'CHECKPOINT_MODE': 'none'})[0]
        self.assertEqual(workshop.stage_chart_lines(profile, self.results, 16), [])


class ProgressionChartTests(SpainResults):
    def test_table_then_both_charts(self):
        self.add('baseline', 'probe-socket-eager', 'sdpa-socket', 'efa', 'probe-efa-eager', 'sdpa')
        tool = load_tool('progression')
        for markdown in (False, True):
            output = io.StringIO()
            with patch.dict(os.environ, {'LLM_RESULTS_DIR': str(self.results), 'GPUS_PER_NODE': '8'}), \
                    redirect_stdout(output):
                self.assertEqual(tool.main(['--markdown'] if markdown else []), 0)
            lines = output.getvalue().splitlines()
            efa = next(line for line in lines if line.startswith(('efa ', '| efa |')))
            self.assertIn('2.29x', efa)
            chart = lines.index(CUMULATIVE_TITLE)
            self.assertLess(lines.index('(2) probe-efa-eager vs baseline: not compared, different workload'), chart)
            self.assertEqual(bar(lines[chart + 3])[1], '3.89x  (SDPA +1.60x)')
            self.assertIn('Training goodput, M tokens per GPU-hour (higher is better)', lines[chart:])
            self.assertEqual(lines[chart - 1], '```text' if markdown else '')
            self.assertEqual(lines[-1] == '```', markdown)
            self.assert_narrow(lines[chart:])


class ProfileChartTests(SpainResults):
    def test_each_read_table_joins_the_chart_of_the_next_profile(self):
        tool = load_tool('profile-table')
        key = tool.table_key('slowest', 5)
        for name in ('profile-socket-eager', 'profile-efa-eager'):
            (self.results / name).mkdir()
            (self.results / name / 'run.json').write_text('{}')
        tool.save_table(self.results / 'profile-socket-eager', key, 'Profile of profile-socket-eager',
                        PROFILES['profile-socket-eager'])
        lines = tool.chart_lines(self.results, 'profile-efa-eager', 'slowest', 5, PROFILES['profile-efa-eager'])
        self.assertEqual(lines[0], 'Where the time of update 5 goes, for the slowest rank of each profile')
        # Attention core, other compute, communication not overlapped and GPU idle, at 0.1 s per cell.
        self.assertEqual([bar(lines[1]), bar(lines[3])], [('█' * 7 + '▓' * 6 + '▒' * 22, '3.519 s'),
                                                          ('█' * 8 + '▓' * 6 + '▒', '1.451 s')])
        self.assertEqual([lines[1][:24], lines[3][:24]], ['  profile-socket-eager  ', '  profile-efa-eager     '])
        self.assertEqual(lines[2:3] + lines[4:], [
            '                        █ 0.751  ▓ 0.565  ▒ 2.193  ░ 0.010 s   input wait (CPU) 0.001 s',
            '                        █ 0.770  ▓ 0.603  ▒ 0.067  ░ 0.011 s   input wait (CPU) 0.001 s',
            '  profile-efa-sdpa      not run yet',
            '                        █ attention core kernels   ▓ other compute kernels',
            '                        ▒ communication not overlapped   ░ GPU idle   one cell = 0.1 s',
            '  input wait (CPU) can overlap the GPU parts, so the bar leaves it out'])
        self.assert_narrow(lines)
        (self.results / 'profile-efa-sdpa').mkdir()
        (self.results / 'profile-efa-sdpa' / 'run.json').write_text('{}')
        lines = tool.chart_lines(self.results, 'profile-efa-eager', 'all', 5, PROFILES['profile-efa-eager'])
        self.assertEqual(lines[0], 'Where the time of update 5 goes, for the mean of all ranks')
        self.assertEqual(lines[1], '  profile-socket-eager  run python3 tools/profile-table.py '
                                   'profile-socket-eager --rank all to add it')
        self.assertEqual(lines[4], '  profile-efa-sdpa      run python3 tools/profile-table.py '
                                   'profile-efa-sdpa --rank all to add it')


class ProtocolChartTests(unittest.TestCase):
    def test_bus_bandwidth_of_both_runs_at_each_size(self):
        directory = SPAIN / 'nccl-protocols' / '20261103T133213Z'
        # The protocol columns of the same run's report, from rank 0's TUNING lines.
        choice = {size << 20: ['RING/LL' if size <= 8 else 'RING/SIMPLE'] for size in (1, 2, 4, 8, 16, 32, 64)}
        simple = {size << 20: ['TREE/SIMPLE' if size <= 8 else 'RING/SIMPLE'] for size in (1, 2, 4, 8, 16, 32, 64)}
        runs = [(label, workshop.nccl_tests_rows((directory / variant / 'all_reduce_perf.txt').read_text()), choices)
                for (variant, label), choices in zip(workshop.PROTOCOL_RUNS, (choice, simple))]
        lines = workshop.protocol_chart_lines(runs)
        self.assertEqual(lines[0], 'Bus bandwidth of all_reduce_perf by message size, GB/s (higher is better)')
        self.assertEqual(len(lines), 15)
        self.assertEqual([line[:31] for line in lines[3:5]], ['    2 MiB  NCCL_PROTO not set  ',
                                                              '           NCCL_PROTO=Simple   '])
        # One scale: 37.80 GB/s fills the bar, so 0.26 GB/s is three eighths of a cell.
        self.assertEqual(bar(lines[3]), ('▍', '0.26  RING/LL'))
        self.assertEqual(bar(lines[8])[1], '13.94  TREE/SIMPLE')
        self.assertEqual(bar(lines[14]), ('█' * 49, '37.80  RING/SIMPLE'))
        self.assertEqual(len({BAR.search(line).start() for line in lines[1:]}), 1)
        self.assertTrue(all(len(line) <= charts.WIDTH for line in lines))


if __name__ == '__main__':
    unittest.main()
