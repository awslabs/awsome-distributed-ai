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

    def test_neighbouring_fills_are_two_shade_steps_apart(self):
        shade = {'█': 4, '▓': 3, '▒': 2, '░': 1}
        self.assertTrue(all(abs(shade[a] - shade[b]) >= 2 for a, b in zip(charts.FILLS[:3], charts.FILLS[1:3])))
        # The update chart's two parts, FILLS[0] and FILLS[3], are two steps apart as well.
        self.assertEqual(shade[charts.FILLS[0]] - shade[charts.FILLS[3]], 2)

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
        self.assertEqual(bar(lines[2]), ('█' * 10 + '░' * 13, '2.29x  (EFA +1.29x)'))
        self.add('sdpa')
        lines = workshop.cumulative_chart_lines(self.results)
        self.assertEqual([bar(line) for line in lines[1:4]], [
            ('█' * 10, '1.00x'),
            ('█' * 10 + '░' * 13, '2.29x  (EFA +1.29x)'),
            ('█' * 10 + '░' * 13 + '▓' * 16, '3.89x  (SDPA +1.60x)')])
        self.assertEqual([line[:28] for line in lines[1:4]], ['  baseline (socket, eager)  ',
                                                             '  efa (efa, eager)          ',
                                                             '  sdpa (efa, sdpa)          '])
        self.assertEqual(len(columns(lines[1:4])), 1)
        self.assertEqual(lines[4], '                            █ baseline 1.00x   ░ EFA +1.29x   ▓ SDPA +1.60x   '
                                   'one cell = 0.1x')
        self.assert_narrow(lines)

    def test_unpaired_failed_and_slower_runs_are_shown_as_such(self):
        self.add('baseline', 'efa', 'sdpa')
        self.edit_summary('sdpa', measured_seconds=6.6)  # synthetic: slower than efa
        lines = workshop.cumulative_chart_lines(self.results)
        # 13.19 / 6.6 = 2.00x: the bar keeps 20 cells of the efa bar and marks its 3 lost cells.
        self.assertEqual(bar(lines[3]), ('█' * 10 + '░' * 10 + '·' * 3, '2.00x  (SDPA -0.29x, slower)'))
        self.assertEqual(lines[4:], ['                            █ baseline 1.00x   ░ EFA +1.29x',
                                     '                            · SDPA -0.29x, lost against the bar before   '
                                     'one cell = 0.1x'])
        self.edit_summary('efa', allocation={'label': 'another allocation'})  # synthetic
        lines = workshop.cumulative_chart_lines(self.results)
        self.assertEqual(lines[2], '  efa (efa, eager)          not compared: different allocation')
        # Without an efa bar, the sdpa bar holds both steps; nothing is split or estimated.
        self.assertEqual(bar(lines[3]), ('█' * 10 + '░' * 10, '2.00x  (EFA + SDPA +1.00x)'))
        run = json.loads((self.results / 'efa' / 'run.json').read_text())
        (self.results / 'efa' / 'run.json').write_text(json.dumps(dict(run, exit_status=4)))
        self.assertEqual(workshop.cumulative_chart_lines(self.results)[2],
                         '  efa (efa, eager)          failed, so it has no bar; run it again')
        shutil.rmtree(self.results / 'baseline')
        lines = workshop.cumulative_chart_lines(self.results)
        self.assertEqual(lines[1:], ['  baseline (socket, eager)  not run yet',
                                     '  efa (efa, eager)          failed, so it has no bar; run it again',
                                     '  sdpa (efa, sdpa)          no baseline to compare with'])


GOODPUT_TITLE = 'Training goodput, from 0 to 1 (higher is better)'
GOODPUT_NOTE = ('  Training goodput = (retained tokens / interval) / (tokens/s of stage sdpa, '
                'which does not save)')
SDPA_RATE = 77274.93663399987  # tokens/s of the Spain sdpa run, tests/fixtures/spain-20261103


class GoodputChartTests(SpainResults):
    def saving_stage(self, mode, goodput, targets=65504):
        """A finished saving stage whose retained rate is goodput times the sdpa rate (synthetic)."""
        stage = workshop.plan('checkpoint', {'LLM_TRANSPORT': 'efa', 'ATTENTION': 'sdpa',
                                             'CHECKPOINT_MODE': mode})[0 if mode == 'sync' else 1]
        directory = self.results / stage['name']
        (directory / 'train').mkdir(parents=True)
        run = dict(plan=stage, status='finished', exit_status=0, checks=[])
        (directory / 'run.json').write_text(json.dumps(run))
        workload = json.loads((SPAIN / 'sdpa' / 'train' / 'summary.json').read_text())['workload']
        (directory / 'train' / 'summary.json').write_text(json.dumps(dict(
            measured_seconds=80.0, measured_useful_tokens=38 * targets, segment_seconds=88.25,
            completed_updates=40, workload=dict(workload, updates=40, warmup_updates=2),
            retained_progress={'retained_progress_tokens': 40 * targets, 'elapsed_seconds': 88.25,
                               'retained_tokens_per_second': goodput * SDPA_RATE})))
        return stage, directory, run

    def interrupt_stage(self, goodput):
        stage = workshop.plan('interrupt', {'LLM_TRANSPORT': 'efa', 'ATTENTION': 'sdpa', 'CHECKPOINT_MODE': 'process'})[0]
        directory = self.results / 'interrupt' / 'campaign'
        directory.mkdir(parents=True)
        (directory.parent / 'run.json').write_text(json.dumps(dict(plan=stage, status='finished', exit_status=0)))
        workload = json.loads((SPAIN / 'sdpa' / 'train' / 'summary.json').read_text())['workload']
        (directory / 'campaign.json').write_text(json.dumps(dict(  # synthetic
            metric='Retained progress', retained_progress_tokens=40 * 65504, elapsed_seconds=190.0,
            retained_tokens_per_second=goodput * SDPA_RATE, completed_updates=40,
            workload=dict(workload, updates=40, warmup_updates=2))))

    def test_saving_bars_need_stage_sdpa_and_interrupt_has_no_row_before_it_runs(self):
        lines = workshop.goodput_chart_lines(self.results)
        self.assertEqual(lines, [GOODPUT_TITLE,
                                 '  ckpt-sync, synchronous saves      not run yet',
                                 '  ckpt-process, process-mode saves  not run yet',
                                 '                                    0' + ' ' * 48 + '1',
                                 '                                    one cell = 0.02',
                                 GOODPUT_NOTE])
        self.saving_stage('sync', 0.25)  # synthetic
        self.assertEqual(workshop.goodput_chart_lines(self.results)[1],
                         '  ckpt-sync, synchronous saves      no completed run of stage sdpa')

    def test_goodput_on_an_axis_from_0_to_1_with_no_throughput_bar(self):
        self.add('sdpa')
        self.saving_stage('sync', 0.25)  # synthetic goodput values
        self.saving_stage('process', 0.5)
        self.interrupt_stage(0.15)
        lines = workshop.goodput_chart_lines(self.results)
        self.assertEqual(lines[0], GOODPUT_TITLE)
        self.assertEqual([line.split('  ')[1] for line in lines[1:4]],
                         ['ckpt-sync, synchronous saves', 'ckpt-process, process-mode saves', 'interrupt, stop and resume'])
        # 0.02 per cell: 1 is 50 cells, so 0.25, 0.5 and 0.15 are 12.5, 25 and 7.5 cells.
        self.assertEqual([bar(line) for line in lines[1:4]], [('█' * 12 + '▌', '0.25'), ('█' * 25, '0.50'),
                                                              ('█' * 7 + '▌', '0.15')])
        self.assertEqual(len(columns(lines[1:4])), 1)
        start = BAR.search(lines[1]).start()
        self.assertEqual((lines[4].index('0'), lines[4].index('1')), (start, start + 49))
        self.assertEqual(lines[5:], ['                                    one cell = 0.02', GOODPUT_NOTE])
        self.assertFalse([line for line in lines if line.startswith('  sdpa') or 'GPU-hour' in line])
        self.assert_narrow(lines)

    def test_a_saving_run_of_other_work_per_update_has_no_bar(self):
        self.add('sdpa')
        self.saving_stage('sync', 0.25, targets=32752)  # synthetic
        self.assertEqual(workshop.goodput_chart_lines(self.results)[1],
                         '  ckpt-sync, synchronous saves      different work per update: targets_per_update')

    def test_result_block_of_a_saving_stage_is_followed_by_the_goodput_chart(self):
        self.add('sdpa')
        stage, directory, run = self.saving_stage('sync', 0.4)  # synthetic
        lines = workshop.result_lines(stage, directory, run, self.results, 16)
        chart = lines.index(GOODPUT_TITLE)
        self.assertEqual(lines[0], 'ckpt-sync (efa, sdpa attention, sync saves after updates 5, 20, 40)')
        self.assertEqual(lines[2:9], [
            '  retained tokens              2,620,160 (updates 1-40, durable at the end)',
            '  interval                     88.25 s, from the end of setup (model loading, FSDP wrapping,',
            '                               optimizer, checkpoint manager) to the end of the final flush and',
            '                               the closing synchronization across ranks',
            '  retained progress rate       30,910 tokens/s (retained tokens / interval)',
            '  fault-free rate              77,275 tokens/s (stage sdpa, which does not save)',
            '  training goodput             0.40 (retained progress rate / fault-free rate)'])
        self.assertEqual(lines[chart - 1], '')
        self.assertEqual([bar(lines[chart + 1])[1], lines[chart + 2]],
                         ['0.40', '  ckpt-process, process-mode saves  not run yet'])
        self.assertEqual(lines[chart + 4].strip(), 'one cell = 0.02')
        self.assert_narrow(lines[chart:])


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
        # 0.1 s per cell, as in Modules 2 to 4, so the EFA bar keeps its length from Module 4.
        self.assertEqual([bar(update[0]), bar(update[2])], [('█' * 3 + '▒' * 11, '1.44 s'),
                                                            ('█' * 3 + '▒' * 5, '0.85 s')])
        self.assertEqual([update[1].strip(), update[3].strip(), update[4].strip()], [
            '█ arithmetic at your GEMM peak 0.28 s   ▒ the rest 1.16 s',
            '█ arithmetic at your GEMM peak 0.28 s   ▒ the rest 0.57 s', 'one cell = 0.1 s'])
        memory = lines[titles[2] + 1:titles[2] + 4]
        # 0.5 GiB per cell: 53 3/8 and 16 5/8 cells.
        self.assertEqual([bar(line) for line in memory[:2]], [('█' * 53 + '▍', '26.68 GiB'), ('█' * 16 + '▋', '8.34 GiB')])
        self.assertEqual(memory[2].strip(), 'one cell = 0.5 GiB')
        self.assertEqual(bar(lines[titles[3] + 3])[1], '3.89x  (SDPA +1.60x)')
        # The charts fit in 100 columns; the block's MFU line is longer, as its EFA bytes line can be.
        self.assert_narrow(lines[titles[1]:])
        # MFU against the dense BF16 peak, and the share of your measured GEMM peak (fixture: 77,275 tokens/s).
        self.assertIn('  MFU                          27.3 %  (11.48 GFLOP per token, dense BF16 peak 202.8 TFLOP/s '
                      'per GPU, estimated)', lines)
        self.assertIn('  share of your GEMM peak      33.0 %  (your GEMM peak 167.9 TFLOP/s per GPU)', lines)

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
            self.assertIn(GOODPUT_TITLE, lines[chart:])
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
        self.assertEqual([bar(lines[1]), bar(lines[3])], [('█' * 7 + '░' * 6 + '▓' * 22, '3.519 s'),
                                                          ('█' * 8 + '░' * 6 + '▓', '1.451 s')])
        self.assertEqual([lines[1][:24], lines[3][:24]], ['  profile-socket-eager  ', '  profile-efa-eager     '])
        self.assertEqual(lines[2:3] + lines[4:], [
            '                        █ 0.751  ░ 0.565  ▓ 2.193  ▒ 0.010 s   input wait (CPU) 0.001 s',
            '                        █ 0.770  ░ 0.603  ▓ 0.067  ▒ 0.011 s   input wait (CPU) 0.001 s',
            '  profile-efa-sdpa      not run yet',
            '                        █ attention core kernels   ░ other compute kernels',
            '                        ▓ communication not overlapped   ▒ GPU idle   one cell = 0.1 s',
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
        self.assertEqual(len(lines), 16)
        self.assertEqual([line[:31] for line in lines[3:5]], ['    2 MiB  NCCL_PROTO not set  ',
                                                              '           NCCL_PROTO=Simple   '])
        # One scale, 1 GB/s per cell: 0.26 GB/s is two eighths of a cell.
        self.assertEqual(bar(lines[3]), ('▎', '0.26  RING/LL'))
        self.assertEqual(bar(lines[8])[1], '13.94  TREE/SIMPLE')
        self.assertEqual(bar(lines[14]), ('█' * 37 + '▊', '37.80  RING/SIMPLE'))
        self.assertEqual(lines[15].strip(), 'one cell = 1 GB/s')
        self.assertEqual(len({BAR.search(line).start() for line in lines[1:15]}), 1)
        self.assertTrue(all(len(line) <= charts.WIDTH for line in lines))


if __name__ == '__main__':
    unittest.main()
