"""CPU tests of the workshop tools (tools/ and lib/workshop.py). Standard library only.

Recorded values come from job 487 (two g7.48xlarge nodes, 16 GPUs, 2,048
positions, eager attention, Socket and EFA). Values marked synthetic are software fixtures, not measurements.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'lib'))
import workshop  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / 'fixtures'
LOSSES = [2.157260686159134, 2.170563116669655, 2.138267382979393, 2.0402668192982674, 2.1076911836862564]
BASELINE = dict(seconds=13.514139526057988, rate=19388.285839048818, peak=28648466432,
                monotonic=[3477353.777025936, 3477357.151745839, 3477360.544529955, 3477363.923061297,
                           3477367.292415476])
EFA = dict(seconds=5.770125308074057, rate=45409.06583664043, peak=28648466432,
           monotonic=[3477404.172973511, 3477405.620809354, 3477407.060770044, 3477408.501891915,
                      3477409.944532604])
ALLOCATION = {'label': 'pcs-g7.48xlarge-2nodes-16gpu', 'gpu_count': 16, 'dp_size': 16}
G7_GPU = 'NVIDIA RTX PRO 4500 Blackwell Server Edition'
HOSTS = ('gpu-node-1', 'gpu-node-2')


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), LAB / 'tools' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows(monotonic, losses, **fields):
    return [dict(update=index + 1, monotonic_seconds=time, global_mean_loss=loss, checkpoint_upload_pending=False,
                 durable_update=0, global_useful_tokens=65504, **fields)
            for index, (time, loss) in enumerate(zip(monotonic, losses))]


def write_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in values))


def summary(stage, seconds, rate, peak, *, tokens=None, **extra):
    workload = dict(sequence_length=stage['sequence_length'], updates=stage['updates'], data=stage['data'],
                    warmup_updates=stage['warmup_updates'], global_batch_samples=stage['global_batch'])
    if stage['saving'] != 'none':
        workload['checkpoint_updates'] = stage['checkpoint_updates']
    result = dict(status='completed', profiled=False, numerical_verification=False, workload=workload,
                  allocation=ALLOCATION, configuration={'attention': stage['attention'], 'deterministic': False},
                  measurement_scope=('training/save/final durable completion' if stage['saving'] != 'none'
                                     else 'continuous post-warmup training'),
                  measured_seconds=seconds, measured_useful_tokens=tokens or workshop.timed_targets(stage),
                  training_throughput_tokens_per_second=rate, segment_seconds=seconds + 9,
                  completed_updates=stage['updates'],
                  ranks=[{'peak_cuda_allocated_bytes': peak, 'gpu_name': G7_GPU}] * 16)
    result.update(extra)
    return result


def nccl_log(transport):
    return (FIXTURES / f'nccl-{transport}-2048.txt').read_text()


def efa_counters(sent, received):
    return {host: {'rdmap0s0': {'tx_bytes': sent, 'rx_bytes': received},
                   'rdmap1s0': {'tx_bytes': sent, 'rx_bytes': received}} for host in HOSTS}


def finished_stage(results, stage, *, exit_status=0, checks=()):
    directory = Path(results) / stage['name']
    (directory / 'train').mkdir(parents=True, exist_ok=True)
    run = dict(stage=stage['name'], plan=stage, allocation_id=42, start_utc=100.0, end_utc=150.0,
               status='finished', exit_status=exit_status, checks=list(checks), failure=None)
    (directory / 'run.json').write_text(json.dumps(run))
    return directory


class StageTableTests(unittest.TestCase):
    def test_every_stage_has_its_config_and_a_launch_command(self):
        for name in workshop.STAGES:
            with self.subTest(stage=name):
                stage = workshop.stage_plan(name)
                self.assertTrue((LAB / stage['config']).is_file())
                self.assertIn(workshop.launch_command(stage, 'label', '/results')[0], ('bash', 'python3'))
        self.assertTrue(workshop.stage_plan('try-microbatch2')['with_change'])
        self.assertFalse(workshop.stage_plan('rerun-microbatch2')['with_change'])
        with self.assertRaisesRegex(workshop.UsageError, 'unknown stage'):
            workshop.stage_plan('probe')


class StageRunTests(unittest.TestCase):
    """run_stage, the body of the batch job, with a fake launcher."""

    def run_stage(self, name, results, launch):
        output = []

        def popen(command, cwd, env, stdout, stderr):
            return SimpleNamespace(poll=lambda: 0, returncode=launch(command, env, stdout))

        with patch.dict(os.environ, SLURM_JOB_ID='42', GPUS_PER_NODE='8'):
            code = workshop.run_stage(workshop.stage_plan(name), results, popen=popen, out=output.append,
                                      efa_reader=lambda log: efa_counters(1, 1))
        return code, output

    def test_baseline_prints_its_rows_and_result(self):
        stage = workshop.stage_plan('baseline')
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            (results / 'baseline').mkdir()
            (results / 'baseline/old.txt').write_text('earlier attempt')

            def launch(command, env, log):
                self.assertEqual((command[:2], env['LLM_TRANSPORT']), (['bash', '2.run-llm.sh'], 'socket'))
                train = results / 'baseline' / 'train'
                train.mkdir()
                log.write(nccl_log('socket'))
                log.flush()
                for rank in range(16):
                    write_rows(train / f'rank-{rank}-updates.jsonl', rows(BASELINE['monotonic'], LOSSES))
                (train / 'summary.json').write_text(json.dumps(summary(stage, BASELINE['seconds'], BASELINE['rate'],
                                                                       BASELINE['peak'])))
                return 0

            code, output = self.run_stage('baseline', results, launch)
            self.assertEqual(code, 0)
            self.assertEqual(len(list((results / 'attempts').glob('baseline-*/old.txt'))), 1)
            run = json.loads((results / 'baseline/run.json').read_text())
            self.assertEqual((run['status'], run['exit_status'], run['launcher_exit']), ('finished', 0, 0))
            self.assertEqual(output[:3], [
                '[aim347] baseline: LLM_TRANSPORT=socket ATTENTION=eager CHECKPOINT_MODE=none',
                '[aim347] 5 updates of 32 sequences x 2,048 positions; updates 2-5 are timed (262,016 targets)',
                f'[aim347] batch job 42, results {results / "baseline"}'])
            self.assertEqual(output[3:9], ['update   seconds      loss', '     1    warmup  2.157261',
                                           '     2     3.375  2.170563', '     3     3.393  2.138267',
                                           '     4     3.379  2.040267', '     5     3.369  2.107691'])
            self.assertEqual(output[10], 'baseline (socket, eager attention, no saving)')

    def test_failed_launch_prints_the_log_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            def launch(command, env, log):
                log.write('\n'.join(f'line {n}' for n in range(60)) + '\n')
                log.flush()
                return 1

            code, output = self.run_stage('efa', Path(directory), launch)
            self.assertEqual(code, 4)
            self.assertIn('efa failed: the launcher exited with status 1', output)
            self.assertEqual(output[-1], 'line 59')


class ProgressionTests(unittest.TestCase):
    def test_a_stage_whose_check_failed_gets_no_ratio(self):
        tool = load_tool('progression')
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, LLM_RESULTS_DIR=directory):
            results = Path(directory)
            for name, exit_status, measured in (('baseline', 0, BASELINE), ('efa', 5, EFA)):
                stage = workshop.stage_plan(name)
                record = summary(stage, measured['seconds'], measured['rate'], 2**30)
                (finished_stage(results, stage, exit_status=exit_status) / 'train/summary.json').write_text(json.dumps(record))
            notes = tool.Notes()
            _, table, _ = tool.collect(results, notes)
            self.assertEqual(table[0]['vs baseline'], '1.00x')
            for column in ('vs baseline', 'vs previous step'):
                self.assertRegex(table[1][column], r'^\(\d\)$')
            self.assertIn('efa vs baseline: not compared, a check of efa failed', notes.texts)


class SyntaxTests(unittest.TestCase):
    def test_every_shell_script_parses(self):
        for path in sorted(LAB.glob('**/*.sh')):
            with self.subTest(script=path.name):
                self.assertEqual(subprocess.run(['bash', '-n', str(path)]).returncode, 0)


if __name__ == '__main__':
    unittest.main()
