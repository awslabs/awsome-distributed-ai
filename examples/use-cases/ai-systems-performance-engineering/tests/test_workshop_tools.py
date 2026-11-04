"""CPU tests of the workshop tools (tools/ and lib/workshop.py). Standard library only.

Recorded values come from job 487 (two g7.48xlarge nodes, 16 GPUs, 2,048
positions, eager attention, Socket and EFA) and the GEMM peak recorded in
VALIDATION.md. Values marked synthetic are software fixtures, not measurements.
"""
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'lib'))
import workshop  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / 'fixtures'
GEMM_PEAK = 171.702538
LOSSES = [2.157260686159134, 2.170563116669655, 2.138267382979393, 2.0402668192982674, 2.1076911836862564]
BASELINE = dict(seconds=13.514139526057988, rate=19388.285839048818, peak=28648466432,
                monotonic=[3477353.777025936, 3477357.151745839, 3477360.544529955, 3477363.923061297,
                           3477367.292415476])
EFA = dict(seconds=5.770125308074057, rate=45409.06583664043, peak=28648466432,
           monotonic=[3477404.172973511, 3477405.620809354, 3477407.060770044, 3477408.501891915,
                      3477409.944532604])
ALLOCATION = {'label': 'pcs-g7.48xlarge-2nodes-16gpu', 'gpu_count': 16, 'dp_size': 16}
HOSTS = ('gpu-node-1', 'gpu-node-2')


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), LAB / 'tools' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def job(transport='socket', attention='eager', saving='none'):
    return {'LLM_TRANSPORT': transport, 'ATTENTION': attention, 'CHECKPOINT_MODE': saving}


def rows(monotonic, losses, **fields):
    return [dict(update=index + 1, monotonic_seconds=time, global_mean_loss=loss, checkpoint_upload_pending=False,
                 durable_update=0, global_useful_tokens=65504, **fields)
            for index, (time, loss) in enumerate(zip(monotonic, losses))]


def write_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in values))


def summary(stage, seconds, rate, peak, *, tokens=None, **extra):
    workload = dict(sequence_length=stage['sequence_length'], updates=stage['updates'], data=stage['data'])
    if stage['saving'] != 'none':
        workload['checkpoint_updates'] = stage['checkpoint_updates']
    result = dict(status='completed', profiled=False, numerical_verification=False, workload=workload,
                  allocation=ALLOCATION, configuration={'attention': stage['attention'], 'deterministic': False},
                  measurement_scope=('training/save/final durable completion' if stage['saving'] != 'none'
                                     else 'continuous post-warmup training'),
                  measured_seconds=seconds, measured_useful_tokens=tokens or workshop.timed_targets(stage),
                  training_throughput_tokens_per_second=rate, segment_seconds=seconds + 9,
                  ranks=[{'peak_cuda_allocated_bytes': peak}] * 16)
    result.update(extra)
    return result


def nccl_log(transport):
    return (FIXTURES / f'nccl-{transport}-2048.txt').read_text()


def with_nccl_proto(log, value='Simple'):
    """Synthetic: the recorded log plus one ENV line per rank, as NCCL_DEBUG_SUBSYS=ENV prints it."""
    ranks = [line.split(' NCCL INFO')[0] for line in log.splitlines() if 'Using network' in line]
    extra = [f'{rank} NCCL INFO NCCL_PROTO set by environment to {value}' for rank in ranks]
    # A rank with a second communicator prints the line again; it still counts once.
    return log + '\n'.join(extra + extra[:1]) + '\n'


def efa_counters(sent, received):
    return {host: {'rdmap0s0': {'tx_bytes': sent, 'rx_bytes': received},
                   'rdmap1s0': {'tx_bytes': sent, 'rx_bytes': received}} for host in HOSTS}


def finished_stage(results, stage, *, exit_status=0, checks=(), invocation='20261103T000000Z'):
    directory = Path(results) / stage['name']
    (directory / 'train').mkdir(parents=True, exist_ok=True)
    run = dict(stage=stage['name'], subcommand=stage['subcommand'], invocation=invocation, stages=[stage['name']],
               plan=stage, job_env=stage['job_env'], allocation_id=42, start_utc=100.0, end_utc=150.0,
               status='finished', exit_status=exit_status, checks=list(checks), failure=None)
    (directory / 'run.json').write_text(json.dumps(run))
    return directory


class JobEnvTests(unittest.TestCase):
    def test_committed_job_env_has_the_three_starting_values(self):
        self.assertEqual(workshop.read_job_env(LAB / 'job.env'), job())

    def test_parser_rejects_unknown_bad_and_duplicate_settings_with_line_numbers(self):
        cases = {'# comment\n\nLLM_TRANSPORT=socket\nATTENTION=eager\nCHECKPOINT_MODE=none\nFOO=bar\n': 'line 6',
                 'LLM_TRANSPORT=socket\nATTENTION=flash\nCHECKPOINT_MODE=none\n': 'line 2: ATTENTION must be one of',
                 'LLM_TRANSPORT=socket\nATTENTION=eager\nLLM_TRANSPORT=efa\nCHECKPOINT_MODE=none\n':
                     'line 3: LLM_TRANSPORT is set a second time',
                 'LLM_TRANSPORT=socket\nATTENTION=eager\n': 'missing CHECKPOINT_MODE',
                 'export LLM_TRANSPORT=socket\nATTENTION=eager\nCHECKPOINT_MODE=none\n': 'line 1'}
        for text, message in cases.items():
            with self.subTest(text=text), self.assertRaisesRegex(workshop.UsageError, message):
                workshop.parse_job_env(text)

    def test_configs_change_only_the_listed_fields(self):
        base = json.loads((LAB / 'configs/llm-zero-smol.json').read_text())
        expected = {'smol-2048': (2048, 5, 1, [5], 'AIM347 workshop job'),
                    'smol-1024': (1024, 5, 1, [5], 'AIM347 half-length probe'),
                    'smol-2048-40': (2048, 40, 2, [5, 20, 40], 'AIM347 saving and interruption')}
        for name, (length, updates, warmup, saves, qualification) in expected.items():
            config = json.loads((LAB / f'configs/{name}.json').read_text())
            self.assertEqual(config, dict(base, sequence_length=length, updates=updates, warmup_updates=warmup,
                                          checkpoint_updates=saves, qualification=qualification))


class PlanTests(unittest.TestCase):
    def test_every_subcommand_plans_its_stages_arguments_config_and_data(self):
        cases = [
            ('baseline', job(), ['baseline'], 'configs/smol-2048.json', 'tokens-2048-160'),
            ('probe', job('efa', 'sdpa'), ['probe-efa-sdpa'], 'configs/smol-1024.json', 'tokens-1024-160'),
            ('profile', job(), ['profile-socket-eager'], 'configs/smol-2048.json', 'tokens-2048-160'),
            ('sdpa-socket', job(), ['sdpa-socket'], 'configs/smol-2048.json', 'tokens-2048-160'),
            ('efa', job('efa'), ['efa'], 'configs/smol-2048.json', 'tokens-2048-160'),
            ('sdpa', job('efa', 'sdpa'), ['sdpa'], 'configs/smol-2048.json', 'tokens-2048-160'),
            ('checkpoint', job('efa', 'sdpa', 'sync'), ['ckpt-sync', 'ckpt-process'], 'configs/smol-2048-40.json',
             'tokens-2048-1280'),
            ('restore', job('efa', 'sdpa', 'process'), ['restore'], 'configs/smol-2048-40.json', 'tokens-2048-1280'),
            ('interrupt', job('efa', 'sdpa', 'sync'), ['interrupt'], 'configs/smol-2048-40.json', 'tokens-2048-1280'),
        ]
        for subcommand, settings, names, config, data in cases:
            with self.subTest(subcommand=subcommand):
                stages = workshop.plan(subcommand, settings)
                self.assertEqual([stage['name'] for stage in stages], names)
                for stage in stages:
                    self.assertEqual((stage['config'], stage['data']), (config, data))
                    arguments = workshop.trainer_arguments(stage, 'label')
                    self.assertEqual(arguments[:16], [
                        '--config', config, '--data', f'/data/{data}', '--model-path', '/data/model',
                        '--attention-implementation', stage['attention'], '--microbatch', '1', '--workers', '2',
                        '--checkpoint-mode', stage['saving'], '--allocation-label', 'label'])
                    self.assertEqual(arguments[-2:], ['--output', f'/results/{stage["name"]}/train'])
        sdpa_socket = workshop.plan('sdpa-socket', job())[0]
        self.assertEqual((sdpa_socket['transport'], sdpa_socket['attention']), ('socket', 'sdpa'))
        self.assertEqual([s['saving'] for s in workshop.plan('checkpoint', job('efa', 'sdpa', 'process'))],
                         ['sync', 'process'])
        self.assertIn('--profile', workshop.trainer_arguments(workshop.plan('profile', job())[0], 'label'))
        restore = workshop.plan('restore', job('efa', 'sdpa', 'process'))[0]
        self.assertEqual(restore['saving'], 'sync')
        arguments = workshop.trainer_arguments(restore, 'label')
        self.assertIn('--resume /results/ckpt-sync/train/checkpoints/update-000020 --interrupt-after-update 25',
                      ' '.join(arguments))
        interrupt = workshop.plan('interrupt', job('efa', 'sdpa', 'sync'))[0]
        command = workshop.launch_command(interrupt, 'label', '/fsx/results', 16)
        self.assertEqual(command[:11], ['python3', '13.measure-recovery.py', '--output', '/fsx/results/interrupt/campaign',
                                        '--allocated-gpus', '16', '--interrupt-after-update', '30', '--', 'bash',
                                        '2.run-llm.sh'])
        self.assertIn('--checkpoint-mode process', ' '.join(command))
        self.assertNotIn('--output', command[9:])

    def test_wrong_job_env_state_names_the_sed_line(self):
        cases = [
            ('efa', job(), "efa expects LLM_TRANSPORT=efa in job.env (it says socket). Change it with: "
                           "sed -i 's/^LLM_TRANSPORT=socket$/LLM_TRANSPORT=efa/' job.env"),
            ('baseline', job('efa'), "sed -i 's/^LLM_TRANSPORT=efa$/LLM_TRANSPORT=socket/' job.env"),
            ('sdpa', job('efa'), "sed -i 's/^ATTENTION=eager$/ATTENTION=sdpa/' job.env"),
            ('checkpoint', job('efa', 'sdpa'), "sed -i 's/^CHECKPOINT_MODE=none$/CHECKPOINT_MODE=sync/' job.env"),
            ('probe', job('efa', 'sdpa', 'sync'), "sed -i 's/^CHECKPOINT_MODE=sync$/CHECKPOINT_MODE=none/' job.env"),
            ('restore', job(), "sed -i 's/^LLM_TRANSPORT=socket$/LLM_TRANSPORT=efa/' job.env"),
            ('interrupt', job('efa', 'eager'), "sed -i 's/^ATTENTION=eager$/ATTENTION=sdpa/' job.env"),
            ('sdpa-socket', job('socket', 'sdpa'), "sed -i 's/^ATTENTION=sdpa$/ATTENTION=eager/' job.env"),
        ]
        for subcommand, settings, message in cases:
            with self.subTest(subcommand=subcommand):
                with self.assertRaises(workshop.UsageError) as raised:
                    workshop.plan(subcommand, settings)
                self.assertIn(message, str(raised.exception))


class MetricTests(unittest.TestCase):
    def test_job_487_fixtures(self):
        self.assertEqual(workshop.flop_per_token(2048), 11_476_217_856)
        self.assertEqual(workshop.flop_per_token(1024), 10_872_238_080)
        self.assertAlmostEqual(workshop.gpu_seconds_per_million_tokens(BASELINE['seconds'], 262016, 16), 825.2, 1)
        self.assertAlmostEqual(workshop.gpu_seconds_per_million_tokens(EFA['seconds'], 262016, 16), 352.4, 1)
        self.assertEqual(round(100 * workshop.mfu(BASELINE['rate'], 2048, GEMM_PEAK, 16), 1), 8.1)
        self.assertEqual(round(100 * workshop.mfu(EFA['rate'], 2048, GEMM_PEAK, 16), 1), 19.0)
        self.assertEqual(round(workshop.compute_only_seconds(65504, 2048, 171.7, 16), 3), 0.274)
        self.assertEqual(workshop.timed_targets(workshop.plan('baseline', job())[0]), 262016)
        self.assertEqual(workshop.timed_targets(workshop.plan('probe', job())[0]), 130944)
        self.assertEqual(workshop.timed_targets(workshop.plan('checkpoint', job('efa', 'sdpa', 'sync'))[0]), 2489152)


class DisplayTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.results = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_five_update_rows_show_warmup_and_intervals(self):
        printer = workshop.RowPrinter(1)
        lines = [printer.header()] + [printer.line(row) for row in rows(BASELINE['monotonic'], LOSSES)]
        self.assertEqual(lines, ['update   seconds      loss',
                                 '     1    warmup  2.157261',
                                 '     2     3.375  2.170563',
                                 '     3     3.393  2.138267',
                                 '     4     3.379  2.040267',
                                 '     5     3.369  2.107691'])

    def test_baseline_result_block(self):
        stage = workshop.plan('baseline', job())[0]
        directory = finished_stage(self.results, stage)
        (directory / 'launch.log').write_text(nccl_log('socket'))
        (directory / 'efa-before.json').write_text(json.dumps(efa_counters(1000, 2000)))
        (directory / 'efa-after.json').write_text(json.dumps(efa_counters(1000, 2000)))
        (self.results / 'ceilings').mkdir()
        (self.results / 'ceilings/ceilings.json').write_text(json.dumps(
            {'gemm_tflops_per_gpu': GEMM_PEAK, 'memory_read_bytes_per_second_per_gpu': 662e9,
             'ridge_flop_per_byte': GEMM_PEAK * 1e12 / 662e9}))
        result = summary(stage, BASELINE['seconds'], BASELINE['rate'], BASELINE['peak'])
        lines = workshop.timed_block(stage, directory, result, workshop.read_ceilings(self.results), 16)
        self.assertEqual(lines, [
            'baseline (socket, eager attention, no saving)',
            '  timed updates 2-5            13.51 s  (262,016 targets)',
            '  tokens/s                     19,388',
            '  GPU-seconds per 1M tokens    825',
            '  MFU vs your GEMM peak        8.1 %    (11.48 GFLOP per token, 171.7 TFLOP/s per GPU)',
            '  peak memory (largest rank)   26.68 GiB',
            '  NCCL network                 Socket on 16 of 16 ranks',
            '  EFA bytes during the run     gpu-node-1 +0 B sent, +0 B received; gpu-node-2 +0 B sent, +0 B received'])
        (self.results / 'ceilings/ceilings.json').unlink()
        lines = workshop.timed_block(stage, directory, result, None, 16)
        self.assertIn('  MFU vs your GEMM peak        n/a (run bash tools/hold-and-measure.sh)', lines)

    def test_efa_and_sdpa_blocks(self):
        efa = workshop.plan('efa', job('efa'))[0]
        directory = finished_stage(self.results, efa)
        (directory / 'launch.log').write_text(nccl_log('efa'))
        write_rows(directory / 'train/rank-0-updates.jsonl', rows(EFA['monotonic'], LOSSES))
        lines = workshop.timed_block(efa, directory, summary(efa, EFA['seconds'], EFA['rate'], EFA['peak']), None, 16)
        self.assertIn('  NCCL network                 AWS Libfabric on 16 of 16 ranks', lines)
        self.assertIn('  provider                     efa, fabric efa-direct (found 2 nics)', lines)
        self.assertIn('  protocol                     RDMA (user set)', lines)
        self.assertIn('  NCCL_PROTO                   not set on any rank', lines)
        self.assertIn('  channels sending from GPU memory (/GDRDMA)  0 of 4 network channel lines', lines)
        (directory / 'launch.log').write_text(with_nccl_proto(nccl_log('efa')))
        lines = workshop.timed_block(efa, directory, summary(efa, EFA['seconds'], EFA['rate'], EFA['peak']), None, 16)
        self.assertEqual(lines[-3:-1], ['  NCCL_PROTO                   Simple on 16 of 16 ranks',
                                        '  channels sending from GPU memory (/GDRDMA)  0 of 4 network channel lines'])
        self.assertIn('  GPU-seconds per 1M tokens    352', lines)
        sdpa = workshop.plan('sdpa', job('efa', 'sdpa'))[0]
        directory = finished_stage(self.results, sdpa)
        (directory / 'launch.log').write_text(nccl_log('efa'))
        synthetic = [loss + 1e-6 for loss in LOSSES]
        write_rows(directory / 'train/rank-0-updates.jsonl', rows(EFA['monotonic'], synthetic))
        run = json.loads((directory / 'run.json').read_text())
        result = summary(sdpa, 4.0, 65504.0, 9 * 2**30)  # synthetic
        (directory / 'train/summary.json').write_text(json.dumps(result))
        lines = workshop.result_lines(sdpa, directory, run, self.results, 16)
        self.assertIn('  losses 1-5 (this run, sdpa)    2.157262 2.170564 2.138268 2.040268 2.107692', lines)
        self.assertIn('  losses 1-5 (stage efa, eager)  2.157261 2.170563 2.138267 2.040267 2.107691', lines)
        self.assertIn('  difference                     +0.000001 +0.000001 +0.000001 +0.000001 +0.000001', lines)

    def checkpoint_rows(self, durable_at_40):
        """Synthetic 40-update rows with saves after 5, 20 and 40."""
        values = []
        for update in range(1, 41):
            pending = (5 <= update < 9) or (20 <= update < 34) or update == 40
            durable = 0 if update < 9 else (5 if update < 34 else 20)
            if update == 40 and durable_at_40:
                pending, durable = False, 40
            values.append(dict(update=update, monotonic_seconds=100.0 + 1.5 * update, global_mean_loss=2.0,
                               checkpoint_upload_pending=pending, durable_update=durable))
        return values

    def test_process_rows_show_save_pending_durable_and_positive_flush(self):
        values = self.checkpoint_rows(durable_at_40=False)
        printer = workshop.RowPrinter(2, [5, 20, 40])
        self.assertEqual(printer.header(), 'update   seconds      loss  save  pending  durable')
        lines = [printer.line(row) for row in values]
        self.assertEqual(lines[0], '     1    warmup  2.000000             no        0')
        self.assertEqual(lines[19], '    20     1.500  2.000000   yes      yes        5')
        self.assertEqual(lines[39], '    40     1.500  2.000000   yes      yes       20')
        events = [dict(phase='save_start', update=40, monotonic_seconds=160.1),
                  dict(phase='globally_complete', update=20, monotonic_seconds=151.0),
                  dict(phase='globally_complete', update=40, monotonic_seconds=172.5)]
        flush = workshop.flush_seconds(values, events, 40)
        self.assertAlmostEqual(flush, 12.5)
        self.assertEqual(workshop.RowPrinter.flush_line(flush, 40), ' flush    12.500         -     -        -       40')

    def test_sync_rows_flush_zero(self):
        values = self.checkpoint_rows(durable_at_40=True)
        events = [dict(phase='globally_complete', update=40, monotonic_seconds=150.0)]
        self.assertEqual(workshop.flush_seconds(values, events, 40), 0.0)
        self.assertEqual(workshop.RowPrinter.flush_line(0.0, 40), ' flush     0.000         -     -        -       40')

    def test_checkpoint_block_prints_goodput_inputs_only(self):
        stage = workshop.plan('checkpoint', job('efa', 'sdpa', 'sync'))[0]
        directory = finished_stage(self.results, stage)
        (directory / 'launch.log').write_text(nccl_log('efa'))
        result = summary(stage, 80.0, 31000.0, 9 * 2**30, segment_seconds=88.25,  # synthetic
                         training_goodput={'retained_progress_tokens': 2620160,
                                           'training_goodput_tokens_per_second': 29690.0,
                                           'tokens_per_allocated_gpu_hour': 6.68e6})
        lines = workshop.timed_block(stage, directory, result, None, 16)
        self.assertEqual(lines[:5], [
            'ckpt-sync (efa, sdpa attention, sync saves after updates 5, 20, 40)',
            '  timed updates 3-40 with saves and final flush  80.00 s  (2,489,152 targets)',
            '  durable at the end           update 40, 2,620,160 tokens',
            '  goodput interval             88.25 s  (from the end of model loading to the end of the final flush)',
            '  GPUs                         16'])
        self.assertFalse([line for line in lines if 'goodput ' in line and 'interval' not in line])
        self.assertFalse([line for line in lines if 'per GPU-hour' in line or '29,690' in line])

    def interrupt_dir(self, pending, durable):
        stage = workshop.plan('interrupt', job('efa', 'sdpa', 'process'))[0]
        directory = finished_stage(self.results, stage)
        campaign = directory / 'campaign'
        values = [dict(update=u, monotonic_seconds=10.0 + u, global_mean_loss=2.0, checkpoint_upload_pending=False,
                       durable_update=5 if u >= 9 else 0) for u in range(1, 31)]
        values[-1].update(checkpoint_upload_pending=pending, durable_update=durable, controlled_interruption=True)
        write_rows(campaign / 'interrupted/rank-0-updates.jsonl', values)
        restored = durable
        write_rows(campaign / 'resumed/rank-0-updates.jsonl',
                   [dict(update=u, monotonic_seconds=60.0 + u, global_mean_loss=2.0, checkpoint_upload_pending=False,
                         durable_update=restored) for u in range(restored + 1, 41)])
        for update in (5, 20) if durable == 20 else (5,):
            (campaign / f'interrupted/checkpoints/update-{update:06d}').mkdir(parents=True)
            (campaign / f'interrupted/checkpoints/update-{update:06d}/COMPLETED.json').write_text('{}')
        (campaign / 'interrupted.log').write_text(nccl_log('efa'))
        (campaign / 'resumed.log').write_text(nccl_log('efa'))
        (campaign / 'campaign.json').write_text(json.dumps(dict(  # synthetic
            interrupted_update=30, restored_update=restored, replayed_update_count=30 - restored,
            elapsed_seconds=190.0, training_goodput_tokens_per_second=13790.3,
            tokens_per_allocated_gpu_hour=3.1e6)))
        return stage, directory

    def test_interrupt_block_names_the_upload_state(self):
        stage, directory = self.interrupt_dir(pending=True, durable=5)
        lines = workshop.interrupt_block(stage, directory, 16)
        self.assertEqual(len([line for line in lines if line.startswith(('    26', '    27', '    28', '    29', '    30'))]), 5)
        self.assertIn('  stopped after update 30; the update-20 upload was still pending (durable update 5)', lines)
        self.assertIn('  completed checkpoints at the stop: update-000005', lines)
        self.assertIn('  resumed from update 5; first update after the resume: 6', lines)
        self.assertIn('  updates run twice: 6-30 (25 updates)', lines)
        self.assertIn('  NCCL network (interrupted run)  AWS Libfabric on 16 of 16 ranks', lines)
        self.assertIn('  NCCL_PROTO                   not set on any rank', lines)
        (directory / 'campaign/interrupted.log').write_text(with_nccl_proto(nccl_log('efa')))
        lines = workshop.interrupt_block(stage, directory, 16)
        self.assertIn('  NCCL_PROTO                   interrupted run: Simple on 16 of 16 ranks; '
                      'resumed run: not set on any rank', lines)
        (directory / 'campaign/resumed.log').write_text(with_nccl_proto(nccl_log('efa')))
        lines = workshop.interrupt_block(stage, directory, 16)
        self.assertIn('  NCCL_PROTO                   Simple on 16 of 16 ranks', lines)
        self.assertIn('  goodput including the stop   13,790 tokens/s, 3.10 M tokens per GPU-hour  '
                      '(first launch to the final durable save)', lines)
        self.directory.cleanup()
        self.directory = tempfile.TemporaryDirectory()
        self.results = Path(self.directory.name)
        stage, directory = self.interrupt_dir(pending=False, durable=20)
        lines = workshop.interrupt_block(stage, directory, 16)
        self.assertIn('  stopped after update 30; the update-20 upload had finished (durable update 20)', lines)
        self.assertIn('  updates run twice: 21-30 (10 updates)', lines)


class NcclTests(unittest.TestCase):
    def test_recorded_socket_and_efa_lines(self):
        socket = workshop.parse_nccl(nccl_log('socket'))
        self.assertEqual((socket['ranks'], socket['networks']), (16, {'Socket': 16}))
        self.assertEqual((socket['network_channels'], socket['gdrdma_channels']), (2, 0))
        self.assertIsNone(workshop.nccl_problem(socket, 'socket', 16))
        self.assertIn('expected AWS Libfabric', workshop.nccl_problem(socket, 'efa', 16))
        efa = workshop.parse_nccl(nccl_log('efa'))
        self.assertEqual((efa['ranks'], efa['networks']), (16, {'AWS Libfabric': 16}))
        self.assertEqual(efa['providers'], ['efa, fabric efa-direct (found 2 nics)'])
        self.assertEqual(efa['protocols'], ['RDMA (user set)'])
        self.assertEqual((efa['network_channels'], efa['gdrdma_channels']), (4, 0))
        fewer = workshop.parse_nccl('\n'.join(nccl_log('efa').splitlines()[1:]))
        self.assertIn('on 15 of 16 ranks', workshop.nccl_problem(fewer, 'efa', 16))
        gdr = workshop.parse_nccl('gpu-node-2:1:1 [7] NCCL INFO Channel 01/0 : 15[7] -> 7[7] [send] via NET/AWS Libfabric/1/GDRDMA')
        self.assertEqual((gdr['network_channels'], gdr['gdrdma_channels']), (1, 1))

    def test_nccl_proto_lines_counted_per_rank(self):
        recorded = workshop.parse_nccl(nccl_log('efa'))
        self.assertEqual(recorded['nccl_protos'], {})
        self.assertEqual(workshop.nccl_proto_text(recorded, 16), 'not set on any rank')
        simple = workshop.parse_nccl(with_nccl_proto(nccl_log('efa')))
        self.assertEqual(simple['nccl_protos'], {'Simple': 16})
        self.assertEqual(workshop.nccl_proto_text(simple, 16), 'Simple on 16 of 16 ranks')
        self.assertEqual(simple['ranks'], 16)


class EfaCounterTests(unittest.TestCase):
    def sysfs(self, root, device, counters, driver='efa'):
        base = Path(root) / device
        (Path(root) / 'drivers' / driver).mkdir(parents=True, exist_ok=True)
        (base / 'device').mkdir(parents=True)
        os.symlink(Path(root) / 'drivers' / driver, base / 'device' / 'driver')
        hw = base / 'ports/1/hw_counters'
        hw.mkdir(parents=True)
        for name, value in counters.items():
            (hw / name).write_text(f'{value}\n')
        return hw

    def test_fake_sysfs_missing_and_unreadable_counters(self):
        with tempfile.TemporaryDirectory() as directory:
            full = Path(directory) / 'full'
            self.sysfs(full / 'class', 'rdmap0s0', {'tx_bytes': 10, 'rx_bytes': 20, 'rdma_write_bytes': 30})
            self.sysfs(full / 'class', 'ens5', {'tx_bytes': 99}, driver='ena')
            devices, errors = workshop.read_efa_counters(full / 'class')
            self.assertEqual((devices, errors), ({'rdmap0s0': {'tx_bytes': 10, 'rx_bytes': 20, 'rdma_write_bytes': 30}}, []))
            empty = Path(directory) / 'empty'
            self.sysfs(empty / 'class', 'rdmap0s0', {})
            missing = workshop.read_efa_counters(empty / 'class')
            self.assertEqual(missing, ({'rdmap0s0': {}}, []))
            lines = [json.dumps(dict(host='gpu-node-1', devices=devices, errors=errors)),
                     json.dumps(dict(host='gpu-node-2', devices=missing[0], errors=missing[1]))]
            if os.geteuid() != 0:
                locked = Path(directory) / 'locked'
                hw = self.sysfs(locked / 'class', 'rdmap0s0', {'tx_bytes': 1, 'rx_bytes': 2})
                (hw / 'tx_bytes').chmod(0)
                unreadable = workshop.read_efa_counters(locked / 'class')
                self.assertEqual(len(unreadable[1]), 1)
                lines.append(json.dumps(dict(host='gpu-node-3', devices=unreadable[0], errors=unreadable[1])))
            before = workshop.merge_efa_readings(lines)
            self.assertEqual(before['gpu-node-2'], {})
            after = json.loads(json.dumps(before))
            after['gpu-node-1']['rdmap0s0'].update(tx_bytes=10 + 4_000_000_000, rx_bytes=20 + 3_000_000_000)
            deltas = workshop.efa_deltas(before, after)
            self.assertIsNone(deltas['gpu-node-2'])
            text = workshop.efa_text(deltas)
            self.assertIn('gpu-node-1 +4.0 GB sent, +3.0 GB received', text)
            self.assertIn('EFA counters unavailable on gpu-node-2', text)
            if os.geteuid() != 0:
                self.assertIn('EFA counters unavailable on gpu-node-3', text)

    def test_unavailable_counters_do_not_fail_an_efa_stage(self):
        stage = workshop.plan('efa', job('efa'))[0]
        with tempfile.TemporaryDirectory() as directory:
            stage_dir = finished_stage(directory, stage)
            (stage_dir / 'launch.log').write_text(nccl_log('efa'))
            (stage_dir / 'train/summary.json').write_text(json.dumps(summary(stage, EFA['seconds'], EFA['rate'], 1)))
            (stage_dir / 'efa-before.json').write_text(json.dumps({'gpu-node-1': {}, 'gpu-node-2': {}}))
            (stage_dir / 'efa-after.json').write_text(json.dumps({'gpu-node-1': {}, 'gpu-node-2': {}}))
            code, checks, _ = workshop.post_run_checks(stage, stage_dir, 0, 16)
            self.assertEqual(code, 0)
            self.assertIn('EFA counters unavailable on gpu-node-1', workshop.efa_text(workshop.efa_deltas({}, {
                'gpu-node-1': {}})))


class ProfileTableTests(unittest.TestCase):
    """Synthetic trace: times in microseconds; update 2 is ProfilerStep#1 on the main thread."""

    def trace(self, scale=1.0):
        main = dict(pid=1, tid=10)
        backward = dict(pid=1, tid=20)
        gpu, nccl_stream = dict(pid=0, tid=7), dict(pid=0, tid=20)

        def x(name, cat, ts, dur, where, **args):
            return dict(ph='X', name=name, cat=cat, ts=ts, dur=dur * scale, args=args, **where)
        return [
            x('ProfilerStep#0', 'user_annotation', 0, 1000, main),
            x('ProfilerStep#1', 'user_annotation', 1000, 2000, main),
            # Update 1 (warmup): its kernel must not count in update 2.
            x('aten::mm', 'cpu_op', 100, 10, main, **{'External id': 1}),
            x('warmup_gemm', 'kernel', 500, 400, gpu, **{'External id': 1}),
            # Update 2: input wait, then attention forward inside attention/core.
            x('input/wait', 'user_annotation', 1000, 100, main),
            x('attention/core', 'user_annotation', 1200, 200, main),
            x('aten::bmm', 'cpu_op', 1250, 10, main, **{'External id': 7, 'Sequence number': 42}),
            x('aten::mm', 'cpu_op', 1450, 10, main, **{'External id': 8, 'Sequence number': 43}),
            x('autograd::engine::evaluate_function: BmmBackward0', 'cpu_op', 2000, 100, backward,
              **{'External id': 30, 'Sequence number': 42}),
            x('aten::bmm', 'cpu_op', 2010, 10, backward, **{'External id': 9}),
            x('autograd::engine::evaluate_function: MmBackward0', 'cpu_op', 2200, 50, backward,
              **{'External id': 31, 'Sequence number': 43}),
            x('aten::mm', 'cpu_op', 2210, 10, backward, **{'External id': 10}),
            x('attention_forward_kernel', 'kernel', 1300, 200, gpu, **{'External id': 7}),
            x('mlp_gemm', 'kernel', 1600, 300, gpu, **{'External id': 8}),
            x('ncclDevKernel_AllGather_RING_LL', 'kernel', 1400, 300, nccl_stream, **{'External id': 50}),
            x('attention_backward_kernel', 'kernel', 2050, 200, gpu, **{'External id': 9}),
            x('mlp_backward_gemm', 'kernel', 2260, 30, gpu, **{'External id': 10}),
            x('Memcpy HtoD', 'gpu_memcpy', 2300, 100, gpu),
        ]

    def test_rows_from_known_overlap(self):
        tool = load_tool('profile-table')
        table = tool.update_table(self.trace(), update=2)
        self.assertAlmostEqual(table['update'], 0.002)
        # compute [1300,1500] [1600,1900] [2050,2250] [2260,2290]
        self.assertAlmostEqual(table['compute'], 0.000730)
        self.assertAlmostEqual(table['nccl'], 0.000300)
        # NCCL [1400,1700] overlaps compute on [1400,1500] and [1600,1700].
        self.assertAlmostEqual(table['exposed'], 0.000100)
        # busy [1300,1900] [2050,2250] [2260,2290] [2300,2400]
        self.assertAlmostEqual(table['idle'], 0.002 - 0.000930)
        self.assertAlmostEqual(table['input_wait'], 0.000100)
        # forward [1300,1500] and the backward of sequence number 42 [2050,2250]
        self.assertAlmostEqual(table['attention'], 0.000400)
        self.assertEqual(table['top'][0][:2], ('mlp_gemm', 1))
        self.assertNotIn('warmup_gemm', [name for name, _, _ in table['top']])
        lines = tool.table_lines('Profile of profile-efa-eager, rank 0, update 2', table)
        self.assertEqual(lines[1], '  update                          0.002 s')
        self.assertTrue(lines[4].startswith('  communication not overlapped    0.000 s     5.0 %'))
        with self.assertRaises(tool.NoSpan):
            tool.update_table(self.trace(), update=4)

    def test_command_reads_the_slowest_rank(self):
        tool = load_tool('profile-table')
        with tempfile.TemporaryDirectory() as directory:
            train = Path(directory) / 'profile-efa-eager' / 'train'
            train.mkdir(parents=True)
            (train.parent / 'run.json').write_text(json.dumps({'start_utc': 1}))
            for rank, gap in ((0, 1.0), (1, 2.5)):
                write_rows(train / f'rank-{rank}-updates.jsonl',
                           [dict(update=1, monotonic_seconds=10.0), dict(update=2, monotonic_seconds=10.0 + gap)])
                (train / f'trace-rank-{rank}.json').write_text(json.dumps({'traceEvents': self.trace(scale=rank + 1)}))
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch.dict(os.environ, LLM_RESULTS_DIR=directory), redirect_stdout(stdout), redirect_stderr(stderr):
                self.assertEqual(tool.main([]), 0)
                self.assertEqual(tool.main(['--rank', 'all']), 0)
                self.assertEqual(tool.main(['--update', '3']), 3)
                self.assertEqual(tool.main(['profile-missing']), 2)
            titles = [line for line in stdout.getvalue().splitlines() if line.startswith('Profile of')]
            self.assertEqual(titles, ['Profile of profile-efa-eager, rank 1 (slowest for update 2), update 2',
                                      'Profile of profile-efa-eager, mean of 2 ranks, update 2'])
            # Rank 1's synthetic durations are twice rank 0's; the mean is one and a half times.
            self.assertIn('  NCCL kernels busy               0.001 s', stdout.getvalue())
            self.assertIn('has no ProfilerStep#2 span for update 3', stderr.getvalue())

    def test_each_table_read_is_kept_for_the_chart_of_the_next_profile(self):
        tool = load_tool('profile-table')
        with tempfile.TemporaryDirectory() as directory:
            for start, name in enumerate(('profile-socket-eager', 'profile-efa-eager'), 1):
                train = Path(directory) / name / 'train'
                train.mkdir(parents=True)
                (train.parent / 'run.json').write_text(json.dumps({'start_utc': start}))
                write_rows(train / 'rank-0-updates.jsonl',
                           [dict(update=1, monotonic_seconds=10.0), dict(update=2, monotonic_seconds=11.0)])
                (train / 'trace-rank-0.json').write_text(json.dumps({'traceEvents': self.trace(scale=start)}))
            stdout = io.StringIO()
            with patch.dict(os.environ, LLM_RESULTS_DIR=directory), redirect_stdout(stdout):
                self.assertEqual(tool.main(['profile-socket-eager']), 0)
                saved = json.loads((Path(directory) / 'profile-socket-eager' / tool.TABLES).read_text())
                self.assertEqual(list(saved), ['rank slowest, update 2'])
                self.assertAlmostEqual(saved['rank slowest, update 2']['table']['update'], 0.002)
                first = stdout.getvalue().splitlines()
                stdout.truncate(0)
                stdout.seek(0)
                # The second call parses only its own trace and draws the first table from the saved file.
                with patch.object(tool, 'update_table', wraps=tool.update_table) as parsed:
                    self.assertEqual(tool.main(['profile-efa-eager']), 0)
                self.assertEqual(parsed.call_count, 1)
            second = stdout.getvalue().splitlines()
            self.assertIn('  profile-efa-eager     run python3 tools/profile-table.py profile-efa-eager to add it', first)
            rows = [line for line in second if line.startswith(('  profile-socket-eager', '  profile-efa-eager'))]
            self.assertEqual([line.split()[-2:] for line in rows], [['0.002', 's'], ['0.004', 's']])


class ProgressionTests(unittest.TestCase):
    def test_rows_ratios_notes_goodput_and_markdown(self):
        tool = load_tool('progression')
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, LLM_RESULTS_DIR=directory):
            results = Path(directory)
            stages = {name: workshop.plan(sub, settings)[index] for name, sub, settings, index in (
                ('baseline', 'baseline', job(), 0), ('efa', 'efa', job('efa'), 0),
                ('sdpa', 'sdpa', job('efa', 'sdpa'), 0), ('probe-socket-eager', 'probe', job(), 0),
                ('ckpt-sync', 'checkpoint', job('efa', 'sdpa', 'sync'), 0),
                ('interrupt', 'interrupt', job('efa', 'sdpa', 'sync'), 0),
                ('profile-efa-eager', 'profile', job('efa'), 0))}
            values = {'baseline': (BASELINE['seconds'], BASELINE['rate']), 'efa': (EFA['seconds'], EFA['rate']),
                      'sdpa': (4.0, 65504.0), 'probe-socket-eager': (13.39, 9779.0),  # synthetic sdpa and probe
                      'ckpt-sync': (80.0, 31114.4), 'profile-efa-eager': (9.0, 1.0)}
            for name, stage in stages.items():
                directory_ = finished_stage(results, stage)
                if name == 'interrupt':
                    (directory_ / 'campaign').mkdir()
                    (directory_ / 'campaign/campaign.json').write_text(json.dumps(dict(
                        elapsed_seconds=190.0, training_goodput_tokens_per_second=13790.3,
                        tokens_per_allocated_gpu_hour=3.1e6, interrupted_update=30, restored_update=5,
                        replayed_update_count=25)))
                    continue
                seconds, rate = values[name]
                extra = {}
                if name == 'ckpt-sync':
                    extra['training_goodput'] = {'retained_progress_tokens': 2620160,
                                                 'training_goodput_tokens_per_second': 29690.0,
                                                 'tokens_per_allocated_gpu_hour': 6.68e6}
                (directory_ / 'train/summary.json').write_text(json.dumps(summary(stage, seconds, rate, 2**30, **extra)))
            (results / 'ceilings').mkdir()
            (results / 'ceilings/ceilings.json').write_text(json.dumps(
                {'gemm_tflops_per_gpu': GEMM_PEAK, 'memory_read_bytes_per_second_per_gpu': 662e9,
                 'ridge_flop_per_byte': GEMM_PEAK * 1e12 / 662e9}))
            notes = tool.Notes()
            ceilings, table = tool.collect(results, notes)
            self.assertEqual([row['_name'] for row in table],
                             ['baseline', 'probe-socket-eager', 'efa', 'sdpa', 'ckpt-sync', 'interrupt'])
            by_name = {row['_name']: row for row in table}
            self.assertEqual(by_name['efa']['vs baseline'], '2.34x')
            self.assertEqual(by_name['efa']['MFU vs your GEMM peak'], '19.0 %')
            self.assertEqual(by_name['baseline']['GPU-s per 1M tokens'], '825')
            self.assertEqual(by_name['efa']['vs previous step'], '2.34x')
            self.assertEqual(by_name['sdpa']['vs previous step'], f'{EFA["seconds"] / 4.0:.2f}x')
            self.assertEqual(by_name['sdpa']['vs baseline'], f'{BASELINE["seconds"] / 4.0:.2f}x')
            self.assertTrue(by_name['probe-socket-eager']['vs baseline'].startswith('('))
            self.assertIn('probe-socket-eager vs baseline: not compared, different workload', notes.texts)
            self.assertEqual(by_name['ckpt-sync']['Goodput tokens/s'], '29,690')
            self.assertEqual(by_name['ckpt-sync']['Goodput M tokens per GPU-hour'], '6.68')
            # The interrupt goodput covers the whole command, so both its cells point to a note.
            self.assertEqual(by_name['interrupt']['vs baseline'], '(1)')
            self.assertEqual(by_name['interrupt']['Goodput tokens/s'], '13,790 (2)')
            self.assertEqual(by_name['interrupt']['Goodput M tokens per GPU-hour'], '3.10 (2)')
            self.assertEqual(notes.texts[1], 'interrupt: goodput over the whole command, from the first launch to the '
                             'end of the final save, including both launches and both model loads. The goodput '
                             'interval of the saving rows starts at the end of model loading, so the two are not compared')
            plain = tool.render(ceilings, table, notes, False)
            self.assertEqual(plain[0], 'GEMM peak 171.7 TFLOP/s per GPU, memory read 662 GB/s per GPU, ridge 259 FLOP per byte')
            self.assertIn('interrupt: stopped after update 30, resumed from update 5, 25 updates ran twice', plain)
            markdown = tool.render(ceilings, table, notes, True)
            self.assertTrue(markdown[2].startswith('| Stage | Transport | Attention | Saving | Work | Seconds |'))
            self.assertTrue(markdown[4].startswith('| baseline | socket | eager | none | 5 x 2,048 | 13.51 | 19,388 |'))
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(tool.main(['--markdown']), 0)
            self.assertNotIn('profile-efa-eager', stdout.getvalue())


class AllocationTests(unittest.TestCase):
    def test_allocation_file_round_trip_and_parsers(self):
        from serve_llm import allocation_cpus
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'allocation.env'
            workshop.write_allocation(path, 42, 'gpu-[1-2]', 192)
            self.assertEqual(workshop.read_allocation(path), {
                'SLURM_JOB_ID': '42', 'SLURM_JOB_NUM_NODES': '2', 'SLURM_JOB_NODELIST': 'gpu-[1-2]',
                'SLURM_JOB_CPUS_PER_NODE': '192(x2)', 'AIM347_ALLOCATION_ID': '42'})
            self.assertEqual(oct(path.stat().st_mode & 0o777), '0o600')
            sourced = subprocess.run(['bash', '-c', f'source {path}; echo "$SLURM_JOB_CPUS_PER_NODE $SLURM_JOB_NODELIST"'],
                                     capture_output=True, text=True, check=True).stdout.strip()
            self.assertEqual(sourced, '192(x2) gpu-[1-2]')
        self.assertEqual(allocation_cpus('192(x2)', 2), 192)
        self.assertEqual(workshop.parse_granted('salloc: Pending job allocation 42\nsalloc: Granted job allocation 42\n'), 42)
        self.assertIsNone(workshop.parse_granted('salloc: error: Unable to allocate resources: Immediate execution impossible'))
        record = workshop.parse_scontrol_record('JobId=42 JobName=aim347-session JobState=RUNNING TimeLimit=01:50:00 '
                                                'NodeList=gpu-[1-2] ReqNodeList=(null) NumNodes=2')
        self.assertEqual((record['NodeList'], record['TimeLimit']), ('gpu-[1-2]', '01:50:00'))
        self.assertEqual(workshop.time_limit_minutes('01:50:00'), 110)
        self.assertEqual(workshop.time_limit_minutes('1-00:00:00'), 1440)
        self.assertEqual(workshop.time_limit_minutes('UNLIMITED'), 'UNLIMITED')


class CeilingTests(unittest.TestCase):
    def test_median_of_sixteen_gpus(self):
        with tempfile.TemporaryDirectory() as directory:
            for host_index, host in enumerate(HOSTS):
                for gpu in range(8):
                    best = 160.0 + host_index * 8 + gpu  # synthetic
                    (Path(directory) / f'gemm-{host}-gpu{gpu}.json').write_text(json.dumps(dict(
                        instance_type='g7.48xlarge', gpu='NVIDIA RTX PRO 4500 Blackwell Server Edition',
                        input_dtype='bf16', accumulate_dtype='fp32', sparse=False, n_elements=8192, warmup_trials=10,
                        trials=30, best_ms=2 * 8192**3 / (best * 1e9), median_ms=1.0, best_tflops_per_gpu=best,
                        median_tflops_per_gpu=best - 20, trial_ms=[1.0] * 30)))
                    (Path(directory) / f'dram-{host}-gpu{gpu}.json').write_text(json.dumps(dict(
                        instance_type='g7.48xlarge', gpu='NVIDIA RTX PRO 4500 Blackwell Server Edition',
                        device_index=0, gpu_uuid_hex='00', method='streaming_read_only_cg',
                        bytes_read_per_trial=2**30, l2_bytes=2**26, warmup_trials=10, trials=30,
                        correctness='passed', best_ms=1.0, median_ms=1.0, best_bytes_per_second=700e9,
                        median_bytes_per_second=(650 + host_index * 8 + gpu) * 1e9, trial_ms=[1.0] * 30)))
            result = workshop.aggregate_ceilings(directory)
        self.assertEqual(result['gpu_count'], 16)
        self.assertEqual(result['gemm_tflops_per_gpu'], 167.5)
        self.assertEqual(result['memory_read_bytes_per_second_per_gpu'], 657.5e9)
        self.assertAlmostEqual(result['ridge_flop_per_byte'], 167.5e12 / 657.5e9)
        lines = workshop.ceilings_lines(result)
        self.assertEqual(lines[1], '  gpu-node-1  160.0 161.0 162.0 163.0 164.0 165.0 166.0 167.0')
        self.assertEqual(lines[3], '  Your GEMM peak (median of 16 GPUs): 167.5 TFLOP/s per GPU')
        self.assertEqual(lines[5], '  gpu-node-1  650 651 652 653 654 655 656 657')
        self.assertEqual(lines[7], '  Your memory-read ceiling (median of 16 GPUs): 658 GB/s per GPU')
        self.assertEqual(lines[8], 'Ridge point: 255 FLOP per byte')


def stub(directory, name, script):
    path = Path(directory) / name
    path.write_text('#!/usr/bin/env bash\n' + script)
    path.chmod(0o755)


class NodeCollectorTests(unittest.TestCase):
    """install-node-collectors.sh with stub id, systemctl, systemd-run and curl on PATH."""

    def run_script(self, *arguments, timer='OnUnitActiveUSec=5s', prom='aim347_lustre_collection_success 1',
                   metrics='aim347_lustre_collection_success 1\nnode_amazonefa_tx_bytes{device="rdmap0s0",port="1"} 7'):
        bin_dir = self.root / 'bin'
        bin_dir.mkdir(exist_ok=True)
        calls = self.root / 'calls'
        stub(bin_dir, 'id', 'echo 0\n')
        stub(bin_dir, 'hostname', 'echo gpu-node-1\n')
        stub(bin_dir, 'systemctl', f'''echo "systemctl $*" >> {calls}
case "$1 $2" in
  "is-active --quiet") [[ $3 == aim347-lustre ]] && {{ [[ -f {self.root}/lustre-running ]]; exit; }}; exit 0 ;;
  "show efa-metrics.timer") echo "TimersMonotonic={{ OnBootUSec=30s ; next_elapse=0 }} {{ {timer} ; next_elapse=1 }}" ;;
esac
exit 0
''')
        stub(bin_dir, 'systemd-run', f'echo "systemd-run $*" >> {calls}\ntouch {self.root}/lustre-running\n'
                                     f"printf '%s\\n' '{prom}' > {self.textfile}/lustre.prom\n")
        stub(bin_dir, 'curl', f"printf '%s\\n' '{metrics}'\n")
        env = dict(os.environ, PATH=f'{bin_dir}:{os.environ["PATH"]}', AIM347_SYSTEMD_DIR=str(self.systemd),
                   AIM347_TEXTFILE_DIR=str(self.textfile))
        result = subprocess.run(['bash', str(LAB / 'facilitator/install-node-collectors.sh'), *arguments], env=env,
                                capture_output=True, text=True, timeout=60)
        return result, calls.read_text().splitlines() if calls.exists() else []

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.systemd = self.root / 'systemd'
        self.textfile = self.root / 'textfile'

    def tearDown(self):
        self.directory.cleanup()

    def test_drop_in_restart_order_single_unit_and_verify(self):
        result, calls = self.run_script('--verify', '--timeout', '5')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip().splitlines()[-1], 'install-node-collectors gpu-node-1: '
                         'efa-metrics.timer 5s ok; lustre counter ok; node-exporter serves both')
        self.assertEqual((self.systemd / 'efa-metrics.timer.d/aim347-5s.conf').read_bytes(),
                         b'[Timer]\nOnUnitActiveSec=\nOnUnitActiveSec=5s\nAccuracySec=1s\n')
        self.assertLess(calls.index('systemctl daemon-reload'), calls.index('systemctl try-restart efa-metrics.timer'))
        started = [call for call in calls if call.startswith('systemd-run')]
        self.assertEqual(started, [f'systemd-run --unit=aim347-lustre --collect --property=Restart=on-failure '
                                   f'/usr/bin/python3 {LAB}/lib/counters.py lustre --output {self.textfile}/lustre.prom'])
        self.assertNotIn('lustre_write_bytes_total', (self.textfile / 'lustre.prom').read_text())
        result, calls = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len([call for call in calls if call.startswith('systemd-run')]), 1)

    def test_verify_names_the_missing_condition(self):
        cases = [(dict(prom='aim347_lustre_collection_success 0'), 'aim347_lustre_collection_success 1 in'),
                 (dict(metrics='aim347_lustre_collection_success 1'), 'node_amazonefa_tx_bytes'),
                 (dict(timer='OnUnitActiveUSec=30s'), 'efa-metrics.timer at 5 s')]
        for options, condition in cases:
            with self.subTest(condition=condition):
                (self.root / 'lustre-running').unlink(missing_ok=True)
                result, _ = self.run_script('--verify', '--timeout', '1', **options)
                self.assertEqual(result.returncode, 1)
                self.assertIn('timed out after 1 s waiting for', result.stderr)
                self.assertIn(condition, result.stderr)

    def test_requires_root(self):
        bin_dir = self.root / 'bin'
        bin_dir.mkdir()
        stub(bin_dir, 'id', 'echo 1000\n')
        result = subprocess.run(['bash', str(LAB / 'facilitator/install-node-collectors.sh')], capture_output=True,
                                text=True, env=dict(os.environ, PATH=f'{bin_dir}:{os.environ["PATH"]}'))
        self.assertEqual(result.returncode, 2)


class PostRunTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.results = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def counters(self, stage_dir, increase=True):
        (stage_dir / 'efa-before.json').write_text(json.dumps(efa_counters(100, 100)))
        after = efa_counters(100, 100)
        if increase:
            for host in HOSTS:
                after[host]['rdmap0s0']['tx_bytes'] += 5
        (stage_dir / 'efa-after.json').write_text(json.dumps(after))

    def restore_dir(self, last_updates, first_update=21):
        stage = workshop.plan('restore', job('efa', 'sdpa', 'sync'))[0]
        stage_dir = finished_stage(self.results, stage)
        for rank, last in enumerate(last_updates):
            start = first_update if rank == 0 else 21
            values = [dict(update=u, monotonic_seconds=float(u), global_mean_loss=2.0, checkpoint_upload_pending=False,
                           durable_update=20) for u in range(start, last + 1)]
            values[-1]['controlled_interruption'] = True
            write_rows(stage_dir / f'train/rank-{rank}-updates.jsonl', values)
        (stage_dir / 'launch.log').write_text(nccl_log('efa'))
        self.counters(stage_dir)
        return stage, stage_dir

    def test_restore_decided_from_rows(self):
        stage, stage_dir = self.restore_dir([25] * 16)
        self.assertEqual(workshop.post_run_checks(stage, stage_dir, 1, 16)[0], 0)
        stage, stage_dir = self.restore_dir([25] * 15 + [24])
        code, _, reason = workshop.post_run_checks(stage, stage_dir, 1, 16)
        self.assertEqual(code, 4)
        self.assertIn('stopped after update 25', reason)
        stage, stage_dir = self.restore_dir([25] * 16, first_update=22)
        code, checks, _ = workshop.post_run_checks(stage, stage_dir, 1, 16)
        self.assertEqual(code, 5)
        self.assertEqual([c['name'] for c in checks if not c['ok']], ['Restore start'])

    def test_interrupt_needs_campaign_and_two_complete_segment_logs(self):
        stage = workshop.plan('interrupt', job('efa', 'sdpa', 'process'))[0]
        stage_dir = finished_stage(self.results, stage)
        (stage_dir / 'campaign').mkdir()
        (stage_dir / 'campaign/campaign.json').write_text('{}')
        (stage_dir / 'campaign/interrupted.log').write_text(nccl_log('efa'))
        (stage_dir / 'campaign/resumed.log').write_text(nccl_log('efa'))
        self.counters(stage_dir)
        self.assertEqual(workshop.post_run_checks(stage, stage_dir, 0, 16)[0], 0)
        self.assertEqual(workshop.post_run_checks(stage, stage_dir, 1, 16)[0], 4)
        (stage_dir / 'campaign/resumed.log').write_text('\n'.join(nccl_log('efa').splitlines()[1:]))
        code, checks, _ = workshop.post_run_checks(stage, stage_dir, 0, 16)
        self.assertEqual(code, 5)
        self.assertEqual([c['detail'] for c in checks if not c['ok']],
                         ['expected AWS Libfabric on 16 ranks, found AWS Libfabric on 15 of 16 ranks'])

    def test_efa_stage_fails_when_read_counters_did_not_increase(self):
        stage = workshop.plan('efa', job('efa'))[0]
        stage_dir = finished_stage(self.results, stage)
        (stage_dir / 'launch.log').write_text(nccl_log('efa'))
        (stage_dir / 'train/summary.json').write_text(json.dumps(summary(stage, EFA['seconds'], EFA['rate'], 1)))
        self.counters(stage_dir)
        self.assertEqual(workshop.post_run_checks(stage, stage_dir, 0, 16)[0], 0)
        after = json.loads((stage_dir / 'efa-after.json').read_text())
        after['gpu-node-2'] = efa_counters(100, 100)['gpu-node-2']
        (stage_dir / 'efa-after.json').write_text(json.dumps(after))
        code, checks, _ = workshop.post_run_checks(stage, stage_dir, 0, 16)
        self.assertEqual(code, 5)
        self.assertEqual([c['detail'] for c in checks if not c['ok']], ['no EFA counter increased on gpu-node-2'])
        self.assertEqual(workshop.post_run_checks(stage, stage_dir, 1, 16)[0], 4)
        socket = workshop.plan('baseline', job())[0]
        socket_dir = finished_stage(self.results, socket)
        (socket_dir / 'launch.log').write_text(nccl_log('socket'))
        (socket_dir / 'train/summary.json').write_text(json.dumps(summary(socket, 13.5, 19388.0, 1)))
        self.counters(socket_dir, increase=False)
        self.assertEqual(workshop.post_run_checks(socket, socket_dir, 0, 16)[0], 0)
        (socket_dir / 'train/summary.json').write_text(json.dumps(dict(summary(socket, 13.5, 19388.0, 1), profiled=True)))
        self.assertEqual(workshop.post_run_checks(socket, socket_dir, 0, 16)[0], 5)


class StageRunTests(unittest.TestCase):
    """The worker and the follower of run.sh with a fake launcher."""

    def test_baseline_stage_end_to_end(self):
        stage = workshop.plan('baseline', job())[0]
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            (results / 'baseline').mkdir()
            (results / 'baseline/old.txt').write_text('earlier attempt')
            launched = []

            def runner(command, cwd, env, stdout, stderr):
                launched.append((command, env['LLM_TRANSPORT']))
                train = results / 'baseline' / 'train'
                train.mkdir()
                stdout.write(nccl_log('socket'))
                for rank in range(16):
                    write_rows(train / f'rank-{rank}-updates.jsonl', rows(BASELINE['monotonic'], LOSSES))
                (train / 'summary.json').write_text(json.dumps(summary(stage, BASELINE['seconds'], BASELINE['rate'],
                                                                       BASELINE['peak'])))
                return subprocess.CompletedProcess(command, 0)

            invocation = dict(invocation='20261103T010203Z', stages=['baseline'], allocation_id=42, pid=None)
            code = workshop.run_stage(stage, invocation, results, runner=runner, world=16, label='label',
                                      efa_reader=lambda log: efa_counters(1, 1))
            self.assertEqual(code, 0)
            self.assertEqual(launched[0][1], 'socket')
            self.assertEqual(launched[0][0][:2], ['bash', '2.run-llm.sh'])
            self.assertEqual(len(list((results / 'attempts').glob('baseline-*/old.txt'))), 1)
            run = json.loads((results / 'baseline/run.json').read_text())
            self.assertEqual((run['status'], run['exit_status'], run['launcher_exit']), ('finished', 0, 0))
            self.assertEqual(run['job_env'], job())
            output = []
            self.assertEqual(workshop.follow_stage('baseline', invocation, results, out=output.append, world=16), 0)
            self.assertEqual(output[:3], [
                '[aim347] baseline: LLM_TRANSPORT=socket ATTENTION=eager CHECKPOINT_MODE=none',
                '[aim347] 5 updates of 32 sequences x 2,048 positions; updates 2-5 are timed (262,016 targets)',
                f'[aim347] allocation 42, results {results / "baseline"}'])
            self.assertEqual(output[3:9], ['update   seconds      loss', '     1    warmup  2.157261',
                                           '     2     3.375  2.170563', '     3     3.393  2.138267',
                                           '     4     3.379  2.040267', '     5     3.369  2.107691'])
            self.assertRegex(output[9], r'^\[aim347\] finished in \d+ s$')
            self.assertEqual(output[10], 'baseline (socket, eager attention, no saving)')
            self.assertEqual(workshop.latest_invocation(results)['stages'], ['baseline'])

    def test_failed_launch_prints_the_log_tail(self):
        stage = workshop.plan('efa', job('efa'))[0]
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)

            def runner(command, cwd, env, stdout, stderr):
                stdout.write('\n'.join(f'line {n}' for n in range(60)) + '\n')
                return subprocess.CompletedProcess(command, 1)
            invocation = dict(invocation='i', stages=['efa'], allocation_id=42, pid=None)
            self.assertEqual(workshop.run_stage(stage, invocation, results, runner=runner, world=16, label='l',
                                                efa_reader=lambda log: {}), 4)
            output = []
            self.assertEqual(workshop.follow_stage('efa', invocation, results, out=output.append, world=16), 4)
            self.assertIn('efa failed: the launcher exited with status 1', output)
            self.assertEqual(output[-1], 'line 59')
            self.assertEqual(len([line for line in output if line.startswith('line ')]), 40)


class CheckReportTests(unittest.TestCase):
    def test_every_check_line(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            data = home / 'data'
            pins = {}
            for name, dataset in workshop.DATASETS.items():
                config = workshop.config_values(dataset['config'])
                tokens = data / name
                tokens.mkdir(parents=True)
                (tokens / 'tokens.bin').write_bytes(b'\0' * (4 * dataset['records'] * config['sequence_length']))
                (tokens / 'lengths.bin').write_bytes(b'\0' * (4 * dataset['records']))
                pins[name] = workshop.dataset_fingerprint(tokens)
                (tokens / 'manifest.json').write_text(json.dumps(dict(
                    {k: config[k] for k in workshop.PROVENANCE_KEYS}, records=dataset['records'],
                    sequence_length=config['sequence_length'], minimum_document_tokens=4096, sha256=pins[name])))
            (data / 'model').mkdir()
            (data / 'model/config.json').write_text('{}')
            config = workshop.config_values('configs/smol-2048.json')
            (data / 'model/aim347-pin.json').write_text(json.dumps(
                {'model_id': config['model_id'], 'model_revision': config['model_revision']}))
            image = home / 'images/aim347-lab.sqsh'
            image.parent.mkdir()
            image.write_bytes(b'image')
            Path(f'{image}.sha256').write_text(f'{"a" * 64}  {image}\n')
            Path(f'{image}.verified').write_text('gpu-node-1 OK t\ngpu-node-2 OK t\n')
            password = home / 'grafana-admin-password'
            password.write_text('secret\n')
            env = dict(AIM347_HOME=str(home), LAB_IMAGE=str(image), LAB_IMAGE_SHA256='a' * 64, LLM_DATA_DIR=str(data),
                       PARTITION='gpu', GPUS_PER_NODE='8', AIM347_INSTANCE_TYPE='g7.48xlarge',
                       AIM347_GRAFANA_URL='https://203.0.113.10/grafana/', AIM347_GRAFANA_PASSWORD_FILE=str(password))
            gpus = [['NVIDIA RTX PRO 4500 Blackwell Server Edition', '32623']] * 8
            reports = [dict(host=host, gpus=gpus, efa=['rdmap83s0', 'rdmap176s0'], error=None) for host in HOSTS]

            def fetch(url):
                if 'grafana' in url:
                    return 200, '{"database": "ok"}'
                return 200, json.dumps({'data': {'result': [{'value': [0, '2']}]}})
            with patch.dict(workshop.EXPECTED_FINGERPRINTS, {name: pins[name] for name in pins if name != 'tokens-2048-1280'}):
                lines, failures = workshop.check_lines(env, 'gpu-node-1 idle\ngpu-node-2 idle\n', reports, fetch)
        status = {line.split()[0]: [] for line in lines[1:-1]}
        for line in lines[1:-1]:
            status[line.split()[0]].append(line.split()[1])
        self.assertEqual(lines[0], 'AIM347 check')
        self.assertIn('  gpus         PASS  gpu-node-1: 8 x NVIDIA RTX PRO 4500 Blackwell Server Edition, 32623 MiB each', lines)
        self.assertIn('  efa          PASS  gpu-node-2: 2 EFA devices (rdmap83s0, rdmap176s0)', lines)
        self.assertIn('  partition    PASS  gpu: 2 nodes (gpu-node-1, gpu-node-2), idle', lines)
        self.assertIn('  grafana      PASS  https://203.0.113.10/grafana/  user admin  password secret', lines)
        self.assertTrue(any(line.startswith('  data         PASS  tokens-2048-160: 160 documents x 2,048 positions, '
                                            'fingerprint ') for line in lines))
        self.assertEqual(status['image'], ['PASS'])
        self.assertEqual(status['data'], ['PASS'] * 3)
        self.assertEqual(status['model'], ['PASS'])
        self.assertEqual(status['monitoring'], ['PASS'])
        # The temporary directory is not on Lustre, so only the FSx check fails.
        self.assertEqual(status['fsx'], ['FAIL'])
        self.assertEqual(failures, 1)
        self.assertEqual(lines[-1], '1 checks failed. Fix the first FAIL line before continuing.')


class ProvisioningTests(unittest.TestCase):
    def test_environment_file(self):
        text = workshop.env_file_text(stage_dir='/fsx/aim347/participant',
                                      companion='/fsx/aim347/participant/companion/examples/use-cases/ai-systems-performance-engineering',
                                      slurm_version='25.11', cpus_per_node=192, instance_type='g7.48xlarge',
                                      nic='enp71s0', image_sha256='b' * 64, grafana_url='https://203.0.113.10/grafana/',
                                      password_file='/home/ubuntu/.aim347/grafana-admin-password')
        keys = [line.split()[1].split('=')[0] for line in text.splitlines()]
        self.assertEqual(keys, ['AIM347_HOME', 'AIM347_COMPANION', 'PATH', 'PARTITION', 'GPUS_PER_NODE',
                                'AIM347_CPUS_PER_NODE', 'AIM347_INSTANCE_TYPE', 'NCCL_SOCKET_IFNAME', 'LAB_IMAGE',
                                'LAB_IMAGE_SHA256', 'LLM_DATA_DIR', 'LLM_RESULTS_DIR', 'LLM_CONTAINER_NAME',
                                'AIM347_ALLOCATION_LABEL', 'AIM347_GRAFANA_URL', 'AIM347_GRAFANA_PASSWORD_FILE',
                                'AIM347_SKIP_LEGACY_ENV'])
        self.assertIn('export PATH="/opt/aws/pcs/scheduler/slurm-25.11/bin:$PATH"\n', text)
        sourced = subprocess.run(['bash', '-c', 'eval "$1"; echo "$NCCL_SOCKET_IFNAME|$LAB_IMAGE|${PATH%%:*}"', 'bash', text],
                                 capture_output=True, text=True, check=True).stdout.strip()
        self.assertEqual(sourced, '=enp71s0|/fsx/aim347/participant/images/aim347-lab.sqsh|'
                                  '/opt/aws/pcs/scheduler/slurm-25.11/bin')

    def test_prepare_event_refuses_other_instance_types_and_bad_usage(self):
        script = LAB / 'facilitator/prepare-event.sh'
        result = subprocess.run(['bash', str(script), '--stage-dir', '/tmp', '--slurm-version', '25.11',
                                 '--instance-type', 'p5.48xlarge', '--grafana-url', 'https://x/grafana/'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stderr.strip(), 'prepare-event: --instance-type must be g7.48xlarge (got p5.48xlarge)')
        self.assertEqual(subprocess.run(['bash', str(script), '--stage-dir'], capture_output=True).returncode, 2)

    def run_prepare_event(self, directory, curl_failures):
        """prepare-event.sh with stub id, scontrol, findmnt, srun, curl, docker and sleep on PATH.

        The stub srun runs a two-node step once per node, and fails the docker build,
        so the script stops at the image step after the monitoring wait."""
        root = Path(directory)
        bin_dir, stage_dir, calls = root / 'bin', root / 'stage', root / 'calls'
        bin_dir.mkdir()
        stage_dir.mkdir()
        stub(bin_dir, 'id', '[[ $1 == -u ]] && echo 0\nexit 0\n')
        stub(bin_dir, 'scontrol', 'case "$2" in hostnames) echo gpu-node-1; echo gpu-node-2 ;; '
                                  'node) echo "NodeName=$4 NodeAddr=10.0.0.1 CPUEfctv=192" ;; esac\n')
        stub(bin_dir, 'findmnt', 'echo lustre\n')
        stub(bin_dir, 'srun', f'''nodes=1
while [[ $1 == -* ]]; do
  [[ $1 == -N2 ]] && nodes=2
  [[ $1 == -w ]] && shift
  shift
done
[[ $1 == ip ]] && {{ echo '[{{"dev": "enp71s0"}}]'; exit 0; }}
[[ $* == *'docker build'* ]] && {{ echo build >> {calls}; exit 1; }}
status=0
for node in $(seq "$nodes"); do SLURMD_NODENAME=gpu-node-$node "$@" || status=1; done
exit $status
''')
        stub(bin_dir, 'curl', f'''echo "curl $SLURMD_NODENAME" >> {calls}
count=$(grep -c "^curl $SLURMD_NODENAME$" {calls})
((count > {curl_failures}))
''')
        stub(bin_dir, 'docker', f'echo "docker $* $SLURMD_NODENAME" >> {calls}\n')
        stub(bin_dir, 'sleep', 'exit 0\n')
        # Slurm version 0.0 keeps a real Slurm directory, if the host has one, off the front of PATH.
        result = subprocess.run(['bash', str(LAB / 'facilitator/prepare-event.sh'), '--stage-dir', str(stage_dir),
                                 '--slurm-version', '0.0', '--instance-type', 'g7.48xlarge',
                                 '--grafana-url', 'https://203.0.113.10/grafana/'],
                                env=dict(os.environ, PATH=f'{bin_dir}:{os.environ["PATH"]}', SLURM_JOB_ID='42',
                                         SLURM_JOB_NODELIST='gpu-node-[1-2]'),
                                capture_output=True, text=True, timeout=120)
        return result, calls.read_text().splitlines() if calls.exists() else []

    def test_prepare_event_waits_for_the_monitoring_install_before_the_image_build(self):
        with tempfile.TemporaryDirectory() as directory:
            result, calls = self.run_prepare_event(directory, curl_failures=2)
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertIn('prepare-event: checks ok (', result.stdout)
        self.assertIn('prepare-event: monitoring install ok (', result.stdout)
        self.assertIn('building or importing aim347-lab:', result.stderr)
        for node in ('gpu-node-1', 'gpu-node-2'):
            self.assertEqual(calls.count(f'curl {node}'), 3)
            self.assertEqual(calls.count(f'docker info {node}'), 1)
        self.assertEqual(calls[-1], 'build')

    def test_prepare_event_stops_when_the_monitoring_install_does_not_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            result, calls = self.run_prepare_event(directory, curl_failures=1000)
        self.assertEqual(result.returncode, 2)
        self.assertIn('prepare-event: the PCS monitoring install has not finished on gpu-node-1 after 30 minutes; '
                      'read /var/log/amazon/pcs/lifecycle/actions/nodeReady/install-monitoring.log on that node',
                      result.stderr)
        self.assertIn('prepare-event: the PCS monitoring install did not finish on both GPU nodes', result.stderr)
        self.assertEqual(calls.count('curl gpu-node-1'), 180)
        self.assertNotIn('build', calls)

    def test_launcher_names_the_container_only_when_asked(self):
        with tempfile.TemporaryDirectory() as directory:
            bin_dir = Path(directory)
            stub(bin_dir, 'srun', f'printf "%s\\n" "$@" > {directory}/srun-args\n')
            stub(bin_dir, 'scontrol', 'case "$2" in hostnames) echo gpu-node-1; echo gpu-node-2 ;; '
                                      'node) echo "NodeName=gpu-node-1 NodeAddr=10.0.0.1" ;; esac\n')
            env = dict(os.environ, PATH=f'{bin_dir}:{os.environ["PATH"]}', SLURM_JOB_ID='42', SLURM_JOB_NUM_NODES='2',
                       SLURM_JOB_NODELIST='gpu-node-[1-2]', SLURM_JOB_CPUS_PER_NODE='192(x2)', GPUS_PER_NODE='8',
                       LAB_IMAGE='/fsx/image.sqsh', LLM_DATA_DIR='/fsx/data', LLM_RESULTS_DIR='/fsx/results',
                       NCCL_SOCKET_IFNAME='=enp71s0', LLM_TRANSPORT='efa')
            for name, expected in (('aim347-lab', ['--container-name=aim347-lab']), ('', [])):
                with self.subTest(name=name):
                    subprocess.run(['bash', str(LAB / '2.run-llm.sh'), '--config', 'configs/smol-2048.json'],
                                   env=dict(env, LLM_CONTAINER_NAME=name), check=True, capture_output=True)
                    arguments = (Path(directory) / 'srun-args').read_text().splitlines()
                    self.assertEqual([a for a in arguments if a.startswith('--container-name')], expected)
                    self.assertEqual(arguments[-2:], ['--config', 'configs/smol-2048.json'])

    def test_shell_syntax_of_every_new_script(self):
        scripts = sorted((LAB / 'tools').glob('*.sh')) + [LAB / 'facilitator/prepare-event.sh',
                                                           LAB / 'facilitator/install-node-collectors.sh',
                                                           LAB / '2.run-llm.sh', LAB / 'lib/nccl-protocols-rank.sh']
        self.assertEqual(len(scripts), 10)
        for script in scripts:
            with self.subTest(script=script.name):
                subprocess.run(['bash', '-n', str(script)], check=True)
        if shutil.which('shellcheck'):
            subprocess.run(['shellcheck', '-x', '-P', 'SCRIPTDIR/..', *map(str, scripts)], check=True, cwd=LAB)


class NcclProtocolSettingTests(unittest.TestCase):
    """NCCL_PROTO=Simple wherever the launchers set OFI_NCCL_PROTOCOL=RDMA for G7."""

    def protocol_environment(self, instance_type):
        script = (f'source {LAB / "lib/instance-type.sh"}\n'
                  f'detect_instance_type() {{ printf "%s\\n" {instance_type}; }}\n'
                  'apply_g7_protocol\n'
                  'printf "OFI_NCCL_PROTOCOL=%s NCCL_PROTO=%s\\n" "${OFI_NCCL_PROTOCOL-unset}" "${NCCL_PROTO-unset}"\n')
        # stdin from /dev/null: bash reads the bashrc files when its input is a socket.
        result = subprocess.run(['bash', '-euc', script], capture_output=True, text=True, stdin=subprocess.DEVNULL,
                                env={'PATH': os.environ['PATH'], 'NCCL_PROTO': '^LL'}, check=True)
        return result.stderr, result.stdout.strip()

    def test_g7_selects_rdma_and_simple_and_says_so(self):
        stderr, values = self.protocol_environment('g7.48xlarge')
        self.assertEqual(values, 'OFI_NCCL_PROTOCOL=RDMA NCCL_PROTO=Simple')
        self.assertRegex(stderr, r'^host=\S+ instance_type=g7\.48xlarge OFI_NCCL_PROTOCOL=RDMA NCCL_PROTO=Simple\n$')
        stderr, values = self.protocol_environment('p5.48xlarge')
        self.assertEqual((stderr, values), ('', 'OFI_NCCL_PROTOCOL=unset NCCL_PROTO=^LL'))

    def test_legacy_job_passes_nccl_proto_into_every_container_that_gets_the_protocol(self):
        lines = [line for line in (LAB / 'lib/job.sh').read_text().splitlines() if '--container-env=' in line]
        self.assertEqual(len(lines), 2)
        for line in lines:
            names = line.split('--container-env=', 1)[1].split()[0].split(',')
            self.assertIn('OFI_NCCL_PROTOCOL', names)
            self.assertIn('NCCL_PROTO', names)

    def test_training_launcher_sets_simple_for_efa_unless_the_caller_sets_nccl_proto(self):
        with tempfile.TemporaryDirectory() as directory:
            bin_dir = Path(directory)
            stub(bin_dir, 'srun', 'printf "%s\\n" "OFI_NCCL_PROTOCOL=${OFI_NCCL_PROTOCOL-unset}" '
                                  '"NCCL_PROTO=${NCCL_PROTO-unset}" "NCCL_DEBUG_SUBSYS=$NCCL_DEBUG_SUBSYS" "$@"\n')
            stub(bin_dir, 'scontrol', 'case "$2" in hostnames) echo gpu-node-1; echo gpu-node-2 ;; '
                                      'node) echo "NodeName=gpu-node-1 NodeAddr=10.0.0.1" ;; esac\n')
            env = dict(os.environ, PATH=f'{bin_dir}:{os.environ["PATH"]}', SLURM_JOB_ID='42', SLURM_JOB_NUM_NODES='2',
                       SLURM_JOB_NODELIST='gpu-node-[1-2]', SLURM_JOB_CPUS_PER_NODE='192(x2)', GPUS_PER_NODE='8',
                       LAB_IMAGE='/fsx/image.sqsh', LLM_DATA_DIR='/fsx/data', LLM_RESULTS_DIR='/fsx/results',
                       NCCL_SOCKET_IFNAME='=enp71s0')
            env.pop('NCCL_PROTO', None)
            # A caller's NCCL_PROTO is kept, and an empty one is unset so that NCCL chooses.
            for transport, caller, proto in (('efa', None, 'Simple'), ('socket', None, 'unset'),
                                             ('efa', 'LL', 'LL'), ('efa', '', 'unset')):
                with self.subTest(transport=transport, caller=caller):
                    run_env = dict(env, LLM_TRANSPORT=transport)
                    if caller is not None:
                        run_env['NCCL_PROTO'] = caller
                    output = subprocess.run(['bash', str(LAB / '2.run-llm.sh')], env=run_env,
                                            check=True, capture_output=True, text=True).stdout.splitlines()
                    self.assertEqual(output[:3], ['OFI_NCCL_PROTOCOL=RDMA', f'NCCL_PROTO={proto}',
                                                  'NCCL_DEBUG_SUBSYS=INIT,NET,ENV'])
                    names = next(a for a in output if a.startswith('--container-env=')).split('=', 1)[1].split(',')
                    self.assertIn('NCCL_PROTO', names)


class NcclProtocolToolTests(unittest.TestCase):
    """tools/nccl-protocols.sh: the srun step, the rank script and the report, with a fake srun."""

    ENV = dict(GPUS_PER_NODE='8', AIM347_CPUS_PER_NODE='192', LAB_IMAGE='/fsx/aim347/images/aim347-lab.sqsh',
               LLM_CONTAINER_NAME='aim347-lab', LLM_RESULTS_DIR='/fsx/aim347/results')
    # Synthetic rows in the nccl-tests 2.18.3 layout: size, count, type, redop, root, then
    # time, algbw, busbw and #wrong out of place and in place.
    TABLES = {
        'nccl-choice': [(1048576, 6593.2, 0.16, 0.30), (2097152, 15322.0, 0.14, 0.26),
                        (16777216, 1007.5, 16.65, 31.22)],
        'simple': [(1048576, 484.6, 2.16, 4.06), (2097152, 846.0, 2.48, 4.65), (16777216, 1007.0, 16.66, 31.24)],
    }
    CHOICES = {'nccl-choice': {1048576: 'RING LL', 2097152: 'RING LL', 16777216: 'RING SIMPLE'},
               'simple': {1048576: 'RING SIMPLE', 2097152: 'RING SIMPLE', 16777216: 'RING SIMPLE'}}

    def write_run(self, directory, variant, wrong=0):
        directory.mkdir(parents=True, exist_ok=True)
        rows = [f'{size:>12}{size // 4:>14}     float     sum      -1{time:>9.1f}{algbw:>8.2f}{busbw:>8.2f}'
                f'{wrong:>8}{time:>9.1f}{algbw:>8.2f}{busbw:>8.2f}{0:>8}' for size, time, algbw, busbw in self.TABLES[variant]]
        (directory / 'all_reduce_perf.txt').write_text(
            '# nccl-tests version 2.18.3 nccl-headers=23004 nccl-library=23004\n'
            '#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong\n'
            + '\n'.join(rows) + '\n# Out of bounds values : 0 OK\n')
        log = with_nccl_proto(nccl_log('efa')) if variant == 'simple' else nccl_log('efa')
        for size, choice in self.CHOICES[variant].items():
            algo, proto = choice.split()
            log += f'gpu-node-1:1769341:1769341 [0] NCCL INFO AllReduce: {size} Bytes -> Algo {algo} proto {proto} channel{{Lo..Hi}}={{0..3}}\n' * 3
        (directory / 'nccl-gpu-node-1-1769341.log').write_text(log)

    def test_srun_step_runs_one_rank_per_gpu_in_the_training_image(self):
        command = workshop.protocol_command('simple', Path('/fsx/aim347/results/nccl-protocols/T'), self.ENV)
        self.assertEqual(command[:9], ['srun', '--nodes=2', '--ntasks=16', '--ntasks-per-node=8', '--cpus-per-task=24',
                                       '--mpi=pmix', '--cpu-bind=none', '--time=10',
                                       '--container-image=/fsx/aim347/images/aim347-lab.sqsh'])
        self.assertIn('--container-name=aim347-lab', command)
        self.assertIn(f'--container-mounts={workshop.LAB}:/opt/aim347,/fsx/aim347/results:/fsx/aim347/results', command)
        self.assertIn('--container-env=NCCL_SOCKET_IFNAME', command)
        self.assertEqual(command[-4:], ['bash', 'lib/nccl-protocols-rank.sh', 'simple',
                                        '/fsx/aim347/results/nccl-protocols/T/simple'])

    def test_rank_script_differs_only_in_nccl_proto(self):
        with tempfile.TemporaryDirectory() as directory:
            stub(Path(directory), 'env', 'for name in NCCL_PROTO OFI_NCCL_PROTOCOL NCCL_NET FI_PROVIDER NCCL_DEBUG_SUBSYS '
                                         'NCCL_DEBUG_FILE OMPI_MCA_btl_tcp_if_include PMIX_MCA_gds; do\n'
                                         '  printf "%s=%s\\n" "$name" "${!name-unset}"; done; printf "%s\\n" "$@"\n')
            env = dict(os.environ, PATH=f'{directory}:{os.environ["PATH"]}', NCCL_SOCKET_IFNAME='=enp71s0',
                       NCCL_PROTO='LL')
            outputs = {}
            for variant in ('nccl-choice', 'simple'):
                outputs[variant] = subprocess.run(['bash', str(LAB / 'lib/nccl-protocols-rank.sh'), variant, '/logs'],
                                                  env=env, check=True, capture_output=True, text=True,
                                                  stdin=subprocess.DEVNULL).stdout.splitlines()
            self.assertEqual(outputs['nccl-choice'][0], 'NCCL_PROTO=unset')
            self.assertEqual(outputs['simple'][0], 'NCCL_PROTO=Simple')
            self.assertEqual(outputs['nccl-choice'][1:], outputs['simple'][1:])
            self.assertEqual(outputs['simple'][1:8], ['OFI_NCCL_PROTOCOL=RDMA', 'NCCL_NET=AWS Libfabric', 'FI_PROVIDER=efa',
                                                      'NCCL_DEBUG_SUBSYS=INIT,NET,ENV,TUNING',
                                                      'NCCL_DEBUG_FILE=/logs/nccl-%h-%p.log',
                                                      'OMPI_MCA_btl_tcp_if_include=enp71s0', 'PMIX_MCA_gds=hash'])
            self.assertEqual(outputs['simple'][8:], ['LD_PRELOAD=/opt/nccl/build/lib/libnccl.so',
                                                     '/opt/nccl-tests/build/all_reduce_perf', '-b', '1M', '-e', '64M',
                                                     '-f', '2', '-g', '1', '-w', '5', '-n', '20', '-c', '1'])
            result = subprocess.run(['bash', str(LAB / 'lib/nccl-protocols-rank.sh'), 'll', '/logs'], env=env,
                                    capture_output=True, text=True, stdin=subprocess.DEVNULL)
            self.assertEqual(result.returncode, 2)

    def test_report_puts_both_runs_side_by_side(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            for variant in ('nccl-choice', 'simple'):
                self.write_run(out / variant, variant)
            lines, code = workshop.protocol_report(out, 16)
            self.assertEqual(code, 0)
            self.assertEqual(lines[:5], [
                '            NCCL_PROTO not set                      NCCL_PROTO=Simple',
                '     size   time (us)  busbw (GB/s)  NCCL ran       time (us)  busbw (GB/s)  NCCL ran',
                '    1 MiB      6593.2          0.30  RING/LL            484.6          4.06  RING/SIMPLE',
                '    2 MiB     15322.0          0.26  RING/LL            846.0          4.65  RING/SIMPLE',
                '   16 MiB      1007.5         31.22  RING/SIMPLE       1007.0         31.24  RING/SIMPLE'])
            self.assertEqual(lines[5], '[aim347] NCCL_PROTO in the NCCL logs: not set on any rank (NCCL_PROTO not set); '
                                       'Simple on 16 of 16 ranks (NCCL_PROTO=Simple)')
            self.assertEqual(lines[6], '[aim347] NCCL logs, with the algorithm and protocol of every collective on rank 0: '
                                       f'{out / "nccl-choice"}, {out / "simple"}')
            self.write_run(out / 'simple', 'simple', wrong=3)
            lines, code = workshop.protocol_report(out, 16)
            self.assertEqual((code, lines[-1]), (5, '[aim347] wrong values: NCCL_PROTO=Simple at 1 MiB, 2 MiB, 16 MiB'))
            (out / 'simple/all_reduce_perf.txt').write_text('')
            lines, code = workshop.protocol_report(out, 16)
            self.assertEqual((code, lines), (4, [f'[aim347] no all_reduce_perf table for NCCL_PROTO=Simple in {out}']))

    def test_tool_runs_both_steps_and_refuses_while_a_stage_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            results, state = Path(directory) / 'results', Path(directory) / 'state'
            state.mkdir()
            env = dict(self.ENV, LLM_RESULTS_DIR=str(results))
            launched = []

            def runner(command, stdout, stderr):
                launched.append(command)
                self.write_run(Path(command[-1]), command[-2])
                stdout.write(Path(command[-1], 'all_reduce_perf.txt').read_text())
                return subprocess.CompletedProcess(command, 0)
            output = io.StringIO()
            with redirect_stdout(output):
                code = workshop.run_protocols(state, runner=runner, env=env)
            self.assertEqual(code, 0)
            self.assertEqual([command[-2] for command in launched], ['nccl-choice', 'simple'])
            text = output.getvalue().splitlines()
            self.assertEqual(text[0], '[aim347] all_reduce_perf, 1 MiB to 64 MiB, on 16 GPUs over EFA with OFI_NCCL_PROTOCOL=RDMA')
            self.assertEqual(text[2:4], ['[aim347] run 1 of 2: NCCL_PROTO not set', '[aim347] run 2 of 2: NCCL_PROTO=Simple'])
            self.assertIn('    1 MiB      6593.2          0.30  RING/LL            484.6          4.06  RING/SIMPLE', text)

            def failing(command, stdout, stderr):
                stdout.write('srun: error: launch failed\n')
                return subprocess.CompletedProcess(command, 1)
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(workshop.run_protocols(state, runner=failing, env=env), 4)
            self.assertIn('srun: error: launch failed', output.getvalue())

            (state / 'running').write_text(json.dumps(dict(invocation='i', stages=['efa'], pid=os.getpid())))
            errors = io.StringIO()
            with redirect_stderr(errors):
                self.assertEqual(workshop.run_protocols(state, runner=runner, env=env), 2)
            self.assertIn('a stage is still running', errors.getvalue())


if __name__ == '__main__':
    unittest.main()
