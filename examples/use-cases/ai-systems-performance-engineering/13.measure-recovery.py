#!/usr/bin/env python3
"""Measure one fresh interrupted training campaign and its real DCP resume.

Run inside one retained Slurm allocation. The command after -- must launch
lib/train_llm.py on that allocation, with --checkpoint-mode sync or async.
Both subprocesses and the recovery gap use this controller's monotonic clock.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent / 'lib'))
from llm_metrics import training_goodput


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allocated-gpus', type=int, required=True)
    parser.add_argument('--interrupt-after-update', type=int, required=True)
    parser.add_argument('--diagnostic-profile', action='store_true',
                        help='collect phase traces; retain full campaign interval but prohibit performance acceptance')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.allocated_gpus < 0 or args.interrupt_after_update < 1:
        parser.error('provide a training launch command, allocation and interruption index')
    if any(item.split('=')[0] in ('--output', '--resume', '--interrupt-after-update') for item in command):
        parser.error('campaign sets output, resume and interruption arguments')
    if any(item.split('=')[0] in ('--profile', '--verify-numerics') for item in command):
        parser.error('campaign goodput excludes profiler and correctness-snapshot runs')
    if args.diagnostic_profile:
        command = command + ['--profile']
    args.output.mkdir(parents=True, exist_ok=False)
    interrupted = args.output / 'interrupted'
    resumed = args.output / 'resumed'
    events = []
    started = time.perf_counter()
    started_utc = time.time()

    def event(phase, **fields):
        row = dict(phase=phase, elapsed_seconds=time.perf_counter() - started, **fields)
        events.append(row)
        with (args.output / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')

    def launch(phase, extra):
        event(phase + '_start')
        with (args.output / (phase + '.log')).open('w') as log:
            result = subprocess.run(command + extra, stdout=log, stderr=subprocess.STDOUT)
        event(phase + '_exit', returncode=result.returncode)
        return result.returncode

    code = launch('interrupted', ['--output', str(interrupted),
                                  '--interrupt-after-update', str(args.interrupt_after_update)])
    if code == 0:
        raise RuntimeError('expected controlled interruption, but the first launch succeeded')
    # Require the intentional update point on every rank, not an arbitrary launch
    # failure. Pending DCP metadata alone is never a recovery point.
    rank_logs = sorted(interrupted.glob('rank-*-updates.jsonl'))
    if not rank_logs:
        raise RuntimeError('first launch failed before producing update evidence')
    for path in rank_logs:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if (not rows or rows[-1]['update'] != args.interrupt_after_update
                or rows[-1].get('controlled_interruption') is not True):
            raise RuntimeError(f'failure did not reach controlled interruption: {path}')
    markers = sorted((interrupted / 'checkpoints').glob('update-*/COMPLETED.json'))
    if not markers:
        raise RuntimeError('interruption left no completed checkpoint to restore')
    restored = json.loads(markers[-1].read_text())
    expected_rank_logs = {f'rank-{rank}-updates.jsonl'
                          for rank in range(restored['workload']['data_parallel_size'])}
    if {path.name for path in rank_logs} != expected_rank_logs:
        raise RuntimeError('interrupted segment lacks evidence from every rank')
    event('recovery_selected', checkpoint=str(markers[-1].parent),
          durable_update=restored['completed_updates'], useful_tokens=restored['useful_tokens'])
    code = launch('resumed', ['--output', str(resumed), '--resume', str(markers[-1].parent)])
    if code:
        raise RuntimeError('resume failed; campaign logs retained without success credit')
    elapsed = time.perf_counter() - started
    finished_utc = time.time()
    summary = json.loads((resumed / 'summary.json').read_text())
    if (summary['status'] != 'completed' or summary.get('profiled') != args.diagnostic_profile
            or summary.get('numerical_verification')):
        raise RuntimeError('campaign requires completed training with the requested profiling policy and no numerical snapshots')
    final_markers = sorted((resumed / 'checkpoints').glob('update-*/COMPLETED.json'))
    if not final_markers:
        raise RuntimeError('resume did not publish a final durable checkpoint')
    final = json.loads(final_markers[-1].read_text())
    if (final['completed_updates'] != summary['completed_updates']
            or final['completed_updates'] <= args.interrupt_after_update
            or final['workload'] != restored['workload']
            or final['useful_tokens'] != summary['final_progress_tokens']):
        raise RuntimeError('final checkpoint lacks new progress or changes the workload')
    if summary['allocation']['gpu_count'] != args.allocated_gpus:
        raise RuntimeError('campaign GPU allocation differs from resumed training')
    if args.diagnostic_profile:
        for segment in (interrupted, resumed):
            for rank in range(restored['workload']['data_parallel_size']):
                for kind in ('setup-trace', 'trace'):
                    trace = segment / f'{kind}-rank-{rank}.json'
                    trace_events = json.loads(trace.read_text())['traceEvents']
                    if not trace_events:
                        raise RuntimeError(f'empty diagnostic trace: {trace}')
    result = training_goodput(
        0, final['useful_tokens'], elapsed, args.allocated_gpus * elapsed,
        scope='fresh campaign launch through final resumed process exit, including initial model load, training, save, failure detection, downtime, child-step relaunch/wait, restore, replay and final durable flush; allocation retained throughout; initial Slurm allocation wait and pre-campaign preparation excluded')
    result.update(status='completed', source='fresh controlled interruption and resume',
                  evidence_kind=summary['evidence_kind'], started_utc_seconds=started_utc,
                  finished_utc_seconds=finished_utc, allocation=summary['allocation'],
                  events=events, profiled=summary['profiled'], numerical_verification=summary['numerical_verification'],
                  workload=summary['workload'], configuration=summary['configuration'],
                  restored_update=restored['completed_updates'],
                  interrupted_update=args.interrupt_after_update,
                  completed_updates=final['completed_updates'],
                  replayed_update_count=args.interrupt_after_update - restored['completed_updates'],
                  sources={'interrupted': str(interrupted), 'resumed': str(resumed)},
                  allocation_scope='same GPUs retained for entire campaign; excludes preparation before campaign launch')
    result['performance_eligible'] = not args.diagnostic_profile
    (args.output / 'campaign.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
