#!/usr/bin/env python3
"""Measure one fresh interrupted training campaign and its real DCP resume.

Run inside one retained Slurm allocation. The command after -- must launch
lib/train_llm.py on that allocation, with --checkpoint-mode sync or process.
Both subprocesses and the recovery gap use this controller's monotonic clock.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent / 'lib'))
from llm_metrics import retained_progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interrupt-after-update', type=int, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('provide a training launch command')
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
    # The first launch must have ended at the stop, not at an earlier error: the last row of
    # rank 0 is the stop update and carries the trainer's mark.
    log = interrupted / 'rank-0-updates.jsonl'
    rows = [json.loads(line) for line in log.read_text().splitlines()] if log.is_file() else []
    if (not rows or rows[-1]['update'] != args.interrupt_after_update
            or rows[-1].get('controlled_interruption') is not True):
        raise RuntimeError(f'the first launch failed before the stop after update {args.interrupt_after_update}')
    # Only a checkpoint with its completion marker is a restart point.
    markers = sorted((interrupted / 'checkpoints').glob('update-*/COMPLETED.json'))
    if not markers:
        raise RuntimeError('interruption left no completed checkpoint to restore')
    restored = json.loads(markers[-1].read_text())
    event('recovery_selected', checkpoint=str(markers[-1].parent),
          durable_update=restored['completed_updates'], useful_tokens=restored['useful_tokens'])
    code = launch('resumed', ['--output', str(resumed), '--resume', str(markers[-1].parent)])
    if code:
        raise RuntimeError('resume failed; campaign logs retained without success credit')
    elapsed = time.perf_counter() - started
    finished_utc = time.time()
    summary = json.loads((resumed / 'summary.json').read_text())
    # The resumed trainer exits 0 only after its final save is complete.
    result = retained_progress(
        0, summary['final_progress_tokens'], elapsed, initial_updates=0,
        durable_updates=summary['completed_updates'],
        scope='from just before the first launch to the exit of the resumed launch, including both launches with their model loads, training, saves, the controlled stop, checkpoint selection, restore, replay and the final flush; allocation retained throughout; initial Slurm allocation wait and pre-campaign preparation excluded')
    result.update(status='completed', source='fresh controlled interruption and resume',
                  evidence_kind=summary['evidence_kind'], started_utc_seconds=started_utc,
                  finished_utc_seconds=finished_utc, allocation=summary['allocation'],
                  events=events, profiled=summary['profiled'],
                  workload=summary['workload'], configuration=summary['configuration'],
                  restored_update=restored['completed_updates'],
                  interrupted_update=args.interrupt_after_update,
                  completed_updates=summary['completed_updates'],
                  replayed_update_count=args.interrupt_after_update - restored['completed_updates'],
                  sources={'interrupted': str(interrupted), 'resumed': str(resumed)},
                  allocation_scope='same GPUs retained for entire campaign; excludes preparation before campaign launch')
    (args.output / 'campaign.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
