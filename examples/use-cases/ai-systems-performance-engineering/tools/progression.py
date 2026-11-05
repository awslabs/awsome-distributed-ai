#!/usr/bin/env python3
"""The table of all your runs, in the order of the workshop's main path.

Usage: python3 tools/progression.py [--markdown]

Reads every stage directory under $LLM_RESULTS_DIR except profile-*, restore
and attempts/. A ratio is shown only for a pair of runs of the same work on
the same allocation; otherwise the cell refers to a note below the table.
Below the table and its notes it prints two charts: the speed-up of the main
path over baseline, step by step, and the training goodput of the saving runs.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
import workshop  # noqa: E402
from llm_metrics import paired_training_report  # noqa: E402

COLUMNS = ('Stage', 'Transport', 'Attention', 'Saving', 'Work', 'Seconds', 'Tokens/s', 'GPU-s per 1M tokens',
           'MFU', 'Share of your GEMM peak', 'Peak memory GiB', 'vs baseline', 'vs previous step', 'Goodput')
PREVIOUS = {'efa': 'baseline', 'sdpa': 'efa', 'sdpa-socket': 'baseline'}
# Optional A: each try-<change> and its rerun follow the main path, compared with stage sdpa.
TRY_ORDER = tuple(f'{prefix}-{change}' for change in workshop.TRY_CHANGES for prefix in ('try', 'rerun'))
PREVIOUS.update({name: 'sdpa' for name in TRY_ORDER})


def stage_names(results):
    names = [path.parent.name for path in Path(results).glob('*/run.json')
             if not path.parent.name.startswith('profile-') and path.parent.name not in ('restore', 'attempts')]
    known = [name for name in workshop.PROGRESSION_ORDER + TRY_ORDER if name in names]
    return known + sorted(name for name in names if name not in known)


class Notes:
    def __init__(self):
        self.texts = []

    def mark(self, text):
        if text not in self.texts:
            self.texts.append(text)
        return f'({self.texts.index(text) + 1})'


def ratio(before, after, label, notes):
    try:
        return f"{paired_training_report(before, after)['speedup_ratio']:.2f}x"
    except ValueError as error:
        reason = str(error).replace('paired training mismatch: ', 'different ')
        return notes.mark(f'{label}: not compared, {reason}')


def goodput(results, record, progress, gpu, name, notes):
    """The training goodput of a saving stage or interrupt, or a note with the reason it has none."""
    if not progress or 'retained_tokens_per_second' not in progress:
        return notes.mark(f'{name}: no retained progress in its result file, so no goodput')
    try:
        value = workshop.stage_goodput(results, record, progress, gpu)['training_goodput']
    except ValueError as error:
        return notes.mark(f'{name}: goodput not computed, {error}')
    if value > 1:
        return f"{value:.2f} {notes.mark(f'{name}: goodput {value:.2f}, {workshop.ABOVE_ONE}')}"
    return f'{value:.2f}'


def mfu_cell(summary, stage, world, notes, gpus):
    """MFU against the dense BF16 peak of the GPU that summary.json names, or a note with the reason."""
    gpu = workshop.gpu_name(summary)
    if gpu is None:
        return notes.mark(f"{stage['name']}: MFU n/a, summary.json names no single GPU")
    try:
        value = workshop.mfu(summary['training_throughput_tokens_per_second'], stage['sequence_length'], gpu, world)
    except ValueError as error:
        return notes.mark(f"{stage['name']}: MFU n/a, {error}")
    gpus.add(gpu)
    return f'{100 * value:.1f} %'


def denominator_lines(ceilings, gpus, world):
    """The two denominators: of MFU, the dense BF16 peak; of the share, your measured GEMM peak."""
    lines = []
    for gpu in sorted(gpus):
        peak, estimated = workshop.dense_peak(gpu)
        lines.append(f'MFU: the model\'s FLOP per second / ({world} x the dense BF16 peak, {peak:g} TFLOP/s per GPU '
                     f"for {gpu}{', estimated from NVIDIA specifications' if estimated else ''})")
    lines.append('Share of your GEMM peak: the same FLOP per second / '
                 + (f"({world} x your GEMM peak, {ceilings['gemm_tflops_per_gpu']:.1f} TFLOP/s per GPU)" if ceilings
                    else 'n/a (run bash tools/hold-and-measure.sh)'))
    return lines


def collect(results, notes):
    results = Path(results)
    ceilings = workshop.read_ceilings(results)
    world = workshop.world_size()
    summaries, rows, gpus = {}, [], set()
    for name in stage_names(results):
        run = workshop.read_json(results / name / 'run.json') or {}
        if run.get('status') != 'finished':
            notes.mark(f'{name} is still running; run the table again when it has finished')
            continue
        if run.get('exit_status') == 4:
            notes.mark(f'{name} failed, so it has no row; run it again')
            continue
        stage = run['plan']
        row = dict(_name=name, Stage=name, Transport=stage['transport'], Attention=stage['attention'], Saving=stage['saving'],
                   Work=f"{stage['updates']} x {stage['sequence_length']:,}")
        if run.get('exit_status') == 5:
            failed = ', '.join(check['name'] for check in run.get('checks', []) if not check['ok'])
            row['Stage'] += ' ' + notes.mark(f'{name}: a check failed ({failed})')
        if stage['kind'] == 'interrupt':
            campaign = workshop.read_json(results / name / 'campaign' / 'campaign.json') or {}
            once = notes.mark('interrupt: one command that stopped and resumed the job, not one run')
            interval = notes.mark('interrupt: goodput over the whole command, from just before the first launch to the '
                                  'exit of the resumed launch, including both launches and both model loads. The '
                                  'interval of the saving rows runs from the end of setup (model loading, FSDP wrapping, '
                                  'optimizer, checkpoint manager) to the end of the final flush and the closing '
                                  'synchronization across ranks, so the saving rows leave the model load out')
            row.update({'Seconds': f"{campaign.get('elapsed_seconds', 0):.2f}",
                        'vs baseline': once,
                        'Goodput': f"{goodput(results, campaign, campaign, workshop.campaign_gpu(results / name / 'campaign'), name, notes)} {interval}"})
            row['_campaign'] = campaign
            rows.append(row)
            continue
        summary = workshop.read_json(results / name / 'train' / 'summary.json')
        if not summary:
            notes.mark(f'{name} has no summary.json, so it has no row')
            continue
        summaries[name] = summary
        seconds, tokens = summary['measured_seconds'], summary['measured_useful_tokens']
        rate = summary['training_throughput_tokens_per_second']
        row.update({'Seconds': f'{seconds:.2f}', 'Tokens/s': f'{rate:,.0f}',
                    'GPU-s per 1M tokens': f'{workshop.gpu_seconds_per_million_tokens(seconds, tokens, world):.0f}',
                    'MFU': mfu_cell(summary, stage, world, notes, gpus),
                    'Share of your GEMM peak': (
                        f"{100 * workshop.gemm_share(rate, stage['sequence_length'], ceilings['gemm_tflops_per_gpu'], world):.1f} %"
                        if ceilings else 'n/a'),
                    'Peak memory GiB': f'{workshop.peak_memory_gib(summary):.2f}'})
        if stage['saving'] != 'none':
            row['Goodput'] = goodput(results, summary, summary.get('retained_progress'), workshop.gpu_name(summary),
                                     name, notes)
        rows.append(row)
    baseline = summaries.get('baseline')
    for row in rows:
        name = row['_name']
        if name in summaries and baseline:
            row['vs baseline'] = ratio(baseline, summaries[name], f'{name} vs baseline', notes)
        previous = PREVIOUS.get(name)
        if previous and name in summaries and previous in summaries:
            row['vs previous step'] = ratio(summaries[previous], summaries[name], f'{name} vs {previous}', notes)
    return ceilings, rows, denominator_lines(ceilings, gpus, world)


def render(ceilings, rows, notes, markdown, denominators=()):
    cells = [[row.get(column, '') for column in COLUMNS] for row in rows]
    lines = [workshop.ceilings_summary(ceilings), '']
    if markdown:
        lines.append('| ' + ' | '.join(COLUMNS) + ' |')
        lines.append('|' + '|'.join('---' for _ in COLUMNS) + '|')
        lines += ['| ' + ' | '.join(cell) + ' |' for cell in cells]
    else:
        widths = [max([len(column)] + [len(cell[i]) for cell in cells]) for i, column in enumerate(COLUMNS)]
        lines.append('  '.join(column.ljust(width) for column, width in zip(COLUMNS, widths)).rstrip())
        lines += ['  '.join(value.ljust(width) for value, width in zip(cell, widths)).rstrip() for cell in cells]
    for row in rows:
        campaign = row.get('_campaign')
        if campaign:
            lines += ['', f"interrupt: stopped after update {campaign.get('interrupted_update')}, resumed from update "
                          f"{campaign.get('restored_update')}, {campaign.get('replayed_update_count')} updates ran twice"]
    if notes.texts:
        lines.append('')
        lines += [f'({index}) {text}' for index, text in enumerate(notes.texts, 1)]
    if denominators:
        lines += [''] + list(denominators)
    return lines


def chart_lines(results, markdown):
    lines = workshop.cumulative_chart_lines(results) + [''] + workshop.goodput_chart_lines(results)
    return ['', '```text'] + lines + ['```'] if markdown else [''] + lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--markdown', action='store_true', help='print a Markdown table for your notes')
    args = parser.parse_args(argv)
    try:
        results = workshop.results_dir()
    except workshop.UsageError as error:
        print(error, file=sys.stderr)
        return 2
    notes = Notes()
    ceilings, rows, denominators = collect(results, notes)
    print('\n'.join(render(ceilings, rows, notes, args.markdown, denominators) + chart_lines(results, args.markdown)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
