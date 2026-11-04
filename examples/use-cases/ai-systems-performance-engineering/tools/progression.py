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
           'MFU vs your GEMM peak', 'Peak memory GiB', 'vs baseline', 'vs previous step', 'Goodput tokens/s',
           'Goodput M tokens per GPU-hour')
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


def collect(results, notes):
    results = Path(results)
    ceilings = workshop.read_ceilings(results)
    world = workshop.world_size()
    summaries, rows = {}, []
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
            interval = notes.mark('interrupt: goodput over the whole command, from the first launch to the end of the '
                                  'final save, including both launches and both model loads. The goodput interval of '
                                  'the saving rows starts at the end of model loading, so the two are not compared')
            row.update({'Seconds': f"{campaign.get('elapsed_seconds', 0):.2f}",
                        'vs baseline': once,
                        'Goodput tokens/s': f"{campaign.get('training_goodput_tokens_per_second', 0):,.0f} {interval}",
                        'Goodput M tokens per GPU-hour':
                            f"{campaign.get('tokens_per_allocated_gpu_hour', 0) / 1e6:.2f} {interval}"})
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
                    'MFU vs your GEMM peak': (f"{100 * workshop.mfu(rate, stage['sequence_length'], ceilings['gemm_tflops_per_gpu'], world):.1f} %"
                                              if ceilings else 'n/a'),
                    'Peak memory GiB': f'{workshop.peak_memory_gib(summary):.2f}'})
        goodput = summary.get('training_goodput')
        if goodput:
            row['Goodput tokens/s'] = f"{goodput['training_goodput_tokens_per_second']:,.0f}"
            row['Goodput M tokens per GPU-hour'] = f"{goodput['tokens_per_allocated_gpu_hour'] / 1e6:.2f}"
        rows.append(row)
    baseline = summaries.get('baseline')
    for row in rows:
        name = row['_name']
        if name in summaries and baseline:
            row['vs baseline'] = ratio(baseline, summaries[name], f'{name} vs baseline', notes)
        previous = PREVIOUS.get(name)
        if previous and name in summaries and previous in summaries:
            row['vs previous step'] = ratio(summaries[previous], summaries[name], f'{name} vs {previous}', notes)
    return ceilings, rows


def render(ceilings, rows, notes, markdown):
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
    ceilings, rows = collect(results, notes)
    print('\n'.join(render(ceilings, rows, notes, args.markdown) + chart_lines(results, args.markdown)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
