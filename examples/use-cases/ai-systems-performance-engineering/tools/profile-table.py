#!/usr/bin/env python3
"""Read one profiled update of a profile-* stage into a fixed table.

Usage: python3 tools/profile-table.py [STAGE] [--rank N|slowest|all] [--update U]

The window of update u is the CPU span ProfilerStep#<u-1> on the main thread.
Each update ends with a host synchronization, so its GPU work finishes inside
the span. Busy times are unions of intervals over all streams. Below the
table, a chart puts this update beside the same update of the other profile
stages whose tables you have read. Standard library only.
"""
import argparse
import bisect
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
import charts  # noqa: E402
import workshop  # noqa: E402

# Each table read is kept in its stage directory, so that the chart of a later
# profile can show it without parsing that trace again.
TABLES = 'profile-tables.json'

NCCL_PREFIXES = ('ncclDevKernel', 'ncclKernel')
BACKWARD_PREFIX = 'autograd::engine::evaluate_function:'


class NoSpan(LookupError):
    """The trace has no ProfilerStep span for the update (exit 3)."""


def merge(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def length(intervals):
    return sum(end - start for start, end in intervals)


def overlap(first, second):
    """Length of the intersection of two merged interval lists."""
    total, j = 0, 0
    for start, end in first:
        while j < len(second) and second[j][1] <= start:
            j += 1
        k = j
        while k < len(second) and second[k][0] < end:
            total += min(end, second[k][1]) - max(start, second[k][0])
            k += 1
    return total


def clipped(events, low, high):
    return merge((max(e['ts'], low), min(e['ts'] + e['dur'], high)) for e in events
                 if e['ts'] < high and e['ts'] + e['dur'] > low)


class Spans:
    """Non-overlapping CPU spans of one thread, searchable by time."""

    def __init__(self, events):
        self.spans = sorted((e['ts'], e['ts'] + e['dur']) for e in events)
        self.starts = [start for start, _ in self.spans]

    def contains(self, time):
        index = bisect.bisect_right(self.starts, time) - 1
        return index >= 0 and time <= self.spans[index][1]


def complete(events):
    return [e for e in events if e.get('ph') == 'X' and 'dur' in e]


def attention_ids(events, main, low, high):
    """External ids of CPU ops that launch attention-core kernels.

    Forward: ops on the main thread inside an attention/core range. Backward:
    ops inside an autograd function whose sequence number belongs to one of
    those forward ops.
    """
    cpu = [e for e in events if e.get('cat') == 'cpu_op' and e['ts'] < high and e['ts'] + e['dur'] > low]
    ranges = Spans([e for e in events if e.get('name') == 'attention/core' and (e['pid'], e['tid']) == main])
    forward = [op for op in cpu if (op['pid'], op['tid']) == main and ranges.contains(op['ts'])]
    numbers = {op['args']['Sequence number'] for op in forward
               if isinstance(op.get('args', {}).get('Sequence number'), int) and op['args']['Sequence number'] >= 0}
    functions = {}
    for op in cpu:
        if op['name'].startswith(BACKWARD_PREFIX) and op.get('args', {}).get('Sequence number') in numbers:
            functions.setdefault((op['pid'], op['tid']), []).append(op)
    inside = {thread: Spans(ops) for thread, ops in functions.items()}
    ids = {op['args']['External id'] for op in forward if 'External id' in op.get('args', {})}
    for op in cpu:
        spans = inside.get((op['pid'], op['tid']))
        if spans and spans.contains(op['ts']) and 'External id' in op.get('args', {}):
            ids.add(op['args']['External id'])
    return ids


def update_table(events, update, first_update=1):
    """Seconds of each row for one update of one rank's trace."""
    events = complete(events)
    name = f'ProfilerStep#{update - first_update}'
    span = next((e for e in events if e.get('name') == name), None)
    if span is None:
        raise NoSpan(name)
    low, high = span['ts'], span['ts'] + span['dur']
    main = (span['pid'], span['tid'])
    kernels = [e for e in events if e.get('cat') == 'kernel' and e['ts'] < high and e['ts'] + e['dur'] > low]
    copies = [e for e in events if e.get('cat') in ('gpu_memcpy', 'gpu_memset')]
    nccl = [k for k in kernels if k['name'].startswith(NCCL_PREFIXES)]
    compute = [k for k in kernels if not k['name'].startswith(NCCL_PREFIXES)]
    compute_busy, nccl_busy = clipped(compute, low, high), clipped(nccl, low, high)
    waits = [e for e in events if e.get('name') == 'input/wait' and (e['pid'], e['tid']) == main]
    ids = attention_ids(events, main, low, high)
    attention = [k for k in kernels if k.get('args', {}).get('External id') in ids]
    totals = {}
    for kernel in kernels:
        seconds = (min(kernel['ts'] + kernel['dur'], high) - max(kernel['ts'], low)) / 1e6
        count, total = totals.get(kernel['name'], (0, 0.0))
        totals[kernel['name']] = (count + 1, total + seconds)
    top = sorted(totals.items(), key=lambda item: -item[1][1])[:5]
    return dict(update=(high - low) / 1e6,
                compute=length(compute_busy) / 1e6,
                nccl=length(nccl_busy) / 1e6,
                exposed=(length(nccl_busy) - overlap(nccl_busy, compute_busy)) / 1e6,
                idle=(high - low - length(clipped(kernels + copies, low, high))) / 1e6,
                input_wait=length(clipped(waits, low, high)) / 1e6,
                attention=length(clipped(attention, low, high)) / 1e6,
                top=[(name, count, seconds) for name, (count, seconds) in top])


def mean_table(tables):
    rows = {key: sum(t[key] for t in tables) / len(tables) for key in tables[0] if key != 'top'}
    totals = {}
    for table in tables:
        for name, count, seconds in table['top']:
            c, s = totals.get(name, (0, 0.0))
            totals[name] = (c + count, s + seconds)
    rows['top'] = sorted(((name, round(c / len(tables)), s / len(tables)) for name, (c, s) in totals.items()),
                         key=lambda row: -row[2])[:5]
    return rows


def table_lines(title, table):
    def share(seconds):
        return f'{100 * seconds / table["update"]:5.1f} %' if table['update'] else '    - %'
    lines = [title, f"  {'update':<32}{table['update']:.3f} s"]
    for label, key, note in (
            ('compute kernels busy', 'compute', ''),
            ('NCCL kernels busy', 'nccl', ''),
            ('communication not overlapped', 'exposed', '   (NCCL busy while no compute kernel runs)'),
            ('GPU idle', 'idle', '   (no kernel and no copy running)')):
        lines.append(f'  {label:<32}{table[key]:.3f} s   {share(table[key])}{note}')
    lines.append(f"  {'input wait (CPU)':<32}{table['input_wait']:.3f} s            (input/wait ranges)")
    lines.append(f"  {'attention core kernels':<32}{table['attention']:.3f} s   {share(table['attention'])}"
                 '   (forward and backward of attention/core)')
    lines.append('  top 5 kernels by time')
    for name, count, seconds in table['top']:
        lines.append(f'    {name[:60]:<60}   {count:>5}   {seconds:.3f} s')
    return lines


def table_key(rank, update):
    return f'rank {rank}, update {update}'


def save_table(stage_dir, key, title, table):
    tables = workshop.read_json(stage_dir / TABLES, {})
    tables[key] = dict(title=title, table=table)
    try:
        workshop.write_atomic(stage_dir / TABLES, json.dumps(tables, indent=2) + '\n')
    except OSError:
        pass


def breakdown_row(label, table):
    """The update as a partition: attention core, other compute, NCCL alone, idle, copies only."""
    other = table['compute'] - table['attention']
    copies = max(0.0, table['update'] - table['compute'] - table['exposed'] - table['idle'])
    segments = [(table['attention'], charts.FILLS[0]), (other, charts.FILLS[1]),
                (table['exposed'], charts.FILLS[2]), (table['idle'], charts.FILLS[3])]
    detail = (f"{charts.FILLS[0]} {table['attention']:.3f}  {charts.FILLS[1]} {other:.3f}  "
              f"{charts.FILLS[2]} {table['exposed']:.3f}  {charts.FILLS[3]} {table['idle']:.3f}")
    if copies >= 0.0005:
        segments.append((copies, charts.LOST))
        detail += f'  {charts.LOST} {copies:.3f}'
    return dict(label=label, segments=segments, text=f"{table['update']:.3f} s",
                detail=detail + f" s   input wait (CPU) {table['input_wait']:.3f} s")


def chart_lines(results, name, rank, update, table):
    """This stage's update beside the same update of the other profile stages, from their saved tables."""
    key = table_key(rank, update)
    names = list(workshop.PROFILE_ORDER) + sorted(path.parent.name for path in Path(results).glob('profile-*/run.json')
                                                  if path.parent.name not in workshop.PROFILE_ORDER)
    rows = []
    for other in names:
        if other == name:
            rows.append(breakdown_row(other, table))
            continue
        saved = (workshop.read_json(Path(results) / other / TABLES, {}) or {}).get(key)
        if saved:
            rows.append(breakdown_row(other, saved['table']))
        elif (Path(results) / other / 'run.json').is_file():
            option = '' if rank == 'slowest' else f' --rank {rank}'
            rows.append(dict(label=other, note=f'run python3 tools/profile-table.py {other}{option} to add it'))
        else:
            rows.append(dict(label=other, note='not run yet'))
    which = {'slowest': 'the slowest rank of each profile', 'all': 'the mean of all ranks'}.get(rank, f'rank {rank}')
    lines = charts.stacked_chart(f'Where the time of update {update} goes, for {which}', rows, [
        (charts.FILLS[0], 'attention core kernels'), (charts.FILLS[1], 'other compute kernels'),
        (charts.FILLS[2], 'communication not overlapped'), (charts.FILLS[3], 'GPU idle')], 's', minimum_scale=0.1)
    if any(charts.LOST in segment for row in rows for _, segment in row.get('segments', [])):
        lines.append(f'  {charts.LOST} copies only (no kernel running)')
    return lines + ['  input wait (CPU) can overlap the GPU parts, so the bar leaves it out']


def intervals_by_rank(train, update):
    gaps = {}
    for path in workshop.rank_logs(train):
        rank = int(path.name.split('-')[1])
        rows = {row['update']: row['monotonic_seconds'] for row in workshop.read_rows(path)}
        if update in rows and update - 1 in rows:
            gaps[rank] = rows[update] - rows[update - 1]
    return gaps


def latest_profile_stage(results):
    stages = [path.parent for path in Path(results).glob('profile-*/run.json')]
    if not stages:
        return None
    return max(stages, key=lambda stage: (workshop.read_json(stage / 'run.json') or {}).get('start_utc', 0)).name


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('stage', nargs='?', help='a profile-* stage (default: the most recent one)')
    parser.add_argument('--rank', default='slowest', help='N, slowest (default) or all')
    parser.add_argument('--update', type=int, help='the update to read (default: the last one)')
    args = parser.parse_args(argv)
    try:
        results = workshop.results_dir()
    except workshop.UsageError as error:
        print(error, file=sys.stderr)
        return 2
    name = args.stage or latest_profile_stage(results)
    if not name or not (results / name / 'train').is_dir():
        print(f'no profile stage {name or "found"} in {results}. Run: bash tools/run.sh profile', file=sys.stderr)
        return 2
    train = results / name / 'train'
    first_rows = workshop.read_rows(train / 'rank-0-updates.jsonl')
    if not first_rows:
        print(f'{train} has no update rows', file=sys.stderr)
        return 2
    first_update = first_rows[0]['update']
    update = args.update or first_rows[-1]['update']
    if args.rank == 'all':
        ranks = sorted(int(p.name.split('-')[2].split('.')[0]) for p in train.glob('trace-rank-*.json'))
        title = f'Profile of {name}, mean of {len(ranks)} ranks, update {update}'
    elif args.rank == 'slowest':
        gaps = intervals_by_rank(train, update)
        rank = max(gaps, key=gaps.get) if gaps else 0
        ranks = [rank]
        title = (f'Profile of {name}, rank {rank} (slowest for update {update}), update {update}' if gaps
                 else f'Profile of {name}, rank {rank}, update {update}')
    else:
        try:
            ranks = [int(args.rank)]
        except ValueError:
            parser.error('--rank takes a rank number, slowest or all')
        title = f'Profile of {name}, rank {ranks[0]}, update {update}'
    tables = []
    for rank in ranks:
        trace = train / f'trace-rank-{rank}.json'
        if not trace.is_file():
            print(f'trace missing: {trace}', file=sys.stderr)
            return 2
        try:
            tables.append(update_table(json.loads(trace.read_text())['traceEvents'], update, first_update))
        except NoSpan as error:
            print(f'{trace} has no {error} span for update {update}', file=sys.stderr)
            return 3
    if not tables:
        print(f'no traces in {train}', file=sys.stderr)
        return 2
    table = tables[0] if len(tables) == 1 else mean_table(tables)
    save_table(results / name, table_key(args.rank, update), title, table)
    print('\n'.join(table_lines(title, table) + [''] + chart_lines(results, name, args.rank, update, table)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
