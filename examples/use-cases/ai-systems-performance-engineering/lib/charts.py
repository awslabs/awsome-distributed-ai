"""Bar charts as text, for the participant's terminal. Standard library only.

The Session Manager terminal in the browser cannot open image files, so the
workshop tools print their charts with Unicode block characters, at most WIDTH
columns wide, under the result they belong to. A chart draws only measured
values: a row without a value prints its reason in place of the bar.
"""
import math

WIDTH = 100
EIGHTHS = ' ▏▎▍▌▋▊▉'
FULL = '█'
# Fills of the parts of a stacked bar, in the order the parts are drawn. Each of the first
# three differs from the next by at least two shade steps (█ to ░, ░ to ▓), so that
# neighbouring parts stay apart on a light background too.
FILLS = ('█', '░', '▓', '▒')
LOST = '·'


def nice_scale(maximum, cells):
    """The smallest of 1, 2, 2.5 and 5 times a power of ten that fits maximum into cells."""
    if maximum <= 0 or cells < 1:
        return 1.0
    exponent = math.floor(math.log10(maximum / cells))
    for power in (exponent, exponent + 1):
        for step in (1, 2, 2.5, 5):
            scale = step * 10 ** power
            if maximum / scale <= cells + 1e-9:
                return scale
    return 10 ** (exponent + 2)


def scale_text(scale):
    return f'{scale:g}'


def solid(value, scale):
    """One bar in full blocks and an eighth block for the remainder."""
    eighths = max(0, round(value / scale * 8))
    return FULL * (eighths // 8) + (EIGHTHS[eighths % 8] if eighths % 8 else '')


def stacked(segments, scale):
    """Segments [(value, fill), ...] in whole cells: the bar has the rounded total, and the
    cells left after each segment's whole part go to the largest remainders."""
    shares = [max(0.0, value) / scale for value, _ in segments]
    cells = [math.floor(share) for share in shares]
    spare = round(sum(shares)) - sum(cells)
    for index in sorted(range(len(shares)), key=lambda i: cells[i] - shares[i])[:max(0, spare)]:
        cells[index] += 1
    return ''.join(fill * count for (_, fill), count in zip(segments, cells))


def layout(rows):
    """(label width, bar cells) so that a row fits in WIDTH columns."""
    label = max(len(row['label']) for row in rows)
    text = max([len(row.get('text') or '') for row in rows] + [0])
    return label, max(20, WIDTH - 2 - label - 2 - 2 - text)


def row_line(label, label_width, bar, cells, text):
    return f'  {label:<{label_width}}  {bar:<{cells}}  {text}'.rstrip()


def key_lines(indent, entries):
    """Legend entries after an indent, two spaces apart, wrapped at WIDTH."""
    lines, line = [], ''
    for entry in entries:
        if line and indent + len(line) + 3 + len(entry) > WIDTH:
            lines.append(' ' * indent + line)
            line = ''
        line = f'{line}   {entry}' if line else entry
    return lines + ([' ' * indent + line] if line else [])


def bar_chart(title, rows, unit, notes=()):
    """Solid bars. rows: dicts with label, value (or None), text, and note when value is None."""
    label_width, cells = layout(rows)
    maximum = max([row['value'] for row in rows if row.get('value') is not None] + [0])
    scale = nice_scale(maximum, cells)
    lines = [title]
    for row in rows:
        if row.get('value') is None:
            lines.append(row_line(row['label'], label_width, row['note'], 0, ''))
        else:
            lines.append(row_line(row['label'], label_width, solid(row['value'], scale), cells, row['text']))
    if maximum:
        lines += key_lines(label_width + 4, [f'one cell = {scale_text(scale)} {unit}'])
    return lines + [f'  {note}' for note in notes]


def stacked_chart(title, rows, legend, unit, minimum_scale=0.0):
    """Stacked bars on one scale. rows: dicts with label, segments [(value, fill)], text and
    detail (a line under the bar), or note when the row has no bar. legend: [(fill, name)],
    printed before the scale; empty when each detail line names its segments. minimum_scale
    keeps one value per cell while the bars fit, so that charts of different runs compare."""
    label_width, cells = layout(rows)
    maximum = max([sum(max(0.0, v) for v, _ in row['segments']) for row in rows if row.get('segments')] + [0])
    scale = max(minimum_scale, nice_scale(maximum, cells))
    lines = [title]
    for row in rows:
        if not row.get('segments'):
            lines.append(row_line(row['label'], label_width, row['note'], 0, ''))
            continue
        lines.append(row_line(row['label'], label_width, stacked(row['segments'], scale), cells, row['text']))
        if row.get('detail'):
            lines.append(f"  {'':<{label_width}}  {row['detail']}")
    return lines + key_lines(label_width + 4, [f'{fill} {name}' for fill, name in legend]
                             + [f'one cell = {scale_text(scale)} {unit}'])


def cumulative_chart(title, rows):
    """Each bar is the bar before it plus its own change, in times the first row.

    rows: dicts with label, change (the name of the step that this row adds), total
    (times the first row, None without a measured pair) and note (why there is no total).
    A step that made the job slower marks the part it lost with LOST. After a row
    without a total, the next bar's segment holds every step since the last bar.
    """
    label_width = max(len(row['label']) for row in rows)
    totals = [row['total'] for row in rows if row.get('total') is not None]
    texts, legend = [], []
    previous, pending = None, []
    for row in rows:
        if row.get('total') is None:
            texts.append(None)
            pending.append(row['change'])
            continue
        if previous is None:
            texts.append(f"{row['total']:.2f}x")
            legend.append((FILLS[0], f"{row['change']} {row['total']:.2f}x"))
        else:
            step = row['total'] - previous
            name = ' + '.join(pending + [row['change']])
            texts.append(f"{row['total']:.2f}x  ({name} {step:+.2f}x{', slower' if step < 0 else ''})")
            legend.append((LOST, f'{name} {step:+.2f}x, lost against the bar before') if step < 0
                          else (FILLS[len(legend) % len(FILLS)], f'{name} {step:+.2f}x'))
        previous, pending = row['total'], []
    text_width = max([len(text) for text in texts if text] + [0])
    cells = max(20, WIDTH - 2 - label_width - 2 - 2 - text_width)
    # Ten cells for 1.00x while the bars fit, so that the chart keeps its scale as it grows.
    scale = max(0.1, nice_scale(max(totals + [0]), cells))
    lines = [title]
    kept, fill_index = None, 0
    for row, text in zip(rows, texts):
        if text is None:
            lines.append(row_line(row['label'], label_width, row['note'], 0, ''))
            continue
        count = round(row['total'] / scale)
        if kept is None:
            kept, shown = [FILLS[0]] * count, FILLS[0] * count
        elif count >= len(kept):
            fill_index += 1
            kept = kept + [FILLS[fill_index % len(FILLS)]] * (count - len(kept))
            shown = ''.join(kept)
        else:
            fill_index += 1
            shown = ''.join(kept[:count]) + LOST * (len(kept) - count)
            kept = kept[:count]
        lines.append(row_line(row['label'], label_width, shown, cells, text))
    if not totals:
        return lines
    return lines + key_lines(label_width + 4, [f'{fill} {name}' for fill, name in legend]
                             + [f'one cell = {scale_text(scale)}x'])
