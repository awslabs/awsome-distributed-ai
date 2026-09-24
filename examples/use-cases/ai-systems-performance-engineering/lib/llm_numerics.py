"""Compare saved full FSDP shards after short correctness-only executions."""
import json
import math
from pathlib import Path


def tensor_differences(before, after):
    """Accumulate in float64, bounded to one tensor chunk at a time on CPU."""
    import torch
    if before.keys() != after.keys():
        raise ValueError('parameter names differ')
    sums = dict(reference_squared=0.0, actual_squared=0.0, error_squared=0.0, dot=0.0)
    maximum = 0.0
    elements = 0
    for name, reference in before.items():
        actual = after[name]
        if reference.shape != actual.shape or reference.dtype != actual.dtype:
            raise ValueError(f'tensor layout differs: {name}')
        reference, actual = reference.reshape(-1), actual.reshape(-1)
        for start in range(0, reference.numel(), 1048576):
            left = reference[start:start + 1048576].to(torch.float64)
            right = actual[start:start + 1048576].to(torch.float64)
            if not (torch.isfinite(left).all() and torch.isfinite(right).all()):
                raise ValueError(f'nonfinite numerical state: {name}')
            difference = right - left
            sums['reference_squared'] += left.dot(left).item()
            sums['actual_squared'] += right.dot(right).item()
            sums['error_squared'] += difference.dot(difference).item()
            sums['dot'] += left.dot(right).item()
            maximum = max(maximum, difference.abs().max().item())
            elements += left.numel()
    return dict(sums, max_absolute_error=maximum, elements=elements)


def numerical_report(before, after, policy):
    import torch
    before, after = Path(before), Path(after)
    left = json.loads((before / 'summary.json').read_text())
    right = json.loads((after / 'summary.json').read_text())
    for key in ('workload', 'allocation', 'completed_updates', 'initial_progress_tokens'):
        if left[key] != right[key]:
            raise ValueError(f'numerical comparison changed {key}')
    if left['configuration'].get('deterministic', False) != right['configuration'].get('deterministic', False):
        raise ValueError('numerical comparison changed determinism policy')
    for row in (left, right):
        if row['status'] != 'completed' or not row['numerical_verification'] or row['profiled']:
            raise ValueError('requires completed correctness-only runs')
    world = left['allocation']['dp_size']
    if type(world) is not int or world < 1:
        raise ValueError('numerical comparison requires positive rank count')

    totals = {}
    for rank in range(world):
        reference = torch.load(before / f'numerics-rank-{rank}.pt', map_location='cpu', weights_only=True)
        actual = torch.load(after / f'numerics-rank-{rank}.pt', map_location='cpu', weights_only=True)
        if set(reference) != {'parameters', 'gradients', 'parameter_deltas'} or reference.keys() != actual.keys():
            raise ValueError('missing numerical state')
        for state in (reference, actual):
            if any(values.keys() != state['parameters'].keys() for values in state.values()):
                raise ValueError('numerical components must cover every trainable parameter')
        for component in reference:
            row = tensor_differences(reference[component], actual[component])
            total = totals.setdefault(component, {key: 0 for key in row})
            for key, value in row.items():
                total[key] = max(total[key], value) if key == 'max_absolute_error' else total[key] + value
        del reference, actual
    for component, row in totals.items():
        if not row['elements']:
            raise ValueError(f'empty numerical component: {component}')
        row['relative_l2_error'] = math.sqrt(row['error_squared'] / row['reference_squared']) if row['reference_squared'] else (0.0 if not row['error_squared'] else math.inf)
        denominator = math.sqrt(row['reference_squared'] * row['actual_squared'])
        row['cosine_similarity'] = row['dot'] / denominator if denominator else (1.0 if not row['error_squared'] else 0.0)
        row['passed'] = (row['relative_l2_error'] <= policy[component]['maximum_relative_l2_error']
                         and row['cosine_similarity'] >= policy[component]['minimum_cosine_similarity'])
    # Every rank logs the same reduced loss; check all ranks still agree.
    max_loss_error = 0.0
    rank_zero_rows = None
    expected_updates = list(range(1, left['completed_updates'] + 1))
    if left['completed_updates'] != left['workload']['updates']:
        raise ValueError('completed updates must match the declared workload')
    if left['initial_progress_tokens'] != 0:
        raise ValueError('numerical comparison requires fresh runs from the fixed initial state')
    for rank in range(world):
        rows = [[json.loads(line) for line in (root / f'rank-{rank}-updates.jsonl').read_text().splitlines()] for root in (before, after)]
        if any([row['update'] for row in run] != expected_updates for run in rows) or not expected_updates:
            raise ValueError('update evidence must cover every expected update')
        if rank_zero_rows is None:
            rank_zero_rows = rows
        for run, rank_zero in zip(rows, rank_zero_rows):
            if any(row[key] != zero[key] for row, zero in zip(run, rank_zero)
                   for key in ('global_mean_loss', 'global_useful_tokens')):
                raise ValueError('reduced loss or global token count differs across ranks')
        for original, changed in zip(*rows):
            for key in ('update', 'global_useful_tokens', 'local_useful_tokens'):
                if original[key] != changed[key]:
                    raise ValueError(f'update membership/accounting differs: {key}')
            loss = original['global_mean_loss']
            error = abs(changed['global_mean_loss'] - loss) / max(abs(loss), 1e-12)
            if not math.isfinite(error):
                raise ValueError('nonfinite loss difference')
            max_loss_error = max(max_loss_error, error)
    return dict(passed=all(row['passed'] for row in totals.values()) and max_loss_error <= policy['maximum_loss_relative_error'],
                tensors=totals, maximum_loss_relative_error=max_loss_error, policy=policy,
                scope='full saved local shards, final-update gradients and initial-to-final parameter deltas; short-run equivalence, not convergence')
