"""Judging and loading for this example: the three TPOT rules, the attainment test (default 90%), the qualifying-rate
walk, goodput, the promise-map grid, and reading stored ramp files. Definitions: README sections
"Three TPOT definitions" and "How a qualifying rate and its goodput are computed".
"""
import hashlib
import json
import math
from pathlib import Path

TPOT_METRICS = ("mean", "itl-p99", "worst-itl")
TTFT_METRICS = ("per-call", "p95", "p99")

# Version 2 adds per-call "chunks"; every record carries the key, an empty list on a failed call.
SCHEMA_VERSION = 2

# Neither a rate nor None ("qualified at no rate"). Compare with ==: copies come back from JSON.
UNAVAILABLE = "unavailable (file predates per-chunk recording, or its per-chunk recording is incomplete)"

# A call that never connected says nothing about the stack it was aimed at.
CLIENT_PATH_ERRORS = ("ConnectError",)

# Fixed default axes, so promise maps from different runs line up.
DEFAULT_TTFT_AXIS_MS = tuple(range(500, 5001, 500))
DEFAULT_TPOT_AXIS_MS = tuple(range(10, 101, 10))


def validate_grid_axis(values):
    """De-duplicated so a repeated bound is one column, not a duplicated cell; ValueError lists every bad bound."""
    bad = [v for v in values if not math.isfinite(v) or v <= 0]
    if bad:
        raise ValueError(f"needs finite positive bounds, got {bad}")
    return tuple(sorted(set(values)))


def percentile(values, fraction):
    """Nearest-rank, not interpolated as in vllm bench serve; None values are dropped."""
    values = sorted(x for x in values if x is not None)
    return None if not values else values[max(0, math.ceil(len(values) * fraction) - 1)]


def worst_itl_ms(record):
    """Longest raw gap between consecutive chunks, not divided by token count: a batched chunk
    after a stall is one pause the client waited through."""
    chunks = record.get("chunks") or []
    if len(chunks) < 2:
        return None
    return max(b["t_ms"] - a["t_ms"] for a, b in zip(chunks, chunks[1:]))


def pooled_itls_ms(records):
    """Each gap divided by its chunk's token count, repeated once per token, as
    sglang.bench_serving reports ITL."""
    pooled = []
    for r in records:
        chunks = r.get("chunks") or []
        for previous, current in zip(chunks, chunks[1:]):
            if current["n"] > 0:
                pooled += [(current["t_ms"] - previous["t_ms"]) / current["n"]] * current["n"]
    return pooled or None


def chunks_recorded(records):
    return bool(records) and all("chunks" in r for r in records)


def chunk_data_complete(records):
    """Whether every successful call carries chunks; pooling a partial subset would understate gaps."""
    ok = [r for r in records if r.get("ok")]
    return all(r.get("chunks") for r in ok) if ok else True


def gap_data_available(records):
    """False means itl-p99 and worst-itl report UNAVAILABLE. True with zero gaps (every call failed,
    or single-chunk calls) is a real measurement, not missing data."""
    return chunks_recorded(records) and chunk_data_complete(records)


def raw_gap_stats(records):
    """Descriptive (gap_count, p99 of raw gaps); never used to judge a rate."""
    gaps = []
    for r in records:
        chunks = r.get("chunks") or []
        gaps += [b["t_ms"] - a["t_ms"] for a, b in zip(chunks, chunks[1:])]
    return len(gaps), (percentile(gaps, .99) if gaps else None)


def summarize(records, duration_s, gpus, ttft_ms, tpot_ms, attainment, offered_tasks_s, max_client_lag_ms, tpot_metric="mean", start_skew_ms=None, start_skew_tolerance_ms=500.0, ttft_metric="per-call"):
    # start_skew_ms is an input, not a post-hoc override, so re-scoring reproduces the verdict.
    assert tpot_metric in TPOT_METRICS
    assert ttft_metric in TTFT_METRICS
    ok = [r for r in records if r["ok"]]
    attempts = [r for r in records if r.get("sent", False)]
    lag = max((r["client_launch_lag_ms"] for r in records), default=0)
    unreachable = sum(r.get("error", "").startswith(CLIENT_PATH_ERRORS) for r in records)
    client_valid = lag <= max_client_lag_ms and not unreachable and (start_skew_ms is None or start_skew_ms <= start_skew_tolerance_ms)
    # Without complete chunk data every gap field is None, never a zero.
    available = gap_data_available(records)
    itls = pooled_itls_ms(ok) if available else None
    itl_p99 = percentile(itls, .99) if itls is not None else None
    worsts = [worst_itl_ms(r) for r in ok] if available else [None] * len(ok)
    worsts_present = [w for w in worsts if w is not None]
    ttft_values = [r.get("ttft_ms") for r in ok]
    ttft_p95 = percentile(ttft_values, .95)
    ttft_p99 = percentile(ttft_values, .99)
    if ttft_metric == "per-call":
        ttft_good = [r for r in records if r["ok"] and r.get("ttft_ms") is not None and r["ttft_ms"] <= ttft_ms]
        meets_ttft = True
    else:
        ttft_good = ok
        pooled_ttft = ttft_p95 if ttft_metric == "p95" else ttft_p99
        meets_ttft = pooled_ttft is not None and pooled_ttft <= ttft_ms
    # itl-p99 judges the pooled p99 once per rate; mean and worst-itl judge each call.
    def tpot_ok(r):
        if tpot_metric == "itl-p99":
            return True
        if tpot_metric == "worst-itl":
            w = worst_itl_ms(r) if available else None
            return w is not None and w <= tpot_ms
        return r.get("tpot_ms") is not None and r["tpot_ms"] <= tpot_ms
    good = [r for r in ttft_good if tpot_ok(r)]
    hit = len(good) / len(records) if records else 0.0
    meets_tpot = tpot_metric in ("mean", "worst-itl") or (itl_p99 is not None and itl_p99 <= tpot_ms)
    meets_joint = hit >= attainment and meets_ttft and meets_tpot
    # Descriptive only; gap_above_bound_fraction is a share of completed calls, not planned ones.
    gap_count, gap_p99 = raw_gap_stats(ok) if available else (None, None)
    gap_above_bound = (sum(1 for w in worsts_present if w > tpot_ms) / len(ok)) if available and ok else None
    worst_itl_p50 = percentile(worsts, .5) if worsts_present else None
    worst_itl_p90 = percentile(worsts, .9) if worsts_present else None
    return {"offered_tasks_per_s": offered_tasks_s, "actual_sent_calls_per_s": len(attempts) / duration_s,
            "measurement_including_drain_s": duration_s, "planned_calls": len(records), "sent_calls": len(attempts),
            "completed_calls": len(ok), "failed_or_skipped_calls": len(records) - len(ok),
            "p90_ttft_ms_success_only": percentile([r.get("ttft_ms") for r in ok], .9),
            "p95_ttft_ms_success_only": percentile([r.get("ttft_ms") for r in ok], .95),
            "p99_ttft_ms_success_only": percentile([r.get("ttft_ms") for r in ok], .99),
            "p90_tpot_ms_success_only": percentile([r.get("tpot_ms") for r in ok], .9),
            "itl_ms_p50": percentile(itls, .5) if itls is not None else None,
            "itl_ms_p95": percentile(itls, .95) if itls is not None else None,
            "itl_ms_p99": itl_p99,
            "worst_itl_ms_p50": worst_itl_p50, "worst_itl_ms_p90": worst_itl_p90,
            "worst_itl_ms_p95": percentile(worsts, .95) if worsts_present else None,
            "worst_itl_ms_p99": percentile(worsts, .99) if worsts_present else None,
            "worst_itl_ms_p999": percentile(worsts, .999) if worsts_present else None,
            # Same values as worst_itl_ms_p50/p90, under the evidence table's name.
            "longest_gap_ms_p50": worst_itl_p50, "longest_gap_ms_p90": worst_itl_p90,
            "gap_count": gap_count, "gap_ms_p99": gap_p99, "gap_above_bound_fraction": gap_above_bound,
            "gap_data_available": available,
            "slo_ttft_ms": ttft_ms, "slo_tpot_ms": tpot_ms, "slo_attainment_fraction": hit,
            "tpot_metric": tpot_metric, "ttft_metric": ttft_metric,
            "attainment_target_fraction": attainment, "meets_joint_slo": meets_joint and client_valid,
            "successful_calls_per_s": len(ok) / duration_s, "useful_calls_per_s": len(good) / duration_s,
            "useful_calls_per_s_per_gpu": len(good) / duration_s / gpus, "gpu_count": gpus,
            "max_client_launch_lag_ms": lag, "unreachable_calls": unreachable, "client_valid": client_valid,
            "start_skew_ms": start_skew_ms, "start_skew_tolerance_ms": start_skew_tolerance_ms if start_skew_ms is not None else None,
            "prefix_overlap_fraction_min": min((r["prefix_overlap_fraction"] for r in records if r.get("prefix_overlap_fraction") is not None), default=None),
            # An int, 0 when nothing reports cached tokens, as stored ramp files already carry.
            "cached_tokens_sum_reported": sum(r.get("cached_tokens") or 0 for r in records),
            "input_tokens_sum_sent": sum(r.get("input_tokens", 0) for r in attempts)}


def stops_ramp(summary, stop_failure_fraction):
    """The ramp's stop rule; check_runs applies the same rule to stored files."""
    return summary["failed_or_skipped_calls"] / max(1, summary["planned_calls"]) > stop_failure_fraction


def highest_qualifying(rows, ttft_ms, tpot_ms, attainment, score):
    """Walk the planned rates upward and stop at the first rate any run fails or did not reach;
    each metric is the minimum across runs at the last qualifying rate."""
    by_rate = {}
    for run in {row['run'] for row in rows}:
        ran = {row['offered']: row for row in rows if row['run'] == run}
        for rate in next(iter(ran.values()))['rates']:
            by_rate.setdefault(rate, []).append(score(ran[rate], ttft_ms, tpot_ms, attainment) if rate in ran else None)
    top = None
    for rate in sorted(by_rate):
        if not all(s is not None and s['meets_joint_slo'] for s in by_rate[rate]):
            break
        top = rate
    if top is None:
        return None
    scores = by_rate[top]
    return {'offered': top, 'useful_per_gpu': min(s['useful_calls_per_s_per_gpu'] for s in scores),
            'attainment': min(s['slo_attainment_fraction'] for s in scores),
            'planned_calls': min(s['planned_calls'] for s in scores), 'repeats': len(scores)}


def verdict(by_arch):
    """(top rate, winner, leaders). An UNAVAILABLE side yields 'incomparable', never a win over a
    stack that was not measured."""
    if any(b == UNAVAILABLE for b in by_arch.values()):
        return None, 'incomparable', []
    numeric = {a: b for a, b in by_arch.items() if isinstance(b, dict)}
    top = max((b['offered'] for b in numeric.values()), default=None)
    lead = sorted(a for a, b in numeric.items() if b['offered'] == top) if top is not None else []
    return top, (lead[0] if len(lead) == 1 else 'tie' if lead else 'neither'), lead


def promise_grid(rows_by_arch, ttft_axis, tpot_axis, attainment, score_factory, availability, definitions=TPOT_METRICS):
    """{definition: {(ttft_ms, tpot_ms): {"by_arch", "top", "winner", "lead"}}}."""
    grid = {}
    for definition in definitions:
        score = score_factory(definition)
        cells = {}
        for ttft in ttft_axis:
            for tpot in tpot_axis:
                by_arch = {}
                for arch, rows in rows_by_arch.items():
                    # Index, never .get(..., True): an unvouched pair must fail, not default to available.
                    if not availability[(arch, definition)]:
                        by_arch[arch] = UNAVAILABLE
                    else:
                        by_arch[arch] = highest_qualifying(rows, ttft, tpot, attainment, score)
                top, winner, lead = verdict(by_arch)
                cells[(ttft, tpot)] = {"by_arch": by_arch, "top": top, "winner": winner, "lead": lead}
        grid[definition] = cells
    return grid


# ---------------------------------------------------------------------------------------------
# Stored ramp files.

# Equal across every file of one comparison.
RAMP_MATCHED = ('instance_type', 'region', 'evidence_scope', 'model', 'model_revision', 'shape_sha256', 'gpu_count',
                'offered_window_s', 'timeout_s', 'max_client_lag_ms', 'context_length_tokens', 'skip_warmup',
                'stop_failure_fraction', 'tpot_metric', 'ttft_metric')
ARGUMENTS = ('max_client_lag_ms', 'duration_s', 'timeout_s', 'context_length_tokens', 'skip_warmup', 'stop_failure_fraction', 'rates')
REQUIRED_METADATA = ('architecture', 'instance_type', 'region', 'evidence_scope', 'model', 'model_revision', 'shape', 'seed', 'phase')
REQUIRED_REQUEST = ('ok', 'sent', 'client_launch_lag_ms')
REQUIRED_SUMMARY = ('gpu_count', 'offered_tasks_per_s', 'measurement_including_drain_s', 'attainment_target_fraction',
                    'slo_ttft_ms', 'slo_tpot_ms')
SUMMARY_KEYS = ('actual_sent_calls_per_s', 'planned_calls', 'sent_calls', 'completed_calls', 'failed_or_skipped_calls', 'p90_ttft_ms_success_only',
                'p90_tpot_ms_success_only', 'slo_attainment_fraction', 'meets_joint_slo', 'successful_calls_per_s',
                'useful_calls_per_s', 'useful_calls_per_s_per_gpu', 'max_client_launch_lag_ms', 'client_valid')


def load_rows(directories):
    """An unreadable or incomplete file stops everything, unlike check_reproduces' per-file refusal."""
    rows = []
    for directory in directories:
        files = sorted(Path(directory).glob('*-measured.json'))
        if not files:
            raise SystemExit(f'No measured files in {directory}; nothing was scored')
        for path in files:
            try:
                run = json.loads(path.read_text())
            except ValueError as exc:
                raise SystemExit(f'{path}: not readable as JSON ({exc}); the ramp may still be writing it') from None
            absent = [k for k in ('metadata', 'summary', 'requests') if k not in run]
            if absent:
                raise SystemExit(f"{path}: not a ramp result file, missing {', '.join(absent)}")
            m, s = run['metadata'], run['summary']
            arguments = m.get('arguments') or {}
            missing = ([f'metadata.{k}' for k in REQUIRED_METADATA if k not in m]
                       + [f'summary.{k}' for k in REQUIRED_SUMMARY + SUMMARY_KEYS if k not in s]
                       + [f'metadata.arguments.{k}' for k in ARGUMENTS if arguments.get(k) is None])
            if missing:
                raise SystemExit(f"{path}: written by an older or different harness, missing {', '.join(missing)}; "
                                 'score only ramps written by this revision of benchmark.py')
            if m['phase'] != 'measured':
                raise SystemExit(f"{path}: metadata.phase is {m['phase']!r}; only measured windows are scored")
            bad = next((i for i, r in enumerate(run['requests']) if not all(k in r for k in REQUIRED_REQUEST)), None)
            if bad is not None:
                raise SystemExit(f"{path}: requests[{bad}] lacks one of {', '.join(REQUIRED_REQUEST)}")
            if len(set(arguments['rates'])) != len(arguments['rates']):
                raise SystemExit(f"{path}: metadata.arguments.rates repeats a rate ({arguments['rates']})")
            rows.append({
                'source': str(path), 'run': str(directory), 'ramp_id': arguments.get('ramp_id'),
                'rates': list(arguments['rates']), 'architecture': m['architecture'], 'instance_type': m['instance_type'],
                'region': m['region'], 'evidence_scope': m['evidence_scope'], 'model': m['model'],
                'model_revision': m['model_revision'], 'shape': m['shape']['name'],
                'shape_sha256': hashlib.sha256(json.dumps(m['shape'], sort_keys=True).encode()).hexdigest(),
                'traffic_sha256': m.get('traffic_sha256'), 'seed': m['seed'],
                'offered_window_s': arguments['duration_s'], 'offered_task_count': s.get('offered_task_count'),
                'timeout_s': arguments['timeout_s'],
                'context_length_tokens': arguments['context_length_tokens'],
                'skip_warmup': arguments['skip_warmup'],
                'stop_failure_fraction': arguments['stop_failure_fraction'],
                'gpu_count': s['gpu_count'], 'offered': s['offered_tasks_per_s'],
                'duration_s': s['measurement_including_drain_s'], 'attainment_target': s['attainment_target_fraction'],
                'max_client_lag_ms': arguments['max_client_lag_ms'], 'stored': s, 'requests': run['requests'],
                # Defaulted, not in ARGUMENTS: older files lack these and must still load.
                'tpot_metric': arguments.get('tpot_metric', 'mean'),
                'ttft_metric': s.get('ttft_metric', 'per-call'),
                'schema_version': m.get('schema_version'),
            })
    return rows


def score(row, ttft_ms, tpot_ms, attainment, tpot_metric=None, ttft_metric=None):
    """summarize() for a loaded row; None keeps the mode the row was recorded with."""
    stored = row['stored']
    # Stored start skew is replayed so a merged run's verdict reproduces.
    tolerance = stored.get('start_skew_tolerance_ms')
    return summarize(row['requests'], row['duration_s'], row['gpu_count'], ttft_ms, tpot_ms, attainment,
                     row['offered'], row['max_client_lag_ms'], tpot_metric or row['tpot_metric'],
                     stored.get('start_skew_ms'), 500.0 if tolerance is None else tolerance,
                     ttft_metric or row['ttft_metric'])


def score_factory_for(definition, ttft_metric=None):
    """A score(row, ttft_ms, tpot_ms, attainment) fixed to one TPOT rule and TTFT mode."""
    return lambda row, ttft, tpot, att: score(row, ttft, tpot, att, tpot_metric=definition, ttft_metric=ttft_metric)


def check_reproduces(row):
    """Refuse a file whose stored summary does not come back from its own requests at its recorded
    objective, before any other objective is applied to it."""
    stored = row['stored']
    again = score(row, stored['slo_ttft_ms'], stored['slo_tpot_ms'], stored['attainment_target_fraction'])
    # Checked when present, so a corrupted descriptive field fails too; older files lack them.
    optional = ('unreachable_calls', 'tpot_metric', 'ttft_metric', 'itl_ms_p50', 'itl_ms_p95', 'itl_ms_p99',
                'p95_ttft_ms_success_only', 'p99_ttft_ms_success_only',
                'worst_itl_ms_p50', 'worst_itl_ms_p90', 'worst_itl_ms_p95', 'worst_itl_ms_p99', 'worst_itl_ms_p999',
                'start_skew_ms', 'start_skew_tolerance_ms',
                'gap_count', 'gap_ms_p99', 'longest_gap_ms_p50', 'longest_gap_ms_p90',
                'gap_above_bound_fraction', 'gap_data_available')
    for key in SUMMARY_KEYS + tuple(k for k in optional if k in stored):
        if again[key] != stored[key]:
            raise SystemExit(f"{row['source']}: re-scoring at the recorded objective changed {key} "
                             f"({stored[key]} -> {again[key]}); refusing to report")


def _fmt_rates(xs):
    return ', '.join(f'{x:g}' for x in xs)


def check_runs(group):
    """Refuse files that one ramp invocation could not have written."""
    seen = {}
    for row in group:
        key = (row['architecture'], row['offered'], row['seed'])
        if key in seen:
            raise SystemExit(f"{row['source']} and {seen[key]} share architecture, rate and seed; "
                             'a repeat needs another --seed, and each run is passed once')
        seen[key] = row['source']
    for run in sorted({row['run'] for row in group}):
        rows = [row for row in group if row['run'] == run]
        if len({(row['architecture'], row['seed'], tuple(row['rates']), row['ramp_id']) for row in rows}) != 1:
            raise SystemExit(f'{run}: files from more than one ramp invocation; give each ramp its own output directory')
        order, ran = rows[0]['rates'], {row['offered']: row for row in rows}
        done = order[:len(ran)]
        if set(done) != set(ran):
            raise SystemExit(f'{run}: measured {_fmt_rates(sorted(ran))} of {_fmt_rates(order)} tasks/s run in that order; '
                             'a ramp with a skipped rate is not comparable')
        stopped = [x for x in done if stops_ramp(ran[x]['stored'], rows[0]['stop_failure_fraction'])]
        if stopped and stopped[0] != done[-1]:
            raise SystemExit(f'{run}: the ramp stopped at {stopped[0]:g} tasks/s, yet the directory also holds later rates; '
                             'give each ramp its own output directory')
        if done != order and not stopped:
            raise SystemExit(f'{run}: ended at {done[-1]:g} of {_fmt_rates(order)} tasks/s without exceeding its failure fraction, '
                             'so the ramp was interrupted; rerun it')
        below = sorted(x for x in order[len(done):] if x < done[-1])
        if below:
            raise SystemExit(f'{run}: stopped at {done[-1]:g} tasks/s before the lower rates {_fmt_rates(below)}; '
                             'only rates above a stop count as misses, so rerun those rates')


def check_matched(shape, group):
    hashed = [bool(row['traffic_sha256']) for row in group]
    keys = RAMP_MATCHED + ('traffic_sha256',)
    if not any(hashed):
        keys = RAMP_MATCHED
        print(f'{shape}: these runs predate the traffic hash; compared on the shape definition only')
    elif not all(hashed):
        raise SystemExit(f'{shape}: some runs record a traffic hash and some do not; score runs from one harness revision')
    for key in keys:
        values = {row['source']: row[key] for row in group}
        if len(set(values.values())) != 1:
            raise SystemExit(f'{shape}: runs differ in {key}: ' + ', '.join(f'{source} has {value}' for source, value in sorted(values.items()))
                             + '; compare one budget, one workload and one ramp protocol at a time')


def check_completeness(shape, group):
    """Enough for a qualifying rate; a promise map also needs matched_protocol to agree."""
    check_matched(shape, group)
    check_runs(group)
    return sorted({row['architecture'] for row in group})


def matched_protocol(group, archs):
    """{arch: sorted (seed, rates) pairs}."""
    return {a: sorted({(row['seed'], tuple(row['rates'])) for row in group if row['architecture'] == a}) for a in archs}


def availability_by_arch(rows_by_arch, definitions=TPOT_METRICS):
    """{(arch, definition): bool}; a gap rule needs complete chunk data on every row of the arch."""
    return {(arch, d): (d == 'mean' or all(gap_data_available(r['requests']) for r in rows))
            for arch, rows in rows_by_arch.items() for d in definitions}
