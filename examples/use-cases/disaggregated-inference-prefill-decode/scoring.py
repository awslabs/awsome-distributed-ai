"""Judging for this example: the three TPOT rules, the qualifying rate and goodput as the original lab computes them,
the promise-map grid, and reading stored ramp files. report.html's glossary states the same definitions.
"""
import hashlib
import json
import math
from pathlib import Path

TPOT_METRICS = ("mean", "itl-p99", "worst-itl")
TTFT_METRICS = ("per-call", "p95", "p99")

# Neither a rate nor None ("qualified at no rate"). Compare with ==: copies come back from JSON.
UNAVAILABLE = "unavailable (a file of this stack has no per-chunk timing)"

# Fixed default axes, so promise maps from different runs line up.
DEFAULT_TTFT_AXIS_MS = tuple(range(500, 5001, 500))
DEFAULT_TPOT_AXIS_MS = tuple(range(10, 101, 10))

# The summary keys the original lab's benchmark.py wrote; check_reproduces compares only these.
ORIGINAL_SUMMARY_KEYS = ("offered_tasks_per_s", "actual_sent_calls_per_s", "measurement_including_drain_s", "planned_calls",
                         "sent_calls", "completed_calls", "failed_or_skipped_calls", "p90_ttft_ms_success_only",
                         "p90_tpot_ms_success_only", "slo_ttft_ms", "slo_tpot_ms", "slo_attainment_fraction",
                         "attainment_target_fraction", "meets_joint_slo", "successful_calls_per_s", "useful_calls_per_s",
                         "useful_calls_per_s_per_gpu", "gpu_count", "max_client_launch_lag_ms", "prefix_overlap_fraction_min",
                         "cached_tokens_sum_reported", "input_tokens_sum_sent", "client_valid")

# Measurement settings outside the original grouping key; report.html marks a pair whose files differ in one.
PAIR_SETTINGS = ("duration_s", "timeout_s", "context_length_tokens", "skip_warmup", "stop_failure_fraction",
                 "max_client_lag_ms", "traffic_sha256")


def validate_grid_axis(values):
    """De-duplicated so a repeated bound is one column, not a duplicated cell; ValueError lists every bad bound."""
    bad = [v for v in values if not math.isfinite(v) or v <= 0]
    if bad:
        raise ValueError(f"needs finite positive bounds, got {bad}")
    return tuple(sorted(set(values)))


def percentile(values, fraction):
    """Nearest-rank, not interpolated as np.percentile in sglang.bench_serving and vllm bench serve; None values are dropped."""
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
            # benchmark.py records a chunk only when it adds tokens; a chunk with n == 0 would hide the gap before it.
            if current["n"] > 0:
                pooled += [(current["t_ms"] - previous["t_ms"]) / current["n"]] * current["n"]
    return pooled or None


def _chunk_ok(c):
    return isinstance(c, dict) and all(isinstance(c.get(k), (int, float)) and not isinstance(c.get(k), bool) for k in ("t_ms", "n"))


def gap_data_available(records):
    """False means itl-p99 and worst-itl report UNAVAILABLE: some call has no chunks key (a file written
    before chunks were recorded), a completed call has no chunks, or a chunk lacks its time or token count."""
    return (all("chunks" in r and isinstance(r["chunks"], list) and all(_chunk_ok(c) for c in r["chunks"]) for r in records)
            and all(r.get("chunks") for r in records if r.get("ok")))


def summarize(records, duration_s, gpus, ttft_ms, tpot_ms, attainment, offered_tasks_s, actual_calls_s,
              tpot_metric="mean", ttft_metric="per-call"):
    """One file's verdict before the client check. A call passes the TTFT side when its TTFT is within ttft_ms (per-call),
    or when it completed (p95/p99, where that percentile of the completed calls' TTFT must be within ttft_ms). A good call
    passes the TTFT side and, under mean, its own mean TPOT within tpot_ms, or, under worst-itl, its longest undivided gap
    within tpot_ms; under itl-p99 every call that passes the TTFT side is good, and the file's pooled per-token gap p99
    must be within tpot_ms. The file meets the objective when good calls / planned calls reaches attainment and those
    percentiles hold. Under mean and per-call this is the original lab's summary."""
    assert tpot_metric in TPOT_METRICS
    assert ttft_metric in TTFT_METRICS
    ok = [r for r in records if r["ok"]]
    attempts = [r for r in records if r.get("sent", False)]
    ttft_values = [r.get("ttft_ms") for r in ok]
    # Gap statistics only from complete chunk records; the mean rule does not read chunks.
    gaps = gap_data_available(records)
    if ttft_metric == "per-call":
        ttft_good = [r for r in ok if r.get("ttft_ms") is not None and r["ttft_ms"] <= ttft_ms]
        meets_ttft = True
    else:
        pooled_ttft = percentile(ttft_values, .95 if ttft_metric == "p95" else .99)
        # benchmark.py records a TTFT for every completed call, so the percentile covers all of them.
        ttft_good = ok
        meets_ttft = pooled_ttft is not None and pooled_ttft <= ttft_ms
    itls = pooled_itls_ms(ok) if gaps else None
    itl_p99 = percentile(itls, .99) if itls is not None else None

    def tpot_ok(r):
        if tpot_metric == "itl-p99":
            return True
        if tpot_metric == "worst-itl":
            w = worst_itl_ms(r) if gaps else None
            return w is not None and w <= tpot_ms
        return r.get("tpot_ms") is not None and r["tpot_ms"] <= tpot_ms
    good = [r for r in ttft_good if tpot_ok(r)]
    hit = len(good) / len(records) if records else 0.0
    meets_tpot = tpot_metric != "itl-p99" or (itl_p99 is not None and itl_p99 <= tpot_ms)
    return {"offered_tasks_per_s": offered_tasks_s, "actual_sent_calls_per_s": actual_calls_s,
            "measurement_including_drain_s": duration_s, "planned_calls": len(records), "sent_calls": len(attempts),
            "completed_calls": len(ok), "failed_or_skipped_calls": len(records) - len(ok),
            "p90_ttft_ms_success_only": percentile(ttft_values, .9),
            "p90_tpot_ms_success_only": percentile([r.get("tpot_ms") for r in ok], .9),
            "slo_ttft_ms": ttft_ms, "slo_tpot_ms": tpot_ms, "slo_attainment_fraction": hit,
            "attainment_target_fraction": attainment, "meets_joint_slo": hit >= attainment and meets_ttft and meets_tpot,
            "successful_calls_per_s": len(ok) / duration_s, "useful_calls_per_s": len(good) / duration_s,
            "useful_calls_per_s_per_gpu": len(good) / duration_s / gpus, "gpu_count": gpus,
            "max_client_launch_lag_ms": max((r.get("client_launch_lag_ms", 0) for r in records), default=0),
            "prefix_overlap_fraction_min": min((r["prefix_overlap_fraction"] for r in records if r.get("prefix_overlap_fraction") is not None), default=None),
            "cached_tokens_sum_reported": sum(r.get("cached_tokens") or 0 for r in records),
            "input_tokens_sum_sent": sum(r.get("input_tokens", 0) for r in attempts),
            # Added for the gap rules and report.html; a ConnectError is counted here and stays a failed call.
            "tpot_metric": tpot_metric, "ttft_metric": ttft_metric,
            "p95_ttft_ms_success_only": percentile(ttft_values, .95), "p99_ttft_ms_success_only": percentile(ttft_values, .99),
            "itl_ms_p99": itl_p99, "longest_gap_ms_p90": percentile([worst_itl_ms(r) for r in ok], .9) if gaps else None,
            "unreachable_calls": sum(str(r.get("error", "")).startswith("ConnectError") for r in records)}


def client_check(summary, max_client_lag_ms, recorded=None):
    """The original lab's client check: a rate whose launch lag passed the bound is invalid. Without a recorded
    bound, the file's recorded client_valid stands."""
    summary["client_valid"] = summary["max_client_launch_lag_ms"] <= max_client_lag_ms if max_client_lag_ms is not None else recorded is not False
    summary["meets_joint_slo"] &= summary["client_valid"]
    return summary


def highest_qualifying(rows, ttft_ms, tpot_ms, attainment, score):
    """The original lab's rule: the highest offered rate at which some file meets the objective, seeds pooled.
    Goodput is the largest among the files that met it at that rate."""
    scored = [(row, score(row, ttft_ms, tpot_ms, attainment)) for row in rows]
    met = [s for _row, s in scored if s["meets_joint_slo"]]
    if not met:
        return None
    top = max(s["offered_tasks_per_s"] for s in met)
    at = [s for s in met if s["offered_tasks_per_s"] == top]
    return {"offered": top, "useful_per_gpu": max(s["useful_calls_per_s_per_gpu"] for s in at),
            "files_met": len(at), "files_measured": sum(1 for row, _s in scored if row["offered"] == top)}


def verdict(by_arch):
    """(top rate, winner, leaders). An UNAVAILABLE side yields 'incomparable', never a win over a
    stack that was not measured."""
    if any(b == UNAVAILABLE for b in by_arch.values()):
        return None, 'incomparable', []
    numeric = {a: b for a, b in by_arch.items() if isinstance(b, dict)}
    top = max((b['offered'] for b in numeric.values()), default=None)
    lead = sorted(a for a, b in numeric.items() if b['offered'] == top) if top is not None else []
    return top, (lead[0] if len(lead) == 1 else 'tie' if lead else 'neither'), lead


def promise_grid(rows_by_arch, ttft_axis, tpot_axis, attainment, score_factory, availability):
    """{definition: {(ttft_ms, tpot_ms): {"by_arch", "top", "winner", "lead"}}}."""
    grid = {}
    for definition in TPOT_METRICS:
        score = score_factory(definition)
        cells = {}
        for ttft in ttft_axis:
            for tpot in tpot_axis:
                by_arch = {arch: highest_qualifying(rows, ttft, tpot, attainment, score) if availability[(arch, definition)] else UNAVAILABLE
                           for arch, rows in rows_by_arch.items()}
                top, winner, lead = verdict(by_arch)
                cells[(ttft, tpot)] = {"by_arch": by_arch, "top": top, "winner": winner, "lead": lead}
        grid[definition] = cells
    return grid


def availability_by_arch(rows_by_arch):
    """{(arch, definition): bool}; a gap rule needs complete chunk data on every file of the stack."""
    return {(arch, d): (d == 'mean' or all(gap_data_available(r['requests']) for r in rows))
            for arch, rows in rows_by_arch.items() for d in TPOT_METRICS}


# ---------------------------------------------------------------------------------------------
# Stored ramp files.

def load_file(path):
    """One *-measured.json as a row; ValueError or KeyError when it cannot be read as one."""
    run = json.loads(Path(path).read_text(encoding="utf-8"))
    m, s, requests = run["metadata"], run["summary"], run["requests"]
    if not (isinstance(m, dict) and isinstance(s, dict) and isinstance(requests, list)):
        raise ValueError("metadata, summary or requests has the wrong type")
    arguments = m.get("arguments") if isinstance(m.get("arguments"), dict) else {}
    for r in requests:
        if not isinstance(r, dict) or "ok" not in r:
            raise ValueError("a call record has no ok field")
    for key in ("measurement_including_drain_s", "gpu_count"):
        if not (isinstance(s.get(key), (int, float)) and s[key] > 0):
            raise ValueError(f"summary.{key} is not a positive number")
    rates = arguments.get("rates") if isinstance(arguments.get("rates"), list) else []
    return {"source": str(path), "architecture": m["architecture"], "instance_type": m["instance_type"], "region": m["region"],
            "evidence_scope": m["evidence_scope"], "model_revision": m["model_revision"], "shape": m["shape"]["name"],
            "shape_sha256": hashlib.sha256(json.dumps(m["shape"], sort_keys=True).encode()).hexdigest(),
            "seed": m.get("seed"), "rates": rates, "offered": s["offered_tasks_per_s"],
            "duration_s": s["measurement_including_drain_s"], "gpu_count": s["gpu_count"],
            "max_client_lag_ms": arguments.get("max_client_lag_ms"),
            # As text, so a value of any type can be compared and shown; None means the file does not record it.
            "settings": {k: None if v is None else str(v) for k, v in ((k, m.get(k) if k == "traffic_sha256" else arguments.get(k)) for k in PAIR_SETTINGS)},
            "stored": s, "requests": requests}


def score(row, ttft_ms, tpot_ms, attainment, tpot_metric="mean", ttft_metric="per-call"):
    """summarize() and the client check for a loaded row, at the given objective and modes."""
    records = row["requests"]
    s = summarize(records, row["duration_s"], row["gpu_count"], ttft_ms, tpot_ms, attainment, row["offered"],
                  sum(r.get("sent", False) for r in records) / row["duration_s"], tpot_metric, ttft_metric)
    return client_check(s, row["max_client_lag_ms"], row["stored"].get("client_valid"))


def score_factory_for(definition, ttft_metric="per-call"):
    """A score(row, ttft_ms, tpot_ms, attainment) fixed to one TPOT rule and TTFT mode."""
    return lambda row, ttft, tpot, att: score(row, ttft, tpot, att, tpot_metric=definition, ttft_metric=ttft_metric)


def check_reproduces(row):
    """ValueError when the file's stored summary does not come back from its own call records at the objective it
    recorded (mean, per-call): the records and the summary disagree, so the file cannot be interpreted."""
    stored = row["stored"]
    again = score(row, stored["slo_ttft_ms"], stored["slo_tpot_ms"], stored["attainment_target_fraction"])
    for key in ORIGINAL_SUMMARY_KEYS:
        if key in stored and again[key] != stored[key]:
            raise ValueError(f"its call records give {key} {again[key]}, its stored summary {stored[key]}")
