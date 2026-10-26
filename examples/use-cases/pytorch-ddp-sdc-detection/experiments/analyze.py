#!/usr/bin/env python3
"""Summarize one run of run_order.py:  python3 analyze.py <results dir> [--out summary.json]

Reads order.json and norms.json.gz (standard library only) and reports:
  - every scenario: the alarm, the step relative to the snapshot, and the state afterwards
  - healthy runs: steps, audits, comparisons, weight checks, alarms, and a 95% upper bound on
    the false-alarm rate (rule of three: 3 / N when 0 events were seen in N trials)
  - the gradient-norm outlier test (|norm - median| / MAD per step, per rank): its healthy
    spread, a threshold of 1.5 x the highest healthy score in the first 800 healthy steps,
    false alarms at that threshold, and each finite fault's score
  - estimated average step times for common schedules, including the weight check
"""
import argparse
import gzip
import json
import math
import os
from statistics import median


def mad_scores(norms):
    """Per-rank outlier scores for one step: |x - median| / MAD. NaN or Inf -> inf."""
    if not all(math.isfinite(x) for x in norms):
        return [math.inf if not math.isfinite(x) else 0.0 for x in norms]
    m = median(norms)
    dev = [abs(x - m) for x in norms]
    mad = median(dev)
    return [(d / mad if mad > 0 else (math.inf if d > 0 else 0.0)) for d in dev]


def _finite(o):
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_finite(v) for v in o]
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out")
    a = ap.parse_args()
    recs = json.load(open(os.path.join(a.dir, "order.json")))
    with gzip.open(os.path.join(a.dir, "norms.json.gz"), "rt") as f:
        norms = json.load(f)
    by_kind = {}
    for r in recs:
        by_kind.setdefault(r["kind"], []).append(r)
    env = by_kind["env"][0]
    world = env["world"]
    out = {"env": {k: env[k] for k in ("torch", "cuda", "nccl", "world", "nodes", "gpus_per_node", "gpu")}}

    # healthy runs and the false-alarm bound
    healthy = by_kind.get("healthy", [])
    audits = sum(h["audit_steps"] for h in healthy)
    checks = sum(h["weight_checks"] for h in healthy)
    alarms = sum(h["n_alarms"] for h in healthy)
    out["healthy"] = {
        "runs": [{k: h[k] for k in ("name", "minutes", "steps", "audit_steps", "pair_comparisons",
                                    "weight_checks", "n_alarms", "median_step_ms")} for h in healthy],
        "total_steps": sum(h["steps"] for h in healthy), "total_audit_steps": audits,
        "total_pair_comparisons": audits * world, "total_weight_checks": checks,
        "total_alarms": alarms,
        "false_alarm_95pct_upper_bound_per_audit_step": 3 / audits if audits and not alarms else None,
        "false_alarm_95pct_upper_bound_per_weight_check": 3 / checks if checks and not alarms else None}

    # gradient-norm outlier test
    healthy_names = [h["name"] for h in healthy]
    calib_run = "healthy_production_schedule" if "healthy_production_schedule" in norms else healthy_names[0]
    calib = norms[calib_run][:800]
    calib_max = max(max(mad_scores(v)) for _, v in calib)
    k = math.ceil(1.5 * calib_max)
    per_run = {}
    for name in healthy_names:
        steps = norms.get(name, [])
        mx = [max(mad_scores(v)) for _, v in steps]
        held_out = mx[800:] if name == calib_run else mx
        per_run[name] = {"steps": len(mx), "max_score": max(mx) if mx else None,
                         "steps_flagged_at_k": sum(1 for s in held_out if s > k),
                         "evaluated_steps": len(held_out)}
    faults = []
    for sc in by_kind.get("scenario", []):
        if not sc["faults"] or sc["faults"][0]["kind"] == "weight":
            continue
        base = sc["base_step"]
        culprits = {f["rank"] for f in sc["faults"]}
        start = min(f["start"] for f in sc["faults"])
        steps = [(s, v) for s, v in norms.get(sc["name"], []) if s >= start]
        culprit_scores, others_over = [], 0
        for s, v in steps:
            sc_ = mad_scores(v)
            culprit_scores.append(max(sc_[r] for r in culprits))
            others_over += any(sc_[r] > k for r in range(world) if r not in culprits)
        faults.append({"scenario": sc["name"], "fault_steps_seen": len(steps),
                       "culprit_max_score": max(culprit_scores) if culprit_scores else None,
                       "culprit_steps_over_k": sum(1 for x in culprit_scores if x > k),
                       "steps_where_another_rank_was_over_k": others_over,
                       "caught_by_workflow": (sc["alarm"] or {}).get("check"),
                       "alarm_rel_step": sc["alarm_rel_step"], "fault_start_rel": start - base})
    out["outlier_test"] = {"calibration_run": calib_run, "calibration_steps": len(calib),
                           "calibration_max_score": calib_max, "k": k,
                           "healthy": per_run, "faults": faults}

    # scenarios
    out["scenarios"] = []
    for sc in by_kind.get("scenario", []):
        al = sc["alarm"] or {}
        rel = (lambda s: None if s is None else s - sc["base_step"])
        w = al.get("suspect_window")
        out["scenarios"].append({
            "name": sc["name"], "alarm_rel_step": sc["alarm_rel_step"], "check": al.get("check"),
            "nonfinite": al.get("nonfinite"), "mismatched_pairs": al.get("mismatched_pairs"),
            "culprits": al.get("culprits"), "suspects": al.get("suspects"),
            "verdicts": al.get("verdicts"), "suspect_window_rel": w and [rel(w[0]), rel(w[1])],
            "state_after": sc["state_after"],
            "saves": [{"step_rel": rel(s["step"]),
                       "tags_rel": {t: rel(v) for t, v in s["tags"].items()},
                       "saved_copy_matches_majority": s["saved_copy_matches_majority"]}
                      for s in sc["saves"]]})

    # timing and schedule estimates (including the weight check every 10 steps)
    t = by_kind["timing"][0]
    est = {}
    for mode, audit_ms in (("deterministic", t["loop_audit_det_ms"]), ("default", t["loop_audit_default_ms"])):
        for every in (10, 50, 100):
            avg = ((every - 1) * t["loop_nonaudit_ms"] + audit_ms) / every + t["weight_check_ms"] / 10
            est[f"{mode}_audits_every_{every}"] = {"ms": round(avg, 1),
                                                   "over_plain_pct": round(100 * (avg / t["plain_ddp_ms"] - 1))}
    out["timing"] = {k_: v for k_, v in t.items() if k_ not in ("kind", "utc")}
    out["schedule_estimates_with_weight_check_every_10"] = est
    for kind in ("topology_unit", "fingerprint_unit", "majority_unit", "resume_check"):
        if kind in by_kind:
            out[kind] = {k_: v for k_, v in by_kind[kind][0].items() if k_ not in ("kind", "utc")}
    s = json.dumps(_finite(out), indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(s + "\n")
    print(s)


if __name__ == "__main__":
    main()
