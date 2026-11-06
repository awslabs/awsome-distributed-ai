#!/usr/bin/env python3
"""Collect this lab's EKS evidence and summarize measured ramp files under the three TPOT rules."""
import argparse
import csv
import json
import math
from pathlib import Path
from deployment import kubectl, owned_namespace, read_config
import report
from scoring import (DEFAULT_TPOT_AXIS_MS, DEFAULT_TTFT_AXIS_MS, ORIGINAL_SUMMARY_KEYS, PAIR_SETTINGS, TPOT_METRICS, TTFT_METRICS, UNAVAILABLE,
                     availability_by_arch, check_reproduces, gap_data_available, highest_qualifying, load_file,
                     promise_grid, score, score_factory_for, validate_grid_axis)

GROUP_KEY = ("architecture", "instance_type", "region", "evidence_scope", "model_revision", "shape_sha256", "gpu_count",
             "slo_ttft_ms", "slo_tpot_ms", "attainment_target_fraction")
# summary.csv column prefixes for the two gap rules; the original columns hold the mean rule.
RULE_PREFIX = {"itl-p99": "itl_p99", "worst-itl": "worst_itl"}
OBJECTIVE_FLAGS = (("slo_ttft_ms", "ttft_slo_ms"), ("slo_tpot_ms", "tpot_slo_ms"), ("attainment_target_fraction", "attainment_fraction"))


def objective(m, a):
    """(ttft_ms, tpot_ms, attainment, ttft_metric): the one recorded with the file unless given on the command line."""
    values = tuple(getattr(a, flag) if getattr(a, flag) is not None else m["stored"][key] for key, flag in OBJECTIVE_FLAGS)
    return values + (a.ttft_metric or "per-call",)


def overridden(m, a):
    """The objective items given on the command line with a value other than the one recorded with the file."""
    return [flag for key, flag in OBJECTIVE_FLAGS if getattr(a, flag) is not None and getattr(a, flag) != m["stored"][key]] + \
        (["ttft_metric"] if a.ttft_metric not in (None, "per-call") else [])


def summary_row(m, a):
    """One summary.csv row: the original lab's columns, in its order, at the objective in use, then the added ones.
    Every file gets the same columns, whichever harness version wrote it."""
    ttft_ms, tpot_ms, attainment, ttft_metric = objective(m, a)
    s = score(m, ttft_ms, tpot_ms, attainment, ttft_metric=ttft_metric)
    original = [k for k in ORIGINAL_SUMMARY_KEYS if k != "client_valid"]
    row = {"source": m["source"], **{k: m[k] for k in ("architecture", "instance_type", "region", "evidence_scope", "model_revision", "shape", "shape_sha256")},
           **{k: s[k] for k in original}, "offered_window_s": m["stored"].get("offered_window_s"),
           "offered_task_count": m["stored"].get("offered_task_count"), "client_valid": s["client_valid"],
           **{k: v for k, v in s.items() if k not in original and k != "client_valid"},
           "seed": m["seed"], "objective_given_on_command_line": bool(overridden(m, a))}
    available = gap_data_available(m["requests"])
    for d, prefix in RULE_PREFIX.items():
        g = score(m, ttft_ms, tpot_ms, attainment, tpot_metric=d, ttft_metric=ttft_metric)
        for key in ("meets_joint_slo", "slo_attainment_fraction", "useful_calls_per_s_per_gpu"):
            row[f"{prefix}_{key}"] = g[key] if available else UNAVAILABLE
    row["gap_data_available"] = available
    return row


def evidence_row(m, row, stack_available):
    """One report.html evidence row: a file's statistics and its verdict under each rule; a rule the stack
    cannot be judged under shows UNAVAILABLE, as the verdict does."""
    ev = {"architecture": m["architecture"], "offered": m["offered"], "seed": m["seed"], "source": m["source"],
          "planned_rates": m["rates"], "max_client_lag_bound_ms": m["max_client_lag_ms"],
          **{k: row[k] for k in ("planned_calls", "completed_calls", "instance_type", "gpu_count", "measurement_including_drain_s",
                                 "p90_ttft_ms_success_only", "p95_ttft_ms_success_only", "p99_ttft_ms_success_only",
                                 "p90_tpot_ms_success_only", "itl_ms_p99", "longest_gap_ms_p90", "client_valid",
                                 "max_client_launch_lag_ms", "unreachable_calls")}}
    ev.update({"mean_pass": row["meets_joint_slo"], "mean_attainment": row["slo_attainment_fraction"], "mean_goodput": row["useful_calls_per_s_per_gpu"]})
    for d, prefix in RULE_PREFIX.items():
        judged = stack_available[d]
        ev.update({f"{d}_pass": row[f"{prefix}_meets_joint_slo"] if judged else UNAVAILABLE,
                   f"{d}_attainment": row[f"{prefix}_slo_attainment_fraction"] if judged else UNAVAILABLE,
                   f"{d}_goodput": row[f"{prefix}_useful_calls_per_s_per_gpu"] if judged else UNAVAILABLE})
    return ev


def pair_marks(files):
    """Settings, planned rates and seeds that differ between the files of a unified/disaggregated pair."""
    marks = []
    unknown = sorted({m["architecture"] for m in files if m["settings"]["traffic_sha256"] is None})
    if unknown:
        marks.append(f"traffic_sha256 unknown for {', '.join(unknown)} (files written before the traffic hash was recorded), so the same traffic is not confirmed")
    for key in PAIR_SETTINGS:
        values = {m["architecture"]: set() for m in files}
        for m in files:
            values[m["architecture"]].add(m["settings"].get(key))
        # The stacks are compared with each other, as planned rates and seeds are below.
        if len({frozenset(v - {None}) for v in values.values()}) > 1:
            shown = "; ".join(f"{arch} {', '.join('unknown' if v is None else str(v) for v in sorted(vs, key=str))}" for arch, vs in sorted(values.items()))
            marks.append(f"{key} differs ({shown})")
    for label, get in (("planned rates", lambda m: m["rates"]), ("seeds", lambda m: [m["seed"]])):
        by_arch = {}
        for m in files:
            by_arch.setdefault(m["architecture"], set()).update(get(m))
        if len({frozenset(v) for v in by_arch.values()}) > 1:
            marks.append(f"{label} differ (" + "; ".join(f"{arch} {', '.join(f'{x:g}' if isinstance(x, (int, float)) else str(x) for x in sorted(v, key=str))}"
                                                     for arch, v in sorted(by_arch.items())) + ")")
    return marks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True, help="Directories containing rate-measured.json files")
    p.add_argument("--output", required=True)
    p.add_argument("--config", help="Also collect live resources and engine logs from the owned namespace")
    p.add_argument("--ttft-ms", type=float, nargs="+", default=list(DEFAULT_TTFT_AXIS_MS), help="report.html promise-map TTFT columns, ms")
    p.add_argument("--tpot-ms", type=float, nargs="+", default=list(DEFAULT_TPOT_AXIS_MS), help="report.html promise-map TPOT rows, ms: per output token under mean and itl-p99, one raw gap under worst-itl")
    p.add_argument("--no-report", action="store_true", help="Skip report.html; summary.csv and qualified-rates.json are unaffected")
    p.add_argument("--ttft-slo-ms", type=float, help="TTFT bound, ms. Default: the one recorded with each file")
    p.add_argument("--tpot-slo-ms", type=float, help="TPOT bound, ms: per output token under mean and itl-p99, one raw gap under worst-itl. Default: recorded")
    p.add_argument("--attainment-fraction", type=float, help="Share of planned calls that must be good, e.g. 0.9. Default: recorded")
    p.add_argument("--ttft-metric", choices=TTFT_METRICS, help="per-call (default, as recorded), or a pooled p95/p99 of each file's completed calls")
    a = p.parse_args()
    for flag, value in (("--ttft-slo-ms", a.ttft_slo_ms), ("--tpot-slo-ms", a.tpot_slo_ms)):
        if value is not None and not (math.isfinite(value) and value > 0):
            p.error(f"{flag} needs a finite positive bound")
    if a.attainment_fraction is not None and not (math.isfinite(a.attainment_fraction) and 0 < a.attainment_fraction <= 1):
        p.error("--attainment-fraction must be in (0, 1]; use 0.9 for 90 percent")
    for name in ("ttft_ms", "tpot_ms"):
        try:
            setattr(a, name, list(validate_grid_axis(getattr(a, name))))
        except ValueError as exc:
            p.error(f"--{name.replace('_', '-')} {exc}")
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    measured, left_out = [], {}
    for run in a.runs:
        for f in sorted(Path(run).glob("*-measured.json")):
            try:
                m = load_file(f)
                check_reproduces(m)
            except (OSError, ValueError, KeyError, TypeError, AttributeError, ZeroDivisionError) as exc:
                left_out[str(f)] = f"{type(exc).__name__}: {exc}"
                continue
            measured.append(m)
    if not measured:
        # Nothing of this run to draw, so an earlier report.html must not look like its result.
        if (out / "report.html").exists():
            (out / "report.html").unlink()
            print(f"Removed the earlier {out / 'report.html'}: this run drew no promise map")
        raise SystemExit(("No measured file could be read; no summary was fabricated" if left_out else "No measured files found; no summary was fabricated")
                         + "".join(f"\n{k}: {v}" for k, v in sorted(left_out.items())))
    rows = [summary_row(m, a) for m in measured]
    with (out / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Group by hardware, workload, thresholds and GPU budget. Never mix the scaling arm with the iso-resource arm.
    groups = {}
    for m, r in zip(measured, rows):
        groups.setdefault(tuple(r[k] for k in GROUP_KEY), []).append((m, r))
    ttft_metric = a.ttft_metric or "per-call"
    qualified, by_group = [], {}
    for key, members in groups.items():
        files = [m for m, _r in members]
        available = availability_by_arch({key[0]: files})
        by_rule = {}
        for d in TPOT_METRICS:
            hq = highest_qualifying(files, key[7], key[8], key[9], score_factory_for(d, ttft_metric)) if available[(key[0], d)] else UNAVAILABLE
            by_rule[d] = hq
        by_group[key] = by_rule
        qualified.append({"group": list(key), "highest_tested_offered_tasks_per_s_meeting_slo": max((r["offered_tasks_per_s"] for _m, r in members if r["meets_joint_slo"]), default=None),
                          "max_observed_useful_calls_per_s_per_gpu": max(r["useful_calls_per_s_per_gpu"] for _m, r in members),
                          "qualification": "Finite-window observation including drain; extend duration and repeat before claiming sustained capacity",
                          "ttft_metric": ttft_metric,
                          "by_rule": {d: (v if v == UNAVAILABLE else {"highest_tested_offered_tasks_per_s_meeting_slo": v["offered"] if v else None,
                                                                       "useful_calls_per_s_per_gpu_at_that_rate": v["useful_per_gpu"] if v else None})
                                      for d, v in by_rule.items()}})
    (out / "qualified-rates.json").write_text(json.dumps(qualified, indent=2))

    # report.html pairs the unified and disaggregated groups that share the rest of the key.
    pairs = {}
    for key in groups:
        pairs.setdefault(key[1:], {})[key[0]] = key
    notices = [f"left out {source}, which could not be read as a ramp file ({reason}); measure that rate again" for source, reason in sorted(left_out.items())]
    changed = sorted({flag for m in measured for flag in overridden(m, a)})
    if changed:
        notices.append(f"judged at {', '.join(f.replace('_', '-') for f in changed)} given on the command line, not the value recorded with the files")
    for m in measured:
        n = sum(str(r.get("error", "")).startswith("ConnectError") for r in m["requests"])
        if n:
            notices.append(f"{m['source']}: {n} of {len(m['requests'])} calls could not connect; if they cluster or are many, rerun the ramp from this rate into the same --output")
        if not gap_data_available(m["requests"]):
            notices.append(f"{m['source']} has no per-chunk timing, so itl-p99 and worst-itl cannot judge its stack; measure that rate again to judge them")
    sections = []
    for rest, archs in sorted(pairs.items(), key=lambda kv: str(kv[0])):
        shape = groups[next(iter(archs.values()))][0][0]["shape"]
        if set(archs) != {"unified", "disaggregated"}:
            notices.append(f"{shape} ({', '.join(sorted(archs))}, {rest[5]} GPUs) has no unified and disaggregated pair with the same hardware, workload, "
                           "GPU count and objective, so it has no promise map; measure the other stack with the same settings to compare")
            continue
        files_by_arch = {arch: [m for m, _r in groups[k]] for arch, k in archs.items()}
        members = [x for k in archs.values() for x in groups[k]]
        availability = availability_by_arch(files_by_arch)
        marks = pair_marks([m for m, _r in members])
        notices += [f"{shape}: {mark}; measure again with them aligned to compare like with like" for mark in marks]
        sections.append({
            "shape": shape, "archs": sorted(archs), "objective": (rest[6], rest[7]), "attainment": rest[8], "ttft_metric": ttft_metric,
            "objective_given": [n for n, flag in (("TTFT bound", "ttft_slo_ms"), ("TPOT bound", "tpot_slo_ms"), ("attainment target", "attainment_fraction"), ("TTFT judging mode", "ttft_metric")) if flag in changed],
            "qualified": {(arch, d): by_group[k][d] for arch, k in archs.items() for d in TPOT_METRICS},
            "grid": None if a.no_report else promise_grid(files_by_arch, a.ttft_ms, a.tpot_ms, rest[8], lambda d: score_factory_for(d, ttft_metric), availability),
            "ttft_axis": a.ttft_ms, "tpot_axis": a.tpot_ms, "marks": marks,
            "evidence": [evidence_row(m, r, {d: availability[(m["architecture"], d)] for d in TPOT_METRICS}) for m, r in members],
        })
    report_path = out / "report.html"
    report_error = None
    if a.no_report or not sections:
        if report_path.exists():
            report_path.unlink()
            print(f"Removed the earlier {report_path}: this run drew no promise map")
    else:
        try:
            report_path.write_text(report.render_sections(sections, notices))
            print(f"Wrote {report_path.resolve()}")
        except Exception as exc:
            if report_path.exists():
                report_path.unlink()
                print(f"Removed the earlier {report_path}: this run drew no promise map")
            report_error = f"report.html rendering failed ({type(exc).__name__}: {exc})"
            print(f"{report_error}; continuing without it")
    if not report_path.exists():
        for notice in notices:
            print(notice)
    if a.config:
        c = read_config(a.config)
        owned_namespace(c)
        (out / "config.json").write_text(json.dumps(c, indent=2))
        for resource in ("pods", "deployments", "services", "events"):
            (out / f"{resource}.json").write_text(kubectl(c, "get", resource, "-o", "json", capture_output=True).stdout)
        pods = json.loads((out / "pods.json").read_text())["items"]
        for pod in pods:
            name = pod["metadata"]["name"]
            if pod["metadata"].get("labels", {}).get("app.kubernetes.io/part-of") != "aim345-lab":
                continue
            (out / f"{name}.log").write_text(kubectl(c, "logs", name, "--all-containers=true", "--tail=-1", capture_output=True).stdout)
            if pod["metadata"]["labels"].get("component") == "engine" and pod["status"]["phase"] == "Running":
                ports = (30000, 30001) if c.get('placement') == 'packed' else (30000,)
                for port in ports:
                    for endpoint in ("metrics", "server_info"):
                        code = f"import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:{port}/{endpoint}').read().decode())"
                        suffix = f"{port}-{endpoint}" if c.get('placement') == 'packed' else endpoint
                        (out / f"{name}-{suffix}.txt").write_text(kubectl(c, "exec", name, "-c", "engine", "--", "python3", "-c", code, capture_output=True).stdout)
    print(json.dumps({"summary_csv": str(out / "summary.csv"), "measured_rate_rows": len(rows), "qualified_rates": str(out / "qualified-rates.json"),
                      "report_html": str(report_path) if report_path.exists() else None, "left_out": left_out}))
    reasons = [f"left out {len(left_out)} file(s) that could not be read as ramp files:\n" + "\n".join(f"{k}: {v}" for k, v in sorted(left_out.items()))] if left_out else []
    if report_error:
        reasons.append(report_error)
    if reasons:
        raise SystemExit("\n\n".join(reasons))

if __name__ == "__main__":
    main()
