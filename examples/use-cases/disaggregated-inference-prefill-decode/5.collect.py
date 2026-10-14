#!/usr/bin/env python3
"""Score measured ramp files under the three TPOT rules and write summary.csv, qualified-rates.json and
report.html; with --config, also collect this lab's EKS evidence. README sections "5.collect.py's output"
and "Score against another objective".
"""
import argparse
import csv
import io
import json
import math
from pathlib import Path
from deployment import kubectl, owned_namespace, read_config
import report
from scoring import (DEFAULT_TPOT_AXIS_MS, DEFAULT_TTFT_AXIS_MS, TPOT_METRICS, TTFT_METRICS, UNAVAILABLE,
                     availability_by_arch, check_completeness, check_reproduces, gap_data_available,
                     highest_qualifying, load_rows, matched_protocol, promise_grid, score, score_factory_for,
                     validate_grid_axis)

SUMMARY_LEADING_FIELDS = ("shape", "stack", "seed", "rate", "definition", "source",
                          "meets_joint_slo", "slo_attainment_fraction", "useful_calls_per_s_per_gpu")


# (key recorded in each file's summary, the option that overrides it, its name in report.html)
OBJECTIVE_FIELDS = (("slo_ttft_ms", "--ttft-slo-ms", "TTFT bound"), ("slo_tpot_ms", "--tpot-slo-ms", "TPOT bound"),
                    ("attainment_target_fraction", "--attainment-fraction", "attainment target"),
                    ("ttft_metric", "--ttft-metric", "TTFT judging mode"))


def _attr(flag):
    return flag[2:].replace("-", "_")


def write_csv(path, records):
    """One CSV from records whose keys may differ."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(dict.fromkeys(k for r in records for k in r)))
    writer.writeheader()
    writer.writerows(records)
    Path(path).write_text(buf.getvalue())


def file_objective(m, given):
    """(ttft_ms, tpot_ms, attainment, ttft_metric) for one file."""
    # Overrides never touch the loaded rows, so check_matched and check_reproduces still see the recorded values.
    recorded = {**m['stored'], "ttft_metric": m['ttft_metric']}
    return tuple(given[_attr(flag)] if given[_attr(flag)] is not None else recorded[key] for key, flag, _name in OBJECTIVE_FIELDS)


def _rate(value):
    """A rate, None (qualifies nowhere) or scoring.UNAVAILABLE."""
    return value['offered'] if isinstance(value, dict) else value


def _goodput(value):
    """Goodput at the qualifying rate, in the same state as _rate."""
    return value['useful_per_gpu'] if isinstance(value, dict) else value


def _console_rate(value):
    rate = _rate(value)
    if rate == UNAVAILABLE:
        return "unavailable"
    return "none" if rate is None else str(rate)


def _console_goodput(value):
    goodput = _goodput(value)
    if goodput == UNAVAILABLE:
        return "unavailable"
    return "none" if goodput is None else f"{goodput:g}"


def row_evidence(m, given):
    """summary.csv rows for one measured file, one per TPOT rule. A gap rule without complete chunk
    data is UNAVAILABLE rather than the conservative failure scoring.summarize would return."""
    ttft_ms, tpot_ms, attainment, ttft_metric = file_objective(m, given)
    available = gap_data_available(m['requests'])
    rows = []
    for d in TPOT_METRICS:
        s = score(m, ttft_ms, tpot_ms, attainment, tpot_metric=d, ttft_metric=ttft_metric)
        row = {"shape": m['shape'], "stack": m['architecture'], "seed": m['seed'], "rate": m['offered'], "definition": d,
               "source": m['source'], "run": m['run'], "instance_type": m['instance_type'], "region": m['region'],
               "evidence_scope": m['evidence_scope'], "model_revision": m['model_revision'], "shape_sha256": m['shape_sha256'],
               "ttft_bound_ms": ttft_ms, "tpot_bound_ms": tpot_ms, **s,
               "offered_window_s": m['offered_window_s'], "offered_task_count": m['offered_task_count']}
        if d != 'mean' and not available:
            for key in ("meets_joint_slo", "slo_attainment_fraction", "useful_calls_per_s_per_gpu"):
                row[key] = UNAVAILABLE
        row["schema_version"] = m.get('schema_version')
        rows.append({**{k: row[k] for k in SUMMARY_LEADING_FIELDS}, **row})
    return rows


def wide_evidence(shape_long_rows):
    """One evidence row per measured file (keyed by source, so repeats never merge), rules side by side."""
    by_key = {}
    for r in shape_long_rows:
        entry = by_key.setdefault(r["source"], {"architecture": r["stack"], "offered": r["rate"], "seed": r["seed"], "run": r["run"],
                                                 "planned_calls": r["planned_calls"], "completed_calls": r["completed_calls"],
                                                 "gap_count": r["gap_count"], "gap_ms_p99": r["gap_ms_p99"],
                                                 "longest_gap_ms_p50": r["longest_gap_ms_p50"], "longest_gap_ms_p90": r["longest_gap_ms_p90"],
                                                 "gap_above_bound_fraction": r["gap_above_bound_fraction"],
                                                 "instance_type": r["instance_type"], "gpu_count": r["gpu_count"],
                                                 "measurement_including_drain_s": r["measurement_including_drain_s"],
                                                 "p90_ttft_ms_success_only": r["p90_ttft_ms_success_only"],
                                                 "p95_ttft_ms_success_only": r["p95_ttft_ms_success_only"],
                                                 "p99_ttft_ms_success_only": r["p99_ttft_ms_success_only"],
                                                 "p90_tpot_ms_success_only": r["p90_tpot_ms_success_only"],
                                                 "itl_ms_p99": r["itl_ms_p99"]})
        entry[f"{r['definition']}_pass"] = r["meets_joint_slo"]
        entry[f"{r['definition']}_goodput"] = r["useful_calls_per_s_per_gpu"]
        # Masked with the verdict: under itl-p99 attainment alone says nothing about the pooled gate.
        entry[f"{r['definition']}_attainment"] = UNAVAILABLE if r["meets_joint_slo"] == UNAVAILABLE else r["slo_attainment_fraction"]
    return [by_key[k] for k in sorted(by_key)]


def score_shape(group, ttft_axis, tpot_axis, given):
    """Objective, qualifying rates and, for a matched unified/disaggregated pair, the promise grid.
    SystemExit means nothing in this shape can be scored."""
    shape = group[0]['shape']
    archs = check_completeness(shape, group)
    objectives = {file_objective(m, given) for m in group}
    if len(objectives) != 1:
        differing = [flag for i, (_key, flag, _name) in enumerate(OBJECTIVE_FIELDS) if len({o[i] for o in objectives}) > 1]
        raise SystemExit(f"{shape}: runs were recorded with different values for {', '.join(differing)}; pass them to score at one objective")
    ttft_ms, tpot_ms, attainment, ttft_metric = objectives.pop()
    objective = (ttft_ms, tpot_ms)
    rows_by_arch = {arch: [m for m in group if m['architecture'] == arch] for arch in archs}
    availability = availability_by_arch(rows_by_arch)

    qualified = {}
    for d in TPOT_METRICS:
        for arch in archs:
            qualified[(arch, d)] = (UNAVAILABLE if not availability[(arch, d)] else
                                    highest_qualifying(rows_by_arch[arch], ttft_ms, tpot_ms, attainment, score_factory_for(d, ttft_metric)))

    comparison_note, grid = None, None
    if archs != ['disaggregated', 'unified']:
        comparison_note = f"{shape}: a promise map compares unified with disaggregated, found {archs}; the per-architecture qualifying rate above is still reported, with no promise map"
    else:
        protocol = matched_protocol(group, archs)
        if protocol[archs[0]] != protocol[archs[1]]:
            comparison_note = (f"{shape}: the architectures were not run with the same seeds and planned rates; "
                                "the per-architecture qualifying rate above is still reported, with no promise map")
        else:
            grid = promise_grid(rows_by_arch, ttft_axis, tpot_axis, attainment, lambda d: score_factory_for(d, ttft_metric), availability)
    return archs, objective, attainment, ttft_metric, qualified, grid, comparison_note


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="Directories containing rate-measured.json files")
    p.add_argument("--output", required=True)
    p.add_argument("--config", help="Also collect live resources and engine logs from the owned namespace; omit for an offline re-render")
    p.add_argument("--ttft-ms", type=float, nargs="+", default=list(DEFAULT_TTFT_AXIS_MS), help="report.html promise-map TTFT columns, ms")
    p.add_argument("--tpot-ms", type=float, nargs="+", default=list(DEFAULT_TPOT_AXIS_MS), help="report.html promise-map TPOT rows, ms/output token")
    p.add_argument("--no-report", action="store_true", help="Skip report.html; summary.csv and qualified-rates.json are unaffected")
    p.add_argument("--ttft-slo-ms", type=float, help="TTFT bound, ms. Default: the one recorded with the runs")
    p.add_argument("--tpot-slo-ms", type=float, help="TPOT bound, ms/output token. Default: the one recorded with the runs")
    p.add_argument("--attainment-fraction", type=float, help="Share of planned calls that must be good, e.g. 0.9. Default: recorded")
    p.add_argument("--ttft-metric", choices=TTFT_METRICS, help="per-call, or a pooled p95/p99 per rate. Default: recorded")
    a = p.parse_args()
    for flag, value in (("--ttft-slo-ms", a.ttft_slo_ms), ("--tpot-slo-ms", a.tpot_slo_ms)):
        if value is not None and not (math.isfinite(value) and value > 0):
            p.error(f"{flag} needs a finite positive bound")
    if a.attainment_fraction is not None and not (math.isfinite(a.attainment_fraction) and 0 < a.attainment_fraction <= 1):
        p.error("--attainment-fraction must be in (0, 1]; use 0.9 for 90 percent")
    given = {_attr(flag): getattr(a, _attr(flag)) for _key, flag, _name in OBJECTIVE_FIELDS}
    objective_given = [name for _key, flag, name in OBJECTIVE_FIELDS if given[_attr(flag)] is not None]
    for name, attr in (("--ttft-ms", "ttft_ms"), ("--tpot-ms", "tpot_ms")):
        try:
            setattr(a, attr, list(validate_grid_axis(getattr(a, attr))))
        except ValueError as exc:
            p.error(f"{name} {exc}")
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    loaded = load_rows(a.runs)
    # An unparseable file stops everything (load_rows); a file that no longer reproduces its stored
    # summary is refused alone, so the other files still get their rows.
    measured, refused_files = [], {}
    for m in loaded:
        try:
            check_reproduces(m)
            measured.append(m)
        except SystemExit as exc:
            refused_files[m['source']] = str(exc)

    # Before shape grouping, so a refused shape's files still get summary.csv rows.
    long_rows = [row for m in measured for row in row_evidence(m, given)]
    write_csv(out / "summary.csv", long_rows)

    qualified_by_definition, objectives_by_shape, refused, report_sections = {d: {} for d in TPOT_METRICS}, {}, {}, []
    for shape in sorted({m['shape'] for m in measured}):
        group = [m for m in measured if m['shape'] == shape]
        try:
            archs, objective, attainment, ttft_metric, qualified, grid, comparison_note = score_shape(group, a.ttft_ms, a.tpot_ms, given)
        except SystemExit as refusal:
            refused[shape] = str(refusal)
            continue
        objectives_by_shape[shape] = {"ttft_bound_ms": objective[0], "tpot_bound_ms": objective[1],
                                       "attainment_target_fraction": attainment, "ttft_metric": ttft_metric}
        for d in TPOT_METRICS:
            qualified_by_definition[d][shape] = {arch: {"highest_qualifying_offered_tasks_per_s": _rate(qualified[(arch, d)]),
                                                         "useful_calls_per_s_per_gpu": _goodput(qualified[(arch, d)])}
                                                  for arch in archs}
        print(f"\n{shape}: {', '.join(sorted(archs))} at TTFT<={objective[0]:g}ms ({ttft_metric}), TPOT<={objective[1]:g}ms, attainment target {attainment:g}")
        print("Qualifying rate (tasks/s):")
        print("stack           " + "".join(f"{d:>14}" for d in TPOT_METRICS))
        for arch in sorted(archs):
            print(f"{arch:<16}" + "".join(f"{_console_rate(qualified[(arch, d)]):>14}" for d in TPOT_METRICS))
        print("Goodput at that rate (useful calls/s/gpu):")
        print("stack           " + "".join(f"{d:>14}" for d in TPOT_METRICS))
        for arch in sorted(archs):
            print(f"{arch:<16}" + "".join(f"{_console_goodput(qualified[(arch, d)]):>14}" for d in TPOT_METRICS))
        if comparison_note:
            print(comparison_note)
        if grid is not None:
            report_sections.append({"shape": shape, "archs": archs, "objective": objective, "attainment": attainment, "ttft_metric": ttft_metric,
                                     "objective_given": objective_given,
                                     "qualified": {(a_, d): qualified[(a_, d)] for a_ in archs for d in TPOT_METRICS},
                                     "grid": grid, "ttft_axis": a.ttft_ms, "tpot_axis": a.tpot_ms,
                                     "evidence": wide_evidence([r for r in long_rows if r["shape"] == shape])})

    (out / "qualified-rates.json").write_text(json.dumps({
        "objectives_by_shape": objectives_by_shape, "by_definition": qualified_by_definition, "refused": refused}, indent=2))
    print(f"\nWrote {out / 'summary.csv'}, {out / 'qualified-rates.json'}")

    # A stale report.html would look current next to this run's files.
    report_path = out / "report.html"
    if a.no_report or not report_sections:
        if report_path.exists():
            report_path.unlink()
    else:
        try:
            report_path.write_text(report.render_sections(report_sections))
            print(f"Wrote {report_path.resolve()}")
        except Exception as exc:
            if report_path.exists():
                report_path.unlink()
            print(f"report.html rendering failed, continuing without it ({type(exc).__name__}: {exc})")

    # Live evidence after the result files, so a failing kubectl call cannot lose them.
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
    print(json.dumps({"summary_csv": str(out / "summary.csv"), "measured_files": len(measured),
                       "qualified_rates": str(out / "qualified-rates.json"),
                       "report_html": str(report_path) if report_path.exists() else None,
                       "refused_files": refused_files}))

    exit_reasons = []
    if refused_files:
        exit_reasons.append(f"{len(refused_files)} file(s) refused by check_reproduces (not included in summary.csv or "
                             "anything scored above -- every other measured file still got its rows):\n"
                             + "\n".join(f"{source}: {reason}" for source, reason in sorted(refused_files.items())))
    if refused:
        exit_reasons.append(f"{len(refused)} shape(s) refused; summary.csv lists every measured file "
                             "regardless, report.html only the shapes that could be grouped, and "
                             "qualified-rates.json's \"refused\" names each reason:\n"
                             + "\n".join(f"{shape}: {reason}" for shape, reason in sorted(refused.items())))
    if exit_reasons:
        raise SystemExit("\n\n".join(exit_reasons))


if __name__ == "__main__":
    main()
