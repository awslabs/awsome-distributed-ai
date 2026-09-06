#!/usr/bin/env python3
"""Collect this lab's EKS evidence and summarize measured ramp files."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from deployment import kubectl, owned_namespace, read_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True, help="Directories containing rate-measured.json files")
    p.add_argument("--output", required=True)
    p.add_argument("--config", help="Also collect live resources and engine logs from the owned namespace")
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in a.runs:
        for f in sorted(Path(run).glob("*-measured.json")):
            r = json.loads(f.read_text())
            m = r["metadata"]
            rows.append({"source": str(f), "architecture": m["architecture"], "instance_type": m["instance_type"], "region": m["region"], "evidence_scope": m["evidence_scope"], "model_revision": m["model_revision"], "shape": m["shape"]["name"], "shape_sha256": hashlib.sha256(json.dumps(m["shape"], sort_keys=True).encode()).hexdigest(), **r["summary"]})
    if not rows:
        raise SystemExit("No measured files found; no summary was fabricated")
    with (out / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Group by hardware, workload, thresholds and GPU budget. Never mix the scaling arm with the iso-resource arm.
    groups = {}
    for r in rows:
        key = tuple(r[k] for k in ("architecture", "instance_type", "region", "evidence_scope", "model_revision", "shape_sha256", "gpu_count", "slo_ttft_ms", "slo_tpot_ms", "attainment_target_fraction"))
        groups.setdefault(key, []).append(r)
    qualified = []
    for key, rs in groups.items():
        eligible = [r for r in rs if r["meets_joint_slo"]]
        qualified.append({"group": list(key), "highest_tested_offered_tasks_per_s_meeting_slo": max((r["offered_tasks_per_s"] for r in eligible), default=None), "max_observed_useful_calls_per_s_per_gpu": max(r["useful_calls_per_s_per_gpu"] for r in rs), "qualification": "Finite-window observation including drain; extend duration and repeat before claiming sustained capacity"})
    (out / "qualified-rates.json").write_text(json.dumps(qualified, indent=2))
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
                for endpoint in ("metrics", "server_info"):
                    code = f"import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:30000/{endpoint}').read().decode())"
                    (out / f"{name}-{endpoint}.txt").write_text(kubectl(c, "exec", name, "-c", "engine", "--", "python3", "-c", code, capture_output=True).stdout)
    print(json.dumps({"summary_csv": str(out / "summary.csv"), "measured_rate_rows": len(rows), "qualified_rates": str(out / "qualified-rates.json")}))

if __name__ == "__main__":
    main()
