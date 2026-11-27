#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Write the bundled Grafana dashboards: python3 gen_dashboards.py [--check].

monitoring-health.json shows the stack itself (targets, scrapes, Pushgateway groups), so an install
with no exporter still has a working Home. gpu-dcgm.json reads an existing DCGM exporter, for example
the GPU Operator's, added with --scrape-service; this tool installs no exporter.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DS = {"type": "prometheus", "uid": "prometheus"}


def panel(pid, title, kind, exprs, x, y, w=12, h=8, unit=None, legend=None, instant=False):
    p = {"id": pid, "title": title, "type": kind, "datasource": DS, "gridPos": {"x": x, "y": y, "w": w, "h": h},
         "targets": [{"refId": chr(65 + i), "expr": e, "legendFormat": (legend or [""] * len(exprs))[i],
                      "instant": instant, "range": not instant} for i, e in enumerate(exprs)],
         "fieldConfig": {"defaults": {"unit": unit} if unit else {}, "overrides": []}, "options": {}}
    if kind == "table":
        p["transformations"] = [{"id": "merge"}, {"id": "organize", "options": {"excludeByName": {"Time": True}}}]
        for t in p["targets"]:
            t["format"] = "table"
    return p


def dashboard(uid, title, panels, variables=()):
    return {"uid": uid, "title": title, "schemaVersion": 39, "version": 1, "editable": False, "tags": ["eks-in-cluster-monitoring"],
            "time": {"from": "now-1h", "to": "now"}, "refresh": "30s", "timezone": "browser",
            "templating": {"list": list(variables)}, "panels": panels}


def health():
    return dashboard("monitoring-health", "Monitoring health", [
        panel(1, "Scrape targets up", "table", ["max by (job, namespace, service, instance) (up)"], 0, 0, 24, 8, instant=True),
        panel(2, "Scrape duration", "timeseries", ["max by (job) (scrape_duration_seconds)"], 0, 8, unit="s", legend=["{{job}}"]),
        panel(3, "Samples scraped per job", "timeseries", ["sum by (job) (scrape_samples_scraped)"], 12, 8, legend=["{{job}}"]),
        panel(4, "Samples ingested per second", "timeseries", ["rate(prometheus_tsdb_head_samples_appended_total[5m])"], 0, 16, unit="ops"),
        panel(5, "TSDB size", "timeseries", ["prometheus_tsdb_storage_blocks_bytes + prometheus_tsdb_wal_storage_size_bytes"], 12, 16, unit="bytes"),
        # One row per group with every grouping label (aggregating would merge groups that differ in a key such
        # as run); push_time_seconds is epoch seconds and Grafana's date units take milliseconds.
        panel(6, "Pushgateway groups: last push", "table", ["push_time_seconds * 1000"], 0, 24, 24, 8, unit="dateTimeAsIso", instant=True),
    ])


def gpu():
    v = [{"name": "instance", "type": "query", "datasource": DS, "query": "label_values(DCGM_FI_DEV_GPU_UTIL, Hostname)",
          "refresh": 2, "multi": True, "includeAll": True, "current": {"text": "All", "value": "$__all"}}]
    sel = '{Hostname=~"$instance"}'
    return dashboard("gpu-dcgm", "GPUs (DCGM exporter)", [
        panel(1, "GPU utilisation", "timeseries", [f"DCGM_FI_DEV_GPU_UTIL{sel}"], 0, 0, unit="percent", legend=["{{Hostname}} GPU {{gpu}}"]),
        panel(2, "Framebuffer used", "timeseries", [f"DCGM_FI_DEV_FB_USED{sel}"], 12, 0, unit="decmbytes", legend=["{{Hostname}} GPU {{gpu}}"]),
        panel(3, "Power", "timeseries", [f"DCGM_FI_DEV_POWER_USAGE{sel}"], 0, 8, unit="watt", legend=["{{Hostname}} GPU {{gpu}}"]),
        panel(4, "Temperature", "timeseries", [f"DCGM_FI_DEV_GPU_TEMP{sel}"], 12, 8, unit="celsius", legend=["{{Hostname}} GPU {{gpu}}"]),
        panel(5, "XID errors (last value)", "table", [f"max by (Hostname, gpu) (DCGM_FI_DEV_XID_ERRORS{sel})"], 0, 16, 24, 8, instant=True),
    ], v)


FILES = {"monitoring-health.json": health, "gpu-dcgm.json": gpu}


def main():
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("--check", action="store_true", help="exit 1 if a file differs from what this script writes")
    args = a.parse_args()
    stale = []
    for name, build in FILES.items():
        text = json.dumps(build(), indent=2, sort_keys=True) + "\n"
        path = HERE / name
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(name)
        else:
            path.write_text(text)
    if stale:
        print("gen_dashboards: out of date: " + ", ".join(stale) + "; run python3 gen_dashboards.py", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
