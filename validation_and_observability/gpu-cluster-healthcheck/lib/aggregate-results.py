#!/usr/bin/env python3
"""Aggregate per-node GPU health check results into a cluster summary.

Reads individual check result JSON files from a results directory and
produces a consolidated cluster-level report with per-node status.

Usage:
    python3 aggregate-results.py --results-dir /tmp/gpu-healthcheck-12345
    python3 aggregate-results.py --results-dir /tmp/gpu-healthcheck-12345 --format table
"""

import argparse
import glob
import json
import os
from datetime import datetime, timezone

SEVERITY_PRIORITY = {
    "ISOLATE": 4,
    "REBOOT": 3,
    "RESET": 2,
    "MONITOR": 1,
    "PASS": 0,
    "SKIP": 0,
    "": 0,
}

ACTIONS = {
    "ISOLATE": "Drain node from Slurm, initiate instance replacement",
    "REBOOT":  "Drain/cordon node, reboot (scontrol reboot nextstate=resume)",
    "RESET":   "Attempt GPU reset via nvidia-smi --gpu-reset",
    "MONITOR": "Keep in service, flag for review",
    "PASS":    "No action required",
}


def load_results(results_dir: str) -> list:
    """Load all check result JSON files from a results directory."""
    results = []
    pattern = os.path.join(results_dir, "check-*.json")
    for filepath in sorted(glob.glob(pattern)):
        try:
            with open(filepath) as f:
                data = json.load(f)
                data["_source_file"] = os.path.basename(filepath)
                results.append(data)
        except (json.JSONDecodeError, OSError) as e:
            results.append({
                "_source_file": os.path.basename(filepath),
                "status": "ERROR",
                "error": str(e),
            })
    return results


def aggregate_node_results(results: list) -> dict:
    """Aggregate results for a single node."""
    node_summary = {
        "hostname": "",
        "instance_type": "",
        "checks": [],
        "overall_status": "PASS",
        "overall_severity": "PASS",
        "overall_action": ACTIONS["PASS"],
        "pass_count": 0,
        "fail_count": 0,
        "warn_count": 0,
        "skip_count": 0,
        "error_count": 0,
    }

    # Empty results directory means checks didn't run -- treat as FAIL
    if not results:
        node_summary["overall_status"] = "FAIL"
        node_summary["overall_severity"] = "RESET"
        node_summary["overall_action"] = ACTIONS["RESET"]
        node_summary["checks"].append({
            "check": "(no results)",
            "status": "FAIL",
            "severity": "RESET",
            "details": "No check-*.json files found -- checks may not have run",
        })
        node_summary["fail_count"] = 1
        return node_summary

    max_severity = 0

    for result in results:
        if not node_summary["hostname"] and result.get("hostname"):
            node_summary["hostname"] = result["hostname"]
        if not node_summary["instance_type"] and result.get("instance_type"):
            node_summary["instance_type"] = result["instance_type"]

        # Normalize schema: fall back from overall_status/overall_severity
        # to status/severity so richer producers (parse-dcgm-results.py) work.
        status = (
            result.get("status")
            or result.get("overall_status")
            or "UNKNOWN"
        )
        severity = (
            result.get("severity")
            or result.get("overall_severity")
            or ""
        )

        check_entry = {
            "check": result.get("check", result.get("_source_file", "unknown")),
            "status": status,
            "severity": severity,
            "details": result.get("details", ""),
        }
        node_summary["checks"].append(check_entry)

        if status == "PASS":
            node_summary["pass_count"] += 1
        elif status == "FAIL":
            node_summary["fail_count"] += 1
        elif status == "WARN":
            node_summary["warn_count"] += 1
        elif status == "SKIP":
            node_summary["skip_count"] += 1
        else:
            # ERROR, UNKNOWN, or any unrecognised status -- treat as FAIL
            # so we never silently report a healthy node on bad data.
            node_summary["error_count"] += 1
            node_summary["fail_count"] += 1

        sev_priority = SEVERITY_PRIORITY.get(severity, 0)
        if sev_priority > max_severity:
            max_severity = sev_priority

    if max_severity > 0:
        for sev_name, sev_val in SEVERITY_PRIORITY.items():
            if sev_val == max_severity:
                node_summary["overall_severity"] = sev_name
                node_summary["overall_action"] = ACTIONS.get(sev_name, "Review manually")
                break

    if node_summary["fail_count"] > 0:
        node_summary["overall_status"] = "FAIL"
    elif node_summary["warn_count"] > 0:
        node_summary["overall_status"] = "WARN"

    return node_summary


def aggregate_cluster(results_dirs: list) -> dict:
    """Aggregate results across multiple nodes into cluster summary."""
    cluster = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "node_count": len(results_dirs),
        "nodes": [],
        "overall_status": "PASS",
        "overall_severity": "PASS",
        "overall_action": ACTIONS["PASS"],
        "summary": {
            "nodes_pass": 0,
            "nodes_fail": 0,
            "nodes_warn": 0,
        },
    }

    max_severity = 0

    for results_dir in results_dirs:
        results = load_results(results_dir)
        node_summary = aggregate_node_results(results)
        cluster["nodes"].append(node_summary)

        if node_summary["overall_status"] == "FAIL":
            cluster["summary"]["nodes_fail"] += 1
        elif node_summary["overall_status"] == "WARN":
            cluster["summary"]["nodes_warn"] += 1
        else:
            cluster["summary"]["nodes_pass"] += 1

        sev_priority = SEVERITY_PRIORITY.get(node_summary["overall_severity"], 0)
        if sev_priority > max_severity:
            max_severity = sev_priority

    if max_severity > 0:
        for sev_name, sev_val in SEVERITY_PRIORITY.items():
            if sev_val == max_severity:
                cluster["overall_severity"] = sev_name
                cluster["overall_action"] = ACTIONS.get(sev_name, "Review manually")
                break

    if cluster["summary"]["nodes_fail"] > 0:
        cluster["overall_status"] = "FAIL"
    elif cluster["summary"]["nodes_warn"] > 0:
        cluster["overall_status"] = "WARN"

    return cluster


def format_table(cluster: dict) -> str:
    """Format cluster results as a text table."""
    lines = []
    lines.append(f"GPU Health Check Cluster Summary -- {cluster['timestamp']}")
    lines.append(f"{'=' * 72}")
    lines.append(
        f"Nodes: {cluster['node_count']}  |  "
        f"Pass: {cluster['summary']['nodes_pass']}  |  "
        f"Fail: {cluster['summary']['nodes_fail']}  |  "
        f"Warn: {cluster['summary']['nodes_warn']}"
    )
    lines.append(
        f"Overall: {cluster['overall_status']} "
        f"(severity: {cluster['overall_severity']})"
    )
    lines.append(f"Action: {cluster['overall_action']}")
    lines.append(f"{'=' * 72}")
    lines.append("")

    for node in cluster["nodes"]:
        status_indicator = {
            "PASS": "[PASS]",
            "FAIL": "[FAIL]",
            "WARN": "[WARN]",
        }.get(node["overall_status"], "[????]")

        lines.append(
            f"{status_indicator} {node['hostname']} "
            f"({node['instance_type']}) -- {node['overall_severity']}"
        )
        for check in node["checks"]:
            check_status = {
                "PASS": "  OK ",
                "FAIL": " FAIL",
                "WARN": " WARN",
                "SKIP": " SKIP",
            }.get(check["status"], " ??? ")
            lines.append(f"  [{check_status}] {check['check']}: {check['details']}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate GPU health check results into cluster summary"
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        nargs="+",
        help="One or more results directories to aggregate",
    )
    parser.add_argument(
        "--format",
        choices=["json", "table"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--output",
        help="Output file path (default: stdout)",
    )
    args = parser.parse_args()

    cluster = aggregate_cluster(args.results_dir)

    if args.format == "table":
        output = format_table(cluster)
    else:
        output = json.dumps(cluster, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
            f.write("\n")
    else:
        print(output)


if __name__ == "__main__":
    main()
