#!/usr/bin/env python3
"""Determine overall severity from check result files in a results directory.

Reads check-*.json files and outputs status:severity (e.g., "pass:PASS" or
"fail:ISOLATE") to stdout.  Uses the same SEVERITY_PRIORITY as
lib/aggregate-results.py.
"""

import glob
import json
import os
import sys

SEVERITY_PRIORITY = {
    "ISOLATE": 4,
    "REBOOT": 3,
    "RESET": 2,
    "MONITOR": 1,
    "PASS": 0,
    "SKIP": 0,
    "": 0,
}

SEVERITY_NAMES = {v: k for k, v in SEVERITY_PRIORITY.items() if k}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Determine severity from check results")
    parser.add_argument(
        "--results-dir",
        default="/tmp/gpu-healthcheck-agent",
        help="Directory containing check-*.json files",
    )
    args = parser.parse_args()

    pattern = os.path.join(args.results_dir, "check-*.json")
    files = sorted(glob.glob(pattern))

    if not files:
        print("fail:RESET")
        return

    max_severity = 0
    has_fail = False
    has_warn = False

    for filepath in files:
        try:
            with open(filepath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            has_fail = True
            max_severity = max(max_severity, SEVERITY_PRIORITY.get("RESET", 2))
            continue

        status = data.get("status") or data.get("overall_status") or "UNKNOWN"
        severity = data.get("severity") or data.get("overall_severity") or ""

        if status == "FAIL":
            has_fail = True
        elif status == "WARN":
            has_warn = True

        sev_val = SEVERITY_PRIORITY.get(severity, 0)
        if sev_val > max_severity:
            max_severity = sev_val

    severity_name = SEVERITY_NAMES.get(max_severity, "PASS")

    if has_fail:
        print(f"fail:{severity_name}")
    elif has_warn:
        print(f"warn:{severity_name}")
    else:
        print(f"pass:{severity_name}")


if __name__ == "__main__":
    main()
