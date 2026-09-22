#!/usr/bin/env python3
"""Parse DCGM diagnostic JSON output into severity classification.

Reads dcgmi diag -j output from stdin and produces a structured JSON report
with per-GPU results, overall severity, and recommended actions.

Usage:
    dcgmi diag -r 2 -j | python3 parse-dcgm-results.py --level 2
    dcgmi diag -r 4 -j | python3 parse-dcgm-results.py --level 4
"""

import argparse
import json
import sys
from datetime import datetime, timezone

# Severity classification based on DCGM warning levels
SEVERITY_MAP = {
    3: "ISOLATE",   # Critical -- drain and replace
    2: "RESET",     # Recoverable -- reboot and retest
    1: "MONITOR",   # Informational -- log and continue
    0: "PASS",      # No issue
}

# Action mapping for each severity level
ACTIONS = {
    "ISOLATE": "Drain node from Slurm, initiate instance replacement",
    "REBOOT":  "Drain/cordon node, reboot (scontrol reboot nextstate=resume)",
    "RESET":   "Reboot the instance (nvidia-smi --gpu-reset is often insufficient)",
    "MONITOR": "Keep in service, flag for review",
    "PASS":    "No action required",
}

# DCGM test names for human-readable output
DCGM_TEST_NAMES = {
    "deployment": "Deployment Readiness",
    "pcie": "PCIe Bandwidth",
    "memory": "GPU Memory",
    "sm_stress": "SM Stress",
    "diagnostic": "Diagnostic",
    "targeted_stress": "Targeted Stress",
    "targeted_power": "Targeted Power",
    "memory_bandwidth": "Memory Bandwidth",
    "eud": "Extended Utility Diagnostic (EUD)",
    "pulse": "Pulse Power Test",
    "context_create": "Context Create",
    "sm_perf": "SM Performance",
    "membw": "Memory Bandwidth",
}


def parse_dcgm_json(raw_input: str) -> dict:
    """Parse the JSON portion from dcgmi output.

    dcgmi may emit non-JSON text before the JSON payload.
    This function extracts and parses the JSON content.
    """
    # Try parsing the full input first
    try:
        return json.loads(raw_input)
    except json.JSONDecodeError:
        pass

    # Look for JSON object boundaries
    start = raw_input.find("{")
    if start == -1:
        raise ValueError("No JSON object found in dcgmi output")

    try:
        value, _ = json.JSONDecoder().raw_decode(raw_input[start:])
        return value
    except json.JSONDecodeError:
        pass

    raise ValueError("Unable to parse JSON from dcgmi output")


# DCGM error_severity is a different enum from legacy warning_level.
# NVIDIA/DCGM dcgmlib/dcgm_errors.h: MONITOR=1, ISOLATE=2,
# UNKNOWN=3, TRIAGE=4, CONFIG=5, RESET=6. Preserve the original fields.
DCGM_ERROR_SEVERITY = {0: 0, 1: 1, 2: 3, 3: 2, 4: 2, 5: 2, 6: 2}


def classify_results(dcgm_data: dict, diag_level: int) -> dict:
    """Classify actual test records; missing or skipped coverage cannot pass."""
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "diag_level": diag_level,
        "test_summary": [],
        "warnings": [],
    }
    if not isinstance(dcgm_data, dict):
        dcgm_data = {"runtime_error": "DCGM output must be a JSON object"}
    root = dcgm_data.get("DCGM Diagnostic",
                         dcgm_data.get("DCGM GPU Diagnostic", dcgm_data))
    if not isinstance(root, dict):
        root = dcgm_data
    categories = root.get("test_categories", root.get("categories", []))
    tests = [(category.get("category", "unknown"), test)
             for category in categories for test in category.get("tests", [])]
    tests.extend(("unknown", test) for test in root.get("tests", []))
    runtime_errors = [source[key] for source in (dcgm_data, root)
                      for key in ("runtime_error", "error") if source.get(key)]
    global_errors = dcgm_data.get("global_errors", []) or root.get("global_errors", [])
    max_level = 2 if runtime_errors or not tests else 0
    if global_errors:
        result["global_errors"] = global_errors
        for error in global_errors:
            try:
                code = int(error.get("error_severity", -1))
            except (ValueError, TypeError):
                code = -1
            max_level = max(max_level, DCGM_ERROR_SEVERITY.get(code, 2))
    if runtime_errors:
        result["runtime_errors"] = runtime_errors
    if not tests:
        result["error"] = "No diagnostic test results were found"

    def classify_entry(entry):
        status = str(entry.get("status", "UNKNOWN")).upper()
        try:
            level = int(entry.get("warning_level", 0))
        except (ValueError, TypeError):
            level = 2
        level = level if level in SEVERITY_MAP else 2
        warnings = entry.get("warnings", [])
        if isinstance(warnings, dict):
            warnings = [warnings]
        for warning in warnings:
            try:
                value = int(warning.get("error_severity", -1))
            except (ValueError, TypeError):
                value = -1
            level = max(level, DCGM_ERROR_SEVERITY.get(value, 2))
        if status == "FAIL" and level == 0:
            level = 3
        elif status in ("WARN", "WARNING", "SKIP", "SKIPPED", "NOT_RUN", "NOT RUN"):
            level = max(level, 1)
        elif status not in ("PASS", "FAIL"):
            level = max(level, 2)
        return status, level

    for category, test in tests:
        details = []
        levels = []
        states = []
        summary = test.get("test_summary", {})
        entries = test.get("results", [])
        for entry in entries:
            status, level = classify_entry(entry)
            levels.append(level)
            states.append(status)
            details.append(dict(entry, gpu_id=entry.get("gpu_id", entry.get("gpuId", entry.get("entity_id", "N/A"))),
                                severity=SEVERITY_MAP[level], action=ACTIONS[SEVERITY_MAP[level]]))
            for warning in entry.get("warnings", []):
                result["warnings"].append(dict(test=test.get("name", "unknown"),
                                                entity_id=entry.get("entity_id"), raw=warning))
            if entry.get("warning"):
                result["warnings"].append(dict(test=test.get("name", "unknown"),
                                                gpu_id=details[-1]["gpu_id"], raw=entry["warning"]))
        if summary:
            status, level = classify_entry(summary)
            # A failed summary without its own warning repeats entity failures;
            # retain their supplied severity rather than inventing ISOLATE.
            if (status == "FAIL" and "FAIL" in states and not summary.get("warnings")
                    and not summary.get("warning_level")):
                level = max(levels)
            levels.append(level)
            states.append(status)
        # A summary alone cannot prove that any GPU was exercised.
        if not entries:
            levels.append(1 if states and all(s in ("SKIP", "SKIPPED", "NOT_RUN", "NOT RUN") for s in states) else 2)
        level = max(levels, default=2)
        status = "FAIL" if level >= 2 else ("WARN" if level else "PASS")
        result["test_summary"].append(dict(name=test.get("name", "unknown"),
            display_name=DCGM_TEST_NAMES.get(test.get("name"), test.get("name", "unknown")),
            category=category, status=status, severity=SEVERITY_MAP[level],
            raw_summary=summary, gpu_details=details))
        max_level = max(max_level, level)
    severity = SEVERITY_MAP[max_level]
    status = "FAIL" if max_level >= 2 else ("WARN" if max_level else "PASS")
    result.update(status=status, severity=severity, overall_status=status,
                  overall_severity=severity, overall_action=ACTIONS[severity])
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Parse DCGM diagnostic JSON output into severity classification"
    )
    parser.add_argument(
        "--level",
        type=int,
        choices=[1, 2, 3, 4],
        default=2,
        help="DCGM diagnostic level (default: 2)",
    )
    args = parser.parse_args()

    raw_input = sys.stdin.read().strip()
    if not raw_input:
        print(json.dumps({
            "error": "No input received",
            "status": "FAIL",
            "severity": "RESET",
            "overall_status": "FAIL",
            "overall_severity": "RESET",
            "overall_action": ACTIONS["RESET"],
        }), file=sys.stdout)
        sys.exit(1)

    try:
        dcgm_data = parse_dcgm_json(raw_input)
    except ValueError as e:
        print(json.dumps({
            "error": str(e),
            "status": "FAIL",
            "severity": "RESET",
            "overall_status": "FAIL",
            "overall_severity": "RESET",
            "overall_action": ACTIONS["RESET"],
        }), file=sys.stdout)
        sys.exit(1)

    result = classify_results(dcgm_data, args.level)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
