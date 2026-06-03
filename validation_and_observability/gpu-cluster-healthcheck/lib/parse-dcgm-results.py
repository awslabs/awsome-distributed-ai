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
    "RESET":   "Attempt GPU reset via nvidia-smi --gpu-reset",
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

    # Find matching closing brace
    depth = 0
    for i, ch in enumerate(raw_input[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw_input[start : i + 1])
                except json.JSONDecodeError:
                    break

    raise ValueError("Unable to parse JSON from dcgmi output")


def classify_results(dcgm_data: dict, diag_level: int) -> dict:
    """Classify DCGM diagnostic results by severity.

    Args:
        dcgm_data: Parsed DCGM JSON output
        diag_level: Diagnostic level (2 or 4)

    Returns:
        Structured result dict with per-GPU and overall classification
    """
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "diag_level": diag_level,
        # Canonical fields consumed by the aggregator
        "status": "PASS",
        "severity": "PASS",
        # Detailed fields for richer consumers
        "overall_status": "PASS",
        "overall_severity": "PASS",
        "overall_action": ACTIONS["PASS"],
        "test_summary": [],
        "warnings": [],
    }

    max_severity_level = 0

    # Navigate DCGM JSON structure
    # DCGM output structure varies by version; handle common formats
    tests = []

    if "DCGM GPU Diagnostic" in dcgm_data:
        diag_root = dcgm_data["DCGM GPU Diagnostic"]
        if "test_categories" in diag_root:
            for category in diag_root["test_categories"]:
                cat_name = category.get("category", "unknown")
                for test in category.get("tests", []):
                    tests.append({
                        "category": cat_name,
                        "test": test,
                    })
    elif "categories" in dcgm_data:
        for category in dcgm_data["categories"]:
            cat_name = category.get("category", "unknown")
            for test in category.get("tests", []):
                tests.append({
                    "category": cat_name,
                    "test": test,
                })
    elif "tests" in dcgm_data:
        for test in dcgm_data["tests"]:
            tests.append({
                "category": "unknown",
                "test": test,
            })

    # Process each test
    for test_entry in tests:
        test = test_entry["test"]
        test_name = test.get("name", "unknown")
        human_name = DCGM_TEST_NAMES.get(test_name, test_name)

        test_result = {
            "name": test_name,
            "display_name": human_name,
            "category": test_entry["category"],
            "status": "PASS",
            "severity": "PASS",
            "gpu_details": [],
        }
        test_max_severity_level = 0

        # Check per-GPU results
        results_list = test.get("results", [])
        for gpu_result in results_list:
            gpu_id = gpu_result.get("gpu_id", gpu_result.get("gpuId", "N/A"))
            status = gpu_result.get("status", "PASS")
            warning = gpu_result.get("warning", "")
            raw_warning_level = gpu_result.get("warning_level", 0)
            info = gpu_result.get("info", "")

            # Safely coerce warning_level to int (DCGM may emit it as a string)
            try:
                warning_level = int(raw_warning_level)
            except (TypeError, ValueError):
                warning_level = 0

            gpu_entry = {
                "gpu_id": gpu_id,
                "status": status,
                "warning": warning,
                "warning_level": warning_level,
                "info": info,
            }

            # Classify severity
            if warning_level > 0:
                severity = SEVERITY_MAP.get(warning_level, "MONITOR")
                gpu_entry["severity"] = severity
                gpu_entry["action"] = ACTIONS.get(severity, "Review manually")

                max_severity_level = max(max_severity_level, warning_level)
                test_max_severity_level = max(test_max_severity_level, warning_level)

                if status.upper() == "FAIL":
                    test_result["status"] = "FAIL"
            elif status.upper() == "FAIL":
                gpu_entry["severity"] = "ISOLATE"
                gpu_entry["action"] = ACTIONS["ISOLATE"]
                max_severity_level = max(max_severity_level, 3)
                test_max_severity_level = max(test_max_severity_level, 3)
                test_result["status"] = "FAIL"
            else:
                gpu_entry["severity"] = "PASS"

            test_result["gpu_details"].append(gpu_entry)

            if warning:
                result["warnings"].append({
                    "gpu_id": gpu_id,
                    "test": test_name,
                    "warning": warning,
                    "level": warning_level,
                })

        # Set per-test severity as the max over all its GPU entries
        if test_max_severity_level > 0:
            test_result["severity"] = SEVERITY_MAP.get(
                test_max_severity_level, "MONITOR"
            )

        result["test_summary"].append(test_result)

    # Set overall severity
    if max_severity_level > 0:
        overall_severity = SEVERITY_MAP.get(max_severity_level, "MONITOR")
        result["overall_status"] = "FAIL" if max_severity_level >= 2 else "WARN"
        result["overall_severity"] = overall_severity
        result["overall_action"] = ACTIONS.get(overall_severity, "Review manually")
    else:
        # Check if any tests had non-PASS status without warning levels
        has_failures = any(
            t["status"] == "FAIL" for t in result["test_summary"]
        )
        if has_failures:
            result["overall_status"] = "FAIL"
            result["overall_severity"] = "ISOLATE"
            result["overall_action"] = ACTIONS["ISOLATE"]

    # Keep status/severity in sync with overall_* for aggregator compatibility
    result["status"] = result["overall_status"]
    result["severity"] = result["overall_severity"]

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
