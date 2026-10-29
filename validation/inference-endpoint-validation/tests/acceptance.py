# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Black-box acceptance runner for the inference endpoint validator.

Reads scenarios.json, starts the mock server per scenario, runs
validate_endpoint.py as a subprocess, and verifies exit codes, JSON
invariants, Markdown format, and secret redaction.

Usage::

    python3 tests/acceptance.py                 # progress; prints METRIC acceptance_pct=<value>
    python3 tests/acceptance.py --require-all   # exits 0 only when 100%
"""

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time

# --- Constants ---

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
VALIDATOR = os.path.join(PROJECT_DIR, "validate_endpoint.py")
MOCK_SERVER = os.path.join(SCRIPT_DIR, "mock_openai_server.py")
SCENARIOS_FILE = os.path.join(SCRIPT_DIR, "scenarios.json")

ALL_CHECK_IDS = [
    "models.list", "server_version", "health.ready", "health.consistent",
    "chat.basic", "chat.determinism", "chat.stream", "chat.stream_usage",
    "chat.stream_parity", "chat.max_tokens", "chat.stop",
    "chat.template_leak", "completions.basic", "tools.call",
    "structured_output", "errors.unknown_model", "errors.malformed_json",
    "auth.required", "stream.cancel", "concurrency.parallel",
    "latency.smoke",
]


def _find_free_port():
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(port, timeout=10):
    """Wait until a port is accepting connections."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.1)
    return False


def _start_mock(mock_args, port):
    """Start the mock server as a subprocess, return (proc, port)."""
    cmd = [sys.executable, MOCK_SERVER, "--port", str(port)] + mock_args
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # Read the port line from stdout
    line = proc.stdout.readline().strip()
    try:
        actual_port = int(line)
    except (ValueError, TypeError):
        actual_port = port
    return proc, actual_port


def _stop_mock(proc):
    """Stop the mock server process."""
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def _run_validator(port, validator_args, env_overrides, json_path, md_path, timeout=30):
    """Run validate_endpoint.py and return (exit_code, stdout, stderr)."""
    cmd = [
        sys.executable, VALIDATOR,
        "--base-url", "http://127.0.0.1:%d" % port,
        "--json", json_path,
        "--markdown", md_path,
        "--timeout", "10",
    ] + validator_args

    env = os.environ.copy()
    # Clear OPENAI_API_KEY by default so it does not leak into tests
    env.pop("OPENAI_API_KEY", None)
    env.update(env_overrides)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
    return proc.returncode, stdout, stderr


def run_scenario(scenario):
    """Run one scenario and return (passed, failures) where failures is a list of strings."""
    name = scenario["name"]
    failures = []

    json_fd, json_path = tempfile.mkstemp(suffix=".json", prefix="iev_")
    md_fd, md_path = tempfile.mkstemp(suffix=".md", prefix="iev_")
    os.close(json_fd)
    os.close(md_fd)

    mock_proc = None
    port = 0

    try:
        mock_args = scenario.get("mock_args")
        env_overrides = scenario.get("env", {})
        validator_args = list(scenario.get("validator_args", []))

        if mock_args is None:
            # Unreachable scenario: no server started
            port = _find_free_port()
        else:
            port = _find_free_port()
            mock_proc, port = _start_mock(mock_args, port)
            if not _wait_for_port(port, timeout=10):
                failures.append("mock server did not start on port %d" % port)
                return (False, failures)

        exit_code, stdout, stderr = _run_validator(
            port, validator_args, env_overrides, json_path, md_path,
        )

        # --- Check exit code ---
        expected_exit = scenario["expected_exit_code"]
        if exit_code != expected_exit:
            failures.append(
                "exit_code: expected %d, got %d" % (expected_exit, exit_code)
            )

        # --- Check JSON report ---
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                report = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError, OSError) as exc:
            failures.append("JSON report unreadable: %s" % exc)
            return (len(failures) == 0, failures)

        # All 21 check IDs present in catalog order
        report_ids = [c["id"] for c in report.get("checks", [])]
        if report_ids != ALL_CHECK_IDS:
            failures.append(
                "JSON checks: expected %d IDs in catalog order, got %s"
                % (len(ALL_CHECK_IDS), report_ids)
            )

        # Check statuses match expected
        status_map = {c["id"]: c["status"] for c in report.get("checks", [])}

        for cid in scenario.get("expected_fail", []):
            actual = status_map.get(cid)
            if actual != "fail":
                failures.append(
                    "check %s: expected fail, got %s" % (cid, actual)
                )
        for cid in scenario.get("expected_warn", []):
            actual = status_map.get(cid)
            if actual != "warn":
                failures.append(
                    "check %s: expected warn, got %s" % (cid, actual)
                )
        for cid in scenario.get("expected_pass", []):
            actual = status_map.get(cid)
            if actual != "pass":
                failures.append(
                    "check %s: expected pass, got %s" % (cid, actual)
                )
        for cid in scenario.get("expected_skip", []):
            actual = status_map.get(cid)
            if actual != "skip":
                failures.append(
                    "check %s: expected skip, got %s" % (cid, actual)
                )

        # Summary counts match
        summary = report.get("summary", {})
        actual_counts = {"pass": 0, "fail": 0, "warn": 0, "skip": 0}
        for c in report.get("checks", []):
            s = c.get("status", "")
            if s in actual_counts:
                actual_counts[s] += 1
        for key in ("pass", "fail", "warn", "skip"):
            if summary.get(key) != actual_counts[key]:
                failures.append(
                    "summary.%s: expected %d, got %s"
                    % (key, actual_counts[key], summary.get(key))
                )

        # exit_code field matches actual
        if report.get("exit_code") != exit_code:
            failures.append(
                "JSON exit_code field: expected %d, got %s"
                % (exit_code, report.get("exit_code"))
            )

        # Unreachable: error must be set
        if scenario["expected_exit_code"] == 2 and mock_args is None:
            if not report.get("error"):
                failures.append("unreachable scenario: 'error' field not set in JSON")

        # --- Check Markdown report ---
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                md_text = f.read()
        except (FileNotFoundError, OSError) as exc:
            failures.append("Markdown report unreadable: %s" % exc)
            md_text = ""

        if md_text:
            md_lines = md_text.split("\n")
            # First line must be exactly "## Test Results"
            if not md_lines or md_lines[0] != "## Test Results":
                failures.append(
                    "Markdown: first line expected '## Test Results', got %r"
                    % (md_lines[0] if md_lines else "")
                )
            # Table header
            header_line = "| Check | Severity | Status | Detail |"
            if header_line not in md_text:
                failures.append("Markdown: table header not found")
            # One row per check with ID in backticks
            for cid in ALL_CHECK_IDS:
                pattern = "| `%s` |" % cid
                if pattern not in md_text:
                    failures.append("Markdown: missing row for check %s" % cid)
            # Summary line present
            if "Summary:" not in md_text:
                failures.append("Markdown: 'Summary:' line not found")

        # --- Secret leak check ---
        secret = env_overrides.get("OPENAI_API_KEY", "")
        if secret:
            for label, content in [
                ("JSON", json.dumps(report)),
                ("Markdown", md_text),
                ("stdout", stdout),
                ("stderr", stderr),
            ]:
                if secret in content:
                    failures.append("secret leaked in %s" % label)

    finally:
        _stop_mock(mock_proc)
        for p in (json_path, md_path):
            try:
                os.unlink(p)
            except OSError:
                pass

    return (len(failures) == 0, failures)


def main():
    require_all = "--require-all" in sys.argv

    with open(SCENARIOS_FILE, "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    total = len(scenarios)
    passed = 0
    failed_names = []

    for i, scenario in enumerate(scenarios):
        name = scenario["name"]
        sys.stdout.write("[%d/%d] %s ... " % (i + 1, total, name))
        sys.stdout.flush()
        ok, failures = run_scenario(scenario)
        if ok:
            passed += 1
            print("PASS")
        else:
            failed_names.append(name)
            print("FAIL")
            for f in failures:
                print("  - %s" % f)

    pct = (passed / total * 100) if total > 0 else 0
    print()
    print("METRIC acceptance_pct=%.1f" % pct)
    print("METRIC scenarios_passed=%d" % passed)
    print("METRIC scenarios_total=%d" % total)

    if failed_names:
        print()
        print("Failed scenarios: %s" % ", ".join(failed_names))

    if require_all and passed < total:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
