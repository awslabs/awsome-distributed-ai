#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Send Prometheus text exposition to a Pushgateway group, delete a group, or list groups.

  some-producer | python3 push.py put --job J --group k=v [--group k=v ...] --via kubectl --namespace N
  python3 push.py put    --job J --group run=1 --via url --url http://pushgateway.N.svc:9091 < metrics.txt
  python3 push.py delete --job J --group run=1 --via kubectl --namespace N
  python3 push.py list   [--job J] [--group k=v ...] --via kubectl --namespace N

put replaces the whole group (HTTP PUT), so send every series of a group in one call. Pushgateway
never expires groups: delete the groups of a finished run. --via kubectl sends through the Kubernetes
API server's Service proxy, so it works from any host with a kubeconfig and needs, on services/proxy
in the monitoring namespace, update for put, delete for delete and get for list. --via url sends
directly, for example from a pod in the cluster. Standard library only.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import re
import subprocess
import sys
import urllib.error
import urllib.request

METRIC = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")
LABEL = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
PLAIN = re.compile(r"^[A-Za-z0-9_.-]+$")
SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{(.*)\})?\s+(\S+)(\s+-?\d+)?$")
LABEL_PAIR = re.compile(r'\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"((?:[^"\\]|\\.)*)"\s*(,|$)')
VERB = {"put": "update", "delete": "delete", "list": "get"}


class Invalid(ValueError):
    pass


def parse_labels(text: str) -> dict:
    out, pos = {}, 0
    while pos < len(text):
        m = LABEL_PAIR.match(text, pos)
        if not m:
            raise Invalid(f"cannot parse labels {{{text}}}")
        name = m.group(1)
        value = unescape(m.group(2))
        if name.startswith("__"):
            raise Invalid(f"label {name} is reserved")
        if name in out:
            raise Invalid(f"label {name} appears twice")
        out[name] = value
        pos = m.end()
    return out


def unescape(text: str) -> str:
    """A label value as the exposition format escapes it: \\\\, \\" and \\n."""
    out, i = [], 0
    while i < len(text):
        c = text[i]
        if c == "\\":
            if i + 1 >= len(text) or text[i + 1] not in '\\"n':
                raise Invalid(f"unsupported escape in label value {text!r}")
            out.append("\n" if text[i + 1] == "n" else text[i + 1])
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def validate(body: str, group: dict, job: str = "") -> int:
    """Check what the Pushgateway would reject, and a little more; returns the number of samples."""
    types, seen, n = {}, set(), 0
    for lineno, line in enumerate(body.splitlines(), 1):
        if not line.strip():
            continue
        if line.startswith("#"):
            parts = line.split(None, 3)
            if len(parts) >= 4 and parts[1] == "TYPE":
                if parts[2] in types:
                    raise Invalid(f"line {lineno}: second TYPE for {parts[2]}")
                if parts[3] not in ("gauge", "untyped"):
                    raise Invalid(f"line {lineno}: {parts[2]} is {parts[3]}; only gauge and untyped are pushed")
                types[parts[2]] = parts[3]
            continue
        m = SAMPLE.match(line)
        if not m:
            raise Invalid(f"line {lineno}: not a sample: {line[:80]}")
        name, labels, value = m.group(1), parse_labels(m.group(3) or ""), m.group(4)
        if m.group(5):
            raise Invalid(f"line {lineno}: {name} carries a timestamp, which the Pushgateway refuses")
        try:
            v = float(value)
        except ValueError:
            raise Invalid(f"line {lineno}: value {value!r} is not a number") from None
        if math.isnan(v) or math.isinf(v):
            raise Invalid(f"line {lineno}: {name} is {value}; push finite values")
        for k, want in dict(group, **({"job": job} if job else {})).items():
            if k in labels and labels[k] != want:
                raise Invalid(f"line {lineno}: {name} has {k}={labels[k]!r} but the group says {want!r}")
        key = (name, tuple(sorted(labels.items())))
        if key in seen:
            raise Invalid(f"line {lineno}: {name}{{{m.group(3) or ''}}} appears twice")
        seen.add(key)
        n += 1
    if not n:
        raise Invalid("no samples on input; to remove a group use delete")
    return n


def segment(key: str, value: str) -> str:
    """One label of the group path, per the Pushgateway URL rules."""
    if not LABEL.match(key):
        raise Invalid(f"group label {key!r} is not a label name")
    if value == "":
        return f"/{key}@base64/="
    if PLAIN.match(value) and value.strip("."):
        return f"/{key}/{value}"
    return f"/{key}@base64/" + base64.urlsafe_b64encode(value.encode()).decode()


def group_path(job: str, group: dict) -> str:
    if not job:
        raise Invalid("--job is empty")
    if "job" in group:
        raise Invalid("job is given by --job, not --group")
    path = "/metrics" + (f"/job/{job}" if PLAIN.match(job) and job.strip(".") else "/job@base64/" + base64.urlsafe_b64encode(job.encode()).decode())
    for k in sorted(group):
        path += segment(k, group[k])
    return path


def send(a: argparse.Namespace, method: str, path: str, body: str | None) -> str:
    if a.via == "url":
        req = urllib.request.Request(a.url.rstrip("/") + path, data=body.encode() if body is not None else None, method=method,
                                     headers={"Content-Type": "text/plain; version=0.0.4"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode()
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"{method} {path}: HTTP {exc.code}: {exc.read().decode()[:500]}") from None
    base = ["kubectl"] + (["--context", a.context] if a.context else [])
    raw = f"/api/v1/namespaces/{a.namespace}/services/{a.service}:http/proxy{path}"
    verb = {"PUT": "replace", "DELETE": "delete", "GET": "get"}[method]
    args = base + [verb, "--raw", raw] + (["-f", "-"] if body is not None else [])
    r = subprocess.run(args, input=body, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"kubectl {verb} --raw {raw}: {(r.stderr or r.stdout).strip()[:500]}")
    return r.stdout


def precheck(a: argparse.Namespace) -> None:
    if a.via != "kubectl":
        return
    base = ["kubectl"] + (["--context", a.context] if a.context else [])
    verb = VERB[a.command]
    # Asked for this Service by name, so a grant narrowed with resourceNames: [<service>:http] passes too.
    r = subprocess.run(base + ["auth", "can-i", verb, f"services/{a.service}:http", "--subresource=proxy", "-n", a.namespace],
                       capture_output=True, text=True)
    if r.stdout.strip() != "yes":
        raise RuntimeError(f"this identity may not {verb} services/proxy for {a.service}:http in {a.namespace}, which {a.command} needs")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("command", choices=["put", "delete", "list"])
    p.add_argument("--job")
    p.add_argument("--group", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--via", choices=["kubectl", "url"], required=True)
    p.add_argument("--url", help="Pushgateway base URL, with --via url")
    p.add_argument("--context")
    p.add_argument("--namespace", default="in-cluster-monitoring")
    p.add_argument("--service", default="pushgateway")
    p.add_argument("--input", default="-", help="exposition file for put; - is stdin")
    a = p.parse_args(argv)
    try:
        group = {}
        for kv in a.group:
            if "=" not in kv:
                raise Invalid(f"--group {kv!r} must be key=value")
            k, v = kv.split("=", 1)
            if k in group:
                raise Invalid(f"--group {k} given twice")
            group[k] = v
        if a.via == "url" and not a.url:
            raise Invalid("--via url needs --url")
        if a.command in ("put", "delete") and not a.job:
            raise Invalid(f"{a.command} needs --job")
        precheck(a)
        if a.command == "list":
            data = json.loads(send(a, "GET", "/api/v1/metrics", None))["data"]
            want = dict(group, **({"job": a.job} if a.job else {}))
            for g in data:
                labels = g["labels"]
                if all(labels.get(k) == v for k, v in want.items()):
                    print(json.dumps({"labels": labels, "push_time_seconds": g.get("push_time_seconds", {}).get("metrics", [{}])[0].get("value")}))
            return 0
        path = group_path(a.job, group)
        if a.command == "delete":
            send(a, "DELETE", path, None)
            print(f"push: deleted group {path}")
            return 0
        body = sys.stdin.read() if a.input == "-" else open(a.input, encoding="utf-8").read()
        n = validate(body, group, a.job)
        send(a, "PUT", path, body if body.endswith("\n") else body + "\n")
        print(f"push: sent {n} samples to {path}")
    except Invalid as exc:
        print(f"push: invalid: {exc}", file=sys.stderr)
        return 2
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        print(f"push: failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
