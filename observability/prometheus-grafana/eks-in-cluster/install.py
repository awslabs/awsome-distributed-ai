#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Install, check or remove an in-cluster Prometheus, Pushgateway and Grafana on EKS.

  python3 install.py apply  --owner OWNER [--namespace N] [--scrape-namespace NS ...] [--scrape-service NS/SVC:PORT ...]
  python3 install.py check  [--namespace N] [--show-password]
  python3 install.py delete --owner OWNER [--namespace N]

Every object carries the owner and instance labels; apply and delete refuse an object that
exists without them or with another owner, and delete refuses a namespace that holds anything this
tool did not create. Needs only Python 3.9+ and kubectl.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import secrets
import subprocess
import sys
import time

import monitoring_render as R

NAMESPACED_IGNORED = {"events", "events.events.k8s.io"}


class Refused(Exception):
    """A precondition failed before anything was written."""


CLUSTER_SCOPED = {"namespaces", "clusterroles.rbac.authorization.k8s.io", "clusterrolebindings.rbac.authorization.k8s.io"}

class Kube:
    def __init__(self, context: str | None):
        self.base = ["kubectl"] + (["--context", context] if context else [])

    def run(self, args, stdin: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
        r = subprocess.run(self.base + list(args), input=stdin, capture_output=True, text=True)
        if check and r.returncode:
            raise RuntimeError(f"kubectl {' '.join(args)}: {r.stderr.strip() or r.stdout.strip()}")
        return r

    def get(self, kind: str, name: str, namespace: str | None = None) -> dict | None:
        args = ["get", kind, name, "-o", "json"] + (["-n", namespace] if namespace else [])
        r = self.run(args, check=False)
        if r.returncode:
            if "NotFound" in r.stderr or "not found" in r.stderr:
                return None
            raise RuntimeError(f"kubectl {' '.join(args)}: {r.stderr.strip()}")
        return json.loads(r.stdout)

    def json(self, args) -> dict:
        return json.loads(self.run(list(args) + ["-o", "json"]).stdout)

    def can_i(self, verb: str, resource: str, namespace: str | None = None) -> bool:
        # No namespace means every namespace for a namespaced kind (the -A listings), not the current one.
        scope = ["-n", namespace] if namespace else ([] if resource.split("/")[0] in CLUSTER_SCOPED else ["--all-namespaces"])
        args = ["auth", "can-i", verb, resource] + scope
        return self.run(args, check=False).stdout.strip() == "yes"

    def proxy_get(self, namespace: str, service: str, path: str) -> str:
        return self.run(["get", "--raw", f"/api/v1/namespaces/{namespace}/services/{service}:http/proxy{path}"]).stdout


def owned(obj: dict, o: R.Options) -> bool:
    lab = obj.get("metadata", {}).get("labels") or {}
    return lab.get(R.MANAGED_BY) == R.TOOL and lab.get(R.INSTANCE_LABEL) == o.namespace and lab.get(R.OWNER_LABEL) == o.owner


def describe(obj: dict) -> str:
    m = obj["metadata"]
    return f"{obj['kind']} {m.get('namespace', '') + '/' if m.get('namespace') else ''}{m['name']}"


def foreign(kube: Kube, o: R.Options, objects: list) -> list:
    """Existing objects among `objects` that this owner does not own, as readable strings."""
    out = []
    for obj in objects:
        m = obj["metadata"]
        have = kube.get(obj["kind"], m["name"], m.get("namespace"))
        if have is not None and not owned(have, o):
            lab = have["metadata"].get("labels") or {}
            who = lab.get(R.OWNER_LABEL)
            out.append(f"{describe(obj)} exists " + (f"with owner {who!r}" if who else "and was not created by this tool"))
    return out


KINDS = {"Namespace": "namespaces", "ServiceAccount": "serviceaccounts", "ConfigMap": "configmaps",
         "Service": "services", "Deployment": "deployments.apps", "Secret": "secrets",
         "Role": "roles.rbac.authorization.k8s.io", "RoleBinding": "rolebindings.rbac.authorization.k8s.io",
         "ClusterRole": "clusterroles.rbac.authorization.k8s.io",
         "ClusterRoleBinding": "clusterrolebindings.rbac.authorization.k8s.io"}


def require(kube: Kube, needs: list) -> None:
    """needs: (verb, kind, namespace or None). Refuses, naming every missing permission, before any write."""
    missing = []
    for verb, kind, ns in sorted(set(needs), key=str):
        if not kube.can_i(verb, KINDS.get(kind, kind), ns):
            missing.append(f"{verb} {KINDS.get(kind, kind)}" + (f" in {ns}" if ns else ""))
    if missing:
        raise Refused("missing permissions: " + "; ".join(missing))


def apply_needs(o: R.Options, objects: list) -> list:
    needs = [("get", "namespaces", None), ("list", "rolebindings.rbac.authorization.k8s.io", None)]
    for obj in objects:
        ns = obj["metadata"].get("namespace")
        needs += [(v, obj["kind"], ns) for v in ("get", "create", "patch")]
    needs += [("create", "Secret", o.namespace), ("get", "Secret", o.namespace),
              ("get", "deployments.apps", o.namespace), ("watch", "deployments.apps", o.namespace), ("list", "pods", o.namespace)]
    # prune(), and removing grants of namespaces taken off the list, run after the writes
    needs += [(v, k, o.namespace) for k in ("deployments.apps", "services", "configmaps", "serviceaccounts", "secrets")
              for v in ("list", "delete")]
    needs += [(v, k, None) for k in ("clusterroles.rbac.authorization.k8s.io", "clusterrolebindings.rbac.authorization.k8s.io")
              for v in ("get", "delete")]
    needs += [(v, k, None) for k in ("roles.rbac.authorization.k8s.io", "rolebindings.rbac.authorization.k8s.io")
              for v in ("list", "delete")]
    return needs


def server_side_apply(kube: Kube, objects: list) -> None:
    # Every object here was verified as this owner's (or created just now by this call), so taking over
    # a field another manager touched (kubectl scale, kubectl edit) is intended: the render is the truth.
    if objects:
        kube.run(["apply", "--server-side", "--force-conflicts", f"--field-manager={R.FIELD_MANAGER}", "-f", "-"],
                 stdin=json.dumps({"apiVersion": "v1", "kind": "List", "items": objects}))


# What claim() creates for each kind: only fields that never change, so the create's field manager
# co-owns nothing that a later render may drop (server-side apply removes a field only when no other
# manager owns it). Kinds inside the monitoring namespace need no claim: the namespace claim decides.
STUB_FIELDS = {"Namespace": (), "Secret": ("type", "data"), "Role": (), "ClusterRole": (),
               "RoleBinding": ("roleRef", "subjects"), "ClusterRoleBinding": ("roleRef", "subjects")}


def claim(kube: Kube, o: R.Options, obj: dict) -> None:
    """Create obj if absent, atomically; then require that what exists is this owner's.

    Two installers racing for the same name both reach create; the loser gets AlreadyExists, re-reads,
    and is refused if the winner was another owner."""
    m = obj["metadata"]
    if kube.get(obj["kind"], m["name"], m.get("namespace")) is None:
        stub = {"apiVersion": obj["apiVersion"], "kind": obj["kind"], "metadata": m}
        stub.update({f: obj[f] for f in STUB_FIELDS[obj["kind"]] if f in obj})
        r = kube.run(["create", "-f", "-"], stdin=json.dumps(stub), check=False)
        if r.returncode and "AlreadyExists" not in r.stderr and "already exists" not in r.stderr:
            raise RuntimeError(f"kubectl create {describe(obj)}: {r.stderr.strip()}")
    have = kube.get(obj["kind"], m["name"], m.get("namespace"))
    if have is None or not owned(have, o):
        who = ((have or {}).get("metadata", {}).get("labels") or {}).get(R.OWNER_LABEL)
        raise Refused(f"{describe(obj)} exists " + (f"with owner {who!r}" if who else "and was not created by this tool"))


def ensure_secret(kube: Kube, o: R.Options) -> None:
    secret = {"apiVersion": "v1", "kind": "Secret", "type": "Opaque", "metadata": R.meta(o, "grafana-admin"),
              "data": {"admin-password": base64.b64encode(secrets.token_urlsafe(24).encode()).decode()}}
    # Created once and never rewritten. The value goes to kubectl on stdin, never to a log or stdout.
    claim(kube, o, secret)


def owned_grants(kube: Kube, o: R.Options) -> dict:
    """{namespace: [(kind, name)]} of this owner's grants anywhere. Fails closed: an unlistable state refuses."""
    sel = f"{R.MANAGED_BY}={R.TOOL},{R.INSTANCE_LABEL}={o.namespace},{R.OWNER_LABEL}={o.owner}"
    found = {}
    for kind in ("rolebindings", "roles"):
        r = kube.run(["get", f"{kind}.rbac.authorization.k8s.io", "-A", "-l", sel, "-o", "json"], check=False)
        if r.returncode:
            raise Refused(f"cannot list {kind} across namespaces ({r.stderr.strip()}); this owner's grants cannot be accounted for")
        for i in json.loads(r.stdout)["items"]:
            if i["metadata"]["name"] == f"{R.READ_ROLE}-{o.namespace}" and owned(i, o):
                found.setdefault(i["metadata"]["namespace"], []).append((i["kind"], i["metadata"]["name"]))
    return found


def remove_grant(kube: Kube, ns: str, items: list) -> None:
    for kind, name in sorted(items, reverse=True):  # RoleBinding before Role
        kube.run(["delete", kind.lower(), name, "-n", ns, "--ignore-not-found"])


def prune(kube: Kube, o: R.Options, doc: dict) -> None:
    """Remove this owner's objects that the current render no longer contains."""
    want = {(x["kind"], x["metadata"]["name"]) for x in doc["objects"]} | {("Secret", "grafana-admin")}
    sel = f"{R.MANAGED_BY}={R.TOOL},{R.INSTANCE_LABEL}={o.namespace},{R.OWNER_LABEL}={o.owner}"
    for kind in ("deployments.apps", "services", "configmaps", "serviceaccounts", "secrets"):
        items = kube.json(["get", kind, "-n", o.namespace, "-l", sel])["items"]
        for i in items:
            if (i["kind"], i["metadata"]["name"]) not in want and owned(i, o):
                kube.run(["delete", kind, i["metadata"]["name"], "-n", o.namespace, "--ignore-not-found"])
    if not doc["cluster"]:
        for kind in ("clusterrolebinding", "clusterrole"):
            have = kube.get(kind, f"{o.namespace}-prometheus")
            if have is not None and owned(have, o):
                kube.run(["delete", kind, f"{o.namespace}-prometheus"])


def apply(kube: Kube, o: R.Options, wait_s: int) -> None:
    doc = R.render(o)
    present = {ns for ns in doc["grants"] if kube.get("namespace", ns) is not None}
    absent = sorted(set(doc["grants"]) - present)
    grants = [g for ns in sorted(present) for g in doc["grants"][ns]]
    everything = [doc["namespace"]] + doc["objects"] + grants + doc["cluster"]
    require(kube, apply_needs(o, everything))
    clash = foreign(kube, o, everything)
    if clash:
        raise Refused("; ".join(clash))
    stale = owned_grants(kube, o)  # read before writing, so a listing failure refuses with nothing written
    claim(kube, o, doc["namespace"])
    ensure_secret(kube, o)
    for obj in doc["cluster"] + grants:
        claim(kube, o, obj)
    server_side_apply(kube, doc["objects"] + doc["cluster"] + grants)
    for ns in sorted(set(stale) - present):  # a namespace taken off --scrape-namespace loses this owner's grant there
        remove_grant(kube, ns, stale[ns])
    prune(kube, o, doc)
    deadline = time.monotonic() + wait_s  # one budget for all three, so --timeout bounds the whole wait
    for name in ("prometheus", "pushgateway", "grafana"):
        left = max(1, int(deadline - time.monotonic()))
        kube.run(["-n", o.namespace, "rollout", "status", f"deployment/{name}", f"--timeout={left}s"])
    for ns in absent:
        print(f"install: namespace {ns} does not exist yet; run apply again with the same --scrape-namespace after it is created")
    print(f"install: {o.namespace} ready (owner {o.owner}); open Grafana with: kubectl -n {o.namespace} port-forward service/grafana 3000:80")


# Aggregated APIs whose objects mirror others (PodMetrics mirror Pods) and carry no uid of their own.
SKIP_GROUPS = (".metrics.k8s.io", ".custom.metrics.k8s.io", ".external.metrics.k8s.io")


def inventory(kube: Kube, o: R.Options) -> list:
    """Objects in the monitoring namespace that delete would remove although this tool did not create them."""
    r = kube.run(["api-resources", "--namespaced", "--verbs=list", "-o", "name"], check=False)
    if r.returncode:
        # kubectl lists what it resolved and exits 1 when an aggregated API is unavailable. Only the
        # metrics APIs, which mirror other objects, are safe to go without.
        broken = set(re.findall(r"([a-z0-9.-]+\.[a-z]+)/v[0-9a-z]+", r.stderr))
        if not r.stdout.strip() or not broken or not all(("." + g).endswith(SKIP_GROUPS) for g in broken):
            raise Refused(f"cannot list the cluster's resource kinds ({r.stderr.strip()[:300]}); delete refuses what it cannot inspect")
    kinds = [k for k in r.stdout.split() if k not in NAMESPACED_IGNORED and not k.endswith(SKIP_GROUPS)]
    items = []
    for kind in kinds:
        r = kube.run(["get", kind, "-n", o.namespace, "-o", "json", "--ignore-not-found"], check=False)
        if r.returncode:
            raise Refused(f"cannot list {kind} in {o.namespace} ({r.stderr.strip()}); delete refuses what it cannot inspect")
        if r.stdout.strip():
            items += json.loads(r.stdout).get("items", [])
    ours = {i["metadata"].get("uid") for i in items if owned(i, o)} - {None}
    services = {i["metadata"]["name"] for i in items if i.get("kind") == "Service" and owned(i, o)}
    changed = True
    while changed:  # follow ownerReferences: Deployment -> ReplicaSet -> Pod, Service -> EndpointSlice
        changed = False
        for i in items:
            uid = i["metadata"].get("uid")
            if uid and uid not in ours and any(ref.get("uid") in ours for ref in i["metadata"].get("ownerReferences") or []):
                ours.add(uid)
                changed = True
    left = []
    for i in items:
        m = i["metadata"]
        if m.get("uid") in ours or (i.get("kind"), m.get("name")) in {("ConfigMap", "kube-root-ca.crt"), ("ServiceAccount", "default")}:
            continue
        if i.get("kind") == "Endpoints" and m.get("name") in services:
            continue
        left.append(describe(i))
    return left


def delete(kube: Kube, o: R.Options, wait_s: int) -> None:
    require(kube, [("get", "namespaces", None), ("delete", f"namespaces/{o.namespace}", None),
                   ("list", "rolebindings.rbac.authorization.k8s.io", None), ("list", "roles.rbac.authorization.k8s.io", None),
                   ("delete", "rolebindings.rbac.authorization.k8s.io", None), ("delete", "roles.rbac.authorization.k8s.io", None)]
            + [(v, k, None) for k in ("clusterroles.rbac.authorization.k8s.io", "clusterrolebindings.rbac.authorization.k8s.io")
               for v in ("get", "delete")])
    ns_obj = kube.get("namespace", o.namespace)
    if ns_obj is not None and not owned(ns_obj, o):
        raise Refused(f"namespace {o.namespace} is not an install of owner {o.owner!r}; nothing deleted")
    if ns_obj is not None:
        left = inventory(kube, o)
        if left:
            raise Refused(f"namespace {o.namespace} holds objects this tool did not create, so it is kept: " + "; ".join(left))
    grants = owned_grants(kube, o)  # refuses, with nothing deleted, if the grants cannot be listed
    for ns, items in sorted(grants.items()):
        remove_grant(kube, ns, items)
    for kind in ("clusterrolebinding", "clusterrole"):
        name = f"{o.namespace}-prometheus"
        have = kube.get(kind, name)
        if have is not None and owned(have, o):
            kube.run(["delete", kind, name])
    if ns_obj is not None:
        kube.run(["delete", "namespace", o.namespace, "--wait=true", f"--timeout={wait_s}s"])
        print(f"install: removed {o.namespace} (owner {o.owner})")
    else:
        print(f"install: namespace {o.namespace} does not exist; removed {sum(map(len, grants.values()))} grant object(s)")


# check's exit status when the install works and only an exporter named with --scrape-service has no target up.
EXPORTER_DOWN = 4


def check(kube: Kube, o: R.Options, show_password: bool) -> int:
    problems = []
    ns_obj = kube.get("namespace", o.namespace)
    if ns_obj is None:
        print(f"check: namespace {o.namespace} does not exist")
        return 1
    lab = ns_obj["metadata"].get("labels") or {}
    print(f"check: {o.namespace} owner {lab.get(R.OWNER_LABEL)!r}, managed by {lab.get(R.MANAGED_BY)!r}")
    for name, path in (("prometheus", "/-/ready"), ("pushgateway", "/-/ready"), ("grafana", "/api/health")):
        try:
            kube.proxy_get(o.namespace, name, path)
            print(f"check: {name} answers {path}")
        except RuntimeError as exc:
            problems.append(f"{name} does not answer {path}: {exc}")
    # An exporter named with --scrape-service is a target the install was asked for: one that has no up
    # target after a short wait (a misspelled Service or port, or an exporter that is down) is a problem.
    try:
        cfg = json.loads(kube.get("configmap", "prometheus-config", o.namespace)["data"]["prometheus.yml"])
        wanted = [j["job_name"] for j in cfg["scrape_configs"] if j["job_name"].startswith("service_")]
    except (RuntimeError, KeyError, TypeError, ValueError) as exc:
        wanted = []
        problems.append(f"cannot read the Prometheus configuration: {exc}")
    deadline = time.monotonic() + (60 if wanted else 0)
    missing = []
    while True:
        try:
            targets = json.loads(kube.proxy_get(o.namespace, "prometheus", "/api/v1/targets?state=active"))["data"]["activeTargets"]
        except (RuntimeError, KeyError, ValueError) as exc:
            problems.append(f"cannot read targets: {exc}")
            targets, wanted = [], []
            break
        up = {tgt["labels"].get("job") for tgt in targets if tgt.get("health") == "up"}
        missing = [j for j in wanted if j not in up]
        if not missing or time.monotonic() > deadline:
            break
        time.sleep(5)
    for tgt in targets:
        print(f"check: target {tgt['labels'].get('job')} {tgt['labels'].get('instance')} {tgt['health']}")
    # An exporter is someone else's workload and may not run yet (a GPU Operator installed after the
    # cluster), so it is reported with its own exit status rather than as a broken install.
    exporters = [f"no target of {j} (a --scrape-service exporter) is up" for j in missing if wanted]
    if show_password:
        s = kube.get("secret", "grafana-admin", o.namespace)
        print("check: Grafana admin password " + base64.b64decode(s["data"]["admin-password"]).decode())
    for p in problems + exporters:
        print(f"check: PROBLEM {p}")
    return 1 if problems else (EXPORTER_DOWN if exporters else 0)


def options(a: argparse.Namespace) -> R.Options:
    images = dict(R.IMAGES)
    for k in images:
        v = getattr(a, f"{k}_image")
        if v:
            images[k] = v
    sel = {}
    for kv in a.node_selector:
        if "=" not in kv:
            raise ValueError(f"--node-selector {kv!r} must be key=value")
        k, v = kv.split("=", 1)
        sel[k] = v
    # render needs no owner; a placeholder keeps the labels valid in its output.
    return R.Options(namespace=a.namespace, owner=a.owner or "render-only", scrape_namespaces=a.scrape_namespace,
                     scrape_services=a.scrape_service, node_selector=sel, tolerations=a.toleration, root_url=a.root_url,
                     extra_origins=a.extra_origin, anonymous_viewer=a.anonymous_viewer, dashboards=a.dashboard,
                     home=a.home, cluster_wide_discovery=a.cluster_wide_discovery, images=images,
                     retention_size=a.retention_size, retention_time=a.retention_time, scrape_interval=a.scrape_interval)



def duration(text: str) -> str:
    """A Prometheus duration such as 15s, 24h or 1h30m, checked before anything is written."""
    if not re.fullmatch(r"(\d+(ms|s|m|h|d|w|y))+", text):
        raise argparse.ArgumentTypeError(f"{text!r} is not a Prometheus duration such as 15s or 24h")
    return text


def size(text: str) -> str:
    if not re.fullmatch(r"\d+(B|KB|MB|GB|TB|PB|EB)", text):
        raise argparse.ArgumentTypeError(f"{text!r} is not a size such as 1GB")
    return text


def positive_int(text: str) -> int:
    if not re.fullmatch(r"[1-9]\d*", text):
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number of at least 1")
    return int(text)

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("command", choices=["apply", "check", "delete", "render"])
    p.add_argument("--context")
    p.add_argument("--namespace", default="in-cluster-monitoring")
    p.add_argument("--owner", help="who owns this install (label value); required for apply and delete")
    p.add_argument("--scrape-namespace", action="append", default=[])
    p.add_argument("--scrape-service", action="append", default=[], metavar="NS/SERVICE:PORT")
    p.add_argument("--node-selector", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--toleration", action="append", default=[], metavar="KEY[=VALUE]:EFFECT",
                   help="let the pods run on nodes with this taint; none by default")
    p.add_argument("--root-url", default="http://localhost:3000/")
    p.add_argument("--extra-origin", action="append", default=[])
    p.add_argument("--anonymous-viewer", action="store_true")
    p.add_argument("--dashboard", action="append", default=[], metavar="FILE")
    p.add_argument("--home", default=R.DEFAULT_HOME, metavar="FILE_NAME")
    p.add_argument("--cluster-wide-discovery", action="store_true")
    p.add_argument("--retention-time", default="24h", type=duration)
    p.add_argument("--retention-size", default="1GB", type=size)
    p.add_argument("--scrape-interval", default="15s", type=duration)
    for k in R.IMAGES:
        p.add_argument(f"--{k}-image")
    p.add_argument("--timeout", type=positive_int, default=300, help="seconds to wait for rollouts and namespace deletion")
    p.add_argument("--show-password", action="store_true")
    return p


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    if a.command in ("apply", "delete") and not a.owner:
        print(f"install: {a.command} needs --owner", file=sys.stderr)
        return 2
    try:
        o = options(a)
        if a.command == "render":
            print(json.dumps(R.render(o), indent=2))
            return 0
        kube = Kube(a.context)
        if a.command == "apply":
            apply(kube, o, a.timeout)
        elif a.command == "delete":
            delete(kube, o, a.timeout)
        else:
            return check(kube, o, a.show_password)
    except (Refused, ValueError) as exc:
        print(f"install: refused: {exc}", file=sys.stderr)
        return 3
    except (RuntimeError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"install: failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
