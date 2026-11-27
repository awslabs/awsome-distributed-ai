# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Kubernetes objects of the in-cluster monitoring stack, rendered from one Options value.

Pure functions: nothing here talks to a cluster, so every object the installer creates can be
unit-tested. install.py applies what render() returns.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

TOOL = "eks-in-cluster-monitoring"
DOMAIN = "awsome-distributed-ai"
INSTANCE_LABEL = f"{DOMAIN}/monitoring-instance"
OWNER_LABEL = f"{DOMAIN}/monitoring-owner"
MANAGED_BY = "app.kubernetes.io/managed-by"
SCRAPE_LABEL = f"{DOMAIN}/scrape"
TARGET_LABEL_PREFIX = f"{DOMAIN}/target-"
METRICS_PORT = "metrics"
FIELD_MANAGER = TOOL
SERVICE_ACCOUNT = "prometheus"
READ_ROLE = "monitoring-read"
# Labels the scrape config sets itself; a target-<name> Service label never overrides them.
RESERVED_TARGET_LABELS = ("namespace", "service", "pod", "job", "instance")

IMAGES = {
    "prometheus": "quay.io/prometheus/prometheus:v3.15.1@sha256:fa3302279710be2f21a8301ec17a8b4f2301fd829bd113839b3b6b02b6f191a2",
    "pushgateway": "quay.io/prometheus/pushgateway:v1.11.5@sha256:7332ee96119c0b927a3e64971189d884718939b6102c3e9e047bc11e83715d6f",
    "grafana": "docker.io/grafana/grafana:13.3.0@sha256:488984b716c0862639f91a698adbe2862677a5a07bad42da5f03f3030883b205",
}
DASHBOARDS = Path(__file__).resolve().parent / "dashboards"
DEFAULT_HOME = "monitoring-health.json"

DNS_LABEL = re.compile(r"^[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?$")
LABEL_VALUE = re.compile(r"^(([A-Za-z0-9][-A-Za-z0-9_.]{0,61})?[A-Za-z0-9])?$")
LABEL_KEY = re.compile(r"^[A-Za-z0-9]([-A-Za-z0-9_.]{0,61}[A-Za-z0-9])?$")
SCRAPE_SERVICE = re.compile(r"^([a-z0-9-]+)/([a-z0-9-]+):([A-Za-z0-9-]+)$")
TOLERATION = re.compile(r"^([A-Za-z0-9./-]+)(?:=([A-Za-z0-9._-]*))?:(NoSchedule|PreferNoSchedule|NoExecute)$")


@dataclass
class Options:
    namespace: str
    owner: str
    scrape_namespaces: list = field(default_factory=list)
    scrape_services: list = field(default_factory=list)  # "namespace/service:port"
    node_selector: dict = field(default_factory=dict)
    tolerations: list = field(default_factory=list)  # "KEY[=VALUE]:EFFECT"
    root_url: str = "http://localhost:3000/"
    extra_origins: list = field(default_factory=list)
    anonymous_viewer: bool = False
    dashboards: list = field(default_factory=list)  # extra dashboard files
    home: str = DEFAULT_HOME
    cluster_wide_discovery: bool = False
    scrape_interval: str = "15s"
    retention_time: str = "24h"
    retention_size: str = "1GB"
    images: dict = field(default_factory=lambda: dict(IMAGES))


def validate(o: Options) -> None:
    """Raise ValueError naming the first input that cannot produce a valid render."""
    if not DNS_LABEL.match(o.namespace):
        raise ValueError(f"--namespace {o.namespace!r} is not a DNS-1123 label")
    if not o.owner or not LABEL_VALUE.match(o.owner):
        raise ValueError(f"--owner {o.owner!r} is not a Kubernetes label value (63 characters of [A-Za-z0-9_.-])")
    for ns in o.scrape_namespaces:
        if not DNS_LABEL.match(ns):
            raise ValueError(f"--scrape-namespace {ns!r} is not a DNS-1123 label")
    for s in o.scrape_services:
        if not SCRAPE_SERVICE.match(s):
            raise ValueError(f"--scrape-service {s!r} is not NAMESPACE/SERVICE:PORT")
    if len(set(o.scrape_services)) != len(o.scrape_services):
        raise ValueError("a --scrape-service is given twice")
    for k, v in o.node_selector.items():
        if not LABEL_KEY.match(k.split("/")[-1]) or not LABEL_VALUE.match(v):
            raise ValueError(f"--node-selector {k}={v} is not a label selector")
    for tol in o.tolerations:
        if not TOLERATION.match(tol):
            raise ValueError(f"--toleration {tol!r} is not KEY[=VALUE]:EFFECT")
    u = urlsplit(o.root_url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ValueError(f"--root-url {o.root_url!r} needs http(s)://host[:port][/path/]")
    if not o.root_url.endswith("/"):
        raise ValueError(f"--root-url {o.root_url!r} must end with /")
    for origin in o.extra_origins:
        e = urlsplit(origin)
        if e.scheme not in ("http", "https") or not e.hostname or e.path not in ("", "/"):
            raise ValueError(f"--extra-origin {origin!r} must be scheme://host[:port]")
    names = [Path(p).name for p in dashboard_files(o)]
    if len(names) != len(set(names)):
        raise ValueError(f"two dashboards share a file name: {names}")
    if o.home not in names:
        raise ValueError(f"--home {o.home!r} is not one of the provisioned dashboards {names}")


def dashboard_files(o: Options) -> list:
    return sorted(str(p) for p in DASHBOARDS.glob("*.json")) + [str(Path(p)) for p in o.dashboards]


POD_SECURITY_LABEL = "pod-security.kubernetes.io/enforce"


def labels(o: Options, component: str = "") -> dict:
    out = {MANAGED_BY: TOOL, INSTANCE_LABEL: o.namespace, OWNER_LABEL: o.owner}
    if component:
        out["app.kubernetes.io/name"] = component
    return out


def meta(o: Options, name: str, component: str = "", namespace: str | None = None) -> dict:
    m = {"name": name, "labels": labels(o, component)}
    if namespace is not False:
        m["namespace"] = namespace or o.namespace
    return m


def scrape_namespaces(o: Options) -> list:
    return sorted(set(o.scrape_namespaces) | {s.split("/")[0] for s in o.scrape_services})


def prometheus_config(o: Options) -> str:
    jobs = [{"job_name": "prometheus", "static_configs": [{"targets": ["localhost:9090"]}]},
            {"job_name": "pushgateway", "honor_labels": True,
             "static_configs": [{"targets": ["pushgateway:9091"]}]}]
    if o.scrape_namespaces:
        relabel = [
            {"source_labels": ["__meta_kubernetes_service_label_" + sanitize(SCRAPE_LABEL)], "regex": re.escape(o.namespace), "action": "keep"},
            {"source_labels": ["__meta_kubernetes_endpointslice_port_name"], "regex": METRICS_PORT, "action": "keep"},
            {"source_labels": ["__meta_kubernetes_endpointslice_endpoint_conditions_ready"], "regex": "true", "action": "keep"},
            {"action": "labelmap", "regex": "__meta_kubernetes_service_label_" + sanitize(TARGET_LABEL_PREFIX) + "(.+)"},
        ] + identity_relabel("labelled-services")
        jobs.append({"job_name": "labelled-services", "sample_limit": 50000,
                     "kubernetes_sd_configs": [{"role": "endpointslice", "namespaces": {"names": sorted(o.scrape_namespaces)}}],
                     "relabel_configs": relabel})
    for s in o.scrape_services:
        ns, svc, port = SCRAPE_SERVICE.match(s).groups()
        port_key = "__meta_kubernetes_endpointslice_port" if port.isdigit() else "__meta_kubernetes_endpointslice_port_name"
        # "_" cannot occur in a namespace or Service name, so the job name is unique per entry.
        jobs.append({"job_name": f"service_{ns}_{svc}_{port}", "sample_limit": 50000,
                     "kubernetes_sd_configs": [{"role": "endpointslice", "namespaces": {"names": [ns]}}],
                     "relabel_configs": [
                         {"source_labels": ["__meta_kubernetes_service_name"], "regex": re.escape(svc), "action": "keep"},
                         {"source_labels": [port_key], "regex": re.escape(port), "action": "keep"},
                         {"source_labels": ["__meta_kubernetes_endpointslice_endpoint_conditions_ready"], "regex": "true", "action": "keep"},
                     ] + identity_relabel(f"service_{ns}_{svc}_{port}")})
    cfg = {"global": {"scrape_interval": o.scrape_interval, "evaluation_interval": o.scrape_interval}, "scrape_configs": jobs}
    return json.dumps(cfg, indent=2, sort_keys=True) + "\n"  # JSON is valid YAML


IDENTITY_SOURCES = {"namespace": "__meta_kubernetes_namespace", "service": "__meta_kubernetes_service_name",
                    "pod": "__meta_kubernetes_pod_name", "instance": "__address__"}


def identity_relabel(job: str) -> list:
    # Set after the labelmap, so a Service label target-job, target-namespace and so on cannot override them.
    rules = [{"target_label": "job", "replacement": job}]
    rules += [{"source_labels": [IDENTITY_SOURCES[name]], "target_label": name}
              for name in RESERVED_TARGET_LABELS if name != "job"]
    return rules


def sanitize(key: str) -> str:
    """How Prometheus turns a Kubernetes label key into a meta label suffix."""
    return re.sub(r"[^A-Za-z0-9_]", "_", key)


def security(uid: int) -> tuple:
    pod = {"runAsNonRoot": True, "runAsUser": uid, "runAsGroup": uid, "fsGroup": uid,
           "seccompProfile": {"type": "RuntimeDefault"}}
    container = {"allowPrivilegeEscalation": False, "readOnlyRootFilesystem": True,
                 "capabilities": {"drop": ["ALL"]}}
    return pod, container


def deployment(o: Options, name: str, uid: int, container: dict, volumes: list, annotations: dict | None = None) -> dict:
    pod_sec, c_sec = security(uid)
    container = dict(container, name=name, securityContext=c_sec)
    spec = {"serviceAccountName": SERVICE_ACCOUNT if name == "prometheus" else "default",
            "automountServiceAccountToken": name == "prometheus",
            "securityContext": pod_sec, "containers": [container], "volumes": volumes}
    if o.node_selector:
        spec["nodeSelector"] = dict(o.node_selector)
    if o.tolerations:
        spec["tolerations"] = [toleration(t) for t in o.tolerations]
    tmpl_meta = {"labels": labels(o, name)}
    if annotations:
        tmpl_meta["annotations"] = annotations
    return {"apiVersion": "apps/v1", "kind": "Deployment", "metadata": meta(o, name, name),
            "spec": {"replicas": 1, "strategy": {"type": "Recreate"},
                     "selector": {"matchLabels": {"app.kubernetes.io/name": name, INSTANCE_LABEL: o.namespace}},
                     "template": {"metadata": tmpl_meta, "spec": spec}}}


def toleration(text: str) -> dict:
    key, value, effect = TOLERATION.match(text).groups()
    if value is None:
        return {"key": key, "operator": "Exists", "effect": effect}
    return {"key": key, "operator": "Equal", "value": value, "effect": effect}


def service(o: Options, name: str, port: int, target: int) -> dict:
    return {"apiVersion": "v1", "kind": "Service", "metadata": meta(o, name, name),
            "spec": {"type": "ClusterIP", "selector": {"app.kubernetes.io/name": name, INSTANCE_LABEL: o.namespace},
                     "ports": [{"name": "http", "port": port, "targetPort": target}]}}


def resources(cpu: str, mem: str) -> dict:
    return {"requests": {"cpu": cpu, "memory": mem}, "limits": {"cpu": cpu, "memory": mem}}


def grafana_env(o: Options) -> list:
    u = urlsplit(o.root_url)
    origin = f"{u.scheme}://{u.netloc}".lower()
    env = [{"name": "GF_SECURITY_ADMIN_USER", "value": "admin"},
           {"name": "GF_SECURITY_ADMIN_PASSWORD", "valueFrom": {"secretKeyRef": {"name": "grafana-admin", "key": "admin-password"}}},
           {"name": "GF_SERVER_ROOT_URL", "value": o.root_url},
           {"name": "GF_SERVER_SERVE_FROM_SUB_PATH", "value": "true" if u.path not in ("", "/") else "false"},
           {"name": "GF_SECURITY_CSRF_TRUSTED_ORIGINS", "value": u.netloc.lower()},
           {"name": "GF_LIVE_ALLOWED_ORIGINS", "value": ",".join([origin] + [x.rstrip("/").lower() for x in o.extra_origins])},
           {"name": "GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH", "value": f"/etc/grafana/dashboards/{o.home}"},
           {"name": "GF_AUTH_ANONYMOUS_ENABLED", "value": "true" if o.anonymous_viewer else "false"},
           {"name": "GF_AUTH_ANONYMOUS_ORG_ROLE", "value": "Viewer"},
           {"name": "GF_USERS_ALLOW_SIGN_UP", "value": "false"},
           {"name": "GF_ANALYTICS_REPORTING_ENABLED", "value": "false"},
           {"name": "GF_ANALYTICS_CHECK_FOR_UPDATES", "value": "false"},
           {"name": "GF_ANALYTICS_CHECK_FOR_PLUGIN_UPDATES", "value": "false"},
           {"name": "GF_NEWS_NEWS_FEED_ENABLED", "value": "false"},
           {"name": "GF_PLUGINS_PREINSTALL_DISABLED", "value": "true"},
           {"name": "GF_PATHS_DATA", "value": "/var/lib/grafana"},
           {"name": "GOMEMLIMIT", "value": "819MiB"}]
    return env


def read_role(o: Options, namespace: str) -> list:
    """Role and RoleBinding in a scraped namespace: what Prometheus may read there, and nothing else."""
    rules = [{"apiGroups": ["discovery.k8s.io"], "resources": ["endpointslices"], "verbs": ["get", "list", "watch"]},
             {"apiGroups": [""], "resources": ["services", "pods"], "verbs": ["get", "list", "watch"]}]
    name = f"{READ_ROLE}-{o.namespace}"
    return [{"apiVersion": "rbac.authorization.k8s.io/v1", "kind": "Role", "metadata": meta(o, name, namespace=namespace), "rules": rules},
            {"apiVersion": "rbac.authorization.k8s.io/v1", "kind": "RoleBinding", "metadata": meta(o, name, namespace=namespace),
             "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "Role", "name": name},
             "subjects": [{"kind": "ServiceAccount", "name": SERVICE_ACCOUNT, "namespace": o.namespace}]}]


def cluster_read(o: Options) -> list:
    name = f"{o.namespace}-prometheus"
    rules = [{"apiGroups": ["discovery.k8s.io"], "resources": ["endpointslices"], "verbs": ["get", "list", "watch"]},
             {"apiGroups": [""], "resources": ["services", "pods"], "verbs": ["get", "list", "watch"]}]
    return [{"apiVersion": "rbac.authorization.k8s.io/v1", "kind": "ClusterRole", "metadata": meta(o, name, namespace=False), "rules": rules},
            {"apiVersion": "rbac.authorization.k8s.io/v1", "kind": "ClusterRoleBinding", "metadata": meta(o, name, namespace=False),
             "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "ClusterRole", "name": name},
             "subjects": [{"kind": "ServiceAccount", "name": SERVICE_ACCOUNT, "namespace": o.namespace}]}]


def render(o: Options) -> dict:
    """{"namespace": obj, "objects": [...in the monitoring namespace...], "grants": {ns: [...]}, "cluster": [...]}.

    The admin password Secret is not rendered: install.py creates it once with a random value.
    """
    validate(o)
    prom_cfg = prometheus_config(o)
    dash = {Path(p).name: Path(p).read_text() for p in dashboard_files(o)}
    provisioning = {
        "datasource.yaml": json.dumps({"apiVersion": 1, "datasources": [{
            "name": "Prometheus", "uid": "prometheus", "type": "prometheus", "access": "proxy",
            "url": "http://prometheus:9090", "isDefault": True, "editable": False,
            "jsonData": {"timeInterval": o.scrape_interval}}]}, indent=2) + "\n",
        "dashboards.yaml": json.dumps({"apiVersion": 1, "providers": [{
            "name": TOOL, "type": "file", "disableDeletion": True, "allowUiUpdates": False,
            "options": {"path": "/etc/grafana/dashboards"}}]}, indent=2) + "\n"}
    ns = {"apiVersion": "v1", "kind": "Namespace", "metadata": meta(o, o.namespace, namespace=False)}
    # Every pod here meets the restricted Pod Security level, so the namespace enforces it.
    ns["metadata"]["labels"][POD_SECURITY_LABEL] = "restricted"
    objs = [
        {"apiVersion": "v1", "kind": "ServiceAccount", "metadata": meta(o, SERVICE_ACCOUNT)},
        {"apiVersion": "v1", "kind": "ConfigMap", "metadata": meta(o, "prometheus-config"), "data": {"prometheus.yml": prom_cfg}},
        {"apiVersion": "v1", "kind": "ConfigMap", "metadata": meta(o, "grafana-provisioning"), "data": provisioning},
        {"apiVersion": "v1", "kind": "ConfigMap", "metadata": meta(o, "grafana-dashboards"), "data": dash},
        deployment(o, "prometheus", 65534, {
            "image": o.images["prometheus"],
            "args": ["--config.file=/etc/prometheus/prometheus.yml", "--storage.tsdb.path=/prometheus",
                     f"--storage.tsdb.retention.time={o.retention_time}", f"--storage.tsdb.retention.size={o.retention_size}",
                     ],
            "ports": [{"containerPort": 9090}], "resources": resources("500m", "1Gi"),
            "readinessProbe": {"httpGet": {"path": "/-/ready", "port": 9090}, "periodSeconds": 5},
            "volumeMounts": [{"name": "config", "mountPath": "/etc/prometheus"}, {"name": "data", "mountPath": "/prometheus"}]},
            [{"name": "config", "configMap": {"name": "prometheus-config"}}, {"name": "data", "emptyDir": {"sizeLimit": "2Gi"}}],
            {f"{DOMAIN}/config-sha256": sha256(prom_cfg)}),
        service(o, "prometheus", 9090, 9090),
        deployment(o, "pushgateway", 65534, {
            "image": o.images["pushgateway"],
            "args": ["--persistence.file=/data/metrics", "--persistence.interval=30s"],
            "ports": [{"containerPort": 9091}], "resources": resources("50m", "64Mi"),
            "readinessProbe": {"httpGet": {"path": "/-/ready", "port": 9091}, "periodSeconds": 5},
            "volumeMounts": [{"name": "data", "mountPath": "/data"}]},
            [{"name": "data", "emptyDir": {"sizeLimit": "64Mi"}}]),
        service(o, "pushgateway", 9091, 9091),
        deployment(o, "grafana", 472, {
            "image": o.images["grafana"], "env": grafana_env(o),
            # GOMEMLIMIT at 80 % of the memory limit makes the Go runtime collect harder before the limit.
            "ports": [{"containerPort": 3000}], "resources": resources("500m", "1Gi"),
            "readinessProbe": {"httpGet": {"path": "/api/health", "port": 3000}, "periodSeconds": 5},
            "volumeMounts": [{"name": "provisioning", "mountPath": "/etc/grafana/provisioning/datasources/monitoring.yaml", "subPath": "datasource.yaml"},
                             {"name": "provisioning", "mountPath": "/etc/grafana/provisioning/dashboards/monitoring.yaml", "subPath": "dashboards.yaml"},
                             {"name": "dashboards", "mountPath": "/etc/grafana/dashboards"},
                             {"name": "data", "mountPath": "/var/lib/grafana"}, {"name": "tmp", "mountPath": "/tmp"}]},
            [{"name": "provisioning", "configMap": {"name": "grafana-provisioning"}},
             {"name": "dashboards", "configMap": {"name": "grafana-dashboards"}},
             {"name": "data", "emptyDir": {"sizeLimit": "256Mi"}}, {"name": "tmp", "emptyDir": {"sizeLimit": "64Mi"}}],
            {f"{DOMAIN}/config-sha256": sha256(json.dumps([provisioning, dash, grafana_env(o)], sort_keys=True))}),
        service(o, "grafana", 80, 3000),
    ]
    grants = {} if o.cluster_wide_discovery else {n: read_role(o, n) for n in scrape_namespaces(o)}
    return {"namespace": ns, "objects": objs, "grants": grants,
            "cluster": cluster_read(o) if o.cluster_wide_discovery else []}


def sha256(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()
