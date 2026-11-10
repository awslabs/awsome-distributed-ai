# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Unit tests; no cluster. python3 -m pytest observability/prometheus-grafana/eks-in-cluster"""
import base64
import http.server
import io
import json
import subprocess
import sys
import threading
import unittest
from unittest import mock
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import install  # noqa: E402
import monitoring_render as R  # noqa: E402
import push  # noqa: E402


def opts(**kw):
    return R.Options(**dict({"namespace": "mon", "owner": "alice"}, **kw))


def all_objects(doc):
    return [doc["namespace"]] + doc["objects"] + [g for v in doc["grants"].values() for g in v] + doc["cluster"]


class RenderTests(unittest.TestCase):
    def test_every_object_carries_owner_and_instance(self):
        doc = R.render(opts(scrape_namespaces=["lab"], cluster_wide_discovery=False))
        for obj in all_objects(doc):
            lab = obj["metadata"]["labels"]
            self.assertEqual((lab[R.MANAGED_BY], lab[R.INSTANCE_LABEL], lab[R.OWNER_LABEL]), (R.TOOL, "mon", "alice"), obj["metadata"]["name"])

    def test_default_creates_nothing_cluster_scoped_but_the_namespace(self):
        doc = R.render(opts(scrape_namespaces=["lab"]))
        self.assertEqual(doc["cluster"], [])
        self.assertTrue(all(o["metadata"].get("namespace") == "mon" for o in doc["objects"]))
        self.assertEqual(sorted(doc["grants"]), ["lab"])
        self.assertEqual([g["kind"] for g in doc["grants"]["lab"]], ["Role", "RoleBinding"])
        self.assertTrue(all(g["metadata"]["namespace"] == "lab" for g in doc["grants"]["lab"]))

    def test_cluster_wide_discovery_is_opt_in(self):
        doc = R.render(opts(scrape_namespaces=["lab"], cluster_wide_discovery=True))
        self.assertEqual([o["kind"] for o in doc["cluster"]], ["ClusterRole", "ClusterRoleBinding"])
        self.assertEqual(doc["grants"], {})
        self.assertNotIn("namespace", doc["cluster"][0]["metadata"])

    def test_no_daemonset_hostport_hostnetwork_crd_or_toleration(self):
        text = json.dumps(R.render(opts(scrape_namespaces=["lab"])))
        for word in ("DaemonSet", "hostPort", "hostNetwork", "CustomResourceDefinition", "tolerations", "prometheus.io/scrape"):
            self.assertNotIn(word, text)

    def test_tolerations_are_opt_in(self):
        doc = R.render(opts(tolerations=["workload=bench:NoSchedule", "nvidia.com/gpu:NoSchedule"]))
        spec = [o for o in doc["objects"] if o["kind"] == "Deployment"][0]["spec"]["template"]["spec"]
        self.assertEqual(spec["tolerations"], [
            {"key": "workload", "operator": "Equal", "value": "bench", "effect": "NoSchedule"},
            {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}])
        with self.assertRaises(ValueError):
            R.validate(opts(tolerations=["nvidia.com/gpu"]))

    def test_the_namespace_enforces_the_restricted_level(self):
        self.assertEqual(R.render(opts())["namespace"]["metadata"]["labels"][R.POD_SECURITY_LABEL], "restricted")

    def test_pods_are_restricted(self):
        for d in [o for o in R.render(opts())["objects"] if o["kind"] == "Deployment"]:
            spec = d["spec"]["template"]["spec"]
            self.assertTrue(spec["securityContext"]["runAsNonRoot"])
            self.assertEqual(spec["securityContext"]["seccompProfile"], {"type": "RuntimeDefault"})
            c = spec["containers"][0]
            self.assertEqual(c["securityContext"]["capabilities"], {"drop": ["ALL"]})
            self.assertFalse(c["securityContext"]["allowPrivilegeEscalation"])
            self.assertTrue(c["securityContext"]["readOnlyRootFilesystem"])
            self.assertEqual(c["resources"]["requests"], c["resources"]["limits"])
            self.assertIn("@sha256:", c["image"])

    def test_scrape_selection_and_label_order(self):
        cfg = json.loads(R.prometheus_config(opts(scrape_namespaces=["lab"])))
        job = next(j for j in cfg["scrape_configs"] if j["job_name"] == "labelled-services")
        self.assertEqual(job["kubernetes_sd_configs"][0]["namespaces"]["names"], ["lab"])
        rl = job["relabel_configs"]
        self.assertEqual(rl[0]["source_labels"], ["__meta_kubernetes_service_label_awsome_distributed_ai_scrape"])
        self.assertEqual(rl[0]["regex"], "mon")
        self.assertEqual(rl[1]["regex"], "metrics")
        lm = [i for i, r in enumerate(rl) if r.get("action") == "labelmap"][0]
        later = [r["target_label"] for r in rl[lm + 1:]]
        for name in ("namespace", "service", "pod", "instance"):
            self.assertIn(name, later)

    def test_target_labels_cannot_override_identity(self):
        cfg = json.loads(R.prometheus_config(opts(scrape_namespaces=["lab"])))
        rl = next(j for j in cfg["scrape_configs"] if j["job_name"] == "labelled-services")["relabel_configs"]
        lm = [i for i, r in enumerate(rl) if r.get("action") == "labelmap"][0]
        after = {r.get("target_label"): r for r in rl[lm + 1:]}
        for name in R.RESERVED_TARGET_LABELS:
            self.assertIn(name, after, name)
        self.assertEqual(after["job"]["replacement"], "labelled-services")

    def test_scrape_service_jobs_are_unique(self):
        cfg = json.loads(R.prometheus_config(opts(scrape_services=["a-b/c:1", "a/b-c:1"])))
        names = [j["job_name"] for j in cfg["scrape_configs"]]
        self.assertEqual(len(names), len(set(names)))
        with self.assertRaises(ValueError):
            R.validate(opts(scrape_services=["a/b:1", "a/b:1"]))

    def test_no_scrape_namespace_means_no_service_job(self):
        cfg = json.loads(R.prometheus_config(opts()))
        self.assertEqual([j["job_name"] for j in cfg["scrape_configs"]], ["prometheus", "pushgateway"])
        self.assertTrue(cfg["scrape_configs"][1]["honor_labels"])

    def test_scrape_service_job(self):
        cfg = json.loads(R.prometheus_config(opts(scrape_services=["gpu-operator/nvidia-dcgm-exporter:9400"])))
        job = cfg["scrape_configs"][-1]
        self.assertEqual(job["job_name"], "service_gpu-operator_nvidia-dcgm-exporter_9400")
        self.assertEqual(job["relabel_configs"][1]["source_labels"], ["__meta_kubernetes_endpointslice_port"])

    def test_grafana_root_url(self):
        env = {e["name"]: e.get("value") for e in R.grafana_env(opts())}
        self.assertEqual(env["GF_SERVER_SERVE_FROM_SUB_PATH"], "false")
        self.assertEqual(env["GF_AUTH_ANONYMOUS_ENABLED"], "false")
        env = {e["name"]: e.get("value") for e in R.grafana_env(opts(root_url="https://Nb.Example.com/proxy/absolute/3000/",
                                                                   extra_origins=["http://localhost:3000"], anonymous_viewer=True))}
        self.assertEqual(env["GF_SERVER_SERVE_FROM_SUB_PATH"], "true")
        self.assertEqual(env["GF_SECURITY_CSRF_TRUSTED_ORIGINS"], "nb.example.com")
        self.assertEqual(env["GF_LIVE_ALLOWED_ORIGINS"], "https://nb.example.com,http://localhost:3000")
        self.assertEqual(env["GF_AUTH_ANONYMOUS_ENABLED"], "true")

    def test_config_change_rolls_the_pod(self):
        a = [o for o in R.render(opts())["objects"] if o["metadata"]["name"] == "prometheus" and o["kind"] == "Deployment"][0]
        b = [o for o in R.render(opts(scrape_namespaces=["x"]))["objects"] if o["metadata"]["name"] == "prometheus" and o["kind"] == "Deployment"][0]
        self.assertNotEqual(a["spec"]["template"]["metadata"]["annotations"], b["spec"]["template"]["metadata"]["annotations"])

    def test_invalid_inputs(self):
        for kw in ({"namespace": "Bad_NS"}, {"owner": ""}, {"owner": "a b"}, {"scrape_namespaces": ["UP"]},
                   {"scrape_services": ["ns/svc"]}, {"root_url": "localhost:3000"}, {"root_url": "http://x/path"},
                   {"extra_origins": ["http://x/path"]}, {"home": "missing.json"}):
            with self.assertRaises(ValueError, msg=kw):
                R.validate(opts(**kw))

    def test_bundled_dashboards_are_generated(self):
        sys.path.insert(0, str(R.DASHBOARDS))
        import gen_dashboards
        for name, build in gen_dashboards.FILES.items():
            self.assertEqual((R.DASHBOARDS / name).read_text(), json.dumps(build(), indent=2, sort_keys=True) + "\n", name)


class FakeKube(install.Kube):
    """A small stateful API server: get, create (AlreadyExists), list by label, delete; records every write."""
    def __init__(self, objects=(), can=True, kinds=("configmaps", "pods", "services"), fail_list=False, race_owner=None,
                 fail_kind=None, targets=None):
        self.base = ["kubectl"]
        self.objects = {}
        for o in objects:
            self.put(o)
        self.writes, self.can, self.kinds, self.fail_list = [], can, kinds, fail_list
        self.race_owner = race_owner  # another owner creates the namespace between our check and our create
        self.fail_kind = fail_kind  # a namespaced kind whose listing is refused
        self.targets = targets or []  # Prometheus's active targets, for check()

    def proxy_get(self, namespace, service, path):
        if "targets" in path:
            return json.dumps({"data": {"activeTargets": self.targets}})
        return "ok"

    def put(self, o):
        m = o["metadata"]
        m.setdefault("uid", f"{o['kind']}-{m['name']}")
        self.objects[(o["kind"].lower(), m.get("namespace"), m["name"])] = o

    def get(self, kind, name, namespace=None):
        return self.objects.get((kind.lower(), namespace, name))

    def can_i(self, verb, resource, namespace=None):
        return self.can if not callable(self.can) else self.can(verb, resource, namespace)

    def run(self, args, stdin=None, check=True):
        args = list(args)
        if args[:2] == ["api-resources", "--namespaced"]:
            return _cp("\n".join(self.kinds))
        if args[0] == "create":
            o = json.loads(stdin)
            if self.race_owner and o["kind"] == "Namespace":
                o = json.loads(stdin)
                o["metadata"]["labels"][R.OWNER_LABEL] = self.race_owner
                self.put(o)
                self.race_owner = None
                return _cp("", 1, "Error from server (AlreadyExists): namespaces exists")
            if self.get(o["kind"], o["metadata"]["name"], o["metadata"].get("namespace")) is not None:
                return _cp("", 1, "Error from server (AlreadyExists)")
            self.put(o)
            self.writes.append((args, stdin))
            return _cp("")
        if args[0] == "get" and "-A" in args:
            if self.fail_list:
                return _cp("", 1, "Forbidden")
            kind = args[1].split(".")[0][:-1]
            return _cp(json.dumps({"items": [o for (k, _, _), o in self.objects.items() if k == kind]}))
        if args[0] == "get" and "-l" in args:
            ns = args[args.index("-n") + 1]
            kind = args[1].split(".")[0][:-1]
            return _cp(json.dumps({"items": [o for (k, n, _), o in self.objects.items() if k == kind and n == ns]}))
        if args[0] == "get" and args[1] == self.fail_kind:
            return _cp("", 1, "Error from server (Forbidden)")
        if args[0] == "get":
            kind, ns = args[1], args[args.index("-n") + 1]
            items = [o for (k, n, _), o in self.objects.items() if n == ns and k + "s" == kind]
            return _cp(json.dumps({"items": items}))
        self.writes.append((args, stdin))
        return _cp("")


def _cp(out, rc=0, err=""):
    import subprocess
    return subprocess.CompletedProcess([], rc, out, err)


def obj(kind, name, namespace=None, owner="alice", instance="mon", managed=True, uid=None, refs=()):
    labels = {}
    if managed:
        labels = {R.MANAGED_BY: R.TOOL, R.INSTANCE_LABEL: instance, R.OWNER_LABEL: owner}
    m = {"name": name, "labels": labels, "uid": uid or f"{kind}-{name}", "ownerReferences": [{"uid": r} for r in refs]}
    if namespace:
        m["namespace"] = namespace
    return {"kind": kind, "metadata": m}


def quiet(fn, *a):
    with redirect_stdout(io.StringIO()):
        return fn(*a)


class PermissionScopeTests(unittest.TestCase):
    def test_a_check_without_a_namespace_covers_every_namespace_for_a_namespaced_kind(self):
        seen = []

        class Recording(install.Kube):
            def run(self, args, stdin=None, check=True):
                seen.append(list(args))
                return type("R", (), {"stdout": "yes", "returncode": 0})()

        kube = Recording(None)
        kube.can_i("list", "rolebindings.rbac.authorization.k8s.io")
        kube.can_i("get", "namespaces")
        kube.can_i("get", "secrets", "mon")
        self.assertEqual(seen, [["auth", "can-i", "list", "rolebindings.rbac.authorization.k8s.io", "--all-namespaces"],
                                ["auth", "can-i", "get", "namespaces"],
                                ["auth", "can-i", "get", "secrets", "-n", "mon"]])


class ArgumentTests(unittest.TestCase):
    def test_durations_sizes_and_timeouts_are_checked_before_anything_runs(self):
        for argv in (["render", "--scrape-interval", "5x"], ["render", "--retention-size", "1G"], ["render", "--timeout", "0"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                install.parser().parse_args(argv)
        a = install.parser().parse_args(["render", "--scrape-interval", "1m30s", "--retention-size", "2GB", "--timeout", "60"])
        self.assertEqual((a.scrape_interval, a.retention_size, a.timeout), ("1m30s", "2GB", 60))


class GrantTests(unittest.TestCase):
    READ = [{"apiGroups": ["discovery.k8s.io"], "resources": ["endpointslices"], "verbs": ["get", "list", "watch"]},
            {"apiGroups": [""], "resources": ["services", "pods"], "verbs": ["get", "list", "watch"]}]

    def test_the_grants_read_endpoints_services_and_pods_and_nothing_else(self):
        role, binding = R.read_role(opts(), "lab")
        self.assertEqual(role["rules"], self.READ)
        self.assertEqual(binding["subjects"], [{"kind": "ServiceAccount", "name": R.SERVICE_ACCOUNT, "namespace": "mon"}])
        cluster_role, cluster_binding = R.cluster_read(opts())
        self.assertEqual(cluster_role["rules"], self.READ)
        self.assertEqual(cluster_binding["roleRef"], {"apiGroup": "rbac.authorization.k8s.io", "kind": "ClusterRole", "name": "mon-prometheus"})


class ScrapeConfigTests(unittest.TestCase):
    def jobs(self, **kw):
        return json.loads(R.prometheus_config(opts(**kw)))["scrape_configs"]

    def test_only_ready_endpoints_are_kept(self):
        for job in self.jobs(scrape_namespaces=["lab"], scrape_services=["gpu/dcgm:9400"])[2:]:
            ready = [r for r in job["relabel_configs"]
                     if r.get("source_labels") == ["__meta_kubernetes_endpointslice_endpoint_conditions_ready"]]
            self.assertEqual(ready, [{"source_labels": ["__meta_kubernetes_endpointslice_endpoint_conditions_ready"],
                                      "regex": "true", "action": "keep"}], job["job_name"])

    def test_identity_labels_come_from_the_discovered_object_after_the_labelmap(self):
        for job in self.jobs(scrape_namespaces=["lab"], scrape_services=["gpu/dcgm:9400"])[2:]:
            rules = job["relabel_configs"]
            last = {r["target_label"]: r for r in rules if "target_label" in r}
            self.assertEqual(last["job"], {"target_label": "job", "replacement": job["job_name"]})
            for name, source in R.IDENTITY_SOURCES.items():
                self.assertEqual(last[name]["source_labels"], [source], (job["job_name"], name))
            labelmap = [i for i, r in enumerate(rules) if r.get("action") == "labelmap"]
            first_identity = min(i for i, r in enumerate(rules) if r.get("target_label") in R.RESERVED_TARGET_LABELS)
            self.assertTrue(all(i < first_identity for i in labelmap), job["job_name"])

    def test_no_job_sends_credentials_to_its_targets(self):
        for job in self.jobs(scrape_namespaces=["lab"], scrape_services=["gpu/dcgm:9400"], cluster_wide_discovery=True):
            for key in ("bearer_token", "bearer_token_file", "authorization", "basic_auth", "oauth2", "tls_config"):
                self.assertNotIn(key, job, (job["job_name"], key))


class OwnershipTests(unittest.TestCase):
    def test_apply_and_delete_refuse_the_same_owner_of_another_install(self):
        k = FakeKube([obj("Namespace", "mon", instance="other")])
        with self.assertRaises(install.Refused):
            install.apply(k, opts(), 1)
        with self.assertRaises(install.Refused):
            install.delete(k, opts(), 1)
        self.assertEqual(k.writes, [])

    def test_delete_refuses_when_a_namespaced_kind_cannot_be_listed(self):
        k = FakeKube([obj("Namespace", "mon"), obj("RoleBinding", "monitoring-read-mon", "lab"), obj("Role", "monitoring-read-mon", "lab")],
                     kinds=("configmaps", "pods", "services", "leases.coordination.k8s.io"), fail_kind="leases.coordination.k8s.io")
        with self.assertRaisesRegex(install.Refused, "cannot list leases"):
            install.delete(k, opts(), 1)
        self.assertFalse(any(w[0][0] == "delete" for w in k.writes))

    def test_delete_checks_every_permission_it_uses_before_removing_anything(self):
        for missing in (("delete", "clusterroles.rbac.authorization.k8s.io"), ("get", "clusterrolebindings.rbac.authorization.k8s.io"),
                        ("delete", "rolebindings.rbac.authorization.k8s.io"), ("delete", "namespaces/mon")):
            k = FakeKube([obj("Namespace", "mon"), obj("ClusterRoleBinding", "mon-prometheus"), obj("ClusterRole", "mon-prometheus")],
                         can=lambda v, r, n, m=missing: (v, r) != m)
            with self.assertRaisesRegex(install.Refused, "missing permissions: " + " ".join(missing)):
                install.delete(k, opts(), 1)
            self.assertEqual(k.writes, [], missing)

    def test_delete_asks_for_the_monitoring_namespace_by_name(self):
        asked = []
        class Recording(install.Kube):
            def run(self, args, stdin=None, check=True):
                asked.append(list(args))
                return _cp("yes")
        Recording(None).can_i("delete", "namespaces/mon")
        self.assertEqual(asked, [["auth", "can-i", "delete", "namespaces/mon"]])
        k = FakeKube([obj("Namespace", "mon")], can=lambda v, r, n: not (v == "delete" and r == "namespaces"))
        quiet(install.delete, k, opts(), 1)
        self.assertIn((["delete", "namespace", "mon", "--wait=true", "--timeout=1s"], None), k.writes)

    def test_delete_goes_past_metrics_apis_that_are_unavailable_and_refuses_any_other(self):
        listed = "configmaps\npods\nservices\n"
        cases = {"error: unable to retrieve the complete list of server APIs: metrics.k8s.io/v1beta1: the server is currently unable to handle the request": True,
                 "error: unable to retrieve the complete list of server APIs: example.com/v1: the server could not find the requested resource": False,
                 "error: something kubectl has not said before": False}
        for stderr, passes in cases.items():
            k = FakeKube([obj("Namespace", "mon")])
            real = k.run
            k.run = lambda args, stdin=None, check=True, real=real, stderr=stderr: (
                _cp(listed, 1, stderr) if list(args)[:2] == ["api-resources", "--namespaced"] else real(args, stdin, check))
            if passes:
                quiet(install.delete, k, opts(), 1)
                self.assertTrue(any(w[0][:3] == ["delete", "namespace", "mon"] for w in k.writes), stderr)
            else:
                with self.assertRaisesRegex(install.Refused, "cannot list the cluster's resource kinds"):
                    install.delete(k, opts(), 1)
                self.assertEqual(k.writes, [], stderr)

    def test_apply_checks_the_watch_its_rollout_wait_needs(self):
        k = FakeKube(can=lambda v, r, n: (v, r) != ("watch", "deployments.apps"))
        with self.assertRaisesRegex(install.Refused, "missing permissions: watch deployments.apps"):
            install.apply(k, opts(), 1)
        self.assertEqual(k.writes, [])

    def test_apply_refuses_unowned_namespace_before_writing(self):
        k = FakeKube([obj("Namespace", "mon", managed=False)])
        with self.assertRaises(install.Refused):
            install.apply(k, opts(), 1)
        self.assertEqual(k.writes, [])

    def test_apply_refuses_other_owner(self):
        k = FakeKube([obj("Namespace", "mon", owner="bob")])
        with self.assertRaisesRegex(install.Refused, "owner 'bob'"):
            install.apply(k, opts(), 1)
        self.assertEqual(k.writes, [])

    def test_apply_refuses_without_each_needed_permission(self):
        for missing in (("patch", "configmaps"), ("create", "secrets"), ("list", "rolebindings.rbac.authorization.k8s.io")):
            k = FakeKube(can=lambda v, r, n, m=missing: (v, r) != m)
            with self.assertRaisesRegex(install.Refused, "missing permissions: " + " ".join(missing)):
                install.apply(k, opts(), 1)
            self.assertEqual(k.writes, [], missing)

    def test_race_with_another_owner_is_refused_without_writing_inside(self):
        k = FakeKube(race_owner="bob")
        with self.assertRaisesRegex(install.Refused, "owner 'bob'"):
            install.apply(k, opts(), 1)
        self.assertEqual(k.writes, [])

    def test_apply_creates_then_applies_with_force_conflicts(self):
        k = FakeKube([obj("Namespace", "lab", managed=False)])
        quiet(install.apply, k, opts(scrape_namespaces=["lab"]), 1)
        creates = [json.loads(w[1])["kind"] for w in k.writes if w[0][0] == "create"]
        self.assertEqual(creates[:2], ["Namespace", "Secret"])
        self.assertIn("RoleBinding", creates)
        ssa = [w for w in k.writes if w[0][:2] == ["apply", "--server-side"]][0]
        self.assertIn("--force-conflicts", ssa[0])
        stubs = {json.loads(w[1])["kind"]: json.loads(w[1]) for w in k.writes if w[0][0] == "create"}
        self.assertEqual(set(stubs["Namespace"]) - {"apiVersion", "kind"}, {"metadata"})
        self.assertIn("roleRef", stubs["RoleBinding"])
        self.assertNotIn("Deployment", stubs)  # created by the server-side apply, so one manager owns every field
        secret = k.get("secret", "grafana-admin", "mon")
        self.assertGreaterEqual(len(base64.b64decode(secret["data"]["admin-password"])), 24)

    def test_existing_secret_is_never_rewritten(self):
        k = FakeKube([obj("Namespace", "mon"), obj("Secret", "grafana-admin", "mon")])
        quiet(install.apply, k, opts(), 1)
        self.assertFalse(any(w[0][0] == "create" and json.loads(w[1])["kind"] == "Secret" for w in k.writes))

    def test_apply_removes_grant_of_a_namespace_taken_off_and_stale_cluster_read(self):
        k = FakeKube([obj("Namespace", "mon"), obj("Namespace", "old", managed=False),
                      obj("RoleBinding", "monitoring-read-mon", "old"), obj("Role", "monitoring-read-mon", "old"),
                      obj("RoleBinding", "theirs", "old"),
                      obj("ClusterRole", "mon-prometheus"), obj("ClusterRoleBinding", "mon-prometheus")])
        quiet(install.apply, k, opts(), 1)
        deletes = [w[0] for w in k.writes if w[0][0] == "delete"]
        self.assertIn(["delete", "rolebinding", "monitoring-read-mon", "-n", "old", "--ignore-not-found"], deletes)
        self.assertIn(["delete", "clusterrolebinding", "mon-prometheus"], deletes)
        self.assertFalse(any("theirs" in d for d in deletes))

    def test_apply_refuses_when_grants_cannot_be_listed(self):
        k = FakeKube(fail_list=True)
        with self.assertRaisesRegex(install.Refused, "cannot list"):
            install.apply(k, opts(), 1)
        self.assertEqual(k.writes, [])

    def test_delete_refuses_other_owner_and_foreign_objects(self):
        k = FakeKube([obj("Namespace", "mon", owner="bob")])
        with self.assertRaises(install.Refused):
            install.delete(k, opts(), 1)
        k = FakeKube([obj("Namespace", "mon"), obj("ConfigMap", "theirs", "mon", managed=False)])
        with self.assertRaisesRegex(install.Refused, "ConfigMap mon/theirs"):
            install.delete(k, opts(), 1)
        self.assertFalse(any(w[0][0] == "delete" for w in k.writes))

    def test_delete_refuses_when_grants_cannot_be_listed(self):
        k = FakeKube([obj("Namespace", "mon")], fail_list=True)
        with self.assertRaisesRegex(install.Refused, "cannot list"):
            install.delete(k, opts(), 1)
        self.assertFalse(any(w[0][0] == "delete" for w in k.writes))

    def test_delete_accepts_dependents_defaults_and_uidless_mirrors(self):
        k = FakeKube([obj("Namespace", "mon"), obj("Service", "grafana", "mon", uid="svc"),
                      obj("Pod", "grafana-x", "mon", managed=False, refs=["rs"]),
                      obj("ReplicaSet", "grafana-rs", "mon", managed=False, uid="rs", refs=["dep"]),
                      obj("Deployment", "grafana", "mon", uid="dep"),
                      obj("ConfigMap", "kube-root-ca.crt", "mon", managed=False),
                      obj("RoleBinding", "monitoring-read-mon", "lab"), obj("Role", "monitoring-read-mon", "lab")],
                     kinds=("configmaps", "pods", "services", "replicasets", "deployments", "pods.metrics.k8s.io"))
        quiet(install.delete, k, opts(), 1)
        deletes = [w[0] for w in k.writes if w[0][0] == "delete"]
        self.assertIn(["delete", "rolebinding", "monitoring-read-mon", "-n", "lab", "--ignore-not-found"], deletes)
        self.assertIn(["delete", "namespace", "mon", "--wait=true", "--timeout=1s"], deletes)


class CheckTests(unittest.TestCase):
    def kube(self, targets):
        cfg = R.prometheus_config(opts(scrape_services=["gpu/dcgm:9400"]))
        return FakeKube([obj("Namespace", "mon"), dict(obj("ConfigMap", "prometheus-config", "mon"), data={"prometheus.yml": cfg})],
                        targets=targets)

    def test_an_explicit_exporter_without_an_up_target_is_reported_with_its_own_status(self):
        clock = [0.0]
        real = install.time.monotonic, install.time.sleep
        install.time.monotonic, install.time.sleep = (lambda: clock[0]), (lambda s: clock.__setitem__(0, clock[0] + s))
        try:
            out = io.StringIO()
            with redirect_stdout(out):
                rc = install.check(self.kube([{"labels": {"job": "prometheus", "instance": "localhost:9090"}, "health": "up"}]), opts(), False)
        finally:
            install.time.monotonic, install.time.sleep = real
        self.assertEqual(rc, install.EXPORTER_DOWN)
        self.assertIn("no target of service_gpu_dcgm_9400", out.getvalue())

    def test_a_component_that_does_not_answer_is_a_problem_even_when_an_exporter_is_also_down(self):
        kube = self.kube([])
        kube.proxy_get = lambda ns, svc, path: (_ for _ in ()).throw(RuntimeError("no endpoints")) if svc == "grafana" else json.dumps({"data": {"activeTargets": []}})
        clock = [0.0]
        with mock.patch.object(install.time, "monotonic", lambda: clock[0]), mock.patch.object(install.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s)):
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(install.check(kube, opts(), False), 1)
        self.assertIn("grafana does not answer /api/health", out.getvalue())

    def test_an_explicit_exporter_with_an_up_target_passes(self):
        with redirect_stdout(io.StringIO()):
            rc = install.check(self.kube([{"labels": {"job": "service_gpu_dcgm_9400", "instance": "10.0.0.1:9400"}, "health": "up"}]), opts(), False)
        self.assertEqual(rc, 0)


class ScrapeServiceTests(unittest.TestCase):
    def test_a_named_port_selects_by_port_name_and_a_number_by_target_port(self):
        jobs = {j["job_name"]: j for j in json.loads(R.prometheus_config(opts(scrape_services=["gpu/dcgm:metrics", "gpu/node:9100"])))["scrape_configs"]}
        def keep(job):
            return [(r["source_labels"], r.get("regex")) for r in job["relabel_configs"] if r.get("action") == "keep"]
        self.assertIn((["__meta_kubernetes_endpointslice_port_name"], "metrics"), keep(jobs["service_gpu_dcgm_metrics"]))
        self.assertIn((["__meta_kubernetes_endpointslice_port"], "9100"), keep(jobs["service_gpu_node_9100"]))


class PushTransportTests(unittest.TestCase):
    """push.py --via url against a local HTTP server, and --via kubectl when the proxy permission is denied."""
    def serve(self, status=200, body=b"{}"):
        seen = []
        class Handler(http.server.BaseHTTPRequestHandler):
            def handle_one(self):
                n = int(self.headers.get("Content-Length") or 0)
                seen.append((self.command, self.path, self.rfile.read(n).decode()))
                self.send_response(status)
                self.end_headers()
                self.wfile.write(body)
            do_PUT = do_DELETE = do_GET = handle_one
            def log_message(self, *a):
                pass
        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_port}", seen

    def test_put_delete_and_list_over_a_url(self):
        url, seen = self.serve(body=json.dumps({"data": [{"labels": {"job": "j", "run": "r"}, "push_time_seconds": {"metrics": [{"value": "1"}]}}]}).encode())
        with redirect_stdout(io.StringIO()) as out, mock.patch("sys.stdin", io.StringIO("x 1\n")):
            self.assertEqual(push.main(["put", "--via", "url", "--url", url, "--job", "j", "--group", "run=r"]), 0)
            self.assertEqual(push.main(["delete", "--via", "url", "--url", url, "--job", "j", "--group", "run=r"]), 0)
            self.assertEqual(push.main(["list", "--via", "url", "--url", url, "--job", "j"]), 0)
        self.assertEqual([(m, p) for m, p, _ in seen], [("PUT", "/metrics/job/j/run/r"), ("DELETE", "/metrics/job/j/run/r"), ("GET", "/api/v1/metrics")])
        self.assertEqual(seen[0][2], "x 1\n")
        self.assertIn('"run": "r"', out.getvalue())

    def test_an_http_error_is_a_failure_with_its_status(self):
        url, _ = self.serve(status=500, body=b"boom")
        with redirect_stderr(io.StringIO()) as err, mock.patch("sys.stdin", io.StringIO("x 1\n")):
            self.assertEqual(push.main(["put", "--via", "url", "--url", url, "--job", "j"]), 1)
        self.assertIn("HTTP 500: boom", err.getvalue())

    def test_a_denied_proxy_permission_stops_before_anything_is_sent(self):
        calls = []
        def run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 1, "no\n", "")
        with mock.patch.object(push.subprocess, "run", run), redirect_stderr(io.StringIO()) as err, mock.patch("sys.stdin", io.StringIO("x 1\n")):
            self.assertEqual(push.main(["put", "--via", "kubectl", "--job", "j"]), 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1:4], ["auth", "can-i", "update"])
        self.assertIn("may not update services/proxy", err.getvalue())


class PushTests(unittest.TestCase):
    def test_group_path_encoding(self):
        self.assertEqual(push.group_path("disagg", {"run": "a b", "instance": "x1", "e": ""}),
                         "/metrics/job/disagg/e@base64/=/instance/x1/run@base64/" + base64.urlsafe_b64encode(b"a b").decode())
        self.assertEqual(push.segment("k", ".."), "/k@base64/" + base64.urlsafe_b64encode(b"..").decode())
        self.assertEqual(push.segment("k", "a/b"), "/k@base64/" + base64.urlsafe_b64encode(b"a/b").decode())
        with self.assertRaises(push.Invalid):
            push.group_path("j", {"job": "x"})
        with self.assertRaises(push.Invalid):
            push.segment("bad-key", "v")

    def test_validate(self):
        good = '# TYPE disagg_x gauge\ndisagg_x{architecture="unified",rate="1"} 2.5\ndisagg_y 3\n'
        self.assertEqual(push.validate(good, {}), 2)
        bad = ['disagg_x 1\ndisagg_x 2\n', 'disagg_x NaN\n', '# TYPE c counter\nc 1\n', 'x{__a="1"} 1\n',
               'x{a="1",a="2"} 1\n', '', 'x{run="b"} 1\n', '1x 2\n', 'x 1 1700000000000\n', 'x{a="\\q"} 1\n']
        for b in bad:
            with self.assertRaises(push.Invalid, msg=b):
                push.validate(b, {"run": "a"})
        with self.assertRaises(push.Invalid):
            push.validate('x{job="other"} 1\n', {}, "mine")
        self.assertEqual(push.validate('x{run="a\\\\b",q="say \\"hi\\"\\n"} 1\n', {"run": "a\\b"}), 1)

    def test_cli_rejects_before_sending(self):
        with redirect_stderr(io.StringIO()):
            self.assertEqual(push.main(["put", "--via", "url"]), 2)
            self.assertEqual(push.main(["put", "--via", "url", "--url", "http://x", "--group", "noeq", "--job", "j"]), 2)


if __name__ == "__main__":
    unittest.main()
