"""Render or apply the lab's namespace-scoped EKS resources."""
import argparse
import json
from pathlib import Path
import re
import subprocess

OWNER = {"app.kubernetes.io/part-of": "aim345-lab"}


def read_config(path):
    c = json.loads(Path(path).read_text())
    assert all("REPLACE" not in str(v) for v in c.values()), "Fill config.example.json first"
    assert re.fullmatch(r"[a-z][a-z0-9-]{0,61}[a-z0-9]", c["namespace"])
    assert c["namespace"].startswith("aim345-"), "Use a dedicated aim345-* namespace"
    assert re.fullmatch(r"[a-f0-9]{40}", c["model_revision"])
    for key in ("image", "router_image"):
        image = c[key]
        assert "@sha256:" in image or (":" in image.rsplit("/", 1)[-1] and not image.endswith(":latest"))
    assert c["gpus_per_worker"] > 0 and c["efa_per_worker"] > 0
    minimum_nodes = 1 if c.get("placement") == "packed" else 2
    assert c.get("placement", "separate-nodes") in ("packed", "separate-nodes")
    assert len(c["nodes"]) == len(set(c["nodes"])) >= minimum_nodes
    assert c["model_cache_host_path"].startswith("/mnt/"), "Use a dedicated cache under /mnt"
    return c


def kubectl(c, *args, **kwargs):
    return subprocess.run(["kubectl", "--context", c["context"], "-n", c["namespace"], *args], check=True, text=True, **kwargs)


def owned_namespace(c, create=False):
    r = subprocess.run(["kubectl", "--context", c["context"], "get", "namespace", c["namespace"], "--ignore-not-found", "-o", "json"], text=True, capture_output=True, check=True)
    priority = subprocess.run(["kubectl", "--context", c["context"], "get", "priorityclass", c["namespace"] + "-nonpreempting", "--ignore-not-found", "-o", "json"], text=True, capture_output=True, check=True)
    if priority.stdout.strip():
        assert json.loads(priority.stdout)["metadata"].get("labels", {}).get("app.kubernetes.io/part-of") == "aim345-lab", "PriorityClass is not owned by this lab"
    if r.stdout.strip():
        assert json.loads(r.stdout)["metadata"].get("labels", {}).get("app.kubernetes.io/part-of") == "aim345-lab", "Namespace is not owned by this lab"
    elif create:
        kubectl(c, "create", "-f", "-", input=json.dumps({"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": c["namespace"], "labels": OWNER}}))
    else:
        raise RuntimeError("Lab namespace does not exist")


def metadata(c, name, labels=None):
    return {"name": name, "namespace": c["namespace"], "labels": OWNER | (labels or {})}


def service(c, name, port, headless=False):
    spec = {"selector": OWNER | {"app": name}, "ports": [{"name": "http", "port": port, "targetPort": port}]}
    if headless:
        spec.update(clusterIP="None", publishNotReadyAddresses=True)
    return {"apiVersion": "v1", "kind": "Service", "metadata": metadata(c, name), "spec": spec}


def worker(c, name, node, role):
    labels = OWNER | {"app": name, "role": role, "component": "engine"}
    model_path = "/models/" + c["model_revision"]
    args = ["--model-path", model_path, "--served-model-name", "aim345", "--host", "0.0.0.0", "--port", "30000", "--tp-size", str(c["gpus_per_worker"]), "--dtype", "bfloat16", "--kv-cache-dtype", "bf16", "--context-length", str(c["context_length_tokens"]), "--chunked-prefill-size", str(c["chunked_prefill_size_tokens"]), "--mem-fraction-static", "0.75", "--enable-metrics", "--stream-interval", "1"]
    if c.get("disable_prefix_cache"):
        args.append("--disable-radix-cache")
    if role != "unified":
        args += ["--disaggregation-mode", role, "--disaggregation-transfer-backend", "nixl"]
        if role == "prefill":
            args += ["--disaggregation-bootstrap-port", "8998"]
    resources = {"nvidia.com/gpu": c["gpus_per_worker"], "vpc.amazonaws.com/efa": c["efa_per_worker"], "cpu": "16", "memory": "128Gi", "ephemeral-storage": c["engine_ephemeral_storage"]}
    env = [{"name": "SGLANG_DISAGGREGATION_NIXL_BACKEND", "value": "LIBFABRIC"}, {"name": "FI_PROVIDER", "value": "efa"}, {"name": "FI_EFA_USE_DEVICE_RDMA", "value": "1"}, {"name": "FI_LOG_LEVEL", "value": "info"}]
    mounts = [{"name": "models", "mountPath": "/models"}, {"name": "shm", "mountPath": "/dev/shm"}]
    download = "from huggingface_hub import snapshot_download; import sys; snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])"
    spec = {
        "hostNetwork": True, "dnsPolicy": "ClusterFirstWithHostNet", "preemptionPolicy": "Never", "priorityClassName": c["namespace"] + "-nonpreempting", "automountServiceAccountToken": False,
        "nodeSelector": {"kubernetes.io/hostname": node, "node.kubernetes.io/instance-type": c["instance_type"]},
        "tolerations": c["tolerations"],
        "initContainers": [{"name": "model", "image": c["image"], "command": ["python3", "-c", download, c["model_id"], c["model_revision"], model_path], "volumeMounts": [mounts[0]], "resources": {"requests": {"cpu": "2", "memory": "4Gi"}}}],
        "containers": [{"name": "engine", "image": c["image"], "imagePullPolicy": "IfNotPresent", "command": ["bash", "-c", 'ulimit -l unlimited; python3 /lab/verify_image.py && exec python3 -m sglang.launch_server "$@"', "engine"], "args": args, "env": env, "resources": {"requests": resources, "limits": resources}, "securityContext": {"capabilities": {"add": ["IPC_LOCK"]}}, "volumeMounts": mounts, "startupProbe": {"httpGet": {"path": "/health", "port": 30000}, "periodSeconds": 10, "timeoutSeconds": 10, "failureThreshold": 180}, "readinessProbe": {"httpGet": {"path": "/health", "port": 30000}, "periodSeconds": 10, "timeoutSeconds": 10}}],
        "volumes": [{"name": "models", "hostPath": {"path": c["model_cache_host_path"], "type": "DirectoryOrCreate"}}, {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "32Gi"}}],
    }
    return {"apiVersion": "apps/v1", "kind": "Deployment", "metadata": metadata(c, name, labels), "spec": {"replicas": 1, "strategy": {"type": "Recreate"}, "selector": {"matchLabels": {"app": name}}, "template": {"metadata": {"labels": labels}, "spec": spec}}}


def render(c, mode, prefill_replicas=1):
    assert prefill_replicas in (1, 2), "This lab implements one or two prefill workers"
    if mode == "unified":
        roles = [("unified-0", c["nodes"][0], "unified"), ("unified-1", c["nodes"][1], "unified")]
    else:
        assert len(c["nodes"]) >= prefill_replicas + 1, "Add a separately allocated node to scale prefill"
        roles = [("prefill-0", c["nodes"][0], "prefill"), ("decode-0", c["nodes"][1], "decode")]
        if prefill_replicas == 2:
            roles.append(("prefill-1", c["nodes"][2], "prefill"))
    items = [{"apiVersion": "scheduling.k8s.io/v1", "kind": "PriorityClass", "metadata": {"name": c["namespace"] + "-nonpreempting", "labels": OWNER}, "value": -10, "preemptionPolicy": "Never", "globalDefault": False, "description": "Nonpreempting priority for the AIM345 lab namespace"}]
    for name, node, role in roles:
        items += [worker(c, name, node, role), service(c, name, 30000, headless=True)]
    # Use individual worker endpoints: one Service balancing replicas hides them from the router.
    url = lambda name: f"http://{name}.{c['namespace']}.svc.cluster.local:30000"
    args = ["--host", "0.0.0.0", "--port", "8000", "--policy", "round_robin", "--disable-retries", "--request-timeout-secs", "330"]
    if mode == "unified":
        args += ["--worker-urls", url("unified-0"), url("unified-1")]
    else:
        args += ["--pd-disaggregation"]
        for name, _, role in roles:
            args += (["--prefill", url(name), "8998"] if role == "prefill" else ["--decode", url(name)])
    labels = OWNER | {"app": "router", "component": "router"}
    pod = {"automountServiceAccountToken": False, "preemptionPolicy": "Never", "priorityClassName": c["namespace"] + "-nonpreempting", "containers": [{"name": "router", "image": c["router_image"], "command": ["python3", "-m", "sglang_router.launch_router"], "args": args, "resources": {"requests": {"cpu": "2", "memory": "4Gi", "ephemeral-storage": c["router_ephemeral_storage"]}, "limits": {"cpu": "4", "memory": "8Gi", "ephemeral-storage": c["router_ephemeral_storage"]}}, "readinessProbe": {"httpGet": {"path": "/health", "port": 8000}, "periodSeconds": 5}}]}
    items += [{"apiVersion": "apps/v1", "kind": "Deployment", "metadata": metadata(c, "router", labels), "spec": {"replicas": 1, "strategy": {"type": "Recreate"}, "selector": {"matchLabels": {"app": "router"}}, "template": {"metadata": {"labels": labels}, "spec": pod}}}, service(c, "router", 8000)]
    return {"apiVersion": "v1", "kind": "List", "items": items}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["unified", "disaggregated", "scale-prefill", "cleanup"])
    p.add_argument("--config", default="config.json")
    p.add_argument("--render", action="store_true")
    a = p.parse_args()
    c = read_config(a.config)
    assert c.get("placement") != "packed", "Use paired.py for packed allocations"
    if a.mode == "cleanup":
        owned_namespace(c)
        kubectl(c, "delete", "namespace", c["namespace"], "--wait=true")
        kubectl(c, "delete", "priorityclass", c["namespace"] + "-nonpreempting", "--ignore-not-found")
        return
    mode = "disaggregated" if a.mode == "scale-prefill" else a.mode
    doc = render(c, mode, 2 if a.mode == "scale-prefill" else 1)
    if a.render:
        print(json.dumps(doc, indent=2))
        return
    owned_namespace(c, create=True)
    if a.mode != "scale-prefill":
        kubectl(c, "delete", "deployment,service", "-l", "app.kubernetes.io/part-of=aim345-lab", "--ignore-not-found", "--wait=true")
        remaining = json.loads(kubectl(c, "get", "pods", "-l", "app.kubernetes.io/part-of=aim345-lab", "-o", "json", capture_output=True).stdout)["items"]
        for pod in remaining:
            kubectl(c, "wait", "--for=delete", "pod/" + pod["metadata"]["name"], "--timeout=180s")
    else:
        # Scaling adds a prefill worker and changes router membership only.
        existing = json.loads(kubectl(c, "get", "deploy", "-o", "json", capture_output=True).stdout)
        assert {"prefill-0", "decode-0", "router"} <= {x["metadata"]["name"] for x in existing["items"]}, "Deploy disaggregated first"
        # Reject configuration drift that would also restart existing engines.
        desired = {x["metadata"]["name"]: x for x in doc["items"] if x["kind"] == "Deployment"}
        for old in existing["items"]:
            if old["metadata"]["name"] not in ("prefill-0", "decode-0"):
                continue
            want = desired[old["metadata"]["name"]]["spec"]["template"]["spec"]
            have = old["spec"]["template"]["spec"]
            for key in ("nodeSelector", "tolerations"):
                assert have[key] == want[key], f"Existing engine {key} changed"
            for key in ("image", "args", "env"):
                assert have["containers"][0][key] == want["containers"][0][key], f"Existing engine {key} changed"
            normalize = lambda resources: {section: {k: str(v) for k, v in quantities.items()} for section, quantities in resources.items()}
            assert normalize(have["containers"][0]["resources"]) == normalize(want["containers"][0]["resources"]), "Existing engine resources changed"
        doc["items"] = [x for x in doc["items"] if x["metadata"]["name"] in ("prefill-1", "router")]
    kubectl(c, "apply", "-f", "-", input=json.dumps(doc))
    Path("results").mkdir(exist_ok=True)
    Path("results/deployment.json").write_text(json.dumps({"config": c, "mode": mode, "prefill_replicas": 2 if a.mode == "scale-prefill" else 1}, indent=2))
    print("Submitted. Wait for each Deployment to become Available; capture pod events if scheduling is blocked.")

if __name__ == "__main__":
    main()
