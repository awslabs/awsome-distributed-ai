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
    assert re.fullmatch(r"[a-f0-9]{40}", c["model_revision"])
    assert c["gpus_per_worker"] > 0 and c["efa_per_worker"] > 0
    minimum_nodes = 1 if c.get("placement") == "packed" else 2
    assert c.get("placement", "separate-nodes") in ("packed", "separate-nodes")
    assert len(c["nodes"]) == len(set(c["nodes"])) >= minimum_nodes
    if c.get('engine_profile') == 'v4-optional' or c['model_id'] == 'deepseek-ai/DeepSeek-V4-Flash':
        assert c.get('engine_profile') == 'v4-optional', 'V4 requires the separately pinned optional engine'
        assert c['model_id'] == 'deepseek-ai/DeepSeek-V4-Flash', 'The optional profile is qualified only for V4'
        assert c['instance_type'] == 'g7.48xlarge' and c['gpus_per_worker'] == 8 and c['efa_per_worker'] == 2
        assert c.get('placement', 'separate-nodes') == 'separate-nodes' and len(c['nodes']) == 2
    return c


def kubectl(c, *args, **kwargs):
    return subprocess.run(["kubectl", "--context", c["context"], "-n", c["namespace"], *args], check=True, text=True, **kwargs)


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
    if c.get('engine_profile') == 'v4-optional':
        # Qualified V4 profile has quantized weights and needs more static memory.
        for flag in ('--dtype', '--kv-cache-dtype'):
            index = args.index(flag)
            del args[index:index + 2]
        args[args.index('--mem-fraction-static') + 1] = '0.90'
        args += ['--max-running-requests', '4', '--cuda-graph-max-bs-decode', '4', '--disable-custom-all-reduce', '--disable-flashinfer-autotune']
    if c.get("disable_prefix_cache"):
        args.append("--disable-radix-cache")
    if role != "unified":
        args += ["--disaggregation-mode", role, "--disaggregation-transfer-backend", "nixl"]
        if role == "prefill":
            args += ["--disaggregation-bootstrap-port", "8998"]
    resources = {"nvidia.com/gpu": c["gpus_per_worker"], "vpc.amazonaws.com/efa": c["efa_per_worker"], "cpu": str(c.get("engine_cpu", "16")), "memory": c.get("engine_memory", "128Gi"), "ephemeral-storage": c["engine_ephemeral_storage"]}
    env = [{"name": "SGLANG_DISAGGREGATION_NIXL_BACKEND", "value": "LIBFABRIC"}, {"name": "FI_PROVIDER", "value": "efa"}, {"name": "FI_EFA_USE_DEVICE_RDMA", "value": "1"}, {"name": "FI_LOG_LEVEL", "value": "info"}]
    mounts = [{"name": "models", "mountPath": "/models"}, {"name": "shm", "mountPath": "/dev/shm"}]
    download = "from huggingface_hub import snapshot_download; import sys; snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])"
    spec = {
        "hostNetwork": True, "dnsPolicy": "ClusterFirstWithHostNet", "preemptionPolicy": "Never", "priorityClassName": c["namespace"] + "-nonpreempting", "automountServiceAccountToken": False,
        "nodeSelector": {"kubernetes.io/hostname": node, "node.kubernetes.io/instance-type": c["instance_type"]},
        "tolerations": c["tolerations"],
        "initContainers": [{"name": "model", "image": c["image"], "command": ["python3", "-c", download, c["model_id"], c["model_revision"], model_path], "volumeMounts": [mounts[0]], "resources": {"requests": {"cpu": "2", "memory": "4Gi"}, "limits": {"cpu": "2", "memory": "4Gi"}}}],
        "containers": [{"name": "engine", "image": c["image"], "imagePullPolicy": "IfNotPresent", "command": ["bash", "-c", 'ulimit -l unlimited; python3 /lab/verify_image.py && exec python3 -m sglang.launch_server "$@"', "engine"], "args": args, "env": env, "resources": {"requests": resources, "limits": resources}, "securityContext": {"capabilities": {"add": ["IPC_LOCK"]}}, "volumeMounts": mounts, "startupProbe": {"httpGet": {"path": "/health", "port": 30000}, "periodSeconds": 10, "timeoutSeconds": 10, "failureThreshold": 180}, "readinessProbe": {"httpGet": {"path": "/health", "port": 30000}, "periodSeconds": 10, "timeoutSeconds": 10}}],
        "volumes": [{"name": "models", "hostPath": {"path": c["model_cache_host_path"], "type": "DirectoryOrCreate"}}, {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "32Gi"}}],
    }
    if c.get('engine_cache_host_path'):
        assert c['engine_cache_host_path'].startswith(c['model_cache_host_path'] + '/')
        spec['volumes'].append({'name': 'engine-cache', 'hostPath': {'path': c['engine_cache_host_path'], 'type': 'DirectoryOrCreate'}})
        mounts.append({'name': 'engine-cache', 'mountPath': '/root/.cache'})
    return {"apiVersion": "apps/v1", "kind": "Deployment", "metadata": metadata(c, name, labels), "spec": {"replicas": 1, "strategy": {"type": "Recreate"}, "selector": {"matchLabels": {"app": name}}, "template": {"metadata": {"labels": labels}, "spec": spec}}}


def render(c, mode):
    if mode == "unified":
        roles = [("unified-0", c["nodes"][0], "unified"), ("unified-1", c["nodes"][1], "unified")]
    else:
        roles = [("prefill-0", c["nodes"][0], "prefill"), ("decode-0", c["nodes"][1], "decode")]
    items = [{"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": c["namespace"], "labels": OWNER}},
             {"apiVersion": "scheduling.k8s.io/v1", "kind": "PriorityClass", "metadata": {"name": c["namespace"] + "-nonpreempting", "labels": OWNER}, "value": -10, "preemptionPolicy": "Never", "globalDefault": False, "description": "Nonpreempting priority for the AIM345 lab namespace"}]
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
    p.add_argument("mode", choices=["unified", "disaggregated", "cleanup"])
    p.add_argument("--config", default="config.json")
    p.add_argument("--render", action="store_true")
    a = p.parse_args()
    c = read_config(a.config)
    assert c.get("placement") != "packed", "Use paired.py for packed allocations"
    if a.mode == "cleanup":
        kubectl(c, "delete", "namespace", c["namespace"], "--wait=true")
        kubectl(c, "delete", "priorityclass", c["namespace"] + "-nonpreempting", "--ignore-not-found")
        return
    doc = render(c, a.mode)
    if a.render:
        print(json.dumps(doc, indent=2))
        return
    # Replace the lab's previous serving resources, and wait until their pods have released the GPUs.
    kubectl(c, "delete", "deployment,service", "-l", "app.kubernetes.io/part-of=aim345-lab", "--ignore-not-found", "--wait=true")
    remaining = json.loads(kubectl(c, "get", "pods", "-l", "app.kubernetes.io/part-of=aim345-lab", "-o", "json", capture_output=True).stdout)["items"]
    for pod in remaining:
        kubectl(c, "wait", "--for=delete", "pod/" + pod["metadata"]["name"], "--timeout=180s")
    kubectl(c, "apply", "-f", "-", input=json.dumps(doc))
    Path("results").mkdir(exist_ok=True)
    Path("results/deployment.json").write_text(json.dumps({"config": c, "mode": a.mode}, indent=2))
    print("Submitted. Wait for each Deployment to become Available; capture pod events if scheduling is blocked.")

if __name__ == "__main__":
    main()
