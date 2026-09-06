#!/usr/bin/env python3
"""Check installed versions, both launch selectors, and EFA RDMA byte deltas."""
import argparse
import asyncio
import json
from pathlib import Path
from benchmark import stream_request
from deployment import kubectl, owned_namespace, read_config
from traffic import tokenizer, initial_prompt
import httpx

COUNTERS = '''import pathlib,json
values={}
for p in pathlib.Path('/sys/class/infiniband').glob('*/ports/*/hw_counters/*'):
    if p.name in ('rdma_write_bytes','rdma_read_bytes','send_bytes','recv_bytes'):
        try: values[str(p)]=int(p.read_text().strip())
        except (OSError,ValueError): pass
print(json.dumps(values))'''


def snapshot(c, pods):
    return {p["metadata"]["name"]: json.loads(kubectl(c, "exec", p["metadata"]["name"], "-c", "engine", "--", "python3", "-c", COUNTERS, capture_output=True).stdout) for p in pods}


async def smoke(c, url):
    tok = tokenizer(c["model_id"], c["model_revision"])
    shape = json.loads(Path("shape-b.json").read_text())
    shape["input_tokens"] = 1024
    import uuid
    ids = initial_prompt(tok, shape, "transport", uuid.uuid4().hex)
    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
        return await stream_request(client, url, ids, 4, 310)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.json")
    p.add_argument("--url", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--bandwidth-gib-s", type=float, help="Measured GPU-buffer fabric bandwidth, never a datasheet peak")
    p.add_argument("--bandwidth-evidence", help="Path to the matching hardware microbenchmark transcript")
    a = p.parse_args()
    c = read_config(a.config)
    owned_namespace(c)
    pods = json.loads(kubectl(c, "get", "pods", "-l", "component=engine", "-o", "json", capture_output=True).stdout)["items"]
    assert len(pods) >= 2
    assert len({x["spec"]["nodeName"] for x in pods}) == len(pods), "Transfer must cross nodes"
    versions = {}
    for pod in pods:
        engine = next(x for x in pod["spec"]["containers"] if x["name"] == "engine")
        args = engine["args"]
        env = {x["name"]: x.get("value") for x in engine["env"]}
        assert args[args.index("--disaggregation-transfer-backend") + 1] == "nixl"
        assert "--disaggregation-ib-device" not in args
        assert env["SGLANG_DISAGGREGATION_NIXL_BACKEND"] == "LIBFABRIC"
        assert env["FI_PROVIDER"] == "efa"
        versions[pod["metadata"]["name"]] = json.loads(kubectl(c, "exec", pod["metadata"]["name"], "-c", "engine", "--", "python3", "/lab/verify_image.py", capture_output=True).stdout)
    before = snapshot(c, pods)
    response = asyncio.run(smoke(c, a.url))
    after = snapshot(c, pods)
    deltas = {pod: {key: after[pod][key] - value for key, value in values.items()} for pod, values in before.items()}
    rdma_bytes = sum(max(0, value) for values in deltas.values() for key, value in values.items() if key.endswith(("rdma_write_bytes", "rdma_read_bytes")))
    result = {"instance_type": c["instance_type"], "evidence_scope": "mechanism-validation", "versions": versions, "pods": pods, "response": response, "before_bytes": before, "after_bytes": after, "delta_bytes": deltas,
              "status": "VALIDATED" if response["ok"] and rdma_bytes > 0 else "UNVALIDATED",
              "scope": "Positive configured path only. Shared-node counters include other traffic; inspect provider/GPU registration logs. This does not validate negative selector controls or GPUDirect absence of staging by itself."}
    if a.bandwidth_gib_s is not None:
        assert a.bandwidth_gib_s > 0 and a.bandwidth_evidence
        transcript = Path(a.bandwidth_evidence).read_text()
        # MLA logical cache, all layers. TP replication and protocol overhead are not included.
        import urllib.request
        cfg = json.load(urllib.request.urlopen(f"https://huggingface.co/{c['model_id']}/resolve/{c['model_revision']}/config.json"))
        logical_bytes = 1024 * cfg["num_hidden_layers"] * (cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"]) * 2
        result["attribution"] = {"input_tokens": 1024, "logical_kv_bytes": logical_bytes, "bf16_bytes_per_element": 2, "measured_fabric_gib_per_s": a.bandwidth_gib_s, "ideal_wire_time_ms": logical_bytes / (a.bandwidth_gib_s * 1024**3) * 1000, "bandwidth_evidence": transcript, "qualification": "Lower-bound estimate; count all layers and tensor-parallel replication before attributing a residual to connection setup"}
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"status": result["status"], "instance_type": c["instance_type"], "rdma_counter_delta_bytes": rdma_bytes, "file": str(out)}))
    if result["status"] != "VALIDATED":
        raise SystemExit(1)

if __name__ == "__main__":
    main()
