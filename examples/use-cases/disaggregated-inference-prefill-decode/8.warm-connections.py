#!/usr/bin/env python3
"""Prime every fixed P/D pair before measurement, then flush prompt caches."""
import argparse
import asyncio
import json
from pathlib import Path
import httpx
from benchmark import stream_request
from deployment import kubectl, owned_namespace, read_config
from traffic import tokenizer, tokens


async def warm(c, url, pairs):
    tok = tokenizer(c["model_id"], c["model_revision"])
    ids = tokens(tok, f"{tok.bos_token}User: Say ready.\n\nAssistant:")
    results = []
    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
        for _ in range(pairs * 2):
            r = await stream_request(client, url, ids, 4, 310)
            results.append(r)
            assert r["ok"], r
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.json")
    p.add_argument("--url", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    c = read_config(a.config)
    owned_namespace(c)
    pods = json.loads(kubectl(c, "get", "pods", "-l", "component=engine", "-o", "json", capture_output=True).stdout)["items"]
    packed = c.get('placement') == 'packed'
    prefill = pods if packed else [x for x in pods if x["metadata"]["labels"]["role"] == "prefill"]
    decode = pods if packed else [x for x in pods if x["metadata"]["labels"]["role"] == "decode"]
    assert prefill and len(decode) == 1, "Fixed round-robin P/D topology required"
    r = asyncio.run(warm(c, a.url, len(prefill)))
    flush = {}
    for pod in pods:
        for port in ((30000, 30001) if packed else (30000,)):
            code = f"import urllib.request; r=urllib.request.urlopen('http://127.0.0.1:{port}/flush_cache'); print(r.status, r.read().decode())"
            key = f"{pod['metadata']['name']}:{port}" if packed else pod['metadata']['name']
            flush[key] = kubectl(c, "exec", pod["metadata"]["name"], "-c", "engine", "--", "python3", "-c", code, capture_output=True).stdout
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"instance_type": c["instance_type"], "evidence_scope": "mechanism-validation", "method": "startup request priming followed by cache flush", "makeConnection_bootstrap_fix": "UNVALIDATED: requires upstream SGLang support; no local patch", "requests": r, "cache_flush": flush}, indent=2))
    print(str(out))

if __name__ == "__main__":
    main()
