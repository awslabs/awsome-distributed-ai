#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0
# Throughput + latency-vs-concurrency probe for the live DP32/EP32 vLLM serve.
# Stdlib only (urllib + concurrent.futures) so it runs in the pod with no pip.
# Fires a fixed set of concurrent chat-completions per concurrency level, measures
# aggregate output tok/s + per-request latency percentiles. 127.0.0.1 loopback only.
# HONEST framing: pod is ~13h old with zombie python3 present (pod-freshness rule I685),
# so this is an AT-SCALE THROUGHPUT + relative latency-shape datapoint, NOT a pristine
# p50 latency baseline. Purpose: first quantitative number at EP32 scale (we only had
# functional 16/16 PASS before).
import json, time, urllib.request, urllib.error, statistics, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "Qwen/Qwen3-30B-A3B-FP8"
MAX_TOKENS = 128
PROMPT = "Explain expert parallelism in mixture-of-experts models in two sentences."
CONCURRENCIES = [1, 8, 16, 32, 64]

def one_request():
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.loads(r.read())
        dt = time.time() - t0
        ct = d.get("usage", {}).get("completion_tokens", 0)
        return (True, dt, ct, r.status if hasattr(r, "status") else 200)
    except urllib.error.HTTPError as e:
        return (False, time.time() - t0, 0, e.code)
    except Exception as e:
        return (False, time.time() - t0, 0, str(e)[:40])

def pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    f = int(k)
    return xs[f] if f + 1 >= len(xs) else xs[f] + (xs[f + 1] - xs[f]) * (k - f)

def sweep(conc):
    t0 = time.time()
    lat, toks, ok, codes = [], [], 0, {}
    with ThreadPoolExecutor(max_workers=conc) as ex:
        futs = [ex.submit(one_request) for _ in range(conc)]
        for fu in as_completed(futs):
            success, dt, ct, code = fu.result()
            lat.append(dt); toks.append(ct)
            if success:
                ok += 1
            codes[str(code)] = codes.get(str(code), 0) + 1
    wall = time.time() - t0
    total_tok = sum(toks)
    return {
        "conc": conc, "ok": ok, "wall_s": round(wall, 2),
        "total_out_tok": total_tok,
        "agg_tok_s": round(total_tok / wall, 1) if wall > 0 else 0,
        "lat_p50_s": round(pct(lat, 50), 2), "lat_p90_s": round(pct(lat, 90), 2),
        "lat_p99_s": round(pct(lat, 99), 2), "lat_max_s": round(max(lat), 2) if lat else 0,
        "codes": codes,
    }

if __name__ == "__main__":
    print("=== DP32/EP32 live-serve throughput probe ===")
    print(f"url={URL} model={MODEL} max_tokens={MAX_TOKENS}")
    print(f"{'conc':>5} {'ok':>4} {'wall_s':>7} {'out_tok':>8} {'agg_tok/s':>10} {'p50_s':>7} {'p90_s':>7} {'p99_s':>7} codes")
    rows = []
    for c in CONCURRENCIES:
        r = sweep(c)
        rows.append(r)
        print(f"{r['conc']:>5} {r['ok']:>4} {r['wall_s']:>7} {r['total_out_tok']:>8} "
              f"{r['agg_tok_s']:>10} {r['lat_p50_s']:>7} {r['lat_p90_s']:>7} {r['lat_p99_s']:>7} {r['codes']}")
    print("JSON_ROWS=" + json.dumps(rows))
