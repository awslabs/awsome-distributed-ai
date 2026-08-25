#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0
# Throughput + latency-vs-concurrency probe for the live trtllm-serve NcclEP EP serve.
# Stdlib only (urllib + concurrent.futures) so it runs in the pod with no pip.
# Methodology (fail-loud, distribution-honest):
#   - each level fires requests-per-level = conc * NREQ_MULT requests (default 5x) at a
#     fixed concurrency, so percentiles describe a distribution, not a single shot
#   - only SUCCESSFUL requests enter the latency percentiles and the token numerator;
#     failures are counted separately and fail the run (a partially-dead serve must not
#     report a *better* p50 because refused connections return fast)
#   - "ignore_eos": true pins every request to exactly max_tokens generated tokens, so
#     the tok/s denominator is fixed by construction rather than model/prompt-dependent
#   - each request carries a unique prompt PREFIX (the index goes first), so a prefix/KV
#     cache cannot serve request N's prefill from request 1's
#   - a missing usage block in a 200 response is a FAILURE (code "200-no-usage"), not a
#     zero-token success that silently deflates throughput
#   - exit 0 only if EVERY request at EVERY level succeeded
import argparse, json, os, time, urllib.request, urllib.error, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Defaults keep the probe runnable standalone inside the pod (127.0.0.1 loopback);
# recipe/benchmark.sh overrides --url (leader IP) and --out (raw JSONL path).
URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = os.environ.get("SERVE_MODEL", "Qwen/Qwen3-30B-A3B")
MAX_TOKENS = 128
PROMPT = "Explain expert parallelism in mixture-of-experts models in two sentences."
CONCURRENCIES = [1, 2, 4]  # sized to the measured serve shape (--max_batch_size 4); raise
                           # SERVE_MAX_BATCH_SIZE/SERVE_MAX_NUM_TOKENS before sweeping higher
NREQ_MULT = 5  # requests per level = conc * this (>=5x so p50/p90 are distributions)

def one_request(url, idx):
    body = json.dumps({
        "model": MODEL,
        # unique prefix per request (index FIRST) so a prefix cache cannot make
        # prefill free for requests 2..N — see benchmarks/README methodology notes
        "messages": [{"role": "user", "content": f"[request {idx}] {PROMPT}"}],
        "max_tokens": MAX_TOKENS,
        "ignore_eos": True,   # pin generated tokens == max_tokens (fixed denominator)
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.loads(r.read())
        dt = time.time() - t0
        usage = d.get("usage")
        if not usage or "completion_tokens" not in usage:
            # a 200 without usage is a malformed success — surface it, don't count 0 tokens
            return (False, dt, 0, "200-no-usage")
        return (True, dt, usage["completion_tokens"], 200)
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

def sweep(conc, url, nreq_mult):
    n = conc * nreq_mult
    t0 = time.time()
    lat, toks, ok, codes = [], [], 0, {}
    with ThreadPoolExecutor(max_workers=conc) as ex:
        futs = [ex.submit(one_request, url, i) for i in range(n)]
        for fu in as_completed(futs):
            success, dt, ct, code = fu.result()
            if success:
                ok += 1
                lat.append(dt); toks.append(ct)   # successes only: failures must not
                # deflate percentiles or ride in the throughput denominator
            codes[str(code)] = codes.get(str(code), 0) + 1
    wall = time.time() - t0
    total_tok = sum(toks)
    return {
        "conc": conc, "n": n, "ok": ok, "wall_s": round(wall, 2),
        "total_out_tok": total_tok,
        "agg_tok_s": round(total_tok / wall, 1) if wall > 0 else 0,
        "lat_p50_s": round(pct(lat, 50), 2), "lat_p90_s": round(pct(lat, 90), 2),
        "lat_p99_s": round(pct(lat, 99), 2), "lat_max_s": round(max(lat), 2) if lat else 0,
        "codes": codes,
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="trtllm-serve NcclEP live-serve throughput/latency probe")
    ap.add_argument("--url", default=URL, help="chat-completions endpoint (default: 127.0.0.1 loopback)")
    ap.add_argument("--out", default=None, help="write one JSON object per concurrency level to this path (JSONL)")
    ap.add_argument("--requests-per-level-mult", type=int, default=NREQ_MULT,
                    help=f"requests per level = concurrency x this (default {NREQ_MULT})")
    args = ap.parse_args()

    print("=== trtllm-serve NcclEP live-serve throughput probe ===")
    print(f"url={args.url} model={MODEL} max_tokens={MAX_TOKENS} (ignore_eos) "
          f"requests/level={args.requests_per_level_mult}x concurrency")
    print(f"{'conc':>5} {'n':>5} {'ok':>5} {'wall_s':>7} {'out_tok':>8} {'agg_tok/s':>10} {'p50_s':>7} {'p90_s':>7} {'p99_s':>7} codes")
    rows = []
    out_fh = open(args.out, "w") if args.out else None
    try:
        for c in CONCURRENCIES:
            r = sweep(c, args.url, args.requests_per_level_mult)
            rows.append(r)
            print(f"{r['conc']:>5} {r['n']:>5} {r['ok']:>5} {r['wall_s']:>7} {r['total_out_tok']:>8} "
                  f"{r['agg_tok_s']:>10} {r['lat_p50_s']:>7} {r['lat_p90_s']:>7} {r['lat_p99_s']:>7} {r['codes']}")
            if out_fh:
                out_fh.write(json.dumps(r) + "\n"); out_fh.flush()
    finally:
        if out_fh:
            out_fh.close()
    print("JSON_ROWS=" + json.dumps(rows))
    # fail-loud contract: EVERY request at EVERY level must have succeeded — one dead
    # backend at one level is a failed run, not a footnote (matches benchmark.sh's gate)
    sys.exit(0 if all(rw["ok"] == rw["n"] for rw in rows) else 1)
