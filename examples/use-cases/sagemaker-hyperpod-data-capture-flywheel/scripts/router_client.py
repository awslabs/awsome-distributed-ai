import json, sys, time, argparse, statistics, urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

V1 = ["ProductFridge", "ProductWasher", "ProductOven", "ProductDishwasher", "ProductMicrowave", "Other"]
V2 = V1[:5] + ["ProductAC", "Other"]

def system_prompt(cats, hint=""):
    return ("You are a support-ticket router for a home-appliance retailer. Route each ticket to the product it "
            'concerns. Respond with ONLY a JSON object {"category": "...", "confidence": <0-1>} where category is '
            "one of: " + ", ".join(cats) + '. Use "Other" if the ticket does not clearly concern one of these products.'
            + (" " + hint if hint else ""))

def route(url, model, cats, ticket, hint=""):
    body = {"model": model, "temperature": 0, "max_tokens": 40,
            "messages": [{"role": "system", "content": system_prompt(cats, hint)}, {"role": "user", "content": ticket}],
            # guided decoding: the model can only emit a valid object with a known category
            "guided_json": {"type": "object", "required": ["category", "confidence"],
                            "properties": {"category": {"enum": cats}, "confidence": {"type": "number"}}}}
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    out = json.load(urllib.request.urlopen(req, timeout=60))
    ms = (time.perf_counter() - t0) * 1000
    try: cat = json.loads(out["choices"][0]["message"]["content"])["category"]
    except Exception: cat = "PARSE_ERROR"
    return cat, ms

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["drive", "eval", "bench"])
    ap.add_argument("file"); ap.add_argument("--port", default="8081"); ap.add_argument("--model", required=True)
    ap.add_argument("--labels", choices=["v1", "v2"], default="v1"); ap.add_argument("--seconds", type=float, default=300)
    ap.add_argument("--concurrency", type=int, default=8); ap.add_argument("--launch-at", type=int, default=400)
    ap.add_argument("--hint", default="", help="extra sentence appended to the system prompt")
    a = ap.parse_args()
    url, cats = f"http://localhost:{a.port}/v1/chat/completions", V1 if a.labels == "v1" else V2
    rows = [json.loads(l) for l in open(a.file)]

    if a.mode == "drive":        # replay tickets in order, paced across --seconds, like production arrivals
        print("RUN_START=" + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), flush=True)
        t0 = time.time()
        for i, r in enumerate(rows):
            time.sleep(max(0, t0 + i * a.seconds / len(rows) - time.time()))
            if i == a.launch_at: print("LAUNCH_TS=" + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), flush=True)
            route(url, a.model, cats, r["ticket"])
            if i % 100 == 99: print(f"  sent {i + 1}/{len(rows)}", flush=True)
        print("RUN_END=" + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time() + 1)))

    elif a.mode == "eval":       # labeled eval: per-class accuracy + confusion
        with ThreadPoolExecutor(a.concurrency) as ex:
            preds = list(ex.map(lambda r: route(url, a.model, cats, r["ticket"], a.hint)[0], rows))
        conf, ok = defaultdict(Counter), Counter()
        for r, p in zip(rows, preds):
            conf[r["label"]][p] += 1; ok[r["label"]] += (p == r["label"])
            if p != r["label"] and len(rows) < 100: print(f"  MISS {r['label']} -> {p}: {r['ticket']}")
        total = sum(ok.values())
        print(f"overall {total}/{len(rows)} = {100 * total / len(rows):.2f}%")
        for lab in V2:
            n = sum(conf[lab].values())
            wrong = {k: v for k, v in conf[lab].items() if k != lab}
            print(f"  {lab:18} {ok[lab]}/{n}" + (f"   misrouted: {wrong}" if wrong else ""))

    else:                        # latency benchmark at fixed concurrency
        tickets = [r["ticket"] for r in rows][:400]
        with ThreadPoolExecutor(a.concurrency) as ex:
            list(ex.map(lambda t: route(url, a.model, cats, t), tickets[:16]))  # warm-up
            t0 = time.perf_counter()
            ms = sorted(m for _, m in ex.map(lambda t: route(url, a.model, cats, t), tickets))
            wall = time.perf_counter() - t0
        q = statistics.quantiles(ms, n=100)
        print(f"port {a.port}: n={len(ms)} p50={q[49]:.1f}ms p90={q[89]:.1f}ms p99={q[98]:.1f}ms "
              f"throughput={len(ms) / wall:.1f} req/s")
