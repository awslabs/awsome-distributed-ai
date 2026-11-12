"""Open-loop task arrivals with sequential calls within each agent task."""
import asyncio
import hashlib
import json
from pathlib import Path
import random
import time
import uuid
import httpx
from scoring import client_check, summarize
from traffic import initial_prompt, continuation, tokenizer


async def stream_request(client, url, input_ids, output_tokens, timeout_s):
    started = time.perf_counter()
    first = last_token = None
    first_count = 0
    text = ""
    meta = {}
    done = False
    chunks = []
    result = {"ok": False, "input_tokens": len(input_ids), "output_tokens": 0}
    try:
        async with asyncio.timeout(timeout_s):
            async with client.stream("POST", url.rstrip("/") + "/generate", json={"input_ids": input_ids, "sampling_params": {"temperature": 0.0, "max_new_tokens": output_tokens, "ignore_eos": True}, "stream": True}) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        done = True
                        break
                    event = json.loads(payload)
                    if "error" in event:
                        raise RuntimeError(str(event["error"]))
                    meta = event.get("meta_info", {})
                    count = meta.get("completion_tokens", 0)
                    if count > result["output_tokens"]:
                        last_token = time.perf_counter()
                        chunks.append({"t_ms": (last_token - started) * 1000, "n": count - result["output_tokens"]})
                        if first is None:
                            first = last_token
                            first_count = count
                    result["output_tokens"] = count
                    text = event.get("text", text)
            if not done or first is None or result["output_tokens"] != output_tokens:
                raise RuntimeError("Incomplete stream or output length mismatch")
            finish_reason = meta.get("finish_reason") or {}
            if isinstance(finish_reason, dict) and finish_reason.get("type") in ("abort", "error"):
                raise RuntimeError(str(finish_reason))
            result["ok"] = True
    except (httpx.HTTPError, TimeoutError, RuntimeError, ValueError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    ended = time.perf_counter()
    # Stream chunks can contain multiple tokens. Record that limitation explicitly.
    token_intervals = result["output_tokens"] - first_count
    result.update(ttft_ms=None if first is None else (first - started) * 1000,
                  tpot_ms=None if token_intervals <= 0 else (last_token - first) * 1000 / token_intervals,
                  latency_ms=(ended - started) * 1000, first_chunk_tokens=first_count,
                  cached_tokens=meta.get("cached_tokens"), text=text,
                  chunks=chunks if result["ok"] else [])
    return result


async def require_reachable_router(url, settle_s=180, poll_s=5):
    """Refuse to measure an endpoint that is unreachable or cannot route yet.

    A `kubectl port-forward` dies silently under the load of the highest rates. Without this
    check the next rate records sent calls with no completions, `None` percentiles and
    `meets_joint_slo: false`, which is indistinguishable from an engine that actually failed,
    and `--stop-failure-fraction` then aborts the remaining rates. A client-side connection
    fault is not a measurement.

    `/health` reports the router process, not the pair: while a worker's circuit breaker is open
    the router answers it 200 and `/generate` 503, and `--disable-retries` turns one such 503
    into a failed task. So probe the path being measured, and wait rather than fail — after an
    engine pod restart the breaker closed within about 90 s in measurement.
    """
    base = url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
            status = (await client.get(base + "/health")).status_code
    except (httpx.HTTPError, TimeoutError) as exc:
        raise SystemExit(f"Router unreachable at {base}/health ({type(exc).__name__}: {exc}). "
                         "The port-forward has most likely dropped: restart it, confirm the pods are still Running, "
                         "and rerun this rate. No measurement file was written.")
    if status != 200:
        raise SystemExit(f"Router at {base}/health answered HTTP {status}, not 200. "
                         "Resolve the endpoint before measuring. No measurement file was written.")

    began = time.monotonic()
    deadline = began + settle_s
    body = {"input_ids": [0], "sampling_params": {"temperature": 0.0, "max_new_tokens": 1}, "stream": False}
    waited = False
    while True:
        try:
            async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
                response = await client.post(base + "/generate", json=body)
            if response.status_code == 200:
                if waited:
                    print(f"Router became routable after {time.monotonic() - began:.0f} s.", flush=True)
                return
            detail = f"HTTP {response.status_code}: {response.text.strip()[:200]}"
        except (httpx.HTTPError, TimeoutError) as exc:
            detail = f"{type(exc).__name__}: {exc}"
        if time.monotonic() >= deadline:
            raise SystemExit(f"Router at {base}/generate is still not routable after {settle_s} s ({detail}). "
                             "A worker circuit breaker is most likely still open after a pod restart: confirm both "
                             "engine pods are Running, then rerun this rate. No measurement file was written.")
        if not waited:
            print(f"Router answers /health but not /generate ({detail}). "
                  f"Waiting up to {settle_s} s for it to become routable.", flush=True)
            waited = True
        await asyncio.sleep(poll_s)


async def run_rate(data, a, rate, phase, tok):
    # No concurrency semaphore: slower responses must not reduce the offered task arrival rate.
    rng = random.Random(a.seed)
    arrivals = []
    when = 0.0
    while when < a.duration_s:
        arrivals.append(when)
        when += rng.expovariate(rate)
    if len(arrivals) > a.max_tasks:
        raise ValueError("Raise --max-tasks deliberately or reduce rate/duration; no arrivals were sent")
    nonce = f"{a.seed}:{phase}:{rate}:{uuid.uuid4().hex}"
    start = time.perf_counter()
    records = []
    async with httpx.AsyncClient(timeout=None, limits=httpx.Limits(max_connections=None, max_keepalive_connections=512), trust_env=False) as client:
        async def task(index, arrival):
            await asyncio.sleep(max(0, start + arrival - time.perf_counter()))
            launch_lag = max(0, time.perf_counter() - start - arrival) * 1000
            item = data["tasks"][index % len(data["tasks"])]
            identity = f"{item['task_id']}-{index}"
            prompt = initial_prompt(tok, data["shape"], identity, nonce)
            overlap = None
            failed = False
            for turn in range(item["calls"]):
                row = {"task_id": identity, "turn_index": turn, "phase": phase, "sent": False, "ok": False, "client_launch_lag_ms": launch_lag, "prefix_overlap_fraction": overlap, "chunks": []}
                if failed:
                    row["error"] = "Skipped after failed predecessor"
                elif len(prompt) + data["shape"]["output_tokens"] > a.context_length_tokens:
                    row["error"] = "Context budget exceeded"
                    failed = True
                else:
                    row["sent"] = True
                    row["input_ids_sha256"] = hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
                    response = await stream_request(client, a.url, prompt, data["shape"]["output_tokens"], a.timeout_s)
                    answer = response.pop("text")
                    row.update(response)
                    if turn == 0 and row.get("ttft_ms") is not None:
                        row["ttft_ms"] += launch_lag
                    failed = not row["ok"]
                    if not failed and turn + 1 < item["calls"]:
                        try:
                            prompt, overlap = continuation(tok, data["shape"], prompt, answer, identity, turn + 1)
                        except ValueError as exc:
                            failed = True
                            row["continuation_error"] = str(exc)
                        await asyncio.sleep(data["shape"].get("tool_think_time_s", 0))
                records.append(row)
        await asyncio.gather(*(task(i, t) for i, t in enumerate(arrivals)))
    elapsed = max(a.duration_s, time.perf_counter() - start)
    summary = summarize(records, elapsed, a.gpus, a.ttft_slo_ms, a.tpot_slo_ms, a.attainment_fraction, rate, sum(r["sent"] for r in records) / elapsed)
    summary["offered_window_s"] = a.duration_s
    summary["offered_task_count"] = len(arrivals)
    client_check(summary, a.max_client_lag_ms)
    return {"metadata": {"architecture": a.architecture, "instance_type": a.instance_type, "region": a.region, "evidence_scope": a.evidence_scope, "model": data["model"], "model_revision": data["revision"], "shape": data["shape"], "traffic_sha256": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(), "phase": phase, "seed": a.seed, "run_nonce": nonce, "arguments": vars(a)}, "summary": summary, "requests": records}


async def ramp(a):
    data = json.loads(Path(a.traffic).read_text())
    tok = tokenizer(data["model"], data["revision"])
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    for rate in a.rates:
        await require_reachable_router(a.url)
        if not a.skip_warmup:
            warm = await run_rate(data, a, rate, "warmup", tok)
            (out / f"{rate:g}-warmup.json").write_text(json.dumps(warm, indent=2))
        measured = await run_rate(data, a, rate, "measured", tok)
        (out / f"{rate:g}-measured.json").write_text(json.dumps(measured, indent=2))
        print(json.dumps({"metadata": measured["metadata"], "summary": measured["summary"]}), flush=True)
        if measured["summary"]["failed_or_skipped_calls"] / max(1, measured["summary"]["planned_calls"]) > a.stop_failure_fraction:
            print("Ramp stopped at configured failure fraction; failures remain in attainment denominator.", flush=True)
            break
