"""Open-loop task arrivals with sequential calls within each agent task."""
import asyncio
import hashlib
import json
import math
from pathlib import Path
import random
import time
import uuid
import httpx
from traffic import initial_prompt, continuation, tokenizer


async def stream_request(client, url, input_ids, output_tokens, timeout_s):
    started = time.perf_counter()
    first = last_token = None
    first_count = 0
    text = ""
    meta = {}
    done = False
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
                  cached_tokens=meta.get("cached_tokens"), text=text)
    return result


def percentile(values, fraction):
    values = sorted(x for x in values if x is not None)
    return None if not values else values[max(0, math.ceil(len(values) * fraction) - 1)]


def summarize(records, duration_s, gpus, ttft_ms, tpot_ms, attainment, offered_tasks_s, actual_calls_s):
    good = [r for r in records if r["ok"] and r.get("ttft_ms") is not None and r.get("tpot_ms") is not None and r["ttft_ms"] <= ttft_ms and r["tpot_ms"] <= tpot_ms]
    ok = [r for r in records if r["ok"]]
    attempts = [r for r in records if r.get("sent", False)]
    hit = len(good) / len(records) if records else 0.0
    return {"offered_tasks_per_s": offered_tasks_s, "actual_sent_calls_per_s": actual_calls_s,
            "measurement_including_drain_s": duration_s, "planned_calls": len(records), "sent_calls": len(attempts),
            "completed_calls": len(ok), "failed_or_skipped_calls": len(records) - len(ok),
            "p90_ttft_ms_success_only": percentile([r.get("ttft_ms") for r in ok], .9),
            "p90_tpot_ms_success_only": percentile([r.get("tpot_ms") for r in ok], .9),
            "slo_ttft_ms": ttft_ms, "slo_tpot_ms": tpot_ms, "slo_attainment_fraction": hit,
            "attainment_target_fraction": attainment, "meets_joint_slo": hit >= attainment,
            "successful_calls_per_s": len(ok) / duration_s, "useful_calls_per_s": len(good) / duration_s,
            "useful_calls_per_s_per_gpu": len(good) / duration_s / gpus, "gpu_count": gpus,
            "max_client_launch_lag_ms": max((r.get("client_launch_lag_ms", 0) for r in records), default=0),
            "prefix_overlap_fraction_min": min((r["prefix_overlap_fraction"] for r in records if r.get("prefix_overlap_fraction") is not None), default=None),
            "cached_tokens_sum_reported": sum(r.get("cached_tokens") or 0 for r in records),
            "input_tokens_sum_sent": sum(r.get("input_tokens", 0) for r in attempts)}


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
                row = {"task_id": identity, "turn_index": turn, "phase": phase, "sent": False, "ok": False, "client_launch_lag_ms": launch_lag, "prefix_overlap_fraction": overlap}
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
    summary["client_valid"] = summary["max_client_launch_lag_ms"] <= a.max_client_lag_ms
    summary["meets_joint_slo"] &= summary["client_valid"]
    return {"metadata": {"architecture": a.architecture, "instance_type": a.instance_type, "region": a.region, "evidence_scope": a.evidence_scope, "model": data["model"], "model_revision": data["revision"], "shape": data["shape"], "phase": phase, "seed": a.seed, "run_nonce": nonce, "arguments": vars(a)}, "summary": summary, "requests": records}


async def ramp(a):
    data = json.loads(Path(a.traffic).read_text())
    tok = tokenizer(data["model"], data["revision"])
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    for rate in a.rates:
        if not a.skip_warmup:
            warm = await run_rate(data, a, rate, "warmup", tok)
            (out / f"{rate:g}-warmup.json").write_text(json.dumps(warm, indent=2))
        measured = await run_rate(data, a, rate, "measured", tok)
        (out / f"{rate:g}-measured.json").write_text(json.dumps(measured, indent=2))
        print(json.dumps({"metadata": measured["metadata"], "summary": measured["summary"]}), flush=True)
        if measured["summary"]["failed_or_skipped_calls"] / max(1, measured["summary"]["planned_calls"]) > a.stop_failure_fraction:
            print("Ramp stopped at configured failure fraction; failures remain in attainment denominator.", flush=True)
            break
