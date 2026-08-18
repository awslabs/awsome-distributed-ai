#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# =============================================================================
# Validation-Split Evaluation
# Scores a held-out parquet file (e.g. mixed-code-math/val.parquet) by:
#   1. Generating responses via vLLM's OpenAI-compatible API
#   2. Scoring them with verl's default_compute_score (via custom_reward_fn),
#      routing code tasks to Sandbox Fusion and math tasks to prime_math.
#
# This is the same reward signal the training loop optimizes, so it directly
# measures whether the checkpoint actually got better at the training
# objective on held-out data.
#
# Submitted as a Ray job by scripts/submit_val_eval.sh.  Can also run standalone
# against a local vLLM endpoint:
#
#   python scripts/eval_val_split.py \
#       --val-parquet /fsx/data/verl/data/mixed-code-math/val.parquet \
#       --vllm-url http://vllm-eval.default.svc.cluster.local:8000/v1 \
#       --served-name qwen3-235b-step200 \
#       --sandbox-url http://sandbox-fusion.default.svc.cluster.local:8080/run_code \
#       --output /fsx/data/verl/eval_results/qwen3-235b-step200/val_split.json \
#       --n-samples 4 \
#       --concurrency 32
# =============================================================================

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Optional progress bar — fall back to plain prints if tqdm not installed.
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(it, **kwargs):
        return it


def _parse_args():
    p = argparse.ArgumentParser(
        description="Score a validation parquet against vLLM + reward fn"
    )
    p.add_argument("--val-parquet", required=True, help="Path to val.parquet")
    p.add_argument(
        "--vllm-url", required=True, help="vLLM OpenAI API base URL (ending in /v1)"
    )
    p.add_argument(
        "--served-name",
        required=True,
        help="--served-model-name set on the vLLM server",
    )
    p.add_argument(
        "--sandbox-url",
        default="",
        help="Sandbox Fusion run_code URL (for code rewards)",
    )
    p.add_argument("--output", required=True, help="Output JSON path")
    p.add_argument(
        "--n-samples",
        type=int,
        default=4,
        help="Responses per prompt (match training n_responses_per_prompt)",
    )
    p.add_argument(
        "--concurrency", type=int, default=32, help="Max concurrent HTTP requests"
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=24576,
        help="Generation max_tokens. Must match training's max_response_length "
             "(conf/config.yaml) or scores are not comparable: a truncated response has "
             "no \\boxed{} to extract and does not compile, so it scores zero. At 16384 "
             "~22%% of every batch was a forced zero and corr(response_length, score) "
             "was -0.457; below that the censoring is worse.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (match training rollout)",
    )
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional: limit number of val samples (0 = all)",
    )
    p.add_argument(
        "--stratify",
        action="store_true",
        help="With --limit, sample evenly across data_source groups (round-robin) "
        "instead of taking the first N rows",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for --stratify sampling",
    )
    p.add_argument(
        "--memory-limit-mb",
        type=int,
        default=1024,
        help="Sandbox Fusion per-exec memory limit",
    )
    return p.parse_args()


def _load_val_parquet(path: str, limit: int = 0, stratify: bool = False, seed: int = 42):
    """Load val parquet and return a list of (idx, data_source, prompt, ground_truth, extra_info) tuples.

    When stratify=True and limit>0, sample `limit` rows spread as evenly as
    possible across data_source groups (round-robin over shuffled groups) instead
    of taking the first N — avoids the apps-heavy head of the parquet dominating
    a small subset, giving a fairer per-source signal.
    """
    import pandas as pd

    df = pd.read_parquet(path)
    required = {"prompt", "data_source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"val parquet missing columns: {missing}. Got: {list(df.columns)}"
        )

    # verl parquet prompts are typically a list of {role, content} dicts, but
    # pandas/pyarrow often materializes them as a numpy ndarray (object dtype),
    # so normalize ndarray/tuple -> list before the message-shape check.
    def _coerce_prompt(p):
        if hasattr(p, "tolist") and not isinstance(p, (str, bytes)):
            p = p.tolist()  # numpy ndarray -> list of dicts
        elif isinstance(p, tuple):
            p = list(p)
        if isinstance(p, list) and p and isinstance(p[0], dict) and "content" in p[0]:
            return list(p)  # list of messages — send as /v1/chat/completions
        return str(p)

    def _extract_ground_truth(row):
        rm = row.get("reward_model")
        if isinstance(rm, dict) and "ground_truth" in rm:
            return rm["ground_truth"]
        return row.get("ground_truth", "")

    # Stratified subset: round-robin across data_source groups so a small `limit`
    # covers all sources rather than the apps-heavy head of the file.
    if stratify and limit and limit < len(df):
        groups = {}
        for ds, g in df.groupby("data_source"):
            groups[ds] = g.sample(frac=1.0, random_state=seed).index.tolist()
        order = sorted(groups)  # deterministic group order
        picked = []
        cyclers = {ds: iter(groups[ds]) for ds in order}
        exhausted = set()
        while len(picked) < limit and len(exhausted) < len(order):
            for ds in order:
                if ds in exhausted:
                    continue
                nxt = next(cyclers[ds], None)
                if nxt is None:
                    exhausted.add(ds)
                    continue
                picked.append(nxt)
                if len(picked) >= limit:
                    break
        df = df.loc[picked]
        limit = 0  # already trimmed

    rows = []
    for i, row in df.iterrows():
        rows.append(
            (
                int(i),
                row.get("data_source", "unknown"),
                _coerce_prompt(row["prompt"]),
                _extract_ground_truth(row),
                row.get("extra_info", None),
            )
        )
        if limit and len(rows) >= limit:
            break
    return rows


async def _vllm_generate(
    session,
    vllm_url: str,
    served_name: str,
    prompt,
    n: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
):
    """Return a list of n response strings."""
    if isinstance(prompt, list):
        # Coerce to plain {role, content} str dicts (parquet may carry numpy types).
        messages = [
            {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
            for m in prompt
        ]
        endpoint = vllm_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": served_name,
            "messages": messages,
            "n": n,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        async with session.post(endpoint, json=payload, timeout=1800) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return [c["message"]["content"] for c in data["choices"]]
    else:
        endpoint = vllm_url.rstrip("/") + "/completions"
        payload = {
            "model": served_name,
            "prompt": prompt,
            "n": n,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        async with session.post(endpoint, json=payload, timeout=1800) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return [c["text"] for c in data["choices"]]


def _score_response(
    compute_score_fn,
    data_source,
    response_text,
    ground_truth,
    extra_info,
    sandbox_url: str,
    memory_limit_mb: int,
):
    """Call custom_reward_fn.compute_score and extract a numeric score."""
    try:
        result = compute_score_fn(
            data_source=data_source,
            solution_str=response_text,
            ground_truth=ground_truth,
            extra_info=extra_info,
            sandbox_fusion_url=sandbox_url or None,
            memory_limit_mb=memory_limit_mb,
        )
    except Exception as e:
        return 0.0, {"error": str(e)[:200]}
    if isinstance(result, dict):
        return float(result.get("score", 0.0)), {
            k: v for k, v in result.items() if k != "score"
        }
    return float(result), {}


async def _worker(sem, session, args, compute_score_fn, row, results):
    async with sem:
        idx, data_source, prompt, ground_truth, extra_info = row
        try:
            responses = await _vllm_generate(
                session,
                args.vllm_url,
                args.served_name,
                prompt,
                args.n_samples,
                args.temperature,
                args.top_p,
                args.max_tokens,
            )
        except Exception as e:
            msg = str(e) or repr(e) or type(e).__name__
            results.append(
                {
                    "idx": idx,
                    "data_source": data_source,
                    "error": f"generation: {msg[:300]}",
                    "scores": [0.0] * args.n_samples,
                }
            )
            return

        scores = []
        extras = []
        for resp in responses:
            score, extra = _score_response(
                compute_score_fn,
                data_source,
                resp,
                ground_truth,
                extra_info,
                args.sandbox_url,
                args.memory_limit_mb,
            )
            scores.append(score)
            extras.append(extra)

        results.append(
            {
                "idx": idx,
                "data_source": data_source,
                "scores": scores,
                "mean_score": sum(scores) / len(scores) if scores else 0.0,
                "pass_any": float(any(s >= 0.5 for s in scores)),
                "extras": extras,
            }
        )


async def main_async(args):
    import aiohttp

    # Import the repo's custom_reward_fn — it wraps verl's default_compute_score
    # with the same Eurus-fallback + sandbox-injection logic used during training.
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    from custom_reward_fn import compute_score  # noqa: E402

    rows = _load_val_parquet(args.val_parquet, args.limit, args.stratify, args.seed)
    print(f"Loaded {len(rows)} samples from {args.val_parquet}", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    results: list = []
    conn = aiohttp.TCPConnector(limit=args.concurrency * 2)
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = [
            asyncio.create_task(
                _worker(sem, session, args, compute_score, row, results)
            )
            for row in rows
        ]
        for _ in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="eval"):
            await _
    return results


def _aggregate(results, n_samples):
    """Summarize results per data_source and overall."""
    from collections import defaultdict

    buckets = defaultdict(list)
    for r in results:
        buckets[r["data_source"]].append(r)
    buckets["__all__"] = results

    summary = {}
    for ds, rows in buckets.items():
        if not rows:
            continue
        means = [r.get("mean_score", 0.0) for r in rows]
        pass_any = [r.get("pass_any", 0.0) for r in rows]
        errors = sum(1 for r in rows if "error" in r)
        summary[ds] = {
            "count": len(rows),
            "errors": errors,
            "mean_score": sum(means) / len(means) if means else 0.0,
            f"pass@{n_samples}": sum(pass_any) / len(pass_any) if pass_any else 0.0,
        }
    return summary


def main():
    args = _parse_args()
    t0 = time.time()

    results = asyncio.run(main_async(args))

    summary = _aggregate(results, args.n_samples)
    elapsed = time.time() - t0

    output = {
        "val_parquet": args.val_parquet,
        "served_name": args.served_name,
        "vllm_url": args.vllm_url,
        "sandbox_url": args.sandbox_url,
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "num_samples": len(results),
        "elapsed_seconds": elapsed,
        "summary": summary,
        "samples": results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(output, f, indent=2, default=str)

    # Echo summary to stdout
    print("\n" + "=" * 60, flush=True)
    print(f"Val-split evaluation complete ({elapsed:.1f}s)", flush=True)
    print(f"Output: {out_path}", flush=True)
    print("=" * 60, flush=True)
    for ds, s in sorted(summary.items()):
        print(
            f"  {ds:30s}  n={s['count']:>6d}  mean={s['mean_score']:.4f}  "
            f"pass@{args.n_samples}={s[f'pass@{args.n_samples}']:.4f}  "
            f"errors={s['errors']}",
            flush=True,
        )
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
