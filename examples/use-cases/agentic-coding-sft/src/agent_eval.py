# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Agentic bug-fix evaluation: base vs fine-tuned, multi-turn with real tool execution.

For every task (make_agent_tasks.py) the model gets a VS Code-agent-style request ("there's a
bug in `f`, fix it"), the tools `bash`, `str_replace_editor`, `submit`, and up to --max_turns
assistant turns. Each turn: the whole conversation is rendered with Qwen3's chat template (the
same template + tool schemas used to build the training data), vLLM generates one assistant
turn, its tool calls run in the task's sandbox, results go back as tool messages.

An episode ends on `submit`, on a turn with no tool call (the agent "answers", as VS Code agents
do), at --max_turns, or when the context is full. Graded by running the ORIGINAL unit tests on
the final repo: resolved = target test passes and no other test broke (submit not required).

Runs in the vllm/vllm-openai image. A LoRA adapter is served via vLLM's LoRA support.
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoTokenizer

from agent_env import TaskEnv, parse_assistant
from agent_tools import SYSTEM, TOOLS

ROOT = os.environ.get("DATA_ROOT", "/fsx")  # FSx for Lustre mount

USER = """I'm working in the Python repository at /testbed. There's a bug in the function \
`{fn}`: it doesn't behave the way its docstring describes, and its unit test fails. Please \
find the bug and fix it. The tests are in /testbed/tests and run with `python -m pytest`. \
Don't modify the tests. Call `submit` when you're done."""

PARSE_ERROR = ("Your last message contained a tool call that could not be parsed. Tool calls "
               "must be a JSON object with \"name\" and \"arguments\" inside <tool_call></tool_call>"
               " tags.")


def server_generator(a):
    """Generation through llama.cpp's llama-server /completion endpoint (raw prompt, so the
    prompt is byte-identical to the vLLM path and only the weights/quantization differ)."""
    import urllib.request

    def one(prompt, seed):
        body = json.dumps({"prompt": prompt, "n_predict": a.max_tokens,
                           "temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k,
                           "min_p": 0.0, "seed": seed, "cache_prompt": True,
                           "return_tokens": False}).encode()
        for attempt in range(5):
            try:
                req = urllib.request.Request(f"{a.server_url}/completion", data=body,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=1800) as r:
                    out = json.loads(r.read())
                return out["content"], int(out.get("tokens_predicted", 0))
            except Exception as e:  # server busy / transient
                err = e
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"llama-server request failed: {err}")

    def generate(prompts, seeds):
        with ThreadPoolExecutor(a.concurrency) as ex:
            return list(ex.map(one, prompts, seeds))

    return generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--label", default="base-1p7b")
    ap.add_argument("--tasks", default=f"{ROOT}/agent_tasks/v1")
    ap.add_argument("--out", default=f"{ROOT}/agent_eval")
    ap.add_argument("--work", default="/tmp/agent_work")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_turns", type=int, default=30)
    ap.add_argument("--max_tokens", type=int, default=2048, help="per assistant turn")
    ap.add_argument("--max_model_len", type=int, default=32768)
    ap.add_argument("--thinking", action="store_true", help="enable Qwen3 thinking mode")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--uid", type=int, default=65534)
    ap.add_argument("--gpu_mem", type=float, default=0.88)
    ap.add_argument("--server_url", default="",
                    help="llama.cpp llama-server URL (GGUF). If set, vLLM is not used; --model "
                         "is then only the tokenizer/chat template source.")
    ap.add_argument("--concurrency", type=int, default=8, help="parallel requests (server)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tasks = [json.loads(l) for l in open(f"{a.tasks}/tasks.jsonl")]
    if a.limit:
        tasks = tasks[:a.limit]
    tok = AutoTokenizer.from_pretrained(a.model)

    if a.server_url:
        generate = server_generator(a)
    else:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_model_len,
                  gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=True, seed=a.seed,
                  enable_lora=bool(a.adapter), max_lora_rank=64 if a.adapter else 16)
        lora = LoRARequest("tuned", 1, a.adapter) if a.adapter else None

        def generate(prompts, seeds):
            sps = [SamplingParams(temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                                  max_tokens=a.max_tokens, seed=sd, skip_special_tokens=False)
                   for sd in seeds]
            outs = llm.generate(prompts, sps, lora_request=lora, use_tqdm=False)
            return [(o.outputs[0].text, len(o.outputs[0].token_ids)) for o in outs]
    print(f"label={a.label} model={a.model} adapter={a.adapter or '-'} tasks={len(tasks)} "
          f"thinking={a.thinking}", flush=True)

    envs = [TaskEnv(t, a.tasks, a.work, a.uid) for t in tasks]
    check = envs[0].bash("python --version; python -m pytest --version; git status | head -1; "
                         "python -m pytest -q -p no:cacheprovider tests 2>&1 | tail -2")
    print("ENV_CHECK:\n" + check, flush=True)
    if "command not found" in check or "No module named pytest" in check:
        raise SystemExit("environment check failed")
    convs = [[{"role": "system", "content": SYSTEM},
              {"role": "user", "content": USER.format(fn=t["entry_point"])}] for t in tasks]
    ep = [{"turns": 0, "ended": None, "gen_tokens": 0, "parse_errors": 0, "last_prompt_tokens": 0,
           "raw": []}
          for _ in tasks]
    active = list(range(len(tasks)))
    t0 = time.time()

    def render(i):
        return tok.apply_chat_template(convs[i], tools=TOOLS, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=a.thinking)

    def step(i, text):
        content, calls, bad = parse_assistant(text)
        env, e = envs[i], ep[i]
        e["turns"] += 1
        e["parse_errors"] += bad
        e["raw"].append(text)
        msg = {"role": "assistant", "content": content}
        if calls:
            msg["tool_calls"] = [{"type": "function", "function": {"name": n, "arguments": g}}
                                 for n, g in calls]
        convs[i].append(msg)
        if not calls:
            if bad:
                convs[i].append({"role": "user", "content": PARSE_ERROR})
                return True
            e["ended"] = "no_tool_call"
            return False
        done = False
        for name, args in calls:
            obs, d = env.call(name, args)
            convs[i].append({"role": "tool", "content": obs})
            done = done or d
        if done:
            e["ended"] = "submit"
            return False
        return True

    for turn in range(a.max_turns):
        if not active:
            break
        prompts, idx = [], []
        for i in active:
            p = render(i)
            n = len(tok(p, add_special_tokens=False)["input_ids"])
            ep[i]["last_prompt_tokens"] = n
            if n + a.max_tokens > a.max_model_len:
                ep[i]["ended"] = "context_full"
                continue
            prompts.append(p)
            idx.append(i)
        if not prompts:
            active = []
            break
        outs = generate(prompts, [a.seed * 100003 + i * 101 + turn for i in idx])
        texts = {}
        for i, (text, ntok) in zip(idx, outs):
            texts[i] = text
            ep[i]["gen_tokens"] += ntok
        with ThreadPoolExecutor(16) as ex:
            keep = list(ex.map(lambda i: step(i, texts[i]), idx))
        active = [i for i, k in zip(idx, keep) if k]
        if turn < 2 and tasks:
            print(f"RENDER_TASK0_TURN{turn + 1} (tail):\n" + render(0)[-1500:], flush=True)
        print(f"[{a.label}] turn {turn + 1}: generated {len(idx)}, still active {len(active)} "
              f"({time.time() - t0:.0f}s)", flush=True)
    for i in active:
        ep[i]["ended"] = "max_turns"

    with ThreadPoolExecutor(16) as ex:
        grades = list(ex.map(lambda env: env.grade(), envs))

    rows = []
    for t, env, e, g, c in zip(tasks, envs, ep, grades, convs):
        rows.append({"id": t["id"], "entry_point": t["entry_point"], "mutation": t["mutation"],
                     **g, **e, **env.stats,
                     "transcript": [{**m, "content": env.to_virtual(m["content"] or "")}
                                    for m in c]})
    n = len(rows)

    def frac(k):
        return round(sum(bool(r[k]) for r in rows) / n, 3)

    calls = sum(r["bash"] + r["view"] + r["create"] + r["str_replace"] + r["insert"]
                + r["undo_edit"] + r["submit"] for r in rows)
    bad = sum(r["parse_errors"] + r["invalid_calls"] for r in rows)
    edits = sum(r["str_replace"] + r["insert"] + r["create"] for r in rows)
    ended = {}
    for r in rows:
        ended[r["ended"]] = ended.get(r["ended"], 0) + 1
    summary = {
        "label": a.label, "model": a.model, "adapter": a.adapter, "n": n,
        "backend": a.server_url or "vllm-bf16",
        "thinking": a.thinking, "max_turns": a.max_turns,
        "resolved": frac("resolved"), "target_fixed": frac("target_fixed"),
        "broke_other_tests": round(sum(not r["others_intact"] for r in rows) / n, 3),
        "touched_target_file": frac("touched_target"),
        "submitted": round(sum(r["ended"] == "submit" for r in rows) / n, 3),
        "ended": ended,
        "tool_calls_per_task": round(calls / n, 1),
        "invalid_call_rate": round(bad / max(calls + bad, 1), 3),
        "edit_attempts_per_task": round(edits / n, 2),
        "edit_error_rate": round(sum(r["edit_errors"] for r in rows) / max(edits, 1), 3),
        "ran_tests_frac": round(sum(r["pytest_runs"] > 0 for r in rows) / n, 3),
        "avg_turns": round(sum(r["turns"] for r in rows) / n, 1),
        "avg_gen_tokens": round(sum(r["gen_tokens"] for r in rows) / n),
        "wall_s": round(time.time() - t0),
    }
    with open(f"{a.out}/{a.label}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    json.dump(summary, open(f"{a.out}/{a.label}.summary.json", "w"), indent=2)
    print("AGENT_EVAL " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()