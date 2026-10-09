# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Prepare SWE-smith agent trajectories for SFT (runs on the CPU node).

Source: SWE-bench/SWE-smith-trajectories, split `tool` (native function calling; Claude 3.7
Sonnet driving SWE-agent). We keep resolved trajectories from the main agent, convert them to
Qwen3's chat format with tool schemas, and pre-tokenize with a loss mask that covers only the
assistant turns (the model's own reasoning text + tool calls), not the task text or tool output.

Writes (HF save_to_disk) under --out:
  train-<L>/  val/     columns: input_ids, labels, n_tokens, instance_id
  heldout_ids.json     instance ids used for val (never trained on)
  stats.json
"""
import argparse
import json
import os
import random
import re

from datasets import load_dataset
from transformers import AutoTokenizer

from agent_tools import SYSTEM, TOOLS  # noqa: E402  (shared with agent_eval.py)

ROOT = os.environ.get("DATA_ROOT", "/fsx")  # FSx for Lustre mount


def text(content):
    if isinstance(content, list):
        return "".join(c.get("text", "") for c in content if isinstance(c, dict))
    return content or ""


# About a third of the `tool` split's assistant messages carry the call as SWE-agent XML in the
# message text (`<function=bash><parameter=command>ls</parameter></function>`) instead of in
# `tool_calls`. Left as text, they teach the model to write XML that no tool-calling runtime
# (Ollama, llama.cpp, VS Code) executes. Convert them to real tool calls.
FN_RE = re.compile(r"<function=([\w\-.]+)>(.*?)</function>", re.S)
PARAM_RE = re.compile(r"<parameter=([\w\-.]+)>(.*?)</parameter>", re.S)


def _param_value(name, raw):
    v = raw[1:] if raw.startswith("\n") else raw
    v = v[:-1] if v.endswith("\n") else v
    if name == "insert_line":
        return int(v.strip())
    if name == "view_range":
        return json.loads(v.strip())
    return v


def xml_calls(content):
    """(content_without_calls, [call]) for SWE-agent XML calls embedded in text."""
    calls = []
    for m in FN_RE.finditer(content):
        args = {p.group(1): _param_value(p.group(1), p.group(2))
                for p in PARAM_RE.finditer(m.group(2))}
        calls.append({"type": "function", "function": {"name": m.group(1), "arguments": args}})
    if not calls:
        return content, []
    return content[:content.find("<function=")].strip(), calls


def convert(row):
    """SWE-smith row -> Qwen3 messages, or None if unusable."""
    msgs = row["messages"]
    if isinstance(msgs, str):
        msgs = json.loads(msgs)
    out = [{"role": "system", "content": SYSTEM}]
    expecting_obs = False
    for m in msgs:
        if m.get("agent", "main") != "main":
            continue
        role = m["role"]
        if role == "system":
            continue
        if role == "user" and not expecting_obs:
            out.append({"role": "user", "content": text(m["content"])})
        elif role == "assistant":
            calls = []
            for tc in m.get("tool_calls") or []:
                f = tc.get("function", {})
                args = f.get("arguments") or "{}"
                try:
                    args = json.loads(args) if isinstance(args, str) else args
                except json.JSONDecodeError:
                    return None
                calls.append({"type": "function",
                              "function": {"name": f.get("name"), "arguments": args}})
            content = text(m["content"]).strip()
            if not calls:
                try:
                    content, calls = xml_calls(content)
                except (ValueError, json.JSONDecodeError):
                    return None
            if "<function=" in content or "<parameter=" in content:
                return None
            msg = {"role": "assistant", "content": content}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
            expecting_obs = bool(calls)
        else:  # tool result (role "tool", or "user" carrying the observation of an XML call)
            obs = text(m["content"])
            if obs.startswith("OBSERVATION:"):
                obs = obs[len("OBSERVATION:"):].lstrip("\n")
            out.append({"role": "tool", "content": obs})
            expecting_obs = False
    if not any(m["role"] == "assistant" for m in out):
        return None
    return out


EMPTY_THINK = "<think>\n\n</think>\n\n"


def tokenize(messages, tok, max_len):
    """input_ids + labels; labels = -100 everywhere except assistant turns (incl. <|im_end|>).

    Train/inference alignment: in non-thinking mode the generation prompt for EVERY turn is
    `<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`, but the Qwen3 template renders that empty
    think block only on the LAST assistant turn of a training conversation. Without fixing this,
    the model learns "after an empty think block comes the final turn" and, at inference, every
    turn looks like the last one (no reasoning text, early submit). So every assistant turn gets
    the empty think block, as an unsupervised prefix, exactly as inference presents it.
    """
    rendered = tok.apply_chat_template(messages, tools=TOOLS, tokenize=False)
    head, end = "<|im_start|>assistant\n", "<|im_end|>"
    rendered = rendered.replace(head + "<think>\n\n</think>\n\n", head)
    rendered = rendered.replace(head, head + EMPTY_THINK)
    head = head + EMPTY_THINK
    spans, start = [], 0
    while True:
        i = rendered.find(head, start)
        if i < 0:
            break
        j = rendered.find(end, i)
        if j < 0:
            break
        spans.append((i + len(head), j + len(end)))
        start = j
    enc = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    if len(ids) > max_len:
        return None
    labels, k = [], 0
    for tid, (a, _) in zip(ids, enc["offset_mapping"]):
        while k < len(spans) and a >= spans[k][1]:
            k += 1
        inside = k < len(spans) and spans[k][0] <= a < spans[k][1]
        labels.append(tid if inside else -100)
    return ids, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/data/agent")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--max_lens", default="8192,16384")
    ap.add_argument("--val", type=int, default=100)
    ap.add_argument("--num_proc", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="rows to read (0 = all)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    max_lens = [int(x) for x in a.max_lens.split(",")]
    top = max(max_lens)

    tok = AutoTokenizer.from_pretrained(a.model)
    probe = tok.apply_chat_template([{"role": "user", "content": "hi"}], tools=TOOLS,
                                    tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
    assert probe.endswith("<|im_start|>assistant\n" + EMPTY_THINK), repr(probe[-80:])
    ds = load_dataset("SWE-bench/SWE-smith-trajectories", split="tool")
    n_all = len(ds)
    ds = ds.filter(lambda r: bool(r["resolved"]), num_proc=a.num_proc)
    if a.limit:
        ds = ds.select(range(min(a.limit, len(ds))))
    n_resolved = len(ds)

    def proc(row):
        msgs = convert(row)
        res = tokenize(msgs, tok, top) if msgs else None
        if res is None:
            return {"input_ids": [], "labels": [], "n_tokens": 0}
        ids, labels = res
        return {"input_ids": ids, "labels": labels, "n_tokens": len(ids)}

    ds = ds.map(proc, num_proc=a.num_proc, remove_columns=[c for c in ds.column_names
                                                           if c != "instance_id"])
    ds = ds.filter(lambda r: r["n_tokens"] > 0, num_proc=a.num_proc)

    # Hold out whole instances (an instance can have several trajectories).
    ids = sorted(set(ds["instance_id"]))
    random.Random(42).shuffle(ids)
    held = set(ids[:a.val])
    val = ds.filter(lambda r: r["instance_id"] in held and r["n_tokens"] <= min(max_lens))
    train = ds.filter(lambda r: r["instance_id"] not in held)
    val.save_to_disk(f"{a.out}/val")
    json.dump(sorted(held), open(f"{a.out}/heldout_ids.json", "w"))

    stats = {"rows_tool_split": n_all, "resolved": n_resolved, "tokenized_le_max": len(ds),
             "val": len(val), "heldout_instances": len(held)}
    for L in max_lens:
        t = train.filter(lambda r: r["n_tokens"] <= L).shuffle(seed=42)
        t.save_to_disk(f"{a.out}/train-{L // 1024}k")
        sup = sum(sum(x != -100 for x in lab) for lab in t.select(range(min(500, len(t))))["labels"])
        tot = sum(t.select(range(min(500, len(t))))["n_tokens"])
        stats[f"train_{L // 1024}k"] = len(t)
        stats[f"train_{L // 1024}k_tokens"] = int(sum(t["n_tokens"]))
        stats[f"train_{L // 1024}k_supervised_frac"] = round(sup / max(tot, 1), 3)
        print(f"wrote {a.out}/train-{L // 1024}k: {len(t)} trajectories", flush=True)
    json.dump(stats, open(f"{a.out}/stats.json", "w"), indent=2)
    print("AGENT_STATS " + json.dumps(stats), flush=True)

    # Show one rendered example (head + first assistant turn) to eyeball the format.
    sample = val[0] if len(val) else train[0]
    txt = tok.decode(sample["input_ids"])
    i = txt.find("<|im_start|>assistant")
    print("SAMPLE_RENDER_HEAD:\n" + txt[:1500])
    print("SAMPLE_FIRST_ASSISTANT:\n" + txt[i:i + 800])
    sup = tok.decode([t for t in sample["labels"] if t != -100])
    print("SAMPLE_SUPERVISED_START:\n" + sup[:600])


if __name__ == "__main__":
    main()