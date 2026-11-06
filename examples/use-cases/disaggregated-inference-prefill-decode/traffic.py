"""Structured synthetic tasks, tokenizer counts, and sequential continuations."""
import argparse
import hashlib
import json
from pathlib import Path
import random
from transformers import AutoTokenizer

MODEL = "deepseek-ai/DeepSeek-V2-Lite-Chat"
REVISION = "85864749cd611b4353ce1decdb286193298f64c7"


def tokenizer(model=MODEL, revision=REVISION):
    return AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=False)


def tokens(tok, text):
    return tok.encode(text, add_special_tokens=False)


def fill(tok, text, count):
    ids = tokens(tok, text)
    assert ids and count >= 0
    return (ids * (count // len(ids) + 1))[:count]


def common_prefix(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def initial_prompt(tok, shape, task_id, run_nonce=""):
    if shape["name"].startswith("B"):
        # The nonce is before the document: per-rate warmup cannot cache the measured document.
        identity = hashlib.sha256(f"{run_nonce}:{task_id}".encode()).hexdigest()
        prefix = tokens(tok, f"{tok.bos_token}User: Record {identity}. ")
        suffix = tokens(tok, "\n\nAssistant:")
        count = shape["input_tokens"] - len(prefix) - len(suffix)
        assert count > 0
        return prefix + fill(tok, shape["document_text"], count) + suffix
    system = fill(tok, shape["system_text"], shape["system_tokens"])
    # Cacheability is a workload property; the server reports actual cached tokens separately.
    if random.Random(f"{shape['seed']}:{task_id}").random() >= shape["shared_system_fraction"]:
        unique = tokens(tok, f"Private task {run_nonce}:{task_id}. ")
        system = (unique + system)[:len(system)]
    prefix = tokens(tok, tok.bos_token) + system + tokens(tok, f"\n\nUser: Investigate order {task_id}. ")
    suffix = tokens(tok, "\n\nAssistant:")
    count = shape["first_input_tokens"] - len(prefix) - len(suffix)
    assert count > 0
    return prefix + fill(tok, shape["tool_observation"], count) + suffix


def continuation(tok, shape, previous, answer, task_id, turn):
    # Append the actual answer before the next simulated tool observation. Never send turns concurrently.
    response = tokens(tok, answer + tok.eos_token + f"User: Tool observation for order {task_id}, step {turn}: ")
    suffix = tokens(tok, "\n\nAssistant:")
    target = max(round(len(previous) / shape["prefix_overlap_target_fraction"]), len(previous) + len(response) + len(suffix) + 32)
    prompt = previous + response + fill(tok, shape["tool_observation"], target - len(previous) - len(response) - len(suffix)) + suffix
    overlap = common_prefix(previous, prompt) / len(prompt)
    if not shape["prefix_overlap_min_fraction"] <= overlap <= shape["prefix_overlap_max_fraction"]:
        raise ValueError(f"Agentic prefix overlap {overlap} outside configured fraction bounds")
    return prompt, overlap


def generate(default_shape):
    p = argparse.ArgumentParser(description="Generate task definitions and representative initial token IDs")
    p.add_argument("--shape", default=default_shape)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--revision", default=REVISION)
    a = p.parse_args()
    shape = json.loads(Path(a.shape).read_text())
    assert shape["tasks"] > 0
    if shape["name"].startswith("A"):
        assert 5 <= shape["calls_per_task_min"] <= shape["calls_per_task_max"] <= 15
        assert 0.5 <= shape["prefix_overlap_min_fraction"] <= shape["prefix_overlap_target_fraction"] <= shape["prefix_overlap_max_fraction"] <= 0.9
        assert 0 <= shape["shared_system_fraction"] <= 1
    rng = random.Random(shape["seed"])
    tok = tokenizer(a.model, a.revision)
    tasks = [{"task_id": f"task-{i}", "calls": rng.randint(shape["calls_per_task_min"], shape["calls_per_task_max"]), "initial_input_ids": initial_prompt(tok, shape, f"task-{i}", "generated")}
             for i in range(shape["tasks"])]
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": a.model, "revision": a.revision, "shape": shape, "tasks": tasks}))
    print(json.dumps({"file": str(out), "task_count": len(tasks), "planned_call_count": sum(x["calls"] for x in tasks), "initial_input_tokens_min": min(len(x["initial_input_ids"]) for x in tasks), "initial_input_tokens_max": max(len(x["initial_input_ids"]) for x in tasks)}))
