# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Side-by-side comparison: base Qwen3-1.7B vs YOUR fine-tune vs the full fine-tune,
on agentic bug-fixing.

Every model gets the same small repository with one bug, the same tools (bash,
str_replace_editor, submit) and the same request. Their tool calls really run in a sandbox, and
the verdict comes from running the repo's unit tests. You see, side by side, what each model did
turn by turn, whether the tests pass, and the change it made.

Modes (interactive menu by default):
  1  Showcase scenarios   a few curated tasks (a mix of wins, a tie and a loss for fine-tuning)
  2  Your own bug         paste a small buggy function + assert tests
  3  Inject a bug         pick a HumanEval function and a bug type; we break it, models fix it
  4  Quick scoreboard     N tasks from the eval set, % fixed per model

Runs on a GPU node in the vLLM image (see kubernetes/compare.yaml-template). One vLLM engine
serves the base model and both LoRA adapters at once.
"""
import warnings

warnings.filterwarnings("ignore")  # before torch/transformers imports (pynvml FutureWarning)

import argparse  # noqa: E402
import ast  # noqa: E402
import difflib
import hashlib
import json
import logging
import os
import random
import re
import shutil
import sys
import time

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from transformers import AutoTokenizer

from agent_env import TaskEnv, parse_assistant
from agent_eval import PARSE_ERROR, USER
from agent_tools import SYSTEM, TOOLS
from make_agent_tasks import mutate, run_check, sites, target_fn, test_file, write_repo

ROOT = os.environ.get("DATA_ROOT", "/fsx")  # FSx for Lustre mount

TASKS_ROOT = os.environ.get("AGENT_TASKS", f"{ROOT}/agent_tasks/v1")  # set by --tasks too
# Picked from the recorded 150-task results to show a mix, not just wins (full fine-tune vs base):
#   HumanEval_7: full fixes, workshop and base don't.  HumanEval_4/5: both fine-tunes fix, base doesn't.
#   HumanEval_46: all three fix.  HumanEval_57: base fixes, full doesn't.
#   Live runs are sampled (temperature 0.7), so outcomes can differ on a rerun.
SHOWCASE = ["HumanEval_7", "HumanEval_4", "HumanEval_5", "HumanEval_46", "HumanEval_57"]
# Quick scoreboard set: 20 eval tasks chosen (random search over 20-task samples) so their
# RECORDED 150-task-eval results match the overall rates: base 10% (8.7), workshop fine-tune
# 35% (33.3), full fine-tune 55% (53.3), base 4B 50% (50.7). Representative, not cherry-picked.
QUICK_SET = ["HumanEval_4", "HumanEval_11", "HumanEval_15", "HumanEval_20", "HumanEval_31",
             "HumanEval_44", "HumanEval_53", "HumanEval_56", "HumanEval_68", "HumanEval_69",
             "HumanEval_74", "HumanEval_79", "HumanEval_87", "HumanEval_90", "HumanEval_112",
             "HumanEval_123", "HumanEval_132", "HumanEval_146", "HumanEval_150", "HumanEval_163"]
# 2-way showcase (base vs the workshop-size fine-tune), picked from the recorded 150-task results
# of the workshop-size adapter as a mix: HumanEval_4/8 fine-tune wins, HumanEval_46 both fix,
# HumanEval_41 base wins. Used when no full adapter is loaded.
SHOWCASE_2WAY = ["HumanEval_4", "HumanEval_8", "HumanEval_46", "HumanEval_41"]
BUG_KINDS = ["comparison", "arithmetic", "boolean", "off_by_one"]
COLORS = {"base": "red", "yours": "cyan", "full": "green"}
TITLES = {"base": "BASE Qwen3-1.7B", "yours": "YOUR fine-tune", "full": "FULL fine-tune"}

con = Console()
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")


# ----------------------------------------------------------------------------- engine
class Engine:
    def __init__(self, a):
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        self.SamplingParams = SamplingParams
        self.a = a
        self.models = {"base": None}
        for i, (name, path) in enumerate((("yours", a.yours), ("full", a.full)), start=1):
            if not path:
                continue
            if os.path.isfile(f"{path}/adapter_config.json"):
                self.models[name] = LoRARequest(name, i, path)
            else:
                con.print(f"[yellow]Skipping {TITLES[name]}: no adapter at {path}[/yellow]")
        self.tok = AutoTokenizer.from_pretrained(a.model)
        # vLLM's engine (and the subprocess it starts) logs a lot: send it to a file by pointing
        # fds 1/2 there while the engine starts, then restore the terminal for our output.
        log = open(a.engine_log, "a")
        saved = os.dup(1), os.dup(2)
        with con.status(f"Loading {a.model} + {len(self.models) - 1} LoRA adapter(s), about "
                        f"1-2 min (engine log: {a.engine_log})"):
            sys.stdout.flush(), sys.stderr.flush()
            os.dup2(log.fileno(), 1), os.dup2(log.fileno(), 2)
            try:
                self.llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_model_len,
                               gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=True,
                               enable_lora=len(self.models) > 1, max_loras=2,
                               max_lora_rank=64, seed=0)
            finally:
                sys.stdout.flush(), sys.stderr.flush()
                os.dup2(saved[0], 1), os.dup2(saved[1], 2)

    def render(self, conv):
        return self.tok.apply_chat_template(conv, tools=TOOLS, tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)

    def generate(self, prompts, seeds, loras):
        sps = [self.SamplingParams(temperature=self.a.temperature, top_p=0.8, top_k=20,
                                   max_tokens=self.a.max_tokens, seed=s,
                                   skip_special_tokens=False) for s in seeds]
        if all(l is None for l in loras):
            loras = None
        outs = self.llm.generate(prompts, sps, lora_request=loras, use_tqdm=False)
        return [(o.outputs[0].text, len(o.outputs[0].token_ids)) for o in outs]


# ----------------------------------------------------------------------------- agent loop
def run(engine, jobs, max_turns, label=""):
    """jobs: list of dicts {model, task, tasks_root, seed}. Runs all episodes in lockstep."""
    eps = []
    for j in jobs:
        env = TaskEnv(j["task"], j["tasks_root"], f"/tmp/compare_work/{j['model']}")
        conv = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER.format(fn=j["task"]["entry_point"])}]
        eps.append({**j, "env": env, "conv": conv, "steps": [], "turns": 0, "ended": None,
                    "tokens": 0})
    active = list(range(len(eps)))
    t0 = time.time()
    for turn in range(max_turns):
        if not active:
            break
        prompts, idx = [], []
        for i in active:
            p = engine.render(eps[i]["conv"])
            if len(engine.tok(p, add_special_tokens=False)["input_ids"]) + engine.a.max_tokens \
                    > engine.a.max_model_len:
                eps[i]["ended"] = "context_full"
                continue
            prompts.append(p)
            idx.append(i)
        if not prompts:
            break
        # seed = task_index * 101 + turn: the same per-request seeds agent_eval.py uses
        outs = engine.generate(prompts, [eps[i]["seed"] + turn for i in idx],
                               [engine.models[eps[i]["model"]] for i in idx])
        still = []
        for i, (text, ntok) in zip(idx, outs):
            if step(eps[i], text, ntok):
                still.append(i)
        active = still
        con.print(f"  [dim]{label} turn {turn + 1}: {len(active)} of {len(eps)} still working "
                  f"({time.time() - t0:.0f}s)[/dim]", end="\r")
    con.print(" " * 80, end="\r")
    for i in active:
        eps[i]["ended"] = "max_turns"
    for e in eps:
        e["grade"] = e["env"].grade()
        pristine = open(f"{e['env'].pristine}/{e['task']['module']}").read()
        final = open(f"{e['env'].dir}/{e['task']['module']}").read()
        e["diff"] = "".join(difflib.unified_diff(pristine.splitlines(True), final.splitlines(True),
                                                 "before", "after", n=0))
    return eps


def step(e, text, ntok):
    content, calls, bad = parse_assistant(text)
    e["turns"] += 1
    e["tokens"] += ntok
    msg = {"role": "assistant", "content": content}
    if calls:
        msg["tool_calls"] = [{"type": "function", "function": {"name": n, "arguments": g}}
                             for n, g in calls]
    e["conv"].append(msg)
    if not calls:
        if bad:
            e["steps"].append("[red]✗ malformed tool call[/red]")
            e["conv"].append({"role": "user", "content": PARSE_ERROR})
            return True
        said = escape(" ".join(content.split())[:60])
        e["steps"].append(f"[yellow]stopped, replied in text[/yellow] [dim]{said}[/dim]")
        e["ended"] = "no_tool_call"
        return False
    done = False
    for name, args in calls:
        obs, d = e["env"].call(name, args)
        e["conv"].append({"role": "tool", "content": obs})
        e["steps"].append(describe(name, args, obs))
        done = done or d
    if done:
        e["ended"] = "submit"
        return False
    return True


def describe(name, args, obs):
    """One short line per tool call."""
    def short(p):
        return escape(str(p).replace("/testbed/", ""))
    if name == "bash":
        cmd = " ".join(str(args.get("command", "")).split())
        line = "$ " + escape(cmd.replace("/testbed/", "")[:48])
        if "pytest" in cmd or "tests/" in cmd:
            m = re.findall(r"(\d+) (passed|failed|error)", obs)
            if m:
                line += "  -> " + ", ".join(f"{n} {w}" for n, w in m)
            elif obs.strip().endswith("OK"):
                line += "  -> [green]OK[/green]"
            elif "Error" in obs or "Traceback" in obs:
                line += "  -> [red]error[/red]"
        return line
    if name == "str_replace_editor":
        cmd, path = args.get("command"), short(args.get("path", ""))
        if cmd == "view":
            return f"view {path}"
        if cmd in ("str_replace", "insert", "create", "undo_edit"):
            ok = ("has been edited" in obs or "created successfully" in obs
                  or "undone successfully" in obs)
            verb = {"str_replace": "edit", "insert": "insert", "create": "create",
                    "undo_edit": "undo"}[cmd]
            mark = "[green]✓[/green]" if ok else "[red]✗ " + (
                "no match" if "did not appear" in obs else "failed") + "[/red]"
            warn = " [yellow](test file!)[/yellow]" if path.startswith("tests/") and ok else ""
            return f"{verb} {path} {mark}{warn}"
        return f"[red]✗ editor: bad command {escape(str(cmd))}[/red]"
    if name == "submit":
        return "[bold]submit[/bold]"
    return f"[red]✗ unknown tool {escape(str(name))}[/red]"


def compress(steps, max_lines):
    """Collapse consecutive repeats (x N) and cap the number of lines."""
    out = []
    for s in steps:
        if out and out[-1][0] == s:
            out[-1][1] += 1
        else:
            out.append([s, 1])
    lines = [f"{s}  [magenta](x{n})[/magenta]" if n > 1 else s for s, n in out]
    if len(lines) > max_lines:
        keep = max_lines - 1
        lines = lines[:keep // 2] + [f"[dim]... {len(lines) - keep} more steps ...[/dim]"] + \
            lines[-(keep - keep // 2):]
    return lines


def verdict(e):
    g = e["grade"]
    if g["resolved"]:
        v = "[bold green]✓ FIXED[/bold green] tests pass"
    elif not g["others_intact"]:
        v = "[bold red]✗ broke other tests[/bold red]"
    else:
        v = "[bold red]✗ NOT FIXED[/bold red]"
    why = {"submit": "submitted", "no_tool_call": "stopped early", "max_turns": "ran out of turns",
           "context_full": "context full"}.get(e["ended"], e["ended"])
    return f"{v}\n{e['turns']} turns, {why}"


def show(title, eps, subtitle=""):
    con.rule(f"[bold]{title}[/bold]")
    if subtitle:
        con.print(subtitle)
    t = Table(box=box.SIMPLE_HEAVY, expand=True, show_lines=False)
    for e in eps:
        t.add_column(TITLES[e["model"]], style=COLORS[e["model"]], ratio=1, overflow="fold")
    cols = [compress(e["steps"], 14) for e in eps]
    for r in range(max(len(c) for c in cols)):
        t.add_row(*[f"[white]{r + 1:>2}[/white] {c[r]}" if r < len(c) else "" for c in cols])
    t.add_section()
    t.add_row(*[verdict(e) for e in eps])
    con.print(t)
    diffs = [e for e in eps if e["diff"]]
    for e in diffs:
        body = "\n".join(l for l in e["diff"].splitlines() if not l.startswith(("---", "+++")))
        con.print(Panel(escape(body[:1200]) or "(no change)", title=f"{TITLES[e['model']]}: change to "
                        f"{e['task']['module']}", border_style=COLORS[e["model"]], expand=False))
    for e in eps:
        if not e["diff"]:
            con.print(f"[{COLORS[e['model']]}]{TITLES[e['model']]}[/]: did not change "
                      f"{e['task']['module']}")


def save(eps, a, tag):
    os.makedirs(a.out, exist_ok=True)
    path = f"{a.out}/{time.strftime('%Y%m%d-%H%M%S')}-{tag}.jsonl"
    with open(path, "w") as f:
        for e in eps:
            f.write(json.dumps({"model": e["model"], "task": e["task"]["id"],
                                "grade": e["grade"], "turns": e["turns"], "ended": e["ended"],
                                "diff": e["diff"], "transcript": e["conv"]}) + "\n")
    con.print(f"[dim]Full transcripts saved to {path}[/dim]")


# ----------------------------------------------------------------------------- task sources
def eval_tasks():
    return [json.loads(l) for l in open(f"{TASKS_ROOT}/tasks.jsonl")]


def humaneval():
    return {r["entry_point"]: r for r in map(json.loads, open(f"{TASKS_ROOT}/humaneval.jsonl"))}


def seed_for(task_id):
    """Same per-task seed as agent_eval.py for eval tasks; a stable hash otherwise."""
    ids = [t["id"] for t in eval_tasks()]
    if task_id in ids:
        return ids.index(task_id) * 101
    return int(hashlib.sha256(task_id.encode()).hexdigest()[:6], 16)


def build_task(task_id, entry, buggy_src, test_src, distractors=2):
    """Write a mini-repo like the eval tasks: buggy module + tests + correct distractors."""
    root = "/tmp/compare_tasks"
    shutil.rmtree(f"{root}/{task_id}", ignore_errors=True)
    he = humaneval()
    others = random.Random(task_id).sample([n for n in he if n != entry], distractors)
    modules = {entry: buggy_src, **{o: he[o]["prompt"] + he[o]["canonical_solution"] for o in others}}
    tests = {entry: test_src, **{o: test_file(o, he[o]["test"]) for o in others}}
    write_repo(f"{root}/{task_id}/repo", modules, tests)
    return {"id": task_id, "entry_point": entry, "module": f"lib/{entry}.py",
            "modules": list(modules)}, root


def own_test_file(entry, asserts):
    body = "\n".join("    " + l for l in asserts.strip().splitlines()) or "    pass"
    return ("import os\nimport sys\n\n"
            "sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n\n"
            f"from lib.{entry} import *  # noqa: F401,F403\n\n\n"
            f"def test_{entry}():\n{body}\n\n\n"
            f"if __name__ == \"__main__\":\n    test_{entry}()\n    print(\"OK\")\n")


def read_block(prompt):
    con.print(prompt + " [dim](finish with a line containing only END)[/dim]")
    lines = []
    while True:
        try:
            l = input()
        except EOFError:
            break
        if l.strip() == "END":
            break
        lines.append(l)
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- modes
def compare_task(engine, a, task, root, title, subtitle=""):
    jobs = [{"model": m, "task": task, "tasks_root": root, "seed": seed_for(task["id"])}
            for m in engine.models]
    eps = run(engine, jobs, a.max_turns, label=task["entry_point"])
    show(title, eps, subtitle)
    save(eps, a, task["id"])
    return eps


def mode_showcase(engine, a):
    tasks = {t["id"]: t for t in eval_tasks()}
    con.print(Panel(
        "Each model gets the same repo with one injected bug and the request:\n"
        f"[italic]{USER.format(fn='<function>')}[/italic]\n\n"
        "These scenarios were picked to show a mix (wins, a tie and a loss). They are sampled "
        "live, so a rerun can differ. Use the quick scoreboard (menu 4) for real rates.",
        title="Showcase", expand=False))
    tally = {m: 0 for m in engine.models}
    for i, tid in enumerate(a.scenarios, 1):
        t = tasks[tid]
        eps = compare_task(engine, a, t, TASKS_ROOT,
                           f"Scenario {i}/{len(a.scenarios)}: fix `{t['entry_point']}`",
                           f"[dim]bug type: {t['mutation']}, repo has {len(t['modules'])} modules"
                           "[/dim]")
        for e in eps:
            tally[e["model"]] += e["grade"]["resolved"]
    summary(tally, len(a.scenarios), "Showcase total")


def mode_paste(engine, a):
    src = read_block("Paste the [bold]buggy function[/bold] (with imports if it needs any):")
    try:
        fns = [n.name for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)]
    except SyntaxError as e:
        con.print(f"[red]That code doesn't parse: {e}[/red]")
        return
    if not fns:
        con.print("[red]No function definition found.[/red]")
        return
    entry = fns[0]
    asserts = read_block(f"Paste [bold]assert tests[/bold] for `{entry}` that a CORRECT version "
                         f"passes, e.g. assert {entry}(...) == ...:")
    test_src = own_test_file(entry, asserts)
    task, root = build_task(f"byo_{entry}", entry, src, test_src)
    probe = TaskEnv(task, root, "/tmp/compare_work/probe")
    if probe.grade()["target_fixed"]:
        con.print("[yellow]Your tests already pass on this code, so there's no bug to fix. "
                  "Add an assert that the buggy version fails.[/yellow]")
        return
    compare_task(engine, a, task, root, f"Your bug: fix `{entry}`")


def mode_inject(engine, a):
    he = humaneval()
    name = input("HumanEval function name (Enter for a random one): ").strip()
    if not name:
        name = random.choice([n for n in he if sites(target_fn(ast.parse(
            he[n]["prompt"] + he[n]["canonical_solution"]), n))])
    if name not in he:
        close = difflib.get_close_matches(name, he, n=5)
        con.print(f"[red]Unknown function. Close matches: {', '.join(close) or 'none'}[/red]")
        return
    r = he[name]
    src = r["prompt"] + r["canonical_solution"]
    kinds = sorted({s[0] for s in sites(target_fn(ast.parse(src), name))})
    if not kinds:
        con.print("[red]No place to inject a bug in this function, pick another.[/red]")
        return
    kind = input(f"Bug type {kinds} (Enter for random): ").strip() or random.choice(kinds)
    if kind not in kinds:
        con.print(f"[red]Pick one of {kinds}[/red]")
        return
    cands = [k for k, s in enumerate(sites(target_fn(ast.parse(src), name))) if s[0] == kind]
    random.shuffle(cands)
    buggy = None
    for k in cands:
        b, _ = mutate(src, name, k)
        if run_check(b, r["test"], name) == "fail":
            buggy = b
            break
    if buggy is None:
        con.print("[yellow]Couldn't find a bug of that type that the tests catch. Try another "
                  "type or function.[/yellow]")
        return
    bug = "".join(difflib.unified_diff((ast.unparse(ast.parse(src)) + "\n").splitlines(True),
                                       buggy.splitlines(True), "correct", "buggy", n=0))
    con.print(Panel(escape("\n".join(l for l in bug.splitlines() if not l.startswith(("---", "+++")))),
                    title=f"Injected bug in {name} (the models don't see this)", expand=False))
    task, root = build_task(f"inject_{name}_{kind}", name, buggy, test_file(name, r["test"]))
    compare_task(engine, a, task, root, f"Injected {kind} bug: fix `{name}`")


def mode_quick(engine, a):
    tasks = eval_tasks()
    index = {t["id"]: i for i, t in enumerate(tasks)}
    pick = [index[t] for t in QUICK_SET][:a.quick]
    if a.quick > len(pick):  # more than the fixed set: add a seeded sample of the rest
        rest = [i for i in range(len(tasks)) if i not in pick]
        pick += random.Random(0).sample(rest, min(a.quick - len(pick), len(rest)))
    jobs = [{"model": m, "task": tasks[i], "tasks_root": TASKS_ROOT, "seed": i * 101}
            for i in pick for m in engine.models]
    con.print(f"Running {len(pick)} tasks x {len(engine.models)} models in parallel, up to "
              f"{a.quick_turns} turns each. This takes roughly {len(pick) // 2 + 3}-{len(pick) + 3} "
              "min. Small samples are noisy: with 20 tasks, +/-10 points is normal.")
    eps = run(engine, jobs, a.quick_turns, label="scoreboard")
    t = Table(title=f"Quick scoreboard: {len(pick)} tasks", box=box.SIMPLE_HEAVY)
    t.add_column("task")
    for m in engine.models:
        t.add_column(TITLES[m], style=COLORS[m], justify="center")
    by = {(e["task"]["id"], e["model"]): e for e in eps}
    tally = {m: 0 for m in engine.models}
    for i in pick:
        tid = tasks[i]["id"]
        row = [tasks[i]["entry_point"]]
        for m in engine.models:
            ok = by[(tid, m)]["grade"]["resolved"]
            tally[m] += ok
            row.append("✓" if ok else "·")
        t.add_row(*row)
    con.print(t)
    summary(tally, len(pick), "Quick scoreboard")
    save(eps, a, f"quick{len(pick)}")


def summary(tally, n, title):
    t = Table(title=title, box=box.ROUNDED)
    t.add_column("model")
    t.add_column("fixed", justify="right")
    for m, k in tally.items():
        t.add_row(f"[{COLORS[m]}]{TITLES[m]}[/]", f"{k}/{n}  ({100 * k / n:.0f}%)")
    con.print(t)
    con.print("[dim]Reference, all 150 eval tasks: base 8.7%, workshop-size fine-tune 33.3%, "
              "full fine-tune 53.3%. The 20-task scoreboard set is chosen to match these "
              "(recorded: 10% / 35% / 55%).[/dim]")


# ----------------------------------------------------------------------------- main
def main():
    global TASKS_ROOT
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", choices=["menu", "showcase", "paste", "inject", "quick"],
                    default="menu")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--yours", default=os.environ.get("YOURS_ADAPTER", ""),
                    help="your LoRA adapter dir, e.g. $DATA_ROOT/runs/<RUN>/adapter")
    ap.add_argument("--full", default=os.environ.get(
        "FULL_ADAPTER", f"{ROOT}/runs/agentic-full/adapter"),
        help="pre-trained full fine-tune adapter dir; pass --full '' for base vs yours only")
    ap.add_argument("--scenarios", default="",
                    help="comma-separated task ids (default: 3-way or 2-way showcase set)")
    ap.add_argument("--tasks", default=TASKS_ROOT, help="agent task dir (tasks.jsonl, humaneval.jsonl)")
    ap.add_argument("--quick", type=int, default=20, help="tasks for the quick scoreboard")
    ap.add_argument("--quick_turns", type=int, default=30,
                    help="turn limit in the scoreboard (30 = same as the 150-task eval)")
    ap.add_argument("--max_turns", type=int, default=20)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--max_model_len", type=int, default=32768)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--gpu_mem", type=float, default=0.88)
    ap.add_argument("--out", default=f"{ROOT}/compare")
    ap.add_argument("--engine_log", default="/tmp/vllm-engine.log")
    a = ap.parse_args()
    TASKS_ROOT = a.tasks

    engine = Engine(a)
    default = SHOWCASE if "full" in engine.models else SHOWCASE_2WAY
    a.scenarios = [s.strip() for s in (a.scenarios or ",".join(default)).split(",") if s.strip()]
    con.print(f"Comparing: " + ", ".join(f"[{COLORS[m]}]{TITLES[m]}[/]" for m in engine.models))
    modes = {"showcase": mode_showcase, "paste": mode_paste, "inject": mode_inject,
             "quick": mode_quick}
    if a.mode != "menu":
        modes[a.mode](engine, a)
        return
    menu = {"1": "showcase", "2": "paste", "3": "inject", "4": "quick"}
    while True:
        con.print(Panel("1  Showcase scenarios\n2  Your own bug (paste a function + tests)\n"
                        "3  Inject a bug into a HumanEval function\n"
                        f"4  Quick scoreboard ({a.quick} tasks)\nq  Quit",
                        title="Base vs fine-tuned", expand=False))
        try:
            c = input("> ").strip().lower()
        except EOFError:
            break
        if c in ("q", "quit", "exit"):
            break
        if c in menu:
            try:
                modes[menu[c]](engine, a)
            except KeyboardInterrupt:
                con.print("\n[yellow]Interrupted, back to the menu.[/yellow]")


if __name__ == "__main__":
    sys.exit(main())