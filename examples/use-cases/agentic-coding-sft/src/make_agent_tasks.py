# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Build agentic bug-fix tasks from HumanEval (runs on the CPU node, training image).

Each task is a tiny repository:
    /testbed/README.md
    /testbed/conftest.py            (makes the repo root importable for pytest)
    /testbed/lib/<fn>.py            one module per function: the TARGET (with an injected bug)
                                    plus 3 distractor modules (correct)
    /testbed/tests/test_<fn>.py     HumanEval's unit tests for each module

The bug is a single AST mutation of HumanEval's canonical solution (comparison flip, +/- swap,
and/or swap, off-by-one constant, ...). A mutation is accepted only if the canonical code passes
the tests, the mutated code fails them, and the mutated code doesn't hang. So every task is
solvable with a one-line fix, and success is graded by running the tests.

Writes <out>/tasks.jsonl and <out>/<task_id>/repo/ (pristine copy, incl. tests).
"""
import argparse
import ast
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile

ROOT = os.environ.get("DATA_ROOT", "/fsx")  # FSx for Lustre mount


CMP_SWAP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
            ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.In: ast.NotIn, ast.NotIn: ast.In,
            ast.Is: ast.IsNot, ast.IsNot: ast.Is}
BIN_SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv, ast.FloorDiv: ast.Mult,
            ast.Mod: ast.FloorDiv}
BOOL_SWAP = {ast.And: ast.Or, ast.Or: ast.And}


def target_fn(tree, name):
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def sites(fn):
    out = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Compare):
            out += [("comparison", node, i) for i, op in enumerate(node.ops) if type(op) in CMP_SWAP]
        elif isinstance(node, ast.BinOp) and type(node.op) in BIN_SWAP:
            out.append(("arithmetic", node, None))
        elif isinstance(node, ast.BoolOp) and type(node.op) in BOOL_SWAP:
            out.append(("boolean", node, None))
        elif isinstance(node, ast.Constant) and type(node.value) is int:
            out.append(("off_by_one", node, None))
    return out


def mutate(src, name, k):
    tree = ast.parse(src)
    kind, node, i = sites(target_fn(tree, name))[k]
    if kind == "comparison":
        node.ops[i] = CMP_SWAP[type(node.ops[i])]()
    elif kind == "arithmetic":
        node.op = BIN_SWAP[type(node.op)]()
    elif kind == "boolean":
        node.op = BOOL_SWAP[type(node.op)]()
    else:
        node.value = node.value + 1
    return ast.unparse(tree) + "\n", kind


def run_check(src, test, entry, timeout=10):
    """'pass' | 'fail' | 'timeout' for module source + HumanEval check()."""
    prog = f"{src}\n\n{test}\n\ncheck({entry})\n"
    with tempfile.TemporaryDirectory() as d:
        open(f"{d}/p.py", "w").write(prog)
        try:
            r = subprocess.run([sys.executable, f"{d}/p.py"], cwd=d, capture_output=True,
                               timeout=timeout, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return "timeout"
    return "pass" if r.returncode == 0 else "fail"


def test_file(entry, test):
    return (
        "import os\nimport sys\n\n"
        "sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n\n"
        f"from lib.{entry} import *  # noqa: F401,F403\n"
        f"from lib.{entry} import {entry}\n\n"
        f"{test.strip()}\n\n\n"
        f"def test_{entry}():\n    check({entry})\n\n\n"
        "if __name__ == \"__main__\":\n"
        f"    test_{entry}()\n    print(\"OK\")\n"
    )


def write_repo(root, modules, tests):
    os.makedirs(f"{root}/lib")
    os.makedirs(f"{root}/tests")
    open(f"{root}/README.md", "w").write(
        "# pyutils\n\nA small collection of Python utility functions, one per module in `lib/`.\n"
        "Unit tests live in `tests/` and run with `python -m pytest`.\n")
    open(f"{root}/conftest.py", "w").write("")
    open(f"{root}/lib/__init__.py", "w").write("")
    for name, src in modules.items():
        open(f"{root}/lib/{name}.py", "w").write(src)
    for name, src in tests.items():
        open(f"{root}/tests/test_{name}.py", "w").write(src)


def grade_dir(root, names, timeout=20):
    """{name: passed} running each test file directly (no pytest dependency)."""
    res = {}
    for n in names:
        try:
            r = subprocess.run([sys.executable, f"tests/test_{n}.py"], cwd=root,
                               capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL)
            res[n] = r.returncode == 0
        except subprocess.TimeoutExpired:
            res[n] = False
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--humaneval", default=f"{ROOT}/data/full/humaneval")
    ap.add_argument("--out", default=f"{ROOT}/agent_tasks/v1")
    ap.add_argument("--distractors", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    shutil.rmtree(a.out, ignore_errors=True)
    os.makedirs(a.out)

    from datasets import load_dataset, load_from_disk  # lazy: compare.py imports this module

    if os.path.isdir(a.humaneval):
        he = list(load_from_disk(a.humaneval))
    else:  # first run: download HumanEval and keep a copy on FSx
        ds = load_dataset("openai_humaneval", split="test")
        ds.save_to_disk(a.humaneval)
        he = list(ds)
    canon = {r["entry_point"]: r["prompt"] + r["canonical_solution"] for r in he}
    tests = {r["entry_point"]: r["test"] for r in he}
    names = [r["entry_point"] for r in he]
    rng = random.Random(a.seed)

    kept, skipped = [], {"no_sites": 0, "no_failing_mutation": 0, "canonical_fails": 0,
                         "repo_check_failed": 0}
    for idx, r in enumerate(he):
        entry = r["entry_point"]
        src = canon[entry]
        if run_check(src, tests[entry], entry) != "pass":
            skipped["canonical_fails"] += 1
            continue
        n_sites = len(sites(target_fn(ast.parse(src), entry)))
        if not n_sites:
            skipped["no_sites"] += 1
            continue
        order = list(range(n_sites))
        random.Random(a.seed * 1000 + idx).shuffle(order)
        bug = None
        for k in order[:20]:
            buggy, kind = mutate(src, entry, k)
            if buggy.strip() == ast.unparse(ast.parse(src)).strip():
                continue
            if run_check(buggy, tests[entry], entry) == "fail":
                bug = (buggy, kind)
                break
        if bug is None:
            skipped["no_failing_mutation"] += 1
            continue

        others = rng.sample([n for n in names if n != entry], a.distractors)
        modules = {entry: bug[0], **{o: canon[o] for o in others}}
        tfiles = {n: test_file(n, tests[n]) for n in modules}
        task_id = r["task_id"].replace("/", "_")
        repo = f"{a.out}/{task_id}/repo"
        write_repo(repo, modules, tfiles)

        # Sanity: target fails, distractors pass; restoring canonical makes everything pass.
        g = grade_dir(repo, modules)
        with tempfile.TemporaryDirectory() as d:
            shutil.copytree(repo, f"{d}/r")
            open(f"{d}/r/lib/{entry}.py", "w").write(src)
            g_fix = grade_dir(f"{d}/r", modules)
        if g[entry] or not all(g[o] for o in others) or not all(g_fix.values()):
            shutil.rmtree(f"{a.out}/{task_id}")
            skipped["repo_check_failed"] += 1
            continue

        kept.append({"id": task_id, "entry_point": entry, "module": f"lib/{entry}.py",
                     "mutation": bug[1], "distractors": others, "modules": list(modules)})

    with open(f"{a.out}/tasks.jsonl", "w") as f:
        for t in kept:
            f.write(json.dumps(t) + "\n")
    # Plain-JSON copy of HumanEval for tools without `datasets` (compare.py runs in the vLLM image).
    with open(f"{a.out}/humaneval.jsonl", "w") as f:
        for r in he:
            f.write(json.dumps({k: r[k] for k in ("task_id", "entry_point", "prompt",
                                                  "canonical_solution", "test")}) + "\n")
    kinds = {}
    for t in kept:
        kinds[t["mutation"]] = kinds.get(t["mutation"], 0) + 1
    print("TASKS " + json.dumps({"kept": len(kept), "skipped": skipped, "mutations": kinds}),
          flush=True)
    t = kept[0]
    print("SAMPLE", t)
    print(open(f"{a.out}/{t['id']}/repo/{t['module']}").read()[:1200])


if __name__ == "__main__":
    main()