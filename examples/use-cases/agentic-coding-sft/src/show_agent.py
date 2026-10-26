# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Print readable excerpts of agent-eval transcripts. usage: show_agent.py <label> [n] [ids...]"""
import json
import os
import sys

ROOT = os.environ.get("DATA_ROOT", "/fsx")  # FSx for Lustre mount
label = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
ids = set(sys.argv[3:])
rows = [json.loads(l) for l in open(f"{ROOT}/agent_eval/{label}.jsonl")]
print("per-task:", [(r["id"].split("_")[1], r["resolved"], r["ended"], r["turns"], r["edit_errors"])
                    for r in rows])
shown = 0
for r in rows:
    if ids and r["id"] not in ids:
        continue
    if shown >= n:
        break
    shown += 1
    print(f"\n===== {r['id']} resolved={r['resolved']} ended={r['ended']} turns={r['turns']} "
          f"edit_errors={r['edit_errors']} mutation={r['mutation']}")
    for m in r["transcript"][2:]:
        if m["role"] == "assistant":
            calls = m.get("tool_calls") or []
            c = (m["content"] or "").strip().replace("\n", " ")[:300]
            print(f"  A: {c}")
            for tc in calls:
                print(f"     CALL {tc['function']['name']} {json.dumps(tc['function']['arguments'])[:400]}")
        elif m["role"] == "tool":
            print(f"  T: {m['content'][:260]!r}")
        else:
            print(f"  U: {m['content'][:200]!r}")