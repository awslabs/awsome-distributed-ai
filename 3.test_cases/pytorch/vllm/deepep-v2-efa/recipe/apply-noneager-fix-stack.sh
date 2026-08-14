#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0
# Fable-confirmed non-eager fix stack onto /opt/vllm @ e2f993dc4 (NONEAGER-ROOTCAUSE 2026-08-13 §2):
# cherry-pick vllm e48592066 (#46404) + 04c2a8dea (#46432) + meta-None H2 guard. Idempotent.
set -uo pipefail
cd /opt/vllm
git config user.email repro@local; git config user.name repro
F=vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_v2.py
if git log --oneline -5 | grep -q 04c2a8de; then echo "cherry-picks already applied"; else
  set -e
  git fetch origin e48592066 04c2a8dea 2>/dev/null || git fetch origin e485920662b04c9615004d1636e259e6989c8f8d 04c2a8dea8b48ec0658d051f37a0d5e163bd5a86 2>/dev/null || { git fetch origin main --depth 500; }
  git cherry-pick e48592066 || { echo CHERRY-PICK-46404-FAILED; git cherry-pick --abort; exit 41; }
  git cherry-pick 04c2a8dea || { echo CHERRY-PICK-46432-FAILED; git cherry-pick --abort; exit 42; }
  set +e
fi
grep -q "_globalize_recv_topk_idx" $F || { echo "ASSERT FAIL: 46432 kernel missing"; exit 43; }
python3 - <<'PY'
import re, sys
p = "vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_v2.py"
c = open(p).read()
if "H2_META_NONE_GUARD" in c:
    print("meta-None guard already applied"); sys.exit(0)
pat = re.compile(r"( *)expert_tokens_meta = mk\.ExpertTokensMetadata\.make_from_list\(\n( *)recv_expert_num_tokens,\n( *)device=expert_x\.device,?\n( *)\)")
m = pat.search(c)
assert m, "meta-None target block not found post-cherry-pick — inspect _receiver manually"
i = m.group(1)
rep = (f"{i}# H2_META_NONE_GUARD (nccl-ep-431): do_cpu_sync=False leaves the list EMPTY;\n"
       f"{i}# a PRESENT 0-element meta violates the decode-mode contract (must be None).\n"
       f"{i}if len(recv_expert_num_tokens) == 0:\n"
       f"{i}    expert_tokens_meta = None\n"
       f"{i}else:\n"
       f"{i}    expert_tokens_meta = mk.ExpertTokensMetadata.make_from_list(\n"
       f"{i}        recv_expert_num_tokens,\n"
       f"{i}        device=expert_x.device,\n"
       f"{i}    )")
c = c[:m.start()] + rep + c[m.end():]
open(p, "w").write(c)
print("meta-None guard applied")
PY
[ $? -eq 0 ] || exit 44
python3 -c "import ast; ast.parse(open('$F').read()); print('py-syntax OK')" || exit 45
git log --oneline -3 | head -3
grep -c H2_META_NONE_GUARD $F
echo "NONEAGER-STACK-APPLIED"
