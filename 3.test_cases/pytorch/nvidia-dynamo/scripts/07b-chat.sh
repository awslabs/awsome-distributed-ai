#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Interactive multi-turn streaming chat with a deployed scenario (keeps history).
# Auto-detects the deployed DynamoGraphDeployment if no name is given.
# Usage: ./scripts/07b-chat.sh [gpt-oss-agg|gpt-oss-disagg|qwen36-agg|qwen36-disagg]
set -euo pipefail
NS=dynamo-system
NAME="${1:-}"
if [ -z "$NAME" ]; then
  DET=$(kubectl get dynamographdeployment -n "$NS" -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null)
  n=$(echo "$DET" | grep -c . || true)
  if   [ "$n" -eq 0 ]; then echo "❌ No DynamoGraphDeployment deployed."; exit 1
  elif [ "$n" -eq 1 ]; then NAME="$DET"; echo "🔎 auto-detected: $NAME"
  elif [ -t 0 ]; then echo "Multiple deployed — pick one:"; select s in $DET; do [ -n "$s" ] && { NAME="$s"; break; }; done
  else echo "❌ Multiple deployed ($(echo $DET)). Pass one as arg."; exit 1; fi
fi
case "$NAME" in
  gpt-oss-*) MODEL="openai/gpt-oss-20b" ;;
  qwen36-*)  MODEL="Qwen/Qwen3.6-27B-FP8" ;;
  *) echo "Unknown scenario: $NAME"; exit 1 ;;
esac
SVC="${NAME}-frontend"; PORT="${LOCAL_PORT:-8000}"

echo "🔌 port-forward svc/${SVC} -> localhost:${PORT} ..."
kubectl port-forward -n "$NS" "svc/${SVC}" "${PORT}:8000" >/tmp/pf-${NAME}.log 2>&1 &
PF=$!; PYF="/tmp/dyn-chat-$$.py"; trap "kill $PF 2>/dev/null; rm -f $PYF" EXIT
for i in $(seq 1 30); do curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1 && break; sleep 1; done

echo "💬 Chatting with ${MODEL}  (commands: 'exit' quit · 'reset' clear history)"
cat > "$PYF" <<'PY'
import os, json, urllib.request
model, base = os.environ["MODEL"], os.environ["BASE"]
msgs = []; short = model.split("/")[-1]
while True:
    try: user = input("\nyou> ").strip()
    except EOFError: break
    if user in ("exit", "quit"): break
    if user == "reset": msgs = []; print("(history cleared)"); continue
    if not user: continue
    msgs.append({"role": "user", "content": user})
    body = json.dumps({"model": model, "messages": msgs, "max_tokens": 2000, "temperature": 0.7, "stream": True}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    answer = ""; dim = False
    print(f"\n{short}> ", end="", flush=True)
    try:
        for raw in urllib.request.urlopen(req, timeout=300):
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"): continue
            data = line[5:].strip()
            if data == "[DONE]": break
            try: delta = json.loads(data)["choices"][0].get("delta", {})
            except Exception: continue
            rc, c = delta.get("reasoning_content"), delta.get("content")
            if rc:
                if not dim: print("\033[2m", end=""); dim = True
                print(rc, end="", flush=True)
            if c:
                if dim: print("\033[0m", end=""); dim = False
                print(c, end="", flush=True); answer += c
        if dim: print("\033[0m", end="")
        print()
    except Exception as e:
        print("  ⚠️ error:", e); msgs.pop(); continue
    msgs.append({"role": "assistant", "content": answer})
print("bye 👋")
PY
MODEL="$MODEL" BASE="http://localhost:${PORT}" python3 "$PYF"
