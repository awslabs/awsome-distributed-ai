#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# ============================================================
# Evaluation Script for SLIME-trained Models
#
# Evaluates a HuggingFace-format checkpoint on AIME-2024 using
# SGLang for inference, then scores with the math reward function.
#
# Usage:
#   bash scripts/evaluate.sh \
#       --model-path /fsx/models/Qwen3-4B-GRPO-step60 \
#       --eval-data /fsx/data/aime-2024/aime-2024.jsonl \
#       --num-samples 16 \
#       --tp-size 2 \
#       --max-tokens 16384
# ============================================================

set -euo pipefail

# Defaults
MODEL_PATH=""
EVAL_DATA="/fsx/data/aime-2024/aime-2024.jsonl"
NUM_SAMPLES=16
TP_SIZE=2
MAX_TOKENS=16384
TEMPERATURE=0.6
TOP_P=0.95
OUTPUT_DIR="/fsx/eval_results"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --eval-data) EVAL_DATA="$2"; shift 2 ;;
        --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
        --tp-size) TP_SIZE="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --top-p) TOP_P="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "${MODEL_PATH}" ]]; then
    echo "Usage: $0 --model-path <path> [--eval-data <path>] [--num-samples N] ..."
    exit 1
fi

MODEL_NAME="$(basename "${MODEL_PATH}")"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="${OUTPUT_DIR}/${MODEL_NAME}_${TIMESTAMP}"
mkdir -p "${RESULT_DIR}"

echo "============================================================"
echo "  SLIME Model Evaluation"
echo "============================================================"
echo "  Model:       ${MODEL_PATH}"
echo "  Eval data:   ${EVAL_DATA}"
echo "  Samples:     ${NUM_SAMPLES} per prompt"
echo "  TP size:     ${TP_SIZE}"
echo "  Max tokens:  ${MAX_TOKENS}"
echo "  Output:      ${RESULT_DIR}"
echo "============================================================"

# ----- Step 1: Start SGLang server -----
echo "[INFO] Starting SGLang server (TP=${TP_SIZE})..."
python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --tp "${TP_SIZE}" \
    --host 0.0.0.0 \
    --port 30000 \
    --mem-fraction-static 0.85 \
    --log-level WARN &

SGLANG_PID=$!

# Wait for server to be ready
echo "[INFO] Waiting for SGLang server to start..."
for i in $(seq 1 120); do
    if curl -s http://localhost:30000/health > /dev/null 2>&1; then
        echo "[INFO] SGLang server ready."
        break
    fi
    if [[ $i -eq 120 ]]; then
        echo "[ERROR] SGLang server failed to start within 120 seconds."
        kill ${SGLANG_PID} 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

# ----- Step 2: Run evaluation -----
echo "[INFO] Running evaluation..."
python3 - <<'EVAL_SCRIPT'
import json
import sys
import os
import re
import asyncio
import aiohttp

EVAL_DATA = os.environ.get("EVAL_DATA", "/fsx/data/aime-2024/aime-2024.jsonl")
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "16"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "16384"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.6"))
TOP_P = float(os.environ.get("TOP_P", "0.95"))
RESULT_DIR = os.environ.get("RESULT_DIR", "/fsx/eval_results")
SERVER_URL = "http://localhost:30000/v1/chat/completions"

# Load evaluation prompts
prompts = []
with open(EVAL_DATA, "r") as f:
    for line in f:
        item = json.loads(line.strip())
        prompts.append(item)

print(f"Loaded {len(prompts)} evaluation prompts")

async def evaluate_prompt(session, prompt_item, sample_idx):
    """Generate a response and check correctness."""
    messages = [{"role": "user", "content": prompt_item.get("prompt", prompt_item.get("question", ""))}]

    payload = {
        "model": "default",
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    }

    try:
        async with session.post(SERVER_URL, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
            result = await resp.json()
            response_text = result["choices"][0]["message"]["content"]

            # Extract \boxed{} answer
            pattern = r"\\boxed\{([^}]*)\}"
            matches = re.findall(pattern, response_text)
            predicted = matches[-1].strip() if matches else ""

            label = prompt_item.get("label", prompt_item.get("answer", ""))

            return {
                "prompt_idx": prompt_item.get("idx", 0),
                "sample_idx": sample_idx,
                "predicted": predicted,
                "label": label,
                "correct": predicted.strip() == label.strip() if predicted else False,
                "response_length": len(response_text),
            }
    except Exception as e:
        return {
            "prompt_idx": prompt_item.get("idx", 0),
            "sample_idx": sample_idx,
            "predicted": "",
            "label": prompt_item.get("label", ""),
            "correct": False,
            "error": str(e),
        }

async def main():
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        for prompt_item in prompts:
            for s in range(NUM_SAMPLES):
                tasks.append(evaluate_prompt(session, prompt_item, s))

        print(f"Evaluating {len(tasks)} total samples...")
        results = await asyncio.gather(*tasks)

    # Compute metrics
    total = len(results)
    correct = sum(1 for r in results if r.get("correct", False))
    accuracy = correct / total if total > 0 else 0

    # Per-prompt pass@k (at least one correct)
    from collections import defaultdict
    prompt_results = defaultdict(list)
    for r in results:
        prompt_results[r["prompt_idx"]].append(r.get("correct", False))

    pass_at_k = sum(1 for prs in prompt_results.values() if any(prs)) / len(prompt_results) if prompt_results else 0

    print(f"\n{'='*60}")
    print(f"  Evaluation Results")
    print(f"{'='*60}")
    print(f"  Total samples:    {total}")
    print(f"  Correct:          {correct}")
    print(f"  Accuracy:         {accuracy:.4f}")
    print(f"  Pass@{NUM_SAMPLES}:          {pass_at_k:.4f}")
    print(f"  Prompts evaluated:{len(prompt_results)}")
    print(f"{'='*60}")

    # Save results
    output_file = os.path.join(RESULT_DIR, "eval_results.json")
    with open(output_file, "w") as f:
        json.dump({
            "metrics": {
                "total_samples": total,
                "correct": correct,
                "accuracy": accuracy,
                "pass_at_k": pass_at_k,
                "k": NUM_SAMPLES,
            },
            "results": results,
        }, f, indent=2)
    print(f"  Results saved to: {output_file}")

asyncio.run(main())
EVAL_SCRIPT

# ----- Step 3: Cleanup -----
echo "[INFO] Stopping SGLang server..."
kill ${SGLANG_PID} 2>/dev/null || true
wait ${SGLANG_PID} 2>/dev/null || true

echo "[INFO] Evaluation complete."
