#!/usr/bin/env bash
# STATUS: Partly verified -- the grader this selects was exercised in the pinned image
#   (\boxed{\frac{1}{2}} against 0.5 grades equivalent). The script itself has not been run
#   end to end on miles.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# ============================================================
# Evaluation Script for miles-trained Models
#
# Evaluates a HuggingFace-format checkpoint on AIME-2024: serves it with SGLang, samples
# responses, and grades them with the same verifier the training reward uses.
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
SERVER_PORT="${SERVER_PORT:-30000}"
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
echo "  miles Model Evaluation"
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
    --port "${SERVER_PORT}" \
    --mem-fraction-static "${SGLANG_MEM_FRACTION:-0.85}" \
    --log-level warning &   # lowercase: uvicorn KeyErrors on "WARN"

SGLANG_PID=$!

# Reap the server on ANY exit, not just the happy path. `set -e` is active, so a non-zero
# exit from the evaluation step below (the high-error-rate abort, or any Python exception)
# would otherwise skip the cleanup at the end of the script and leave a multi-GPU SGLang
# server holding the whole node's memory until someone notices.
cleanup_sglang() {
    kill "${SGLANG_PID}" 2>/dev/null || true
    wait "${SGLANG_PID}" 2>/dev/null || true
}
trap cleanup_sglang EXIT

# Wait for server to be ready
SGLANG_STARTUP_TIMEOUT=${SGLANG_STARTUP_TIMEOUT:-300}
echo "[INFO] Waiting for SGLang server to start (timeout=${SGLANG_STARTUP_TIMEOUT}s)..."
for i in $(seq 1 ${SGLANG_STARTUP_TIMEOUT}); do
    # -f, not just -s: without it curl exits 0 on a 4xx or 5xx body and the loop announces a
    # ready server against an endpoint that is answering with an error.
    if curl -sf "http://localhost:${SERVER_PORT}/health" > /dev/null 2>&1; then
        echo "[INFO] SGLang server ready."
        break
    fi
    if [[ $i -eq ${SGLANG_STARTUP_TIMEOUT} ]]; then
        echo "[ERROR] SGLang server failed to start within ${SGLANG_STARTUP_TIMEOUT} seconds."
        kill ${SGLANG_PID} 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

# ----- Step 2: Run evaluation -----
echo "[INFO] Running evaluation..."
# The heredoc below is quoted, so the Python reads these through the environment rather
# than through shell expansion. They must be exported: a plain assignment stays in the
# shell and the Python silently falls back to its defaults, which would produce
# confident-looking results for parameters the caller never asked for.
export EVAL_DATA NUM_SAMPLES MAX_TOKENS TEMPERATURE TOP_P RESULT_DIR SERVER_PORT
export EVAL_MAX_CONCURRENCY="${EVAL_MAX_CONCURRENCY:-32}"
export EVAL_REQUEST_TIMEOUT="${EVAL_REQUEST_TIMEOUT:-1800}"
export EVAL_ERROR_FRACTION_ABORT="${EVAL_ERROR_FRACTION_ABORT:-0.05}"

python3 - <<'EVAL_SCRIPT'
import json
import sys
import os
import asyncio
import aiohttp

EVAL_DATA = os.environ.get("EVAL_DATA", "/fsx/data/aime-2024/aime-2024.jsonl")
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "16"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "16384"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.6"))
TOP_P = float(os.environ.get("TOP_P", "0.95"))
RESULT_DIR = os.environ.get("RESULT_DIR", "/fsx/eval_results")
SERVER_URL = f"http://localhost:{os.environ.get('SERVER_PORT', '30000')}/v1/chat/completions"
# Bounded concurrency and a timeout sized for the generation, not for the queue. With
# MAX_TOKENS=16384 a single response can take minutes, so a 300s cap applied to every
# request at once made most of them time out and be tallied as wrong answers.
MAX_CONCURRENCY = int(os.environ.get("EVAL_MAX_CONCURRENCY", "32"))
REQUEST_TIMEOUT = float(os.environ.get("EVAL_REQUEST_TIMEOUT", "1800"))
ERROR_FRACTION_ABORT = float(os.environ.get("EVAL_ERROR_FRACTION_ABORT", "0.05"))
GRADER_TIMEOUT_S = float(os.environ.get("EVAL_GRADER_TIMEOUT_S", "10"))

# Load evaluation prompts
prompts = []
with open(EVAL_DATA, "r") as f:
    for line in f:
        item = json.loads(line.strip())
        prompts.append(item)

print(f"Loaded {len(prompts)} evaluation prompts")

def extract_boxed(text):
    r"""Return the content of the last \boxed{...}, honoring nested braces.

    Scanned with a brace counter rather than a regex: a character class cannot match the
    balanced braces in answers like `\boxed{\frac{1}{2}}`.
    """
    out = []
    needle = r"\boxed{"
    start = text.find(needle)
    while start != -1:
        i = start + len(needle)
        depth = 1
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if depth == 0:
            out.append(text[start + len(needle):i])
        start = text.find(needle, start + len(needle))
    return out[-1].strip() if out else ""


# grade_answer_verl first: it is the verifier the training reward uses, and it is in the image.
# math_verify is not -- requirements.txt pins it but the Dockerfile does not install it, so
# preferring it fell through to exact match on every run. Installing math_verify instead would
# put the eval and the reward on different verifiers.
_GRADER = None
_GRADER_NAME = "exact_string_match"
try:
    from miles.rollout.rm_hub.math_utils import grade_answer_verl as _GRADER
    _GRADER_NAME = "grade_answer_verl"
except ImportError:
    try:
        from math_verify import parse as _mv_parse, verify as _mv_verify

        def _GRADER(response_text, label):  # noqa: N802 - slot for the same call shape
            gold = _mv_parse(label if "\\boxed" in label else f"\\boxed{{{label}}}")
            return bool(_mv_verify(gold, _mv_parse(response_text)))

        _GRADER_NAME = "math_verify"
    except ImportError:
        print("WARNING: no semantic grader is importable, so grading falls back to exact string "
              "comparison. Equivalent answers such as 0.5 against 1/2 count as wrong and the "
              "reported accuracy understates the model.", file=sys.stderr)



def grade(response_text: str, predicted: str, label: str):
    """Return (correct, how). Exact string match first, then mathematical equivalence.

    Comparing the extracted \\boxed{} content to the label with == alone counts a correct
    answer wrong whenever it is written differently -- 0.5 against 1/2, a trailing "\\!" from
    LaTeX spacing -- and the reported accuracy then understates the model for a reason that has
    nothing to do with the model.

    The grader takes the whole response, not the extracted answer, because grade_answer_verl does
    its own extraction and may find an answer this one does not. `predicted` is still used for the
    exact-match shortcut and is reported, so a label the grader cannot parse still counts as
    correct when the answer is character-identical.
    """
    if not predicted:
        return False, "no_answer"
    if predicted == label:
        return True, "exact"
    if _GRADER is None:
        return False, "exact_only"
    try:
        if _GRADER(response_text, label):
            return True, "equivalent"
    except Exception as e:  # noqa: BLE001 - a verifier fault must not be reported as a score
        print(f"WARNING: {_GRADER_NAME} failed on label={label!r} predicted={predicted!r}: {e}",
              file=sys.stderr)
        return False, "verifier_error"
    return False, "incorrect"

async def evaluate_prompt(session, prompt_item, prompt_idx, sample_idx, sem):
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
        # Bound the in-flight requests. Submitting every prompt x sample at once makes most
        # requests spend their timeout queued rather than generating, and the tally below
        # counts a timeout as a wrong answer -- so accuracy drops for a reason unrelated to
        # the model.
        async with sem:
            async with session.post(SERVER_URL, json=payload,
                                    timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as resp:
                result = await resp.json()
                response_text = result["choices"][0]["message"]["content"]

                predicted = extract_boxed(response_text)
                # A label may be a number in the JSONL, and str.strip() on an int raises
                # AttributeError -- which the except below would turn into "incorrect" for
                # every sample.
                label = str(prompt_item.get("label", prompt_item.get("answer", ""))).strip()
                #
                # And it needs a ceiling: sympy on model-shaped text can run for a very long
                # time, and one such response would otherwise hold a slot until the script is
                # killed. wait_for abandons the coroutine, not the thread, so a wedged
                # verification leaks one thread for the rest of the run, bounded by the default
                # thread pool rather than by the semaphore, which wait_for has already released.
                # The cheaper trade here; app.py pays for a killable
                # process pool instead because it serves a training run rather than one script.
                try:
                    correct, how = await asyncio.wait_for(
                        asyncio.to_thread(grade, response_text, predicted, label),
                        timeout=GRADER_TIMEOUT_S,
                    )
                except asyncio.TimeoutError:
                    print(f"WARNING: {_GRADER_NAME} exceeded {GRADER_TIMEOUT_S}s on "
                          f"label={label!r} predicted={predicted!r}", file=sys.stderr)
                    correct, how = False, "verifier_error"

                return {
                    "prompt_idx": prompt_idx,
                    "sample_idx": sample_idx,
                    "predicted": predicted,
                    "label": label,
                    "correct": correct,
                    "graded_by": how,
                    "response_length": len(response_text),
                }
    except Exception as e:
        return {
            "prompt_idx": prompt_idx,
            "sample_idx": sample_idx,
            "predicted": "",
            "label": str(prompt_item.get("label", "")),
            "correct": False,
            "error": f"{type(e).__name__}: {e}",
        }

async def main():
    results = []
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    async with aiohttp.ClientSession() as session:
        tasks = []
        # pass@k is keyed on the loop index, not on a field in the data: AIME-2024 as prepared
        # here carries only {"prompt", "label"}, so keying on a missing "idx" would collapse
        # every prompt onto one bucket and turn pass@k into "any sample anywhere was right".
        for prompt_idx, prompt_item in enumerate(prompts):
            for s in range(NUM_SAMPLES):
                tasks.append(evaluate_prompt(session, prompt_item, prompt_idx, s, sem))

        print(f"Evaluating {len(tasks)} total samples "
              f"({MAX_CONCURRENCY} concurrent, {REQUEST_TIMEOUT}s timeout each)...")
        results = await asyncio.gather(*tasks)

    # Compute metrics
    total = len(results)
    correct = sum(1 for r in results if r.get("correct", False))
    errors = sum(1 for r in results if r.get("error"))
    equivalent = sum(1 for r in results if r.get("graded_by") == "equivalent")
    verifier_errors = sum(1 for r in results if r.get("graded_by") == "verifier_error")
    # A verifier fault is not a score, so it does not belong in the denominator. Counting it as
    # an incorrect sample made accuracy depend on how often the verifier broke: one correct
    # answer beside one verifier error reported 0.5, which is not a measurement of the model.
    graded = total - verifier_errors - errors
    accuracy = correct / graded if graded > 0 else 0

    # Per-prompt pass@k (at least one correct)
    from collections import defaultdict
    prompt_results = defaultdict(list)
    for r in results:
        # Same reason as the accuracy denominator: a prompt whose samples all failed to be
        # graded has not been measured, so it is not a pass@k miss either.
        if r.get("graded_by") == "verifier_error" or r.get("error"):
            continue
        prompt_results[r["prompt_idx"]].append(r.get("correct", False))

    pass_at_k = sum(1 for prs in prompt_results.values() if any(prs)) / len(prompt_results) if prompt_results else 0

    print(f"\n{'='*60}")
    print(f"  Evaluation Results")
    print(f"{'='*60}")
    print(f"  Total samples:    {total}")
    print(f"  Graded samples:   {graded} (accuracy denominator)")
    print(f"  Correct:          {correct}")
    print(f"    of which equivalent, not string-identical: {equivalent}")
    print(f"  Errors:           {errors}")
    if verifier_errors:
        print(f"  Verifier errors:  {verifier_errors} (not graded; see warnings above)")
    print(f"  Accuracy:         {accuracy:.4f}  (over graded samples)")
    print(f"  Pass@{NUM_SAMPLES}:          {pass_at_k:.4f}")
    print(f"  Prompts evaluated:{len(prompt_results)}")
    print(f"{'='*60}")
    if len(prompt_results) != len(prompts):
        print(f"  WARNING: {len(prompts)} prompts were loaded but only "
              f"{len(prompt_results)} distinct prompt indices appear in the results.")

    # Save results
    output_file = os.path.join(RESULT_DIR, "eval_results.json")
    with open(output_file, "w") as f:
        json.dump({
            "metrics": {
                "total_samples": total,
                "correct": correct,
                "graded_samples": graded,
                "correct_by_equivalence": equivalent,
                "verifier_errors": verifier_errors,
                "graded_with": _GRADER_NAME,
                "errors": errors,
                "accuracy": accuracy,
                "pass_at_k": pass_at_k,
                "k": NUM_SAMPLES,
                "prompts": len(prompt_results),
            },
            "results": results,
        }, f, indent=2)
    print(f"  Results saved to: {output_file}")

    # A run where a large share of requests failed has not measured accuracy, it has
    # measured the timeout. Exiting non-zero keeps that out of a results table.
    # Both kinds of non-measurement abort the run, because either one makes the number above
    # something other than accuracy. Without counting verifier faults here, a run where every
    # single sample failed to be graded exited zero and published a figure.
    ungraded = errors + verifier_errors
    if total and ungraded / total > ERROR_FRACTION_ABORT:
        print(f"\nERROR: {ungraded}/{total} samples were not graded "
              f"({errors} request failures, {verifier_errors} verifier faults; "
              f"> {ERROR_FRACTION_ABORT:.0%}). The figure above is therefore not this model's "
              "accuracy. For request failures raise SGLANG_MEM_FRACTION, lower "
              "EVAL_MAX_CONCURRENCY, or raise EVAL_REQUEST_TIMEOUT; for verifier faults read the "
              "warnings above. Then re-run.")
        sys.exit(2)
    if total and graded == 0:
        print("\nERROR: no sample was graded, so there is no accuracy to report.")
        sys.exit(2)

asyncio.run(main())
EVAL_SCRIPT

# ----- Step 3: Cleanup -----
# Killing and reaping is the EXIT trap's job, so there is deliberately no kill/wait here: a
# bare `wait` would block on a server that is still running and nothing would ever stop it.
echo "[INFO] Evaluation complete; stopping SGLang server."
