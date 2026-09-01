#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
set -xeuo pipefail

# ---------------------------------------------------------------------------
# Prepare HuggingFaceH4/Multilingual-Thinking dataset for OpenRLHF
#
# OpenRLHF expects JSONL with:
#   - A prompt field (specified via --input_key)
#   - A label field  (specified via --label_key)
#
# For chat-formatted prompts with --apply_chat_template, the prompt field
# should contain a list of message dicts: [{"role": "system", ...}, {"role": "user", ...}]
# stored as a JSON string.
#
# Alternatively, we can use a plain text prompt and let --apply_chat_template
# handle formatting.  For simplicity and to match the veRL approach, we
# store the prompt as a JSON-serialized list of messages and use
# --apply_chat_template during training.
#
# The label field contains the expected language code (e.g. "en", "fr")
# for the reward function.
# ---------------------------------------------------------------------------

DATA_DIR="${RAY_DATA_HOME:-/fsx/openrlhf}/data/multilingual-thinking"
echo "Target data directory: ${DATA_DIR}"

# Get the head pod name
HEAD_POD=$(kubectl get pods -l ray.io/node-type=head \
    -o jsonpath='{.items[0].metadata.name}')

if [ -z "$HEAD_POD" ]; then
    echo "Error: Could not find Ray head pod. Is your cluster running?"
    exit 1
fi

echo "Using Ray head pod: ${HEAD_POD}"

# ---- Python script that runs inside the pod ----
cat > /tmp/prepare_openrlhf_data.py << 'PYEOF'
import json
import os
import re
import sys

from datasets import load_dataset

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
    print("WARNING: langdetect not available - falling back to English")

data_dir = os.environ.get("DATA_DIR", "/fsx/openrlhf/data/multilingual-thinking")
os.makedirs(data_dir, exist_ok=True)

LANG_CODE_MAP = {
    "English": "en", "French": "fr", "German": "de",
    "Spanish": "es", "Italian": "it",
}
SUPPORTED_LANGUAGES = list(LANG_CODE_MAP.keys())


def detect_question_language(text):
    """Return the language name for a question string."""
    if not HAS_LANGDETECT:
        return "English"
    try:
        clean = re.sub(r"[0-9+\-*/=%$€£]", "", text)
        clean = re.sub(r"[^\w\s]", " ", clean)
        if len(clean.strip()) < 10:
            return "English"
        code = detect(clean)
        code_to_name = {v: k for k, v in LANG_CODE_MAP.items()}
        return code_to_name.get(code, "English")
    except Exception:
        return "English"


print("Loading HuggingFaceH4/Multilingual-Thinking ...")
dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking")


def make_openrlhf_row(example):
    """Convert a Multilingual-Thinking example into OpenRLHF JSONL format.

    Output schema:
        prompt: list[dict] — chat messages (JSON-serialized)
        label:  str        — expected language code (e.g. "en")
    """
    messages = example.get("messages", [])
    user_prompt = ""
    for msg in messages:
        if msg["role"] == "user":
            user_prompt = msg["content"]
            break

    lang_name = detect_question_language(user_prompt)
    if lang_name not in SUPPORTED_LANGUAGES:
        lang_name = "English"
    lang_code = LANG_CODE_MAP[lang_name]

    # Build chat messages (same system prompt as veRL)
    prompt_messages = [
        {
            "role": "system",
            "content": f"reasoning language: {lang_name}\nanswer language: {lang_name}",
        },
        {"role": "user", "content": user_prompt},
    ]

    return {
        "prompt": prompt_messages,
        "label": lang_code,
    }


print("Formatting dataset for OpenRLHF ...")
formatted = dataset["train"].map(
    make_openrlhf_row,
    remove_columns=dataset["train"].column_names,
)

# Split: first 20 rows as test, rest as train
total = len(formatted)
eval_size = min(20, total)
test_dataset = formatted.select(range(eval_size))
train_dataset = formatted.select(range(eval_size, total))


def save_jsonl(dataset, path):
    """Save dataset as JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for row in dataset:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


train_path = os.path.join(data_dir, "train.jsonl")
test_path = os.path.join(data_dir, "test.jsonl")

save_jsonl(train_dataset, train_path)
save_jsonl(test_dataset, test_path)

print(f"\nSaved {len(train_dataset)} train rows -> {train_path}")
print(f"Saved {len(test_dataset)} test rows  -> {test_path}")

# Verify format
with open(train_path) as f:
    sample = json.loads(f.readline())
print(f"\nSample row:")
print(f"  prompt: {json.dumps(sample['prompt'], indent=2)}")
print(f"  label:  {sample['label']}")

print("\nDone!")
PYEOF

# Copy and execute on the head pod
echo "Copying script to head pod ..."
kubectl cp /tmp/prepare_openrlhf_data.py \
    "${HEAD_POD}":/tmp/prepare_openrlhf_data.py

echo "Running data preparation ..."
kubectl exec "${HEAD_POD}" -- bash -c \
    "export DATA_DIR=${DATA_DIR} && python3 /tmp/prepare_openrlhf_data.py"

echo "Verifying files ..."
kubectl exec "${HEAD_POD}" -- ls -lh "${DATA_DIR}/"

echo ""
echo "Data preparation complete!"
echo "  Train: ${DATA_DIR}/train.jsonl"
echo "  Test:  ${DATA_DIR}/test.jsonl"
