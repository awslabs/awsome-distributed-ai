#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
set -xeuo pipefail

# ---------------------------------------------------------------------------
# Prepare HuggingFaceH4/Multilingual-Thinking dataset as veRL parquet
#
# This script runs a Python snippet on the Ray head pod to download the
# dataset and format it into the parquet schema veRL expects:
#
#   data_source  (str)       – dataset identifier for reward routing
#   prompt       (list[dict])– chat-template messages (system + user)
#   ability      (str)       – task category
#   reward_model (dict)      – {"style": "rule", "ground_truth": "<lang_code>"}
#   extra_info   (dict)      – optional metadata
#
# The ground_truth is the **expected language code** (en/fr/de/es/it),
# detected from the question text, which the custom reward function uses
# to score language compliance.
# ---------------------------------------------------------------------------

DATA_DIR="${RAY_DATA_HOME}/data/multilingual-thinking"
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
cat > /tmp/download_multilingual_thinking.py << 'PYEOF'
import os, sys, re
from datasets import load_dataset

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
    print("WARNING: langdetect not available – falling back to English")

data_dir = os.environ.get("DATA_DIR", "/fsx/verl/data/multilingual-thinking")
os.makedirs(data_dir, exist_ok=True)

LANG_CODE_MAP = {
    "English": "en", "French": "fr", "German": "de",
    "Spanish": "es", "Italian": "it",
}
SUPPORTED_LANGUAGES = list(LANG_CODE_MAP.keys())

def detect_question_language(text):
    """Return the language **name** for a question string."""
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

print("Loading HuggingFaceH4/Multilingual-Thinking …")
dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking")

def make_verl_row(example):
    """Convert a Multilingual-Thinking example into veRL parquet format."""
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

    # Build the prompt as chat-template messages.
    # The system prompt instructs the model on the reasoning AND answer language.
    prompt_messages = [
        {
            "role": "system",
            "content": f"reasoning language: {lang_name}\nanswer language: {lang_name}",
        },
        {"role": "user", "content": user_prompt},
    ]

    return {
        "data_source": "multilingual-thinking",
        "prompt": prompt_messages,
        "ability": "language",
        "reward_model": {"style": "rule", "ground_truth": lang_code},
        "extra_info": {"language_name": lang_name},
    }

print("Formatting dataset for veRL …")
formatted = dataset["train"].map(
    make_verl_row,
    remove_columns=dataset["train"].column_names,
)

# Split: first 20 rows as test, rest as train
total = len(formatted)
eval_size = min(20, total)
test_dataset  = formatted.select(range(eval_size))
train_dataset = formatted.select(range(eval_size, total))

train_path = os.path.join(data_dir, "train.parquet")
test_path  = os.path.join(data_dir, "test.parquet")

train_dataset.to_parquet(train_path)
test_dataset.to_parquet(test_path)

print(f"\nSaved {len(train_dataset)} train rows -> {train_path}")
print(f"Saved {len(test_dataset)} test rows  -> {test_path}")

sample = train_dataset[0]
print(f"\nSample row:")
print(f"  data_source : {sample['data_source']}")
print(f"  prompt      : {sample['prompt']}")
print(f"  ability     : {sample['ability']}")
print(f"  reward_model: {sample['reward_model']}")
print(f"  extra_info  : {sample['extra_info']}")

print("\nDone!")
PYEOF

# Copy and execute on the head pod
echo "Copying script to head pod …"
kubectl cp /tmp/download_multilingual_thinking.py \
    "${HEAD_POD}":/tmp/download_multilingual_thinking.py

echo "Running data preparation …"
kubectl exec "${HEAD_POD}" -- bash -c \
    "export DATA_DIR=${DATA_DIR} && python3 /tmp/download_multilingual_thinking.py"

echo "Verifying files …"
kubectl exec "${HEAD_POD}" -- ls -lh "${DATA_DIR}/"

echo ""
echo "Data preparation complete!"
echo "  Train: ${DATA_DIR}/train.parquet"
echo "  Test:  ${DATA_DIR}/test.parquet"
