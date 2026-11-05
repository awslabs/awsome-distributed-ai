"""
Step 0: build the medical-reasoning SFT dataset and write it where the training job reads it.

    python 0.prepare_dataset.py --config args-s3.yaml   # S3 variant: <s3_base_uri>/datasets/{train,val}/
    python 0.prepare_dataset.py --config args.yaml      # FSx variant: train_dataset_path / val_dataset_path

Each record keeps one column, "text": the model's chat template applied to a system prompt,
the question, and the answer with its chain of thought in <think> tags. SFTTrainer trains on it.
Run it from the Studio space attached to the Ray cluster.
"""

import argparse
import os
import tempfile
from urllib.parse import urlparse

# The SageMaker Distribution image ships TensorFlow + Keras 3, which transformers 4.x cannot
# import. Turn TF off before transformers is imported.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import yaml

DATASET_ID = "FreedomIntelligence/medical-o1-reasoning-SFT"
DATASET_CONFIG = "en"
# Pinned dataset commit, so every run trains on the same records.
DATASET_REVISION = "fc2c9e8a37b38f38da6d449564a8c350b244aef4"

SYSTEM_PROMPT = """You are a medical reasoning assistant. Respond to user queries
with clear, step-by-step medical reasoning."""


def build_splits(model_id: str, num_samples: int, val_fraction: float, seed: int):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dataset = load_dataset(
        DATASET_ID, DATASET_CONFIG, split=f"train[:{num_samples}]", revision=DATASET_REVISION
    )

    def format_chat(example):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["Question"]},
            {
                "role": "assistant",
                "content": f"<think>\n{example['Complex_CoT']}\n</think>\n\n{example['Response']}",
            },
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    dataset = dataset.map(format_chat, remove_columns=dataset.column_names)
    splits = dataset.train_test_split(test_size=val_fraction, seed=seed)
    return splits["train"], splits["test"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="args-s3.yaml (S3 variant) or args.yaml (FSx variant)")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    script_args = config["script_args"]

    train, val = build_splits(script_args["model_id"], args.num_samples, args.val_fraction, args.seed)
    print(f"Train: {len(train)}, Val: {len(val)}")

    if "s3_base_uri" in config:
        import boto3

        s3_base_uri = config["s3_base_uri"].rstrip("/")
        if "<" in s3_base_uri:
            raise SystemExit(f"Set s3_base_uri in {args.config} to your own bucket and prefix.")
        parsed = urlparse(s3_base_uri)
        bucket, prefix = parsed.netloc, parsed.path.lstrip("/")
        s3 = boto3.client("s3")
        with tempfile.TemporaryDirectory() as tmp:
            for split, ds in (("train", train), ("val", val)):
                local = os.path.join(tmp, f"{split}.json")
                ds.to_json(local)
                key = f"{prefix}/datasets/{split}/{split}.json"
                s3.upload_file(local, bucket, key)
                print(f"Uploaded s3://{bucket}/{key}")
    else:
        for split, ds, out_dir in (
            ("train", train, script_args["train_dataset_path"]),
            ("val", val, script_args["val_dataset_path"]),
        ):
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"{split}.json")
            ds.to_json(path)
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
