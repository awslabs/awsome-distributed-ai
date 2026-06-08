# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Merge LoRA adapter into base model for downstream use.

After SFT training, merge the LoRA checkpoint into the base model
so GRPO can use it as the starting point (Arm D: SFT+GRPO).

Usage:
    python merge_lora_checkpoint.py \
        --base_model Qwen/Qwen2.5-7B-Instruct \
        --adapter_path /fsx/qwen-grpo/checkpoints/multilingual/sft \
        --output_path /fsx/qwen-grpo/models/qwen-sft-multilingual-merged
"""

import argparse
import os
import sys

import torch


def find_adapter_path(checkpoint_dir: str) -> str:
    """Find the latest adapter checkpoint within a training output dir."""
    # Direct adapter files
    if os.path.exists(os.path.join(checkpoint_dir, "adapter_config.json")):
        return checkpoint_dir

    # Look for global_step_* directories
    step_dirs = []
    for d in os.listdir(checkpoint_dir):
        full = os.path.join(checkpoint_dir, d)
        if os.path.isdir(full) and d.startswith("global_step_"):
            step_dirs.append(full)

    if step_dirs:
        # Sort by step number, take latest
        step_dirs.sort(key=lambda x: int(x.split("_")[-1]))
        latest = step_dirs[-1]
        if os.path.exists(os.path.join(latest, "adapter_config.json")):
            return latest
        # Check subdirectories
        for sub in os.listdir(latest):
            sub_path = os.path.join(latest, sub)
            if os.path.isdir(sub_path) and os.path.exists(
                os.path.join(sub_path, "adapter_config.json")
            ):
                return sub_path

    # Look for any adapter_config.json recursively
    for root, dirs, files in os.walk(checkpoint_dir):
        if "adapter_config.json" in files:
            return root

    return checkpoint_dir  # Hope for the best


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter_path", type=str, required=True,
                        help="Path to LoRA checkpoint directory")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output path for merged model")
    parser.add_argument("--push_to_hub", action="store_true", default=False)
    parser.add_argument("--hub_name", type=str, default=None)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Find actual adapter path
    adapter_path = find_adapter_path(args.adapter_path)
    print(f"Base model:   {args.base_model}")
    print(f"Adapter path: {adapter_path}")
    print(f"Output path:  {args.output_path}")

    if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        print(f"ERROR: No adapter_config.json found in {adapter_path}")
        sys.exit(1)

    # Load base model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Merge on CPU to avoid OOM
        trust_remote_code=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and merge adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging weights...")
    model = model.merge_and_unload()

    # Save merged model
    print(f"Saving merged model to {args.output_path}...")
    os.makedirs(args.output_path, exist_ok=True)
    model.save_pretrained(args.output_path, safe_serialization=True)
    tokenizer.save_pretrained(args.output_path)

    print(f"Merged model saved: {args.output_path}")

    # Verify
    files = os.listdir(args.output_path)
    safetensors = [f for f in files if f.endswith(".safetensors")]
    print(f"  Files: {len(files)} total, {len(safetensors)} safetensors shards")

    if args.push_to_hub and args.hub_name:
        print(f"Pushing to hub: {args.hub_name}")
        model.push_to_hub(args.hub_name)
        tokenizer.push_to_hub(args.hub_name)


if __name__ == "__main__":
    main()
