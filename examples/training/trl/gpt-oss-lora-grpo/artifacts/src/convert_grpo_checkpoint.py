# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Convert GRPO checkpoint to PEFT format for inference.

The GRPO checkpoints save LoRA adapters in safetensors format.
This script copies the checkpoint and ensures compatibility with inference_g6e.py.

Usage:
    python convert_grpo_checkpoint.py --checkpoint /fsx/checkpoints/grpo-node-0/checkpoint-1470
    python convert_grpo_checkpoint.py --checkpoint /fsx/checkpoints/grpo-node-0/checkpoint-1470 --output /fsx/checkpoints/checkpoint-1470-peft
"""

import os
import shutil
import argparse
import json
from safetensors.torch import load_file, save_file
import torch


def convert_checkpoint(checkpoint_path: str, output_path: str = None):
    """Convert GRPO checkpoint to PEFT format for inference."""
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return False
    
    # Default output path
    if output_path is None:
        checkpoint_name = os.path.basename(checkpoint_path)
        parent_dir = os.path.dirname(os.path.dirname(checkpoint_path))
        output_path = os.path.join(parent_dir, f"{checkpoint_name}-peft")
    
    print(f"Converting checkpoint: {checkpoint_path}")
    print(f"Output path: {output_path}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Check for adapter files
    safetensors_path = os.path.join(checkpoint_path, "adapter_model.safetensors")
    bin_path = os.path.join(checkpoint_path, "adapter_model.bin")
    config_path = os.path.join(checkpoint_path, "adapter_config.json")
    
    if not os.path.exists(config_path):
        print(f"ERROR: adapter_config.json not found in {checkpoint_path}")
        return False
    
    # Copy adapter config
    shutil.copy(config_path, os.path.join(output_path, "adapter_config.json"))
    print("✓ Copied adapter_config.json")
    
    # Handle adapter weights
    if os.path.exists(safetensors_path):
        # Load safetensors and save as both formats for compatibility
        print(f"Loading adapter weights from safetensors...")
        state_dict = load_file(safetensors_path)
        
        # Save as safetensors (primary)
        save_file(state_dict, os.path.join(output_path, "adapter_model.safetensors"))
        print("✓ Saved adapter_model.safetensors")
        
        # Also save as .bin for older PEFT versions
        torch.save(state_dict, os.path.join(output_path, "adapter_model.bin"))
        print("✓ Saved adapter_model.bin (for compatibility)")
        
    elif os.path.exists(bin_path):
        # Copy existing bin file
        shutil.copy(bin_path, os.path.join(output_path, "adapter_model.bin"))
        print("✓ Copied adapter_model.bin")
    else:
        print(f"ERROR: No adapter weights found in {checkpoint_path}")
        return False
    
    # Copy tokenizer files if present
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json", 
        "special_tokens_map.json",
        "chat_template.jinja"
    ]
    
    for tf in tokenizer_files:
        src = os.path.join(checkpoint_path, tf)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(output_path, tf))
            print(f"✓ Copied {tf}")
    
    # Copy README if present
    readme_path = os.path.join(checkpoint_path, "README.md")
    if os.path.exists(readme_path):
        shutil.copy(readme_path, os.path.join(output_path, "README.md"))
        print("✓ Copied README.md")
    
    print(f"\n✅ Conversion complete!")
    print(f"Output: {output_path}")
    print(f"\nTo use with inference_g6e.py:")
    print(f"  python inference_g6e.py --use_trained --checkpoint_dir {os.path.dirname(output_path)}")
    
    return True


def find_latest_checkpoint(base_dir: str = "/fsx/checkpoints/grpo-node-0"):
    """Find the latest checkpoint in the directory."""
    if not os.path.exists(base_dir):
        return None
    
    checkpoints = []
    for item in os.listdir(base_dir):
        if item.startswith("checkpoint-") and not item.endswith("-peft"):
            checkpoint_path = os.path.join(base_dir, item)
            if os.path.isdir(checkpoint_path):
                try:
                    step = int(item.split("-")[-1])
                    checkpoints.append((step, checkpoint_path))
                except ValueError:
                    continue
    
    if not checkpoints:
        return None
    
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]


def main():
    parser = argparse.ArgumentParser(description="Convert GRPO checkpoint for inference")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to GRPO checkpoint (default: latest in grpo-node-0)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: <checkpoint>-peft in parent dir)")
    parser.add_argument("--grpo_dir", type=str, default="/fsx/checkpoints/grpo-node-0",
                        help="GRPO checkpoints directory")
    
    args = parser.parse_args()
    
    # Find checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(args.grpo_dir)
        if checkpoint_path is None:
            print(f"ERROR: No checkpoints found in {args.grpo_dir}")
            return
        print(f"Found latest checkpoint: {checkpoint_path}")
    
    # Convert
    success = convert_checkpoint(checkpoint_path, args.output)
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main()
