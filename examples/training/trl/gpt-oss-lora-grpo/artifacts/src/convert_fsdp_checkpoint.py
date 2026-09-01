# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Convert FSDP distributed checkpoint to standard PEFT format.
Handles the .distcp sharded format from Accelerate FSDP training.

Usage:
    python convert_fsdp_checkpoint.py --checkpoint /fsx/checkpoints/checkpoint-200
"""

import argparse
import os
import json
import torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import Metadata
from tqdm import tqdm


def convert_checkpoint_v2(checkpoint_path: str, output_path: str = None):
    """Convert FSDP distributed checkpoint to standard PEFT format."""
    
    fsdp_path = os.path.join(checkpoint_path, "pytorch_model_fsdp_0")
    
    if not os.path.exists(fsdp_path):
        raise ValueError(f"FSDP checkpoint not found at {fsdp_path}")
    
    if output_path is None:
        output_path = checkpoint_path + "-peft"
    
    os.makedirs(output_path, exist_ok=True)
    
    print("="*60)
    print("FSDP to PEFT Checkpoint Converter")
    print("="*60)
    print(f"Input:  {fsdp_path}")
    print(f"Output: {output_path}")
    print("="*60)
    
    # Use Accelerate's load_fsdp_model approach
    print("\n[1/3] Loading FSDP checkpoint using torch.distributed.checkpoint...")
    
    from torch.distributed.checkpoint.state_dict_loader import load
    
    # Read metadata first
    reader = FileSystemReader(fsdp_path)
    metadata = reader.read_metadata()
    
    # Create state dict with proper tensor placeholders
    state_dict = {}
    for key, tensor_meta in tqdm(metadata.state_dict_metadata.items(), desc="Creating placeholders"):
        # Create empty tensor with correct shape and dtype
        state_dict[key] = torch.empty(tensor_meta.size, dtype=torch.bfloat16)
    
    print(f"Created {len(state_dict)} tensor placeholders")
    
    # Load the actual data
    print("\n[2/3] Loading tensor data from shards...")
    load(state_dict, checkpoint_id=fsdp_path)
    
    print(f"Loaded {len(state_dict)} tensors")
    
    # Transform keys to PEFT format
    print("\n[3/3] Saving PEFT adapter...")
    peft_state_dict = {}
    for key, value in tqdm(state_dict.items(), desc="Transforming keys"):
        # Remove the leading "model." prefix if present
        new_key = key
        if new_key.startswith("model."):
            new_key = new_key[6:]
        peft_state_dict[new_key] = value
    
    # Save adapter weights
    adapter_path = os.path.join(output_path, "adapter_model.bin")
    torch.save(peft_state_dict, adapter_path)
    print(f"Saved adapter weights to: {adapter_path}")
    
    # Create adapter_config.json
    adapter_config = {
        "base_model_name_or_path": "openai/gpt-oss-20b",
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": 8,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "task_type": "CAUSAL_LM",
    }
    
    config_path = os.path.join(output_path, "adapter_config.json")
    with open(config_path, 'w') as f:
        json.dump(adapter_config, f, indent=2)
    print(f"Saved adapter config to: {config_path}")
    
    # Show sample keys
    print("\nSample keys in adapter:")
    for key in list(peft_state_dict.keys())[:5]:
        print(f"  {key}: {peft_state_dict[key].shape}")
    
    print("\n" + "="*60)
    print("Conversion complete!")
    print(f"PEFT adapter saved to: {output_path}")
    print("="*60)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert FSDP checkpoint to PEFT format")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to FSDP checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: checkpoint-peft)")
    
    args = parser.parse_args()
    convert_checkpoint_v2(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
