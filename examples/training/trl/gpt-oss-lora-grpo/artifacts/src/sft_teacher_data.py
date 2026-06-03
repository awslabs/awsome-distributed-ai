# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
SFT Training on Teacher-Generated Data

Fine-tunes the student model (gpt-oss-20b) on teacher-generated data
before running GRPO. This ensures the student can generate positive samples.

Usage:
    python sft_teacher_data.py --data_path /fsx/teacher_data/teacher_data_sft_latest.jsonl
"""

import os
import json
import torch
import argparse
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Mxfp4Config
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer


def load_teacher_data(data_path: str) -> Dataset:
    """Load teacher-generated data from JSONL file."""
    
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            records.append(record)
    
    print(f"Loaded {len(records)} examples from {data_path}")
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(records)
    return dataset


def main():
    parser = argparse.ArgumentParser(description="SFT on teacher data")
    parser.add_argument("--model_name", type=str, default="openai/gpt-oss-20b",
                        help="Student model to fine-tune")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to teacher-generated SFT data (JSONL)")
    parser.add_argument("--output_dir", type=str, default="/fsx/checkpoints/sft-teacher",
                        help="Output directory for checkpoints")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha")
    
    args = parser.parse_args()
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_rank = int(os.environ.get("RANK", 0))
    
    if world_rank == 0:
        print("=" * 60)
        print("SFT TRAINING ON TEACHER DATA")
        print("=" * 60)
        print(f"Student Model: {args.model_name}")
        print(f"Data Path: {args.data_path}")
        print(f"Output: {args.output_dir}")
        print(f"Epochs: {args.num_epochs}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Learning Rate: {args.learning_rate}")
        print("=" * 60)
    
    # Load dataset
    dataset = load_teacher_data(args.data_path)
    
    if world_rank == 0:
        print(f"Dataset size: {len(dataset)}")
    
    # Load model
    quantization_config = Mxfp4Config(dequantize=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,  # Required: GPT-OSS model uses custom code on HF Hub
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
        device_map="auto",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    
    if world_rank == 0:
        model.print_trainable_parameters()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        gradient_checkpointing=True,
        report_to=[],
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        max_seq_length=args.max_seq_length,
    )
    
    if world_rank == 0:
        print("Starting SFT training on teacher data...")
    
    trainer.train()
    
    # Save final model
    if world_rank == 0:
        print(f"Saving model to {args.output_dir}")
    
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    if world_rank == 0:
        print("SFT training complete!")
        print(f"Checkpoint saved to: {args.output_dir}")
        print("\nNext step: Run GRPO training with this checkpoint:")
        print(f"  python grpo_singlenode.py --peft_checkpoint {args.output_dir}")


if __name__ == "__main__":
    main()
