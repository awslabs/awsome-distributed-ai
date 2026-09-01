# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
GRPO Single-Node Training Script with DDP

Fine-tunes GPT-OSS 20B to improve final answer language matching.
Uses DDP for multi-node distributed training with device_map="auto" for model sharding.

Key features:
- K=8 generations per prompt
- Language-aware reward function (higher penalty for wrong answer language)
- Test prompt generation every 2 steps for monitoring
- Checkpoint every epoch (10 epochs total)
- Starts from converted-peft/lora-checkpoint-1000-peft
- device_map="auto" spreads 20B model across 4 GPUs per node
- DDP syncs gradients between nodes (1 process per node)

Usage (multi-node):
    torchrun --nnodes=4 --nproc_per_node=1 grpo_singlenode.py

Usage (single node):
    python grpo_singlenode.py
"""

import os
import re
import json
import torch
import argparse
from datetime import datetime
from typing import List, Dict, Any
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, GenerationConfig
from peft import PeftModel, LoraConfig, get_peft_model

# Language detection
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
    print("WARNING: langdetect not installed")


LANG_CODE_MAP = {
    "English": "en",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Italian": "it",
}

SUPPORTED_LANGUAGES = list(LANG_CODE_MAP.keys())

# Test prompts for monitoring (Spanish and Italian)
TEST_PROMPTS = [
    ("Cuántos minutos tiene una semana?", "Spanish"),  # How many minutes are in a week?
    ("Qual è più lungo: un miglio o un chilometro?", "Italian"),  # Which is longer: a mile or a kilometer?
]


def extract_reasoning(response: str) -> str:
    """Extract reasoning section from model response."""
    match = re.search(r'assistantanalysis\s*(.*?)\s*assistantfinal', response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return response[:500] if len(response) > 500 else response


def extract_final_answer(response: str) -> str:
    """Extract final answer section."""
    match = re.search(r'assistantfinal\s*(.*)$', response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return response[-200:] if len(response) > 200 else response


def detect_language(text: str) -> str:
    """Detect language of text."""
    if not HAS_LANGDETECT:
        return "unknown"
    try:
        clean_text = re.sub(r'[0-9\+\-\*\/\=\%\$\€\£]', '', text)
        clean_text = re.sub(r'[^\w\s]', ' ', clean_text)
        if len(clean_text.strip()) < 20:
            return "too_short"
        return detect(clean_text)
    except Exception:
        return "error"


def count_sentences(text: str) -> int:
    """Count number of sentences in text."""
    # Split by sentence-ending punctuation
    sentences = re.split(r'[.!?]+', text.strip())
    # Filter out empty strings
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences)


def language_reward_fn(completions: List[str], prompts: List[str] = None, **kwargs) -> List[float]:
    """
    Simplified language-aware reward function.
    
    Priority order:
    1. Answer language (most important): +5.0 / -5.0
    2. Reasoning language: +1.5 / -1.5
    3. Final answer brevity (≤2 sentences): +0.5 / -1.0
    
    Max score: +7.0, Min score: -7.5
    """
    rewards = []
    
    for i, completion in enumerate(completions):
        reward = 0.0
        reasoning = extract_reasoning(completion)
        final_answer = extract_final_answer(completion)
        reasoning_lang = detect_language(reasoning)
        output_lang = detect_language(final_answer)
        
        # Determine expected language from prompt
        expected_code = "en"
        if prompts and i < len(prompts):
            prompt = prompts[i].lower()
            for lang_name, lang_code in LANG_CODE_MAP.items():
                if f"reasoning language: {lang_name.lower()}" in prompt:
                    expected_code = lang_code
                    break
        
        # === 1. ANSWER LANGUAGE (most important - 70% weight) ===
        if output_lang == expected_code:
            reward += 5.0
        else:
            reward -= 5.0
        
        # === 2. REASONING LANGUAGE (20% weight) ===
        if reasoning_lang == expected_code:
            reward += 1.5
        else:
            reward -= 1.5
        
        # === 3. FINAL ANSWER BREVITY (≤2 sentences - 10% weight) ===
        answer_sentences = count_sentences(final_answer)
        if answer_sentences <= 2:
            reward += 0.5
        else:
            reward -= 1.0
        
        rewards.append(reward)
    
    return rewards


def detect_question_language(text: str) -> str:
    """Detect language of question text and return language name."""
    if not HAS_LANGDETECT:
        return "English"
    try:
        clean_text = re.sub(r'[0-9\+\-\*\/\=\%\$\€\£]', '', text)
        clean_text = re.sub(r'[^\w\s]', ' ', clean_text)
        if len(clean_text.strip()) < 10:
            return "English"
        detected_code = detect(clean_text)
        code_to_name = {v: k for k, v in LANG_CODE_MAP.items()}
        return code_to_name.get(detected_code, "English")
    except Exception:
        return "English"



class TestPromptCallback(TrainerCallback):
    """Callback to generate test prompt output every 10 steps."""
    
    def __init__(self, tokenizer, model, output_dir):
        self.tokenizer = tokenizer
        self.model = model
        self.output_dir = output_dir
        self.test_log_path = os.path.join(output_dir, "test_outputs.txt")
        
        # Only rank 0 writes header
        world_rank = int(os.environ.get("RANK", 0))
        if world_rank == 0:
            os.makedirs(output_dir, exist_ok=True)
            with open(self.test_log_path, "w") as f:
                f.write("=" * 80 + "\n")
                f.write("GRPO Test Prompt Outputs\n")
                f.write(f"Test Prompts:\n")
                for prompt, lang in TEST_PROMPTS:
                    f.write(f"  - [{lang}] {prompt}\n")
                f.write(f"Started: {datetime.now().isoformat()}\n")
                f.write("=" * 80 + "\n\n")
    
    def on_step_end(self, args, state, control, **kwargs):
        """Generate test output after step 1, then every 10 steps."""
        if state.global_step == 1 or (state.global_step % 10 == 0 and state.global_step > 0):
            self._generate_test_output(state.global_step)
        return control
    
    def _generate_test_output(self, step: int):
        """Generate and log test prompt output for all test prompts."""
        world_rank = int(os.environ.get("RANK", 0))
        if world_rank != 0:
            return
        
        self.model.eval()
        
        for test_prompt, test_language in TEST_PROMPTS:
            self._evaluate_single_prompt(step, test_prompt, test_language)
        
        self.model.train()
    
    def _evaluate_single_prompt(self, step: int, test_prompt: str, test_language: str):
        """Evaluate a single test prompt."""
        # Build prompt with chat template
        messages = [
            {"role": "system", "content": f"reasoning language: {test_language}\nanswer language: {test_language}"},
            {"role": "user", "content": test_prompt}
        ]
        
        try:
            chat_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            chat_prompt = f"System: reasoning language: {test_language}\nanswer language: {test_language}\nUser: {test_prompt}\nAssistant:"
        
        try:
            with torch.no_grad():
                inputs = self.tokenizer(chat_prompt, return_tensors="pt", truncation=True, max_length=1024)
                
                # Handle device placement
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                generation_config = GenerationConfig(
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
                
                outputs = self.model.generate(**inputs, generation_config=generation_config)
                completion = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:], 
                    skip_special_tokens=True
                )
        except Exception as e:
            completion = f"[Generation error: {str(e)}]"
        
        # Analyze output
        reasoning = extract_reasoning(completion)
        final_answer = extract_final_answer(completion)
        reasoning_lang = detect_language(reasoning)
        output_lang = detect_language(final_answer)
        
        expected_code = LANG_CODE_MAP.get(test_language, "en")
        reasoning_correct = reasoning_lang == expected_code
        answer_correct = output_lang == expected_code
        
        # Calculate reward for this test output
        reward = 0.0
        if output_lang == expected_code:
            reward += 5.0
        else:
            reward -= 5.0
        if reasoning_lang == expected_code:
            reward += 1.5
        else:
            reward -= 1.5
        answer_sentences = count_sentences(final_answer)
        if answer_sentences <= 2:
            reward += 0.5
        else:
            reward -= 1.0
        
        # Log to file
        try:
            with open(self.test_log_path, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"STEP {step} | {datetime.now().strftime('%H:%M:%S')} | {test_language}\n")
                f.write(f"Question: {test_prompt}\n")
                f.write(f"{'='*60}\n")
                f.write(f"Reasoning Language: {reasoning_lang} ({'OK' if reasoning_correct else 'WRONG'})\n")
                f.write(f"Answer Language: {output_lang} ({'OK' if answer_correct else 'WRONG'})\n")
                f.write(f"Total Reward: {reward}\n")
                f.write(f"\nFULL OUTPUT:\n{'-'*40}\n")
                f.write(completion)
                f.write(f"\n{'-'*40}\n")
                f.write(f"\nREASONING SECTION:\n{reasoning[:1000]}\n")
                f.write(f"\nFINAL ANSWER SECTION:\n{final_answer}\n")
        except Exception as e:
            print(f"[Step {step}] Error writing test log: {e}")
        
        # Print full output to console for verification
        print(f"\n{'='*60}")
        print(f"[Step {step}] TEST OUTPUT | {datetime.now().strftime('%H:%M:%S')} | {test_language}")
        print(f"Question: {test_prompt}")
        print(f"{'='*60}")
        print(f"Reasoning Language: {reasoning_lang} ({'OK' if reasoning_correct else 'WRONG'})")
        print(f"Answer Language: {output_lang} ({'OK' if answer_correct else 'WRONG'})")
        print(f"Total Reward: {reward}")
        print(f"\nFULL OUTPUT:\n{'-'*40}")
        print(completion)
        print(f"{'-'*40}")
        print(f"\nFINAL ANSWER:\n{final_answer}")
        print(f"{'='*60}\n")


def prepare_dataset(dataset_name: str, tokenizer, shard_idx: int = 0, num_shards: int = 1, eval_size: int = 20):
    """Prepare dataset with question language detection and optional sharding."""
    dataset = load_dataset(dataset_name)
    
    def format_prompt(example):
        messages = example.get("messages", [])
        user_prompt = ""
        
        for msg in messages:
            if msg["role"] == "user":
                user_prompt = msg["content"]
                break
        
        # Detect question language
        detected_language = detect_question_language(user_prompt)
        if detected_language not in SUPPORTED_LANGUAGES:
            detected_language = "English"
        
        # Build chat messages - include BOTH reasoning and answer language
        chat_messages = [
            {"role": "system", "content": f"reasoning language: {detected_language}\nanswer language: {detected_language}"},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            formatted_prompt = f"System: reasoning language: {detected_language}\nanswer language: {detected_language}\nUser: {user_prompt}\nAssistant:"
        
        return {"prompt": formatted_prompt}
    
    print("Processing dataset...")
    formatted_dataset = dataset["train"].map(
        format_prompt,
        remove_columns=dataset["train"].column_names,
    )
    
    # Split into train and eval
    total_size = len(formatted_dataset)
    eval_indices = list(range(0, min(eval_size, total_size)))
    train_indices = list(range(eval_size, total_size))
    
    eval_dataset = formatted_dataset.select(eval_indices)
    train_dataset = formatted_dataset.select(train_indices)
    
    # Shard training dataset if running in parallel mode
    if num_shards > 1:
        train_dataset = train_dataset.shard(num_shards=num_shards, index=shard_idx)
        print(f"Using shard {shard_idx}/{num_shards}, train size: {len(train_dataset)}")
    
    print(f"Eval dataset size: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--peft_checkpoint", type=str, default="/fsx/checkpoints/converted-peft/lora-checkpoint-1000-peft")
    parser.add_argument("--output_dir", type=str, default="/fsx/checkpoints/grpo-multi")
    parser.add_argument("--num_generations", type=int, default=8, help="K=8 generations per prompt")
    parser.add_argument("--num_epochs", type=int, default=10, help="10 full cycles over dataset")
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--eval_size", type=int, default=20)
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    
    args = parser.parse_args()
    
    # Get distributed info
    world_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    
    is_main = world_rank == 0
    
    if is_main:
        print("="*60)
        print("GRPO Multi-Node Training with DDP")
        print("="*60)
        print(f"Model: {args.model_name_or_path}")
        print(f"PEFT Checkpoint: {args.peft_checkpoint}")
        print(f"Output: {args.output_dir}")
        print(f"K (num_generations): {args.num_generations}")
        print(f"Epochs: {args.num_epochs}")
        print(f"World Size: {world_size}")
        print(f"Test Prompts: {len(TEST_PROMPTS)} prompts")
        print("="*60)
    
    # Validate checkpoint exists
    if not os.path.exists(args.peft_checkpoint):
        print(f"ERROR: PEFT checkpoint not found: {args.peft_checkpoint}")
        return
    
    adapter_safetensors = os.path.join(args.peft_checkpoint, "adapter_model.safetensors")
    adapter_bin = os.path.join(args.peft_checkpoint, "adapter_model.bin")
    if not os.path.exists(adapter_safetensors) and not os.path.exists(adapter_bin):
        print(f"ERROR: No adapter weights found in: {args.peft_checkpoint}")
        return
    
    if is_main:
        print(f"PEFT checkpoint validated: {args.peft_checkpoint}")
    
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        print(f"ERROR: {e}")
        print("GRPOTrainer requires TRL >= 0.14.0")
        return
    
    # Load model
    # Use device_map="auto" to spread 20B model across 4 GPUs within this node
    # This works for both single-node and multi-node (1 process per node)
    if is_main:
        print("Loading base model with device_map='auto'...")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,  # Required: GPT-OSS model uses custom code on HF Hub
        low_cpu_mem_usage=True,
        device_map="auto",  # Spread across 4 GPUs within this node
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load and merge PEFT checkpoint (SFT weights)
    if is_main:
        print(f"Loading PEFT checkpoint: {args.peft_checkpoint}")
    model = PeftModel.from_pretrained(model, args.peft_checkpoint)
    model = model.merge_and_unload()
    if is_main:
        print("SFT weights merged into base model!")
    
    # Add new LoRA for GRPO training
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    
    if is_main:
        model.print_trainable_parameters()
        print("Preparing dataset...")
    
    train_dataset, eval_dataset = prepare_dataset(
        "HuggingFaceH4/Multilingual-Thinking",
        tokenizer,
        eval_size=args.eval_size
    )
    
    # Calculate steps per epoch for checkpointing
    dataset_size = len(train_dataset)
    effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
    steps_per_epoch = max(1, dataset_size // effective_batch)
    
    if is_main:
        print(f"Train dataset size: {dataset_size}")
        print(f"World size (GPUs): {world_size}")
        print(f"Effective batch size: {effective_batch}")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Total steps for {args.num_epochs} epochs: {steps_per_epoch * args.num_epochs}")
    
    # GRPO config
    # Using DDP with device_map="auto" - each node has 1 process with model spread across 4 GPUs
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        logging_steps=1,
        save_steps=steps_per_epoch,
        save_total_limit=args.num_epochs + 2,
        num_generations=args.num_generations,
        max_completion_length=512,
        temperature=0.7,
        bf16=True,
        gradient_checkpointing=True,
        report_to=[],
        ddp_find_unused_parameters=False,
    )
    
    # Create test prompt callback
    test_callback = TestPromptCallback(
        tokenizer=tokenizer,
        model=model,
        output_dir=args.output_dir
    )
    
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=language_reward_fn,
        callbacks=[test_callback],
    )
    
    if is_main:
        print("Starting GRPO training...")
    
    trainer.train()
    
    if is_main:
        print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    if is_main:
        print("GRPO training complete!")
        print(f"Test outputs saved to: {args.output_dir}/test_outputs.txt")


if __name__ == "__main__":
    main()
