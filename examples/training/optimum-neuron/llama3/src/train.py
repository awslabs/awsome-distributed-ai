# Adapted from the official optimum-neuron training example:
# https://github.com/huggingface/optimum-neuron/blob/main/examples/training/llama/finetune_llama.py
#
# This script extends the upstream example with:
#   - Support for local model paths (for Slurm/K8s with shared storage)
#   - Configurable final model save path
#   - Configurable LoRA parameters via CLI

from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, HfArgumentParser

from optimum.neuron import NeuronSFTConfig, NeuronSFTTrainer, NeuronTrainingArguments
from optimum.neuron.models.training import NeuronModelForCausalLM


# =============================================================================
# Data Formatting
# =============================================================================


def format_dolly(example, tokenizer):
    """Format Dolly dataset examples using the tokenizer's chat template."""
    user_content = example["instruction"]
    if len(example["context"]) > 0:
        user_content += f"\n\nContext: {example['context']}"

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example["response"]},
    ]

    return tokenizer.apply_chat_template(messages, tokenize=False)


# =============================================================================
# Training
# =============================================================================


def train(model_id, tokenizer, dataset, training_args, script_args):
    trn_config = training_args.trn_config
    dtype = torch.bfloat16 if training_args.bf16 else torch.float32

    model = NeuronModelForCausalLM.from_pretrained(
        model_id,
        trn_config,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
    )

    lora_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_config = NeuronSFTConfig(
        max_length=script_args.max_seq_length,
        packing=True,
        **training_args.to_dict(),
    )

    trainer = NeuronSFTTrainer(
        args=sft_config,
        model=model,
        peft_config=lora_config,
        processing_class=tokenizer,
        train_dataset=dataset,
        formatting_func=lambda example: format_dolly(example, tokenizer),
    )

    trainer.train()

    if script_args.model_final_path:
        trainer.save_model(script_args.model_final_path)


# =============================================================================
# Script Arguments
# =============================================================================


@dataclass
class ScriptArguments:
    model_id: str = field(
        metadata={
            "help": "Model name on HuggingFace Hub, or path to a local model directory."
        },
    )
    dataset: str = field(
        default="databricks/databricks-dolly-15k",
        metadata={"help": "Dataset name on HuggingFace Hub."},
    )
    max_seq_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Must be a multiple of 2048 when using flash attention."
        },
    )
    model_final_path: str = field(
        default="",
        metadata={
            "help": "Path to save the final model after training (in addition to output_dir checkpoints)."
        },
    )
    lora_r: int = field(default=16, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, NeuronTrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(script_args.dataset, split="train")

    train(
        model_id=script_args.model_id,
        tokenizer=tokenizer,
        dataset=dataset,
        training_args=training_args,
        script_args=script_args,
    )
