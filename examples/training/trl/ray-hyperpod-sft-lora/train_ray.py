"""
Ray TorchTrainer on SageMaker HyperPod — Fine-tune LLM with FSDP + LoRA/QLoRA

Target: Ray cluster created via SageMaker Studio on HyperPod (EKS + KubeRay)
"""

import os
import yaml
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import ray
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, CheckpointConfig, FailureConfig

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset, load_from_disk
from transformers import TrainerCallback

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class ScriptArguments:
    model_id: str = field(default="Qwen/Qwen3-0.6B")
    train_dataset_path: str = field(default="/mnt/fsx/datasets/train")
    val_dataset_path: str = field(default="/mnt/fsx/datasets/val")
    checkpoint_dir: str = field(default="/mnt/fsx/checkpoints")
    output_dir: str = field(default="/mnt/fsx/output")
    token: Optional[str] = field(default=None)
    attn_implementation: str = field(default="flash_attention_2")
    apply_truncation: bool = field(default=True)
    auto_calculate_lengths: bool = field(default=True)
    deserialize_messages: bool = field(default=True)
    load_in_4bit: bool = field(default=False)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    merge_weights: bool = field(default=True)
    early_stopping: bool = field(default=False)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_lora_config(script_args: ScriptArguments) -> LoraConfig:
    return LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )


def get_quantization_config(script_args: ScriptArguments):
    if not script_args.load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


class RayTrainReportCallback(TrainerCallback):
    """Reports metrics and checkpoint path to Ray Train on each save."""

    def on_save(self, args, state, control, **kwargs):
        import ray.train
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        ray.train.report(
            metrics={"loss": state.log_history[-1].get("loss", 0), "step": state.global_step},
            checkpoint=ray.train.Checkpoint.from_directory(checkpoint_dir),
        )


def train_func(config: dict):
    """Per-worker training function executed inside Ray workers."""
    import ray.train

    script_args = ScriptArguments(**config["script_args"])
    sft_config = config["sft_config"]

    logger.info(f"Loading model: {script_args.model_id}")

    quant_config = get_quantization_config(script_args)

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_id,
        quantization_config=quant_config,
        attn_implementation=script_args.attn_implementation,
        torch_dtype=torch.bfloat16,
        token=script_args.token,
        trust_remote_code=True,
    )

    if script_args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = get_lora_config(script_args)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_id,
        token=script_args.token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading datasets from: {script_args.train_dataset_path}")
    try:
        train_dataset = load_from_disk(script_args.train_dataset_path)
    except Exception:
        train_dataset = load_dataset(
            "json",
            data_files=f"{script_args.train_dataset_path}/*.json",
            split="train",
        )

    val_dataset = None
    if script_args.val_dataset_path and os.path.exists(script_args.val_dataset_path):
        try:
            val_dataset = load_from_disk(script_args.val_dataset_path)
        except Exception:
            val_dataset = load_dataset(
                "json",
                data_files=f"{script_args.val_dataset_path}/*.json",
                split="train",
            )

    training_args = SFTConfig(
        output_dir=os.path.join(script_args.checkpoint_dir, "sft"),
        per_device_train_batch_size=sft_config.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=sft_config.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=sft_config.get("gradient_accumulation_steps", 2),
        num_train_epochs=sft_config.get("num_train_epochs", 2),
        learning_rate=sft_config.get("learning_rate", 2e-4),
        lr_scheduler_type=sft_config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=sft_config.get("warmup_ratio", 0.1),
        bf16=True,
        logging_steps=sft_config.get("logging_steps", 10),
        save_strategy="steps",
        save_steps=sft_config.get("save_steps", 100),
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=sft_config.get("eval_steps", 100) if val_dataset else None,
        max_seq_length=sft_config.get("max_seq_length", 2048),
        gradient_checkpointing=sft_config.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fsdp=sft_config.get("fsdp", "full_shard auto_wrap offload"),
        fsdp_config=sft_config.get("fsdp_config", {
            "backward_prefetch": "backward_pre",
            "forward_prefetch": True,
            "use_orig_params": True,
        }),
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        callbacks=[RayTrainReportCallback()],
    )

    train_result = trainer.train()

    metrics = train_result.metrics
    ray.train.report(metrics)

    final_path = os.path.join(script_args.output_dir, "final_model")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    if script_args.merge_weights and not script_args.load_in_4bit:
        logger.info("Merging LoRA weights into base model...")
        from peft import AutoPeftModelForCausalLM

        merged_model = AutoPeftModelForCausalLM.from_pretrained(
            final_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        merged_model = merged_model.merge_and_unload()
        merged_path = os.path.join(script_args.output_dir, "merged_model")
        merged_model.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
        logger.info(f"Merged model saved to: {merged_path}")

    logger.info("Training complete.")


def main(config_path: str = "args.yaml"):
    config = load_config(config_path)

    script_args = ScriptArguments(**config.get("script_args", {}))
    sft_config = config.get("sft_config", {})

    num_workers = config.get("num_workers", 8)
    num_gpus_per_worker = config.get("num_gpus_per_worker", 1)
    use_gpu = config.get("use_gpu", True)
    max_failures = config.get("max_failures", 3)

    os.makedirs(script_args.checkpoint_dir, exist_ok=True)
    os.makedirs(script_args.output_dir, exist_ok=True)

    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=use_gpu,
        resources_per_worker={"GPU": num_gpus_per_worker, "CPU": 8},
    )

    run_config = RunConfig(
        name="hyperpod-ray-finetune",
        storage_path="/mnt/fsx/ray_results",
        checkpoint_config=CheckpointConfig(
            num_to_keep=3,
        ),
        failure_config=FailureConfig(
            max_failures=max_failures,
        ),
    )

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={
            "script_args": script_args.__dict__,
            "sft_config": sft_config,
        },
        scaling_config=scaling_config,
        run_config=run_config,
    )

    result = trainer.fit()
    logger.info(f"Training result: {result}")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="args.yaml")
    args = parser.parse_args()
    main(config_path=args.config)
