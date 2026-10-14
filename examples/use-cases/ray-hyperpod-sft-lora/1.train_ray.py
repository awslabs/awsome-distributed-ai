"""
Ray TorchTrainer on SageMaker HyperPod — Fine-tune an LLM with LoRA (FSx variant)

Target: Ray cluster created via SageMaker Studio on HyperPod (EKS + KubeRay), with an
FSx for Lustre file system mounted at /mnt/fsx on the Studio space and on every Ray pod.

  - the dataset is read from train_dataset_path / val_dataset_path (written by 0.prepare_dataset.py)
  - Ray Train checkpoints go to <storage_path>/<run_name>
  - rank 0 writes the final LoRA adapter and the merged model to output_dir
A gated Hugging Face model needs HF_TOKEN in the pods' environment (see the README);
never put a token in the YAML config.

Heavy ML imports live inside train_func so the driver (Ray head, CPU image) only
needs ray and pyyaml.
"""

import os

# The SageMaker Distribution image ships TensorFlow 2.19 + Keras 3, which transformers 4.x
# cannot import. This is a PyTorch-only job, so turn TF off before transformers is imported.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import logging
from dataclasses import dataclass, field

import yaml
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, CheckpointConfig, FailureConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class ScriptArguments:
    model_id: str = field(default="Qwen/Qwen3-0.6B")
    train_dataset_path: str = field(default="/mnt/fsx/ray-sft-lora/datasets/train")
    val_dataset_path: str = field(default="/mnt/fsx/ray-sft-lora/datasets/val")
    checkpoint_dir: str = field(default="/mnt/fsx/ray-sft-lora/checkpoints")
    output_dir: str = field(default="/mnt/fsx/ray-sft-lora/output")
    attn_implementation: str = field(default="sdpa")
    load_in_4bit: bool = field(default=False)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    merge_weights: bool = field(default=True)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Training (runs inside each Ray Train worker)
# ---------------------------------------------------------------------------

def train_func(config: dict):
    """Per-worker training function executed inside Ray workers."""
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

    import torch
    import ray.train
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    script_args = ScriptArguments(**config["script_args"])
    sft_config = config["sft_config"]
    rank = ray.train.get_context().get_world_rank()

    # 1. Model + LoRA
    # Hugging Face reads HF_TOKEN from the environment when a model is gated.
    logger.info(f"Loading model: {script_args.model_id}")
    quant_config = None
    if script_args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_id,
        quantization_config=quant_config,
        attn_implementation=script_args.attn_implementation,
        torch_dtype=torch.bfloat16,
    )
    if script_args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    model = get_peft_model(
        model,
        LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Dataset: JSON files written by 0.prepare_dataset.py
    logger.info(f"Loading datasets from: {script_args.train_dataset_path}")
    train_dataset = load_dataset(
        "json",
        data_files=f"{script_args.train_dataset_path}/*.json",
        split="train",
    )
    val_dataset = None
    if script_args.val_dataset_path and os.path.isdir(script_args.val_dataset_path):
        val_dataset = load_dataset(
            "json",
            data_files=f"{script_args.val_dataset_path}/*.json",
            split="train",
        )

    # 3. Ray Train checkpoint reporting: rank 0 hands the HF checkpoint to Ray, which
    #    copies it to RunConfig.storage_path. All ranks must call report().
    class RayTrainReportCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            metrics = {"step": state.global_step}
            for entry in reversed(state.log_history):
                if "loss" in entry:
                    metrics["loss"] = entry["loss"]
                    break
            checkpoint = None
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if rank == 0 and os.path.isdir(ckpt_dir):
                checkpoint = ray.train.Checkpoint.from_directory(ckpt_dir)
            ray.train.report(metrics=metrics, checkpoint=checkpoint)

    # FSDP is only turned on when args.yaml sets it (not needed for 0.6B + LoRA).
    fsdp_kwargs = {}
    if sft_config.get("fsdp"):
        fsdp_kwargs = {"fsdp": sft_config["fsdp"], "fsdp_config": sft_config.get("fsdp_config", {})}

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
        save_total_limit=2,
        eval_strategy="steps" if val_dataset is not None else "no",
        eval_steps=sft_config.get("eval_steps", 100) if val_dataset is not None else None,
        max_length=sft_config.get("max_length", 2048),
        gradient_checkpointing=sft_config.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        ddp_find_unused_parameters=False,
        **fsdp_kwargs,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        callbacks=[RayTrainReportCallback()],
    )

    # After a worker or node failure, Ray Train hands the latest reported checkpoint
    # (a full HF checkpoint-N folder) to the restarted workers. Resume from it.
    resume_from = None
    checkpoint = ray.train.get_checkpoint()
    if checkpoint:
        resume_from = checkpoint.to_directory()
        if not os.path.isfile(os.path.join(resume_from, "trainer_state.json")):
            raise RuntimeError(
                f"run_name '{config['run_name']}' already holds a finished run. "
                "Set a new run_name in the YAML config."
            )
        logger.info(f"[rank {rank}] Resuming from checkpoint {resume_from}")

    train_result = trainer.train(resume_from_checkpoint=resume_from)
    metrics = dict(train_result.metrics)
    if val_dataset is not None:
        metrics.update(trainer.evaluate())

    # 4. Save the final adapter (+ merged model). Every rank calls save_model, because
    #    under FSDP it gathers the full state dict; only rank 0 writes the other files.
    final_path = os.path.join(script_args.output_dir, "final_model")
    trainer.save_model(final_path)
    final_checkpoint = None
    if rank == 0:
        tokenizer.save_pretrained(final_path)

        if script_args.merge_weights and not script_args.load_in_4bit:
            logger.info("Merging LoRA weights into base model...")
            from peft import AutoPeftModelForCausalLM

            merged = AutoPeftModelForCausalLM.from_pretrained(final_path, torch_dtype=torch.bfloat16)
            merged = merged.merge_and_unload()
            merged_path = os.path.join(script_args.output_dir, "merged_model")
            merged.save_pretrained(merged_path)
            tokenizer.save_pretrained(merged_path)
            logger.info(f"Merged model saved to: {merged_path}")

        final_checkpoint = ray.train.Checkpoint.from_directory(final_path)

    # Final report includes a checkpoint so Result.metrics/Result.checkpoint are
    # populated on the driver (Ray Train V2 only fills them from checkpoint reports).
    ray.train.report(metrics=metrics, checkpoint=final_checkpoint)
    logger.info("Training complete.")


# ---------------------------------------------------------------------------
# Driver (runs on the Ray head as the job entrypoint)
# ---------------------------------------------------------------------------

def main(config_path: str = "args.yaml"):
    config = load_config(config_path)
    script_args = ScriptArguments(**config.get("script_args", {}))
    run_name = config.get("run_name")
    if not run_name:
        raise ValueError(f"Set run_name in {config_path} (use a new name for each experiment).")

    os.makedirs(script_args.checkpoint_dir, exist_ok=True)
    os.makedirs(script_args.output_dir, exist_ok=True)

    use_gpu = config.get("use_gpu", True)
    resources_per_worker = {"CPU": config.get("cpus_per_worker", 8)}
    if use_gpu:
        resources_per_worker["GPU"] = config.get("num_gpus_per_worker", 1)
    scaling_config = ScalingConfig(
        num_workers=config.get("num_workers", 1),
        use_gpu=use_gpu,
        resources_per_worker=resources_per_worker,
    )

    run_config = RunConfig(
        name=run_name,
        storage_path=config.get("storage_path", "/mnt/fsx/ray-sft-lora/ray_results"),
        checkpoint_config=CheckpointConfig(num_to_keep=3),
        failure_config=FailureConfig(max_failures=config.get("max_failures", 3)),
    )

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={
            "script_args": script_args.__dict__,
            "sft_config": config.get("sft_config", {}),
            "run_name": run_name,
        },
        scaling_config=scaling_config,
        run_config=run_config,
    )

    result = trainer.fit()
    logger.info(f"Final metrics: {result.metrics}")
    logger.info(f"Final checkpoint: {result.checkpoint}")
    logger.info(f"Model artifacts: {script_args.output_dir}")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="args.yaml")
    args = parser.parse_args()
    main(config_path=args.config)
