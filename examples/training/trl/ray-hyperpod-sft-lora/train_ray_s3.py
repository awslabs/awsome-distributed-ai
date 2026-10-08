"""
Ray TorchTrainer on SageMaker HyperPod — Fine-tune an LLM with LoRA (S3 variant, no FSx)

Target: Ray cluster created via SageMaker Studio on HyperPod (EKS + KubeRay).

Shared storage is Amazon S3 instead of FSx for Lustre:
  - the dataset is read from <s3_base_uri>/datasets/{train,val}/ and copied to local disk
  - Ray Train checkpoints go to <s3_base_uri>/ray_results (RunConfig.storage_path)
  - the final LoRA adapter and merged model are uploaded by rank 0 to <s3_base_uri>/output/
The Ray pods get AWS credentials from EKS Pod Identity (no keys in code).

Heavy ML imports live inside train_func so the driver (Ray head, CPU image) only
needs ray, boto3 and pyyaml.
"""

import os

# The SageMaker Distribution image ships TensorFlow 2.19 + Keras 3. transformers 4.x imports its
# TF code paths when TF is present and fails with "Keras 3 ... not yet supported" (seen as
# "Failed to import trl.trainer.sft_trainer"). This is a PyTorch-only job, so turn TF off.
# Must be set before transformers is imported; also passed via runtime_env env_vars.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import logging
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

import yaml
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, CheckpointConfig, FailureConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class ScriptArguments:
    model_id: str = field(default="Qwen/Qwen3-0.6B")
    local_data_dir: str = field(default="/tmp/qwen3-medical/data")
    local_checkpoint_dir: str = field(default="/tmp/qwen3-medical/checkpoints")
    local_output_dir: str = field(default="/tmp/qwen3-medical/output")
    token: Optional[str] = field(default=None)
    attn_implementation: str = field(default="sdpa")
    load_in_4bit: bool = field(default=False)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    merge_weights: bool = field(default=True)


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _split_s3_uri(uri: str):
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Expected an s3:// URI, got: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def s3_download_prefix(s3_uri: str, local_dir: str) -> int:
    """Download every object under s3_uri into local_dir. Returns the file count."""
    import boto3

    bucket, prefix = _split_s3_uri(s3_uri.rstrip("/") + "/")
    s3 = boto3.client("s3")
    count = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix):]
            if not rel or rel.endswith("/"):
                continue
            dest = os.path.join(local_dir, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            s3.download_file(bucket, obj["Key"], dest)
            count += 1
    return count


def s3_upload_dir(local_dir: str, s3_uri: str) -> int:
    """Upload every file under local_dir to s3_uri. Returns the file count."""
    import boto3

    bucket, prefix = _split_s3_uri(s3_uri.rstrip("/") + "/")
    s3 = boto3.client("s3")
    count = 0
    for root, _, files in os.walk(local_dir):
        for name in files:
            path = os.path.join(root, name)
            key = prefix + os.path.relpath(path, local_dir)
            s3.upload_file(path, bucket, key)
            count += 1
    return count


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Training (runs inside each Ray Train worker)
# ---------------------------------------------------------------------------

def train_func(config: dict):
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
    s3_base_uri = config["s3_base_uri"].rstrip("/")
    ctx = ray.train.get_context()
    rank = ctx.get_world_rank()

    # 1. Dataset: S3 -> local disk of this worker
    train_dir = os.path.join(script_args.local_data_dir, "train")
    val_dir = os.path.join(script_args.local_data_dir, "val")
    n_train = s3_download_prefix(f"{s3_base_uri}/datasets/train", train_dir)
    n_val = s3_download_prefix(f"{s3_base_uri}/datasets/val", val_dir)
    logger.info(f"[rank {rank}] Downloaded {n_train} train / {n_val} val files from {s3_base_uri}/datasets")
    if n_train == 0:
        raise RuntimeError(f"No training data found under {s3_base_uri}/datasets/train")

    train_dataset = load_dataset("json", data_files=f"{train_dir}/*.json", split="train")
    val_dataset = (
        load_dataset("json", data_files=f"{val_dir}/*.json", split="train") if n_val else None
    )

    # 2. Model + LoRA
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
        token=script_args.token,
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

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id, token=script_args.token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Ray Train checkpoint reporting: rank 0 hands the HF checkpoint to Ray,
    #    which persists it to RunConfig.storage_path (S3). All ranks must call report().
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

    # FSDP is only turned on when args-s3.yaml sets it (not needed for 0.6B + LoRA).
    fsdp_kwargs = {}
    if sft_config.get("fsdp"):
        fsdp_kwargs = {"fsdp": sft_config["fsdp"], "fsdp_config": sft_config.get("fsdp_config", {})}

    training_args = SFTConfig(
        output_dir=os.path.join(script_args.local_checkpoint_dir, "sft"),
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

    train_result = trainer.train()
    metrics = dict(train_result.metrics)
    if val_dataset is not None:
        metrics.update(trainer.evaluate())

    # 4. Save final adapter (+ merged model) locally and upload to S3 from rank 0
    final_path = os.path.join(script_args.local_output_dir, "final_model")
    trainer.save_model(final_path)
    final_checkpoint = None
    if rank == 0:
        tokenizer.save_pretrained(final_path)
        n = s3_upload_dir(final_path, f"{s3_base_uri}/output/final_model")
        logger.info(f"Uploaded LoRA adapter ({n} files) to {s3_base_uri}/output/final_model")

        if script_args.merge_weights and not script_args.load_in_4bit:
            logger.info("Merging LoRA weights into base model...")
            from peft import AutoPeftModelForCausalLM

            merged = AutoPeftModelForCausalLM.from_pretrained(final_path, torch_dtype=torch.bfloat16)
            merged = merged.merge_and_unload()
            merged_path = os.path.join(script_args.local_output_dir, "merged_model")
            merged.save_pretrained(merged_path)
            tokenizer.save_pretrained(merged_path)
            n = s3_upload_dir(merged_path, f"{s3_base_uri}/output/merged_model")
            logger.info(f"Uploaded merged model ({n} files) to {s3_base_uri}/output/merged_model")

        final_checkpoint = ray.train.Checkpoint.from_directory(final_path)

    # Final report includes a checkpoint so Result.metrics/Result.checkpoint are
    # populated on the driver (Ray Train V2 only fills them from checkpoint reports).
    ray.train.report(metrics=metrics, checkpoint=final_checkpoint)
    logger.info("Training complete.")


# ---------------------------------------------------------------------------
# Driver (runs on the Ray head as the job entrypoint)
# ---------------------------------------------------------------------------

def main(config_path: str = "args-s3.yaml"):
    config = load_config(config_path)
    script_args = ScriptArguments(**config.get("script_args", {}))
    s3_base_uri = config["s3_base_uri"].rstrip("/")

    scaling_config = ScalingConfig(
        num_workers=config.get("num_workers", 1),
        use_gpu=config.get("use_gpu", True),
        resources_per_worker={
            "GPU": config.get("num_gpus_per_worker", 1),
            "CPU": config.get("cpus_per_worker", 8),
        },
    )

    run_config = RunConfig(
        name="hyperpod-ray-finetune",
        storage_path=f"{s3_base_uri}/ray_results",
        checkpoint_config=CheckpointConfig(num_to_keep=3),
        failure_config=FailureConfig(max_failures=config.get("max_failures", 3)),
    )

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={
            "script_args": script_args.__dict__,
            "sft_config": config.get("sft_config", {}),
            "s3_base_uri": s3_base_uri,
        },
        scaling_config=scaling_config,
        run_config=run_config,
    )

    result = trainer.fit()
    logger.info(f"Final metrics: {result.metrics}")
    logger.info(f"Final checkpoint: {result.checkpoint}")
    logger.info(f"Model artifacts: {s3_base_uri}/output/")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="args-s3.yaml")
    args = parser.parse_args()
    main(config_path=args.config)
