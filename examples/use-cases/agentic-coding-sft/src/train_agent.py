# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""LoRA SFT on pre-tokenized agent trajectories (from prep_agent.py), launched by hyperpodrun.

Plain HF Trainer: the dataset already carries input_ids + labels (loss only on assistant
turns), so no chat templating or masking happens here. Long sequences (8-16K tokens) need
gradient checkpointing and Liger's fused linear cross-entropy, which never materialises the
full [seq x 152K vocab] logits tensor.
"""
import json
import os
import time
from dataclasses import dataclass, field

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, Trainer,
                          TrainerCallback, TrainingArguments, set_seed)


ROOT = os.environ.get("DATA_ROOT", "/fsx")  # FSx for Lustre mount


class GpuMemCallback(TrainerCallback):
    """Print peak GPU memory and step timing at every log step (grep for 'GPUMEM')."""

    def __init__(self):
        self.t0 = None

    def on_train_begin(self, args, state, control, **kw):
        torch.cuda.reset_peak_memory_stats()
        self.t0 = time.time()

    def on_log(self, args, state, control, logs=None, **kw):
        if not torch.cuda.is_available() or not state.is_world_process_zero:
            return
        peak = torch.cuda.max_memory_allocated() / 2**30
        reserved = torch.cuda.max_memory_reserved() / 2**30
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        el = time.time() - self.t0
        sps = el / max(state.global_step, 1)
        print(f"GPUMEM step={state.global_step}/{state.max_steps} peak_alloc={peak:.2f}GiB "
              f"peak_reserved={reserved:.2f}GiB total={total:.2f}GiB "
              f"elapsed={el:.0f}s sec_per_step={sps:.2f}", flush=True)


def _patch_liger():
    """liger-kernel 0.6.3 + transformers 4.57: the model forwards extra kwargs (e.g.
    `return_dict`) into liger_fused_linear_cross_entropy, which rejects them. Drop any kwarg
    the function doesn't accept."""
    import inspect

    import liger_kernel.transformers.model.loss_utils as lu
    fn = lu.F.liger_fused_linear_cross_entropy
    ok = set(inspect.signature(fn).parameters)

    def wrapped(*args, **kw):
        return fn(*args, **{k: v for k, v in kw.items() if k in ok})

    lu.F.liger_fused_linear_cross_entropy = wrapped


@dataclass
class Args:
    model_id: str = field(default="Qwen/Qwen3-1.7B")
    train_path: str = field(default=f"{ROOT}/data/agent/train-8k")
    eval_path: str = field(default=f"{ROOT}/data/agent/val")
    max_train_samples: int = field(default=0)
    max_eval_samples: int = field(default=50)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    merge: bool = field(default=False)
    mlflow_tracking_server: str = field(default="", metadata={
        "help": "SageMaker MLflow tracking server ARN; empty = no MLflow logging"})
    mlflow_exp_name: str = field(default="qwen3-agentic-sft")


def collate(batch, pad_id):
    batch = [{k: b[k] for k in ("input_ids", "labels")} for b in batch]
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L] = torch.tensor(b["input_ids"])
        lab[i, :L] = torch.tensor(b["labels"])
        att[i, :L] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def main():
    a, targs = HfArgumentParser((Args, TrainingArguments)).parse_args_into_dataclasses()
    if targs.use_liger_kernel:
        _patch_liger()
    set_seed(targs.seed)
    if a.mlflow_tracking_server:  # optional: HF Trainer -> SageMaker managed MLflow
        os.environ["MLFLOW_TRACKING_URI"] = a.mlflow_tracking_server
        os.environ["MLFLOW_EXPERIMENT_NAME"] = a.mlflow_exp_name
        os.environ.setdefault("MLFLOW_RUN_NAME", os.path.basename(targs.output_dir.rstrip("/")))
        targs.report_to = ["mlflow"]
        targs.run_name = os.environ["MLFLOW_RUN_NAME"]
    else:
        targs.report_to = []
    tok = AutoTokenizer.from_pretrained(a.model_id)

    train = load_from_disk(a.train_path)
    val = load_from_disk(a.eval_path)
    if a.max_train_samples:
        train = train.select(range(min(a.max_train_samples, len(train))))
    if a.max_eval_samples:
        val = val.select(range(min(a.max_eval_samples, len(val))))
    keep = ["input_ids", "labels"]
    train = train.remove_columns([c for c in train.column_names if c not in keep])
    val = val.remove_columns([c for c in val.column_names if c not in keep])
    lens = [len(x) for x in train["input_ids"]]
    if targs.group_by_length:  # similar-length traces per step -> less waiting across GPUs
        train = train.add_column("length", lens)
        targs.length_column_name = "length"
    print(f"train={len(train)} val={len(val)} tokens: mean={sum(lens)/len(lens):.0f} "
          f"max={max(lens)} total={sum(lens)} model={a.model_id}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(a.model_id, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout, bias="none",
        task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                               "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    if targs.gradient_checkpointing:
        model.enable_input_require_grads()
    targs.remove_unused_columns = False
    # Multi-node (DDP) settings; no effect on a single GPU. Reentrant checkpointing + DDP with
    # find_unused_parameters breaks, and LoRA has no unused params, so turn the search off.
    targs.ddp_find_unused_parameters = False
    if targs.gradient_checkpointing:
        targs.gradient_checkpointing_kwargs = {"use_reentrant": False}
    world = int(os.environ.get("WORLD_SIZE", "1"))
    eff = targs.per_device_train_batch_size * targs.gradient_accumulation_steps * world
    if targs.process_index == 0:
        print(f"world_size={world} effective_batch={eff} "
              f"(micro {targs.per_device_train_batch_size} x accum "
              f"{targs.gradient_accumulation_steps} x {world} GPU)", flush=True)

    trainer = Trainer(model=model, args=targs, train_dataset=train, eval_dataset=val,
                      data_collator=lambda b: collate(b, tok.pad_token_id),
                      callbacks=[GpuMemCallback()])
    out = targs.output_dir
    # Resume from the latest checkpoint in output_dir if there is one. This is what makes the
    # HyperPod training operator's process-level restart continue instead of starting over: after
    # a failed worker is restarted, the fresh process picks up the last saved checkpoint.
    from transformers.trainer_utils import get_last_checkpoint
    last_ckpt = get_last_checkpoint(out) if os.path.isdir(out) else None
    if last_ckpt and targs.process_index == 0:
        print(f"RESUMING from {last_ckpt}", flush=True)
    # Open the MLflow run ourselves (rank 0) so the HF callback reuses it and doesn't end it at
    # train end; otherwise the final evaluate() lands in a second, randomly named run. On a restart
    # we reattach to the same run via its id (saved beside the output) so metrics stay in one run.
    mlrun = None
    if a.mlflow_tracking_server and targs.process_index == 0:
        import mlflow
        mlflow.set_tracking_uri(a.mlflow_tracking_server)
        mlflow.set_experiment(a.mlflow_exp_name)
        run_id_file = os.path.join(out, "mlflow_run_id.txt")
        prev_run_id = open(run_id_file).read().strip() if os.path.exists(run_id_file) else None
        mlrun = mlflow.start_run(run_id=prev_run_id, run_name=os.environ.get("MLFLOW_RUN_NAME"))
        if not prev_run_id:
            os.makedirs(out, exist_ok=True)
            with open(run_id_file, "w") as f:
                f.write(mlrun.info.run_id)
    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=last_ckpt)
    wall = time.time() - t0
    ev = trainer.evaluate() if len(val) else None  # collective: every rank must call it

    if trainer.is_world_process_zero():
        trainer.model.save_pretrained(f"{out}/adapter")
        tok.save_pretrained(f"{out}/adapter")
        summary = {"model_id": a.model_id, "train_path": a.train_path,
                   "train_samples": len(train), "train_tokens": sum(lens),
                   "train_runtime_s": round(wall, 1),
                   "tokens_per_s": round(sum(lens) * targs.num_train_epochs / wall, 1)
                   if targs.max_steps <= 0 else None,
                   "global_steps": trainer.state.global_step, "train_loss": result.training_loss,
                   "world_size": world, "effective_batch": eff,
                   "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                   "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
                   "per_device_train_batch_size": targs.per_device_train_batch_size,
                   "gradient_accumulation_steps": targs.gradient_accumulation_steps,
                   "group_by_length": targs.group_by_length,
                   "use_liger_kernel": targs.use_liger_kernel, "lora_r": a.lora_r, "eval": ev}
        print("SUMMARY " + json.dumps(summary), flush=True)
        json.dump(summary, open(f"{out}/summary.json", "w"), indent=2)
        if a.merge:
            merged = trainer.model.merge_and_unload()
            merged.save_pretrained(f"{out}/merged", safe_serialization=True)
            tok.save_pretrained(f"{out}/merged")
    if mlrun is not None:
        import mlflow
        mlflow.end_run()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()