import os, boto3, torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

s3 = boto3.client("s3")
def split(uri): b, _, k = uri[5:].partition("/"); return b, k

# 1. Stage base model + training data from S3
bucket, prefix = split(os.environ["BASE_MODEL_S3"])
for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
    for o in page.get("Contents", []):
        dst = os.path.join("/work/base", os.path.relpath(o["Key"], prefix))
        os.makedirs(os.path.dirname(dst), exist_ok=True); s3.download_file(bucket, o["Key"], dst)
s3.download_file(*split(os.environ["TRAIN_FILE_S3"]), "/work/train.jsonl")
print("[stage] base model and training data downloaded", flush=True)

# 2. LoRA SFT, loss only on the assistant turn (the label), not the prompt
tok = AutoTokenizer.from_pretrained("/work/base")
def encode(ex):
    full = tok.apply_chat_template(ex["messages"], tokenize=False)
    prompt = tok.apply_chat_template(ex["messages"][:-1], tokenize=False, add_generation_prompt=True)
    ids = tok(full, add_special_tokens=False)["input_ids"]
    n = len(tok(prompt, add_special_tokens=False)["input_ids"])
    return {"input_ids": ids, "labels": [-100] * n + ids[n:]}
ds = load_dataset("json", data_files="/work/train.jsonl", split="train")
print(f"[data] {len(ds)} examples", flush=True)
ds = ds.map(encode, remove_columns=ds.column_names)

def collate(batch):
    m = max(len(b["input_ids"]) for b in batch)
    pad = lambda xs, v: [x + [v] * (m - len(x)) for x in xs]
    return {"input_ids": torch.tensor(pad([b["input_ids"] for b in batch], tok.pad_token_id)),
            "labels": torch.tensor(pad([b["labels"] for b in batch], -100)),
            "attention_mask": torch.tensor(pad([[1] * len(b["input_ids"]) for b in batch], 0))}

model = AutoModelForCausalLM.from_pretrained("/work/base", torch_dtype=torch.bfloat16, device_map="cuda:0")
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()
Trainer(model=model, train_dataset=ds, data_collator=collate, args=TrainingArguments(
    output_dir="/tmp/ckpt", num_train_epochs=float(os.getenv("EPOCHS", "2")), learning_rate=1e-4,
    per_device_train_batch_size=8, bf16=True, logging_steps=10, save_strategy="no", report_to=[],
    warmup_ratio=0.1, lr_scheduler_type="cosine")).train()

# 3. Merge the adapter into the base weights (a plain model vLLM can serve) and upload it
model.merge_and_unload().save_pretrained("/work/out", safe_serialization=True)
tok.save_pretrained("/work/out")
bucket, prefix = split(os.environ["OUTPUT_S3"])
for f in os.listdir("/work/out"):
    s3.upload_file(os.path.join("/work/out", f), bucket, f"{prefix.rstrip('/')}/{f}")
print(f"[done] merged model uploaded to {os.environ['OUTPUT_S3']}", flush=True)
