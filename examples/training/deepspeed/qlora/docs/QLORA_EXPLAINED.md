# QLoRA Explained

This document explains the QLoRA (Quantized Low-Rank Adaptation) technique used in this project for efficient fine-tuning of large language models.

## The Problem: Fine-tuning Large Models

Fine-tuning a model like Qwen3-8B traditionally requires:

| Approach | Memory Required | Cost |
|----------|-----------------|------|
| Full fine-tuning (fp32) | ~32 GB per billion params | Very High |
| Full fine-tuning (fp16) | ~16 GB per billion params | High |
| QLoRA (4-bit + LoRA) | ~0.5-1 GB per billion params | Low |

For Qwen3-8B:
- **Full fine-tuning**: ~128 GB (fp16) - requires multiple high-end GPUs
- **QLoRA**: ~8-10 GB per GPU - trainable on consumer-grade hardware

## How QLoRA Works

QLoRA combines three techniques:

### 1. 4-bit Quantization

Quantization reduces the precision of model weights:

```
Original (fp32):  [0.1234567, 0.2345678, ...]  → 32 bits per weight
fp16:             [0.1235, 0.2346, ...]         → 16 bits per weight
4-bit (NF4):      [0.12, 0.23, ...]             → 4 bits per weight
```

**NormalFloat4 (NF4)**: A special 4-bit format optimized for normally distributed weights (which LLM weights typically are).

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # NormalFloat4
    bnb_4bit_use_double_quant=True,  # Quantize the quantization constants too!
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in higher precision
)
```

### 2. Low-Rank Adaptation (LoRA)

Instead of updating all weights, LoRA adds small "adapter" matrices:

```
Original: y = Wx        (W is d×k, e.g., 4096×4096)

With LoRA: y = Wx + BAx  (B is d×r, A is r×k, r << d,k)
                         (e.g., B is 4096×16, A is 16×4096)
```

The rank `r` is typically 8-64, so:
- Original parameters: 4096 x 4096 = 16.7M
- LoRA parameters: (4096 x 16) + (16 x 4096) = 131K
- **Reduction: 99.2%**

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,              # Rank of adaptation
    lora_alpha=32,     # Scaling factor
    lora_dropout=0.05, # Regularization
    target_modules=[   # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
)
```

### 3. Paged Optimizers

Memory spikes during training can cause OOM errors. Paged optimizers move optimizer states to CPU memory when GPU memory is full:

```python
training_args = TrainingArguments(
    optim="paged_adamw_8bit",  # Paged optimizer in 8-bit
    # ...
)
```

## Target Modules Explained

For transformer models like Qwen3, we apply LoRA to:

```
┌─────────────────────────────────────────────────────┐
│                 Transformer Layer                    │
│                                                      │
│  ┌───────────────────────────────────────────────┐  │
│  │            Self-Attention Block                │  │
│  │                                                │  │
│  │  Input → [q_proj] → Q                         │  │
│  │       → [k_proj] → K    → Attention → [o_proj]│  │
│  │       → [v_proj] → V                          │  │
│  │                                                │  │
│  │  LoRA: ✓ q_proj, ✓ k_proj, ✓ v_proj, ✓ o_proj│  │
│  └───────────────────────────────────────────────┘  │
│                         │                            │
│                         ▼                            │
│  ┌───────────────────────────────────────────────┐  │
│  │              MLP Block (SwiGLU)                │  │
│  │                                                │  │
│  │  Input → [gate_proj] ─┐                       │  │
│  │       → [up_proj] ────┼─→ SiLU → [down_proj] │  │
│  │                       │                       │  │
│  │  LoRA: ✓ gate_proj, ✓ up_proj, ✓ down_proj   │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## Hyperparameter Guide

### LoRA Rank (`r`)

| Rank | Parameters | Memory | Quality | Use Case |
|------|------------|--------|---------|----------|
| 4 | Very low | Minimal | Basic | Quick experiments |
| 8 | Low | Low | Good | Small datasets |
| **16** | Medium | Medium | Very Good | **General use** |
| 32 | High | Higher | Excellent | Large datasets |
| 64 | Very high | High | Best | Maximum quality |

### LoRA Alpha

Controls the scaling of LoRA updates:

```
Effective LoRA weight = (alpha / r) × BA
```

Rules of thumb:
- `alpha = r`: Balanced
- `alpha = 2*r`: Stronger LoRA updates (recommended)
- `alpha = 4*r`: Very strong updates

### Learning Rate

QLoRA typically uses higher learning rates than full fine-tuning:

| Method | Typical Learning Rate |
|--------|----------------------|
| Full fine-tuning | 1e-5 to 5e-5 |
| QLoRA | 1e-4 to 3e-4 |

We use `2e-4` in this project.

## Memory Usage Breakdown

### Single GPU (g5.2xlarge, 24GB)

Running QLoRA on a single A10G with `batch_size=1, seq_len=1536`:

```
┌─────────────────────────────────────────┐
│           GPU Memory (24GB)              │
├─────────────────────────────────────────┤
│ Base Model (4-bit):        ~5 GB        │
│ LoRA Adapters:             ~0.1 GB      │
│ Activations (batch=1):     ~8 GB        │
│ Optimizer States:          ~3 GB        │
│ Gradients:                 ~2 GB        │
│ Misc (buffers, etc):       ~2 GB        │
├─────────────────────────────────────────┤
│ Total:                     ~20 GB       │
│ Free:                      ~4 GB        │
└─────────────────────────────────────────┘
```

> A single GPU runs near capacity and is prone to CUDA OOM crashes at
> checkpoint-save points where memory spikes temporarily.

### Multi-GPU with DeepSpeed ZeRO-2 (g5.12xlarge, 4x 24GB)

This project uses 4x A10G GPUs with DDP + DeepSpeed ZeRO-2, which **shards
optimizer states and gradients** across GPUs. Each GPU still holds the full
model parameters, but optimizer memory is divided by the number of GPUs:

```
┌─────────────────────────────────────────┐
│      Per-GPU Memory with ZeRO-2         │
│         (4x A10G, 24GB each)            │
├─────────────────────────────────────────┤
│ Base Model (4-bit):        ~5 GB        │
│ LoRA Adapters:             ~0.1 GB      │
│ Activations (batch=1):     ~5 GB        │
│ Optimizer States (÷4):     ~0.75 GB     │
│ Gradients (÷4):            ~0.5 GB      │
│ Misc (NCCL buffers, etc):  ~3 GB        │
├─────────────────────────────────────────┤
│ Total per GPU:             ~15 GB       │
│ Free per GPU:              ~9 GB        │
└─────────────────────────────────────────┘
```

Key settings that prevent OOM:
- `max_seq_length: 1536` (reduced from 2048, saves ~25% activation memory)
- `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`
- `gradient_checkpointing: true` (trades compute for memory)

## Code Example

Complete QLoRA setup:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Step 1: Create quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Step 2: Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    quantization_config=quantization_config,
    device_map="auto",
)

# Step 3: Prepare for training
model = prepare_model_for_kbit_training(model)

# Step 4: Add LoRA adapters
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# Check trainable parameters
model.print_trainable_parameters()
# Output: trainable params: 41,943,040 || all params: 8,251,297,792 || trainable%: 0.508
```

## Merging LoRA into the Full-Precision Model

After training, you can merge the LoRA adapter into the original bf16 model
for better inference accuracy (no quantization loss):

```python
from peft import PeftModel

# Load base model in full precision (no quantization)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Apply and merge LoRA adapter
model = PeftModel.from_pretrained(model, "outputs/final_model")
model = model.merge_and_unload()  # Permanently merges adapter into weights

# Now 'model' is a standard model with no adapter overhead
# Requires ~16GB VRAM for inference
```

See `src/inference_demo.py` for a complete working example.

## Quality vs. Efficiency Trade-offs

| Configuration | Per-GPU VRAM | Training Speed | Quality |
|---------------|-------------|----------------|---------|
| r=8, batch=1, 1 GPU | ~12 GB | Fast | Good |
| **r=16, batch=1, 4 GPU ZeRO-2** | **~15 GB** | **Medium** | **Very Good** |
| r=32, batch=1, 4 GPU ZeRO-2 | ~18 GB | Slower | Excellent |
| r=64, batch=1, 4 GPU ZeRO-3 | ~14 GB | Slow | Best |

## Common Issues

### 1. Out of Memory

```python
# Reduce batch size
per_device_train_batch_size = 1

# Increase gradient accumulation
gradient_accumulation_steps = 8

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# If still OOM, reduce sequence length
max_seq_length = 1024

# Or switch to DeepSpeed ZeRO-3 for full parameter sharding
```

### 2. Loss Not Decreasing

- Try lower learning rate: `1e-4`
- Increase warmup: `warmup_ratio = 0.1`
- Check data formatting

### 3. Unstable Training

- Use gradient clipping: `max_grad_norm = 0.3`
- Try `fp32` compute dtype instead of `bfloat16`

## References

- [QLoRA Paper](https://arxiv.org/abs/2305.14314) - Original research
- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Foundation technique
- [PEFT Documentation](https://huggingface.co/docs/peft) - Implementation details
- [BitsAndBytes](https://github.com/bitsandbytes-foundation/bitsandbytes) - Quantization library
- [DeepSpeed](https://www.deepspeed.ai/tutorials/zero/) - ZeRO optimization stages
