# NVIDIA Model Optimizer — FP8 Quantization on a Single GPU Instance

This test case demonstrates how to evaluate [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)
(ModelOpt) — a library of inference optimization techniques (quantization, pruning, distillation,
speculative decoding, sparsity) — on a **single GPU EC2 instance**. It runs **FP8 post-training
quantization (PTQ)** on a Hugging Face LLM and serves the compressed checkpoint with
[vLLM](https://github.com/vllm-project/vllm).

Unlike the distributed-training test cases in this repo, ModelOpt PTQ is a single-process,
single-GPU workload — so a single EC2 instance (provisioned here with Terraform) is the right
tool for evaluation. Graduate the proven workflow to EKS/HyperPod when you need repeatable,
multi-node, or production `train -> optimize -> serve` pipelines.

## Repository Structure

```text
3.test_cases/pytorch/model-optimizer/
├── README.md                 # This file
├── terraform/                # Single-GPU EC2 instance + SSM access
│   ├── versions.tf
│   ├── providers.tf
│   ├── variables.tf
│   ├── main.tf
│   ├── outputs.tf
│   └── README.md
└── src/
    ├── requirements.txt      # Pinned: nvidia-modelopt==0.44.0 + example deps
    ├── setup.sh              # venv + install + repo checkout pinned to the matching tag
    ├── quantize_fp8.sh       # FP8 PTQ via examples/llm_ptq/hf_ptq.py
    └── smoke_test_vllm.py    # vLLM offline generation to verify the checkpoint
```

## Compatible Instance Types

The quantization **format** is gated by the GPU **architecture**:

| Format | GPU architecture | Example instance |
|--------|------------------|------------------|
| INT8 / INT4-AWQ | Ampere+ | p4d, g5, g6e |
| **FP8** | Ada / Hopper | **g6e.xlarge** (L40S), p5.48xlarge (H100) |
| **NVFP4** | Blackwell only | p6-b200.48xlarge |

This recipe defaults to **`g6e.xlarge`** (1x L40S, 48 GB) — the cheapest viable single-GPU box
for FP8 (~$1.86/hr on-demand in us-west-2). NVFP4 cannot be exercised on g6e/p5.

> [!NOTE]
> ModelOpt is CUDA/TensorRT-centric. On Trainium/Inferentia (`trn`/`inf`) the analogous
> toolchain is the Neuron SDK, not this library.

## Prerequisites

- Terraform `>= 1.14.0` and AWS credentials (EC2/IAM/SG create permissions)
- The [SSM Session Manager plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html)
- EC2 quota for the chosen GPU instance family

## Step 1 — Provision the instance

```bash
cd terraform
terraform init
terraform apply        # creates IAM role, egress-only SG, GPU instance (default VPC)

# Connect via SSM (no SSH, no key pair):
$(terraform output -raw ssm_start_session_command)
```

Verify the GPU:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
# e.g.  NVIDIA L40S, 46068 MiB, 595.71.05
```

See [`terraform/README.md`](terraform/README.md) for variables and overrides.

## Step 2 — Install ModelOpt

Copy `src/` to the instance (or `git clone` this repo on the box), then:

```bash
cd src
bash setup.sh        # ~90s: venv + nvidia-modelopt==0.44.0 + repo checkout @ 0.44.0
```

## Step 3 — Run FP8 PTQ

```bash
bash quantize_fp8.sh    # defaults to Qwen/Qwen2.5-7B-Instruct (non-gated)
```

Override the model or calibration size via environment variables:

```bash
MODEL=Qwen/Qwen2.5-7B-Instruct CALIB_SIZE=512 bash quantize_fp8.sh
```

## Step 4 — Serve & verify with vLLM

vLLM pins specific torch versions, so install it in a **separate venv**:

```bash
python3.12 -m venv ~/model-optimizer-recipe/vllm-venv
. ~/model-optimizer-recipe/vllm-venv/bin/activate
pip install 'vllm==0.23.0'
python smoke_test_vllm.py --model ~/qwen2.5-7b-fp8
```

## Step 5 — Teardown (cost-critical)

```bash
cd terraform
terraform destroy
```

> [!WARNING]
> The instance bills until destroyed. Run `terraform destroy` as soon as you are done.

## Results (reference run)

Live run on a single `g6e.xlarge` (L40S, driver 595.71.05) in us-west-2:

| Metric | Value |
|--------|-------|
| ModelOpt version | 0.44.0 |
| Install time | ~90 s |
| Calibration + export (Qwen2.5-7B, after model cached) | ~45 s |
| Peak GPU memory during PTQ | ~16.5 GB |
| Checkpoint size | 15 GB BF16 -> **8.2 GB FP8** (~1.8x) |
| vLLM kernel selected | `CutlassFP8ScaledMMLinearKernel` |

Sample generation from the FP8 checkpoint:

```text
PROMPT:  What is the capital of France? Answer in one sentence.
OUTPUT:  The capital of France is Paris.
```

## Known Issues

- **FP8 KV-cache produces garbled output on Ada (L40S).** Quantize **weights only**
  (`--kv_cache_qformat none`, the default in `quantize_fp8.sh`). vLLM warns
  `Using KV cache scaling factor 1.0 for fp8_e4m3` and emits gibberish if FP8 KV-cache is on
  without validated scales. Enable FP8 KV-cache only with explicit accuracy validation.
- **Repo/wheel version skew.** The example scripts are not in the pip wheel; `main` may
  reference internals not in the release. `setup.sh` checks out the tag matching the pinned
  wheel (`0.44.0`) to avoid `ModuleNotFoundError`.
- **The default calibration dataset is gated.** `hf_ptq.py` defaults to a gated Nemotron
  dataset; this recipe passes `--dataset cnn_dailymail` (non-gated) so it is self-contained.
- **`transformers>=5.0` is experimental** in ModelOpt 0.44.0 (emits a `UserWarning`). It works
  for Qwen2.5; pin `transformers` if you hit export issues on other models.

## References

- [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)
- [ModelOpt documentation](https://nvidia.github.io/Model-Optimizer)
- [LLM PTQ example](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/llm_ptq)
- [Introducing NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [vLLM](https://github.com/vllm-project/vllm)
