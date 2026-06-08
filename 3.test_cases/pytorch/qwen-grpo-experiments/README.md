# Qwen2.5-7B GRPO Experiments

Controlled experiments to demonstrate that GRPO improves over SFT-only training,
and to verify that the GRPO failures observed with GPT-OSS-20B were model-specific
(not task-specific).

## Overview

| Phase | Goal | Task | Expected Outcome |
|-------|------|------|-----------------|
| **Phase 1** | Verify GRPO works on Qwen (where GPT-OSS failed) | Multilingual language compliance | GRPO > Base, SFT+GRPO >= SFT |
| **Phase 2** | Demonstrate GRPO improves math reasoning | GSM8K math | SFT+GRPO > SFT > Base |

## Experiment Arms (per phase)

| Arm | Description | Training |
|-----|-------------|----------|
| A | Base model (no fine-tuning) | None |
| B | SFT only | veRL fsdp_sft_trainer + LoRA |
| C | GRPO only (from base) | veRL GRPO + LoRA |
| D | SFT + GRPO | SFT first, merge, then GRPO |

## Infrastructure

- **Cluster**: EKS with g6e.48xlarge (8x L40S per node)
- **Nodes**: 2 (16 GPUs total)
- **Framework**: veRL v0.6.1 + vLLM + Ray + FSDP
- **Model**: Qwen/Qwen2.5-7B-Instruct
- **LoRA**: r=16, alpha=32, all linear layers

## Quick Start

```bash
# 1. Build and push Docker image
./build_push.sh

# 2. Deploy Ray cluster
kubectl apply -f k8s/raycluster.yaml

# 3. Port-forward Ray dashboard
kubectl port-forward svc/qwen-grpo-cluster-head-svc 8265:8265

# 4. Run full pipeline
kubectl exec -it qwen-grpo-cluster-head-xxxxx -- bash
cd /workspace/experiments
./run_all.sh

# Or run individual steps:
./run_all.sh --phase 1 --step preprocess
./run_all.sh --phase 1 --step sft
./run_all.sh --phase 1 --step grpo
./run_all.sh --phase 1 --step eval
```

## Directory Structure

```
qwen-grpo-experiments/
├── README.md                         # This file
├── Dockerfile                        # veRL + EFA image (no model patches needed)
├── build_push.sh                     # Build and push to ECR
├── run_all.sh                        # Master orchestration
├── EXPERIMENT_PLAN_FULL.md           # Detailed experiment design
├── k8s/
│   ├── raycluster.yaml               # RayCluster (2x g6e.48xlarge)
│   └── env_vars.example              # Environment template
└── src/
    ├── data_preprocess_multilingual.py  # Phase 1 data prep
    ├── data_preprocess_math.py          # Phase 2 data prep
    ├── language_reward.py               # Multilingual reward ([-7.5, +7.0])
    ├── language_reward_simple.py        # Binary language reward ([-1, +1])
    ├── math_reward.py                   # Math correctness reward
    ├── run_sft_multilingual.sh          # SFT launch (Phase 1)
    ├── run_grpo_multilingual.sh         # GRPO launch (Phase 1)
    ├── run_sft_math.sh                  # SFT launch (Phase 2)
    ├── run_grpo_math.sh                 # GRPO launch (Phase 2)
    ├── evaluate_multilingual.py         # Language compliance eval
    ├── evaluate_math.py                 # Math accuracy eval (vLLM)
    ├── merge_lora_checkpoint.py         # Merge LoRA into base for Arm D
    └── batch_eval.py                    # Run all evals, produce summary table
```

## Key Design Decisions

1. **No model patches**: Qwen2.5-7B is natively supported by vLLM and veRL (unlike GPT-OSS which required 9 build-time + 4 runtime patches)
2. **LoRA throughout**: r=16, alpha=32 for both SFT and GRPO (keeps experiments comparable)
3. **veRL for both SFT and GRPO**: Same framework for fair comparison
4. **KL loss in GRPO**: `kl_loss_coef=0.001` with `low_var_kl` prevents policy collapse
5. **No FSDP offload**: 7B model fits on L40S without CPU offloading

## Expected Results

Based on DeepSeekMath and Qwen2.5-Math literature:

| Phase | Metric | Expected |
|-------|--------|----------|
| Phase 1 | Language compliance (Arm D vs A) | +40-60% absolute |
| Phase 2 | GSM8K accuracy (Arm D vs B) | +4-6% absolute |
| Phase 2 | GSM8K accuracy (Arm D vs A) | +8-12% absolute |

## Decision Matrix (Phase 1)

| Outcome | Interpretation |
|---------|---------------|
| GRPO works on Qwen (Arm C/D > A) | GPT-OSS architecture was the problem |
| GRPO fails on Qwen too | Multilingual task is inherently hard for GRPO |
| SFT+GRPO > SFT > GRPO > Base | Ideal outcome: both help, combination is best |
