# Qwen2.5-7B GRPO Experiment Plan

## Objective

Two-phase experiment to:
1. **Verify** GRPO works on Qwen2.5-7B for multilingual tasks (proving GPT-OSS-20B architecture was the failure cause)
2. **Demonstrate** GRPO improves math reasoning over SFT-only (replicating DeepSeekMath findings)

## Background

Previous work with GPT-OSS-20B showed GRPO failed across 13 experiments (~120 GPU-hours).
Key question: Was failure due to (a) GPT-OSS architecture (mxfp4, MoE routing, Harmony layers)
or (b) the multilingual task being inherently unsuitable for GRPO?

Literature shows GRPO provides +4-6% absolute on math (DeepSeekMath, Qwen2.5-Math).

---

## Phase 1: Multilingual Verification (~324 GPU-hours)

### Hypothesis
If GRPO works on Qwen2.5-7B for the same multilingual task, then GPT-OSS architecture
was the problem (not the task).

### Arms

| Arm | Training | Expected Accuracy |
|-----|----------|-------------------|
| A | None (base Qwen2.5-7B-Instruct) | ~30-40% |
| B | SFT only (3 epochs) | ~80-90% |
| C | GRPO only (from base, 3 epochs) | ~50-70% |
| D | SFT + GRPO (SFT then GRPO) | ~85-95% |

### Dataset
- `HuggingFaceH4/Multilingual-Thinking`
- 5 languages: English, French, German, Spanish, Italian
- Task: Answer questions in the detected language

### Training Config
- Model: `Qwen/Qwen2.5-7B-Instruct`
- LoRA: r=16, alpha=32, all linear layers
- SFT: lr=2e-5, 3 epochs, cosine schedule
- GRPO: lr=1e-6, 3 epochs, n=8 samples, temp=0.7, KL=0.001

### Evaluation
- 50 test prompts (10 per language)
- Metric: % responses in correct language (via langdetect)
- Automated via `evaluate_multilingual.py`

### Decision Matrix

| Result | Conclusion |
|--------|-----------|
| D > B > C > A | Ideal: SFT+GRPO best, both contribute |
| C > A, D > B | GRPO helps — GPT-OSS was the problem |
| C ≈ A (GRPO fails) | Task IS hard for GRPO (not just model) |
| B ≈ D (GRPO doesn't add) | SFT sufficient, GRPO adds noise |

### GPU Budget
- Arm B (SFT): ~8 hours on 8 GPUs = 64 GPU-hrs
- Arm C (GRPO): ~16 hours on 16 GPUs = 256 GPU-hrs  
- Arm D (GRPO from SFT): Reuse SFT + ~16 hours GRPO = 256 GPU-hrs
- Eval: ~4 GPU-hrs per arm
- **Total Phase 1: ~324 GPU-hours**

---

## Phase 2: Math Reasoning (~480 GPU-hours)

### Hypothesis
SFT+GRPO on GSM8K will outperform SFT-only by 4-6% (consistent with literature).

### Arms

| Arm | Training | Expected GSM8K |
|-----|----------|----------------|
| A | None (base) | ~75-80% |
| B | SFT only (GSM8K CoT, 3 epochs) | ~82-85% |
| C | GRPO only (from base) | ~78-83% |
| D | SFT + GRPO | ~86-90% |

### Dataset
- **Training**: `openai/gsm8k` (7.5K train)
- **Evaluation**: `openai/gsm8k` test (1.3K)
- Format: CoT with `\boxed{answer}` extraction

### Training Config
- Same as Phase 1 (LoRA r=16, alpha=32)
- GRPO response length: 2048 tokens (math needs longer CoT)
- Reward: exact answer match via `math_reward.py`

### Evaluation
- GSM8K test set (1,319 problems)
- Metric: exact match accuracy (predicted vs ground truth)
- Uses vLLM for fast batch inference

### GPU Budget
- Arm B (SFT): ~12 hours on 8 GPUs = 96 GPU-hrs
- Arm C (GRPO): ~24 hours on 16 GPUs = 384 GPU-hrs
- Arm D: Reuse SFT + GRPO = 384 GPU-hrs
- Eval: ~16 GPU-hrs total
- **Total Phase 2: ~480 GPU-hours**

---

## Infrastructure

| Component | Spec |
|-----------|------|
| Instance | g6e.48xlarge (8x NVIDIA L40S, 48GB each) |
| Nodes | 2 (16 GPUs total) |
| Network | EFA (100 Gbps) |
| Storage | FSx for Lustre |
| Orchestration | Ray 2.44 + KubeRay 1.4.2 |
| Framework | veRL v0.6.1, vLLM 0.8.4 |
| Container | Custom (see Dockerfile) |

## Timeline

| Day | Activity |
|-----|----------|
| 1 | Build image, deploy cluster, preprocess data |
| 2-3 | Phase 1: SFT + GRPO training |
| 4 | Phase 1: Evaluation + analysis |
| 5-7 | Phase 2: SFT + GRPO training |
| 8 | Phase 2: Evaluation + final report |

**Total: ~8 days, ~804 GPU-hours**

---

## Success Criteria

1. Phase 1: GRPO improves over base on Qwen (Arm C > A) — confirms GPT-OSS was the blocker
2. Phase 2: SFT+GRPO > SFT-only by >= 3% on GSM8K — replicates literature
3. No training instability (grad norm < 10, no NaN losses)
4. Reproducible: all configs, data, and eval scripts committed
