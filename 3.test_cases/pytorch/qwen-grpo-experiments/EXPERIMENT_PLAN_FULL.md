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

## Phase 2: Math Reasoning

### Phase 2a: GSM8K — COMPLETED (Ceiling Effect)

**Result**: Qwen2.5-7B-Instruct already achieves 90.5% on GSM8K — too strong for GRPO to improve.
GRPO (step 30, best val) = 90.5% (identical to baseline). No headroom.

| Arm | Training | GSM8K Accuracy |
|-----|----------|:--------------:|
| A | None (base) | 90.5% |
| C | GRPO from base (1 epoch) | 90.5% (no change) |

**Lesson**: Need pass@k >> pass@1 for GRPO to help. GSM8K is too easy for this model.

### Phase 2b: MATH (Competition-Level) — IN PROGRESS

### Hypothesis
GRPO on MATH (AMC/AIME difficulty) will show improvement because baseline is ~50-65%,
providing substantial headroom. This replicates DeepSeekMath: MATH 46.8% → 51.7% (+4.9%).

### Arms

| Arm | Training | Expected MATH |
|-----|----------|:-------------:|
| A | None (base Qwen2.5-7B-Instruct) | ~50-65% |
| C | GRPO from base (1 epoch) | ~55-70% (+4-6%) |

### Dataset
- **Training**: `hendrycks/MATH` or `lighteval/MATH` (12.5K problems, competition-level)
- **Evaluation**: MATH test set (5K problems)
- Subjects: Algebra, Counting/Probability, Geometry, Number Theory, Intermediate Algebra, Precalculus, Prealgebra
- Format: CoT with `\boxed{answer}` extraction

### Training Config
- LoRA r=16, alpha=32, all-linear
- GRPO: lr=5e-6, 1 epoch, n=4, temp=0.7, KL=0.01, max_response_length=2048
- Reward: exact answer match (+1.2 correct, -1.2 wrong) via `math_verify` library
- `layered_summon=True`, `load_format=safetensors`, TP=2

### Evaluation
- MATH test set (500 problems, representative subset across difficulty levels 1-5)
- Metric: exact match accuracy
- Greedy decoding, max_new_tokens=2048

### GPU Budget
- Arm C (GRPO): ~16 hours on 8 GPUs = 128 GPU-hrs
- Eval: ~8 GPU-hrs
- **Total Phase 2b: ~136 GPU-hours**

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

1. Phase 1: GRPO improves over base on Qwen (Arm D > B) — confirms GPT-OSS was the blocker ✅ DONE (+6%)
2. Phase 2a: GSM8K baseline too high — ceiling effect documented ✅ DONE (90.5% = no headroom)
3. Phase 2b: GRPO > base by >= 3% on MATH — replicates DeepSeekMath findings (IN PROGRESS)
4. No training instability (grad norm < 10, no NaN losses) ✅ CONFIRMED
5. Reproducible: all configs, data, and eval scripts committed ✅ DONE
