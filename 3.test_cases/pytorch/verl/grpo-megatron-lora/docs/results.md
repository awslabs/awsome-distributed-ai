# Results & Insights

Qwen3-235B-A22B, GRPO + LoRA (rank 128 / alpha 256), verl v0.8.0 + Megatron on SageMaker
HyperPod EKS, 48x B200, Ray/KubeRay, FSx Lustre, vLLM 0.20.2 runtime-LoRA serving.

Four training runs and the measurement work around them. Run 4 is the substantive one; the
follow-on sections are where the publicly defensible numbers live.

## The defensible claim

Both trained arms beat the untrained base on **public, contamination-cleared, deterministic**
math benchmarks. Greedy decoding means no sampling variance, so these need no replicate --
that is a statement about determinism, **not** a claim of replication.

| arm | GSM8K (n=1319, 5-shot, strict-match) | MATH-500 scoreable subset (n=472) |
|---|---|---|
| base | 0.8469 | 0.5212 |
| attention-only (`aug4b-s100`) | **0.8719** -- +0.0250 (2.83 sigma, p=0.006) | 0.5297 -- +0.0085 (0.52 sigma, null) |
| expert-FFN (`expert-s150`) | **0.8704** -- +0.0235 (2.60 sigma, p=0.012) | **0.5551** -- +0.0339 (2.11 sigma, p=0.048) |

The pre-registered primary (base -> `expert-s150`, GSM8K strict-match, win = >= +0.02 at
>= 1.96 sigma) **passes at +0.0235 / 2.60 sigma**. Both arms also clear the ~2.50 sigma
Bonferroni bar on GSM8K. On code, base -> `expert-s150` MBPP pass@4 is **+0.0300 (2.85
sigma)**, genuinely replicated and pooled.

**Expert-FFN placement -- the Run-4 hypothesis -- is NOT SUPPORTED.** 6/6 matched internal
nulls, an MBPP null, and a well-powered public null on GSM8K. The public null carries a
confound worth stating plainly (no matched step-150 control exists), so it is evidence
alongside the matched comparisons rather than a standalone result -- see
[Run 4](#run-4-expert-ffn-lora-placement----null-vs-the-control-2026-08-07--09).

## Five things worth keeping

**1. The internal headline is inflated by distribution.** Internal in-distribution MATH moved
**+0.0784 (7.57 sigma)**; public math moved roughly 3x less (+0.024 to +0.034), and the public
benchmark *closest* to the training distribution (MATH-500, competition-style, like the numina
training mix) moved **least** for the control arm. That is the opposite of the naive
prediction. The defensible capability statement is **"+0.024 to +0.034 on public math
benchmarks"** -- never +0.0784 unqualified.

**2. Never trust a new benchmark until its OWN gold answers score ~1.0 through YOUR scorer.**
This caught two disasters at zero GPU cost. BigCodeBench's canonical solutions scored **3/5**
under lm-eval's `code_eval`, because `reliability_guard()` disables the filesystem, network
and plotting those tasks legitimately need -- wrong executor, so the task is authored but
unusable without BigCodeBench's own Docker sandbox. MATH-500's own golds score **0.9440**
because sympy's `is_equiv` cannot compare tuples, intervals or matrices (20 / 5 / 3 of the 28
failures). MATH-500 *deltas* stay valid -- unscoreable items are fixed by ground truth alone,
so they are identical across arms and cancel in the paired difference -- but full-n absolutes
are ceilinged at 0.944, hence the pre-registered n=472 subset.

**3. The 105.7 GB expert adapter is 128 -> 4, not 128 -> 1.** Expert-parallel rank is 4; each
rank's adapter is materialised 32x by the HF-PEFT export. De-duplication is therefore
**21.67x -> 4.88 GB**, bit-exact verified across all 72,944 tensors -- *not* 342x / 0.8 GB.
"342x inflation vs trained parameters" and "recoverable redundancy" are different ratios and
must never be used interchangeably. A 128 -> 1 collapse would have silently discarded 3 of the
4 learned adapters and still loaded and served fine. De-dup does **not** unblock multi-adapter
serving: the GPU KV-cache slot is sized for the expanded 26.0B adapter regardless of on-disk
layout.

**4. Most of the gain was cheap; the tail was not.** 93% of the base -> step-150 move landed by
**step 50**. Steps 50-150 cost roughly **1,060 GPU-hours for +0.0057 (0.55 sigma)** -- the same
result was reachable for about a third of the spend.

**5. Reserved capacity can vanish before its stated EndTime.** A 6-node p6-b200 reservation was
reclaimed **~3h10m early** (stated end 11:30 UTC, actual ~08:20). Never plan work that must
complete in a reservation's final hours.

## How the numbers were produced

Paired per-problem analysis, not aggregate comparison: `pass@1`/`pass@4` via a validated
paired estimator (35/35 self-test), and a separate `exact_match` sibling for math keyed by
`(doc_id, filter)` because `gsm8k` emits one record per filter (2638 rows for 1319 docs).
Replication pools per-problem and reports the **measured** SE reduction beside the sqrt(R)
ceiling -- measured 1.14-1.18x against a 1.41x ceiling, which is why more *problems*, not more
*repeats*, is the lever that matters.

## Run 1: Eurus-only (mode collapsed)

| Field | Value |
|-------|-------|
| **Model** | Qwen3-235B-A22B, Megatron + LoRA (rank 32) |
| **Dataset** | Eurus (480K samples, math-dominant) |
| **Key params** | kl=0.001, lr=3e-5, max_response=4096 |
| **Duration** | ~99 hours, reached step 1231 / epoch 0 |
| **MLflow run** | `a4c842c5410342cdbec91f482bb9d9c7` |
| **Outcome** | Mode collapse -- stopped manually |

**What happened**: Training reward climbed (0.40 -> 0.57) while validation benchmarks
collapsed. Entropy crashed from 0.55 to 0.014 (model became essentially deterministic).
Code benchmarks peaked at step 50 (first eval) then declined 63-76%: APPS 16.4% -> 3.9%,
CodeContests 16.5% -> 6.2%. Math peaked around steps 200-500 then declined. Classic
reward hacking -- the model found a narrow strategy that scored well on training reward
but generalized poorly.

**Root causes identified**:

1. `kl_loss_coef=0.001` was ~100x too low -- provided zero braking against policy drift
2. `lr=3e-5` too aggressive for a 235B model -- amplified the collapse
3. DAPO-style asymmetric clipping (`clip_ratio_high=0.28`) amplified upward drift
4. Eurus dataset was math-dominant -- model over-specialized on math at the expense of code

**Best checkpoint**: Step 300 (best balanced performance before collapse). Preserved at
`/fsx/data/verl/ckpts/grpo-lora/megatron/Qwen3-235B-A22B/`.

## Run 2: Mixed code/math dataset (baseline control)

| Field | Value |
|-------|-------|
| **Model** | Qwen3-235B-A22B, Megatron + LoRA (rank 32) |
| **Dataset** | mixed-code-math (59K: TACO 43.6%, CC 17.2%, APPS 8.3%, math 30.7%) |
| **Key params** | kl=0.02, lr=1e-5, max_response=6144 |
| **Status** | Running (step 142 / 1851 as of 2026-04-15) |
| **MLflow run** | `b1b140ec344d44ef88041ff931c3739e` |

**Changes from Run 1**:

| Parameter | Run 1 | Run 2 | Why |
|-----------|-------|-------|-----|
| `kl_loss_coef` | 0.001 | **0.02** | 20x increase prevents entropy collapse |
| `learning_rate` | 3e-5 | **1e-5** | Less aggressive updates for 235B |
| `max_response_length` | 4096 | **6144** | More room for coding solutions |
| Dataset | Eurus (math-heavy) | **Mixed 70/30 code/math** | Balanced for agentic coding |
| Checkpoint dir | `ckpts/grpo-lora/` | **`ckpts/mixed-code-math/`** | Separate from Run 1 |

**Early results (step 142)**:

- Entropy stable at 0.78 (no collapse -- KL fix working)
- Code benchmarks already surpassing Run 1 at equivalent steps: APPS 23.2% vs 16.4%, CodeContests 17.0% vs 16.5%, Codeforces 20.3% vs 12.3%
- Math benchmarks solid: AMC/AIME 100%, Olympiads 29.0%
- Truncation at 50-55% (known issue -- may need max_response_length=8192+ in future run)
- SandboxFusion pod unstable (91 restarts in 8 days) -- intermittent code eval failures

## Run 3: KL trust-region test -- entropy collapse RECURRED (2026-08-06)

| Field | Value |
|-------|-------|
| **Model** | Qwen3-235B-A22B, Megatron + LoRA (rank 128 / alpha 256) |
| **Dataset** | mixed-code-math (train byte-identical to Run 2), val = mixed-code-math-valplus |
| **Key params** | **kl=0.001**, lr=1e-5, max_response=32768, max_num_seqs=56 |
| **Duration** | 9.86 h, stopped at step 35 |
| **Outcome** | **Stopped on 3 pre-registered tripwires -- entropy collapse** |

**Why it was run**: Run 2 changed four things at once versus Run 1 and credited `kl_loss_coef` for
fixing the collapse without ever testing it. Three of Run 1's four root causes have since been fixed
independently (truncation, `lr`, data mix), so this isolated KL as the single variable.

**What happened**: `actor/kl_loss` rose from 0.0234 to 0.1230 -- about **4x** the 0.030 plateau both
prior runs held, confirming that `kl_loss_coef=0.02` is indeed what pins it. But `actor/entropy`
collapsed **monotonically 0.7323 -> 0.1303** and was still falling at -0.008/step when stopped, while
the Run-2-style control held 0.61-0.69 flat over the identical step window. Truncation returned
(`response_length/clip_ratio` 0.0 for 18 steps, then 0.0755 / 0.0833). **Reward did not improve** --
`critic/score/mean` 0.4682 vs the control's 0.4735 over steps 21-35.

**Conclusion -- `kl_loss_coef=0.02` is CORRECT, not conservative. Do not lower it.**

With the cap at 32768, `lr` already 1e-5, and data already mixed 70/30, relaxing KL *still* collapses
entropy. So KL is doing independent regularisation work; it is not a proxy for the other three Run-1
root causes. The flat `kl_loss ~= 0.030` is an **equilibrium** where the KL penalty balances the policy
gradient, not a ceiling holding capability back.

**Why collapse is fatal specifically to GRPO**: the advantage is `(r_i - mean(r)) / std(r)` within a
group of `n_responses_per_prompt=4`, so the learning signal *is* the spread across the 4 samples. As
entropy falls the samples become near-identical, `std -> 0`, advantage `-> 0`, and the gradient
vanishes. The policy destroys the signal it learns from, which is why reward went flat as entropy fell.

**Caught in 9.9 h versus Run 1's 99 h**, on tripwires fixed before any data was seen:
`entropy < 0.50`; `clip_ratio > 0.05`; `pg_clipfrac > 0.05`; `corr(resp_len, score) < -0.20`;
`critic/score/mean` more than 2 sigma below the control's matched window. Keep these on any run that
touches KL, entropy, clipping, or the response cap.

**Tripwires are reporting-only.** They raise a flag for human review and **never stop a job
autonomously**; stopping a run always requires an explicit human decision. Where this document
records that a run *was* stopped (Run 3 at step 35), that was a human acting on a flag.

> **Tripwire correction (2026-08-07).** The original set also included "`entropy` fell more than 0.10
> over any 20-step window". **Do not use that rule before step ~20 — it has no specificity during
> warmup.** Applied to the identical step range 1..18, the *healthy* control trips it at -0.1166 and
> the healthy expert-LoRA run at -0.1267, both in window (1,10), which is exactly
> `lr_warmup_steps=10`. Entropy legitimately settles ~0.10-0.13 while the LR ramps. Trusting it would
> have aborted a healthy 48-GPU run.
>
> Use instead: **after step 10, flag for human review if the trailing 10-step linear slope of
> `actor/entropy` is < -0.005/step.** Validated at every eligible step: the control never fires across 135 evaluations
> (worst -0.00351), Run 3 fires at step 20 (worst -0.02478, 4.0x past). Post-warmup slope is
> +0.0017/step in the control and +0.0023/step in the expert-LoRA run versus **-0.0254/step** in
> Run 3 — opposite sign, an order of magnitude apart.
>
> The absolute floor `entropy < 0.50` remains the primary detector: it is unambiguous and, on Run 3,
> fired earlier (step 12 vs step 20). The slope rule earns its place on *slow* drift, which the floor
> would take 25+ steps to reach from a healthy 0.65.

If exploration ever needs raising, the principled lever is `entropy_coeff` (currently 0) -- a separate
term for exploration -- not the KL leash.

## Run 4: Expert-FFN LoRA placement -- NULL vs the control (2026-08-07 → 09)

| Field | Value |
|-------|-------|
| **Model** | Qwen3-235B-A22B, Megatron + LoRA (rank 128 / alpha 256) |
| **Dataset** | mixed-code-math (train byte-identical to Runs 2/3), val = mixed-code-math-valplus |
| **Key params** | kl=0.02, lr=1e-5, max_response=32768, **`target_modules=linear_qkv,linear_proj,linear_fc1,linear_fc2`** |
| **Duration** | 45.05 h, stopped at step 165 (target was 150) |
| **Health** | **0 Traceback / 0 OOM / 0 NCCL / 0 NaN** |
| **Outcome** | **Feasible and trains correctly, but NO measurable capability gain over attention-only** |

**Why**: Runs 1-3 eliminated truncation, context budget, adapter rank and the KL trust region
as the binding constraint on CODE. What was never tested is **placement**: LoRA adapts only
attention, 6.70B of 235B = **2.85% of weights**, while the 227B of MoE experts receive zero
adaptation. Single variable versus the Run-2-style control: only `lora_target_modules` changes.

**Feasibility result (both prior blockers were wrong)** -- see the "LoRA target modules" note
in the Qwen3-235B tuning section above for the full correction. Summary:

| | attention-only (control) | + expert layers (Run 4) |
|---|---|---|
| trainable params | 25,395,200 (0.13%) | **76,185,600 (0.39%)** |
| `actor/grad_norm`, steps 1-8 | 0.036-0.068 | **0.064-0.148 (mean 2.12x)** |
| `actor/ppo_kl` | -2.5e-4 .. 1.2e-4 | unchanged (~1e-4) |
| step time, post-warmup median s11-s58 | 672 s | **749 s (+11.4%, 7.10 sigma)** |
| HF adapter size | 1.63 GB | **105.7 GB (342x inflated)** |
| OOM / Tracebacks | -- | **0 / 0** |

`grad_norm` at 2.12x is the evidence gradient actually reaches the expert adapters; `ppo_kl` at
~1e-4 is the evidence the rollout policy still equals the training policy (so the merge handled
grouped-GEMM expert weights).

> **Step-time correction.** An earlier revision of this table claimed **+3.5%**, measured at step 6
> — before convergence. The honest figure on a matched post-warmup median is **+11.4%** (mean
> difference +127 s at 7.10 sigma). A later 862 s reading was also wrong: it was the median of steps
> 49-58, a window straddling the step-50 validation and a high-`timing_s/gen` cluster. `timing_s/step`
> has **no significant trend** over s11-s58 (+0.62 s/step, 0.50 sigma) and **95% of its variance is
> generation**. Projections from 58 steps (749/797) reproduced at 164 steps (748/798) to within 1 s.

### Result: strong vs base, NULL vs the control

Sample-weighted with a local rescoring script (not shipped); validations at steps 50/100/150.

| | base | s50 | s100 | s150 | base→s150 |
|---|---|---|---|---|---|
| extended MATH (n=4515) | 0.5238 | 0.5965 | 0.6016 | 0.6022 | **+0.0784 (7.57 sigma)** |
| weighted CODE (n=7601) | 0.7221 | 0.7367 | 0.7374 | 0.7344 | +0.0123 (1.71, inconclusive) |

**But every matched comparison against the attention-only control is null** (absolute, identical
rows): legacy MATH +0.0175 / **-0.0020** / +0.0155 at s50/s100/s150 (0.57 / 0.06 / 0.50 sigma);
weighted CODE +0.0019 / +0.0004 / -0.0021 (0.27 / 0.06 / 0.29 sigma). **6 of 6 null.**

The pre-registered gate (extended MATH >= 0.5648 at step 100) **passes at 0.6016**, but the gate is
**base-relative and the control would also pass it** (the control's own base→s100 legacy gain was
+0.0661). Passing it is not evidence that placement helped. The gate was not moved.

The gain is also **saturated**: 93% of the entire base→s150 move landed by step 50, and s100→s150 is
a **well-powered null** (+0.0006, 0.06 sigma, MDE +/-0.0202).

### External paired eval confirms it (the better-powered instrument)

`humaneval_p4` + `mbpp_p4`, runtime-LoRA vLLM, analysed per-problem with a local paired-eval
script (not shipped). Both step-100 arms were measured **twice** and pooled:

| comparison | HE p@1 | HE p@4 | MBPP p@1 | MBPP p@4 |
|---|---|---|---|---|
| **aug4b-s100 vs expert-s100 (POOLED)** | -0.0160 (1.11) | -0.0244 (1.38) | **+0.0018 (0.26)** | +0.0070 (0.87) |
| MDE at 1.96 sigma | +/-0.0284 | +/-0.0347 | **+/-0.0131** | +/-0.0158 |

Nothing clears 1.96, let alone the ~2.50 Bonferroni threshold for 4 comparisons. **MBPP is a
well-powered null.** HumanEval is tighter than the single-run MDE (+/-0.0329) but still cannot
exclude the +0.0259 effect aug4b itself showed over base, and what signal exists points *against*
the expert arm -- so state it as "at best neutral on HumanEval", not as a clean null.

**Conclusion: "attention-only was the binding constraint because the 227B of MoE experts got zero
adaptation" is NOT SUPPORTED.** Tripling trainable parameters and demonstrably delivering gradient to
the experts bought no measurable capability, at +11.4% throughput and a 105.7 GB adapter.

### Follow-on (2026-08-10): PUBLIC math benchmarks — both arms beat base

Every earlier MATH number was on an internal, in-distribution holdout. These are public
benchmarks, verified absent from the training mix (`gsm8k` and `hendrycks/MATH` appear
nowhere in apps/taco/codeforces/codecontests + numina_*). Greedy decoding, so the
measurement is **deterministic** — no replicate needed. Analysed paired per-problem with a
local math paired-eval script (not shipped), cross-validated against two published reference
results.

| arm | GSM8K (n=1319, 5-shot, strict-match) | MATH-500 scoreable subset (n=472) |
|---|---|---|
| base | 0.8469 | 0.5212 |
| **attention-only (aug4b-s100)** | **0.8719** — +0.0250 vs base (**2.83σ**, p=0.006) | 0.5297 — +0.0085 (0.52σ, null) |
| **expert-FFN (expert-s150)** | **0.8704** — +0.0235 vs base (**2.60σ**, p=0.012) | **0.5551** — +0.0339 (**2.11σ**, p=0.048) |

**Both trained arms beat base on public GSM8K past the ~2.50σ Bonferroni bar.** The
pre-registered primary (base → expert-s150 on GSM8K strict-match, win = ≥ +0.02 at ≥ 1.96σ)
**passes at +0.0235 / 2.60σ.** MATH-500 clears 1.96σ but not Bonferroni.

**Placement is still NOT SUPPORTED, now on a well-powered public instrument.** On the
pre-registered primary filter, `aug4b-s100` → `expert-s150` on GSM8K is a null:
**−0.0015 at 0.19σ (55 better / 57 worse, sign p=0.925), MDE ±0.0157**. The
flexible-extract filter corroborates it and is an exact tie: **+0.0000 at 0.00σ
(53 / 53, sign p=1.000), MDE ±0.0153**. Either MDE is *smaller* than the
+0.0235/+0.0250 each arm gains over base, so the instrument demonstrably can detect an
effect of the size training produced, and sees none between placements.

> **Read this comparison as a null on (placement + 50 extra steps), not on placement
> alone.** No matched step-150 control exists — the attention-only arm ran `save_freq=20`
> (20/40/…/140, no step 150), so this pair differs in *both* target modules and training
> length. The step term is very likely negligible: 93% of the base→s150 move landed by step
> 50, and the internal s100→s150 comparison is itself a well-powered null (+0.0006, 0.06σ,
> MDE ±0.0202). But a null on a sum is not strictly a null on each part — two opposing
> effects of about ±0.02 could cancel. The claim rests on this plus the 6/6 matched internal
> nulls and the MBPP null, not on this comparison by itself.

MATH-500 is the one place expert placement has ever pointed positive on a clean instrument:
**+0.0240 at 1.86σ over n=500** (equivalently **+0.0254 at 1.86σ** on the n=472 scoreable
subset), sign p=0.088, |Δ| ≈ 95% of the ±0.0253 MDE. It does **not** reach 1.96σ and must not
be reported as evidence. Because these runs are deterministic a replicate cannot tighten it;
only more problems can (`minerva_math`, n=5000, would take the MDE to ~±0.008).

> **This re-scales the internal headline.** Internal in-distribution MATH moved +0.0784
> (7.57σ); the public gains are **~3x smaller (+0.024 to +0.034)**, and the public benchmark
> *closest* to the training distribution (MATH-500) moves least for the control. So +0.0784
> is substantially an in-distribution effect. The defensible capability claim is
> **"+0.024 to +0.034 on public math benchmarks"**, not +0.0784.

Instrument notes: MATH-500's own gold answers score only 0.9440 through Minerva's `is_equiv`
(sympy cannot compare tuples/intervals/matrices), so full-n MATH-500 absolutes have a 0.944
ceiling and are not leaderboard-comparable — hence the pre-registered n=472 scoreable subset.
GSM8K's gold gate is 1.0000. Recorded in the local measurement notes (not shipped).

> **Always name the denominator on a MATH-500 delta.** The 28 gold-unscoreable items are
> fixed by ground truth alone, so they enter the paired difference as ties and dilute it:
> the same effect reads **+0.0339 over n=472** but **+0.0320 over n=500**
> (`0.0339 × 472/500 = 0.0320`). Both are correct; an unlabelled figure is not. The same
> applies to naming the *pair* on any quoted delta — `aug4b-vs-base` and
> `expert-s150-vs-base` happen to share a mean delta of +0.0259 on HumanEval pass@1 while
> differing completely in per-problem structure.

### Follow-on (2026-08-09): replication + a larger-code-set attempt

- **The one Bonferroni-clearing external number was replicated and shrank.** `expert-s150 vs base`
  MBPP p@4 read **+0.0400 (3.04 sigma)** on a single run; both arms were then re-measured and pooled
  (effective repeats 4->8). The pooled delta is **+0.0300 (2.85 sigma; inverse-variance 3.30 sigma;
  sign p=0.014)** -- still real and past Bonferroni, but ~25% smaller: the single-run point estimate
  rode a favorable replicate (the 2nd replicate alone gave +0.0200). This is a **vs-base** result, not
  a placement result -- aug4b beats base similarly, and the step-100 placement comparison stays null.
- **Base is a STABLE arm.** Its first-ever replicate: HE p@1 0.2774/0.2729, MBPP p@1 0.7845/0.7915. So
  the base denominator every vs-base number rests on is trustworthy; prior HE vs-base noise came from
  the trained arms (§8.2), not base. Measured pooled SE reduction 1.15-1.18x vs the sqrt(2)=1.41x
  ceiling — replication is a weak lever.
- **A larger code eval set (BigCodeBench, n=1140) was authored but is NOT yet usable.** The task
  (`kubernetes/lmeval-tasks/bigcodebench_p4.yaml` + `utils.py` builders, completion-mode, added
  additively) is ready, but a canonical-solution HARD GATE proved lm-eval's `code_eval` executor is
  the WRONG scorer for it: `code_eval`'s `reliability_guard()` disables the filesystem/network/plotting
  that BigCodeBench tasks require, so even ground-truth solutions score only ~3/5. BigCodeBench needs
  its own sandbox executor (its official Docker image) -- a generate-then-score split is the next-
  reservation task. No BigCodeBench model number was produced.

### Serving an expert-LoRA adapter: three non-default settings are REQUIRED

The 342x adapter inflation is a **binding operational constraint**, not a cosmetic wart. Each of
these was diagnosed from a failure; none of them affects scores:

| symptom | root cause | required setting |
|---|---|---|
| vLLM `CrashLoopBackOff` | `No available memory for the cache blocks`; **KV cache -23.21 GiB**. `max_loras=N` sizes every slot for the EXPERT-EXPANDED 26.0B adapter (~25 GiB/GPU/slot) | `VLLM_MAX_LORAS=1` + `VLLM_GPU_MEM_UTIL=0.95` (-> +69.6 GiB) |
| first request HTTP 500 | `TimeoutError: RPC call to sample_tokens timed out` -- the adapter load exceeds the 300 s default | `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600` (adapter then loads in ~38 min) |
| worker dies mid-eval, driver dies with `RuntimeError: Session is closed` | **OOMKilled exit 137** -- HOST RAM. 8 workers x 84.6 GB = ~677 GB with ONE expert adapter vs the container's 1Ti limit | **one adapter per server**; do not co-register a second |

Also measured: an adapter **swap** costs **~25 min**; with a single registered adapter the same
request returns in **3 s**. So arms cannot share a server instance and each needs its own ~50 min
load. **A de-duplicating export is the highest-value fix for iterating on expert-LoRA runs** -- the
inflation exists because `share_expert_adapters=True` has no HF-PEFT representation, so the export
materialises one adapter per expert (26.0B params vs 76.2M trained).

> **De-dup measured (2026-08-09).** The redundancy is **128 experts -> 4 EP-rank groups of 32**
> (verified bit-identical within a group, distinct across groups), NOT 128 identical copies. A
> byte-exact F32 de-dup of the step-150 adapter therefore goes **105.7 GB -> 4.88 GB (21.67x;
> 26.428B -> 1.220B params)**, confirmed by a bit-exact expand round trip on all 72,944 tensors.
> (The "342x / ~0.8 GB" figure conflated inflation-vs-trained-params with recoverable redundancy.)
> This shrinks disk/transfer/load; it does NOT shrink the GPU KV-cache/slot (still the expanded
> 26.0B), so on vLLM 0.20.2 it does not by itself unblock multi-adapter serving. The prototype
> exporter is local and not shipped with this test case.
