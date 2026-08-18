<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->
# Evaluating a Saved Checkpoint

End-to-end guide for taking a verl v0.8.0 training checkpoint (Megatron + LoRA
or FSDP + LoRA) and producing comparable HumanEval / MBPP pass@k numbers plus
a validation-split reward score, then rolling them up into a learning curve
against a baseline.

Worked example throughout: **Qwen3-235B-A22B, mixed-code-math-v2 dataset,
Megatron backend, MLflow training run id `3322f50103bf4ff1` (experiment
`GRPO-Megatron`)**.

## Pipeline Overview

```
┌──────────────────────────────────────────────┐
│ Training checkpoint on FSx (verl v0.8.0)     │
│ ckpts/mixed-code-math-v2/megatron/           │
│   Qwen3-235B-A22B/global_step_<N>/           │
│     actor/                                   │
│       huggingface/adapter/                   │
│         adapter_config.json                  │
│         adapter_model.safetensors  (~388 MB) │
│       huggingface/  (tokenizer + config)     │
│       (dist_ckpt/... Megatron shards)        │
└──────────────────┬───────────────────────────┘
                   │ Phase 1: merge_adapter.sh <N>
                   │           (peft merge_and_unload, 8 GPUs / 1 node)
                   ▼
┌──────────────────────────────────────────────┐
│ Merged HF model on FSx                       │
│ merged/Qwen3-235B-A22B/step_<N>/             │
│   *.safetensors, config.json,                │
│   tokenizer.json, ...                        │
│  + check_merge_parity.py (logit-parity gate) │
└──────────────────┬───────────────────────────┘
                   │ Phase 2: deploy_vllm_eval.sh
                   │  (TP=8 on 1× p6-b200, training ECR image, ~90 min startup)
                   ▼
┌──────────────────────────────────────────────┐
│ vLLM OpenAI-compatible server                │
│ svc/vllm-eval:8000  (/v1)                    │
└───────┬─────────────────────────────┬────────┘
        │                             │
   Phase 3a                       Phase 3b
   submit_lmeval.sh               submit_val_eval.sh
   (CPU driver Job,               (Ray job, vLLM +
    lm-evaluation-harness,         Sandbox Fusion via
    local-completions)             custom_reward_fn.py)
   HumanEval + MBPP               val.parquet reward
        │                             │
        └──────────────┬──────────────┘
                       │ Phase 5: compare_eval_results.py --curve
                       ▼
┌──────────────────────────────────────────────┐
│ Markdown table + MLflow curve                │
│ eval_results/qwen3-235b-curve.md             │
│ MLflow run 3322f50103bf4ff1, step=<N>        │
└──────────────────────────────────────────────┘
```

## Prerequisites

- **Training job stopped** and the Ray worker group scaled down by ≥1 replica
  so vLLM has a free GPU node to land on. Stopping the Ray *job* returns
  logical resources but the Ray *worker pods* keep holding
  `nvidia.com/gpu` reservations. Use:
  ```bash
  ./scripts/scale_ray_workers.sh 0     # or any count < total worker replicas
  ```
- `fsx-utils` pod running (`kubectl get pod fsx-utils`) for read-only ckpt
  inspection.
- `sandbox-fusion` Deployment healthy (`kubectl get pods -l app=sandbox-fusion`)
  — required for the val-split eval's code reward path.
- Training-image tag available in ECR (`${REGISTRY}${IMAGE}:${TAG}`) — the
  vLLM **server** uses this same image for numerical parity with training
  rollouts (see Key Design Choices).
- `env_vars` sourced in your shell.

## Phase 0 — Inspect the Checkpoint (read-only)

Under verl v0.8.0, every Megatron+LoRA save writes a native HF PEFT adapter
alongside the Megatron dist-ckpt shards. Confirm it exists before merging:

```bash
STEP=350
MODEL=Qwen3-235B-A22B

kubectl exec fsx-utils -- ls -lh \
    "/fsx/data/verl/ckpts/mixed-code-math-v2/megatron/${MODEL}/global_step_${STEP}/actor/huggingface/adapter/"
```

Expected contents:

```
adapter_config.json           # r=32, lora_alpha=64, target_modules=[q_proj,k_proj,v_proj,o_proj]
adapter_model.safetensors     # ~388 MB, native HF PEFT format
```

If this directory is missing, the checkpoint predates the v0.8.0 upgrade and
cannot be merged with `merge_adapter.sh` — regenerate a fresh checkpoint from
current `main` or fall back to a compatible verl release.

## Phase 0.5 — Free a GPU Node

vLLM eval needs one full p6-b200 node (8 GPUs, TP=8). Scale the Ray worker
group down first — this evicts one pod and lets Karpenter/HP hand the node
to vLLM:

```bash
./scripts/scale_ray_workers.sh 0     # frees all worker nodes; use a smaller step for partial scale-down
```

This performs an indexed JSON patch on the Ray CR, so it is safe to run
repeatedly.

## Phase 1 — Merge the Adapter into a HuggingFace Model

Under verl v0.8.0 there is no Megatron replay merger — the on-disk adapter
is already in native HF PEFT format, so merging is a plain
`PeftModel.from_pretrained(base, adapter).merge_and_unload()`:

```bash
# The provenance gate is fail-closed: state the rank/alpha the run was trained at, so a
# checkpoint from a different experiment cannot be merged silently. Use
# ALLOW_UNVERIFIED_ADAPTER=1 only if you accept an unverifiable merge.
EXPECT_LORA_RANK=32 EXPECT_LORA_ALPHA=64 ./scripts/merge_adapter.sh 350            # full merge
EXPECT_LORA_RANK=32 EXPECT_LORA_ALPHA=64 DRY_RUN=1 ./scripts/merge_adapter.sh 350  # gates only, no export
```

What it does:

1. Submits a Ray job that pins to a single node (8 GPUs, `device_map=auto`).
2. Loads the base model from `/fsx/data/verl/models/Qwen3-235B-A22B`.
3. Loads the adapter from
   `/fsx/data/verl/ckpts/mixed-code-math-v2/megatron/Qwen3-235B-A22B/global_step_350/actor/huggingface/adapter/`.
4. **Gate A (pre-merge):** asserts adapter tensors are non-zero.
5. Calls `merge_and_unload()`.
6. **Gate B (post-merge):** samples base vs merged weights and asserts
   a non-trivial delta (guards against silent base-only merge).
7. Saves the merged HF model + tokenizer to
   `/fsx/data/verl/merged/Qwen3-235B-A22B/step_350/`.

After it finishes, ALWAYS run the full logit-parity check before spending eval
GPU-hours:

```bash
python scripts/check_merge_parity.py \
    --base-model   /fsx/data/verl/models/Qwen3-235B-A22B \
    --merged-model /fsx/data/verl/merged/Qwen3-235B-A22B/step_350
```

## Phase 2 — Deploy vLLM

```bash
export VLLM_MODEL_PATH=/fsx/data/verl/merged/Qwen3-235B-A22B/step_350
export VLLM_SERVED_NAME=qwen3-235b-step350

./scripts/deploy_vllm_eval.sh
```

What happens:

1. `envsubst` renders `kubernetes/vllm-eval.yaml`.
2. Applies the Deployment + ClusterIP Service (`svc/vllm-eval:8000`, path
   prefix `/v1`).
3. nodeAffinity accepts both `p6-b200.48xlarge` and `ml.p6-b200.48xlarge`.
4. `startupProbe` allows up to ~90 min — 235B MoE at TP=8 is slow to
   materialise from cold FSx.

The Deployment uses the **training ECR image** (`${REGISTRY}${IMAGE}:${TAG}`)
so that vLLM 0.20.2 + `VLLM_LORA_DISABLE_PDL=1` matches the training rollout
stack byte-for-byte.

**Verify:**
```bash
kubectl exec fsx-utils -- \
    wget -qO- http://vllm-eval.default.svc.cluster.local:8000/v1/models
```

**Tear down:**
```bash
./scripts/deploy_vllm_eval.sh --delete
```

## Phase 3a — HumanEval + MBPP via lm-evaluation-harness

```bash
./scripts/submit_lmeval.sh qwen3-235b-step350
LMEVAL_LIMIT=5 ./scripts/submit_lmeval.sh qwen3-235b-step350   # smoke test
```

This applies `kubernetes/lmeval-job.yaml` — a CPU-only driver Job that:

- Runs `python:3.11.15-slim` (the **driver**; no GPU, no model reload).
- Pip-installs CPU `torch==2.13.0+cpu`, `lm-eval[api]==0.4.9` and
  `transformers==4.49.0` at start — the versions the reported runs recorded (no phantom
  `ghcr.io` image, no `ghcr-pull` secret).
- Calls `lm_eval --model local-completions --model_args
  base_url=http://vllm-eval:8000/v1,model=${VLLM_SERVED_NAME} ...
  --tasks humaneval_p4,mbpp_p4` with the chat template **OFF**.
  **Both are mandatory explicit overrides** — see the `--apply_chat_template` note
  in the Design Decisions section below. The script's own defaults
  (`LMEVAL_APPLY_CHAT_TEMPLATE=true`, stock `humaneval,mbpp`) produce wrong numbers.
- Reuses the warm `vllm-eval` service in Phase 2 — no separate GPU cost.

Results land at:
```
/fsx/data/verl/eval_results/qwen3-235b-step350/lmeval/results.json
```

`results.json` follows the standard lm-eval schema:
`results.humaneval.pass@1`, `results.humaneval.pass@10`,
`results.mbpp.pass@1`, `results.mbpp.pass@10`.

**Monitor:**
```bash
kubectl get jobs -l eval-run=qwen3-235b-step350
kubectl logs -f job/lmeval-qwen3-235b-step350
```

## Phase 3b — Validation-Split Reward

```bash
./scripts/submit_val_eval.sh qwen3-235b-step350
VAL_LIMIT=100 ./scripts/submit_val_eval.sh qwen3-235b-step350   # smoke test
```

Submits `scripts/eval_val_split.py` as a Ray job. For each row in
`/fsx/data/verl/data/mixed-code-math/val.parquet` (note: the val parquet is
under `mixed-code-math`, **not** `mixed-code-math-v2` — the `-v2` suffix is
only on the checkpoint tree):

1. Generates `n=4` responses via `svc/vllm-eval` (matches training's
   `n_responses_per_prompt=4`).
2. Scores each response via `scripts/custom_reward_fn.py` — the same wrapper
   the training loop uses, routing code tasks to Sandbox Fusion and math
   tasks to `prime_math`.

This is the only metric that directly measures **"did the checkpoint get
better at what we trained it for on held-out data?"**

Output: `/fsx/data/verl/eval_results/qwen3-235b-step350/val_split.json`
(overall mean reward + per-`data_source` breakdown: codecontests, apps,
taco, numina_*, …).

## Phase 4 — Repeat for Baseline and Additional Steps

Tear down the current server, then redeploy pointed at the base model
(no merge required for the base) or the next step:

```bash
./scripts/deploy_vllm_eval.sh --delete

# Base model (no Phase 1 needed)
export VLLM_MODEL_PATH=/fsx/data/verl/models/Qwen3-235B-A22B
export VLLM_SERVED_NAME=qwen3-235b-base
./scripts/deploy_vllm_eval.sh
./scripts/submit_lmeval.sh qwen3-235b-base
./scripts/submit_val_eval.sh qwen3-235b-base
./scripts/deploy_vllm_eval.sh --delete

# Next step (loop Phases 1 → 3 for each STEP in {50, 350, 750, ...})
```

## Phase 5 — Aggregate, Compare, and Log a Learning Curve

Single-run mode (one step vs baseline):

```bash
python3 scripts/compare_eval_results.py \
    --run qwen3-235b-step350 \
    --baseline qwen3-235b-base \
    --results-dir /fsx/data/verl/eval_results \
    --markdown-out /fsx/data/verl/eval_results/qwen3-235b-step350/comparison.md \
    --mlflow-run-id 3322f50103bf4ff1
```

Learning-curve mode (many steps vs baseline, one row per step + Δ-vs-base row,
per-step MLflow metrics with `step=<N>` so they render as a curve):

```bash
python3 scripts/compare_eval_results.py \
    --curve \
    --steps 50,350,750 \
    --baseline qwen3-235b-base \
    --run-prefix qwen3-235b- \
    --markdown-out /fsx/data/verl/eval_results/qwen3-235b-curve.md \
    --mlflow-run-id 3322f50103bf4ff1
```

The comparator:

1. Loads `lmeval/results.json` for each run and pulls
   `results.humaneval.pass@1`, `results.humaneval.pass@10`,
   `results.mbpp.pass@1`, `results.mbpp.pass@10`.
2. Loads each `val_split.json` (overall + per-`data_source`).
3. Prints a markdown table (single mode) or step-indexed curve table (curve
   mode) with an explicit Δ-vs-base row.
4. If `--mlflow-run-id` and `MLFLOW_TRACKING_URI` are set, logs each metric
   to the training run under `eval/<run>/...` — in curve mode with
   `step=<N>` so MLflow renders it as a curve on the same dashboard as
   training loss.

## What Gets Measured

| Metric | Source | Notes |
|---|---|---|
| `humaneval.pass@1` | lm-eval (local-completions) | Sampling with harness defaults (temperature 0.2). |
| `humaneval.pass@10` | lm-eval (local-completions) | Harness computes pass@10 from its own samples. |
| `mbpp.pass@1` | lm-eval (local-completions) | Same driver, same warm vLLM. |
| `mbpp.pass@10` | lm-eval (local-completions) |  |
| val-split mean reward | `eval_val_split.py` + `custom_reward_fn.py` | Same reward the training loop optimizes. |
| val-split per-`data_source` mean | as above | codecontests, apps, taco, numina_math, numina_amc, … |

## Key Design Choices

- **Native v0.8.0 adapter → trivial peft merge.** Under verl v0.8.0, every
  Megatron+LoRA save emits a native HuggingFace PEFT adapter at
  `actor/huggingface/adapter/`. Merging is a plain
  `PeftModel.from_pretrained(base, adapter).merge_and_unload()` — the old
  ~370-line Megatron-Bridge "replay merger" is gone.
- **vLLM SERVER uses the training ECR image; the lm-eval DRIVER uses a
  published image.** The server (`kubernetes/vllm-eval.yaml`) points at
  `${REGISTRY}${IMAGE}:${TAG}` so vLLM 0.20.2 + `VLLM_LORA_DISABLE_PDL=1`
  matches training rollouts exactly. The lm-eval driver only issues HTTP
  requests, so it uses a plain pinned `python:3.11.15-slim` and installs the CPU-only
  deps it needs — it never loads a model, so it does not need a vLLM image at all.
- **lm-eval uses `local-completions` with the chat template OFF.**
  `local-chat-completions` gives `pass@1=0` on instruct-tuned models because the
  harness's default prompt template is wrong for chat models, so
  `local-completions` is correct.

  **The `--apply_chat_template` half of this guidance is SUPERSEDED (2026-08).**
  It was right when the tasks were stock `humaneval`/`mbpp`, but the custom
  `humaneval_p4`/`mbpp_p4` tasks now carry a **completion-mode `until` list**
  (`['<|im_end|>', '\nclass', '\ndef', '\n#', '\nif', '\nprint']`) that is coupled
  to the template being **off**. Turning the template on makes `"\ndef"` fire on a
  fenced ` ```python ` block and truncate every generation — the stop-sequence
  bug in reverse. Required:

  ```bash
  export LMEVAL_APPLY_CHAT_TEMPLATE=false          # NOT the script default (true)
  export LMEVAL_TASKS="humaneval_p4,mbpp_p4"       # NOT stock humaneval,mbpp
  export LMEVAL_GEN_KWARGS="temperature=0.2,top_p=0.95,max_gen_toks=8192"
  export LMEVAL_CONCURRENCY=32                     # identical across arms is MANDATORY
  export LMEVAL_TIMEOUT=7200
  export LMEVAL_MAX_LENGTH=32768
  ```

  Stock `humaneval,mbpp` default to `max_gen_toks=1024` and reproduce the **retired
  0.2634** HumanEval artifact. `--gen_kwargs` MERGES rather than replaces, so the
  `until` list survives the gen_kwargs above (verified in the accepted `results.json`).

  Absolute HumanEval values in this regime are **not** comparable to the published
  ~0.85 — raw completion mode is off-distribution for a chat-trained model. Only the
  **paired delta** is valid, and only between arms sharing the same regime.
- **Val-split uses the same `custom_reward_fn.py` as training** (including
  Sandbox Fusion for code), so val-split reward is a direct measure of
  training-objective progress on held-out data.
- **Two safety gates in `merge_adapter.sh`** — non-zero adapter tensor check
  pre-merge and base-vs-merged weight delta check post-merge — plus the
  full logit-parity gate in `scripts/check_merge_parity.py` before any
  eval GPU-hours are spent.
- **All eval results live under `/fsx/data/verl/eval_results/<run>/`** and
  replicate to S3 via the existing FSx Data Repository Association.

## Troubleshooting

### lm-eval driver reports `pass@1 = 0` on every problem
Two distinct causes — check them in this order.

1. **You used `--model local-chat-completions`.** On instruct-tuned models this
   uses a chat template that doesn't match what the model was trained on. Fix:
   `--model local-completions`.
2. **You ran the `_p4` tasks with the chat template ON.** The `_p4` `until` list is
   the completion-mode list, so `"\ndef"` fires on a fenced ` ```python ` block and
   truncates every generation to nothing (the stop-sequence bug in reverse). Fix:
   `LMEVAL_APPLY_CHAT_TEMPLATE=false`. **This is NOT the script default** —
   `submit_lmeval.sh` still defaults it to `true`, which is stale.

Diagnose which one you hit by reading the `--log_samples` dump: cause 1 gives
well-formed prose that fails extraction, cause 2 gives near-empty completions.
Do not "fix" a truncation problem by turning the chat template back on — the
template and the `until` list are coupled and must move together.

### lm-eval can't find `humaneval` or `mbpp`
There is a local directory named `humaneval/` or `mbpp/` in the driver's
cwd shadowing the task name. `submit_lmeval.sh` runs the harness from a
clean cwd (`/tmp/lmeval-run`) precisely to avoid this — verify nothing has
mounted a conflicting dir in.

### vLLM worker pods stay `Pending`
Either (a) no GPU node available — run `./scripts/scale_ray_workers.sh 0`,
or (b) the flexible training plan (FTP) has expired and there is no live
capacity. Check node availability:
```bash
kubectl get nodes -l node.kubernetes.io/instance-type=ml.p6-b200.48xlarge
```

### vLLM takes > 30 min to load
Expected. 235B MoE at TP=8 off cold FSx routinely takes 60–90 min to
materialise. The `startupProbe` in `kubernetes/vllm-eval.yaml` allows
this. Stream the pod logs to confirm progress:
```bash
kubectl logs -f deploy/vllm-eval
```
If it fails after ≥90 min, bump `startupProbe.failureThreshold`.

### Sandbox Fusion errors during val-split
Not caused by lm-eval or vLLM. `scripts/eval_val_split.py` counts errors
per `data_source` — if `codecontests`/`apps` error rates are high, roll
sandbox-fusion:
```bash
kubectl rollout restart deploy/sandbox-fusion
```
Then rerun only the val-split step (lm-eval results are unaffected).

### CUDA graph capture OOM on first request
The manifest sets `--enforce-eager`; keep it on. vLLM CUDA graphs don't
play well with 235B MoE router paths and save little throughput on long
generations.
