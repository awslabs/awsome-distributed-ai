# AGENTS.md — grpo-megatron-lora

## What This Test Case Is

Infrastructure-as-code for GRPO training on Kubernetes. Shell scripts, a Dockerfile,
Hydra config (YAML), and one Python entry point (`scripts/submit_training.py`) that
orchestrate the upstream [verl](https://github.com/volcengine/verl) framework with
Megatron-LM for distributed RL fine-tuning of LLMs. Supports LoRA and full fine-tuning.

There are **no Python packages to install locally**. The training code is verl (installed
at job submission time via `scripts/runtime_env.yaml`). The only Python files here are
`scripts/submit_training.py` (Hydra job submitter), `scripts/custom_reward_fn.py`
(reward wrapper), and data prep scripts under `data/`.

## Directory Layout

```
conf/                  # Hydra config groups (THE source of truth for training params)
  config.yaml          # Root config: hyperparams, Ray, secrets, defaults
  backend/             # fsdp.yaml, megatron.yaml
  cluster/             # p6-b200-6node.yaml, p6-b200-1node.yaml
  model/               # qwen3-8b, qwen25-coder-7b, qwen25-72b, qwen3-coder-next, qwen3-30b-a3b, qwen3-235b
  lora/                # enabled.yaml, disabled.yaml
  data/                # eurus.yaml, mixed.yaml, mixed-hard.yaml, mixed-valplus.yaml
  sandbox/             # enabled.yaml, disabled.yaml
  tracking/            # mlflow.yaml, console.yaml
  profiling/           # disabled.yaml, torch.yaml, nsys.yaml, memory.yaml
scripts/
  submit_training.py   # PRIMARY ENTRY POINT — Hydra config -> verl CLI -> ray job submit
  custom_reward_fn.py  # Reward function (copied into Docker image at /workspace/)
  runtime_env.yaml     # Pip packages installed on Ray nodes at job start
  merge_adapter.sh     # EVAL: merge native HF PEFT adapter -> HF via peft merge_and_unload
  merge_adapter.py     # EVAL: Ray-launched merge driver (8 GPUs, device_map=auto, safety gates)
  check_merge_parity.py  # EVAL: logit-parity gate to run after merge_adapter
  deploy_vllm_eval.sh  # EVAL: deploy/teardown vllm-eval Deployment + Service
  submit_lmeval.sh     # EVAL: HumanEval+MBPP via lm-evaluation-harness against vllm-eval
  submit_val_eval.sh   # EVAL: val-split reward as Ray job (uses Sandbox Fusion)
  eval_val_split.py    # EVAL: vLLM + custom_reward_fn scoring of val parquet
  compare_eval_results.py  # EVAL: aggregate lm-eval JSONs + markdown + MLflow (single + curve modes)
  scale_ray_workers.sh # EVAL: indexed JSON patch to free a GPU node for vllm-eval
  test_lmeval_utils.py     # GATE: lm-eval task builders (exits non-zero on failure)
  test_merge_provenance.py # GATE: adapter-merge provenance
  test_reward_routing.py   # GATE: reward routing per data_source
data/                  # prepare_data.py, mix_datasets.py, filter_difficulty.py,
                       #   build_val_holdout.py, submit_data_prep.sh
models/                # Model download helpers (submit_download.sh, per-model scripts)
kubernetes/         # fsx-utils, sandbox-fusion, vllm-eval, lmeval-job, valeval-job,
                       #   eval-mlflow-log-job (all envsubst templates)
  lmeval-tasks/        # Custom lm-eval tasks: humaneval_p4/p10, mbpp_p1/p4/p10,
                       #   bigcodebench_p4 (authored, NOT usable -- see pitfalls), utils.py
lustre/                # FSx PV/PVC/StorageClass (envsubst templates, needs FSX_*)
docs/                  # cluster, configuration, eval-pipeline, results, profiling,
                       #   troubleshooting, parallelism-strategies.svg
Dockerfile             # EFA + Megatron-Bridge + blinker fix + vLLM LoRA patch
build-push.sh          # Docker build + ECR push (at the test-case root, NOT setup/)
env_vars.example       # Template for env_vars (infra-only: AWS, K8s, secrets, NCCL)
```

## Key Commands

### Submit training (primary workflow)
```bash
source env_vars

# Reach the Ray dashboard. Default is a port-forward (no auth needed):
kubectl port-forward -n "${KUBE_NAMESPACE:-default}" svc/<ray-head-svc> 8265:8265
export RAY_ADDRESS="http://localhost:8265"
# Behind an authenticating proxy, also export RAY_HEADERS -- see docs/cluster.md

python3 scripts/submit_training.py                          # defaults: Qwen3-8B, FSDP, 6-node, LoRA, MLflow
python3 scripts/submit_training.py backend=megatron model=qwen25-72b
python3 scripts/submit_training.py cluster=p6-b200-1node tracking=console  # quick validation
python3 scripts/submit_training.py lora=disabled            # full fine-tuning
python3 scripts/submit_training.py training.learning_rate=1e-4 training.train_batch_size=128
python3 scripts/submit_training.py --cfg job                # dry run: print resolved config
```

### Docker build and push
```bash
source env_vars
./build-push.sh
```

### Deploy K8s services
```bash
source env_vars
# lustre/ is applied the same way (needs FSX_* in env_vars) -- see docs/cluster.md
envsubst < lustre/storageclass.yaml | kubectl apply -f -
envsubst < lustre/pv.yaml | kubectl apply -f -
envsubst < lustre/pvc.yaml | kubectl apply -f -
envsubst < kubernetes/sandbox-fusion.yaml | kubectl apply -f -
envsubst < kubernetes/fsx-utils.yaml | kubectl apply -f -
```

### Data prep
```bash
./data/submit_data_prep.sh --datasets eurus apps taco codecontests --output-dir /fsx/data/verl/data
```

### Model download to FSx
```bash
./models/submit_download.sh Qwen/Qwen3-8B /fsx/data/verl/models/Qwen3-8B
```

### Ray job management
```bash
ray job list
ray job logs <submission_id>
ray job stop <submission_id>

# Add --headers "$RAY_HEADERS" to each command only when RAY_HEADERS is set
# (authenticating proxy). A port-forward needs no headers.
```

### Evaluation (post-training)

Full walkthrough: `docs/eval-pipeline.md`.

```bash
# 0. Inspect checkpoint (read-only)
kubectl exec fsx-utils -- ls -lh \
    /fsx/data/verl/ckpts/mixed-code-math-v2/megatron/<model>/global_step_<N>/actor/huggingface/adapter/

# 1. Merge the native HF PEFT adapter into a standalone HF model (peft merge_and_unload)
./scripts/merge_adapter.sh <step>                                # defaults to Qwen3-235B-A22B/mixed-code-math-v2/megatron

# 2. Deploy vLLM (TP=8 on 1 p6-b200 node, uses training ECR image)
export VLLM_MODEL_PATH=/fsx/data/verl/merged/<model>/step_<N>
export VLLM_SERVED_NAME=<run-name>
./scripts/deploy_vllm_eval.sh

# 3. HumanEval + MBPP (pass@1, pass@10) via lm-evaluation-harness + val-split reward
./scripts/submit_lmeval.sh <run-name>
./scripts/submit_val_eval.sh <run-name>

# 4. Compare vs baseline (+ optional MLflow logging)
python3 scripts/compare_eval_results.py \
    --run <run-name> --baseline <base-run-name> \
    --mlflow-run-id "${TRAINING_RUN_ID}"

# Tear down eval server when done
./scripts/deploy_vllm_eval.sh --delete
```

## Architecture: How Training Submits

1. `submit_training.py` loads Hydra config from `conf/` (composable YAML groups)
2. Resolves env vars via `${oc.env:RAY_ADDRESS}`, `${oc.env:MLFLOW_TRACKING_URI}`, etc.
3. Translates config into ~80 verl CLI override args (`actor_rollout_ref.model.path=...`)
4. Calls `ray job submit` with `runtime_env.yaml` (which pip-installs verl + deps on nodes)
5. Ray workers run `python3 -m verl.trainer.main_ppo` with the generated overrides

The two backends diverge at step 3: FSDP uses HuggingFace PEFT for LoRA, Megatron uses
Megatron-Bridge (`use_mbridge=True`). The `conf/backend/*.yaml` files control this split.
`submit_training.py` dispatches via `_build_fsdp_overrides()` vs `_build_megatron_overrides()`.

## Config System (Hydra)

**Training params live in `conf/`, NOT `env_vars`.** `env_vars` holds only infrastructure
settings (AWS, K8s namespace, secrets, NCCL). This is the most common source of confusion.

Config groups (`group=option` on CLI):
- `backend`: `fsdp` (default) | `megatron`
- `cluster`: `p6-b200-6node` (default) | `p6-b200-1node`
- `model`: `qwen3-8b` (default) | `qwen25-coder-7b` | `qwen25-72b` | `qwen3-coder-next` | `qwen3-30b-a3b` | `qwen3-235b`
- `lora`: `enabled` (default) | `disabled`
- `data`: `eurus` | `mixed` | `mixed-hard` | `mixed-valplus`
- `sandbox`: `enabled` (default) | `disabled`
- `tracking`: `mlflow` (default) | `console`
- `profiling`: `disabled` (default) | `torch` | `nsys` | `memory`

Scalar overrides use dotted paths: `training.learning_rate=1e-4`, `cluster.param_offload=true`.

To add a new model: create `conf/model/my-model.yaml` with `name`, `hf_path`, `fsx_path`,
`tensor_parallel_size`, and Megatron parallelism fields. Use it: `model=my-model`.

## Key Dependencies and Versions

| Component       | Version                    | Where                        |
|-----------------|----------------------------|------------------------------|
| verl            | v0.8.0                     | `scripts/runtime_env.yaml`   |
| Megatron-Core   | 0.18.0                     | `scripts/runtime_env.yaml`   |
| Megatron-Bridge | v0.5.0                     | Dockerfile (`--no-deps`)     |
| vLLM            | 0.20.2                     | Base image (verlai/verl)     |
| Ray             | 2.53.0                     | Ray cluster image            |
| Python          | 3.12                       | Inside container             |

## Critical Pitfalls

- **Never commit `env_vars`** — contains `HF_TOKEN` and AWS credentials. Only
  `env_vars.example` is tracked, and it derives `ACCOUNT`/`AWS_REGION`/`REGISTRY`
  dynamically so it holds no account identifiers. (`.gitignore` contains `env_vars`
  and nothing else. `AGENTS.md` itself IS tracked and is meant to ship — keep it
  accurate rather than excluding it.)
- **Megatron LoRA requires Megatron-Bridge** — without `use_mbridge=True` and the
  megatron-bridge package (Dockerfile), LoRA silently fails. Handled automatically
  when `backend=megatron` + `lora=enabled`.
- **FSDP vs Megatron LoRA APIs differ** — FSDP: `lora_rank=`, `target_modules=all-linear`
  (PEFT). Megatron: `model.lora.rank=`, `model.lora.type=`, fused module names
  (`linear_qkv,linear_proj,...`). Never mix them.
- **Parallelism constraint**: `TP x PP` must evenly divide total GPU count. For MoE
  models, `EP` is handled separately by Megatron.
- **vLLM LoRA PDL crash on Blackwell** — vLLM 0.20.2 disables PDL via the
  `VLLM_LORA_DISABLE_PDL=1` env var baked into the Dockerfile. Don't remove it.
- **pyext is broken on Python 3.12** — code reward scoring uses Sandbox Fusion service
  instead of local execution. Sandbox must be deployed for code tasks.
- **runtime_env.yaml installs packages per job** — verl, megatron-core, transformers,
  mlflow are NOT in the Docker image. They're pip-installed on Ray nodes at job start
  via `runtime_env.yaml`. Only Megatron-Bridge stays in the Dockerfile (needs `--no-deps`).
- **Cache dirs must point to /fsx** — `submit_training.py` redirects TRITON_CACHE_DIR,
  TORCH_COMPILE_CACHE_DIR, VLLM_CACHE_ROOT to `/fsx/data/cache/` to prevent /tmp exhaustion
  on Ray nodes.
- **`build-push.sh` is at the test-case root**, not `setup/`. The `setup/` directory no
  longer exists.
- **MoE models** may need per-model overrides for `lora_target_modules`, `lora_merge`,
  `lora_alpha`, `rollout_tensor_parallel_size`, `gpu_memory_utilization`, and
  `ppo_mini_batch_size`. These are resolved in `submit_training.py` via `OmegaConf.select()`
  with model YAML as the primary source.
- **Eval: vLLM container must use the training image** — `kubernetes/vllm-eval.yaml`
  references `${REGISTRY}${IMAGE}:${TAG}` for vLLM 0.20.2 + `VLLM_LORA_DISABLE_PDL=1`
  parity with training rollouts. Using `vllm/vllm-openai:*` images on the **server**
  can give subtly different numbers from training rollouts. (The lm-eval **driver**
  is a different concern — it only makes HTTP calls, so it's safe to use the
  published `vllm/vllm-openai:v0.20.2` there.)
- **Eval: Megatron+LoRA ckpts under verl v0.8.0 write a native HF PEFT adapter.**
  Since verl v0.8.0, every Megatron+LoRA save emits a native HuggingFace PEFT adapter
  at `actor/huggingface/adapter/` (`adapter_config.json` +
  `adapter_model.safetensors`). Merge is a plain `peft merge_and_unload()` via
  `scripts/merge_adapter.sh` — no replay merger. Still run
  `scripts/check_merge_parity.py` after merging before spending eval GPU-hours.
- **Eval: code benchmarks now go through lm-evaluation-harness, not bigcode.**
  Eval code benchmarks now run via lm-evaluation-harness (`local-completions`
  against the `vllm-eval` service), NOT bigcode-harness (which had no working
  OpenAI backend and no published image). Use `local-completions` +
  `--apply_chat_template`; `local-chat-completions` gives `pass@1=0` on
  instruct models.
- **Eval: `lmeval-tasks/bigcodebench_p4.yaml` is authored but NOT USABLE.** lm-eval's
  `code_eval` metric runs candidates under `reliability_guard()`, which disables the
  filesystem, network and plotting that BigCodeBench tasks legitimately require — its own
  canonical solutions score ~3/5 there. It needs BigCodeBench's official Docker executor
  via a generate-then-score split. Do not quote any number produced from this task as-is.
- **Eval: validate a new benchmark's OWN gold answers through the scorer before spending
  GPU.** A sound harness returns ~1.0 on ground truth; anything materially below means the
  instrument is unfit, not that the model is bad. This caught two unusable setups at zero
  GPU cost (see `docs/results.md`). Pre-register the threshold (0.95 was used) so the
  outcome cannot be rationalised afterwards.

## Shell Script Conventions

- Training/data scripts: `#!/usr/bin/env bash` + `set -xeuo pipefail`
- Setup/build scripts: `#!/bin/bash` + `set -euo pipefail`
- Every script has an `=` banner header describing its purpose
- Scripts source `env_vars` relative to their own `BASH_SOURCE` location
- All env vars use `${VAR:-default}` — never assume vars are set
- Naming: underscores in `scripts/` and `data/`, hyphens in `setup/` and root

## Dockerfile Conventions

- Base image: `verlai/verl:${BASE_TAG}` — pin to immutable tag, never `:latest`
- Layer order: system deps -> EFA -> blinker fix -> Megatron-Bridge (`--no-deps`)
- verl + transformers + pip extras go in `runtime_env.yaml`, NOT the Dockerfile
- Required env vars in image: `CUDA_DEVICE_MAX_CONNECTIONS=1`, `NCCL_NVLS_ENABLE=0`, `VLLM_USE_V1=1`
- `custom_reward_fn.py` is COPYed to `/workspace/` — the Hydra config references it at
  `/fsx/data/verl/custom_reward_fn.py` (synced via FSx DRA from S3)

## Testing and Linting

Three self-test scripts act as gates. Each is a plain script (no pytest) that prints
per-check `PASS`/`FAIL` lines and **exits non-zero if any check fails**:

```bash
python3 scripts/test_lmeval_utils.py      # lm-eval task builders + until-list behaviour
python3 scripts/test_merge_provenance.py  # adapter-merge provenance gating
python3 scripts/test_reward_routing.py    # custom_reward_fn routing per data_source
```

**Trust the exit code, not a `grep -c PASS`.** Verified: `test_merge_provenance.py` prints
13 checks but `grep -c PASS` returns **15**, because the code under test emits its own
`[merge_adapter] provenance gate PASS: ...` log lines, which a substring count cannot
distinguish from the harness's `[PASS]` verdicts. A gate whose pass-count is read by a
pattern that conflates subject output with verdict output is not a gate. Each script also
prints an authoritative `N/N checks passed` summary.

There is no CI. For shell scripts use `shellcheck` — note it is **not installed in every
dev environment**, so verify it is present before claiming the shell lint passed:

```bash
command -v shellcheck && shellcheck scripts/*.sh data/*.sh models/*.sh build-push.sh
```

For Python, ruff with upstream verl's settings (no config file is tracked, so pass the
flags explicitly):

```bash
ruff check --line-length 120 --select E,F,UP,B,I,G scripts/ data/ kubernetes/lmeval-tasks/
```

## Detailed Documentation

- `README.md` — short entry point: directory map, prerequisites, quickstart, doc index
- `docs/cluster.md` — cluster topology, Ray, storage, networking, monitoring, and the
  full workload-setup path (sandbox, fsx-utils, data/model staging, Ray auth, MLflow)
- `docs/configuration.md` — every Hydra config group, per-model recommended settings,
  and the parallelism / LoRA / memory / rollout / NCCL / B200 tuning guides
- `docs/eval-pipeline.md` — end-to-end eval workflow (merge adapter -> vLLM -> lm-eval +
  val-split -> MLflow curve)
- `docs/results.md` — Runs 1-4, measured results, MoE-adapter serving constraints, and
  the five insights worth keeping
- `docs/profiling.md` — torch profiler, Nsight, memory profiler, Perfetto
- `docs/troubleshooting.md` — OOM, slow training, slow loads, Megatron-Bridge, sandbox
