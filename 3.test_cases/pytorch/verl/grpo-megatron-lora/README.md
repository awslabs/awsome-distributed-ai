# GRPO + Megatron Training (LoRA & Full Fine-Tuning)

Train large language models with **Group Relative Policy Optimization (GRPO)** and verifiable
rewards (math + code) on Kubernetes. Wraps upstream [verl](https://github.com/volcengine/verl)
with Megatron-LM or FSDP, on Ray/KubeRay over EKS. Supports LoRA (7B to 235B+ MoE) and full
fine-tuning. Tuned for AWS P6-B200 (NVIDIA Blackwell, 8x183 GB per node).

There is **nothing to pip-install locally**. Training code is verl, installed on the Ray nodes
at job submission time via `scripts/runtime_env.yaml`. This test case is config, scripts and
manifests.

---

## Directory map

| Path | What lives there |
|------|------------------|
| `conf/` | **Hydra config — the source of truth for all training params** |
| `scripts/submit_training.py` | **Primary entry point.** Hydra config -> verl CLI -> `ray job submit` |
| `scripts/` | Job submitters, adapter merge + parity gate, eval drivers |
| `data/` | Dataset download, prep, difficulty filtering, mixing, val-holdout builder |
| `models/` | Model download helpers (HF -> FSx). `run_on_cluster.sh` dispatches onto the `fsx-utils` pod |
| `kubernetes/` | `vllm-eval`, `lmeval-job`, `sandbox-fusion`, `fsx-utils` (envsubst templates) |
| `kubernetes/lmeval-tasks/` | Custom lm-eval task YAMLs (pass@4/pass@10 variants) |
| `lustre/` | FSx Lustre PV / PVC / StorageClass (envsubst templates) |
| `docs/` | All reference documentation (see below) |
| `Dockerfile`, `build-push.sh` | Training image (EFA + Megatron-Bridge); build + ECR push |
| `env_vars.example` | Infra + secrets template. Copy to `env_vars` (gitignored) |

## Documentation

| Doc | Read it for |
|-----|-------------|
| [docs/cluster.md](docs/cluster.md) | Cluster topology, compute, Ray, storage, networking, monitoring |
| [docs/configuration.md](docs/configuration.md) | Every config group, per-model recommended settings, tuning guides |
| [docs/eval-pipeline.md](docs/eval-pipeline.md) | End-to-end eval walkthrough with worked examples |
| [docs/results.md](docs/results.md) | **Measured results and the five insights worth keeping** |
| [docs/profiling.md](docs/profiling.md) | Torch profiler, Nsight, memory profiler, Perfetto traces |
| [docs/troubleshooting.md](docs/troubleshooting.md) | OOM, slow training, slow loads, Megatron-Bridge, sandbox |
| [AGENTS.md](AGENTS.md) | Conventions, pitfalls and commands (written for coding agents) |

---

## Prerequisites

- An EKS cluster with GPU nodes (validated on 6 x `p6-b200.48xlarge`, 48 B200 GPUs)
- An FSx for Lustre filesystem (provisioned as the `fsx-lustre` PVC via `lustre/`)
- KubeRay operator installed
- A HuggingFace token with access to the model you intend to train
- `kubectl`, `awscli`, `docker`, `envsubst`, Python 3.12

## Quickstart

```bash
# 1. Configure infrastructure (AWS, K8s namespace, secrets, NCCL). NOT training params.
cp env_vars.example env_vars
vim env_vars                      # KUBE_NAMESPACE, HF_TOKEN, MLFLOW_TRACKING_ARN, ...
source env_vars

# 2. Build and push the training image to ECR
./build-push.sh

# 3. Provision shared storage, then the supporting services.
#    Fill the FSX_* vars in env_vars first -- see docs/cluster.md.
envsubst < lustre/storageclass.yaml          | kubectl apply -f -
envsubst < lustre/pv.yaml                    | kubectl apply -f -
envsubst < lustre/pvc.yaml                   | kubectl apply -f -   # PVC `fsx-lustre`
envsubst < kubernetes/sandbox-fusion.yaml | kubectl apply -f -   # code-reward execution
envsubst < kubernetes/fsx-utils.yaml      | kubectl apply -f -   # FSx shell/inspection pod
kubectl wait --for=condition=ready pod -l app=sandbox-fusion --timeout=300s

# 4. Stage data and model weights onto FSx
./data/submit_data_prep.sh --datasets eurus apps taco codecontests \
    --output-dir /fsx/data/verl/data
./models/submit_download.sh Qwen/Qwen3-8B /fsx/data/verl/models/Qwen3-8B

# 5. Reach the Ray dashboard (port-forward in a second shell)
kubectl port-forward -n "${KUBE_NAMESPACE:-default}" svc/<ray-head-svc> 8265:8265
export RAY_ADDRESS="http://localhost:8265"   # already the env_vars.example default

# 6. Submit training
python3 scripts/submit_training.py --cfg job     # dry run: print the resolved config
python3 scripts/submit_training.py               # Qwen3-8B, FSDP, 6-node, LoRA, MLflow
```

Monitor and manage jobs:

```bash
ray job list
ray job logs  <submission_id>
ray job stop  <submission_id>
```

---

## Configuration

**Training params live in `conf/` (Hydra). `env_vars` is infrastructure only.** That split is
the most common source of confusion in this test case.

Select options with `group=option`; override scalars with dotted paths.

| Group | Options (default first) |
|-------|-------------------------|
| `backend` | `fsdp`, `megatron` |
| `model` | `qwen3-8b`, `qwen25-coder-7b`, `qwen25-72b`, `qwen3-coder-next`, `qwen3-30b-a3b`, `qwen3-235b` |
| `cluster` | `p6-b200-6node`, `p6-b200-1node` |
| `lora` | `enabled`, `disabled` (disabled = full fine-tuning) |
| `data` | `eurus`, `mixed`, `mixed-hard`, `mixed-valplus` |
| `sandbox` | `enabled`, `disabled` |
| `tracking` | `mlflow`, `console` |
| `profiling` | `disabled` (+ torch / nsys / memory presets) |

```bash
# 235B MoE on Megatron with a mixed code/math dataset
python3 scripts/submit_training.py backend=megatron model=qwen3-235b data=mixed

# Single-node smoke test, no MLflow
python3 scripts/submit_training.py cluster=p6-b200-1node tracking=console

# Full fine-tuning instead of LoRA
python3 scripts/submit_training.py lora=disabled

# Override hyperparameters
python3 scripts/submit_training.py training.learning_rate=1e-5 training.train_batch_size=128
```

Full reference, including per-model recommended configurations and the parallelism / LoRA /
memory / rollout / NCCL tuning guides: **[docs/configuration.md](docs/configuration.md)**.

---

## Evaluating a checkpoint

Since verl v0.8.0, every Megatron+LoRA save writes a native HF PEFT adapter to
`actor/huggingface/adapter/`, so merging is a plain `peft merge_and_unload()`. Adapters can
alternatively be served directly as vLLM **runtime LoRA**, skipping the merge entirely.

Scale the Ray worker group down first so vLLM has a free GPU node.

```bash
./scripts/scale_ray_workers.sh 0                      # free a node for vLLM

./scripts/merge_adapter.sh 350                        # merge adapter -> standalone HF model
python3 scripts/check_merge_parity.py \
    --base-model   /fsx/data/verl/models/Qwen3-235B-A22B \
    --merged-model /fsx/data/verl/merged/Qwen3-235B-A22B/step_350

export VLLM_MODEL_PATH=/fsx/data/verl/merged/Qwen3-235B-A22B/step_350
export VLLM_SERVED_NAME=qwen3-235b-step350
./scripts/deploy_vllm_eval.sh                         # vLLM, TP=8, training image

./scripts/submit_lmeval.sh   qwen3-235b-step350       # HumanEval + MBPP (lm-eval)
./scripts/submit_val_eval.sh qwen3-235b-step350       # held-out val-split reward

./scripts/deploy_vllm_eval.sh --delete                # tear down when finished
```

> **Always run `check_merge_parity.py` before spending eval GPU-hours.** It asserts the merged
> model's logits actually diverge from base, which catches a silent base-only merge.

Full walkthrough, including the learning-curve aggregation and MLflow logging:
**[docs/eval-pipeline.md](docs/eval-pipeline.md)**.

---

## Results & insights

Qwen3-235B-A22B, GRPO + LoRA (r=128 / alpha=256), 48x B200, 45 h. Public,
contamination-cleared, deterministic (greedy) benchmarks:

| arm | GSM8K (n=1319, 5-shot, strict) | MATH-500 scoreable subset (n=472) |
|---|---|---|
| base | 0.8469 | 0.5212 |
| attention-only LoRA | **0.8719** (+0.0250, 2.83 sigma) | 0.5297 (+0.0085, null) |
| expert-FFN LoRA | **0.8704** (+0.0235, 2.60 sigma) | **0.5551** (+0.0339, 2.11 sigma) |

Five things worth carrying to the next project:

1. **In-distribution gains overstate capability ~3x.** The internal holdout moved +0.0784;
   public math moved +0.024 to +0.034. Quote the public figures.
2. **Validate a new benchmark's own gold answers through your scorer before spending GPU.**
   Two benchmarks failed this gate at zero cost (one executor was simply wrong for the task).
3. **A 105.7 GB MoE LoRA export is expert-parallel duplication, not one adapter x128.** It
   de-duplicates 21.67x to 4.88 GB, bit-exact. Collapsing it naively would corrupt 3 of 4
   learned adapters and still load fine.
4. **93% of the gain landed by step 50.** Steps 50-150 cost ~1,060 GPU-hours for +0.0057.
5. **Reserved capacity can disappear before its stated end time** (observed: 3h10m early).
   Never schedule work that must finish in a reservation's final hours.

Full run history, the serving constraints for large MoE adapters, and the measurement
methodology: **[docs/results.md](docs/results.md)**.

---

## Tested versions

| Component | Version |
|-----------|---------|
| EKS | 1.33 |
| verl | v0.8.0 |
| Megatron-Core | 0.18.0 |
| Megatron-Bridge | v0.5.0 |
| vLLM | 0.20.2 (base image `verlai/verl:vllm020.dev2`) |
| Ray | 2.53.0 / KubeRay v1.3.0 |
| Python | 3.12 |

## References

- [verl docs](https://verl.readthedocs.io/) —
  [LoRA](https://verl.readthedocs.io/en/latest/advance/ppo_lora.html) ·
  [GRPO](https://verl.readthedocs.io/en/latest/algo/grpo.html) ·
  [Megatron backend](https://verl.readthedocs.io/en/latest/workers/megatron_workers.html) ·
  [perf tuning](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html) ·
  [best practices](https://verl.readthedocs.io/en/latest/perf/best_practices.html)
- [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) ·
  [SandboxFusion](https://github.com/bytedance/SandboxFusion) ·
  [Hydra](https://hydra.cc/docs/intro/)
- [GRPO paper](https://arxiv.org/abs/2402.03300) · [RLVR paper](https://arxiv.org/abs/2506.14245)
