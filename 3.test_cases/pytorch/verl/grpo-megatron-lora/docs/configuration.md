# Configuration Reference

The Hydra config system, every config group, per-model recommended settings, and the
parallelism / LoRA / memory / rollout / NCCL tuning guides.

**Training params live in `conf/`, not `env_vars`.** `env_vars` holds infrastructure only
(AWS, K8s, secrets, NCCL). This is the most common source of confusion in this test case.

Start with [Config Groups](#config-groups); jump to [Recommended Configurations by Model
Size](#recommended-configurations-by-model-size) for a known-good starting point.

Training configuration uses [Hydra](https://hydra.cc/) with composable YAML config
groups. The entry point is `scripts/submit_training.py`, which loads `conf/config.yaml`
as the root config and merges in config groups from subdirectories under `conf/`.

Infrastructure settings (AWS, Docker, K8s, secrets, Ray address, NCCL) remain in
`env_vars` -- see `env_vars.example`. Hydra reads `RAY_ADDRESS` and
`MLFLOW_TRACKING_URI` from the environment via `${oc.env:...}` resolvers. `HF_TOKEN` is
read directly from the shell environment by the data/model staging scripts and pod
manifests, not through Hydra. All training
parameters (model, parallelism, LoRA, hyperparameters, offloading, sandbox, tracking)
are managed exclusively through the config files below.

## Config Groups

Each subdirectory under `conf/` is a config group. Switch variants from the CLI by
passing `group=option`:

| Group | Options | Default | What it controls |
|-------|---------|---------|------------------|
| `backend` | `fsdp`, `megatron` | `fsdp` | Backend engine, fused kernels, Megatron-Bridge, offloading mode |
| `cluster` | `p6-b200-6node`, `p6-b200-1node` | `p6-b200-6node` | Nodes, GPUs, offloading, memory utilization |
| `model` | `qwen3-8b`, `qwen25-coder-7b`, `qwen25-72b`, `qwen3-coder-next`, `qwen3-30b-a3b`, `qwen3-235b` | `qwen3-8b` | Model identity, HF path, FSx path, tensor/pipeline/expert/context parallelism |
| `lora` | `enabled`, `disabled` | `enabled` | LoRA rank, alpha, PEFT target modules (FSDP), mbridge params (Megatron) |
| `data` | `eurus`, `mixed` | `eurus` | Train/val file paths, prompt key, data name for checkpoint separation |
| `sandbox` | `enabled`, `disabled` | `enabled` | SandboxFusion URL, memory limit, concurrency |
| `tracking` | `mlflow`, `console` | `mlflow` | Experiment loggers (MLflow, console) |

Training hyperparameters (learning rate, batch size, epochs, etc.) live directly in
`conf/config.yaml` under the `training:` key and are overridden with dotted paths:

```bash
python3 scripts/submit_training.py training.learning_rate=1e-4 training.train_batch_size=128
```

## Config Directory Structure

```
conf/
├── config.yaml                  # Root config: training hyperparams, Ray, secrets, defaults
├── backend/
│   ├── fsdp.yaml                # FSDP + HuggingFace PEFT backend
│   └── megatron.yaml            # Megatron-Core + Megatron-Bridge backend (fused kernels, mbridge)
├── cluster/
│   ├── p6-b200-6node.yaml       # 6-node production (48 GPUs)
│   └── p6-b200-1node.yaml       # 1-node validation (8 GPUs)
├── model/
│   ├── qwen3-8b.yaml            # Qwen3-8B (TP=2, PP=1)
│   ├── qwen25-coder-7b.yaml     # Qwen2.5-Coder-7B-Instruct (TP=2, PP=1)
│   ├── qwen25-72b.yaml          # Qwen2.5-72B-Instruct (TP=4, PP=4)
│   ├── qwen3-coder-next.yaml    # Qwen3-Coder-Next (TP=4, PP=4)
│   ├── qwen3-30b-a3b.yaml       # Qwen3-30B-A3B MoE (TP=2, PP=2, EP=4, CP=2)
│   └── qwen3-235b.yaml          # Qwen3-235B-A22B MoE (TP=4, PP=3, EP=4)
├── lora/
│   ├── enabled.yaml             # LoRA rank=32, alpha=32, PEFT + mbridge settings
│   └── disabled.yaml            # Full fine-tuning
├── data/
│   ├── eurus.yaml               # Eurus math+code reasoning dataset
│   └── mixed.yaml               # Mixed code/math (TACO+APPS+CodeContests+Eurus, 70/30 split)
├── sandbox/
│   ├── enabled.yaml             # Remote SandboxFusion service
│   └── disabled.yaml            # Local code execution
└── tracking/
    ├── mlflow.yaml              # MLflow + console logging
    └── console.yaml             # Console-only logging
```

## Common Override Examples

```bash
# Defaults (Qwen3-8B, FSDP backend, 6-node, LoRA, MLflow)
python3 scripts/submit_training.py

# Quick validation run
python3 scripts/submit_training.py cluster=p6-b200-1node tracking=console

# Switch model
python3 scripts/submit_training.py model=qwen25-72b

# Megatron backend
python3 scripts/submit_training.py backend=megatron model=qwen25-72b

# Megatron with custom parallelism
python3 scripts/submit_training.py backend=megatron \
  model=qwen3-coder-next model.pipeline_parallel_size=2

# Disable fused kernels on Megatron
python3 scripts/submit_training.py backend=megatron backend.use_fused_kernels=false

# Full fine-tuning (no LoRA)
python3 scripts/submit_training.py lora=disabled

# Tune multiple hyperparameters
python3 scripts/submit_training.py \
  training.learning_rate=1e-4 \
  training.train_batch_size=128 \
  training.n_responses_per_prompt=8

# Override cluster memory settings
python3 scripts/submit_training.py \
  cluster.param_offload=true \
  cluster.optimizer_offload=true \
  cluster.gpu_memory_utilization=0.6

# Dry run (print resolved config, don't submit)
python3 scripts/submit_training.py --cfg job
```

## Hardware Assumptions

| Spec | Value |
|------|-------|
| Instance type | `p6-b200.48xlarge` |
| GPUs per node | 8x NVIDIA Blackwell B200 |
| GPU memory | 183 GB HBM per GPU |
| System memory | 4 TB per node |
| EFA adapters | 32 per node (6.4 Tbps aggregate) |
| NVLink domain | 1.4 TB aggregate GPU memory per node |

Two cluster profiles are referenced throughout:

- **2-node cluster**: 2 nodes x 8 GPUs = **16 GPUs total** (2.9 TB GPU memory)
- **6-node cluster**: 6 nodes x 8 GPUs = **48 GPUs total** (8.8 TB GPU memory)

## Model Selection

Switch models with the `model=` config group override. Each model config sets the HF path,
FSx path, and tensor parallelism:

```bash
# Qwen3-8B (default) — pipeline validation
python3 scripts/submit_training.py model=qwen3-8b

# Qwen2.5-Coder-7B — fast iteration
python3 scripts/submit_training.py model=qwen25-coder-7b

# Qwen2.5-72B — large dense model
python3 scripts/submit_training.py model=qwen25-72b

# Qwen3-Coder-Next — target production model
python3 scripts/submit_training.py model=qwen3-coder-next

# Qwen3-30B-A3B — MoE model (30.5B total, 3.3B active)
python3 scripts/submit_training.py model=qwen3-30b-a3b backend=megatron
```

Model configs are in `conf/model/`. Each config sets tensor parallelism and Megatron
parallelism fields (pipeline, expert, context). The Megatron parallelism fields are
ignored when using the FSDP backend. To add a new model, create a new YAML file:

```yaml
# conf/model/my-model.yaml
name: My-Model
hf_path: org/My-Model
fsx_path: ${cluster.fsx_home}/models/${model.name}
tensor_parallel_size: 2
# Megatron parallelism (ignored by FSDP backend)
pipeline_parallel_size: 1
expert_parallel_size: 1
context_parallel_size: 1
```

Then use it: `python3 scripts/submit_training.py model=my-model`

---

## Recommended Configurations by Model Size

Each table below shows recommended Hydra config values for a specific model size and
training mode. Values are tuned for B200 GPUs (183 GB each). Where the 2-node and 6-node
settings differ, both are shown. Config keys map to Hydra override paths (e.g.,
`training.learning_rate=3e-5` on the CLI). Recommendations are based on the
[verl performance tuning guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html),
[verl LoRA docs](https://verl.readthedocs.io/en/latest/advance/ppo_lora.html),
[verl hardware resource guide](https://verl.readthedocs.io/en/latest/perf/device_tuning.html),
and [verl best practices (DAPO + Qwen3-235B)](https://verl.readthedocs.io/en/latest/perf/best_practices.html).

### Qwen2.5-Coder-7B -- LoRA

A small model that fits comfortably on 2 nodes. LoRA makes training very fast with
minimal memory pressure, so no offloading is needed.

| Config Key | 2-node (16 GPUs) | 6-node (48 GPUs) | Notes |
|----------|:-:|:-:|-------|
| `lora` (group) | `enabled` | `enabled` | |
| `lora.rank` | `32` | `32` | Rank 32 matches full fine-tuning convergence for models <=7B ([verl LoRA docs](https://verl.readthedocs.io/en/latest/advance/ppo_lora.html)) |
| `lora.alpha` | `32` | `32` | Equal to rank is a safe default |
| `training.learning_rate` | `3e-5` | `3e-5` | ~10x higher than full fine-tuning ([verl LoRA docs](https://verl.readthedocs.io/en/latest/advance/ppo_lora.html)) |
| `model.tensor_parallel_size` | `1` | `1` | 7B fits in a single GPU; TP=1 maximizes data parallelism |
| `model.pipeline_parallel_size` | `1` | `1` | No pipeline splitting needed |
| `model.expert_parallel_size` | `1` | `1` | Dense model, no MoE |
| `training.train_batch_size` | `128` | `256` | Scale with GPU count for throughput |
| `training.n_responses_per_prompt` | `8` | `8` | GRPO group size |
| `cluster.gpu_memory_utilization` | `0.85` | `0.85` | vLLM rollout memory fraction; safe on B200 ([verl perf tuning](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html)) |
| `cluster.param_offload` | `false` | `false` | Not needed -- 7B + LoRA fits in GPU memory easily |
| `cluster.optimizer_offload` | `false` | `false` | |
| `backend` (group) | `fsdp` or `megatron` | `fsdp` or `megatron` | FSDP is simpler for 7B; Megatron works but is overkill |

**Example (6-node):**
```bash
python3 scripts/submit_training.py model=qwen25-coder-7b training.train_batch_size=256
```

**Why these values**: With LoRA rank 32, only ~0.3% of the 7B model's parameters are
trainable. At BF16, the full model is ~14 GB, and the LoRA adapter adds negligible memory.
Each B200 GPU has 183 GB, so even without offloading, there is plenty of room for the model,
optimizer states, KV cache, and activations. TP=1 means all 16 (or 48) GPUs run as independent
data-parallel workers, which maximizes generation throughput since more vLLM replicas run
in parallel ([verl perf tuning -- rollout generation](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html#rollout-generation-tuning)).

### Qwen2.5-Coder-7B -- Full Fine-Tuning

Full fine-tuning trains all 7B parameters. Memory requirements jump significantly because
of optimizer states (Adam stores 2 extra copies of each parameter).

| Config Key | 2-node (16 GPUs) | 6-node (48 GPUs) | Notes |
|----------|:-:|:-:|-------|
| `lora` (group) | `disabled` | `disabled` | |
| `training.learning_rate` | `1e-6` | `1e-6` | Standard full fine-tuning LR |
| `model.tensor_parallel_size` | `2` | `2` | Split model across 2 GPUs for optimizer memory headroom |
| `model.pipeline_parallel_size` | `1` | `1` | |
| `training.train_batch_size` | `128` | `256` | |
| `training.n_responses_per_prompt` | `8` | `8` | |
| `cluster.gpu_memory_utilization` | `0.7` | `0.7` | Lower than LoRA -- more GPU memory needed for training state |
| `cluster.param_offload` | `false` | `false` | B200 has enough memory for 7B full fine-tuning |
| `cluster.optimizer_offload` | `false` | `false` | Enable if you see OOM during actor update |
| `backend` (group) | `fsdp` or `megatron` | `fsdp` or `megatron` | |

**Example (6-node):**
```bash
python3 scripts/submit_training.py \
  model=qwen25-coder-7b lora=disabled \
  training.learning_rate=1e-6 training.train_batch_size=256 \
  cluster.gpu_memory_utilization=0.7
```

**Why these values**: Full 7B fine-tuning in BF16 requires ~14 GB for model weights + ~56 GB
for Adam optimizer states (at FP32) + activations per data-parallel replica. With TP=2 this
is split across 2 GPUs, consuming ~35 GB each before activations -- well within B200 capacity.
The `gpu_memory_utilization` is lowered to 0.7 because the vLLM rollout phase must coexist
with the training state in verl's HybridEngine ([verl perf tuning](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html#rollout-generation-tuning)).

### Qwen2.5-72B -- LoRA

72B is the sweet spot for 2-node B200 clusters with LoRA. The model requires tensor
parallelism to fit, and the reference model should offload to CPU to free memory
for rollout.

| Config Key | 2-node (16 GPUs) | 6-node (48 GPUs) | Notes |
|----------|:-:|:-:|-------|
| `lora` (group) | `enabled` | `enabled` | |
| `lora.rank` | `64` | `64` | Rank 64 recommended for 72B; increase to 128 if convergence is slow ([verl LoRA docs](https://verl.readthedocs.io/en/latest/advance/ppo_lora.html)) |
| `lora.alpha` | `32` | `32` | Alpha = rank/2 is common; try alpha = rank if training is unstable |
| `training.learning_rate` | `3e-5` | `3e-5` | |
| `model.tensor_parallel_size` | `4` | `4` | 72B in BF16 = ~144 GB; split across 4 GPUs = ~36 GB each |
| `model.pipeline_parallel_size` | `1` | `1` | PP=1 preferred for LoRA -- avoids pipeline bubble overhead |
| `model.expert_parallel_size` | `1` | `1` | Dense model |
| `training.train_batch_size` | `64` | `256` | Smaller batch on 2-node to fit in memory |
| `training.n_responses_per_prompt` | `5` | `8` | Fewer responses per prompt on 2-node to reduce rollout memory |
| `cluster.gpu_memory_utilization` | `0.5` | `0.6` | Lower on 2-node since training state competes for memory ([verl perf tuning](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html)) |
| `cluster.param_offload` | `false` | `false` | LoRA adapters are small; base model weights are frozen |
| `cluster.optimizer_offload` | `false` | `false` | Only LoRA params have optimizer states |
| Ref model offload | `true` | `true` | Always offload the reference model for 72B ([verl config docs](https://verl.readthedocs.io/en/latest/examples/config.html): "For models >7B, offload ref by default") |
| `backend` (group) | **`megatron`** (recommended) | **`megatron`** | Megatron's 3D HybridEngine reduces peak memory during actor-rollout transitions |

**Example (6-node):**
```bash
python3 scripts/submit_training.py \
  backend=megatron model=qwen25-72b \
  lora.rank=64 lora.alpha=32 \
  training.train_batch_size=256 \
  cluster.gpu_memory_utilization=0.6
```

For **FSDP backend** on 72B LoRA, use these overrides (from [verl LoRA docs -- Qwen2.5-72B reference config](https://verl.readthedocs.io/en/latest/advance/ppo_lora.html#fsdp-backend-usage-guide)):

```bash
python3 scripts/submit_training.py \
  model=qwen25-72b \
  lora.rank=64 lora.alpha=32 \
  cluster.param_offload=true cluster.optimizer_offload=true \
  cluster.gpu_memory_utilization=0.4
```

The FSDP backend automatically sets `use_remove_padding=True`, `layered_summon=True`,
and `ref.fsdp_config.param_offload=True`. Additional FSDP-specific verl overrides for
reference:

```bash
# FSDP-specific settings for 72B LoRA
actor_rollout_ref.model.use_shm=True                    # Preload model to /dev/shm for faster loading
actor_rollout_ref.model.lora_rank=64
actor_rollout_ref.model.lora_alpha=32
actor_rollout_ref.model.target_modules=all-linear
actor_rollout_ref.actor.fsdp_config.param_offload=True   # Offload actor params on FSDP (more aggressive than Megatron)
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
actor_rollout_ref.ref.fsdp_config.param_offload=True
actor_rollout_ref.rollout.load_format=safetensors
actor_rollout_ref.rollout.layered_summon=True             # Gather FSDP shards per-layer to reduce peak memory
actor_rollout_ref.rollout.gpu_memory_utilization=0.4      # Lower for FSDP -- more memory consumed by sharded states
```

**Why these values**: At BF16, 72B model weights consume ~144 GB. With TP=4 on Megatron,
each GPU holds ~36 GB of weights, leaving ~145 GB for KV cache, activations, and LoRA
optimizer states. The reference model (which only does forward passes, no gradients) is
offloaded to CPU since it would otherwise double the weight memory. On a 2-node cluster
(16 GPUs), DP = 16 / (TP x PP) = 16 / 4 = 4 data-parallel replicas. The smaller batch
size (64 vs 256) accounts for fewer DP replicas and tighter memory.

### Qwen2.5-72B -- Full Fine-Tuning

Full fine-tuning of 72B requires pipeline parallelism and aggressive memory management.
The optimizer alone needs ~576 GB (Adam FP32 states for 72B params). This realistically
requires 6 nodes; 2 nodes will need offloading and will be slow.

| Config Key | 2-node (16 GPUs) | 6-node (48 GPUs) | Notes |
|----------|:-:|:-:|-------|
| `lora` (group) | `disabled` | `disabled` | |
| `training.learning_rate` | `1e-6` | `1e-6` | |
| `model.tensor_parallel_size` | `4` | `4` | |
| `model.pipeline_parallel_size` | `2` | `4` | PP splits layers across GPUs in sequence |
| `training.train_batch_size` | `32` | `128` | Small batch on 2-node due to memory constraints |
| `training.n_responses_per_prompt` | `4` | `8` | |
| `cluster.gpu_memory_utilization` | `0.4` | `0.6` | Low on 2-node; optimizer states consume most of GPU memory |
| `cluster.param_offload` | `true` | `false` | Needed on 2-node; 6-node has enough aggregate memory |
| `cluster.optimizer_offload` | `true` | `true` | Recommended even on 6-node for 72B full FT |
| `backend.grad_offload` | `true` | `false` | Only needed on 2-node (Megatron only) |
| Ref model offload | `true` | `true` | |
| `backend` (group) | **`megatron`** | **`megatron`** | FSDP can work but Megatron's PP is critical for 72B full FT |

**Example (6-node):**
```bash
python3 scripts/submit_training.py \
  backend=megatron model=qwen25-72b lora=disabled \
  training.learning_rate=1e-6 training.train_batch_size=128 \
  cluster.optimizer_offload=true cluster.gpu_memory_utilization=0.6
```

**Why these values**: Full 72B fine-tuning in BF16 with Adam requires: ~144 GB (weights) +
~288 GB (optimizer momentum) + ~288 GB (optimizer variance) = ~720 GB minimum, before
activations. On a 2-node cluster with TP=4, PP=2, each GPU holds a fraction of this, but
offloading optimizer states to the 4 TB system memory per node is essential. On 6 nodes
with PP=4, the load is distributed across 48 GPUs (DP = 48 / (4 x 4) = 3 replicas),
and only optimizer offload is needed.

### Qwen3-235B-A22B MoE -- LoRA

> **Warning**: 235B MoE models require substantial GPU memory and inter-node bandwidth.
> A **2-node (16 GPU) cluster is not recommended** for 235B -- the model weights alone
> are ~470 GB in BF16, and with expert parallelism the communication overhead across only
> 2 nodes severely limits throughput. The config below is provided for reference, but
> expect very slow training and possible OOM. **Use 6+ nodes for practical 235B training.**

| Config Key | 2-node (16 GPUs) | 6-node (48 GPUs) | Notes |
|----------|:-:|:-:|-------|
| `lora` (group) | `enabled` | `enabled` | Full FT of 235B is impractical; LoRA is essential |
| `lora.rank` | `128` | `32` | Rank 32 validated on 6-node; increase if convergence is slow |
| `lora.alpha` | `64` | `64` | |
| `lora.megatron.target_modules` | `linear_qkv,linear_proj` | same | Attention-only default. Adding `linear_fc1,linear_fc2` (expert MLP) is **verified working** as of 2026-08-07 — see note below |
| `training.learning_rate` | `1e-5` | `1e-5` | 3e-5 caused mode collapse in initial run; 1e-5 validated stable |
| `training.kl_loss_coef` | `0.02` | `0.02` | KL=0.001 caused entropy crash; 0.02 keeps entropy stable at ~0.78. **Isolated and confirmed by Run 3 (2026-08-06) — do NOT lower.** |
| `training.max_response_length` | `4096` | `6144` | 6144 recommended for coding tasks; 4096 had 45%+ truncation |
| `model.tensor_parallel_size` | `4` | `4` | |
| `model.pipeline_parallel_size` | `2` | `3` | PP=3 validated on 6-node (96 layers / 3 = 32 per stage) |
| `model.expert_parallel_size` | `2` | `4` | Distribute experts across GPUs ([verl best practices](https://verl.readthedocs.io/en/latest/perf/best_practices.html)) |
| `training.train_batch_size` | `32` | `96` | |
| `training.n_responses_per_prompt` | `4` | `4` | |
| `cluster.gpu_memory_utilization` | `0.4` | `0.7` | |
| `cluster.param_offload` | `true` | `false` | Required on 2-node |
| `cluster.optimizer_offload` | `true` | `false` | |
| Ref model offload | `true` | `true` | |
| `backend` (group) | **`megatron`** (required) | **`megatron`** | Only Megatron supports expert parallelism |

> **LoRA target modules**: ~~Use `linear_qkv,linear_proj` (attention layers only), not
> `linear_fc1,linear_fc2` (expert MLP layers). Including expert layers causes a naming
> mismatch when syncing LoRA weights to the vLLM rollout engine (Megatron fused names
> vs HuggingFace names), resulting in runtime errors. This is tracked in
> [verl#5479](https://github.com/volcengine/verl/issues/5479).~~
>
> **CORRECTED 2026-08-07 — expert-layer LoRA WORKS on this stack. Both documented
> blockers were wrong.** An expert-FFN run
> (`target_modules=linear_qkv,linear_proj,linear_fc1,linear_fc2`, r=128/alpha=256,
> `lora_merge=true`, verl v0.8.0 + Megatron-Bridge v0.5.0, EP=4, `moe_grouped_gemm=True`):
>
> | claimed blocker | source | outcome |
> |---|---|---|
> | vLLM weight-sync naming mismatch -> runtime errors (verl#5479) | this note | **Does not reproduce.** 0 Tracebacks, 0 KeyError/naming/size-mismatch hits across init and 7 steps. Rollouts ran normally. |
> | ~1.68B trainable params/GPU -> ~20 GB Adam state -> OOM | local follow-on notes | **Wrong by 33x.** Measured `Adapter parameters: 76,185,600` (0.39%). No OOM. |
>
> **Why verl#5479 does not bite here: `lora_merge: true`.** verl merges the adapter into the
> base weights *before* syncing to vLLM, so what crosses the boundary is merged model
> weights, not adapter tensors under Megatron-fused names. The naming mismatch has nothing
> to bite on. (If you ever set `lora_merge: false` for an MoE model, expect verl#5479 to return.)
>
> **Why the OOM estimate was 33x off:** Megatron-Bridge `peft/lora.py` sets
> `share_expert_adapters: bool = True` by default — one adapter shared across all local
> experts on the EP rank, not one per expert. verl never references that field, so the
> per-expert branch is unreachable. 128 experts / EP=4 = 32 experts per GPU, hence the 32x.
>
> Measured cost and evidence that it is really training:
>
> | | attention-only (control) | + expert layers |
> |---|---|---|
> | trainable params | 25,395,200 (0.13%) | **76,185,600 (0.39%)** |
> | `actor/grad_norm`, steps 1-8 | 0.036-0.068 | **0.064-0.148 (1.70-2.74x, mean 2.12x)** |
> | `actor/ppo_kl`, steps 1-6 | -2.5e-4 .. 1.2e-4 | **-2.2e-4 .. 5e-5 (same magnitude)** |
> | step time (converged) | 635 s | **657 s (+3.5%)** |
>
> The `grad_norm` rise is the proof gradient actually reaches the expert adapters — had
> `linear_fc1`/`linear_fc2` matched nothing, it would be indistinguishable from the control.
> The small `ppo_kl` is the proof the rollout policy still equals the training policy, i.e.
> the merge handled grouped-GEMM expert weights correctly; a botched sync would show up as
> a large or erratic `ppo_kl` even without raising an error.
>
> **Still unverified:** the eval-time path (checkpoint -> `actor/huggingface/adapter/` ->
> `scripts/merge_adapter.sh` -> vLLM serve) for an adapter containing expert modules. Run
> `scripts/check_merge_parity.py` before spending eval GPU-hours on one.
>
> Rationale for trying it at all: attention-only LoRA adapts 6.7B of 235B = **2.85% of
> weights**; the 227B of MoE experts get zero adaptation. See Run 4 below.

For MoE-specific Megatron optimizations (automatically applied by `submit_training.py`
when the model has `expert_parallel_size > 1`):

```bash
+actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
+actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
+actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
```

**Why these values**: Qwen3-235B-A22B has 235B total parameters but only ~22B are active
per token (MoE routing). In BF16 the full model is ~470 GB. With TP=4 and EP=4 on 6 nodes,
expert weights are distributed across 4 groups while attention layers are split across 4
GPUs per group. DP = 48 / (TP x PP x EP) = 48 / (4 x 4 x 1) = 3 when EP is handled
separately by Megatron. The `moe_grouped_gemm` and `moe_permute_fusion` flags are
performance optimizations from the verl best practices guide that batch expert computation.

### Qwen3-30B-A3B MoE -- LoRA

> **Note**: Qwen3-30B-A3B is a 30.5B MoE model with only 3.3B active parameters per token.
> At ~61 GB in BF16 (16 shards), it fits comfortably on a single node but benefits from
> multi-node parallelism for throughput. Config derived from the upstream verl reference
> script `run_qwen3moe-30b_megatron_lora.sh`.

| Config Key | 6-node (48 GPUs) | Notes |
|----------|:-:|-------|
| `lora` (group) | `enabled` | |
| `lora.rank` | `32` | Global default; matches upstream |
| `model.lora_alpha` | `64` | Per-model override (2x ratio); set in model YAML |
| `lora.megatron.target_modules` | `linear_qkv,linear_proj,linear_fc1,linear_fc2` | Global default; MoE routers excluded |
| `training.learning_rate` | `3e-6` | Matches upstream verl script |
| `model.tensor_parallel_size` | `2` | Actor TP |
| `model.rollout_tensor_parallel_size` | `8` | Full-node TP for vLLM rollout inference |
| `model.pipeline_parallel_size` | `2` | |
| `model.expert_parallel_size` | `4` | 128/4 = 32 experts per EP group |
| `model.context_parallel_size` | `2` | Sequence-level parallelism |
| `model.gpu_memory_utilization` | `0.25` | Lower for MoE expert routing headroom |
| `model.recompute` | `uniform/full/1` | Activation recomputation for memory savings |
| `backend` (group) | **`megatron`** (required) | Only Megatron supports expert parallelism |

```bash
# Launch Qwen3-30B-A3B training
python3 scripts/submit_training.py model=qwen3-30b-a3b backend=megatron
```

**Why these values**: With TP=2 and PP=2, the 48 layers split into 24 per pipeline stage,
and attention heads split 16/1 (Q/KV) per TP rank. EP=4 distributes 128 experts into groups
of 32 per rank. The rollout uses TP=8 (full node width) for maximum vLLM inference throughput
since MoE models have sparse activation patterns. Activation recomputation (`uniform/full/1`)
trades compute for memory, recomputing every layer's activations during backward.

---

## Parallelism Tuning Guide

These control how the model is split across GPUs. The fundamental constraint is:
**TP x PP must evenly divide the total GPU count**. Data parallelism (DP) is computed
automatically: `DP = total_GPUs / (TP x PP)`. Expert parallelism (EP) applies only to MoE
models and is handled separately by Megatron.

All parallelism settings are in the model config group (`conf/model/*.yaml`):

```bash
# Override from CLI
python3 scripts/submit_training.py model.tensor_parallel_size=4 model.pipeline_parallel_size=2
```

**When to increase each** ([verl best practices](https://verl.readthedocs.io/en/latest/perf/best_practices.html)):

- **TP** (tensor parallelism): Increase first when a single GPU can't hold the model weights. Each parameter consumes `2 / TP` bytes in BF16. Keep TP <= 8 to avoid excessive all-reduce communication.
- **PP** (pipeline parallelism): Add when TP alone isn't enough. Introduces pipeline bubbles (idle time), so only use when necessary. Effective for very large models (70B+).
- **EP** (expert parallelism): Only for MoE models. Align with TP for optimal communication patterns.
- **CP** (context parallelism): Only for sequences > 32k tokens. Splits the sequence dimension.

**Rollout TP**: The vLLM rollout engine's `tensor_model_parallel_size` is set automatically
from `model.tensor_parallel_size` by `submit_training.py`.

## LoRA Tuning Guide

Switch between LoRA and full fine-tuning with the `lora` config group. Override individual
settings with dotted paths:

```bash
python3 scripts/submit_training.py lora=enabled lora.rank=64 lora.alpha=32
python3 scripts/submit_training.py lora=disabled  # full fine-tuning
```

**LoRA rank recommendations** (from [verl LoRA docs](https://verl.readthedocs.io/en/latest/advance/ppo_lora.html#best-practices-and-notes)):

| Model Size | `lora.rank` | Convergence vs. Full FT | Notes |
|------------|-------------|-------------------------|-------|
| 0.5B-7B | 32 | Near-identical | Rank 32 tested on 0.5B with convergence parity |
| 14B-32B | 64 | Near-identical | |
| 72B | 64-128 | 128 matches full FT for 32B+ | Increase to 128 if training loss plateaus |
| 235B+ MoE | 128-256 | Higher rank needed for MoE | MoE routers excluded by default |

**Learning rate**: LoRA requires ~10x higher LR than full fine-tuning ([verl LoRA docs](https://verl.readthedocs.io/en/latest/advance/ppo_lora.html#best-practices-and-notes)):

```bash
python3 scripts/submit_training.py training.learning_rate=3e-5   # LoRA (default)
python3 scripts/submit_training.py training.learning_rate=1e-6   # Full fine-tuning
```

**Megatron vs. FSDP LoRA** -- the two backends use different LoRA implementations:

| Aspect | Megatron Backend | FSDP Backend |
|--------|-----------------|--------------|
| LoRA library | Megatron-Bridge native | HuggingFace PEFT |
| Target modules syntax | `linear_qkv,linear_proj,...` (fused names) | `all-linear` |
| Requires mbridge | Yes (`use_mbridge=True`) | No |
| Weight sync to vLLM | Merge or separate adapters | `layered_summon=True` for per-layer gather |
| Config location | `conf/lora/enabled.yaml` → `megatron:` subsection | `conf/lora/enabled.yaml` → top-level keys |

When LoRA is disabled (`lora=disabled`), `submit_training.py` omits all LoRA overrides.
Full FT requires significantly more memory -- see the per-model tables above for
offloading recommendations.

## Memory Optimization Guide

These settings trade compute speed for lower GPU memory usage. On B200 GPUs (183 GB each),
you often don't need them for LoRA training, but they become essential for full fine-tuning
of large models. Override via the `cluster` and `backend` config groups:

```bash
python3 scripts/submit_training.py \
  cluster.param_offload=true \
  cluster.optimizer_offload=true \
  cluster.gpu_memory_utilization=0.6

# Megatron-only: gradient offloading
python3 scripts/submit_training.py backend=megatron backend.grad_offload=true
```

**When to enable offloading** ([verl Megatron workers docs](https://verl.readthedocs.io/en/latest/workers/megatron_workers.html#offload)):

| Scenario | `cluster.param_offload` | `cluster.optimizer_offload` | `backend.grad_offload` |
|----------|:-:|:-:|:-:|
| 7B LoRA | `false` | `false` | `false` |
| 7B full FT | `false` | `false` | `false` |
| 72B LoRA | `false` | `false` | `false` |
| 72B full FT (2-node) | `true` | `true` | `true` |
| 72B full FT (6-node) | `false` | `true` | `false` |
| 235B MoE LoRA (2-node) | `true` | `true` | `false` |
| 235B MoE LoRA (6-node) | `false` | `false` | `false` |

**Reference model offloading**: The reference model (frozen copy used for KL divergence)
should always be offloaded for models > 7B. `submit_training.py` sets this automatically.
Per the [verl config docs](https://verl.readthedocs.io/en/latest/examples/config.html):
*"For models larger than 7B, it's recommended to turn on offload for ref by default."*

**`cluster.gpu_memory_utilization`** controls how much GPU memory vLLM claims during the
rollout generation phase. In verl's HybridEngine, the same GPUs run both training (actor
update) and inference (vLLM rollout), so this value must leave enough room for training state:

| Mode | Recommended range | Why |
|------|:-:|-----|
| LoRA (small model) | `0.80-0.90` | Training state is minimal (only LoRA adapters) |
| LoRA (large model) | `0.50-0.70` | Base model weights + optimizer for LoRA params |
| Full FT (no offload) | `0.40-0.60` | Optimizer states consume most GPU memory |
| Full FT (with offload) | `0.60-0.80` | Offloading frees GPU memory for vLLM |

Higher `gpu_memory_utilization` means more KV cache for vLLM, which improves rollout
throughput. Push it as high as possible without triggering OOM ([verl best practices](https://verl.readthedocs.io/en/latest/perf/best_practices.html)).

## Rollout & Generation Settings

These configure the vLLM inference engine used during GRPO rollout generation.
Override via `training.*` and `cluster.*` Hydra paths:

```bash
python3 scripts/submit_training.py \
  training.n_responses_per_prompt=8 \
  training.max_prompt_length=2048 \
  training.max_response_length=4096 \
  cluster.gpu_memory_utilization=0.85
```

Key rollout tuning knobs (set automatically by `submit_training.py`):

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `rollout.enforce_eager=True` | `True` | Disables CUDA graphs; frees memory but slightly slower. Set `False` with `cudagraph_capture_sizes` if memory allows ([verl best practices](https://verl.readthedocs.io/en/latest/perf/best_practices.html)) |
| `rollout.free_cache_engine=True` | `True` | Offloads KV cache after generation -- essential for HybridEngine |
| `rollout.enable_chunked_prefill=True` | `True` | Improves GPU utilization during prefill (Megatron backend) |
| `rollout.max_num_batched_tokens` | `prompt+response` | Maximum tokens per batch; set >= `max_prompt_length + max_response_length` ([verl best practices](https://verl.readthedocs.io/en/latest/perf/best_practices.html)) |
| `rollout.max_num_seqs` | `training.max_num_seqs` | Concurrent sequences; increase if GPU cache utilization is low ([verl perf tuning](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html)) |
| `rollout.temperature=1.0` | `1.0` | Sampling temperature; keep 1.0 for training diversity |
| `rollout.top_p=1.0` | `1.0` | Nucleus sampling; 1.0 disables filtering |

**Dynamic batching** (enabled by default for all runs):

```bash
actor_rollout_ref.actor.use_dynamic_bsz=True
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=<max_prompt_length + max_response_length>
```

Dynamic batch sizing adapts the actual batch size per forward pass based on sequence length,
ensuring each GPU processes a similar number of tokens. This improves throughput and reduces
memory waste from padding ([verl perf tuning -- dynamic batch](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html#tuning-for-dynamic-batch-size)).
Set `ppo_max_token_len_per_gpu` to at least 2x `(max_prompt_length + max_response_length)`.

## Training Hyperparameters

Key hyperparameters are in `conf/config.yaml` under `training:`. Override with dotted paths:

```bash
python3 scripts/submit_training.py \
  training.kl_loss_coef=0.01 \
  training.ppo_mini_batch_size=64 \
  training.train_batch_size=256
```

The verl-specific algorithm settings below are hardcoded in `submit_training.py`'s
`_build_shared_overrides()` function. Understanding them helps with tuning.

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `algorithm.adv_estimator=grpo` | `grpo` | Use GRPO advantage estimator (no critic model) ([verl GRPO docs](https://verl.readthedocs.io/en/latest/algo/grpo.html)) |
| `actor.use_kl_loss=True` | `True` | KL divergence regularization against reference model |
| `actor.kl_loss_coef=0.02` | `0.02` | KL penalty weight; increase to prevent mode collapse. Initial runs with 0.001 led to entropy crash and reward hacking. **Run 3 (2026-08-06) re-tested 0.001 as a single variable with truncation/lr/data-mix already fixed and entropy still collapsed 0.73 -> 0.13 in 35 steps — this value is load-bearing regularization, not a legacy patch. Do NOT lower it.** |
| `actor.kl_loss_type=low_var_kl` | `low_var_kl` | Low-variance KL estimator (k3); recommended for GRPO ([verl GRPO docs](https://verl.readthedocs.io/en/latest/algo/grpo.html)) |
| `actor.clip_ratio_low=0.2` | `0.2` | Lower PPO clip bound |
| `actor.clip_ratio_high=0.28` | `0.28` | Upper clip bound (asymmetric/DAPO-style); allows upward updates more aggressively |
| `actor.clip_ratio_c=10.0` | `10.0` | Token-level importance weight clipping |
| `actor.loss_agg_mode=token-mean` | `token-mean` | Token-level loss averaging; recommended over `seq-mean-token-mean` for long-CoT stability ([verl GRPO docs](https://verl.readthedocs.io/en/latest/algo/grpo.html)) |
| `actor.ppo_mini_batch_size=32` | `32` | Global mini-batch size for actor updates |
| `actor.ppo_micro_batch_size_per_gpu=2` | `2` | Samples per GPU per forward pass; increase until OOM for throughput ([verl perf tuning](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html#batch-size-tuning)) |

**Batch size hierarchy** ([verl config docs](https://verl.readthedocs.io/en/latest/examples/config.html)):

```
train_batch_size (global, per iteration)
  └── ppo_mini_batch_size (global, per optimization step)
        └── ppo_micro_batch_size_per_gpu (local, per GPU forward pass)
```

`training.train_batch_size` determines how many prompts are sampled per GRPO iteration.
Each prompt generates `training.n_responses_per_prompt` responses, so total trajectories =
`train_batch_size x n_responses_per_prompt`. These are split into mini-batches of
`training.ppo_mini_batch_size` for the policy update, and each mini-batch is further
split into micro-batches for gradient accumulation.

## NCCL & Communication Settings

These are set in the Dockerfile or `env_vars` and are required for correct operation:

```bash
# Required (set in Dockerfile)
export CUDA_DEVICE_MAX_CONNECTIONS=1   # Required by Megatron-LM for correct TP communication
export NCCL_NVLS_ENABLE=0             # Disable NVLink SHARP -- avoids hangs on some topologies
export VLLM_USE_V1=1                  # Use vLLM v1 engine (required by verl)

# EFA networking (set in env_vars, propagated to Ray workers via shell environment)
export NCCL_DEBUG=INFO                # Log NCCL operations for debugging
export NCCL_IB_DISABLE=0             # Enable InfiniBand/EFA
export NCCL_NET_GDR_LEVEL=2          # GPU Direct RDMA level for EFA
```

---

## Hardware-Specific Tips for B200 GPUs

The NVIDIA Blackwell B200 (183 GB HBM) offers significantly more memory than H100
(80 GB) or H200 (141 GB). This changes several tuning decisions:

1. **Less offloading needed**: Most configurations that require `param_offload` and
   `optimizer_offload` on H100/H200 can run without offloading on B200. Only full
   fine-tuning of 72B+ models on small clusters needs offloading.

2. **Higher `gpu_memory_utilization`**: With 183 GB per GPU, you can safely push
   `gpu_memory_utilization` to 0.85 for LoRA workloads. On H100/H200, the verl docs
   recommend 0.4-0.5 for the same models.

3. **Larger micro-batch sizes**: The extra memory allows `ppo_micro_batch_size_per_gpu`
   of 4-8 instead of 1-2, improving training throughput. Increase until you hit OOM.

4. **Lower TP for smaller models**: A 7B model fits entirely in one B200 GPU (14 GB in BF16).
   Use TP=1 to maximize data parallelism and rollout throughput. On H100, you'd need TP=2.

5. **EFA networking**: P6-B200 has 32 EFA adapters per node (6.4 Tbps). Ensure EFA is
   properly configured for multi-node training -- check with `fi_info -p efa`. The
   `NCCL_NET_GDR_LEVEL=2` setting enables GPU Direct RDMA for lowest-latency communication.

6. **NVLink domain**: All 8 B200 GPUs within a single node share a 1.4 TB NVLink domain.
   Intra-node TP communication (the most frequent parallel operation) runs over NVLink
   at full bandwidth. Keep TP <= 8 to stay within a single NVLink domain.

---

## Backend-Specific Notes

### Megatron Backend

The Megatron backend is recommended for models > 32B and all MoE architectures. It requires
[Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) for LoRA support.

**Required for LoRA** (automatically set by `submit_training.py` when `backend=megatron`
and `lora=enabled`):

```bash
actor_rollout_ref.actor.megatron.use_mbridge=True
actor_rollout_ref.actor.megatron.vanilla_mbridge=False
```

**Fused kernel flags** (configured in `conf/backend/megatron.yaml` under `fused_kernels:`,
gated by `backend.use_fused_kernels` which defaults to `true`):

```bash
# These are automatically emitted by submit_training.py when backend=megatron
+actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
+actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True
+actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True
+actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True

# To disable fused kernels:
python3 scripts/submit_training.py backend=megatron backend.use_fused_kernels=false

# To selectively disable individual kernels:
python3 scripts/submit_training.py backend=megatron \
  backend.fused_kernels.gradient_accumulation_fusion=false
```

**Gradient checkpointing** (Megatron equivalent -- use for large models or full fine-tuning):

```bash
+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
```

Per the [verl best practices](https://verl.readthedocs.io/en/latest/perf/best_practices.html),
these trade computation for memory by recomputing activations during the backward pass.

### FSDP Backend

The FSDP backend is simpler to set up and recommended for models <= 72B. It uses
HuggingFace PEFT for LoRA.

**Key FSDP-specific settings** (from the [verl perf tuning guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html)):

```bash
actor_rollout_ref.model.use_remove_padding=True          # Sequence packing -- removes padding for efficiency
actor_rollout_ref.model.enable_gradient_checkpointing=True # Recompute activations to save memory
actor_rollout_ref.model.use_shm=True                      # Preload model to /dev/shm
actor_rollout_ref.actor.fsdp_config.fsdp_size=8           # FSDP sharding group size (typically GPUs per node)
actor_rollout_ref.actor.fsdp_config.forward_prefetch=True  # Overlap next all-gather with computation
```

**FSDP2 migration** (optional, requires PyTorch 2.1+):

```bash
actor_rollout_ref.actor.strategy="fsdp2"  # 7% lower GPU memory, 1.5% throughput improvement
```

Per the [verl perf tuning guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html#migrating-to-fsdp2),
FSDP2 offers better composability with DTensor and per-parameter sharding.

# Backend Comparison

| Feature | Megatron | FSDP |
|---------|----------|------|
| Max model size | 1T+ | ~200B |
| LoRA implementation | Megatron-Bridge native | HuggingFace PEFT |
| Pipeline parallelism | Yes | No |
| Expert parallelism (MoE) | Yes | No |
| Setup complexity | Higher | Lower |
| Throughput (large models) | Higher | Lower |

**Recommendation**: Use Megatron for models >32B or MoE architectures.

## Optimizing for Qwen3-Coder-Next

Once Qwen3-Coder-Next is released, consider these optimizations:

### Code-Specific Settings

```bash
# Longer responses for code generation
python3 scripts/submit_training.py model=qwen3-coder-next training.max_response_length=8192
```

To adjust sampling temperature, modify the `_build_shared_overrides()` function in
`submit_training.py` or pass it as a verl override (not yet exposed in Hydra config):

```bash
# As a verl override:
actor_rollout_ref.rollout.temperature=1.1
```

### If Model is MoE (Mixture of Experts)

```bash
# Enable expert parallelism
python3 scripts/submit_training.py backend=megatron \
  model=qwen3-coder-next model.expert_parallel_size=4

# Add MoE optimizations as verl overrides:
+actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
+actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
+actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=True
```
