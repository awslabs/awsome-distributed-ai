# Verification log

What was run, on what, and what came back. Every metric below was read out of the trainer's
own TensorBoard event files rather than retyped from a terminal, and every run named here has
an event file behind it.

This file exists because an earlier version of the README's verification table contained a
row that was wrong, and a log is what would have caught it. See "A correction" at the end.

## Environment

| | |
|---|---|
| Cluster | Amazon EKS, KubeRay, 2 nodes |
| Instance | `p5en.48xlarge` (H200 141GB x8 per node) |
| Interconnect | EFA between nodes, GPUDirect RDMA |
| Shared storage | FSx for Lustre, PERSISTENT_2, mounted at `/fsx` |
| Base image | `radixark/miles@sha256:ca0bb593dd6f4011b444f64d478b72c213e4c70421f4d7f94e593a709562429e` (tag `dev-202607310056`, cu13) plus the AWS EFA layer built by `miles.Dockerfile` |
| Model | Qwen3-4B, converted to Megatron format with `scripts/convert_checkpoint.sh hf2megatron` |
| Data | DAPO-Math-17k for training prompts, AIME-2024 for eval, `deepscaler` reward |

## Run: the shipped 4B recipe, unmodified

This is the reader's path: the shipped `env_vars.colocated.example` with only the
cluster-specific values filled in (model and data paths, checkpoint and TensorBoard
directories, namespace, FSx claim) and `NUM_ROLLOUT=1` to keep it short. No changes to the
recipe itself.

```
ENV_FILE=env_upstream bash recipe/run_grpo_qwen3_4b.sh
```

Submitted as Ray job `raysubmit_GdnWJy6KJsJDmsF2`, terminal status `SUCCEEDED`.

The flags the job actually received, from the job log rather than from the recipe source:

```
bash grpo_launch.sh \
  --hf-checkpoint /fsx/models/Qwen3-4B \
  --ref-load /fsx/models/Qwen3-4B_torch_dist \
  --load /fsx/runs/upstream_verify/ckpt/qwen3-4b-grpo/ \
  --save /fsx/runs/upstream_verify/ckpt/qwen3-4b-grpo/ \
  --save-interval 1000 \
  --prompt-data /fsx/data/dapo-math-17k/dapo-math-17k.jsonl \
  --input-key prompt --label-key label ...
```

Results, read back from `/fsx/tb/upstream_verify` (78 scalar tags recorded):

| metric | value | reading |
|---|---|---|
| `rollout/raw_reward` | 0.5156 | the model answers correctly about half the time, which is the expected range for 4B on this data |
| `rollout/repetition_frac` | 0.0 | no degenerate generation |
| `rollout/truncated_ratio` | 0.4844 | about half the responses hit the 8192-token cap, normal for this prompt set |
| `train/grad_norm` | 0.6479 | finite and unremarkable |
| `perf/step_time` | 273.3 s | one full rollout-plus-train cycle |

Logs kept in full: 2,242 lines of job log and 2,276 lines of submission log.

## Why reward and repetition are the metrics to check, not the exit code

`rollout/repetition_frac` and `rollout/raw_reward` are what separate a run that works from a
run that merely finishes. A misconfigured 30B MoE run -- one whose SGLang rollout runs the
MoE tensor-parallel and expert-parallel at once (`moe_tp>1` and `moe_ep>1`) -- exits 0, prints
an unremarkable loss, and produces nothing usable: repetition 0.48 to 0.70, reward pinned at
0.0. If you verify by exit code you will record it as working -- which is exactly what happened
once, before the metric was checked. The same 30B model with the shipped rollout geometry
(`moe_tp=1`, pure expert-parallel) trains cleanly: reward 0.578, repetition 0.0. The metric,
not the exit code, is what tells the two apart -- and what pointed to the rollout geometry as
the cause. See "30B MoE root cause" below.

For comparison, on the same cluster and recipe:

| | dense 4B | 30B MoE, `moe_tp=1` (shipped) | 30B MoE, `moe_tp>1` (misconfigured) |
|---|---|---|---|
| `rollout/repetition_frac` | 0.0 | 0.0 | 0.48 to 0.70 |
| `rollout/raw_reward` | 0.516 | 0.578 | 0.0 |
| `rollout/truncated_ratio` | 0.484 | 0.42 | 0.97 to 0.99 |
| exit status | SUCCEEDED | SUCCEEDED | SUCCEEDED |

## Two-node EFA

The 2-node rows in the README's table come from earlier runs on this same cluster with the
worker replica count at 2. NCCL all_reduce busbw measured 190 to 257 GB/s with `efa-direct`
and GPUDirect RDMA and no TCP fallback; `docs/EFA_2NODE.md` has the security-group
prerequisite and the failure signature when it is missing.

`docs/EFA_2NODE.md` is a fabric measurement, though, not a training one, so on its own it does
not support the "3 rollout cycles" part of that row. The GRPO run behind it, Qwen3-4B dense
colocated across 2 nodes (actor 2x8, rollout sharing the same 16 GPU):

| rollout | `rollout/raw_reward` | `actor_train_tflops` | `perf/step_time` |
|---|---|---|---|
| 0 | 0.477 | 100.4 | 119.2s |
| 1 | 0.523 | 245.4 | 73.1s |
| 2 | 0.492 | 235.7 | 81.6s |

All three cycles completed, job `SUCCEEDED` in 676s. Weight sync, rollout, reference log-probs
and the Megatron backward all crossed the node boundary over EFA. `ppo_kl` stayed 0.0 at
dropout 0, matching the single-node arm. Reward moving in the 0.48-0.52 band over three steps
is not a training result -- three optimizer steps show nothing about convergence -- it is
evidence that the loop closes end to end on two nodes, which is what that row claims.

One detail worth repeating from that document, because it cost real time: requesting fewer
EFA devices than the instance exposes lets the device plugin pick different cards on each
node, and NCCL then fails with `NET/OFI Unexpected number of remote rails`. It only shows up
on point-to-point traffic (pipeline or expert parallel), so an all-reduce smoke test passes
and the problem surfaces later.

## What this log does not cover

- The disaggregated reward service. Present in the repository, never deployed.
- Any B300 hardware.
- Checkpoint save-back. `save_model()` fails with a pickle-truncation error inside Megatron's
  distributed checkpoint save; `SAVE_INTERVAL` ships above the step count so a run does not
  trigger it. Known Issues item 1.

## Ray head must run on a GPU (CUDA-capable) node, not a CPU-only node

An earlier version of this test case placed the Ray head on a CPU-only node (to keep it off
the expensive GPU pool). That shape is not safe for miles and the shipped manifest no longer
uses it. The reason is in the framework, not the recipe: miles's Ray control actors are
created with `num_gpus=0` (e.g. `create_rollout_manager` in `miles/ray/placement_group.py`
does `RolloutManager.options(num_cpus=1, num_gpus=0)`), but their module import pulls in
Megatron / `transformer_engine`, whose shared library `dlopen`s `libcuda.so.1` at import time.
Ray is then free to place such a zero-GPU actor on any node with a spare CPU -- including a
CPU-only head node -- where the import hard-fails:

```
OSError: libcuda.so.1: cannot open shared object file: No such file or directory
  ... File ".../transformer_engine/common/__init__.py", ... _load_core_library()
  ray.exceptions.ActorDiedError: RolloutManager.__init__() ... (TemporaryActor, ip=<head node>)
```

So any CPU-only node in the miles Ray cluster is a latent hazard: whether a run survives
depends on whether Ray happens to place the Megatron-importing actor on a GPU worker instead.
The earlier "head-on-CPU SUCCEEDED (reward 0.531)" run was that placement luck, not a
guarantee. The fix is to keep every node in the Ray cluster CUDA-capable: the shipped
`raycluster.yaml` now schedules the head onto the GPU pool (`CPU_NODE_ROLE` defaults to
`GPU_NODE_ROLE`) with a `nvidia.com/gpu` toleration; the head still runs `num-gpus 0` and
consumes no GPU, so on a colocated run it simply co-locates on a worker's GPU node at no extra
cost, and the ~18 GB image is already cached there. This also removes the head from the CPU
Karpenter pool, sidestepping the "underutilized" consolidation churn that pool's 30s policy
caused. The verified metrics for the colocated dense 4B run (reward 0.53, repetition 0.0,
weight_version uniform) are unchanged; only the head's node placement changed.

Note (upstream): the sibling `slime` test case ships the same head-on-CPU shape and the same
`num_gpus=0` control-actor pattern, so it shares this latent hazard; worth raising upstream.

## Second review round: defects found by reading, and what was re-run to confirm

A later review pass went through the reader's path again rather than re-running the same job,
and found six defects that no amount of re-running would have surfaced, because the affected
paths either are not on the default path or fail silently. They are listed here because the
fixes changed shipped files, and a log that only records successes is not much of a log.

| what was wrong | why it was silent |
|---|---|
| `kubernetes/reward-service.yaml` referenced `${REWARD_IMAGE}`, which no env file defined | `envsubst` renders an undefined variable as the empty string, so the Deployment went out with `image: ""` |
| `raycluster.yaml` hardcoded `replicas: 1` with a comment telling the reader to edit it for 2-node runs | a 2-node config (the 30B MoE block sets `ACTOR_NUM_NODES=2`) then started 8 GPU worth of workers and the job waited on placement rather than erroring. Now driven by `${ACTOR_NUM_NODES}` |
| `raycluster.yaml` requested `vpc.amazonaws.com/efa: 8` of the node's 15 allocatable | a partial request breaks point-to-point NCCL (see the rails note above) but passes an all-reduce smoke test. Now `${EFA_PER_NODE}` |
| `scripts/evaluate.sh` keyed pass@k on `prompt_item.get("idx", 0)` | the prepared AIME-2024 file has only `prompt` and `label`, so all 30 prompts collapsed onto key 0 and pass@k became "any of the 480 samples was correct". Reproduced: for a model correct on 1 prompt of 30, the old code reports 1.0000 where the truth is 0.0333 |
| `scripts/evaluate.sh` extracted answers with `\\boxed\{([^}]*)\}` | the character class stops at the first brace, so `\boxed{\frac{1}{2}}` yielded `\frac{1` and was scored wrong. Now a brace-counting scan |
| `scripts/evaluate.sh` submitted every request at once under a 300s timeout | with `MAX_TOKENS=16384` most requests time out in the queue, and the handler counts a timeout as an incorrect answer, so accuracy sags for a reason unrelated to the model. Now bounded concurrency, a generation-sized timeout, and a non-zero exit when the error rate exceeds 5% |

The README's hardware table also said "EFA per node: 16 devices" while EKS reports 15
allocatable on `p5en.48xlarge`; `docs/EFA_2NODE.md` still described a partial EFA request as a
throughput trade-off rather than a correctness problem; and step 4's prose offered "exec into
its head" for the checkpoint conversion two lines above a code block that correctly targets a
worker. All three are corrected.

What was re-verified on hardware after these changes, rather than assumed:

- All four manifests were rendered through `envsubst` from the shipped example env and
  validated with `kubectl apply --dry-run=server` against a live EKS cluster. All four are
  accepted, including `reward-service.yaml`, which previously could not be.
- `raycluster.yaml` was rendered at both `ACTOR_NUM_NODES=1` and `=2` and validated at each;
  `replicas`, `minReplicas` and `maxReplicas` track the value, and `vpc.amazonaws.com/efa`
  resolves to 15, matching what the node reports as allocatable.
- The `\boxed{}` extractor and the pass@k indexing were checked against the actual
  `aime-2024.jsonl` on the cluster, which is where the missing `idx` field was confirmed.
- `bash -n` passes on every script, and the Python inside the `evaluate.sh` heredoc compiles.

## slime-parity comparison (model x topology coverage)

The sibling `3.test_cases/pytorch/slime` README lists a "Supported Model Sizes" matrix
(Qwen3-4B colocated, GLM-Z1-9B colocated, Qwen3-30B-A3B disaggregated, Qwen2.5-72B
disaggregated) but publishes no measured results for it. The runs below cover that matrix on
miles, `NUM_ROLLOUT=2`, reading metrics from the trainer's event files. Rows marked
(shipped recipe) use the recipes in this test case unmodified; rows marked (campaign config)
used an authored model script or a `--colocate`-conditional recipe variant and are reported as
findings, not as shipped support.

Hardware shorthand used in the HW column:

- **P5ENx1** = 1x `p5en.48xlarge` (8x H200, 141 GiB each).
- **P5ENx2** = 2x `p5en.48xlarge` (16x H200).
- **P6B300x2** = 2x `p6-b300.48xlarge` (16x B300, 288 GiB each) -- the expected-compatible target
  for cases that do not fit H200; not run here.

The HW column names the configuration a row was actually run on, so a "does not fit" is scoped to
that hardware rather than read as a property of the model.

| Model | Layout | HW | Result | reward | repetition | notes |
|---|---|---|---|---|---|---|
| Qwen3-4B dense | colocated (shipped recipe) | P5ENx1 | SUCCEEDED | 0.53 | 0.0 | head co-located on GPU pool, above |
| Qwen3-4B dense | disaggregated `COLOCATE=false` (shipped recipe) | P5ENx2 | SUCCEEDED | 0.52 | 0.0 | weight sync over NCCL/EFA (`weight_version` 2, mixed 0.0); worker `replicas` must cover actor+rollout GPUs |
| GLM-Z1-9B dense | colocated TP2 (campaign config) | P5ENx1 | SUCCEEDED | 0.68 | 0.0 | TP>1 requires `CUDA_DEVICE_MAX_CONNECTIONS=1` in the Ray runtime-env |
| Qwen3-30B-A3B MoE | colocated, `moe_tp=1` pure EP (shipped recipe) | P5ENx2 | SUCCEEDED | 0.578 | 0.0 | shipped 30B block, EP_SIZE=ROLLOUT_GPUS_PER_ENGINE=2 so the rollout MoE is pure expert-parallel; trains cleanly, comparable to dense 4B. `--use-distributed-optimizer` shards the 30B optimizer state to fit H200 |
| Qwen3-30B-A3B MoE | SGLang `moe_tp>1` and `moe_ep>1`, fusion left on (campaign config) | P5ENx2 | completes, degenerate | 0.0 | 0.56 to 0.80 | root cause, below: a FlashInfer allreduce-fusion bug drops the moe-tp reduce in this combined path. Reproduced disaggregated (per-engine 4 / EP 2 -> moe_tp=2) and in stock serving |
| Qwen3-30B-A3B MoE | colocated, `moe_tp=2` x `moe_ep=2`, fusion disabled (recipe auto) | P5ENx2 | SUCCEEDED | 0.555 | 0.0 | per-engine 4 / EP 2 -> moe_tp=2; the recipe adds `--sglang-enforce-disable-flashinfer-allreduce-fusion` automatically and the combined geometry trains cleanly, comparable to pure EP (0.578) |
| Qwen2.5-72B dense | disaggregated TP4 PP2, actor8+rollout8 (campaign config) | P5ENx2 | OOM | -- | -- | did not fit on P5ENx2: ~144 GiB/GPU (8-way shard, DP=1 so Adam cannot shard) vs 141 GiB. Expected to fit **P6B300x2** (288 GiB) or an H200 layout with optimizer sharding (DP>1 / TP8 / offload) -- not run. `rms_norm_eps` is 1e-6, not 1e-5 |

Takeaways: the disaggregated (`COLOCATE=false`) weight-sync path works on miles and is now
verified, not just argv-rendered. The dense 4B and GLM-Z1-9B cases train cleanly, and the 30B
MoE trains cleanly too once the SGLang rollout runs pure expert-parallel (`moe_tp=1`): reward
0.578, no repetition, comparable to the dense 4B run. What degenerates is not "the 30B MoE" or
"SGLang expert parallelism" -- pure expert-parallel (`moe_ep=8`) and pure tensor-parallel
(`moe_ep=1`) both generate cleanly -- but specifically the combined path where the rollout MoE
runs tensor-parallel AND expert-parallel at once (`moe_tp>1` and `moe_ep>1`). The earlier
"colocated 30B degenerates" reading conflated the Megatron TP/EP labels with the SGLang rollout
geometry; the run that degenerated had the rollout at `moe_tp>1`, and the shipped colocated
block (`moe_tp=1`) does not. See "30B MoE root cause" below.
The 72B dense case did not fit the 16-GPU disaggregated layout on H200 in the configurations
tried (DP=1 leaves the optimizer unshardable; DP>1 / TP8 / offload were not attempted) -- which
is consistent with slime listing 72B as a config without measured evidence.

On the comparison with slime specifically: slime's "Supported Model Sizes" table lists
Qwen3-30B-A3B and Qwen2.5-72B as parallelism configurations (TP/PP and rollout/training GPU
counts), but ships no runnable env for them and reports no reward/success metric, so it is not
evidence that either trains -- the 30B is verified here (reward 0.578) and unverified on slime,
and the 72B is unverified on both:

- The 72B layout slime tabulates (TP4 PP2, training on 8 GPUs, so 8-way sharding with DP=1) needs
  roughly 18 GiB weights + 18 GiB grads + ~108 GiB Adam state = ~144 GiB per GPU with a naive
  distributed optimizer that cannot shard at DP=1. That exceeds the H200's 141 GiB here (hence the
  OOM) and is well above the H100 80 GiB in slime's own table -- i.e. the tabulated layout does not
  fit either card as written, which is why "slime does it on p5" has no measured run behind it.
  Fitting 72B GRPO needs optimizer sharding (DP>1, i.e. more actor GPUs / the colocated-16 layout),
  heavier model parallel (TP8), or CPU/optimizer offload -- none of which were attempted here.

## 30B MoE root cause

The 30B MoE degeneration is not "the 30B model" and not "SGLang expert parallelism". It is a
single, narrow condition in the SGLang build shipped in the miles image: the rollout MoE
corrupts its own output when it runs **tensor-parallel and expert-parallel at the same time**
(`moe_tp>1` and `moe_ep>1`). Either axis alone is fine.

SGLang derives the rollout MoE geometry from two recipe flags:
`moe_ep = --sglang-expert-parallel-size` (EP_SIZE) and
`moe_tp = --rollout-num-gpus-per-engine / EP_SIZE`. So the trigger is set by the ratio of the
per-engine GPU count to EP_SIZE, not by the Megatron TP/EP -- which is why labelling the runs
by Megatron TP/EP hid it.

GRPO, same 30B model and cluster, `NUM_ROLLOUT=2`, metrics from the trainer's event files:

| rollout geometry | `moe_tp` x `moe_ep` | reward | repetition |
|---|---|---|---|
| colocated, per-engine 2, EP 2 (shipped) | 1 x 2 (pure EP) | 0.578 | 0.0 |
| colocated, per-engine 2, EP 1 | 2 x 1 (pure TP) | 0.531 | 0.0 |
| disaggregated, per-engine 4, EP 2 | 2 x 2 (combined) | 0.0 | 0.56 |

Reproduced without any RL or weight-update code, in a stock `sglang.Engine` on the same
`/fsx/models/Qwen3-30B-A3B`, `temperature=0.0`, on math prompts (4-gram repetition score):

| `tp_size` | `ep_size` | `moe_tp` x `moe_ep` | mean repetition | sample output |
|---|---|---|---|---|
| 8 | 8 | 1 x 8 (pure EP) | 0.009 | coherent ("...Okay, so I need to solve the equation 3x + 7 = 22...") |
| 8 | 4 | 2 x 4 (combined) | 0.807 | " 7. 7. 7. 7..." |
| 8 | 2 | 4 x 2 (combined) | 0.327 | ",,,,, and and and", "10101010...", "aaaa..." |

That the stock engine reproduces it rules out the miles RL path (weight sync, the on-policy
topk branch, Megatron) as the cause; it is in SGLang's serving path.

What the code shows (SGLang `0.5.16.dev` in the image):

- `models/qwen3_moe.py` `forward_normal` runs two post-experts all-reduces -- an expert-parallel
  one over the moe-ep group (guarded by `self.ep_size = moe_ep_size > 1`) and a tensor-parallel
  one over the moe-tp group (guarded by `self.tp_size = moe_tp_size > 1`). With `moe_tp=1` the
  second is skipped; with `moe_ep=1` the first is skipped; only the combined case runs both.
- `should_skip_post_experts_all_reduce` returns `False` for both paths in this configuration
  (triton runner, no dp-attention, no flashinfer/deepep A2A, no reduce-scatter), so neither
  reduce is being dropped -- a missing reduce is not the cause.
- the standard dispatcher's `local_expert_mapping` is built from `moe_ep_rank` only, and the
  moe-tp ranks inside one expert-parallel group correctly share it, so the global->local expert
  mapping is not itself wrong.

The `forward_normal` reduces, the moe-ep/moe-tp process groups (orthogonal by construction: for
`tp=8, ep=2` the ep groups are `{0,4},{1,5},{2,6},{3,7}` and the tp groups `{0,1,2,3},{4,5,6,7}`),
the weight sharding and the dispatcher's `local_expert_mapping` are all correct. The defect is in
the **FlashInfer allreduce+RMSNorm fusion**. On SM90/SM100 it is auto-enabled for Qwen3-MoE
(`tp_size>1`, no dp-attention, `moe_a2a=none`) regardless of `moe_ep`/`moe_tp`. With it on, both
post-experts reduces in `forward_normal` are skipped and deferred to the next layer's fused
`layernorm.forward_with_allreduce_fusion`; that fused reduce
(`layernorm.py::_forward_with_allreduce_fusion`, `flashinfer_comm_fusion.py`) picks its group with
`if moe_ep_size>1: moe_ep_group else: moe_tp_group`, treating the two axes as mutually exclusive.
When both are `>1` it reduces over the moe-ep group only and never over the moe-tp group, so each
rank keeps a partial sum over the intermediate dimension -- a total collapse from layer 0. Pure EP
and pure TP escape it because `_MOE_EP`/`_MOE_TP` alias the full TP group, so either branch reduces
over all ranks. Causal proof: rerunning the same combined config (`tp=8, ep=2, moe_tp=2`, triton)
with `enforce_disable_flashinfer_allreduce_fusion=True` generates cleanly (4-gram repetition 0.006,
same as pure EP). This is a genuine SGLang bug; a minimal stock-`sglang.Engine` reproducer is
captured for an upstream report.

The recipe's response is to disable the fusion for the affected geometry rather than forbid it:
`run_grpo_qwen3_30b_a3b.sh` computes `moe_tp = ROLLOUT_GPUS_PER_ENGINE / EP_SIZE`, and when both
`moe_tp>1` and `moe_ep>1` it adds `--sglang-enforce-disable-flashinfer-allreduce-fusion` so the
combined geometry trains correctly too; pure EP (the shipped default) and pure TP keep the fusion.

## A correction

This log has now corrected the 30B MoE row twice, and both corrections are worth keeping
visible. First, an early table listed the configuration as simply "Verified" on the strength
of a smoke run that exited 0 while producing reward 0.0 and repetition 0.96 -- "the job
completed" written up as "the configuration works". The table was split to separate those
claims. Second, the follow-up reading -- "the 30B MoE degenerates, cause suspected in SGLang
expert parallelism" -- was itself too broad: it generalised from runs that happened to have
the rollout at `moe_tp>1`, and it labelled runs by Megatron TP/EP, which is not what sets the
SGLang rollout geometry. Measuring the model across the actual rollout geometries showed the
shipped colocated block trains cleanly (reward 0.578) and isolated the real trigger. The
lesson both times is the same: read the metric, name the exact variable, and do not let a
plausible summary outrun the measurement.
