<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Megatron-Bridge + UCCL-EP over EFA

This directory is the **library-level**, **model-agnostic** home for
[NVIDIA Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) test cases
that run Mixture-of-Experts (MoE) training on Amazon EKS with
[UCCL-EP](https://github.com/uccl-project/uccl) carrying the expert-parallel
all-to-all over **AWS EFA**.

The crux is replacing NVIDIA [DeepEP](https://github.com/deepseek-ai/DeepEP) (which is
built on NVSHMEM + InfiniBand verbs and does **not** run on EFA) with UCCL's EFA-native
drop-in — **without patching Megatron-Core**. UCCL ships a top-level `deep_ep` shadow
module; because it installs into `site-packages`, `import deep_ep` resolves to UCCL's
EFA RDMA implementation. Megatron-Core's MoE `flex`/`deepep` dispatcher then sends its
all-to-all bytes over EFA via UCCL + GDRCopy instead of over IB verbs via NVSHMEM.

## Layout

The container environment (Dockerfile + its build/validation scripts) lives here at the
library level and is **shared by every model** under it. Per-model recipes (checkpoint
conversion, the SFT `conf`, deployment manifests, benchmarks) live in a model subdirectory.

```text
megatron-bridge/                  # <library> — model-agnostic environment
├── Dockerfile                    # NGC NeMo base + EFA/GDRCopy + UCCL + deep_ep shadow
├── 1.build-and-push.sh           # build the shared env image and push the pinned tag to ECR
├── 2.sanity-singlenode.sh        # single-node 8-GPU deep_ep/EFA/EP smoke gate (run in the image)
├── test_megatron_bridge_uccl.py  # CI build smoke test for the shared image
└── kimi-k2/                      # <model> — Kimi K2 full-parameter SFT recipe
    ├── README.md
    ├── 1.convert-checkpoint.sh
    ├── conf/                     # SFT ConfigContainer (mounted into the image at runtime)
    ├── kubernetes/
    └── benchmarks/
```

The image is **model-agnostic**: SFT configs are **not** baked in. Each model mounts its
own `conf/` at `/workspace/conf` at runtime (e.g. via a ConfigMap — see the model's
`kubernetes/README.md`), so one image serves every model under this library.

## Shared environment workflow

These two steps build and validate the shared image and apply to **all** models. Run them
from this directory, then continue in the model subdirectory.

### 1. Build the environment image and push to ECR

`1.build-and-push.sh` builds `Dockerfile`, creates the ECR repository if needed
(`megatron-bridge-uccl`), logs in, and pushes the pinned tag `nemo-26.04.01-uccl-0dc87eb`
(no `latest`). The Dockerfile starts from `nvcr.io/nvidia/nemo:26.04.01` (CUDA 13.1,
PyTorch 2.11), which already ships **Megatron-Bridge v0.4.2**, Megatron-Core `0.17.1` (with
the `flex`/`deepep` dispatcher), and TransformerEngine — these are **not** reinstalled. (The
older `25.11.01` base shipped Megatron-Bridge 0.2.0, whose flex/deepep GPU allowlist rejects
p6-b300; 0.4.0+ fixes it — see the Dockerfile header.) It then strips the IB fabric, lays down
GDRCopy `v2.5.2` + the EFA installer `1.48.0`, and builds UCCL (pinned to commit `0dc87eb`) /
UCCL-EP for `sm_103` (B300) plus the `deep_ep` shadow.

```bash
bash 1.build-and-push.sh
# Image: <account>.dkr.ecr.us-west-2.amazonaws.com/megatron-bridge-uccl:nemo-26.04.01-uccl-0dc87eb
```

### 2. Single-node sanity gate

**Do not skip this.** It is far cheaper to fail on 1 node than to burn 32 capacity-block
nodes. `2.sanity-singlenode.sh` runs a single-node, 8-GPU smoke test **inside the image**
that confirms the UCCL `deep_ep` wrapper is active, EFA is present, and the
Megatron-Core flex/deepep dispatcher config is wired.

```bash
# inside the container on one p6-b300.48xlarge node:
bash 2.sanity-singlenode.sh
```

> **Note (nemo:26.04.01 / Megatron-Core 0.17.1):** Gates 1–4 (EFA device present, UCCL
> `deep_ep` active with `Buffer`, MCore flex/deepep config fields, NCCL all-reduce over
> EFA) pass. **Gate 5** — a hand-rolled `MoEFlexTokenDispatcher` micro-step — is **stale**
> on Core 0.17.1: its standalone setup predates the `ProcessGroupCollection` API and
> raises a `pg_collection` error. This is **not** a UCCL/image fault — the real
> `pretrain()` path builds the process groups internally and dispatches correctly (the
> multi-node benchmark runs clean through `MoEFlexTokenDispatcher(backend="deepep")`).
> Treat the multi-node run / [`benchmarks`](kimi-k2/benchmarks/RESULTS.md) as the
> authoritative end-to-end dispatch check until Gate 5 is ported to the 0.17.1 API.

## Models

| Model | Directory | Recipe |
|-------|-----------|--------|
| [Kimi K2](https://huggingface.co/moonshotai/Kimi-K2-Base) (1.04T MoE) | [`kimi-k2/`](kimi-k2/) | Full-parameter SFT on 32× p6-b300 (256× B300) |

To add a model: create `megatron-bridge/<model>/` with its `conf/`, deployment manifests,
and a model README. Reuse the shared image from step 1 (mount the model's `conf` at runtime)
— do **not** add a second Dockerfile.

## Benchmark result — UCCL-EP vs NCCL all-to-all (256× B300)

The headline measurement this environment was built for: swapping **only** the
Megatron-Core MoE token dispatcher — NCCL all-to-all (baseline) vs UCCL's EFA-native
`deep_ep` (treatment) — on a live 32× p6-b300.48xlarge (256× B300) block. Model:
DeepSeek-V3 256-expert MoE (Kimi-K2 family), TP8/PP8/EP32/DP4, seq 4096, bf16, balanced
routing. Everything else is held byte-identical across arms. Full methodology, caveats,
and raw numbers: [`kimi-k2/benchmarks/RESULTS.md`](kimi-k2/benchmarks/RESULTS.md).

At the throughput-efficient operating point (micro-batch ≥ 4), **UCCL `deep_ep` is
~36% faster than NCCL all-to-all**, and the advantage **holds under deployment-realistic
1F1B overlap**. NCCL wins only at micro-batch 1 (64 tiny dispatches — UCCL-EP's
per-dispatch overhead unamortized), an operating point no throughput-tuned run uses.

| micro-batch | overlap | NCCL all-to-all | UCCL `deep_ep` | dispatcher delta |
|------------:|---------|----------------:|---------------:|------------------|
| 1 | off | 12.54 s | 14.12 s | NCCL **+12.6%** faster |
| 4 | off |  9.77 s | **6.26 s** | UCCL **−36.0%** faster |
| 4 | on  |  5.98 s | **3.84 s** | UCCL **−35.8%** faster |

> Mean training-iteration time (lower = better), median over 16 steady-state iters
> after warmup, 0 stalls, EFA-active on every rank. Work-equivalence (no token dropping)
> verified two ways: drop-free config + an iteration-1 loss match (deepep 11.897349 vs
> alltoall 11.897517). `overlap=on` is a separate within-regime A/B (VPP=2 + recompute
> off on both arms) — do not subtract its numbers against the `overlap=off` rows.

## References

- [NVIDIA Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)
- [Megatron-Bridge docs](https://docs.nvidia.com/nemo/megatron-bridge/)
- [UCCL project](https://github.com/uccl-project/uccl)
- Sibling case: [`../megatron-lm`](../megatron-lm) (EFA/GDRCopy Dockerfile + PyTorchJob template)
