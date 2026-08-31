<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Qwen3-235B-A22B (128-expert MoE) — NCCL vs DeepEP+UCCL vs DeepEP+NVSHMEM

A **three-way** Megatron-Core MoE token-dispatcher comparison on up to **8× p6-b300**
(64× B300), measured as end-to-end **training-step throughput**. The only thing that
changes between arms is the expert-parallel all-to-all backend:

| Arm | `MOE_DISPATCHER` | Image | all-to-all transport |
|-----|------------------|-------|----------------------|
| **NCCL all-to-all** (baseline) | `alltoall` | UCCL image | NCCL all-to-all over EFA |
| **DeepEP + UCCL** | `deepep` | UCCL image | UCCL EFA-native `deep_ep` |
| **DeepEP + NVSHMEM** | `deepep` | **NVSHMEM image** | NVIDIA DeepEP over NVSHMEM-libfabric/EFA |

The two `deepep` arms run the **identical** Megatron-Core flex/deepep code path; they differ
only in which `deep_ep` the image provides (see [`../README.md`](../README.md) for the
`EP_BACKEND` build arg). Everything else — model, data, parallelism, precision, seed, EFA env,
`MOE_A2A_OVERLAP` — is held byte-identical across arms within a run.

> **The DeepEP+NVSHMEM arm is new to this library.** The parent README originally noted that
> NVIDIA DeepEP "does not run on EFA". That is no longer true: the
> [`deepep-benchmark`](../../../../micro-benchmarks/expert-parallelism/deepep-benchmark)
> patches DeepEP to run **NVSHMEM over libfabric** (host-proxy, IBGDA off) on EFA, and this
> test case reuses that build (vendored at [`../deepep/`](../deepep/)) as a Megatron-Core
> dispatcher backend.

## Why Qwen3-235B-A22B (and not DSV3) for the 8-node comparison

The DSV3 dispatcher A/B ran at **EP32 on 32 nodes** to fit the 671B model. On 8 nodes the
671B DSV3 cannot reach the EP32 regime without OOM. **Qwen3-235B-A22B** (235B total / 22B
active, **128 experts**, top-8, 94 layers, hidden 4096, no shared expert) fits 8 nodes at
**both EP16 and EP32**, so the comparison sweeps the variable the all-to-all turns on — the
**EP degree** — at a realistic operating point (EP32 is the Qwen3-235B point cited in the
DSV3 reference table). It is built from the shipped Megatron-Bridge recipe
`megatron.bridge.recipes.qwen.qwen3_moe.qwen3_235b_a22b_pretrain_config` with **mock data +
random-init weights** (we measure step time, not loss).

## Parallelism (world = 64 = TP·PP·DP; EP divides TP·DP and the 128 experts)

| EP | TP | PP | DP | experts/rank |
|---:|---:|---:|---:|---:|
| 16 | 8 | 4 | 2 | 8 |
| 32 | 8 | 2 | 4 | 4 |

`PP = WORLD / EP` (TP8 fixed, ETP=1). The table above is the **8-node target** (WORLD=64).
The measured campaign in [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) ran on **4 nodes**
(WORLD=32 → EP16=PP2/DP2, EP32=PP1/DP4) because the shared cluster had only 6 of 8 B300 nodes
free; the EP degree (the variable that matters) is preserved either way.

Qwen3's 94 layers (= 2·47) are not VPP-divisible, so
the `overlap=on` cells round `num_layers` up to the nearest `PP·VPP` multiple (94 → 96) and
disable the embedding/loss pipeline accounting — making `overlap=on` a separate within-regime
A/B (never subtract its numbers against `overlap=off`).

## Run it

```bash
# from megatron-bridge/ — build both images once (see ../README.md):
bash 1.build-and-push.sh                       # UCCL    -> nemo-26.04.01-uccl-0dc87eb
EP_BACKEND=nvshmem bash 1.build-and-push.sh    # NVSHMEM -> nemo-26.04.01-deepep-nvshmem-567632d-cu13

# single-node bring-up gate per image (run inside the image on one p6-b300 node):
EP_BACKEND=uccl    bash 2.sanity-singlenode.sh
EP_BACKEND=nvshmem bash 2.sanity-singlenode.sh

# the full 3-way x EP{16,32} campaign on 8 nodes:
CTX=<kube-context> \
UCCL_IMG=<acct>.dkr.ecr.<region>.amazonaws.com/megatron-bridge-uccl:nemo-26.04.01-uccl-0dc87eb \
NVSHMEM_IMG=<acct>.dkr.ecr.<region>.amazonaws.com/megatron-bridge-uccl:nemo-26.04.01-deepep-nvshmem-567632d-cu13 \
  bash qwen3-235b/benchmarks/run-qwen3-campaign.sh
```

The campaign ([`benchmarks/run-qwen3-campaign.sh`](benchmarks/run-qwen3-campaign.sh)) runs
each arm serially via the shared [`../run-ab-rawpods.sh`](../run-ab-rawpods.sh), gates every
run on EFA-active on all ranks, and parses the tree with
[`../bench/parse-runs.py`](../bench/parse-runs.py) into `index.csv`. Measured results:
[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

**Confounder guard.** Every rank of every run must log
`NET/OFI Selected provider is efa` (true EFA RDMA, no socket fallback); the NVSHMEM arm must
also show NVSHMEM-over-libfabric init. Work-equivalence (no token dropping from the NVSHMEM
arm's put_signal + host-proxy shim) is checked with `LOSS_PROBE=1`: iteration-1 loss must
match the `alltoall` arm to ~5 significant figures.
