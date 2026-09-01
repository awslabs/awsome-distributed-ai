<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Qwen3-30B-A3B (128-expert MoE) — NCCL vs DeepEP+UCCL vs DeepEP+NVSHMEM (p5.48xlarge / H100)

The same three-way MoE token-dispatcher comparison as [`../qwen3-235b/`](../qwen3-235b/), but on
**`p5.48xlarge` (H100-80GB, sm_90, 32× EFA / node)** using **Qwen3-30B-A3B** — the shipped smaller
Qwen3 MoE (30B total / 3B active, **128 experts**, top-8, 48 layers, hidden 2048). Qwen3-235B does
**not** fit H100-80GB at EP16/EP32, so the H100 comparison uses the 30B sibling, which fits with
**recompute off** (the cleanest dispatcher-isolation regime).

This pairs with the B300 / Qwen3-235B results to show the dispatcher comparison **across two
fabrics/GPUs**: p6-b300 (B300, 16×400G EFA, ~6.4 Tbps/node) vs p5.48xlarge (H100, 32×100G EFA,
~3.2 Tbps/node).

## EP > 8 (internode all-to-all)

With **TP8** occupying one node's NVLink domain, **EP > 8 forces the expert all-to-all across
nodes** (EP16 spans 2 nodes, EP32 spans 4) — i.e. genuine internode EFA traffic, not intranode
NVLink. Both swept points satisfy this:

| EP | TP | PP | DP | nodes the EP group spans |
|---:|---:|---:|---:|---|
| 16 | 8 | 4 | 2 | 2 (internode) |
| 32 | 8 | 2 | 4 | 4 (internode) |

`PP = WORLD/EP` (WORLD = 64 = 8 nodes × 8 H100). 48 layers is VPP-divisible for both, so
`overlap=on` needs no layer round-up (unlike Qwen3-235B's 94).

## Shared harness

This model reuses the shared Qwen3 bench and campaign (one bench, selected by `QWEN3_SIZE`):
- bench: [`../qwen3-235b/benchmarks/bench_qwen3_pretrain.py`](../qwen3-235b/benchmarks/bench_qwen3_pretrain.py) (`QWEN3_SIZE=30b`)
- campaign: [`../qwen3-235b/benchmarks/run-qwen3-campaign.sh`](../qwen3-235b/benchmarks/run-qwen3-campaign.sh) (`MODEL=qwen3-30b`)

Both DeepEP arms need **sm_90** images (the B300 images are Blackwell-only); build them with:

```bash
# from megatron-bridge/ — add Hopper to the arch list:
docker build -f Dockerfile --build-arg EP_BACKEND=uccl    --build-arg TORCH_CUDA_ARCH_LIST="9.0a+PTX" \
  -t <repo>:nemo-26.04.01-uccl-0dc87eb-sm90 .
docker build -f Dockerfile --build-arg EP_BACKEND=nvshmem --build-arg DEEPEP_GPU_ARCH="90" \
  -t <repo>:nemo-26.04.01-deepep-nvshmem-567632d-cu13-sm90 .
```

Run the campaign on 8× p5.48xlarge (recompute off — 30B fits H100):

```bash
CTX=<ctx> NS=kimi-k2-bench MODEL=qwen3-30b NNODES=8 EPS="16 32" CELLS="4:off 1:off 4:on" RECOMPUTE="" \
TRAIN_ITERS=12 GLOBAL_BATCH=256 INSTANCE_TYPE=p5.48xlarge EFA_PER_NODE=32 \
UCCL_IMG=<repo>:nemo-26.04.01-uccl-0dc87eb-sm90 \
NVSHMEM_IMG=<repo>:nemo-26.04.01-deepep-nvshmem-567632d-cu13-sm90 \
  bash ../qwen3-235b/benchmarks/run-qwen3-campaign.sh
```

Measured numbers: [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).
