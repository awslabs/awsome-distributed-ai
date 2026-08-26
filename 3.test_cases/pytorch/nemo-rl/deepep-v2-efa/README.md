<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0 -->
# NeMo-RL + DeepEP V2 — MoE expert-parallel all-to-all over AWS EFA (NCCL-GIN CPU-proxy)

This test case wires **[NeMo-RL](https://github.com/NVIDIA-NeMo/RL)** (GRPO post-training) and
**[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)** (the training backend) to **DeepEP V2's
NCCL backend** (`ElasticBuffer`, merged upstream in
[deepseek-ai/DeepEP#605](https://github.com/deepseek-ai/DeepEP/pull/605)) on **AWS EFA**. Unlike
the NVSHMEM-path DeepEP samples (e.g. the SGLang sibling), V2's network traffic rides **NCCL's GIN
(GPU-Initiated Networking) device API**, which on EFA means **aws-ofi-nccl with GIN compiled in**,
running GIN's **CPU-proxy** mode. The image is built NGC-from-scratch from public sources only; the
recipe takes you from image build → static verify → a cross-node dispatch/combine probe → an N-step
Megatron MoE training gate — and documents, honestly, which parts of the full GRPO rollout path
still depend on draft upstream PRs.

## The mechanism chain

```
NeMo-RL (GRPO) ── nemo_rl.models.megatron ── Megatron-core MoE flex dispatcher (fused_a2a)
                     └─ deep_ep.ElasticBuffer (DeepEP V2, NCCL backend)
                           └─ NCCL 2.30.4 GIN device API (NCCL_GIN_TYPE=2, CPU-proxy)
                                 └─ aws-ofi-nccl @9c44d34 (ncclGinPlugin, gdrcopy)
                                       └─ libfabric ── EFA (efa-direct, SRD)
```

## Measured vs staged (read this first)

**Measured (2026-05-06, "Wave 28"):** the NeMo-RL 0.5.0rc0 full stack ran E2E on **2×
p5.48xlarge (16× H100)** over EFA with this exact mechanism chain — imports + trainer bring-up,
rollout shape `[64, 8192]` generated in **9.45 s**, Megatron "Shape Y" 3-step training with loss
**26.41 → 24.62**, `ElasticBuffer` confirmed as the active buffer class, and EFA cross-node
hardware TX counters advanced. That run used a privately prebuilt image cascade, which this repo
does not allow as a base — **this folder re-cuts the same substrate NGC-from-scratch from public
sources.**

**Staged, NOT re-measured:** the image assembly in this folder is **build-staged and has not been
cluster-re-run**; no performance numbers are published from this folder. The **full GRPO
rollout-over-DeepEP path depends on 3 draft upstream PRs** (opt-in image layer, default **OFF** —
see below). What the recipe gates re-verify on the **baseline (upstream-only)** image: static
substrate asserts (`verify-image.sh`), cross-node `ElasticBuffer` dispatch/combine with an EFA
TX-counter assert (`run-rollout-probe.sh`), and a loss-decreasing Megatron MoE train step on the
stock `alltoall` dispatcher (`train-step.sh`). Never read a build-gate as an E2E pass.

## Pins (every one justified)

| Component | Pin | Why |
|---|---|---|
| Base image | `nvcr.io/nvidia/pytorch:26.02-py3` | Same NGC base as the slime RL sibling. Bakes torch 2.11/CUDA 13 with TransformerEngine/apex/flash-attn compiled against that exact ABI — what megatron.core's H100 path needs. |
| EFA installer | 1.48.0 | The userspace of the measured NCCL-GIN substrate. Bumping is a re-measure event. |
| NCCL | `v2.30.4-1` (source build) | GIN device API generation (`nccl_device.h` asserted at build). Same NCCL line the NGC base bakes — the ld.so.conf override introduces no drift; source-built so DeepEP has one controlled header/lib root. |
| aws-ofi-nccl | commit `9c44d34` + PR#1351 head `c2e773d` | The GIN CPU-proxy plugin pins of the measured substrate (same as the TensorRT-LLM NcclEP sibling). Immutable SHAs — `refs/pull/N/head` is a moving ref. |
| gdrcopy | commit `c91ad9f` (= v2.5.2) | GIN **requires** gdrcopy compiled in (trap 4). Commit pin, not tag. |
| DeepEP | `01dc3aa` (upstream `deepseek-ai/DeepEP` main) | Carries EPv2 (ElasticBuffer + NCCL backend) and is the **exact base of draft PR #612**, so the opt-in layer applies `--check`-clean. Stock upstream — NOT a fork. |
| Megatron-LM | `19deef67` (main) | The **exact base of draft PR #4632** (ElasticBuffer in the flex dispatcher). |
| NeMo-RL | `46be4e8` | The **exact base (parent commit) of draft PR #2410** — declares `requires-python ">=3.12"`, which the py3.12 NGC base satisfies. `requirements.txt` is generated from this revision. NOT #2411's base `cc75cad` (which bumps `requires-python` to `>=3.13.13` and hard-fails `pip install -e` on this base). The measured Wave-28 evidence ran 0.5.0rc0; re-pinning to a release tag is a re-measure event. |
| GPU arch | `TORCH_CUDA_ARCH_LIST=9.0` (H100/H200) | The only measured arch. Blackwell needs an arch-list override and a re-measure. |

## The 3 draft upstream PRs (opt-in layer, default OFF)

Baked only with `--build-arg APPLY_DRAFT_ROLLOUT_PATCHES=1`; commits pinned at immutable SHAs in
[`patches/apply_nemo_rl_patches.py`](patches/apply_nemo_rl_patches.py), applied fail-loud,
self-neutralizing once merged upstream. **The baseline image has zero dependence on them.**
PR states below are as of 2026-08-25 — check them before relying on this table.

| PR | State | What it carries |
|---|---|---|
| [deepseek-ai/DeepEP#612](https://github.com/deepseek-ai/DeepEP/pull/612) | open | EFA awareness: auto-QP capped at `EP_EFA_MAX_QPS` (aws-ofi-nccl's GIN request ring is 128 slots; upstream's auto formula overruns it → `CUDA_ERROR_LAUNCH_FAILED` at first dispatch), `get_rdma_gbs()` EFA fast path (SM auto-sizing), dispatch scaleout-interval tuning. |
| [NVIDIA/Megatron-LM#4632](https://github.com/NVIDIA/Megatron-LM/pull/4632) | open | DeepEP **V2 ElasticBuffer** support in the MoE flex dispatcher (`fused_a2a.py`) — without it, `--moe-enable-deepep` binds the V1 NVSHMEM `Buffer`, which this NCCL-GIN image intentionally does not build. |
| [NVIDIA-NeMo/RL#2410](https://github.com/NVIDIA-NeMo/RL/pull/2410) | draft, closed unmerged | `LD_LIBRARY_PATH` re-export for OFI plugin discovery in NeMo-RL's own containers, plus the worked 2-node EFA GRPO recipe config (`examples/configs/recipes/llm/aws-efa-grpo-qwen3-30ba3b-2n8g-megatron.yaml`) the full rollout path uses. Applied at its parent commit `46be4e8` (= `NEMO_RL_SHA`), so it lands `--check`-clean. |

> **Not applied: [NVIDIA-NeMo/RL#2411](https://github.com/NVIDIA-NeMo/RL/pull/2411)** (deep_ep pin bump). Its base is `cc75cad` — 116 commits ahead of `46be4e8`, across a `requires-python` bump to `>=3.13.13` — so it neither applies to this tree nor belongs on this py3.12 base, and it is metadata-only anyway (deep_ep is built from `/opt/DeepEP`, not NeMo-RL's pin). Retained here as a note so the pin history is auditable.

## The integration traps

1. **NeMo-RL's pyproject pins torch exactly** (`torch==2.9.0` at the pinned SHA). Letting pip
   resolve it would *downgrade* the NGC-baked torch 2.11 and orphan the baked
   TransformerEngine/apex/flash-attn ABI. The Dockerfile installs NeMo-RL `--no-deps` and carries
   the rest of its dependency set in [`requirements.txt`](requirements.txt) (re-diff on every
   `NEMO_RL_SHA` bump).
2. **Upstream DeepEP's SM/QP auto-sizers are EFA-blind** at the pinned SHA: auto-QP
   (`num_sms*16+1`) overruns aws-ofi-nccl's 128-slot GIN request ring (hard assert surfaced as
   `CUDA_ERROR_LAUNCH_FAILED` at the *first* dispatch), and `get_rdma_gbs()` reads 0 on EFA.
   The probe passes `num_allocated_qps`/`num_sms`/`num_qps` **explicitly** (`EP_NUM_QPS=2` is the
   value the #612 evidence validated on p5en) so the baseline image stays probeable; the opt-in
   layer (#612) fixes the auto-sizers.
3. **Two NCCLs in one image — the resolved path matters.** The NGC base bakes its own libnccl in a
   different directory; if it wins the loader search, you silently run a non-GIN-verified copy.
   The image ranks the source build first via `/etc/ld.so.conf.d/00-nccl-gin.conf`, and
   `verify-image.sh` asserts **which path** `libnccl.so.2` resolves to *and* its version string.
4. **GIN needs gdrcopy at compile time and gdrdrv at run time.** An aws-ofi-nccl built without
   `gdrapi.h` carries "GDRCopy support not available at compile time" and GIN init fails at run
   time; the setup script asserts that string is *absent* from the built plugin. At run time,
   clusters without a gdrdrv device plugin need `privileged: true` (the unprivileged device cgroup
   blocks `open("/dev/gdrdrv")` with EPERM even after CAP_MKNOD — the manifest header documents the
   trade-off).
5. **The EFA installer's NGC auto-detect reroutes the plugin install.** On an NGC base the
   installer would install the stock `libnccl-ofi-ngc` plugin — which does not carry GIN — and two
   plugins on the loader path is a which-one-won guessing game. The Dockerfile passes
   `--disable-ngc --disable-build-ngc` and builds the GIN plugin from source.
6. **`flex` on an unpatched image is refused, not degraded.** Megatron's flex dispatcher with
   `moe_enable_deepep` needs #4632; on the baseline image `train-step.sh` exits with the reason
   instead of failing later inside megatron — and it never silently falls back to a different
   dispatcher than the one you asked for.
7. **Opt-in flags use `"${VAR:-default}"` in `env_vars.example`** so a value pre-set on the command
   line (`APPLY_DRAFT_ROLLOUT_PATCHES=1 docker build ...`) survives the file being sourced after
   it. A hardcoded `export VAR="0"` silently clobbers the build you thought you asked for.
8. **CPU-proxy means CPU is on the data path.** `NCCL_GIN_TYPE=2` runs GIN's proxy threads on the
   host cores; the manifest pins requests == limits (Guaranteed QoS) so CFS throttling under node
   pressure cannot silently degrade the transport.

## Runtime requirements (baked in the image ENV, repeated in the manifest and scripts)

| Env | Why |
|---|---|
| `NCCL_GIN_TYPE=2`, `NCCL_GIN_ENABLE=1` | GIN CPU-proxy — the EFA-viable GIN mode |
| `OFI_NCCL_GIN_GDAKI=0` | GPU-initiated GIN is not the shipped path on EFA |
| `OFI_NCCL_GIN_MAX_REQUESTS=512` | GIN request-ring depth of the measured substrate |
| `FI_PROVIDER=efa`, `FI_EFA_USE_DEVICE_RDMA=1` | EFA with GPU-direct RDMA |
| `FI_EFA_ENABLE_SHM_TRANSFER=0`, `FI_EFA_FORK_SAFE=1` | no SHM shortcut; fork-safe for the proxy |
| `NCCL_NET_PLUGIN=/opt/aws-ofi-nccl/lib/libnccl-net-ofi.so` | the GIN-capable plugin, explicitly |
| `NCCL_NVLS_ENABLE=0` | prevents NVLS init failures on H100/H200 |
| `DEEP_EP_USE_V2_SHIM=0` | V2-native path, no compatibility shim |
| `EP_EFA_MAX_QPS=2`, `EP_EFA_RDMA_GBS=25.0` | read only by the **patched** deep_ep (#612); inert on baseline |

## Hardware requirements

2× `p5.48xlarge` (8× H100, 32 EFA NICs/node — the measured topology) or `p5en.48xlarge`
(8× H200, 16 EFA NICs/node; set `EFA_PER_NODE=16`). One CPU node for the Ray head. FSx for
Lustre PVC (full GRPO path only — the recipe gates need no shared storage). 2Mi hugepages
pre-allocated on the GPU nodes (`hugepages-2Mi: 5120Mi` per pod, or the pods sit Pending).

## Prerequisites

- An EKS / SageMaker HyperPod EKS cluster with the EFA device plugin, the NVIDIA device plugin,
  and the [KubeRay operator](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started.html).
- Docker + network access to `nvcr.io`, `pypi.org`, `github.com`, `efa-installer.amazonaws.com`.
- An image registry you own (ECR); **do not point at anyone's private registry**.

## Quick Start

### 1. Configure environment variables

```bash
cp env_vars.example env_vars   # env_vars is gitignored
vim env_vars                   # REGISTRY/IMAGE/TAG, NAMESPACE, FSX_CLAIM, EFA_PER_NODE, ...
source env_vars
```

### 2. Build and push (baseline, and optionally the draft-PR flavor)

```bash
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${REGISTRY}
aws ecr create-repository --repository-name ${IMAGE} --region ${AWS_REGION} || true

# baseline (upstream-only)
docker build -f nemo-rl.Dockerfile -t ${FULL_IMAGE} .
# opt-in flavor with the 3 draft PRs baked (use a DISTINCT tag — never overwrite the baseline)
docker build -f nemo-rl.Dockerfile --build-arg APPLY_DRAFT_ROLLOUT_PATCHES=1 \
  -t ${FULL_IMAGE}-draftprs .

docker push ${FULL_IMAGE}
```

### 3. Gate the image before any cluster deploy

```bash
recipe/verify-image.sh ${FULL_IMAGE}
# ... ALL CHECKS PASS
```

### 4. Deploy the Ray cluster

```bash
kubectl create namespace ${NAMESPACE} || true
kubectl create secret generic hf-token --from-literal=HF_TOKEN=${HF_TOKEN} -n ${NAMESPACE}
envsubst < kubernetes/raycluster.yaml | kubectl apply -f -
kubectl -n ${NAMESPACE} get pods -w   # 1 head + ${NUM_NODES} workers
```

### 5. Cross-node rollout probe (minutes, no weights)

Drives the real `deep_ep.ElasticBuffer` dispatch/combine across the node boundary against a
closed-form oracle, and asserts the EFA hardware TX counters advanced:

```bash
W0=$(kubectl -n ${NAMESPACE} get pod -l ray.io/node-type=worker -o jsonpath='{.items[0].metadata.name}')
W1=$(kubectl -n ${NAMESPACE} get pod -l ray.io/node-type=worker -o jsonpath='{.items[1].metadata.name}')
W0_IP=$(kubectl -n ${NAMESPACE} get pod ${W0} -o jsonpath='{.status.podIP}')
kubectl -n ${NAMESPACE} exec ${W1} -c ray-worker -- bash -lc \
  "nohup /opt/run-rollout-probe.sh worker ${W0_IP} 1 > /tmp/probe.log 2>&1 &"
kubectl -n ${NAMESPACE} exec ${W0} -c ray-worker -- /opt/run-rollout-probe.sh leader ${W0_IP}
# ... ROLLOUT-PROBE PASS — ElasticBuffer dispatch/combine over EFA verified (tx +NNNB)
```

### 6. N-step MoE training gate

Baseline image → the stock `alltoall` dispatcher (no deep_ep on the data path, still NCCL over
EFA); patched image → `MOE_DISPATCHER=flex` for the DeepEP V2 ElasticBuffer path:

```bash
kubectl -n ${NAMESPACE} exec ${W1} -c ray-worker -- bash -lc \
  "nohup /opt/train-step.sh worker ${W0_IP} 1 > /tmp/train.log 2>&1 &"
kubectl -n ${NAMESPACE} exec ${W0} -c ray-worker -- /opt/train-step.sh leader ${W0_IP}
# ... TRAIN-STEP-PASS dispatcher=alltoall world=16 ep=16
# on the -draftprs image:
#   MOE_DISPATCHER=flex /opt/train-step.sh leader ${W0_IP}
```

### 7. Full GRPO run (STAGED — draft-PR image only, not re-measured from this folder)

The patched image carries the worked 2-node EFA GRPO recipe config from NeMo-RL#2410 at
`/opt/NeMo-RL/examples/configs/recipes/llm/aws-efa-grpo-qwen3-30ba3b-2n8g-megatron.yaml`
(Qwen3-30B-A3B, Megatron backend, `moe_token_dispatcher_type=flex`, `moe_enable_deepep=true`).
Stage the model with `kubernetes/data-prep-pod.yaml`, then launch per
[NeMo-RL's GRPO docs](https://github.com/NVIDIA-NeMo/RL) from the Ray head with that config.
Treat results as your own measurement — this folder publishes none for this path.

## Known limitations (honest list)

- **Build-staged, not cluster-re-run.** The NGC-from-scratch assembly here reproduces the measured
  Wave-28 mechanism chain from public sources, but this exact image has not itself been re-run on
  a cluster. The recipe gates exist so you (or we, next capacity window) can re-verify cheaply.
- **The full rollout path is draft-PR-dependent.** Three upstream PRs, closed-unmerged upstream
  (see the table). If upstream supersedes them, the patch layer fails loud or self-neutralizes —
  either way the image never ships an ambiguous patch state.
- **Baseline DeepEP gates use explicit SM/QP counts** (trap 2). A probe pass with explicit counts
  does not certify upstream's auto-sizing on EFA — that certification is exactly PR #612.
- **NCCL topology XML on 32-NIC p5 nodes:** stock NCCL can hit the open issue
  [NVIDIA/nccl#2160](https://github.com/NVIDIA/nccl/issues/2160) (`NCCL_TOPO_XML_MAX_NODES=256`
  overflow during intra-node XML fusion). If NCCL init fails with a topo-XML error on
  p5.48xlarge, rebuild Layer 4 with the define raised (documented one-liner in the issue) — not
  baked here because it is a non-upstream one-off.
- **No performance numbers.** Dispatch/combine latency and GRPO throughput on this substrate are
  future work; for kernel-level EP benchmarks on the same NCCL-GIN substrate see
  [`micro-benchmarks/expert-parallelism`](../../../../micro-benchmarks/expert-parallelism).

## File structure

```
deepep-v2-efa/
├── README.md                      <- you are here
├── nemo-rl.Dockerfile             <- NGC-from-scratch image (baseline + opt-in draft-PR layer)
├── setup_nemo_rl_deepep_efa.sh    <- aws-ofi-nccl GIN + DeepEP V2 source builds (in-tree, COPY'd)
├── requirements.txt               <- NeMo-RL deps minus the NGC-baked ABI anchors
├── env_vars.example               <- copy to env_vars (gitignored), fill in, source
├── patches/
│   └── apply_nemo_rl_patches.py   <- the 3 draft PRs, pinned SHAs, fail-loud, self-neutralizing
├── recipe/
│   ├── verify-image.sh            <- static substrate gate (run before any deploy)
│   ├── run-rollout-probe.sh       <- cross-node ElasticBuffer probe + EFA TX-counter assert
│   ├── probe_rollout.py           <- the torchrun probe body (oracle-checked dispatch/combine)
│   ├── train-step.sh              <- N-step Megatron MoE training gate launcher
│   └── train_moe_step.py          <- the torchrun train body (loss finite+decreasing+agreeing)
└── kubernetes/
    ├── raycluster.yaml            <- 1 CPU head + N GPU workers (EFA, hugepages, Guaranteed QoS)
    └── data-prep-pod.yaml         <- stage model/dataset onto FSx (full GRPO path only)
```

## References

- [DeepEP](https://github.com/deepseek-ai/DeepEP) — EPv2 / ElasticBuffer (PR#605)
- [aws-ofi-nccl](https://github.com/aws/aws-ofi-nccl) — the GIN-capable NCCL network plugin
- [NeMo-RL](https://github.com/NVIDIA-NeMo/RL) and [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- Sibling test cases: [`slime`](../../slime) (RL on HyperPod EKS as a Ray cluster — this folder
  mirrors its shape), [`sglang/dsr1-deepep-efa`](../../sglang/dsr1-deepep-efa) (the NVSHMEM-path
  DeepEP serving sample)
- [`micro-benchmarks/expert-parallelism`](../../../../micro-benchmarks/expert-parallelism) —
  kernel-level EP benchmarks, including a DeepEP V2 (NCCL GIN) benchmark
