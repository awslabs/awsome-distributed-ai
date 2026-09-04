<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0 -->
# NVIDIA Dynamo serving DeepEP-V2 MoE all-to-all over EFA

Serve a Mixture-of-Experts model with **NVIDIA Dynamo** (`dynamo.frontend` OpenAI ingress +
`dynamo.vllm` engine) using **DeepEP-V2** (`ElasticBuffer`) expert-parallel all-to-all, routed over
**AWS EFA** via the NCCL-GIN **CPU-proxy** path (`NCCL_GIN_TYPE=2`). `dynamo.vllm` wraps the same vLLM
engine as the sibling `../../vllm/deepep-v2-efa` sample (it forwards every unknown CLI flag straight
into vLLM's `AsyncEngineArgs`), so **nothing about the DeepEP-V2 / EFA transport changes** — this
sample adds only an OpenAI-compatible frontend and a DP/EP-aware worker wrapper on top of that proven
substrate. It is the V2 / NCCL-GIN counterpart to the NVSHMEM-backed `../../sglang/dsr1-deepep-efa`
sample: no NVSHMEM, no IBGDA — DeepEP-V2's `ElasticBuffer` drives the dispatch/combine collectives
over `aws-ofi-nccl`'s GIN plugin on `efa-direct`.

Validated on **2× and 4× p5en.48xlarge (H200)**, `Qwen/Qwen3-30B-A3B-FP8`: **measured at DP16/EP16**
(the concurrency sweeps in `benchmarks/`); **DP32/EP32 functionally validated** (16/16 HTTP 200
bring-up, no measured sweep — the shipped manifest is the 2-node/EP16 shape).

## Relationship to the vLLM sample

This sample is the vLLM-DeepEP-V2 substrate (`../../vllm/deepep-v2-efa`) **plus a Dynamo serving
front**. Everything below the serving layer is shared and byte-identical:

| Shared with `../../vllm/deepep-v2-efa` (identical) | Dynamo-specific (the only delta) |
|---|---|
| `Dockerfile` Layers 1–5b (NGC base, EFA, torch cu13, gdrcopy, aws-ofi-nccl GIN + #1351, DeepEP-V2, pinned vLLM wheel) | `Dockerfile` Layer 5c: `pip install --no-deps ai-dynamo{,-runtime}==1.3.1` |
| `setup_deepep_v2_efa.sh`, `recipe/build_deepep.sh`, `recipe/run-kernel-test.sh`, `recipe/verify-image.sh`, `recipe/benchmark*.{sh,py}` | `recipe/serve.sh`: launches `dynamo.frontend` + `dynamo.vllm` (vs `vllm serve`) |
| The proxy-Gin/EFA env contract, the DP-coordinator flags, the model, the EP-divisibility preflight | `kubernetes/dynamo-deepep-v2-2node.yaml`: readiness probe + names |

`dynamo.vllm` uses `parse_known_args`, so the DeepEP-V2 backend selection (`--all2all-backend
deepep_v2`), the DP-coordinator wiring (`--data-parallel-*`), and the eager/revision flags all pass
through into vLLM unchanged — the `COMMON` flag block in `recipe/serve.sh` is identical to the vLLM
sample's. If you only need a raw OpenAI endpoint with no Dynamo router/planner, the vLLM sample is the
simpler choice; use this one when you want Dynamo's frontend (and a path to its KV-router / planner)
in front of the same DeepEP-V2/EFA engine.

## How DeepEP-V2 gets onto EFA

DeepEP's default transport is NVSHMEM/IBGDA, which EFA does not provide. The V2 (`ElasticBuffer`) path
instead runs its dispatch/combine over `aws-ofi-nccl`'s **GIN CPU-proxy** (`NCCL_GIN_TYPE=2`,
`OFI_NCCL_GIN_GDAKI=0`) on the `efa-direct` fabric. Three things make this work, and two of them are
non-obvious integration fixes, not config:

### Integration fixes baked into this sample

1. **`EP_REUSE_NCCL_COMM=0` (required, or serve init segfaults).** Upstream DeepEP flipped this default
   to `1` (reuse torch's NCCL comm). Under vLLM, torch creates NCCL comms lazily and has run no
   collective on the EP group before `ElasticBuffer` construction, so `_comm_ptr()` returns `0` →
   `ncclTeamWorld(nullptr)` → deterministic segfault on all ranks. Setting `0` restores DeepEP's
   create-own-comm path. (Env, set in `recipe/serve.sh` and `kubernetes/`.)
2. **The gdrcopy forced-PCIe capability** for the GIN plugin on gdrdrv-2.4 hosts, via
   `OFI_NCCL_GDRCOPY_FORCED_PCIE_COPY=1` — the parameterized fix from
   [aws/aws-ofi-nccl#1351](https://github.com/aws/aws-ofi-nccl/pull/1351), cherry-picked at a pinned SHA
   in `setup_deepep_v2_efa.sh` (no local patch file).
3. **DeepEP-V2 source** = `b306af06` + [PR#612](https://github.com/deepseek-ai/DeepEP/pull/612) (EFA
   auto-QP cap), pinned to the PR's **immutable head SHA** (a bare `refs/pull/N/head` is a moving ref).

### eager vs non-eager (see `benchmarks/`)

| Mode | Status | How |
|---|---|---|
| `--enforce-eager` | **Serves, zero extra patches — the shipped, measured, supported path** | the default this sample ships (`SERVE_ENFORCE_EAGER=1`) |
| default compilation (CUDA graphs) | **NOT supported at the shipped pin** | needs the empty-`ExpertTokensMetadata` guard ([vLLM #52632](https://github.com/vllm-project/vllm/pull/52632), merged 2026-08-20), which is **only on the vLLM 0.26 line** — and 0.26 separately regresses the `deepep_v2` combine on this DeepEP/EFA substrate (see below), so we cannot pin forward to reach it. The historical non-eager `benchmarks/` tables were taken on this pin with #52632 applied as an **unmerged cherry-pick**. |

The shipped vLLM pin is `e2f993dc4` — the merge commit of [PR#41183](https://github.com/vllm-project/vllm/pull/41183),
the first (and, on this EFA/DeepEP substrate, only measured-working) `deepep_v2` backend, `0.22.1rc1.dev283`.
At this pin, default (non-eager) compilation crashes deterministically ~48 s into startup in `profile_run`
(`deepep_v2.py` combine); `--enforce-eager` avoids it and is the path this sample ships and supports.

> **Why the pin is not bumped forward to pick up #52632:** vLLM 0.26 (the line that carries #52632)
> regresses the `deepep_v2` combine on this exact DeepEP `_C.so` / EFA substrate — a DP16/EP16 serve
> faults `CUDA_ERROR_LAUNCH_FAILED (719)` in `profile_run` → `determine_available_memory` during
> KV-cache sizing (measured 2026-09-04, eager **and** non-eager), while the standalone cross-node
> kernel-test still passes on the same image. So the fault is 0.26's DeepEP-V2 driver, not the kernels
> or EFA. `e2f993dc4` (0.22) is the newest vLLM proven to serve `deepep_v2` DP16/EP16 over EFA; the pin
> stays there until a newer vLLM line is re-measured green on this substrate.

<!-- MD028: separate the two blockquotes so the blank line is not read as inside one quote -->

> **Shared-experts caveat when bumping the pin:** at `e2f993dc4` there is a second, independent
> non-eager crash cause that bites models with shared experts (DeepSeek-V3/R1, DeepSeek-V2-Lite —
> models `serve.sh` explicitly supports): the `-1` sentinel expert IDs that vLLM
> [#46432](https://github.com/vllm-project/vllm/pull/46432) legitimately produces reach
> `moe_align_sum_kernels.cu`, which guards `expert_id >= num_experts` but not `expert_id < 0`
> (out-of-bounds atomic write). Upstream fixed it in
> [#47785](https://github.com/vllm-project/vllm/pull/47785) (merged 2026-07-10). Qwen3-30B-A3B has
> no shared experts, which is why the sweeps here were green without it. Any pin past
> [#52632](https://github.com/vllm-project/vllm/pull/52632)'s merge also includes #47785 (it merged
> earlier), so the "bump the pin past #52632" guidance below picks up both fixes.

## Prerequisites

- An EKS cluster of p5en.48xlarge (H200) with EFA + the EFA K8s device plugin (the shipped launcher);
  the container also runs under raw `docker run` on any 2 EFA hosts if you wire the rendezvous by hand.
- An ECR repo you own (set in `setup/env_vars`); this sample never hardcodes a registry.
- Hugging Face access for the model (`Qwen/Qwen3-30B-A3B-FP8` is public, no token required).

## Build

```bash
cp setup/env_vars.example setup/env_vars && $EDITOR setup/env_vars   # set REGISTRY, IMAGE_TAG
bash setup/build-push.sh
```

The image is NGC-from-scratch (`FROM nvcr.io/nvidia/cuda:...`). `setup_deepep_v2_efa.sh` builds
aws-ofi-nccl (GIN + the #1351 param) and stages DeepEP-V2 source; the `_C.so` is compiled in-pod on
first boot (needs a live CUDA context) by `recipe/`-invoked `build_deepep.sh`. Dynamo is added in
Layer 5c as `pip install --no-deps ai-dynamo{,-runtime}==1.3.1` — `--no-deps` is load-bearing: without
it, pip would pull `ai-dynamo`'s `vllm[...]==0.23.0` dependency and overwrite the pinned `VLLM_SHA`
build (and drag `nvidia-nccl-cu13` back to torch's 2.28.x, undoing the ABI re-pin). The in-tree
`Dockerfile` is the canonical, reviewable build. The shipped vLLM pin (`e2f993dc4`) is the **exact**
substrate the `benchmarks/` tables were measured on, so a rebuild reproduces them — see
`benchmarks/README.md`.

The one image name used everywhere is `${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}` from
`setup/env_vars` (`build-push.sh` builds and pushes exactly that; point the manifest's `image:` at
the same).

## Smoke-test the EFA transport before loading the model

**1. Static image check (single node, no rendezvous):**

```bash
source setup/env_vars && bash recipe/verify-image.sh "${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
```

Asserts (fail-loud) the efa provider resolves (`fi_info` — the live `efa-direct` fabric check runs
when `/dev/infiniband` is present on the host), a single pinned libnccl wins **with the GIN/LSA
symbols present**, the GIN plugin exports `ncclGinPlugin`, and the DeepEP-V2 source (ElasticBuffer +
the kernel-smoke test) is staged at `/opt/DeepEP`. The `import deep_ep` assertion happens in-pod
after `build_deepep.sh`, not here — the `_C.so` is deliberately not baked into the image.

**2. Cross-node kernel smoke (prove bytes move over EFA before the model load):**

```bash
# in the pods/containers, one per node — runs DeepEP-V2's own elastic EP test
bash /opt/run-kernel-test.sh leader <leader-ip>            # node 0
bash /opt/run-kernel-test.sh worker <leader-ip> 1          # node 1
```

Runs `DeepEP/tests/elastic/test_ep.py` across the nodes with the exact proxy-Gin/EFA env the serve
uses, and only prints `KERNEL-TEST PASS` when the test passes **and** the `NCCL_DEBUG=INFO` log shows
the `efa-direct` banner (so a green result cannot be a silent TCP/SHM fallback). This is the one step
that cannot hang for hours — run it before committing a node to the multi-hundred-GB weight load.

For a standalone DeepEP-V2 dispatch/combine benchmark of the fabric itself (numbers, not just
pass/fail — and without any vLLM in the loop), use the repo's runnable V2 micro-benchmark:
[`micro-benchmarks/expert-parallelism/deepep-v2-benchmark/`](../../../../micro-benchmarks/expert-parallelism/deepep-v2-benchmark/)
(own image + Slurm launchers; same NCCL-GIN/EFA transport this sample serves over).

## Serve

Dynamo splits the serving front from the engine, so the launcher is **one invocation per node** with a
role (see the header of `recipe/serve.sh` for the topology):

```bash
# eager (default). SERVE_DP = total data-parallel = EP size; SERVE_DP_LOCAL = GPUs/node.
# node 0 (leader): dynamo.frontend on :8000 (OpenAI HTTP) + dynamo.vllm (non-headless) DP ranks 0..7
SERVE_DP=16 bash recipe/serve.sh leader <leader-ip>
# node 1 (worker): dynamo.vllm --headless, DP start-rank 8 (= ORDINAL × SERVE_DP_LOCAL)
SERVE_DP=16 bash recipe/serve.sh worker <leader-ip> 8
```

Only the leader runs the frontend, so the OpenAI API is on **the leader pod's** `:8000`
(`dynamo-deepep-v2-0`), not on the workers. Discovery is the **file backend**
(`--discovery-backend file`), which is node-0-local (frontend ↔ leader engine only); the worker is
`--headless` and joins the DP group purely through vLLM's native data-parallel RPC to the leader over
EFA, so there is **no etcd/NATS and no shared discovery volume** to provision.

Serve **eager** — it is the shipped default (`SERVE_ENFORCE_EAGER=1`) and the only supported mode at
the shipped pin. Default (CUDA-graph) compilation crashes at this pin (`e2f993dc4`) in `profile_run`;
the guard that fixes it ([vLLM #52632](https://github.com/vllm-project/vllm/pull/52632)) is only on the
vLLM 0.26 line, which separately regresses the `deepep_v2` combine here (see "eager vs non-eager"), so
`SERVE_ENFORCE_EAGER=0` is not a supported path in this sample — the knob still exists but warns and
will crash. Kubernetes: `kubectl apply -f kubernetes/` (2-node StatefulSet + headless service; the
proxy-Gin env contract + EFA device requests are set there).

### Readiness on Dynamo — why `/health` alone lies

The Kubernetes leader probe requires `/health` to be 200 **AND** carry a non-empty `endpoints` array —
not a bare `curl -sf /health`. This is deliberate and Dynamo-specific:

- `dynamo.frontend`'s `/health` returns **200 as soon as the HTTP server binds** (~6 s). Its
  `ServiceObserver` defaults to `Ready` and only leaves `Ready` on shutdown — **nothing flips it to
  Ready on engine registration**. So a bare `/health` check reports "healthy" with an empty
  `instances: []`, *hours* before the multi-hundred-GB model finishes loading — which would defeat the
  `startupProbe`'s 60-minute compile+load budget.
- The engine registers with the frontend **only after the weight load** (`dynamo.vllm`'s
  `register_model()` runs after `AsyncLLM.from_vllm_config`). Once it does, `/health`'s response body
  populates its `endpoints`/`instances` arrays.
- Therefore the honest readiness signal is **`/health` is 200 *and* has ≥1 endpoint**. The probe uses
  `curl -sf .../health | grep -qF '"endpoints":["'`, which matches only a populated array (an empty
  `"endpoints":[]` fails the match).

Workers are `--headless` (no HTTP server, and they never register a discovery endpoint), so their
probe is **process-liveness only** (`pgrep -f dynamo.vllm`). A worker's startup passes on its first
tick and a wedged-but-alive worker stays Ready — a known limitation, not a readiness check. A worker's
true progress is observed through the leader coming Ready: the DP rendezvous cannot complete (and the
leader cannot register its engine) until every worker has joined. This pairs with
`publishNotReadyAddresses: true` on the headless Service — workers must resolve the leader's DNS
A-record *before* the leader is Ready, or the rendezvous deadlocks.

## Benchmark

```bash
# from inside the leader pod (both scripts are baked into the image at /opt):
kubectl -n dynamo-deepep exec dynamo-deepep-v2-0 -- env OUT_ROOT=/work/benchmarks bash /opt/benchmark.sh 127.0.0.1

# or from outside the cluster, via a port-forward to the leader pod:
kubectl -n dynamo-deepep port-forward pod/dynamo-deepep-v2-0 8000:8000 &
bash recipe/benchmark.sh 127.0.0.1
```

Concurrency sweep 1/8/16/32/64 (5× requests per level), writes `benchmarks/raw/`. Exits non-zero
unless **every** request at **every** level succeeded. See `benchmarks/README.md` for the measured
eager and non-eager tables + environment provenance.

## Known limitations

- Measured on **H200 (p5en, `sm_90`) only**; no Blackwell serving run is in this sample. The manifest's
  `DEEPEP_ARCH_LIST=10.0` (p6-b200) / `10.3` (p6-b300) knobs are **documented but not verified** at the
  shipped DeepEP pin: on 2× p6-b300 the DeepEP runtime JIT produced no loadable kernel (a Blackwell PTX
  codegen failure at CUDA 13.0 on the `deepseek-ai/DeepEP@b306af06` lineage). The
  [`amazon-contributing/DeepEP`](https://github.com/amazon-contributing/DeepEP) fork the canonical
  benchmark pins carries the `st.bulk` 64-bit-operand fix
  ([#3](https://github.com/amazon-contributing/DeepEP/pull/3)) that makes CUDA 13.0 work on Blackwell;
  moving this sample's `DEEPEP_SHA`/`DEEPEP_REPO` to that fork is the intended path to enabling those
  rows, and should be re-verified on Blackwell before the knobs are advertised as working.
- The `benchmarks/` numbers are an **at-scale throughput + relative-latency** datapoint (fixed 128-token
  greedy decode, single sweep per mode), **not** a tuned per-token-latency (TTFT) baseline.
- **Only eager (`--enforce-eager`) serving is supported at the shipped pin.** Default-compilation
  (non-eager) serving needs the empty-`ExpertTokensMetadata` guard
  ([vLLM #52632](https://github.com/vllm-project/vllm/pull/52632), merged 2026-08-20), which is only on
  the vLLM 0.26 line — and 0.26 separately regresses the `deepep_v2` combine on this DeepEP/EFA substrate
  (`CUDA_ERROR_LAUNCH_FAILED (719)` in `profile_run`, measured 2026-09-04), so the pin cannot be moved
  forward to reach the guard. This sample therefore carries **no build-time patches** and ships eager.
  The non-eager numbers in `benchmarks/` were measured on this same pin (`e2f993dc4`) with #52632 applied
  as an unmerged cherry-pick; they are **historical** — a stock rebuild of this sample serves eager only.
- Only a **Kubernetes** launcher is shipped and exercised (`kubernetes/`). No Slurm/Pyxis `.sbatch` is
  provided because none was run; the raw two-node `recipe/serve.sh` path is the manual fallback.
- The Dynamo front is `dynamo.frontend` + `dynamo.vllm` with the **file** discovery backend and vLLM's
  native DP coordinator for cross-node fan-out. The Dynamo **KV router / planner / disaggregated
  prefill-decode** paths are **not** exercised here — this sample proves DeepEP-V2 MoE all-to-all over
  EFA under a Dynamo front, not those higher-level Dynamo features.
- `setup_deepep_v2_efa.sh` is a **documented variant** of the repo's canonical V2/GIN provisioner,
  [`micro-benchmarks/expert-parallelism/deepep-v2-benchmark/setup_deepep_gin.sh`](../../../../micro-benchmarks/expert-parallelism/deepep-v2-benchmark/setup_deepep_gin.sh)
  (which appeared 2026-08-24). When the canonical moves, that is the file to track. Three deliberate
  divergences justify a separate script here; the next reader should know they are choices, not drift:
  1. **The unmerged aws-ofi-nccl #1351 parameter.** This sample cherry-picks
     [aws/aws-ofi-nccl#1351](https://github.com/aws/aws-ofi-nccl/pull/1351)
     (`OFI_NCCL_GDRCOPY_FORCED_PCIE_COPY`) at a pinned head SHA; the canonical builds a stock GIN NCCL
     and does not carry it. Until #1351 merges, this recipe cannot be a thin call into the canonical.
  2. **CPU-proxy (`NCCL_GIN_TYPE=2`), not EFA-GDA.** This is the GDAKI-off, CPU-proxy transport that is
     viable on EFA today; the canonical benchmark's defaults and NCCL build target a different point in
     that design space.
  3. **Coupling to the vLLM wheel's torch/NCCL ABI.** The DeepEP `_C.so` here is built in-pod against
     the exact `torch 2.11+cu130` / `nvidia-nccl-cu13 2.30.4` the pinned vLLM wheel drags in (Dockerfile
     Layer 5b re-pins it), so the toolchain is wheel-driven rather than a standalone NCCL build tree.

  The **DeepEP source** is the one divergence with a concrete cost, called out at
  `setup_deepep_v2_efa.sh` (see the header note there): the canonical pins the
  [`amazon-contributing/DeepEP`](https://github.com/amazon-contributing/DeepEP) fork, which carries the
  Blackwell `st.bulk` 64-bit-operand fix ([amazon-contributing/DeepEP#3](https://github.com/amazon-contributing/DeepEP/pull/3),
  merged 2026-08-24); this sample pins `deepseek-ai/DeepEP@b306af06`+PR#612, which does not. The
  `benchmarks/` numbers were measured on H200 (`sm_90`), where this does not bite — see the Blackwell
  caveat under **Known limitations** before using the `DEEPEP_ARCH_LIST=10.x` knob.
- `setup_deepep_v2_efa.sh` is deliberately **outside** `.github/workflows/deepep-vendor-sync.yml`. That
  CI gates only the NVSHMEM `setup_deepep_efa.sh` vendored copy (canonical at
  `micro-benchmarks/expert-parallelism/deepep-benchmark/`) — a different script — so this V2/GIN variant
  is correctly not in that workflow. Do not add it to that workflow.
