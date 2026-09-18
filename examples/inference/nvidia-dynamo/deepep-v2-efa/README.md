<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0 -->
# NVIDIA Dynamo serving DeepEP-V2 MoE all-to-all over EFA

Serve a Mixture-of-Experts model with **NVIDIA Dynamo** (`dynamo.frontend` OpenAI ingress +
`dynamo.vllm` engine) using **DeepEP-V2** (`ElasticBuffer`) expert-parallel all-to-all, routed over
**AWS EFA** via the NCCL-GIN **CPU-proxy** path (`NCCL_GIN_TYPE=2`). `dynamo.vllm` wraps the same vLLM
engine as the sibling **vLLM DeepEP-V2 sample**
([PR #1230](https://github.com/awslabs/awsome-distributed-ai/pull/1230) — the `../../vllm/deepep-v2-efa`
relative links throughout this sample resolve once it merges) (it forwards every unknown CLI flag
straight into vLLM's `AsyncEngineArgs`), so **nothing about the DeepEP-V2 / EFA transport changes** — this
sample adds only an OpenAI-compatible frontend and a DP/EP-aware worker wrapper on top of that proven
substrate. It is the V2 / NCCL-GIN counterpart to the NVSHMEM-backed `../../sglang/dsr1-deepep-efa`
sample: no NVSHMEM, no IBGDA — DeepEP-V2's `ElasticBuffer` drives the dispatch/combine collectives
over `aws-ofi-nccl`'s GIN plugin on `efa-direct`.

Validated on **2× and 4× p5en.48xlarge (H200)**, `Qwen/Qwen3-30B-A3B-FP8`: **measured at DP16/EP16**
(the concurrency sweeps in `benchmarks/` — measured on the sibling vLLM sample's image, not this
Dynamo image; see `benchmarks/README.md`); **DP32/EP32 functionally validated** (16/16 HTTP 200
bring-up, no measured sweep — the shipped manifest is the 2-node/EP16 shape).

## Relationship to the vLLM sample

This sample is the vLLM-DeepEP-V2 substrate (`../../vllm/deepep-v2-efa`,
[PR #1230](https://github.com/awslabs/awsome-distributed-ai/pull/1230)) **plus a Dynamo serving
front**. Everything below the serving layer is shared by design — same layers, same scripts, same
env contract. (This sample's substrate pins may lead the sibling's while the two reviews converge;
the recipes re-align as the sibling's review lands.) **Merge order: #1230 lands first** — this
sample's relative sibling links and its byte-identity claims point at that folder, so until #1230
merges they have nothing on `main` to resolve against.

| Shared with `../../vllm/deepep-v2-efa` (identical) | Dynamo-specific (the only delta) |
|---|---|
| `Dockerfile` Layers 1–5b (NGC base, EFA, torch cu13, gdrcopy, aws-ofi-nccl GIN v1.21.1, DeepEP-V2, pinned vLLM wheel) | `Dockerfile` Layer 5c: `pip install --no-deps ai-dynamo{,-runtime}==1.3.1` |
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
2. **gdrcopy compiled into the GIN plugin, with gdrdrv ≥ 2.5 on the host.** aws-ofi-nccl is built
   from source at the released `v1.21.1` tag with `--with-gdrcopy`, and the build asserts gdrcopy
   support landed (a gdrapi-less plugin fails `nccl_ofi_gin_init` at serve). The release attempts
   the forced-PCIe pin path by default and falls back on failure, so no
   `OFI_NCCL_GDRCOPY_FORCED_PCIE_COPY` override is needed; the gdrdrv **kernel module ≥ 2.5** is a
   host prerequisite instead (see Prerequisites).
3. **DeepEP-V2 source** = the
   [`amazon-contributing/DeepEP`](https://github.com/amazon-contributing/DeepEP) fork at a pinned SHA
   (`97d8f9bc`) — the same source the repo's canonical V2/GIN provisioner pins ("the benchmark
   supports no other source"). The fork carries the in-tree successors of deepseek
   [PR#612](https://github.com/deepseek-ai/DeepEP/pull/612)'s EFA work: the QP count clamps from `_C`
   runtime constants and the RDMA link rate is probed from sysfs, so no `EP_EFA_MAX_QPS` /
   `EP_EFA_RDMA_GBS` env exists (or is set) in this sample.

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
- The **`gdrdrv` kernel module ≥ 2.5 loaded on the host** (`cat /sys/module/gdrdrv/version`;
  `/dev/gdrdrv` must exist). GIN needs GDRCopy at run time; the manifest's `privileged: true` lets the
  container open the host's `/dev/gdrdrv`, but privileged cannot conjure the device node if the module
  was never loaded — and the shipped plugin (released `v1.21.1`) carries no gdrdrv-2.4 workaround. The
  AWS GPU AMIs ship it; if absent, `sudo modprobe gdrdrv` (gdrcopy ≥ 2.5, matching the image's
  `c91ad9f`/v2.5.2 userspace build).
- **2Mi hugepages pre-allocated on the compute nodes.** The manifest requests `hugepages-2Mi: 5120Mi`
  per pod (= 2560 × 2Mi pages); the consumer is libfabric's EFA provider bounce-buffer pools.
  Kubernetes only *accounts* hugepages — allocation is a node-bootstrap step (kernel cmdline
  `hugepages=2560`, or `sysctl vm.nr_hugepages=2560` via a privileged DaemonSet, before the kubelet
  starts). Without it the pods sit **Pending** with an unsatisfiable `hugepages-2Mi` request. The
  nodes the measured runs used had them pre-allocated.
- **Room for the weights where `emptyDir` actually lands.** The manifest's `work` volume
  (`emptyDir: { sizeLimit: 900Gi }`) is backed by the filesystem holding `/var/lib/kubelet` — on a
  stock EKS AMI that is the **EBS root volume**, not the p5en instance-store NVMe. A multi-hundred-GB
  model download there fills the root disk and triggers a **node-wide `DiskPressure` eviction**, not
  a pod-level failure. Size the root volume for your model, or remap `/var/lib/kubelet` (or the
  volume) onto the instance store in the node bootstrap.
- **`cpuManagerPolicy: static` on the kubelet** if you want what the manifest's Guaranteed QoS
  (requests == limits) is there for: exclusive core pinning for the NCCL-GIN **CPU-proxy** threads,
  which sit on the network data path. Without the static policy, Guaranteed QoS does **not** exempt
  the pod from CFS throttling at its `cpu` limit (see the note in the manifest).
- An ECR repo you own (set in `setup/env_vars`); this sample never hardcodes a registry.
- Hugging Face access for the model (`Qwen/Qwen3-30B-A3B-FP8` is public, no token required).

## Build

```bash
cp setup/env_vars.example setup/env_vars && $EDITOR setup/env_vars   # set REGISTRY, IMAGE_TAG
bash setup/build-push.sh
```

The image is NGC-from-scratch (`FROM nvcr.io/nvidia/cuda:...`). `setup_deepep_v2_efa.sh` builds
aws-ofi-nccl (GIN, released `v1.21.1`) and stages the DeepEP-V2 source; the `_C.so` is compiled in-pod on
first boot (needs a live CUDA context) by `recipe/`-invoked `build_deepep.sh`. Dynamo is added in
Layer 5c as `pip install --no-deps ai-dynamo{,-runtime}==1.3.1` — `--no-deps` is load-bearing: it
keeps pip from re-resolving `ai-dynamo`'s nine unconditional dependencies (`transformers`,
`prometheus-client`, `msgspec`, `pyzmq`, …) over the versions the pinned vLLM wheel installed. (Its
`vllm[...]==0.23.0` entry is an opt-in extra a bare install never resolves — see the Dockerfile
Layer-5c comment for the full mechanism.) The in-tree
`Dockerfile` is the canonical, reviewable build. The shipped vLLM pin (`e2f993dc4`) matches the
substrate the `benchmarks/` tables were measured on — but those tables were measured on the
**sibling vLLM sample's image** with an earlier probe revision, **not on this Dynamo image**, so a
rebuild of this sample does **not** reproduce them — see `benchmarks/README.md` for the full
provenance.

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
- Therefore the honest readiness signal is **`/health` is 200 *and* has ≥1 endpoint**. The probe
  checks that structurally — `curl -sf .../health | python3 -c 'import json,sys; sys.exit(0 if
  json.load(sys.stdin).get("endpoints") else 1)'` — so it does not depend on how the frontend
  happens to serialize the body: a populated `endpoints` array passes, an empty one (or non-JSON)
  fails, and a formatting change across a Dynamo bump cannot turn a succeeding load into a probe
  kill.

Workers are `--headless` (no HTTP server, and they never register a discovery endpoint), so their
probe is **process-liveness only** (`pgrep -f dynamo.vllm`). A worker's startup passes on its first
tick and a wedged-but-alive worker stays Ready — a known limitation, not a readiness check. A worker's
true progress is observed through the leader coming Ready: the DP rendezvous cannot complete (and the
leader cannot register its engine) until every worker has joined. This pairs with
`publishNotReadyAddresses: true` on the headless Service — workers must resolve the leader's DNS
A-record *before* the leader is Ready, or the rendezvous deadlocks.

### Update / recovery — all pods together, never one at a time

The DP rendezvous is **one-shot**: a restarted pod cannot rejoin a group whose other members kept
running. The StatefulSet therefore ships `updateStrategy: OnDelete`, and the single update **and**
recovery procedure is to delete all pods together so the group re-forms from scratch:

```bash
kubectl -n dynamo-deepep delete pod -l app=dynamo-deepep-v2   # roll a new image tag / recover a wedged group
```

The default rolling update would replace pods one at a time, each replacement joining a collective
that no longer exists while the survivor keeps reporting Ready. The leader also carries a
`livenessProbe` (same exec as readiness, gated by the `startupProbe`) so frontend death or a wedged
leader turns into a visible restart — after which the all-pods delete above is the recovery.

## Benchmark

```bash
# from inside the leader pod (both scripts are baked into the image at /opt):
kubectl -n dynamo-deepep exec dynamo-deepep-v2-0 -- env OUT_ROOT=/work/benchmarks bash /opt/benchmark.sh 127.0.0.1

# or from outside the cluster, via a port-forward to the leader pod:
kubectl -n dynamo-deepep port-forward pod/dynamo-deepep-v2-0 8000:8000 &
bash recipe/benchmark.sh 127.0.0.1
```

Concurrency sweep 1/8/16/32/64 (5× requests per level), writes `benchmarks/raw/`. Exits non-zero
unless **every** request at **every** level succeeded.

**Persist the raws off-pod before deleting the pods** — `/work` is an `emptyDir` and dies with the
pod (the original sweeps' raws were lost exactly this way):

```bash
kubectl -n dynamo-deepep cp dynamo-deepep-v2-0:/work/benchmarks "./benchmarks-raw-$(date -u +%Y%m%dT%H%M%SZ)"
```

See `benchmarks/README.md` for the measured eager and non-eager tables + environment provenance
(measured on the sibling vLLM sample's image — not re-measured on this Dynamo image).

## Known limitations

- Measured on **H200 (p5en, `sm_90`) only**; no Blackwell serving run is in this sample. The manifest's
  `DEEPEP_ARCH_LIST=10.0` (p6-b200) / `10.3` (p6-b300) knobs are **documented but not verified**: an
  earlier bring-up on 2× p6-b300 hit a Blackwell PTX codegen failure at CUDA 13.0 on the
  `deepseek-ai/DeepEP@b306af06` lineage this sample previously pinned. The shipped pin is now the
  [`amazon-contributing/DeepEP`](https://github.com/amazon-contributing/DeepEP) fork, which carries the
  `st.bulk` 64-bit-operand fix ([#3](https://github.com/amazon-contributing/DeepEP/pull/3)) that removes
  that specific codegen failure — but no Blackwell run exists at the shipped pin, so re-verify on p6
  before advertising those rows as working.
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
  (which appeared 2026-08-24). When the canonical moves, that is the file to track. Two deliberate
  divergences justify a separate script here; the next reader should know they are choices, not drift:
  1. **CPU-proxy (`NCCL_GIN_TYPE=2`), not EFA-GDA.** This is the GDAKI-off, CPU-proxy transport that is
     viable on EFA today; the canonical benchmark's defaults and NCCL build target a different point in
     that design space.
  2. **Coupling to the vLLM wheel's torch/NCCL ABI.** The DeepEP `_C.so` here is built in-pod against
     the exact `torch 2.11+cu130` / `nvidia-nccl-cu13 2.30.4` the pinned vLLM wheel drags in (Dockerfile
     Layer 5b re-pins it), so the toolchain is wheel-driven rather than a standalone NCCL build tree.
     (The aws-ofi-nccl plugin itself is the canonical's version — released `v1.21.1`, built from source
     so gdrcopy support is compiled in by construction.)

  The **DeepEP source** matches the canonical: both pin the
  [`amazon-contributing/DeepEP`](https://github.com/amazon-contributing/DeepEP) fork — this sample at
  the immutable SHA `97d8f9bc` rather than the canonical's floating `main`, so the build is
  reproducible. See the Blackwell caveat under **Known limitations** before using the
  `DEEPEP_ARCH_LIST=10.x` knob.
- `setup_deepep_v2_efa.sh` is deliberately **outside** `.github/workflows/deepep-vendor-sync.yml`. That
  CI gates only the NVSHMEM `setup_deepep_efa.sh` vendored copy (canonical at
  `micro-benchmarks/expert-parallelism/deepep-benchmark/`) — a different script — so this V2/GIN variant
  is correctly not in that workflow. Do not add it to that workflow.
