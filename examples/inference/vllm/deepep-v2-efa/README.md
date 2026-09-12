<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0 -->
# vLLM with DeepEP-V2 MoE all-to-all over EFA (eager + non-eager)

Serve a Mixture-of-Experts model on vLLM with **DeepEP-V2** (`ElasticBuffer`) expert-parallel
all-to-all, routed over **AWS EFA** via the NCCL-GIN **CPU-proxy** path (`NCCL_GIN_TYPE=2`). This is the
V2 / NCCL-GIN counterpart to the NVSHMEM-backed `../../sglang/dsr1-deepep-efa` sample: no NVSHMEM, no
IBGDA — DeepEP-V2's `ElasticBuffer` drives the dispatch/combine collectives over `aws-ofi-nccl`'s GIN
plugin on `efa-direct`.

Validated on **2× and 4× p5en.48xlarge (H200)**, `Qwen/Qwen3-30B-A3B-FP8`: **measured at DP16/EP16**
(the concurrency sweeps in `benchmarks/`); **DP32/EP32 functionally validated** (16/16 HTTP 200
bring-up, no measured sweep — the shipped manifest is the 2-node/EP16 shape).

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

### eager vs non-eager (both measured; see `benchmarks/`)

| Mode | Status | How |
|---|---|---|
| `--enforce-eager` | **Serves, zero extra patches** | the default this sample ships |
| default compilation (CUDA graphs) | **Unblocked upstream — no patch shipped here** | vLLM [#46404](https://github.com/vllm-project/vllm/pull/46404) + [#46432](https://github.com/vllm-project/vllm/pull/46432) and the empty-`ExpertTokensMetadata` guard ([vLLM #52632](https://github.com/vllm-project/vllm/pull/52632), merged 2026-08-20) are all in main. The `VLLM_SHA` pin is now on #52632's merge commit, so default compilation works without `--enforce-eager` and with no build-time patch step. |

At stock `e2f993dc4` (the first commit with the `deepep_v2` backend), default compilation crashes
deterministically ~48 s into startup in `profile_run` (`deepep_v2.py` combine). `--enforce-eager` avoids
it and is the path this sample ships and supports. Historical non-eager measurements (taken with the
then-unmerged guard) remain in `benchmarks/` for reference.

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
first boot (needs a live CUDA context) by `recipe/`-invoked `build_deepep.sh`. The in-tree
`Dockerfile` is the canonical, reviewable build. The published benchmark numbers were taken with it at
the previous pin (`e2f993dc4`); the pin has since moved to `14617c2b` (vLLM #52632's merge commit) and
the tables have **not** been re-measured on it — see `benchmarks/README.md`.

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

```bash
# eager (default). SERVE_DP = total data-parallel = EP size; SERVE_DP_LOCAL = GPUs/node.
SERVE_DP=16 bash recipe/serve.sh leader <leader-ip>       # on node 0
SERVE_DP=16 bash recipe/serve.sh worker <leader-ip> 8     # on node 1
```

Default compilation (CUDA graphs) works with the shipped pin: the upstream guard
([vLLM #52632](https://github.com/vllm-project/vllm/pull/52632)) merged 2026-08-20 and the `VLLM_SHA`
pin is now on its merge commit, so you can drop `--enforce-eager` — no patch step. The knob is
`SERVE_ENFORCE_EAGER`: it defaults to `1` (eager, the mode the published tables were measured with);
set `SERVE_ENFORCE_EAGER=0` to serve with default compilation. Re-measure at your concurrency before
quoting non-eager numbers — the `benchmarks/` tables are eager and were taken on the previous pin.
Kubernetes: `kubectl apply -f kubernetes/` (2-node StatefulSet + headless service; the proxy-Gin env
contract + EFA device requests are set there).

## Benchmark

```bash
# from inside the leader pod (both scripts are baked into the image at /opt):
kubectl -n vllm-deepep exec vllm-deepep-v2-0 -- env OUT_ROOT=/work/benchmarks bash /opt/benchmark.sh 127.0.0.1

# or from outside the cluster, via a port-forward to the leader pod:
kubectl -n vllm-deepep port-forward pod/vllm-deepep-v2-0 8000:8000 &
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
- Default-compilation (non-eager) serving needs the empty-`ExpertTokensMetadata` guard
  ([vLLM #52632](https://github.com/vllm-project/vllm/pull/52632), merged 2026-08-20). This sample
  carries **no build-time patches**: the `VLLM_SHA` pin is now on #52632's merge commit, so non-eager
  works with zero recipe changes (`SERVE_ENFORCE_EAGER=0`). The non-eager numbers in `benchmarks/` were
  measured on the **previous** pin (`e2f993dc4`) with that guard applied as an unmerged patch; they have
  **not** been re-measured on the shipped pin — treat them as historical, not as what a rebuild produces.
- Only a **Kubernetes** launcher is shipped and exercised (`kubernetes/`). No Slurm/Pyxis `.sbatch` is
  provided because none was run; the raw two-node `recipe/serve.sh` path is the manual fallback.
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
