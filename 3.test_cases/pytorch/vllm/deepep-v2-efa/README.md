<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0 -->
# vLLM with DeepEP-V2 MoE all-to-all over EFA (eager + non-eager)

Serve a Mixture-of-Experts model on vLLM with **DeepEP-V2** (`ElasticBuffer`) expert-parallel
all-to-all, routed over **AWS EFA** via the NCCL-GIN **CPU-proxy** path (`NCCL_GIN_TYPE=2`). This is the
V2 / NCCL-GIN counterpart to the NVSHMEM-backed `../../sglang/dsr1-deepep-efa` sample: no NVSHMEM, no
IBGDA — DeepEP-V2's `ElasticBuffer` drives the dispatch/combine collectives over `aws-ofi-nccl`'s GIN
plugin on `efa-direct`.

Validated on **2× and 4× p5en.48xlarge (H200)**, `Qwen/Qwen3-30B-A3B-FP8`, DP16/EP16 and DP32/EP32.

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
| default compilation (CUDA graphs) | **Pending one upstream fix — no patch shipped here** | vLLM [#46404](https://github.com/vllm-project/vllm/pull/46404) + [#46432](https://github.com/vllm-project/vllm/pull/46432) are merged; the remaining piece is the empty-`ExpertTokensMetadata` guard, now filed upstream ([vLLM #52632](https://github.com/vllm-project/vllm/pull/52632)). Once it merges, bump the vLLM pin past it and serve without `--enforce-eager` — no build-time patch step. |

At stock `e2f993dc4` (the first commit with the `deepep_v2` backend), default compilation crashes
deterministically ~48 s into startup in `profile_run` (`deepep_v2.py` combine). `--enforce-eager` avoids
it and is the path this sample ships and supports. Historical non-eager measurements (taken with the
then-unmerged guard) remain in `benchmarks/` for reference.

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
first boot (needs a live CUDA context) by `recipe/`-invoked `build_deepep.sh`.

> **Fast path (no clone):** the same image is also buildable from a single self-fetching Dockerfile —
> `curl -fsSL <gist-raw>/10-Dockerfile.selfcontained -o Dockerfile && docker build -t vllm-deepep-v2 .`
> — see the gist referenced in `benchmarks/README.md`. The in-tree `Dockerfile` (vendored-script) is the
> canonical, reviewable build.

## Smoke-test the EFA transport before loading the model

**1. Static image check (single node, no rendezvous):**
```bash
bash recipe/verify-image.sh $REGISTRY/vllm-deepep-v2:$IMAGE_TAG
```
Asserts (fail-loud) `fi_info -p efa` shows `efa-direct`, the GIN plugin exports `ncclGinPlugin`, and
`import deep_ep` resolves `ElasticBuffer`.

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

## Serve
```bash
# eager (default). SERVE_DP = total data-parallel = EP size; SERVE_DP_LOCAL = GPUs/node.
SERVE_DP=16 bash recipe/serve.sh leader <leader-ip>       # on node 0
SERVE_DP=16 bash recipe/serve.sh worker <leader-ip> 8     # on node 1
```
Default compilation (CUDA graphs) is not enabled in this sample until the upstream guard
([vLLM #52632](https://github.com/vllm-project/vllm/pull/52632)) merges; then bump the vLLM pin past it and drop `--enforce-eager` — no patch step.
Kubernetes: `kubectl apply -f kubernetes/` (2-node StatefulSet + headless service; the proxy-Gin env
contract + EFA device requests are set there).

## Benchmark
```bash
bash recipe/benchmark.sh $LEADER_IP    # concurrency sweep 1/8/16/32/64, writes benchmarks/raw/
```
See `benchmarks/README.md` for the measured eager and non-eager tables + environment provenance.

## Known limitations
- Measured on **H200 (p5en) only**; no Blackwell serving run is in this sample.
- The `benchmarks/` numbers are an **at-scale throughput + relative-latency** datapoint (fixed 128-token
  greedy decode, single sweep per mode), **not** a tuned per-token-latency (TTFT) baseline.
- Default-compilation (non-eager) serving is deliberately **not shipped** here: it requires the
  empty-`ExpertTokensMetadata` guard now filed upstream ([vLLM #52632](https://github.com/vllm-project/vllm/pull/52632)). This sample carries **no
  build-time patches** — when the guard merges, a vLLM pin bump enables non-eager with zero recipe changes.
  The non-eager numbers in `benchmarks/` are historical measurements taken with that guard applied.
- Only a **Kubernetes** launcher is shipped and exercised (`kubernetes/`). No Slurm/Pyxis `.sbatch` is
  provided because none was run; the raw two-node `recipe/serve.sh` path is the manual fallback.
- `setup_deepep_v2_efa.sh` is first-party-authored for the V2 / NCCL-GIN path and is deliberately
  **outside** `.github/workflows/deepep-vendor-sync.yml` (that CI gates the NVSHMEM `setup_deepep_efa.sh`
  vendored copy — a different script). Do not add this script to that workflow.
