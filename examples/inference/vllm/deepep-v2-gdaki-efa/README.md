<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0 -->
# vLLM with DeepEP-V2 MoE all-to-all over EFA — GDAKI (GPU-initiated) variant

Serve a Mixture-of-Experts model on vLLM with **DeepEP-V2** (`ElasticBuffer`) expert-parallel
all-to-all, routed over **AWS EFA** via the NCCL-GIN **GDAKI** path (`NCCL_GIN_TYPE=3`,
`OFI_NCCL_GIN_GDAKI=1`) — the **GPU-initiated** transport where the GPU kernel posts the RDMA work
requests (WQEs) itself, instead of handing them to a CPU proxy thread. This is the GDAKI counterpart
to the CPU-proxy sample [`../deepep-v2-efa`](../deepep-v2-efa) (`NCCL_GIN_TYPE=2`): **same base, same
torch/NCCL/NVSHMEM/vLLM/DeepEP pins — the only deltas are the GDAKI transport stack.** Neither uses
NVSHMEM or IBGDA; DeepEP-V2's `ElasticBuffer` drives the dispatch/combine collectives over
`aws-ofi-nccl`'s GIN plugin on `efa-direct`.

Validated on **2× and 4× p5en.48xlarge (H200)**, `Qwen/Qwen3-30B-A3B-FP8`, DP16/EP16 and DP32/EP32.

## How the GDAKI transport gets onto EFA

DeepEP's default transport is NVSHMEM/IBGDA, which EFA does not provide. The proxy sibling runs the
V2 `ElasticBuffer` dispatch/combine over `aws-ofi-nccl`'s **GIN CPU-proxy** (`NCCL_GIN_TYPE=2`). This
sample instead uses **GDAKI** (`NCCL_GIN_TYPE=3`, `OFI_NCCL_GIN_GDAKI=1`), where the GPU kernel posts
WQEs directly. GDAKI on EFA needs a newer transport substrate than the proxy path, built here from
public source:

1. **rdma-core @ master post-[PR#1701](https://github.com/linux-rdma/rdma-core/pull/1701)** (merged
   2026-08-04) — adds the EFA completion-counter verbs (`efadv` comp-cntr) that libfabric's efa
   provider and GDAKI's hw-counter mode consume. No release tag carries it yet, so pinned by SHA.
2. **libfabric @ main post-[PR#12591](https://github.com/ofiwg/libfabric/pull/12591)** (merged
   2026-07-28) — makes `prov/efa` consume the comp-cntr caps; built against the rdma-core above.
3. **aws-ofi-nccl @ `a3d2680` `--enable-gdaki`** (SHA pin, **no local patches**) — `a3d2680`
   carries [PR#1311](https://github.com/aws/aws-ofi-nccl/pull/1311) (the per-platform EFA hw-counter
   tristate `OFI_NCCL_GDAKI_EFA_HW_COUNTER`) and the GIN seq-space aliasing fix. It does **not** carry
   [PR#1351](https://github.com/aws/aws-ofi-nccl/pull/1351) (the `OFI_NCCL_GDRCOPY_FORCED_PCIE_COPY`
   forced-PCIe override): the maintainer **closed #1351** in favour of requiring **GDRCopy 2.5+** on the
   node, so this sample follows that guidance — no cherry-pick, and GDRCopy 2.5+ is a documented
   [node precondition](#prerequisites) instead.

Everything else — `EP_REUSE_NCCL_COMM=0` (or serve init segfaults; DeepEP must create its own comm
because torch's is lazy/null under vLLM), the DeepEP source pin
([`amazon-contributing/DeepEP`](https://github.com/amazon-contributing/DeepEP) `97d8f9bc`, the AWS
EPv2/NCCL-GIN fork — same fork the house V2 canonical
[`setup_deepep_gin.sh`](../../../../micro-benchmarks/expert-parallelism/deepep-v2-benchmark/setup_deepep_gin.sh)
clones; it carries the EFA delta **in-code**, including both halves of the superseded draft
[deepseek-ai/DeepEP#612](https://github.com/deepseek-ai/DeepEP/pull/612) — the `get_rdma_gbs()` sysfs
link-rate fast path and the auto-QP overflow clamp), and the vLLM wheel-pin — is the standard
V2/GDAKI contract.

**Net:** this folder is **post-merge upstream SHA pins + the amazon-contributing/DeepEP fork (an
immutable SHA), with zero local source patches.** The fork supersedes the earlier `deepseek-ai`
base + draft PR#612 (pinning the pre-fix upstream fork-point forfeited exactly those fixes). The
merged-upstream substrate fixes (rdma-core #1701, libfabric #12591, aws-ofi-nccl #1311, carried by
the `a3d2680` pin) are plain SHA pins, not patches.

### eager vs non-eager (both measured; see `benchmarks/`)

| Mode | Status | How |
|---|---|---|
| `--enforce-eager` | **Serves, zero extra patches** | the default this sample ships |
| default compilation (CUDA graphs) | **Pending one upstream fix — no patch shipped here** | vLLM [#46404](https://github.com/vllm-project/vllm/pull/46404) + [#46432](https://github.com/vllm-project/vllm/pull/46432) are merged; the remaining empty-`ExpertTokensMetadata` guard is filed upstream ([vLLM #52632](https://github.com/vllm-project/vllm/pull/52632)). Once merged, bump the vLLM pin past it and serve without `--enforce-eager` — no build-time patch step. |

At stock `e2f993dc4` (the first commit with the `deepep_v2` backend), default compilation crashes
deterministically during startup in `profile_run` (`deepep_v2.py` combine, a Triton illegal-memory
access). This is **transport-independent** — the same crash and the same fix as the proxy sample,
because the vLLM pin is identical. `--enforce-eager` avoids it; the fix stack above lets default
compilation serve. Numbers for both modes are in `benchmarks/`.

## Prerequisites

- An EKS cluster of p5en.48xlarge (H200) with EFA + the EFA K8s device plugin (the shipped launcher);
  the container also runs under raw `docker run` on any 2 EFA hosts if you wire the rendezvous by hand.
- **Node GDRCopy >= 2.5:** the GIN path needs the host `gdrdrv` kernel module at GDRCopy 2.5 or newer
  (check `cat /sys/module/gdrdrv/version`, or `modinfo gdrdrv | grep ^version`). This replaces the
  older aws-ofi-nccl [PR#1351](https://github.com/aws/aws-ofi-nccl/pull/1351) forced-PCIe workaround,
  which the maintainer closed in favour of requiring GDRCopy 2.5+ on the node — so this sample carries
  **no** `OFI_NCCL_GDRCOPY_FORCED_PCIE_COPY` and no #1351 cherry-pick; make sure the node driver is 2.5+
  (`recipe/verify-image.sh` and the K8s launcher fail loud if it is older).
- **Node efa.ko for GDAKI hw-counter mode:** the byte-level hardware completion counter needs node
  `efa.ko >= 3.3.0` (efa_linux_3.3.0, 2026-07-28; check `cat /sys/module/efa/version`). On older
  nodes set `OFI_NCCL_GDAKI_EFA_HW_COUNTER=off` (the kubernetes/ YAML default) and GDAKI runs in its
  non-hw-counter mode — the transport still works; only the byte-level `/sys` counters are absent.
- An ECR repo you own (set in `setup/env_vars`); this sample never hardcodes a registry.
- Hugging Face access for the model (`Qwen/Qwen3-30B-A3B-FP8` is public, no token required).

## Build

```bash
cp setup/env_vars.example setup/env_vars && $EDITOR setup/env_vars   # set REGISTRY, IMAGE_TAG
bash setup/build-push.sh
```

The image is NGC-from-scratch (`FROM nvcr.io/nvidia/cuda:...`). It builds rdma-core (post-#1701) and
libfabric (post-#12591) from source, then `setup_deepep_v2_gdaki_efa.sh` builds aws-ofi-nccl
`--enable-gdaki` at its pinned SHA (no local patches) against them and stages DeepEP-V2 source
(`b306af06` + upstream PR#612); the `_C.so` is compiled in-pod on first boot (needs a live CUDA
context) by `recipe/build_deepep.sh`.

## Smoke-test the substrate before loading the model

**1. Static image check (single node, no rendezvous):**

```bash
bash recipe/verify-image.sh $REGISTRY/vllm-deepep-v2-gdaki-efa:$IMAGE_TAG
```

Asserts (fail-loud) the EFA provider resolves, the rdma-core comp-cntr verbs are present, a single
NCCL 2.30.4 wins on the linker path, the GIN plugin exports `ncclGinPlugin`, GDAKI + the
`OFI_NCCL_GDAKI_EFA_HW_COUNTER` param are compiled in, and (when run on a node) the host gdrdrv is
GDRCopy 2.5+.

**2. Cross-node kernel smoke (prove bytes move over EFA before the model load):**

```bash
# in the pods/containers, one per node — runs DeepEP-V2's own elastic EP test on the GDAKI transport
bash /opt/run-kernel-test.sh leader <leader-ip>            # node 0
bash /opt/run-kernel-test.sh worker <leader-ip> 1          # node 1
```

Runs `DeepEP/tests/elastic/test_ep.py` across the nodes with the exact GDAKI-Gin/EFA env the serve
uses, and only prints `KERNEL-TEST PASS` when the test passes **and** the `NCCL_DEBUG=INFO` log shows
the `efa-direct` banner (so a green result cannot be a silent TCP/SHM fallback). It also warns on the
known `GDAKI-CQE ... status 9` upstream signature. This is the one step that cannot hang for hours —
run it before committing a node to the multi-hundred-GB weight load.

## Serve

```bash
# eager (default). SERVE_DP = total data-parallel = EP size; SERVE_DP_LOCAL = GPUs/node.
SERVE_DP=16 bash recipe/serve.sh leader <leader-ip>       # on node 0
SERVE_DP=16 bash recipe/serve.sh worker <leader-ip> 8     # on node 1
# non-eager (CUDA graphs): apply the fix stack first, then serve without --enforce-eager
# non-eager: pending the upstream guard ([vLLM #52632](https://github.com/vllm-project/vllm/pull/52632)) — pin bump enables it, no patch step
```

Kubernetes: `kubectl apply -f kubernetes/` (2-node StatefulSet + headless service; the GDAKI-Gin env
contract, the `OFI_NCCL_GDAKI_EFA_HW_COUNTER` tristate, and EFA device requests are all set there).

## Benchmark

```bash
bash recipe/benchmark.sh $LEADER_IP    # concurrency sweep 1/8/16/32/64, writes benchmarks/raw/
```

See [`benchmarks/README.md`](benchmarks/README.md) for the measured eager, non-eager, and the
same-node-set **GDAKI-vs-proxy transport A/B** tables (EP16 + EP32) + environment provenance.

## Known limitations

- Measured on **H200 (p5en) only**; no Blackwell serving run is in this sample.
- The transport A/B is **one sweep per arm/scale**; the 2×2 (EP16+EP32, GDAKI+proxy, 10 paired
  comparisons, GDAKI ≥ proxy in all 10 by +0.2–3.2%) shows a consistent direction, not a
  statistically tight interval.
- The validation nodes ran efa.ko 3.0.x/3.1.x, so the cross-node proof is **functional** (efa-direct
  boot banner + coherent EP output), **not** a byte-level `/sys` hw-counter tally — that needs
  efa.ko ≥ 3.3.0. See `benchmarks/README.md` caveats.
- The non-eager fix stack pins two already-merged upstream PRs + a one-line guard staged for upstream;
- Default-compilation serving is deliberately not shipped until the upstream guard ([vLLM #52632](https://github.com/vllm-project/vllm/pull/52632)) merges; the non-eager numbers in `benchmarks/` are historical measurements taken with that guard applied. Also,
  vLLM [#47785](https://github.com/vllm-project/vllm/pull/47785) (a compiled align-sum kernel fix)
  post-dates the pinned `e2f993dc4` and is **documented-missing** — a source cherry-pick is inert
  under the precompiled wheel; the swap seam is a newer `VLLM_SHA` + wheel URL (see the Dockerfile).
- Only a **Kubernetes** launcher is shipped and exercised (`kubernetes/`). No Slurm/Pyxis `.sbatch`
  is provided because none was run; the raw two-node `recipe/serve.sh` path is the manual fallback.
- `setup_deepep_v2_gdaki_efa.sh` is first-party-authored for the V2 / NCCL-GIN GDAKI path and is
  deliberately **outside** `.github/workflows/deepep-vendor-sync.yml` (that CI gates the NVSHMEM
  `setup_deepep_efa.sh` vendored copy — a different script). Do not add this script to that workflow.
