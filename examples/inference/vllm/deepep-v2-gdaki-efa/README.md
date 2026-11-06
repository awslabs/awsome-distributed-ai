<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0 -->
# vLLM with DeepEP-V2 MoE all-to-all over EFA — built for GDAKI (GPU-initiated), not yet shown on hardware

This sample is built to serve a Mixture-of-Experts model on vLLM with **DeepEP-V2** (`ElasticBuffer`)
expert-parallel all-to-all routed over **AWS EFA** via the NCCL-GIN **GDAKI** path (`NCCL_GIN_TYPE=5`,
EFA-GDA; NCCL selects the GIN backend, and the pinned plugin builds GDAKI in when configure finds
libfabric's GDA ops + hw-counter ABI and CUDA): the **GPU-initiated** transport, where the GPU kernel
posts the RDMA work requests (WQEs) itself instead of handing them to a CPU proxy thread. **GDAKI has
not yet been shown on hardware with this sample**; [How to confirm GDAKI is active](#how-to-confirm-gdaki-is-active)
is the pass/fail test. This is the GDAKI counterpart
to the CPU-proxy sample [`../deepep-v2-efa`](../deepep-v2-efa) (`NCCL_GIN_TYPE=2`): **same base and
torch/NVSHMEM/DeepEP pins; the deltas are the GDAKI transport stack (EFA installer 1.48.0 here, 1.50.0 there), NCCL 2.31.2 (the proxy sample pins 2.30.7) and the vLLM pin (`e2f993dc4` here, `14617c2b` there).** Neither uses
NVSHMEM or IBGDA; DeepEP-V2's `ElasticBuffer` drives the dispatch/combine collectives over
`aws-ofi-nccl`'s GIN plugin on `efa-direct`.

**Hardware runs so far: CPU proxy only.** The 2026-08-14 serves on 2× and 4× p5en.48xlarge (H200),
`Qwen/Qwen3-30B-A3B-FP8`, DP16/EP16 and DP32/EP32 (NCCL 2.30.4 and an earlier DeepEP tree, see the
[provenance table](benchmarks/README.md#environment-provenance)) ran the plugin's v13 CPU-proxy GIN
table, not GDAKI. Evidence: a same-day `NCCL_DEBUG=INFO` boot log (one non-eager DP16/EP16 serve of an
image built from the same pins) shows `NET/Plugin: Loaded gin plugin Libfabric (v13)` at line 351 and
`Assigned GIN plugin Libfabric to comm` at line 354, and the same pair on all 8 ranks it captured. The
2026-08-14 env-flip A/B (`NCCL_GIN_TYPE` 3 vs 2): CPU proxy, both arms the same transport configuration;
the A/B shows no difference. The shipped pin set has not been run (a DeepEP `nccl.cu` compile gate only; see
[Known limitations](#known-limitations)).

## How the sample is built for GDAKI on EFA

DeepEP's default transport is NVSHMEM/IBGDA, which EFA does not provide. The proxy sibling runs the
V2 `ElasticBuffer` dispatch/combine over `aws-ofi-nccl`'s **GIN CPU-proxy** (`NCCL_GIN_TYPE=2`). This
sample is instead configured for **GDAKI** (`NCCL_GIN_TYPE=5`, EFA-GDA), where the GPU kernel posts
WQEs directly. GDAKI on EFA needs a newer transport substrate than the proxy path, built here from
public source:

1. **rdma-core @ master post-[PR#1701](https://github.com/linux-rdma/rdma-core/pull/1701)** (merged
   2026-08-04) — adds the EFA completion-counter verbs (`efadv` comp-cntr) that libfabric's efa
   provider and GDAKI's hw-counter mode consume. No release tag carries it yet, so pinned by SHA.
2. **libfabric @ main post-[PR#12591](https://github.com/ofiwg/libfabric/pull/12591)** (merged
   2026-07-28) — makes `prov/efa` consume the comp-cntr caps; built against the rdma-core above.
3. **aws-ofi-nccl @ `a3d2680` with GDAKI** (SHA pin, **no local patches**; GDAKI is detected at configure time, and the build fails if `ncclGinPlugin_v14` is not exported) — `a3d2680`
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
base + the open, unmerged PR#612 (pinning the pre-fix upstream fork-point forfeited exactly those fixes). The
merged-upstream substrate fixes (rdma-core #1701, libfabric #12591, aws-ofi-nccl #1311, carried by
the `a3d2680` pin) are plain SHA pins, not patches.

### eager vs non-eager (both measured on the CPU proxy; see `benchmarks/`)

| Mode | Status | How |
|---|---|---|
| `--enforce-eager` | **Served on the CPU proxy (2026-08-14), zero extra patches** | the default this sample ships |
| default compilation (CUDA graphs) | **Not shipped at this pin — no patch here** | vLLM [#46404](https://github.com/vllm-project/vllm/pull/46404) + [#46432](https://github.com/vllm-project/vllm/pull/46432) are in the pin; the empty-`ExpertTokensMetadata` guard ([vLLM #52632](https://github.com/vllm-project/vllm/pull/52632), **merged 2026-08-20**) post-dates the pinned `e2f993dc4`. Enabling it is a `VLLM_SHA` pin bump past #52632's merge commit plus a re-measure — no build-time patch step. The proxy sibling (`../deepep-v2-efa`) has taken that bump; this sample has not been re-measured on it. |

At stock `e2f993dc4` (the first commit with the `deepep_v2` backend), default compilation crashes
deterministically during startup in `profile_run` (`deepep_v2.py` combine, a Triton illegal-memory
access). It is the same crash, with the same vLLM-side fix, as in the proxy sample: both were
measured at vLLM `e2f993dc4`, both on the CPU proxy (the proxy sample has since moved to `14617c2b`). `--enforce-eager` avoids it. The non-eager numbers in `benchmarks/`
are historical — taken with the then-unmerged guard applied — and the shipped recipe carries no patch.

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
  nodes set `OFI_NCCL_GDAKI_EFA_HW_COUNTER=off` (the kubernetes/ YAML default); the plugin is built to
  run GDAKI in its non-hw-counter mode there, without the byte-level `/sys` counters (not yet shown on
  hardware).
- An ECR repo you own (set in `setup/env_vars`); this sample never hardcodes a registry.
- Hugging Face access for the model (`Qwen/Qwen3-30B-A3B-FP8` is public, no token required).

## Build

```bash
cp setup/env_vars.example setup/env_vars && $EDITOR setup/env_vars   # set REGISTRY, IMAGE_TAG
bash setup/build-push.sh
```

The image is NGC-from-scratch (`FROM nvcr.io/nvidia/cuda:...`). It builds rdma-core (post-#1701) and
libfabric (post-#12591) from source, then `setup_deepep_v2_gdaki_efa.sh` builds aws-ofi-nccl
with GDAKI at its pinned SHA (no local patches) against them and stages the DeepEP-V2 source from
the [`amazon-contributing/DeepEP`](https://github.com/amazon-contributing/DeepEP) fork at `97d8f9bc`
(SHA pin, no local patches); the `_C.so` is compiled in-pod on first boot for the arch the node runs
(`DEEPEP_ARCH_LIST`; a design choice, since nvcc needs no GPU) by `recipe/build_deepep.sh`.

## Smoke-test the substrate before loading the model

**1. Static image check (single node, no rendezvous):**

```bash
bash recipe/verify-image.sh $REGISTRY/vllm-deepep-v2-gdaki-efa:$IMAGE_TAG
```

Asserts (fail-loud) the EFA provider resolves, the rdma-core comp-cntr verbs are present, a single
NCCL 2.31.2 wins on the linker path, the GIN plugin exports `ncclGinPlugin`, GDAKI + the
`OFI_NCCL_GDAKI_EFA_HW_COUNTER` param are compiled in, and (when run on a node) the host gdrdrv is
GDRCopy 2.5+.

**2. Cross-node kernel smoke (prove bytes move over EFA before the model load):**

```bash
# in the pods/containers, one per node — runs DeepEP-V2's own elastic EP test with the GDAKI env
bash /opt/run-kernel-test.sh leader <leader-ip>            # node 0
bash /opt/run-kernel-test.sh worker <leader-ip> 1          # node 1
```

Runs `DeepEP/tests/elastic/test_ep.py` across the nodes with the exact GDAKI-Gin/EFA env the serve
uses, and only prints `KERNEL-TEST PASS` when the test passes **and** the `NCCL_DEBUG=INFO` log shows
the `efa-direct` banner (so a green result cannot be a silent TCP/SHM fallback). A `KERNEL-TEST PASS`
shows that EFA carried the traffic; it is not the GDAKI check (see
[How to confirm GDAKI is active](#how-to-confirm-gdaki-is-active)). It also warns on the
known `GDAKI-CQE ... status 9` upstream signature. This is the one step that cannot hang for hours —
run it before committing a node to the multi-hundred-GB weight load.

## How to confirm GDAKI is active

This is the pass/fail test for a hardware run of this sample. Run the kernel smoke (its default is
`NCCL_DEBUG=INFO`) or the serve with `SERVE_NCCL_DEBUG=INFO`, and set `NCCL_DEBUG_SUBSYS=INIT,NET,ENV` in
the same environment, for example
`NCCL_DEBUG_SUBSYS=INIT,NET,ENV bash /opt/run-kernel-test.sh leader <leader-ip>`. NCCL's default,
`INIT,BOOTSTRAP,ENV` (`src/debug.cc:55-56`), leaves out the plugin's `NET` line. The `kubernetes/` YAML
does not set `NCCL_DEBUG_SUBSYS`: on that path add `- { name: NCCL_DEBUG_SUBSYS, value: "INIT,NET,ENV" }`
to the container `env:` (the YAML already sets `SERVE_NCCL_DEBUG=INFO`, which `serve.sh` turns into
`NCCL_DEBUG=INFO`). Without `NCCL_DEBUG=INFO` none of the four lines below is logged, and without `NET`
the plugin's line (row 4) is not, so such a run proves nothing either way. GDAKI counts as shown only if
**every rank** logs all four lines below; with the settings above, any other result means GDAKI did not
run.

| Line to find (after `NCCL INFO`) | Printed by | What it shows |
|---|---|---|
| `NCCL_GIN_TYPE set by environment to 5.` | NCCL `v2.31.2-1` `src/misc/param.cc:100` | GIN type 5 (EFA-GDA) was requested |
| `GIN/Plugin: Loaded gin plugin Libfabric_GDAKI (v14)` | NCCL `v2.31.2-1` `src/plugin/gin/gin_v14.cc:17` | NCCL loaded the plugin's v14 table, which is its GDAKI table (aws-ofi-nccl `a3d2680` `src/rdma/gin/nccl_ofi_gin_gdaki.cpp:1225-1226`) |
| `devCommCreate: creating <n> contexts: <m> GIN connections with <k> contexts each (<r> contexts total requested)` | NCCL `v2.31.2-1` `src/gin/gin_host.cc:263-265` | NCCL started setting up a GIN device communicator on a kept backend (a failure after this line logs the `DevComm setup failed` line below); with type 5 requested, NCCL keeps only type-5 backends (`src/plugin/gin.cc:144-148`, `src/gin/gin_host.cc:423-433`) |
| `NET/OFI gin GDAKI: createContext done (nranks=… rank=… …)` | aws-ofi-nccl `a3d2680` `src/rdma/gin/nccl_ofi_gin_gdaki.cpp:883-889` | the plugin created its GPU-initiated contexts and returned a type-5 device handle (`:877`) |

NCCL `v2.31.2-1` logs no line that names the GIN type it selected (it logs a type only when it skips or
removes a backend, or when a setup fails), so rows 1 and 3 together are the evidence for type 5. Also
check that no rank logs `GIN/Plugin: Loaded gin plugin Libfabric (v13)` (`src/plugin/gin/gin_v13.cc:37`,
the plugin's CPU-proxy table), `GIN/Plugin: Skipping plugin Libfabric_GDAKI …` (`src/plugin/gin.cc:145`,
`:156`; skip lines for NCCL's built-in GIN plugins are normal, including
`GIN/Plugin: Skipping plugin Libfabric index <i> type 2: NCCL_GIN_TYPE=5 requested`, because NCCL's
built-in GIN proxy takes the name of the RMA plugin it runs over, here aws-ofi-nccl's `Libfabric`,
`src/gin/gin_host_proxy.cc:698`),
`GIN/Plugin: Failed to initialize any GIN plugin` (`src/plugin/gin.cc:304`),
`GIN backend type 5 not supported on all ranks, removing from active backends` (`src/plugin/gin.cc:369`)
or `GIN: DevComm setup failed with backend type 5` (`src/gin/gin_host.cc:437`).

## Serve

```bash
# eager (default). SERVE_DP = total data-parallel = EP size; SERVE_DP_LOCAL = GPUs/node.
SERVE_DP=16 bash recipe/serve.sh leader <leader-ip>       # on node 0
SERVE_DP=16 bash recipe/serve.sh worker <leader-ip> 8     # on node 1
# non-eager (CUDA graphs): not shipped at this pin — bump VLLM_SHA past vLLM #52632 (merged 2026-08-20)
# and re-measure before serving without --enforce-eager; no patch step
```

Kubernetes: `kubectl apply -f kubernetes/` (2-node StatefulSet + headless service; the GDAKI-Gin env
contract, the `OFI_NCCL_GDAKI_EFA_HW_COUNTER` tristate, and EFA device requests are all set there).

## Benchmark

```bash
bash recipe/benchmark.sh $LEADER_IP    # concurrency sweep 1/8/16/32/64, writes benchmarks/raw/
```

See [`benchmarks/README.md`](benchmarks/README.md) for the measured eager, non-eager and env-flip A/B
tables (EP16 + EP32) + environment provenance: CPU proxy, both arms the same transport configuration;
the A/B shows no difference.

## Known limitations

- **The NCCL 2.31.2 / `NCCL_GIN_TYPE=5` pin set has not been run.** It passes a DeepEP `nccl.cu`
  compile gate only. NCCL 2.31.2 is the first release that defines GIN type 5 (EFA-GDA), the type the
  pinned aws-ofi-nccl's GDAKI table reports, and NCCL keeps a GIN plugin only when its type matches a
  set `NCCL_GIN_TYPE`; so this sample sets 5, as the repo's
  [`deepep-v2-benchmark`](../../../../micro-benchmarks/expert-parallelism/deepep-v2-benchmark) does.
  The tables in `benchmarks/` were recorded with `NCCL_GIN_TYPE=3` on NCCL 2.30.4, where NCCL ran the
  plugin's v13 CPU-proxy GIN table, so no GDAKI run exists yet; a run counts only if it passes
  [How to confirm GDAKI is active](#how-to-confirm-gdaki-is-active).
- Measured (CPU proxy) on **H200 (p5en) only**; no Blackwell serving run is in this sample.
- The `NCCL_GIN_TYPE` 3-vs-2 A/B (**one sweep per arm/scale**) is CPU proxy, both arms the same
  transport configuration; the A/B shows no difference. Its per-row deltas (+0.2–3.2%) are between two
  runs of one transport configuration and say nothing about GDAKI. Whether both arms used the identical
  image build is not recorded.
- The 2026-08-14 CPU-proxy runs used nodes with efa.ko 3.0.x/3.1.x, so their cross-node proof is **functional** (efa-direct
  boot banner + coherent EP output), **not** a byte-level `/sys` hw-counter tally — that needs
  efa.ko ≥ 3.3.0. See `benchmarks/README.md` caveats.
- Default-compilation serving is not shipped at this pin: the empty-`ExpertTokensMetadata` guard ([vLLM #52632](https://github.com/vllm-project/vllm/pull/52632)) merged 2026-08-20 but post-dates the pinned `e2f993dc4`; enabling it is a pin bump + re-measure, and the non-eager numbers in `benchmarks/` are historical measurements taken with that guard applied. Also,
  vLLM [#47785](https://github.com/vllm-project/vllm/pull/47785) (a compiled align-sum kernel fix)
  post-dates the pinned `e2f993dc4` and is **documented-missing** — a source cherry-pick is inert
  under the precompiled wheel; the swap seam is a newer `VLLM_SHA` + wheel URL (see the Dockerfile).
- Only a **Kubernetes** launcher is shipped and exercised (`kubernetes/`). No Slurm/Pyxis `.sbatch`
  is provided because none was run; the raw two-node `recipe/serve.sh` path is the manual fallback.
- `setup_deepep_v2_gdaki_efa.sh` is first-party-authored for the V2 / NCCL-GIN GDAKI path and is
  deliberately **outside** `.github/workflows/deepep-vendor-sync.yml` (that CI gates the NVSHMEM
  `setup_deepep_efa.sh` vendored copy — a different script). Do not add this script to that workflow.
