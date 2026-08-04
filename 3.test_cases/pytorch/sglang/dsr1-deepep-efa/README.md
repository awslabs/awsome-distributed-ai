<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# DeepSeek-R1 on SGLang with DeepEP over EFA

Serve **DeepSeek-R1** (671B, FP8, 256 experts, top-8 routing) with SGLang across two
`p5.48xlarge` / `p5en.48xlarge` nodes, using **DeepEP** MoE dispatch/combine kernels running over
**NVSHMEM's libfabric/EFA transport** — GPU expert-parallel all-to-all on AWS EFA with no
InfiniBand anywhere.

The same image also serves as the head-to-head harness: `recipe/serve.sh` selects between the
DeepEP all-to-all, SGLang's ordinary fused-MoE all-to-all, and pure tensor parallelism, so all
three can be measured on identical hardware and fabric. `recipe/serve-pd.sh` does the same in a
**4-node prefill/decode-disaggregated (2P2D)** topology, with the KV cache moving over EFA RDMA via
Mooncake — which is where the interesting result is, because **prefill and decode pick different
winners** ([`benchmarks/`](./benchmarks/README.md)).

| | |
|---|---|
| Base image | `lmsysorg/sglang:v0.5.13.post1-cu130` |
| DeepEP | `deepseek-ai/DeepEP` @ `567632d` + the EFA patch (pre-EPv2, NVSHMEM backend) |
| NVSHMEM | `v3.7.0-0`, built with **only** the libfabric transport (IBRC/IBGDA off) |
| Mooncake | `kvcache-ai/Mooncake` @ main, `-DUSE_EFA=ON` (KV transfer for PD-disaggregation) |
| GPU arch | Hopper `sm_90` + Blackwell `sm_100`/`sm_103` — **serving validated on H100/H200 only**; see [Blackwell](#blackwell-b200--b300) |

> SGLang 0.5.13.post1 already carries the EFA-protocol change upstream, so no SGLang patch is
> needed. Older bases required one.

## How DeepEP gets onto EFA

DeepEP's internode kernels were written for **IBGDA** (GPU-initiated RDMA over InfiniBand), which
EFA does not provide. [`setup_deepep_efa.sh`](./setup_deepep_efa.sh) rewrites those paths onto
NVSHMEM host-proxy QP APIs. That script is a **vendored copy** of the canonical one in
[`micro-benchmarks/expert-parallelism/deepep-benchmark`](../../../../micro-benchmarks/expert-parallelism/deepep-benchmark) —
see that README for the patch rationale and the kernel-level benchmarks. This directory covers
only what it takes to get that build working **inside a serving engine**, which is where the
non-obvious failure modes are.

### The three integration traps

**1. A stock `deep_ep` in the base image silently shadows the EFA build.**
The SGLang image ships its own `deep_ep` in `/usr/local/lib/python3.12/dist-packages`, built for
the standard NVSHMEM **IBGDA** transport. That directory sits *ahead* of our venv on `sys.path`,
so leaving it in place means `import deep_ep` resolves to the IBGDA build: the server starts, the
logs say DeepEP, and **you believe you are running on EFA while you are not**. The Dockerfile
uninstalls it with the *system* pip before building (the build script's own `pip uninstall` runs
inside the venv and cannot touch a system package), then **asserts** that `deep_ep` resolves under
`/opt/deepep-venv` — so this fails the build rather than shipping a misleading image.

**2. The build needs a venv, and the runtime needs it not to matter.**
The base image is PEP 668 `EXTERNALLY-MANAGED`, so DeepEP is built into
`/opt/deepep-venv` created with `--system-site-packages` (it must compile against the base image's
PyTorch 2.11). But the SGLang launcher runs under the default `python3`, which would not see it —
so the build writes a `.pth` into the system `dist-packages` pointing at the venv's
site-packages.

**3. Two NVSHMEM versions collide.**
The base image's torch pulls pip `nvidia-nvshmem-cu13` (**v3.4.5**) as a dependency of
`libtorch_nvshmem.so`. Its `libnvshmem_host.so.3` loads *ahead* of ours, but DeepEP's device cubin
is compiled against **v3.7.0** — every internode and low-latency run then aborts at init with
*"NVSHMEM device library version does not match host library version"*. The Dockerfile overwrites
the pip copy with the v3.7.0 build so the single soname resolves to v3.7.0 for both consumers
(replacing rather than uninstalling keeps `torch_nvshmem` loadable). The launchers also
`LD_PRELOAD` it as a belt-and-braces measure.

`recipe/verify-image.sh` checks all three in one shot.

### And a fourth, if you run PD-disaggregated: Mooncake silently falls back to TCP

**Without `MOONCAKE_PROTOCOL=efa`, Mooncake moves the KV cache over TCP sockets instead of EFA
RDMA.** The server still comes up, still returns correct tokens, and still passes a smoke test —
then deadlocks under sustained high-concurrency prefill with `KVTransferError ... session is not
alive`. The only evidence is one line in the server log:

```
transfer_engine_py.cpp:241] Installing TCP transport (auto_discover disabled in EFA build)   # WRONG
efa_transport.cpp:1025]     EfaTransport: Initialized EFA device rdmap160s0 ...              # right
```

On EFA hardware there is no reason to run the KV path any other way, so `recipe/serve-pd.sh` always
sets it — and prints the grep above, because the only way to know which transport you got is to
check. Do that on every role before benchmarking. The fallback is not backend-specific: it wedges
pure TP too, which never loads DeepEP.

### Runtime requirements that are easy to miss

| Setting | Why |
|---|---|
| `--device /dev/gdrdrv` | **In addition to** `/dev/infiniband`. The libfabric/EFA transport uses GDRCopy to register GPU memory; without it internode runs fail with *"GDRCopy support not enabled. Unable to register gpu memory handle info."* |
| `NVSHMEM_REMOTE_TRANSPORT=libfabric` | NVSHMEM is built with **only** the libfabric transport, so the default `ibrc` finds nothing → *"Peer GPU not accessible / building transport map failed"*. |
| `NVSHMEM_LIBFABRIC_PROVIDER=efa` | Select the EFA provider. |
| `NVSHMEM_NETDEVS_POLICY=EXTERNAL_SHARING_PCIE_SWITCH_NIC_EXCLUSIVE` | Gives each GPU exclusive use of the NIC on its own PCIe switch. Recommended for RDMA performance on p5/p5en, which pair multiple EFA NICs across PCIe switches. |
| `NCCL_NET_PLUGIN=ofi` — **short name, not a path** | The EFA installer's `libnccl-ofi-ngc-v3` ships only `libnccl-net-ofi.so`, not the `libnccl-net.so` NCCL auto-loads, so NCCL logs *"NET/Plugin: Could not find: libnccl-net.so"* and **silently uses TCP sockets** (~14 GB/s vs ~400 GB/s) — visible only as 3–5× worse prefill TTFT. NCCL templates the value into `libnccl-net-<value>.so`, so an absolute path becomes a bogus filename and falls back too. |
| `NVSHMEM_DISABLE_CUDA_VMM` | **Leave it unset for low-latency kernels** (VMM enabled), or the RDMA-buffer `cudaMemset` fails with *"invalid argument"* (`deep_ep.cpp:371`) — measured, and the one direction that matters. Because the server uses `--deepep-mode auto` (low-latency on the decode path), `recipe/serve.sh` leaves VMM enabled. The scripts set it to `1` for `normal`, but that is belt-and-braces, not a requirement: `normal` also runs clean with VMM enabled ([matrix](./benchmarks/README.md#the-deepep-mode--cuda-vmm-coupling-measured-both-ways)). |
| `--network host --ipc host --ulimit memlock=-1 --shm-size 32g` | EFA needs host networking, IPC and unlimited locked memory. |
| `MOONCAKE_PROTOCOL=efa` — **PD-disaggregated only** | KV cache over EFA RDMA. Omitting it is a *silent* fallback to TCP, not an error. See trap 4 above. |

The four transport variables (`NVSHMEM_REMOTE_TRANSPORT`, `NVSHMEM_LIBFABRIC_PROVIDER`,
`NVSHMEM_NETDEVS_POLICY`, `NCCL_NET_PLUGIN`) are baked into the image as `ENV`; the launchers
re-export them anyway so the transport config is visible at the launch surface and overridable per
run. The device flags and `MOONCAKE_PROTOCOL` have to come from the launcher.

### If you enable DP-attention

`--dp-size 16 --enable-dp-attention` is orthogonal to the MoE backend and worth a lot on decode
(~+25–30% for the EP backends), but it needs two things or the server does not start:

| Setting | Why |
|---|---|
| Pre-compile DeepGEMM on **both** prefill nodes (`recipe/serve-pd.sh precompile <rank>`) | DP+EP uses `num_groups=16` grouped-GEMM shapes that are absent from the cache. Their first-time JIT compile is slow enough to trip DeepEP's dispatch warmup timeout — *"DeepEP error: timeout (dispatch CPU)"*. A single-node precompile cannot initialise 16-rank EP, so it must run on both. |
| `--cuda-graph-bs 128 --max-running-requests 256` on the **decode** role, with `--mem-fraction-static 0.78` | Otherwise all 16 DP ranks each capture the full default batch-size list and CUDA-graph capture OOMs. The symptom is *"scheduler died"* during startup, not an OOM message. |

`recipe/serve-pd.sh` applies the second automatically when `DP_ATTENTION=1`; the precompile step is
a separate command because it only needs running once per host.

## Prerequisites

- **2 nodes** for the colocated recipe, **4** for the PD-disaggregated one (2 prefill + 2 decode),
  `p5.48xlarge` / `p5en.48xlarge` (8×H100/H200) or `p6-b200`/`p6-b300` (8×B200/B300), EFA, Docker
  with GPU support.
- `/dev/infiniband` and `/dev/gdrdrv` present on the host (`ls /dev/gdrdrv`). If `gdrdrv` is
  missing, install GDRCopy on the host — the in-container library cannot create the device node.
- **~640 GB of fast local NVMe per node** for the DeepSeek-R1 FP8 weights, present on *every*
  node (`HF_CACHE_DIR`).
- (Optional) A HuggingFace token. DeepSeek-R1 is **not** gated — the weights download without one;
  a token only helps with rate limits.

## Build

No GPU is needed at build time, but NVSHMEM, DeepEP and Mooncake all compile from source — budget
**40–60 minutes** on a 32-core box and build on something large. The build context is this
directory.

```bash
cp setup/env_vars.example setup/env_vars
"${EDITOR:-vi}" setup/env_vars         # fill in every REPLACE_ME
grep REPLACE_ME setup/env_vars         # should print nothing
source setup/env_vars

setup/build-push.sh                    # local build
setup/build-push.sh --push             # and push to ECR
```

Then verify, on a GPU node:

```bash
recipe/verify-image.sh
```

Build args worth knowing: `SGLANG_BASE` (base tag — must be a `cu130` build, see below),
`DEEPEP_COMMIT` (hard-checked by the setup script — do not change without changing the script's
pin), `NVSHMEM_TAG`, `GDRCOPY_VERSION`, and `TORCH_CUDA_ARCH_LIST`.

### Blackwell (B200 / B300)

`TORCH_CUDA_ARCH_LIST` defaults to `9.0;10.0;10.3`, so one image covers `p5`/`p5en` (`sm_90`),
`p6-b200` (`sm_100`) and `p6-b300` (`sm_103`). Build single-arch to cut compile time:

```bash
docker build --build-arg TORCH_CUDA_ARCH_LIST=10.3 -t sglang-deepep-efa:b300 .    # B300 only
docker build --build-arg TORCH_CUDA_ARCH_LIST=9.0  -t sglang-deepep-efa:hopper .  # Hopper only
```

Three things to know before running this on Blackwell:

- **Keep CUDA 13.** Beyond codegen, CUDA ≤ 12.9 CUPTI returns `CUPTI_ERROR_INVALID_DEVICE` on
  B200/B300, and the `internode`/`low_latency` tests profile their kernels during tuning — so they
  fail *after* the correctness checks pass, which reads as an unrelated bug.
- **`9.0` is not just a subset.** `setup_deepep_efa.sh` enables DeepEP's aggressive PTX
  instructions only when the arch list is exactly `9.0` (they are Hopper-specific). A multi-arch
  image therefore builds Hopper *without* them — pass `TORCH_CUDA_ARCH_LIST=9.0` to reproduce the
  H200 numbers in [`benchmarks/`](./benchmarks/README.md) exactly.
- **A `p6-b300.48xlarge` has 16 EFA NICs, not `p5`'s 32.** Nothing in the launchers needs changing,
  but `IFACE` in `setup/env_vars` is not `enp71s0` there — check `ip -br link`.

**Serving on Blackwell is not validated here.** The DeepEP-EFA kernels are — the same
`567632d` + EFA patch, same NVSHMEM 3.7.0, is measured out to 256 ranks on `p6-b300` in
[`ep-backend-comparison`](../../../../micro-benchmarks/expert-parallelism/ep-backend-comparison/RESULTS.md).
The serving-side gap matters, because **on B300 the published SGLang comparison goes the other
way**: see [Blackwell: expect DeepEP to lose at 2 nodes](./benchmarks/README.md#blackwell-expect-deepep-to-lose-at-2-nodes).

## Smoke-test the EFA transport before loading the model

Confirm the kernels actually move bytes over EFA before spending time on a 640 GB load. On
**both** nodes, same command with `RANK` varying:

```bash
source setup/env_vars
recipe/run-kernel-test.sh intranode    0     # node 0 only, NVLink, no NVSHMEM
recipe/run-kernel-test.sh internode    0     # node 0   } run both, in parallel
recipe/run-kernel-test.sh internode    1     # node 1   } RDMA over EFA
recipe/run-kernel-test.sh low_latency  0
recipe/run-kernel-test.sh low_latency  1
```

Each test prints `passed` for its correctness checks before reporting bandwidth. For the full
kernel matrix (Slurm and EKS launchers, 1–32 nodes, Blackwell too) use
[`deepep-benchmark`](../../../../micro-benchmarks/expert-parallelism/deepep-benchmark) instead —
this script is only a pre-flight check on the serving image.

## Serve

On **both** nodes (rank 0 must be `NODE_0_IP`):

```bash
source setup/env_vars
recipe/serve.sh 0        # on node 0
recipe/serve.sh 1        # on node 1
docker logs -f r1-deepep # wait for "The server is fired up"
```

R1 takes several minutes to load. Then, from node 0:

```bash
curl -s localhost:30000/health && echo OK
```

Swap the MoE backend by restarting with `MOE_BACKEND=baseline` or `MOE_BACKEND=tp`.

## Serve PD-disaggregated (2P2D, 4 nodes)

Two prefill nodes and two decode nodes, each role its own TP16/EP16 group, KV cache flowing
prefill→decode over EFA, one router in front. Fill in the PD block of `setup/env_vars` first.

```bash
source setup/env_vars

# Only for DeepEP + DP_ATTENTION=1 — warm the DeepGEMM cache on BOTH prefill nodes first.
DP_ATTENTION=1 recipe/serve-pd.sh precompile 0
DP_ATTENTION=1 recipe/serve-pd.sh precompile 1

recipe/serve-pd.sh serve prefill 0      # prefill node 0
recipe/serve-pd.sh serve prefill 1      # prefill node 1
recipe/serve-pd.sh serve decode  0      # decode node 0
recipe/serve-pd.sh serve decode  1      # decode node 1

# Wait for all four to print "The server is fired up", then:
recipe/serve-pd.sh router               # on $ROUTER_IP
curl -s localhost:8000/health && echo OK

recipe/serve-pd.sh stop                 # on every node, when done
```

Before trusting any number, confirm the KV path came up on EFA rather than TCP:

```bash
docker logs r1-pd-prefill 2>&1 | grep -E 'EfaTransport|Installing TCP transport'
```

`UCCL=1` benchmarks **UCCL-EP** in either topology, from the second image in this directory,
[`Dockerfile.uccl`](./Dockerfile.uccl):

```bash
docker build -t ucclep-sglang-efa:latest -f Dockerfile.uccl .
```

Same base, EFA and Mooncake stack as `Dockerfile`, with UCCL's `ep/deep_ep_wrapper` in place of
DeepEP/NVSHMEM. The wrapper exposes UCCL under DeepEP's Python API, so `--moe-a2a-backend deepep`
drives it unchanged and the image is the only variable. Hopper-only; for Blackwell or for kernels
without a serving engine see
[`uccl-ep-benchmark`](../../../../micro-benchmarks/expert-parallelism/uccl-ep-benchmark), and for
UCCL under vLLM see [`dsv3-uccl-nixl`](../../vllm/dsv3-uccl-nixl).

Point `IMAGE_URI` at it and set `UCCL=1`: that switches on `--privileged` (UCCL registers GPU memory
through dma-buf/ibverbs, which DeepEP does not need), drops the NVSHMEM environment UCCL has no use
for, drops the 2P2D decode role's `--mem-fraction-static` to 0.70, and makes `DEEPEP_MODE`
mandatory — with `--deepep-mode auto` UCCL's prefill path segfaults at startup. It is worth the
trouble for one reason: **UCCL-EP wins decode**, by 6–37% over DeepEP colocated and, with
DP-attention on the disaggregated decode role, at 4094 tok/s / TPOT 28 ms — the fastest decode
measured here ([`benchmarks/`](./benchmarks/README.md#decode-both-pinned-low_latency)).

## Benchmark

Prefill and decode are swept separately, because the two stages stress the MoE all-to-all very
differently:

```bash
recipe/benchmark.sh prefill      # --random-output-len 1 => TTFT is ~pure prefill
recipe/benchmark.sh decode       # short in, long out => TPOT-dominated
```

Raw JSON lands in `benchmarks/raw/$MOE_BACKEND/`. For the 2P2D deployment use the PD sweeps
instead — same split, driven through the router, with the concurrency pushed much further:

```bash
recipe/benchmark-pd.sh decode        # run decode FIRST; see the script header for why
recipe/benchmark-pd.sh prefill
```

Measured numbers, with caveats to read before quoting anything:

[`benchmarks/README.md`](./benchmarks/README.md) — all H200, in two parts: the colocated 2-node
topology this sample's `serve.sh` launches, and the 4-node 2P2D topology with four MoE backends
(DeepEP / baseline / pure TP / UCCL-EP) × prefill and decode, ± DP-attention.

The short version: **no single backend wins both stages.** DeepEP takes prefill decisively —
161.5k input tok/s at 1K×conc256, 3× everything else, and 21–32% ahead of UCCL-EP at every
colocated point — while on decode at 16 GPUs it is last, and pure TP ≈ baseline lead. UCCL-EP beats
DeepEP on decode in both topologies, reaching 4094 tok/s / TPOT 28 ms with DP-attention.
Because a PD deployment runs the two stages on separate nodes, the candidate configuration these
numbers point to is **DeepEP on prefill, UCCL-EP + DP-attention on decode** — assembled from
per-stage measurements and **not yet run end to end as a single deployment**. Treat the per-stage
results as directional rather than statistically tight: they are single-pass, and the smallest margin
promoted to a conclusion here is about 6%. DeepEP losing decode at this scale is expected rather
than a defect in the EFA port: it is built for large-scale EP where experts are spread thin enough
that every token must cross the network. What this sample demonstrates is that the DeepEP kernels
are **correct and run at IB-class bandwidth over EFA**, and plug into a real R1 server end to end.

## Keeping the vendored script in sync

`setup_deepep_efa.sh` is a verbatim copy of the canonical script. If that one changes, re-copy:

```bash
cp ../../../../micro-benchmarks/expert-parallelism/deepep-benchmark/setup_deepep_efa.sh \
   setup_deepep_efa.sh
```

Docker cannot `COPY` from outside the build context, which is why it is vendored rather than
referenced. Drift should be guarded in CI alongside the other vendored copy — see
[`.github/workflows/deepep-vendor-sync.yml`](../../../../.github/workflows/deepep-vendor-sync.yml).

## Known limitations

- **Launch surface**: validated as raw `docker run` on EC2 across 2 nodes (colocated) and 4 nodes
  (2P2D). No Slurm or Kubernetes launchers are provided here, because none were exercised — for
  cluster-scheduled kernel benchmarks use `deepep-benchmark`, which has both.
- **2P2D input length tops out around 64K.** 128K prefill computes fine but the Mooncake/EFA KV
  transfer times out (`EFA submitSlicesOnPeer: CQ drain wr_depth=256, max=256`); a larger `MC_MAX_WR`
  or chunked KV transfer is the untried fix. 64K at concurrency ≥4 exhausts the 2-node prefill KV
  pool.
- **Serving is validated on Hopper only.** The image builds for Blackwell (`sm_100`/`sm_103`) and
  the DeepEP-EFA kernels are measured on `p6-b300` in `ep-backend-comparison`, but no end-to-end
  serving run on Blackwell is in this repo. Do not assume the H100/H200 backend ranking transfers —
  it does not ([`benchmarks/`](./benchmarks/README.md#blackwell-expect-deepep-to-lose-at-2-nodes)).
- **DeepEP pinned to `567632d`** (pre-EPv2). The setup script hard-checks the tree and its EFA
  patch only applies at that commit. EPv2 restructures the kernels and moves to the NCCL GIN
  backend, which is out of scope.
- **`low_latency` requires CUDA VMM enabled.** Without it the low-latency RDMA-buffer `cudaMemset`
  fails with *"invalid argument"* (`deep_ep.cpp:371`), at init, before any bandwidth is printed.
  `recipe/run-kernel-test.sh` sets VMM per test and `recipe/serve.sh` derives it from `DEEPEP_MODE`,
  so pass the mode rather than setting VMM by hand.

  **The converse — that `normal` requires VMM *off* — did not reproduce when it was tested
  directly**, so do not treat it as a rule. Two-node `internode` on `p5.48xlarge` runs clean with
  VMM left enabled: 60.97 GB/s BF16 dispatch (RDMA) against 61.07 with VMM off, both ranks exit 0,
  no NVSHMEM topology or transport-map error in either log. The scripts still set
  `NVSHMEM_DISABLE_CUDA_VMM=1` for `normal`, now as harmless belt-and-braces rather than a
  requirement — it costs nothing measurable and the original failure was real on some host, just not
  one that has been re-identified. `recipe/serve-pd.sh` does not set it at all, and its
  `normal`-pinned prefill role initialises NVSHMEM with VMM enabled on both H200 and B200 (the
  latter reported by @KeitaW), which is consistent with the p5 measurement. See
  [the matrix](./benchmarks/README.md#the-deepep-mode--cuda-vmm-coupling-measured-both-ways).
