<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# DeepSeek-R1 on vLLM with DeepEP over EFA (PD-disaggregated)

Serve **DeepSeek-R1** (671B, FP8, 256 experts, top-8 routing) with vLLM across four
`p5.48xlarge` / `p5en.48xlarge` nodes in a **2P2D prefill/decode-disaggregated** deployment, using
**DeepEP** MoE dispatch/combine kernels over **NVSHMEM's libfabric/EFA transport** and moving KV
cache prefill→decode over EFA via **NixlConnector** — GPU expert-parallel all-to-all plus
disaggregated KV on AWS EFA, with no InfiniBand anywhere.

`recipe/serve-pd.sh` selects between the DeepEP all-to-all and vLLM's ordinary all-to-all, so both
can be measured on identical hardware and fabric.

| | |
|---|---|
| Base image | `vllm/vllm-openai:v0.23.0` |
| DeepEP | `deepseek-ai/DeepEP` @ `567632d` + the EFA patch (pre-EPv2, NVSHMEM backend) |
| NVSHMEM | `v3.7.0-0`, built with **only** the libfabric transport (IBRC/IBGDA off) |
| KV transport | NIXL `LIBFABRIC` plugin (ships in the base image) |
| Mooncake | `kvcache-ai/Mooncake` @ main, `-DUSE_EFA=ON` (alternative KV connector) |
| GPU arch | Hopper `sm_90` (validated on H100 and H200) |

For the same model and fabric served by **SGLang** — including a colocated (non-PD) 2-node
configuration — see the sibling sample
[`3.test_cases/pytorch/sglang/dsr1-deepep-efa`](../../sglang/dsr1-deepep-efa).

## How DeepEP gets onto EFA

DeepEP's internode kernels were written for **IBGDA** (GPU-initiated RDMA over InfiniBand), which
EFA does not provide. [`setup_deepep_efa.sh`](./setup_deepep_efa.sh) rewrites those paths onto
NVSHMEM host-proxy QP APIs. That script is a **vendored copy** of the canonical one in
[`micro-benchmarks/expert-parallelism/deepep-benchmark`](../../../../micro-benchmarks/expert-parallelism/deepep-benchmark) —
see that README for the patch rationale and the kernel-level benchmarks. This directory covers only
what it takes to get that build working **inside vLLM**, which is where the non-obvious failure
modes are.

### The build traps

**1. The vLLM base ships a slim CUDA toolkit.** NVSHMEM and DeepEP include `nvml.h`, `cusparse.h`
and `nvtx3/`, none of which are present. The Dockerfile fills the gaps from two sources —
`cuda-nvml-dev-13-0` from the CUDA apt repo already configured in the base, and the `nvidia-cu13`
pip package's include dir for the rest — but **symlinks only the missing headers**. Putting the
whole pip include dir on `CPATH` shadows the toolkit's own `cuda_runtime.h` / `crt/host_runtime.h`
and breaks nvcc's generated stubs with `'__cudaLaunch' was not declared`.

**2. `libcuda.so` lives only in `stubs/`.** DeepEP's final link passes `-lcuda` against
`-L/usr/local/cuda/lib64`, and this base keeps the stub only in `lib64/stubs`, which is not a
default linker search path → `/usr/bin/ld: cannot find -lcuda`. The Dockerfile symlinks it into
`lib64` for link time, and `libcuda.so.1` inside `stubs` so the freshly built extension can be
imported for the assert below on a GPU-less build host. Neither symlink is on the image's runtime
`LD_LIBRARY_PATH`, where a stub would shadow the real driver.

**3. A stock `deep_ep` in the base image silently shadows the EFA build.** The vLLM image ships its
own `deep_ep` in `/usr/local/lib/python3.12/dist-packages`, built for the standard NVSHMEM
**IBGDA** transport. That directory sits *ahead* of our build on `sys.path`, so leaving it means
`import deep_ep` resolves to the IBGDA build: the server starts, the logs say DeepEP, and **you
believe you are running on EFA while you are not**. The Dockerfile uninstalls it with the *system*
pip before building, then **asserts** `deep_ep` resolves under `/opt/deepep-venv` — so this fails
the build rather than shipping a misleading image.

> Unlike the SGLang base, this base is *not* PEP 668 `EXTERNALLY-MANAGED`, so the venv is not
> strictly required here. It is used anyway because it puts the EFA build at a *distinguishable
> path*, which is what makes that assert meaningful. A `.pth` in the system `dist-packages` exposes
> it to the default `python3` that vLLM and Ray run under.

**4. Two NVSHMEM versions collide.** The base image's torch pulls pip `nvidia-nvshmem-cu13`
(**v3.4.5**) as a dependency of `libtorch_nvshmem.so`. Its `libnvshmem_host.so.3` loads *ahead* of
ours, but DeepEP's device cubin is compiled against **v3.7.0** — every internode and low-latency
run then aborts at init with *"NVSHMEM device library version does not match host library
version"*. The Dockerfile overwrites the pip copy with the v3.7.0 build so the single soname
resolves to v3.7.0 for both consumers (replacing rather than uninstalling keeps `torch_nvshmem`
loadable). The launchers also `LD_PRELOAD` it as a belt-and-braces measure.

`recipe/verify-image.sh` checks traps 2–4 in one shot, plus the NIXL LIBFABRIC plugin.

### The serving traps

**5. Multi-node TP requires Ray.** `vllm serve --nnodes/--node-rank` — the launch shape the SGLang
sample uses — fails with `collective_rpc should not be called on follower node`. Each role must be
a **Ray cluster**: `ray start --head` on one node, `ray start --address=…` on the other, then a
single `vllm serve --tensor-parallel-size 16 --distributed-executor-backend ray` on the head, which
Ray places across both nodes. `recipe/serve-pd.sh` has a `raystart` stage for exactly this.

**6. Restarting an engine inside a live Ray cluster leaks the GPU placement group.** After a
`pkill`, the next `vllm serve` cannot obtain 16 GPUs and hangs. **Always tear the role's Ray
cluster down and recreate it when switching backends** — `serve-pd.sh stop` then `raystart`.

**7. JSON-valued flags must not cross a nested shell.** `--kv-transfer-config` and
`--compilation-config` lose their quotes through `docker exec bash -c`, and vLLM rejects them with
`Invalid JSON: key must be a string`. `serve-pd.sh` writes the launch line into a file inside the
container and runs that. (The original workaround was to drop `--compilation-config` and use
`--enforce-eager` on decode — which is why the decode numbers in `benchmarks/RESULTS.md` are not a
fair vLLM result. See the caveats there.)

### Runtime requirements that are easy to miss

| Setting | Why |
|---|---|
| `--device /dev/gdrdrv` | **In addition to** `/dev/infiniband`. The libfabric/EFA transport uses GDRCopy to register GPU memory; without it internode runs fail with *"GDRCopy support not enabled. Unable to register gpu memory handle info."* |
| `NVSHMEM_REMOTE_TRANSPORT=libfabric` | NVSHMEM is built with **only** the libfabric transport, so the default `ibrc` finds nothing → *"Peer GPU not accessible / building transport map failed"*. |
| `NVSHMEM_LIBFABRIC_PROVIDER=efa` | Select the EFA provider. |
| `NVSHMEM_NETDEVS_POLICY=EXTERNAL_SHARING_PCIE_SWITCH_NIC_EXCLUSIVE` | Gives each GPU exclusive use of the NIC on its own PCIe switch. Worth +8–12% on p5en (16 NICs); a **no-op on p5** (32 NICs). |
| `NVSHMEM_DISABLE_CUDA_VMM` — **regime-specific** | Set it to `1` for the **normal** dispatch/combine kernels, or NVSHMEM topology / transport-map init fails in-container. **Leave it unset for low-latency kernels**, or the RDMA-buffer `cudaMemset` fails with *"invalid argument"* (`deep_ep.cpp:371`). Because the decode role uses `deepep_low_latency`, `recipe/serve-pd.sh` leaves VMM **enabled**; `recipe/run-kernel-test.sh` sets it per test. |
| `VLLM_NIXL_SIDE_CHANNEL_HOST` / `_PORT` | NixlConnector's out-of-band handshake. Must be the role's **head IP**, and the two roles must use different ports if they ever share a host. |
| A persistent `/root/.cache` mount | vLLM's DeepGEMM warmup with DeepEP is ~1664 kernels / **~15 min per role** and has no pre-cache path. Mounting the cache pays it once per host. |
| `--network host --ipc host --ulimit memlock=-1 --shm-size 32g` | EFA needs host networking, IPC and unlimited locked memory. |

The NVSHMEM/EFA settings are baked into the image as `ENV`; the launchers re-export them anyway so
the transport config is visible at the launch surface and overridable per run.

## Prerequisites

- **4 nodes**, `p5.48xlarge` or `p5en.48xlarge` (8×H100/H200, EFA), Docker with GPU support, all in
  the same VPC/placement group. (2 nodes are enough for `recipe/run-kernel-test.sh`.)
- `/dev/infiniband` and `/dev/gdrdrv` present on the host (`ls /dev/gdrdrv`). If `gdrdrv` is
  missing, install GDRCopy on the host — the in-container library cannot create the device node.
- **~640 GB of fast local NVMe per node** for the DeepSeek-R1 FP8 weights, **pre-staged on all
  four nodes** at `$MODEL_DIR`. vLLM is pointed at a local directory rather than a HF repo id here
  precisely so a 4-node cluster does not race four concurrent 640 GB downloads:
  ```bash
  hf download deepseek-ai/DeepSeek-R1 --local-dir /opt/dlami/nvme/DeepSeek-R1
  ```
- A HuggingFace token with DeepSeek-R1 access.

## Build

No GPU is needed at build time, but NVSHMEM, DeepEP and Mooncake all compile from source — budget
**40–60 minutes** on a 32-core box. The build context is this directory.

```bash
cp setup/env_vars.example setup/env_vars
$EDITOR setup/env_vars                 # fill in every REPLACE_ME
grep REPLACE_ME setup/env_vars         # should print nothing
source setup/env_vars

setup/build-push.sh                    # local build
setup/build-push.sh --push             # and push to ECR
```

Then verify, on a GPU node:

```bash
recipe/verify-image.sh
```

Build args worth knowing: `VLLM_BASE` (base tag), `DEEPEP_COMMIT` (hard-checked by the setup
script — do not change without changing the script's pin), `NVSHMEM_TAG`, `GDRCOPY_VERSION`,
`VLLM_ROUTER_VERSION`, and `TORCH_CUDA_ARCH_LIST` (`9.0`; this sample is Hopper-only).

## Smoke-test the EFA transport before loading the model

Confirm the kernels actually move bytes over EFA before spending time on a 640 GB load and a
15-minute DeepGEMM warmup. On **both** nodes of a role, same command with `RANK` varying:

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

## Serve (2P2D)

Four stages. Stage 1 runs on all four nodes, stage 2 on the two role heads, stage 3 on the router
host:

```bash
source setup/env_vars

# 1. Start the Ray container on each node (role + head|worker):
recipe/serve-pd.sh raystart prefill head       # on $PREFILL_HEAD_IP
recipe/serve-pd.sh raystart prefill worker     # on $PREFILL_WORKER_IP
recipe/serve-pd.sh raystart decode  head       # on $DECODE_HEAD_IP
recipe/serve-pd.sh raystart decode  worker     # on $DECODE_WORKER_IP

# Confirm each role's Ray cluster really sees 16 GPUs before serving:
docker exec r1-vllm-prefill ray status

# 2. Launch one engine per role, on the role HEAD only:
recipe/serve-pd.sh serve prefill                # on $PREFILL_HEAD_IP
recipe/serve-pd.sh serve decode                 # on $DECODE_HEAD_IP
docker exec r1-vllm-prefill tail -f /serve.log  # wait for "Application startup complete"

# 3. Front both roles with the router:
recipe/serve-pd.sh router                       # on $ROUTER_HOST
```

R1 takes several minutes to load, plus ~15 min of DeepGEMM warmup per role the first time on a
given host. Then, from the router host:

```bash
curl -s localhost:$ROUTER_PORT/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-ai/DeepSeek-R1","prompt":"The capital of France is","max_tokens":16}'
```

To switch backends, tear down and rebuild the Ray clusters (trap 6 above — this is not optional):

```bash
recipe/serve-pd.sh stop                 # on every node
MOE_BACKEND=baseline recipe/serve-pd.sh raystart prefill head    # ... and so on
```

## Benchmark

Prefill and decode are swept separately, because the two stages stress the MoE all-to-all very
differently. **Run `decode` first** on freshly started engines — the prefill sweep's larger batches
can destabilise a 2-node TP16 prefill role, and a wedged prefill role costs you the decode numbers
too.

```bash
recipe/benchmark.sh decode       # short in, out=128 => TPOT-dominated
recipe/benchmark.sh prefill      # out=1 => TTFT is ~pure prefill
```

Raw JSON and logs land in `benchmarks/raw/$MOE_BACKEND/`. `BENCH_TOOL=vllm` (default) uses
`vllm bench serve` from this image; `BENCH_TOOL=sglang` uses `sglang.bench_serving`, which is what
produced the numbers in `benchmarks/RESULTS.md`. Either way, **do not pass `--pd-separated`** —
that flag is `sglang_router`'s PD protocol and `vllm-router` does not implement it.

See [`benchmarks/RESULTS.md`](./benchmarks/RESULTS.md) for measured numbers, and read the caveats
there first: the decode numbers were taken with **CUDA graphs off** and are a slow lower bound, not
a fair vLLM result, and at 2 nodes per role DeepEP does not beat the ordinary all-to-all anyway.
What this sample demonstrates is that DeepEP-over-EFA plus disaggregated KV over EFA works end to
end on a real R1 deployment.

## Keeping the vendored script in sync

`setup_deepep_efa.sh` is a verbatim copy of the canonical script. If that one changes, re-copy:

```bash
cp ../../../../micro-benchmarks/expert-parallelism/deepep-benchmark/setup_deepep_efa.sh \
   setup_deepep_efa.sh
```

Docker cannot `COPY` from outside the build context, which is why it is vendored rather than
referenced. Drift is guarded in CI —
see [`.github/workflows/deepep-vendor-sync.yml`](../../../../.github/workflows/deepep-vendor-sync.yml).

## Known limitations

- **Decode numbers are not a fair vLLM result.** They were measured with `--enforce-eager` on the
  decode role. `recipe/serve-pd.sh` ships the correct `FULL_DECODE_ONLY` cudagraph config, but the
  re-run under it was not completed. See `benchmarks/RESULTS.md` caveat 2.
- **DeepEP prefill on the out=1 PD path is unstable**: repeated `EngineDeadError`
  (`sample_tokens timed out`). The baseline arm completes. Unresolved.
- **Launch surface**: validated as raw `docker run` + Ray on EC2 across 4 nodes. No Slurm or
  Kubernetes launchers are provided, because none were exercised — for cluster-scheduled kernel
  benchmarks use `deepep-benchmark`, which has both.
- **Hopper only** (`sm_90`). The kernel benchmark image covers Blackwell; this one was not tested
  there.
- **DeepEP pinned to `567632d`** (pre-EPv2). The setup script hard-checks the tree and its EFA
  patch only applies at that commit. EPv2 restructures the kernels and moves to the NCCL GIN
  backend, which is out of scope.
- **No colocated (non-PD) recipe.** vLLM here is PD-disaggregated only; for a colocated 2-node
  comparison use the SGLang sample.
