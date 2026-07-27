<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# DeepSeek-R1 on SGLang with DeepEP over EFA

Serve **DeepSeek-R1** (671B, FP8, 256 experts, top-8 routing) with SGLang across two
`p5.48xlarge` / `p5en.48xlarge` nodes, using **DeepEP** MoE dispatch/combine kernels running over
**NVSHMEM's libfabric/EFA transport** — GPU expert-parallel all-to-all on AWS EFA with no
InfiniBand anywhere.

The same image also serves as the head-to-head harness: `recipe/serve.sh` selects between the
DeepEP all-to-all, SGLang's ordinary fused-MoE all-to-all, and pure tensor parallelism, so all
three can be measured on identical hardware and fabric.

| | |
|---|---|
| Base image | `lmsysorg/sglang:v0.5.13.post1-cu130` |
| DeepEP | `deepseek-ai/DeepEP` @ `567632d` + the EFA patch (pre-EPv2, NVSHMEM backend) |
| NVSHMEM | `v3.7.0-0`, built with **only** the libfabric transport (IBRC/IBGDA off) |
| Mooncake | `kvcache-ai/Mooncake` @ main, `-DUSE_EFA=ON` (KV transfer for PD-disaggregation) |
| GPU arch | Hopper `sm_90` (validated on H100 and H200) |

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

### Runtime requirements that are easy to miss

| Setting | Why |
|---|---|
| `--device /dev/gdrdrv` | **In addition to** `/dev/infiniband`. The libfabric/EFA transport uses GDRCopy to register GPU memory; without it internode runs fail with *"GDRCopy support not enabled. Unable to register gpu memory handle info."* |
| `NVSHMEM_REMOTE_TRANSPORT=libfabric` | NVSHMEM is built with **only** the libfabric transport, so the default `ibrc` finds nothing → *"Peer GPU not accessible / building transport map failed"*. |
| `NVSHMEM_LIBFABRIC_PROVIDER=efa` | Select the EFA provider. |
| `NVSHMEM_NETDEVS_POLICY=EXTERNAL_SHARING_PCIE_SWITCH_NIC_EXCLUSIVE` | Gives each GPU exclusive use of the NIC on its own PCIe switch. Recommended for RDMA performance on p5/p5en, which pair multiple EFA NICs across PCIe switches. |
| `NVSHMEM_DISABLE_CUDA_VMM` — **regime-specific** | Set it to `1` for the **normal** dispatch/combine kernels, or NVSHMEM topology / transport-map init fails in-container. **Leave it unset for low-latency kernels**, or the RDMA-buffer `cudaMemset` fails with *"invalid argument"* (`deep_ep.cpp:371`). Because the server uses `--deepep-mode auto` (low-latency on the decode path), `recipe/serve.sh` leaves VMM **enabled**; `recipe/run-kernel-test.sh` sets it per test. |
| `--network host --ipc host --ulimit memlock=-1 --shm-size 32g` | EFA needs host networking, IPC and unlimited locked memory. |

The first four are baked into the image as `ENV`; the launchers re-export them anyway so the
transport config is visible at the launch surface and overridable per run.

## Prerequisites

- **2 nodes**, `p5.48xlarge` or `p5en.48xlarge` (8×H100/H200, EFA), Docker with GPU support.
- `/dev/infiniband` and `/dev/gdrdrv` present on the host (`ls /dev/gdrdrv`). If `gdrdrv` is
  missing, install GDRCopy on the host — the in-container library cannot create the device node.
- **~640 GB of fast local NVMe per node** for the DeepSeek-R1 FP8 weights, present on *both*
  nodes (`HF_CACHE_DIR`).
- A HuggingFace token with DeepSeek-R1 access.

## Build

No GPU is needed at build time, but NVSHMEM, DeepEP and Mooncake all compile from source — budget
**40–60 minutes** on a 32-core box and build on something large. The build context is this
directory.

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

Build args worth knowing: `SGLANG_BASE` (base tag), `DEEPEP_COMMIT` (hard-checked by the setup
script — do not change without changing the script's pin), `NVSHMEM_TAG`, `GDRCOPY_VERSION`, and
`TORCH_CUDA_ARCH_LIST` (`9.0`; this sample is Hopper-only).

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

## Benchmark

Prefill and decode are swept separately, because the two stages stress the MoE all-to-all very
differently:

```bash
recipe/benchmark.sh prefill      # --random-output-len 1 => TTFT is ~pure prefill
recipe/benchmark.sh decode       # short in, long out => TPOT-dominated
```

Raw JSON lands in `benchmarks/raw/$MOE_BACKEND/`.

See [`benchmarks/RESULTS.md`](./benchmarks/RESULTS.md) for measured numbers, and read the caveats
there before quoting anything: **at 2 nodes / 16 GPUs, DeepEP does not beat the ordinary
all-to-all or pure TP.** That is the expected result, not a defect in the EFA port — DeepEP is
built for large-scale EP (experts spread thin across many nodes so every token must cross the
network), and the crossover is not reachable at this scale. What this sample demonstrates is that
the DeepEP kernels are **correct and run at IB-class bandwidth over EFA**, and plug into a real
R1 server end to end.

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

- **Launch surface**: validated as raw `docker run` on EC2 across 2 nodes. No Slurm or Kubernetes
  launchers are provided here, because none were exercised — for cluster-scheduled kernel
  benchmarks use `deepep-benchmark`, which has both.
- **Hopper only** (`sm_90`). The kernel benchmark image covers Blackwell; this one was not tested
  there.
- **DeepEP pinned to `567632d`** (pre-EPv2). The setup script hard-checks the tree and its EFA
  patch only applies at that commit. EPv2 restructures the kernels and moves to the NCCL GIN
  backend, which is out of scope.
- `--deepep-mode auto` leaves CUDA VMM enabled for the whole server, since the decode path needs
  it. If you pin `--deepep-mode normal`, set `NVSHMEM_DISABLE_CUDA_VMM=1` as well.
