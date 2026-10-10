# NCCL EP Benchmark (NCCL GIN CPU proxy over EFA)

[NCCL EP](https://github.com/NVIDIA/nccl-extensions/tree/main/nccl_ep) (`nccl_ep`, part of
[NVIDIA/nccl-extensions](https://github.com/NVIDIA/nccl-extensions)) is NVIDIA's expert-parallel
dispatch/combine library for Mixture-of-Experts (MoE) models. It is built on NCCL's device API, and
its internode traffic goes through **NCCL GIN (GPU-Initiated Networking)**.

This directory runs NVIDIA's own NCCL EP test programs on two `p5en.48xlarge` nodes over AWS EFA,
with NCCL's **GIN CPU proxy** (`NCCL_GIN_TYPE=2`) as the internode transport. No framework is
involved. It provides:

- **[`nccl-ep.Dockerfile`](./nccl-ep.Dockerfile)** and **[`setup_nccl_ep_efa.sh`](./setup_nccl_ep_efa.sh)**:
  an image built from the NGC CUDA 13.0.2 devel base with the EFA installer's user space (which
  bundles the aws-ofi-nccl plugin), one NCCL 2.32.3 from the `nvidia-nccl-cu13` wheel, and NCCL EP
  built from source against that NCCL. The base image, the EFA installer, the NCCL wheel, NCCL EP
  and GDRCopy are pinned by digest, SHA-256 or commit, and the build stops on any mismatch.
- **[`kubernetes/`](./kubernetes/)**: two pods, one per node, each holding all 8 GPUs and all 16 EFA
  devices of its node.
- **[`scripts/`](./scripts/)**: a launcher (`run_nccl_ep_efa.sh up|run|down|all`), in-pod helpers,
  and [`check_run.py`](./scripts/check_run.py), which turns the run's records into pass/fail gates
  in `RESULT.json`.

| Step | Program | Ranks | Checks |
|------|---------|-------|--------|
| Correctness | `ep_test -a ht -L fl` | 16 (2 nodes x 8) | NCCL EP's built-in dispatch and combine oracle on every rank, HIGH_THROUGHPUT algorithm, FLAT layout |
| Timing, only after correctness passes | `ep_bench --algorithm ht --layout flat ... --validate` | 16 | A timing loop that validates its own output at the end |

For each step the launcher also records which transport NCCL selected, from its init log lines, and
the EFA hardware byte counters before and after.

## ⚠️ Version constraints

> - **NCCL >= 2.31.2-1** for GIN on EFA (the [AWS GIN guide](https://github.com/aws/aws-ofi-nccl/blob/68b016a24a4f0e69f6688d0e10ed886657af2d4a/doc/gin-getting-started.md#requirements)).
>   This recipe builds against and runs **NCCL 2.32.3**, from the `nvidia-nccl-cu13==2.32.3` wheel.
> - **aws-ofi-nccl >= 1.21.0** (same guide). EFA installer 1.50.0 bundles 1.21.1.
> - **nccl-extensions at commit `901c4e65`** (NCCL EP 0.2.0). Its only release, v0.1.0, predates the
>   runtime NCCL version check below, and the PyPI `nccl-extensions` 0.1.0 wheel pins NCCL 2.30.7,
>   below the GIN floor.
> - **One NCCL per process.** `ncclEpCreateGroup` accepts a runtime NCCL equal to the one it was
>   built against; a newer runtime only when the build is >= 2.31, or 2.30.5 to 2.31 with
>   `NCCL_DEV_API_JIT` enabled (the default when unset); and fails with `are incompatible` otherwise
>   (`validateNcclRuntimeVersion`, nccl-extensions
>   [`93a50470`](https://github.com/NVIDIA/nccl-extensions/commit/93a50470)). The image holds exactly
>   one `libnccl.so.2`: the base image's own NCCL is removed, and the build checks for a second copy.
> - **16-NIC `p5en.48xlarge`.** On 32-NIC `p5.48xlarge`, released NCCL hits
>   [NVIDIA/nccl#2160](https://github.com/NVIDIA/nccl/issues/2160) ("too many XML nodes" during
>   topology fusion). Its fix (`4ae8e52`) is on NCCL's dev branch and not in `v2.32.3-1`.
> - **HIGH_THROUGHPUT + FLAT** is the validated mode. LOW_LATENCY is not exercised here.

Pinned versions (checked 2026-10-10; the full digests and checksums are in `nccl-ep.Dockerfile`):

| Component | Pin | Notes |
|-----------|-----|-------|
| Base image | `nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:5dc1bca2...` | Provides the `nvcc` that NCCL EP also uses at run time to compile its JIT kernels |
| EFA installer | 1.50.0, tarball SHA-256 `fa6dff85...` | Newest available. Bundles libfabric 2.6.0amzn1.0, aws-ofi-nccl 1.21.1 and Open MPI 4.1.7 |
| aws-ofi-nccl | 1.21.1, from the installer (`/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so`) | Newest release. The installer selects its NGC plugin package on an NGC base |
| NCCL | `nvidia-nccl-cu13==2.32.3` wheel, SHA-256 `14597230...`, unpacked as `NCCL_HOME=/opt/nccl` | Newest release. Headers and `libnccl.so.2` come from the same wheel |
| nccl-extensions | `901c4e65d6c3c8141499902c99838573de43a253` (2026-10-01), NCCL EP 0.2.0 | `main` is ahead, including changes to the HT dispatch paths. Moving the pin means re-validating |
| GDRCopy (user space) | v2.5.2 (`c91ad9f`) | Meets the guide's >= 2.5 floor. v2.6 adds a DMA-BUF fallback that is not validated here. NCCL's proxy does not use GDRCopy by default (below) |
| Open MPI | 4.1.7, from the installer (`/opt/amazon/openmpi`) | Launches the tests over TCP only |

## How the EFA support works

NCCL EP calls NCCL's device API, and NCCL GIN carries the internode traffic. On EFA, aws-ofi-nccl
supports two GIN modes ([AWS GIN guide](https://github.com/aws/aws-ofi-nccl/blob/68b016a24a4f0e69f6688d0e10ed886657af2d4a/doc/gin-getting-started.md)):
a proxy mode, where the GPU queues network operations and a CPU thread issues them, and EFA-GDA,
where the kernels post network operations themselves (`NCCL_GIN_TYPE=5`, which has its own host
requirements). This benchmark uses the proxy mode: NCCL 2.32's host GIN proxy runs over the
aws-ofi-nccl RMA plugin and libfabric's `efa` provider.

The launcher pins the transport rather than relying on NCCL's default selection, and the checker
fails any rank whose log does not show that each setting took effect:

| Variable | Value | Purpose |
|----------|-------|---------|
| `NCCL_GIN_TYPE` | `2` | GIN CPU proxy. Every rank must log `NCCL_GIN_TYPE set by environment to 2` and `GIN Proxy will not be using GDRCopy`. |
| `NCCL_NET_PLUGIN`, `NCCL_GIN_PLUGIN` | `/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so` | The aws-ofi-nccl plugin from the EFA installer, for NET and RMA (`Using network Libfabric`, `RMA/Plugin: Assigned plugin Libfabric to comm`). |
| `FI_PROVIDER`, `FI_EFA_USE_DEVICE_RDMA` | `efa`, `1` | libfabric's EFA provider with device RDMA (`NET/OFI Selected provider is efa`). |
| `NCCL_NVLS_ENABLE` | `0` | NVLink SHARP off. |
| `NCCL_DEBUG`, `NCCL_DEBUG_SUBSYS` | `INFO`, `INIT,ENV,NET` | The init lines the checker reads. |

Ranks started over ssh do not inherit the image `ENV`, so `scripts/pod_tools.sh` passes every
setting to each rank with `mpirun -x`. MPI itself stays on TCP (`pml ob1`, `btl self,vader,tcp`),
so its traffic does not enter the EFA counters the run uses as transport evidence.

The AWS GIN guide lists GDRCopy >= 2.5, runtime and `gdrdrv` kernel module, among its
requirements. NCCL's own GIN proxy initializes GDRCopy only when `NCCL_GDRCOPY_ENABLE=1` (default 0;
NCCL v2.32.3-1 [`src/init.cc:152-162`](https://github.com/NVIDIA/nccl/blob/v2.32.3-1/src/init.cc#L152-L162))
and otherwise logs `GIN Proxy will not be using GDRCopy`
([`src/gin/gin_host_proxy.cc:471`](https://github.com/NVIDIA/nccl/blob/v2.32.3-1/src/gin/gin_host_proxy.cc#L471)).
The pods therefore run unprivileged and without `/dev/gdrdrv`. The image still ships the GDRCopy
2.5.2 user space.

## Prerequisites

- **Cluster:** an EKS cluster with two `p5en.48xlarge` nodes (8x H200, 16 EFA each) that have no
  GPUs allocated, and the [NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin) and
  [AWS EFA device plugin](https://github.com/aws/eks-charts/tree/master/stable/aws-efa-k8s-device-plugin).
  The launcher refuses a node where live pods request GPUs; it never preempts.
- **Hosts:** the EFA kernel driver (the image ships only the user space; run the EFA installer on
  the node to install the driver) and an NVIDIA driver that supports CUDA 13.0. The `gdrdrv` module
  is not needed (see above).
- **Namespace:** an existing namespace. The launcher never creates or deletes one.
- **Network:** TCP between the two nodes within the node CIDR, on port 2222 (the per-run sshd) and
  on Open MPI's dynamic ports. EKS node security groups usually allow node-to-node traffic. The run
  adds no security-group rules and opens nothing to the internet.
- **Workstation:** `kubectl`, `envsubst` (gettext), `python3`, and Docker with BuildKit plus a
  registry you own if you build the image.

## Build the image

Push the image to a registry you own, for example Amazon ECR:

```bash
export AWS_REGION=$(aws ec2 describe-availability-zones --output text --query 'AvailabilityZones[0].[RegionName]')
export ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export REGISTRY=${ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com
export IMAGE_URI=${REGISTRY}/nccl-ep-efa:efa1.50.0-nccl2.32.3-901c4e65
aws ecr create-repository --repository-name nccl-ep-efa --region ${AWS_REGION} 2>/dev/null || true
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${REGISTRY}
```

The build context is this directory:

```bash
cd micro-benchmarks/expert-parallelism/nccl-ep-benchmark
DOCKER_BUILDKIT=1 docker build --progress=plain -f nccl-ep.Dockerfile -t ${IMAGE_URI} .
docker push ${IMAGE_URI}
```

`setup_nccl_ep_efa.sh` performs the build in five stages, one Docker layer each: `deps`, `gdrcopy`,
`efa`, `nccl` and `nccl-ep`. Each stage fails closed. The build stops on:

- a wrong SHA-256, commit or version string;
- a second `libnccl` file in the image;
- a test binary that does not load `libnccl_ep.so.0` and `libnccl.so.2`;
- a library that does not resolve through the runtime library path the launcher passes to the ranks;
- a `libnccl_ep` without the runtime NCCL version check;
- a header probe that does not print `NCCL_VERSION_CODE` 23203.

The build identity is written to `/opt/nccl-ep/BUILD-RECORD.txt`: the nccl-extensions commit, the
wheel SHA-256, the header version code and the SHA-256 of `libnccl.so.2`, `libnccl_ep.so.0.2.0`,
`ep_test`, `ep_bench` and the aws-ofi-nccl plugin. `libnccl_ep` and the test binaries are not
byte-reproducible across builds; the commit, the wheel and the header code are the identity.

**To skip building your own image:** set `IMAGE_URI` to the plain NGC base
(`nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6`)
and `BUILD_IN_POD=1`. The launcher then runs the same `setup_nccl_ep_efa.sh all` inside both pods
before the test, which needs outbound access to the Ubuntu archive, PyPI, GitHub and the EFA
installer site.

## Run

```bash
cd kubernetes
cp env_vars.example env_vars    # set KUBE_CONTEXT, NAMESPACE, IMAGE_URI, NODE_A, NODE_B
source env_vars
kubectl --context "$KUBE_CONTEXT" create namespace "$NAMESPACE"   # only if it does not exist yet
../scripts/run_nccl_ep_efa.sh all   # up -> run -> down
```

`nccl-ep-2node.yaml` holds only the two pods, so applying it never changes the namespace, and
`kubectl delete -f` on a rendered copy removes only the two pods. `up` checks that the namespace
exists and stops if it cannot read it.

The steps can also be run one at a time with `up`, `run` and `down`. Every run writes a new
`results/nccl-ep-efa-<UTC>/` directory (set `OUT_ROOT` to put it elsewhere) and never reuses one. The
launcher exits 0 only if every gate in `RESULT.json` passes.

What `run` does, in order:

1. **Preflight**, failing closed: two Running hostNetwork pods on different nodes; 8 GPUs and 16 EFA
   devices per pod; no GPU compute processes; `fi_info -p efa` finds the provider; exactly one
   `libnccl` file in each pod; matching build records in both pods.
2. **Per-run sshd** in the second pod: bound to that node's address on port 2222, key-only, accepting
   only the first pod's address, with a key generated for this run.
3. **Correctness:** EFA counters, then `mpirun` with 16 ranks of `ep_test -a ht -L fl`, then EFA
   counters again, then a sweep for leftover processes by exact PID, then a copy of every rank's
   output.
4. **Timing**, only if every correctness gate passed: the same sequence with `ep_bench`
   (`EP_BENCH_ARGS` overrides its arguments).
5. **Close out:** GPU readback on both pods, and the sshd stopped with its keys removed.
6. **`check_run.py summarize`** writes `RESULT.json`.

`all` then runs **`down`**, which deletes both pods and confirms they are gone, also after a failed
run. Set `KEEP_PODS=1` to keep them for inspection. If `up` itself stops, for example because the
pods did not become Ready within 30 minutes, `all` exits without `down`: run `down` yourself.

**Why two pods instead of an MPIJob.** The sibling `deepep-v2-benchmark` launches through the
Kubeflow MPI Operator. This one does not need an operator: the launcher creates two plain pods, starts the
per-run sshd in the second, and runs `mpirun` from the first. Keeping the whole run in one launcher
lets it refuse busy nodes, read the EFA counters around each step, and confirm the teardown.

## Expected output

From a passing run on two `p5en.48xlarge` nodes (abridged; host and PID prefixes removed).

Every rank, in `results/<run>/correctness/<pod>/mpi/1/rank.NN/stdout`:

```text
NCCL INFO NCCL version 2.32.3+cuda13.4
NCCL INFO NET/OFI Selected provider is efa, fabric is efa-direct (found 16 nics)
NCCL INFO RMA/Plugin: Assigned plugin Libfabric to comm
NCCL INFO NCCL_GIN_TYPE set by environment to 2.
NCCL INFO GIN/Plugin: Skipping plugin Libfabric_GDAKI index 0 type 5: NCCL_GIN_TYPE=2 requested
NCCL INFO Using network Libfabric
Rank 0: Testing ncclEpCreateGroup with algorithm: HIGH_THROUGHPUT
NCCL INFO devCommCreate: creating 33 contexts: 1 GIN connections with 33 contexts each (33 contexts total requested)
NCCL INFO GIN Proxy will not be using GDRCopy
Rank 0: Verifying recv_topk_weights and recv_topk_idx (HT+FLAT, 2D)
Rank 0: HIGH_THROUGHPUT Dispatch flow passed successfully
Rank 0: Combine verification PASSED! All 50 tokens with 7168 elements each correctly combined
[MPI Rank 0] Success
```

Rank 0 of the timing step prints the configuration `ep_bench` parsed
([`ep_bench.cu:5789-5842`](https://github.com/NVIDIA/nccl-extensions/blob/901c4e65d6c3c8141499902c99838573de43a253/nccl_ep/ep_bench.cu#L5789-L5842)),
then the summary. The timing gate reads what ran from this block, not from the arguments, because
`getopt` keeps the last value of a repeated flag:

```text
=== NCCL EP Performance Benchmark ===
Configuration:
  Algorithm:       HIGH_THROUGHPUT
  Layout:          flat
  Ranks:           16
  Shared SMs:      auto
  Dispatch SMs:    inherit
  Tokens:          4096
  Hidden:          7168
  Top-k:           8
  Experts:         256 (local: 16)
  Dispatch recipe: none
  Combine recipe:  none
  Validate mode:   enabled
...
=== Summary (High Throughput recipe none, across 16 ranks) ===
Dispatch:    total=... us (min=..., max=...)
Combine:     total=... us (min=..., max=...)
Total (D+C): avg=... us, min=... us, max=... us
...
Global validation: Dispatch=PASSED, Combine=PASSED
```

CUPTI errors on hosts that restrict GPU profiling are printed but not fatal; the checker reads only
the host-observed lines.

- **EFA counters:** a positive delta in the summed RDMA byte counters (`rdma_read_bytes`,
  `rdma_read_resp_bytes`, `rdma_write_bytes`, `rdma_write_recv_bytes`) on both nodes for each step,
  about 4.7 MB per node for `ep_test` and about 117 GB per node for the default `ep_bench` arguments.
- **`RESULT.json`:** `"verdict": "PASS"`, with the gates `preflight`, `correctness_and_transport`,
  `timing` and `lifecycle` all `PASS`.

## Known limitations

1. **GIN proxy only.** EFA-GDA (`NCCL_GIN_TYPE=5`) is a separate path with its own requirements (EFA
   driver >= 3.3.0, `PeerMappingOverride` in the NVIDIA driver, signal limits; see the AWS GIN
   guide) and is not covered here.
2. **HIGH_THROUGHPUT + FLAT only.** LOW_LATENCY and the other layouts are not exercised.
3. **HT needs 1 + 2 x dispatch SMs GIN contexts per rank.** NCCL EP refuses fewer and, with the
   automatic setting `ep_test` and `ep_bench` use, creates exactly that many: 33 at the default of 16
   dispatch SMs (nccl-extensions `901c4e65`: `nccl_ep/nccl_ep.cc:1849-1856` and `2319-2323`,
   `nccl_ep/device/ht_ep_configs.cuh:17,26,58`). The checker requires at least that many on every
   rank. For `ep_test`, which has no SM option, the budget is 16. For `ep_bench` it is the
   `Dispatch SMs` value it prints, else its `Shared SMs`, else 16, the order in which the library
   resolves them (`nccl_ep/nccl_ep.cc:2175-2206`); a run with `--max-num-sms 8` needs 17. The
   environment overrides `NCCL_EP_DISPATCH_SMS` and `NCCL_EP_COMM_SMS` take precedence in the library
   (`nccl_ep/nccl_ep.cc:2213-2231`) but are not printed. The launcher does not set them; a run that
   lowers the budget through them fails this gate.
4. **GDRCopy.** `NCCL_GDRCOPY_ENABLE=1` is not tested here. If you enable it, the AWS GIN guide's
   floor applies: GDRCopy >= 2.5 for both the runtime and the host `gdrdrv` module.
5. **32-NIC `p5.48xlarge`:** see the version constraints. That 16-NIC hosts are unaffected by
   NVIDIA/nccl#2160 is an inference, not a measurement.
6. **Frameworks.** A framework process (for example PyTorch for Megatron-LM or NeMo RL) that uses
   this library must load the same `libnccl.so.2`. That is not covered here.
7. **Scale and hardware.** Validated shape: two `p5en.48xlarge` nodes, 8 ranks per node, H200, an `sm_90`
   build (`NVCC_GENCODE`). `INSTANCE_TYPE`, `GPU_PER_NODE` and `EFA_PER_NODE` in `env_vars` select other
   node types, but only this shape is validated; more nodes and Blackwell are not covered.
8. **Timing is not a performance comparison.** `ep_bench` reports host-observed averages from one
   launch. Repeat on fresh pods before comparing numbers.

## Troubleshooting

- **No EFA counters in the pod.** Some container views of `/sys` omit `hw_counters`. Mount the
  host's `/sys` read-only at `/hostsys`; `efa_counters.sh` uses it automatically.
- **The first run is slow.** NCCL EP compiles kernels with `nvcc` on first use and caches them in
  `/work/nccl-ep-run/jit-cache` for the life of the pod.
- **ssh or mpirun fails.** Port 2222 must be free on the node, and the node CIDR must allow TCP
  between the two nodes. `MPI_CIDR` defaults to the /16 of the first pod's address; override it if
  your VPC differs.
- **`NCCL EP requires NCCL Device API support, but Device API is not supported`.** No GIN backend
  came up. Check the rank log for `NCCL_GIN_TYPE set by environment to 2` and
  `RMA/Plugin: Assigned plugin Libfabric to comm`, and check that `NCCL_NET_PLUGIN` points at
  `/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so`.
- **Pods older than 6 hours.** The launcher warns when a pod is more than 6 hours old. Rerun on fresh
  pods (`down`, then `up`) before trusting any timing.
