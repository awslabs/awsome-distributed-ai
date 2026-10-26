# NCCL EP Benchmark (NCCL GIN CPU proxy over EFA)

[NCCL EP](https://github.com/NVIDIA/nccl-extensions/tree/main/nccl_ep) (`nccl_ep`, part of
[NVIDIA/nccl-extensions](https://github.com/NVIDIA/nccl-extensions)) is NVIDIA's expert-parallel
dispatch/combine library for Mixture-of-Experts (MoE) models. It is built on NCCL's device API, and
its internode traffic goes through **NCCL GIN (GPU-Initiated Networking)**.

This directory runs NVIDIA's own NCCL EP test programs on two nodes over AWS EFA, with NCCL's
**GIN CPU proxy** (`NCCL_GIN_TYPE=2`) as the internode transport: the GPU writes each request into a
queue in host memory, and a CPU thread inside NCCL posts it to the NIC through the aws-ofi-nccl plugin
and libfabric's `efa` provider. No framework is involved.

- [`nccl-ep.Dockerfile`](./nccl-ep.Dockerfile): the image.
- [`kubernetes/nccl-ep-mpijob.yaml`](./kubernetes/nccl-ep-mpijob.yaml): one MPIJob, two workers, one rank per GPU.
- [`kubernetes/env_vars.example`](./kubernetes/env_vars.example): the variables the manifest uses.

| Step | Program | Checks |
|------|---------|--------|
| Correctness | `ep_test -a ht -L fl` | NCCL EP's built-in dispatch and combine check on every rank, HIGH_THROUGHPUT algorithm, FLAT layout |
| Timing, only if `ep_test` exits 0 | `ep_bench --algorithm ht --layout flat --tokens 4096 --hidden 7168 --top-k 8 --experts 256 --validate` | A timing loop that validates its own output at the end |

Both programs exit non-zero when a result is wrong, so the job's exit status is the result.

The AWS GIN guide lists GDRCopy >= 2.5, runtime and `gdrdrv` kernel module, among its requirements, and
they apply in proxy mode too: NCCL's proxy calls aws-ofi-nccl's GIN initialization, which loads
`libgdrapi.so`, opens a GDRCopy handle and fails without them. NCCL's own proxy copies through GDRCopy
only when `NCCL_GDRCOPY_ENABLE=1` (default 0), which is why the log says
`GIN Proxy will not be using GDRCopy`.

## ⚠️ Version constraints

> - **NCCL >= 2.31.2** and **aws-ofi-nccl >= 1.21.0** for GIN on EFA (the
>   [AWS GIN guide](https://github.com/aws/aws-ofi-nccl/blob/68b016a24a4f0e69f6688d0e10ed886657af2d4a/doc/gin-getting-started.md)).
> - **16-NIC `p5en.48xlarge`.** On 32-NIC `p5.48xlarge`, NCCL hits
>   [NVIDIA/nccl#2160](https://github.com/NVIDIA/nccl/issues/2160) (the topology XML node limit during
>   intra-node fusion). Its fix (`4ae8e52`) is on NCCL's dev branch, not in `v2.32.3-1`.

| Component | Pin | Notes |
|-----------|-----|-------|
| Base image | `nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu24.04` | Also provides the `nvcc` that NCCL EP uses at run time to compile its JIT kernels |
| NCCL | `libnccl2` and `libnccl-dev` `2.32.3-1+cuda13.4` | From the NVIDIA CUDA apt repository, which the base image configures. Replaces the base image's held NCCL 2.28.3, so the image has one NCCL, the one NCCL EP is built against |
| EFA installer | 1.50.0 | Installs the user space only: libfabric 2.6.0amzn1.0, aws-ofi-nccl 1.21.1 and Open MPI 4.1.7 |
| aws-ofi-nccl | 1.21.1, from the EFA installer | `/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so`, the NET and GIN plugin |
| nccl-extensions | `901c4e65d6c3c8141499902c99838573de43a253` | NCCL EP 0.2.0, which no release contains yet; the PyPI `nccl-extensions` wheel pins NCCL 2.30.7, below the 2.31.2 minimum, and ships no test programs. Moving the pin means re-validating |
| GDRCopy | `v2.5.2` (tag) | User space; the host supplies `gdrdrv`. v2.6 adds a DMA-BUF fallback that is not validated here |
| Open MPI | 4.1.7, from the EFA installer | Launches the ranks; MPI traffic stays on TCP |

## Build the image

The Dockerfile copies nothing from the build context. Build it and push it to a registry you own:

```bash
cd micro-benchmarks/expert-parallelism/nccl-ep-benchmark
export IMAGE_URI=<your-registry>/nccl-ep-efa:<tag>
docker build -f nccl-ep.Dockerfile -t ${IMAGE_URI} .
docker push ${IMAGE_URI}
```

`NVCC_GENCODE` gives the image kernels for sm_90, sm_100 and sm_103 (`p5`/`p5en`, `p6-b200`, `p6-b300`);
only `p5en.48xlarge` was run with it. On a GPU whose architecture is missing, the kernels return zeroed
buffers without a CUDA error, and `ep_test` catches it.

## Run

Prerequisites:

- An EKS cluster with the [NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin), the
  [AWS EFA device plugin](https://github.com/aws/eks-charts/tree/master/stable/aws-efa-k8s-device-plugin)
  and the [Kubeflow MPI Operator](https://github.com/kubeflow/mpi-operator) (`kubeflow.org/v2beta1`; this
  recipe was run with 0.6.0), for example installed as in
  [`deepep-v2-benchmark/kubernetes`](../deepep-v2-benchmark/kubernetes/README.md#prerequisites).
- Two nodes of one instance type in one Availability Zone, with none of their GPUs or EFA devices
  allocated: each worker requests all of them. The pods tolerate the `nvidia.com/gpu`, `workload=bench`
  and `capacity-reservation` taints; for other taints, edit the two `tolerations` blocks.
- On those nodes: the EFA kernel driver (the image has only the user space), an NVIDIA driver that
  supports CUDA 13.0, and the `gdrdrv` module exposed to the pods. The workers set
  `NVIDIA_GDRCOPY=enabled`; with the NVIDIA device plugin, install it with `gdrcopyEnabled=true`.
- Nodes that can pull `IMAGE_URI`: through the node role for Amazon ECR, or with an `imagePullSecret` on
  the namespace's default service account.
- An existing namespace; the manifest does not create one.

Render only the listed variables, so the launcher script's own shell expansions survive. The render
needs GNU `envsubst` (package `gettext-base`); other `envsubst` implementations ignore the variable list
and blank the launcher script's shell variables.

```bash
cd kubernetes
cp env_vars.example env_vars   # then edit env_vars
source env_vars
envsubst '$NAMESPACE $IMAGE_URI $INSTANCE_TYPE $NUM_NODES $GPU_PER_NODE $EFA_PER_NODE' \
  < nccl-ep-mpijob.yaml | kubectl --context "$KUBE_CONTEXT" apply -f -
```

The operator creates the launcher pod (`nccl-ep-efa-launcher-<suffix>`) together with the workers
(`launcherCreationPolicy: AtStartup`, the default); its script waits until each worker's sshd accepts
connections. Watch for it (Ctrl-C when it is Running), then follow its log:

```bash
kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE" get pods -l training.kubeflow.org/job-name=nccl-ep-efa -w
```

```bash
kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE" logs -f --tail=-1 \
  -l training.kubeflow.org/job-name=nccl-ep-efa,training.kubeflow.org/job-role=launcher
```

The exit status is the result: the launcher pod ends `Completed` when `ep_test` and `ep_bench` both
exit 0, and `Error` otherwise. After a failed `ep_test`, the log ends with
`ep_test exited <rc>; not running ep_bench`. `runPolicy.activeDeadlineSeconds` (3600 s) stops a hung
run; `backoffLimit: 0` means no retry. The script starts with `ulimit -n 65536`: with the open-files
limit inherited from the container runtime, each `mpirun` stalled for minutes before ssh reached the
workers. The MPIJob name is fixed, so delete the job, which removes its pods, before the next run:

```bash
kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE" delete mpijob nccl-ep-efa
```

## Expected output

The launcher log shows `ls -l /dev/gdrdrv` from each worker, the `rdma_write_bytes` counter of every
EFA device on each worker before and after each program (on the p5en nodes used here these reads printed
`No such file or directory`; see How to read the numbers), and the programs' output. `--tag-output`
prefixes each line with its MPI rank; prefixes, host names and PIDs are removed below. Each rank logs
lines containing these three strings, which show the transport:

```text
NET/OFI Selected provider is efa
RMA/Plugin: Assigned plugin Libfabric to comm
GIN Proxy will not be using GDRCopy
```

From two `p5en.48xlarge` nodes, rank 0 and its worker (abridged):

```text
nccl-ep-efa-worker-0: crw-rw-rw-. 1 root root 237, 0 Oct 11 17:02 /dev/gdrdrv
Rank 0: Testing ncclEpCreateGroup with algorithm: HIGH_THROUGHPUT
...
Rank 0: HIGH_THROUGHPUT mode recv_topk_weights and recv_topk_idx verification passed
Rank 0: HIGH_THROUGHPUT Dispatch flow passed successfully
Rank 0: Combine verification PASSED! All 50 tokens with 7168 elements each correctly combined
[MPI Rank 0] Success
...
=== Summary (High Throughput recipe none, across 16 ranks) ===
NOTE: total time = kernel time + memcpyD2D + misc
--- BW based on total time ---
Dispatch:    total=7790.44 us (min=7460.35, max=8292.64)
             recv: total_bw=49.21  nvl_bw=24.63  rdma_bw=24.58 GB/s
             send: total_bw=49.21  nvl_bw=24.63  rdma_bw=24.58 GB/s
Combine:     total=2530.62 us (min=2301.09, max=2831.58)
             send: total_bw=151.49  nvl_bw=75.83  rdma_bw=75.67 GB/s
             recv: total_bw=151.49  nvl_bw=75.83  rdma_bw=75.67 GB/s
Total (D+C): avg=10329.67 us, min=9921.98 us, max=10759.10 us
...
Logical payload bytes (tokens + forwarded scales, per-rank avg): total_send=383.37 MB (26726 tokens), rdma_send=191.48 MB (13471 tokens), rdma_recv=191.48 MB (13402 tokens), total_recv=383.37 MB (26850 tokens)
...
Global validation: Dispatch=PASSED, Combine=PASSED
```

## How to read the numbers

These are proxy-mode results. `ep_bench` averages CUDA-event timings over the ranks: dispatch
7790.44 us and combine 2530.62 us in the run above, one launch. Dispatch reaches about 15% of EFA line
rate (58.5 MB in 7.790 ms is 7.5 GB/s per GPU, against 50 GB/s on `p5en.48xlarge`: 16 x 200 Gb/s per node
over 8 GPUs), and combine about 46%. NCCL starts one proxy thread by default (`NCCL_GIN_PROXY_NTHREADS`):
in one launch per setting, 2 threads lowered dispatch from 7790 us to 4621 us with the same bytes on the
wire, 4 threads gave the same as 2, and combine stayed at 2531-2599 us; the manifest keeps NCCL's default
of 1.

The expected EFA volume: with 256 experts split 128 per node and 8 chosen per token, 99.65% of tokens
have at least one expert on the other node, and NCCL EP sends each such token across once. A rank's
4,096 tokens of 7,168 BF16 values therefore put about 58.5 MB on the wire per dispatch, and combine
returns the same amount, so each pass should write about 117.0 MB per rank. Dividing 58.5 MB by the
dispatch or combine time gives the per-GPU rate to compare with the line rate above. `ep_bench`'s own
`rdma_send` and `rdma_bw` figures count the logical payload (each destination copy), about 3.2x the
bytes the NICs carry, because a token crosses to the other node once and is forwarded over NVLink
there; read the wire volume from the EFA counters.

To check the volume against the counters, take the increase of `rdma_write_bytes` over `ep_bench`,
summed over a node's EFA devices, and divide it by the node's 8 ranks and by the 61 passes `ep_bench`
makes (10 warm-up, 50 timed, 1 validation): 58.295 GB per node, or 119.5 MB per rank per pass, in the
run above. MPI stays on TCP (`--mca pml ob1 --mca btl self,vader,tcp`), so the counters show NCCL EP
traffic only. If those reads print `No such file or directory`, the pods do not see `hw_counters`; read
the same files on the node, for example from a pod that mounts the host `/sys` read-only.

## Known limitations

1. **GIN proxy only.** EFA-GDA (`NCCL_GIN_TYPE=5`) is not covered here. NCCL EP's device communicator
   requires GIN strong and VA signals, which the aws-ofi-nccl 1.21.1 EFA-GDA backend does not support.
   With NCCL's symmetric GIN kernels left on (`NCCL_SYM_GIN_KERNELS_ENABLE=1`, the default),
   `ncclEpCreateGroup` fails with `GIN strong signals are required, but the GIN plugin does not support them`.
2. **HIGH_THROUGHPUT + FLAT only.** LOW_LATENCY and the other layouts are not exercised.
3. **Frameworks.** A framework process (for example PyTorch for Megatron-LM or NeMo RL) that uses this
   library must load the same `libnccl.so.2`. That is not covered here.
4. **Scale and hardware.** Validated: two `p5en.48xlarge` nodes (16 ranks) from an image built with this
   Dockerfile, in four launches (`ep_test` passed on 16 of 16 ranks and `ep_bench` validated in each); and
   two `p6-b200.48xlarge` nodes in the
   [run reported in the review](https://github.com/awslabs/awsome-distributed-ai/pull/1283#pullrequestreview-5481141372)
   of this recipe, launched with the `eff9dea2` scripts, not this MPIJob: that revision rebuilt for sm_100,
   and a simplified build (apt NCCL, version pins, GDRCopy v2.5.2) run as a shell script in fresh pods from
   the same NGC base, not a `docker build` of this Dockerfile.
   `p6-b300.48xlarge` is built for (sm_103) but not run; for `p5.48xlarge`, see the version constraints.
   More than two nodes is not covered.
5. **Timing is not a performance comparison.** `ep_bench` reports averages over ranks of CUDA-event timings
   from one launch. Repeat on fresh pods before comparing numbers.
