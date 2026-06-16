# Compute Tests (Tests 4-6)

Validates CPU queue, GPU families (single-NIC G-series + multi-NIC P-series),
and NCCL multi-node communication over EFA.

---

## Test 4: G-series GPU (single NIC)

Single-NIC GPU instances (`g5`/`g6`) use `add-cng.yaml` — deploy them as the On-Demand
CNG, e.g. `OnDemandInstanceType=g6.12xlarge`, `OnDemandQueueName=gpu-g6` (see
[README Example 2](../README.md#5-usage-examples)).

```bash
srun --partition=gpu-g6 --nodes=1 --gres=gpu:1 \
  --container-image=docker://nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

**Expected:** the container runs and `nvidia-smi` lists the node's GPU(s) — confirms
single-NIC GPU + Pyxis on a G-series queue.

---

## Test 5: P5/P6 GPU (multi-NIC)

Multi-NIC GPU node groups, selected automatically by `PseriesInstanceType`
(`p5.48xlarge`/`p5e`/`p5en` → `add-cng-p5.yaml`; `p6-b200.48xlarge` → `add-cng-p6-b200`;
`p6-b300.48xlarge` → `add-cng-p6-b300`). The EFA interface count is derived from the
type — no parameter to set. A one-line interactive `srun` is enough for a GPU sanity
check (no batch script needed); set `--partition` to your GPU queue:

```bash
srun --partition=gpu-p6b200 --nodes=1 --gres=gpu:8 \
  --container-image=docker://nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

**Expected:** `nvidia-smi` lists 8 GPUs of the expected model (H100 / H200 / B200 /
B300). This confirms the multi-NIC launch template booted and Pyxis works on the GPU
node. EFA itself is exercised by Test 6.

---

## Test 6: NCCL multi-node (EFA)

2-node × 8-GPU `all_reduce_perf` over EFA, using the repo's canonical launcher
[`micro-benchmarks/nccl-tests/slurm/nccl-tests-container.sbatch`](../../../micro-benchmarks/nccl-tests/slurm/nccl-tests-container.sbatch)
(it reads `$IMAGE`, default `/fsx/nccl-tests.sqsh`). Only two PCS-specific deltas:

1. **Import the image on the login node** — `enroot import` builds its overlayfs on the
   node-local root disk (the login node has a 300 GiB root via `RootVolumeSize`); FSx
   Lustre can't host that overlay, so only the resulting `.sqsh` goes to shared `/fsx`.
   Pin a specific image tag for reproducible numbers (don't use `latest`):

   ```bash
   # On the login node (direct, not a batch job). enroot URI form is
   # docker://[REGISTRY#]REPO:TAG — the registry needs a '#', or it 401s on Docker Hub.
   TAG=cuda12.8.1-efa1.43.2-ofiv1.16.3-ncclv2.27.7-1-testsv2.16.9
   enroot import -o /fsx/nccl-tests.sqsh "docker://public.ecr.aws#hpc-cloud/nccl-tests:${TAG}"
   ```

2. **Submit with your GPU queue** — set the partition to the queue you deployed
   (`gpu-p5` / `gpu-p6b200` / `gpu-p6b300`); the canonical script defaults to 2 nodes,
   8 tasks/node:

   ```bash
   cd /fsx && sbatch --partition=gpu-p6b200 \
     /fsx/awsome-distributed-ai/micro-benchmarks/nccl-tests/slurm/nccl-tests-container.sbatch
   ```

**Expected** (in `nccl-all_reduce_perf_<jobid>.out`):
- EFA is the provider: `NET/OFI Selected provider is efa, fabric is efa-direct (found N nics)`
  (N = EFA interface count: 32 for p5/p5e, 16 for p5en, 8 for p6-b200, 16 for p6-b300).
- Correctness: `# Out of bounds values : 0 OK`, every size `#wrong: 0`.
- `busbw` rises with message size — on 2× p6-b300, ~654 GB/s at 16 GiB but **~751 GB/s at
  64 GiB** (16 cards aren't saturated at 16 GiB); ~480 GB/s on 2× p5.

> **Sizing the sweep on B300.** The canonical script sweeps to `-e 16G`. On p6-b300 that
> under-reports peak bandwidth — raise it to `-e 64G` (edit the `all_reduce_perf` line) to
> see the cards saturate. **Don't go to 128 GiB/256 GiB**: the all_reduce buffer exceeds
> B300 GPU memory and the job is OOM-killed. For a higher peak, add nodes, not buffer size.

---
