# Expert-Parallelism Backend Comparison on EFA

Head-to-head MoE dispatch/combine micro-benchmark across 3 EP backends, with raw NCCL all-to-all retained as a separately labeled transport reference. Runs within a result set use the same EP world size, problem shape, common dependency stack, and GPU nodes. The latest campaign adds DeepEP V2 over NCCL GIN EFA-GDA on B200 at EP16 and EP32. The historical B300 campaign used 64 ranks and also exercised a 256-rank transport reference.

This directory provides comparison orchestration and log collation. Backend build and launch harnesses live in sibling directories or the linked DeepEP V2 contribution.

| Config | What it is | Source benchmark |
|---|---|---|
| **NCCL** (reference) | Raw all-to-all over EFA. It moves bytes but does *not* do token routing or combine-reduction, so it is not an EP-backend row. | [`nccl-alltoall.yaml`](nccl-alltoall.yaml) (built from [`../../nccl-tests`](../../nccl-tests)) |
| **UCCL** | DeepEP-style dispatch/combine over the UCCL all-to-all backend. | [`../uccl-ep-benchmark/kubernetes`](../uccl-ep-benchmark/kubernetes) |
| **DeepEP V1 NVSHMEM** | DeepEP V1 dispatch/combine over NVSHMEM libfabric/EFA. | [`../deepep-benchmark/kubernetes`](../deepep-benchmark/kubernetes) |
| **DeepEP V2 NCCL GIN** | DeepEP V2 `ElasticBuffer` dispatch/combine over NCCL GIN EFA-GDA. | [DeepEP V2 benchmark PR 1234](https://github.com/awslabs/awsome-distributed-ai/pull/1234) and [`deepep_v2_selected_cases.py`](deepep_v2_selected_cases.py) |

> **Backend naming.** DeepEP V1 at commit `567632d` has no internode path without an external transport, so this comparison names its NVSHMEM transport explicitly. DeepEP V2 is a distinct NCCL GIN backend and does not replace the raw NCCL transport reference. Raw NCCL remains context only and is never treated as an equal dispatch/combine measurement.

## Matched configuration (what makes the numbers comparable)

Compare backends only within the same platform campaign. B200, B300, and H100 measurements are kept on separate result pages. The latest B200 campaign used the following matched shapes:

| Parameter | Value |
|---|---|
| World size | EP16: 2 nodes × 8 GPUs = 16 ranks; EP32: 4 nodes × 8 GPUs = 32 ranks |
| `num-tokens` | 4096 (internode) / 128 (low-latency) |
| `hidden` | 7168 |
| `num-topk` | 8 |
| `num-experts` | 256 |
| dispatch dtype | FP8 and BF16 |
| combine dtype | BF16 headline; DeepEP V2 also prints FP8 diagnostic data |

The historical B300 and H100 result pages use 4-node and 8-node topologies with the same token, hidden-size, top-k, and expert values. The UCCL manifests bake these arguments into the `python3 bench/test_*.py` invocation; the DeepEP V1 test hard-codes its config in-image. Before running V1, confirm the image config and align UCCL to it:

```bash
# Read the DeepEP test config from the NVSHMEM image and match UCCL's CLI args to it.
docker run --rm ${NVSHMEM_IMAGE_URI} sed -n '1,60p' /DeepEP/tests/test_internode.py
```

If the DeepEP values differ from 4096/7168/8/256, edit the bench args in
`../uccl-ep-benchmark/kubernetes/test-*.yaml` to match.

## Prerequisites

- EKS cluster with EFA + GPU nodes; NVIDIA device plugin + AWS EFA device plugin; Kubeflow MPI
  Operator (`kubectl get crd mpijobs.kubeflow.org`). See each benchmark's `kubernetes/README.md`.
- The container images in ECR:
  - DeepEP V1 NVSHMEM: `../deepep-benchmark/deepep.Dockerfile` (CUDA 13, `sm_90`+`sm_100`)
  - UCCL: `../uccl-ep-benchmark/uccl-ep.Dockerfile` (CUDA 13; pinned UCCL commit; Hopper + Blackwell via PTX)
  - DeepEP V2 NCCL GIN: the standalone build and launch workflow from [PR 1234](https://github.com/awslabs/awsome-distributed-ai/pull/1234), with all revisions pinned to the campaign being reproduced
  - NCCL: **reuse the NVSHMEM/DeepEP image** — it already builds `/opt/nccl-tests/build/alltoall_perf`
    with `sm_100` gencode, so no separate `nccl-tests` build is needed for the baseline.
- For EFA-GDA, verify the host EFA driver and GDRCopy requirements documented by the DeepEP V2 harness before scheduling a multi-node run.

## Account / cluster safety (run first)

```bash
aws sts get-caller-identity                          # confirm the target account
kubectl config current-context                       # confirm the target cluster
kubectl get nodes -l node.kubernetes.io/instance-type=p6-b300.48xlarge   # confirm $NUM_NODES schedulable
kubectl get crd mpijobs.kubeflow.org                 # confirm MPI Operator
```

## Run order

Run backends serially on the same named node set. Smoke each image on 1 node first. An intranode smoke is NVLink-only and is not a scored EFA comparison row. The commands below cover the existing DeepEP V1, UCCL, and raw NCCL EKS manifests. Run DeepEP V2 with the linked harness and the matched wrapper in this directory, using 4,096 tokens/rank for prefill and 128 tokens/rank for decode.

`deepep_v2_selected_cases.py` matches the synthetic DeepEP V2 revision pinned in [`RESULTS-b200.md`](RESULTS-b200.md). Review its upstream `Namespace` fields before using it with a different DeepEP V2 revision.

```bash
cp env_vars.example env_vars   # then edit image URIs / topology
source env_vars

# 1) DeepEP V1 NVSHMEM
( cd ../deepep-benchmark/kubernetes
  IMAGE_URI=$NVSHMEM_IMAGE_URI NUM_NODES=$NUM_NODES \
  envsubst '$IMAGE_URI $INSTANCE_TYPE $GPU_PER_NODE $EFA_PER_NODE $NUM_NODES' < test-internode.yaml | kubectl apply -f -
  # ...wait, save logs, delete. Then low-latency -- see the override note below. )

# 2) UCCL (UCCL-EP) — one MPI rank per GPU (NP = NUM_NODES * GPU_PER_NODE)
( cd ../uccl-ep-benchmark/kubernetes
  IMAGE_URI=$UCCL_IMAGE_URI NUM_NODES=$NUM_NODES NP=$NP \
  envsubst '$IMAGE_URI $INSTANCE_TYPE $GPU_PER_NODE $EFA_PER_NODE $NUM_NODES $NP' < test-internode.yaml | kubectl apply -f -
  # ...then test-low-latency.yaml (already pinned to --num-experts=256) )

# 3) DeepEP V2 NCCL GIN
# Run an INFO-level admission first and require NCCL_GIN_TYPE=5, a successful
# Libfabric_GDAKI context, a nonzero GIN layout, and bidirectional EFA deltas.
# Use deepep_v2_selected_cases.py inside the pinned V2 image, then repeat with
# NCCL_DEBUG=WARN for the scored run.
# The multi-node launcher must provide WORLD_SIZE, RANK, MASTER_ADDR, and
# MASTER_PORT to each node. Inside each node's container, run one of:
# python3 deepep_v2_selected_cases.py --num-processes=8 --num-tokens=4096 --hidden=7168 --num-topk=8 --num-experts=256
# python3 deepep_v2_selected_cases.py --num-processes=8 --num-tokens=128  --hidden=7168 --num-topk=8 --num-experts=256

# 4) NCCL reference (reuses the DeepEP V1 image's alltoall_perf)
IMAGE_URI=$NCCL_IMAGE_URI \
envsubst '$IMAGE_URI $INSTANCE_TYPE $GPU_PER_NODE $EFA_PER_NODE $NUM_NODES $NP' < nccl-alltoall.yaml | kubectl apply -f -
```

> **DeepEP low-latency at 8 nodes — required override.** The merged DeepEP low-latency manifest
> runs `python3 /DeepEP/tests/test_low_latency.py` with no args, so it uses the upstream default
> `--num-experts=288`. The test asserts `num_experts % num_ranks == 0`; at 8 nodes (64 ranks),
> `288 % 64 ≠ 0` and it aborts. Match the comparison's 256 by patching the rendered manifest:
> ```bash
> cd ../deepep-benchmark/kubernetes
> IMAGE_URI=$NVSHMEM_IMAGE_URI NUM_NODES=$NUM_NODES \
> envsubst '$IMAGE_URI $INSTANCE_TYPE $GPU_PER_NODE $EFA_PER_NODE $NUM_NODES' < test-low-latency.yaml \
>   | sed 's#test_low_latency.py#test_low_latency.py --num-experts 256#' | kubectl apply -f -
> ```
> (DeepEP internode defaults are already 4096/7168/8/**256**, so internode needs no override.)

Save each launcher log (`kubectl logs <…-launcher> > <name>.log`) and **delete the job before
the next run** (MPIJob names are fixed; re-applying collides, and each job needs all 8 nodes):

```bash
kubectl delete mpijob deepep-internode uccl-ep-internode nccl-alltoall   # etc.
```

## Collate

```bash
python3 collect_results.py \
    --nvshmem-internode nvshmem_internode.log \
    --nvshmem-lowlat   nvshmem_lowlat.log \
    --uccl-internode   uccl_internode.log \
    --uccl-lowlat      uccl_lowlat.log \
    --deepep-v2-prefill deepep_v2_prefill.log \
    --deepep-v2-decode  deepep_v2_decode.log \
    --nccl             nccl_alltoall.log
```

For DeepEP V1 and UCCL internode logs, the parser reports the RDMA leg of the `Best dispatch/combine` line, not the intra-node NVL value printed beside it. For DeepEP V2, it reports rank-zero SO bandwidth, SU bandwidth, and latency for each dispatch dtype and operation. For raw NCCL, it reports bus bandwidth at the EP per-rank payload size, approximately 56 MiB, plus the asymptotic peak. Eyeball one real launcher log against the parser before trusting it.

Results are recorded per platform: [`RESULTS-b200.md`](RESULTS-b200.md) (B200 with DeepEP V2), [`RESULTS.md`](RESULTS.md) (B300 historical), and [`RESULTS-p5.md`](RESULTS-p5.md) (P5/H100 historical). For other instance types set `INSTANCE_TYPE` and `EFA_PER_NODE` to the devices actually exposed by the target nodes.

## Scaling beyond 8 nodes (256-rank findings)

The historical UCCL and DeepEP V1 matrix was pushed to 16 and 32 nodes, or 128 and 256 ranks, on a 32-node `p6-b300.48xlarge` Capacity Block on 14 July 2026. Those V1-era kernels hit implementation limits between 65 and 256 ranks; only the raw NCCL reference ran at 256 ranks. These findings do not establish a DeepEP V2 limit. Details and per-limit source citations are in [`RESULTS.md`](RESULTS.md). Operational notes for rerunning the V1-era matrix at scale:

- **HT internode**: DeepEP asserts at >160 ranks (`NUM_MAX_NVL_PEERS 8 × NUM_MAX_RDMA_PEERS 20`,
  `kernels/configs.cuh`) and its stock combine tuning tables already abort at 16 nodes; UCCL
  overflows an `int32` buffer bound above 64 ranks. Treat the HT comparison as an
  **8-nodes-per-EP-domain benchmark** — which matches how training deploys these kernels
  (EP32/EP64 groups inside a larger world).
- **Low-latency**: both implementations cap between 64 and 128 ranks (UCCL: compile-time
  signaling-buffer arena; NVSHMEM/DeepEP: libfabric host-proxy retry exhaustion with moving
  victims per run).
- **GDRCopy at scale (NVSHMEM)**: past ~1 GiB of LL buffer, NVSHMEM grows its symmetric heap
  dynamically and must register each chunk over libfabric via **GDRCopy inside the container**.
  The manifests set `NVIDIA_GDRCOPY=enabled`, but some clusters' nvidia container toolkit
  ignores it — if every rank dies at `mem_heap.cpp:1361 register_mem_handle failed` after a
  `GDRCopy support not enabled` warning, hostPath-mount `/dev/gdrdrv` into the worker
  (requires `privileged: true`) and ensure the host loads `gdrdrv` (gdrcopy-loader DaemonSet
  or DLAMI).
- **NCCL at 32 nodes** works unmodified (`NUM_NODES=32`, `NP=256`); expect matched-size busbw
  to drop vs 8 nodes (fan-out cost).

## Caveats

- **NCCL is a reference, not an equal.** `alltoall_perf` busbw is pure transport throughput, while EP dispatch/combine includes routing and reduction and uses backend-specific bandwidth accounting. Treat matched-size NCCL bus bandwidth as transport context, not as a hard ceiling or an EP-backend row.
- **Internode = RDMA leg.** DeepEP/UCCL print both an RDMA (cross-node) and an NVL (intra-node)
  bandwidth on the same line; only the RDMA number reflects the inter-node transport being
  compared.
- **DeepEP V2 accounting differs.** V2 prints SO and SU bandwidth plus per-operation latency. SO is the cross-node leg, but it is not numerically interchangeable with the V1/UCCL RDMA accounting. Use latency as the primary cross-backend metric and retain backend-native bandwidth as directional evidence.
- **`num-experts` must divide the world size.** Both tests assert `num_experts % num_ranks == 0`.
  At 8 nodes (64 ranks) the comparison uses 256 (= 4/rank). The DeepEP low-latency default (288)
  is not divisible by 64 and must be overridden (see the run-order note).
- **Toolchain.** Verify the exact toolchain per result page. The B200 comparison used CUDA 13.0.3 and the same vLLM wheel in all 3 EP-backend images. The historical raw NCCL reference shares the DeepEP V1 image.
- **UCCL bench scripts** are pulled from upstream `uccl/ep/bench` at image-build time and pinned
  via `UCCL_COMMIT`. If upstream renames CLI flags, adjust the bench args in the UCCL
  manifests.
