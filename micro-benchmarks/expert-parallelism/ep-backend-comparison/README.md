# Expert-Parallelism Backend Comparison (NCCL vs UCCL vs NVSHMEM) on EKS

Head-to-head MoE dispatch/combine micro-benchmark across three communication backends, run
at the **same EP world size** on the same GPU nodes (designed for 8× `p6-b300.48xlarge`,
64 ranks). This directory is the orchestration layer; the benchmarks themselves live in the
sibling directories.

| Config | What it is | Source benchmark |
|---|---|---|
| **NCCL** (baseline) | Raw all-to-all over EFA. The transport-level **reference ceiling** — moves bytes, but does *not* do token routing or combine-reduction. | [`nccl-alltoall.yaml`](nccl-alltoall.yaml) (built from [`../../nccl-tests`](../../nccl-tests)) |
| **UCCL** | DeepEP-style dispatch/combine over the UCCL all-to-all backend. | [`../uccl-ep-benchmark/kubernetes`](../uccl-ep-benchmark/kubernetes) |
| **NVSHMEM** | DeepEP dispatch/combine over NVSHMEM (libfabric/EFA). | [`../deepep-benchmark/kubernetes`](../deepep-benchmark/kubernetes) |

> **Why no "DeepEP without a backend"?** DeepEP at the pinned commit (`567632d`, pre-EPv2) has
> no internode dispatch/combine path without a transport backend, so a literal "no-backend
> DeepEP on 8 nodes" does not exist. The NCCL all-to-all stands in as the neutral baseline and
> is labelled as a transport ceiling, not as an equal dispatch/combine number.

## Matched configuration (what makes the numbers comparable)

All runs use the **same EP problem size** — otherwise the table is meaningless:

| Parameter | Value |
|---|---|
| World size | 8 nodes × 8 GPU = **64 ranks** |
| `num-tokens` | 4096 (internode) / 128 (low-latency) |
| `hidden` | 7168 |
| `num-topk` | 8 |
| `num-experts` | 256 (divides evenly across 64 ranks) |
| dtype | bf16 |

The UCCL manifests bake these args into `torchrun`; the DeepEP test hard-codes its config
in-image. **Before running, confirm the DeepEP image's config is the anchor** and align UCCL to
it:

```bash
# Read the DeepEP test config from the NVSHMEM image and match UCCL's CLI args to it.
docker run --rm ${NVSHMEM_IMAGE_URI} sed -n '1,60p' /DeepEP/tests/test_internode.py
```

If the DeepEP values differ from 4096/7168/8/256, edit the `torchrun` args in
`../uccl-ep-benchmark/kubernetes/test-*.yaml` to match.

## Prerequisites

- EKS cluster with EFA + GPU nodes; NVIDIA device plugin + AWS EFA device plugin; Kubeflow MPI
  Operator (`kubectl get crd mpijobs.kubeflow.org`). See each benchmark's `kubernetes/README.md`.
- The container images in ECR:
  - NVSHMEM: `../deepep-benchmark/deepep.Dockerfile` (CUDA 13, `sm_90`+`sm_100`)
  - UCCL: `../uccl-ep-benchmark/uccl-ep.Dockerfile` (pinned UCCL commit; single arch via `GPU_SM`, default `sm_100` for B300)
  - NCCL: **reuse the NVSHMEM/DeepEP image** — it already builds `/opt/nccl-tests/build/alltoall_perf`
    with `sm_100` gencode, so no separate `nccl-tests` build is needed for the baseline.

## Account / cluster safety (run first)

```bash
aws sts get-caller-identity                          # confirm the target account
kubectl config current-context                       # confirm the target cluster
kubectl get nodes -l node.kubernetes.io/instance-type=p6-b300.48xlarge   # confirm 8 schedulable
kubectl get crd mpijobs.kubeflow.org                 # confirm MPI Operator
```

## Run order (serial — each config needs all 8 nodes)

**Smoke first.** Before any 8-node job, run the single-node `test-intranode.yaml` for each EP
image. It validates the image, that `sm_100` actually runs on B300, and the launch path in
minutes instead of failing eight nodes deep. Intranode is NVLink-only (same for every backend),
so it is a smoke test, not a comparison row.

```bash
cp env_vars.example env_vars   # then edit image URIs / topology
source env_vars

# 1) NVSHMEM (DeepEP)
( cd ../deepep-benchmark/kubernetes
  IMAGE_URI=$NVSHMEM_IMAGE_URI NUM_NODES=$NUM_NODES \
  envsubst '$IMAGE_URI $INSTANCE_TYPE $GPU_PER_NODE $EFA_PER_NODE $NUM_NODES' < test-internode.yaml | kubectl apply -f -
  # ...wait, save logs, delete. Then low-latency -- see the override note below. )

# 2) UCCL (UCCL-EP)
( cd ../uccl-ep-benchmark/kubernetes
  IMAGE_URI=$UCCL_IMAGE_URI NUM_NODES=$NUM_NODES \
  envsubst '$IMAGE_URI $INSTANCE_TYPE $GPU_PER_NODE $EFA_PER_NODE $NUM_NODES' < test-internode.yaml | kubectl apply -f -
  # ...then test-low-latency.yaml (already pinned to --num-experts=256) )

# 3) NCCL baseline (reuses the DeepEP image's alltoall_perf)
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
    --nccl             nccl_alltoall.log
```

The parser reports, for internode, the **RDMA** leg of the "Best dispatch/combine" line (the
cross-node bottleneck — *not* the intra-node NVL number printed on the same line), and for the
NCCL baseline the busbw **at the EP per-rank payload size** (`num_tokens*hidden*2`, ≈56 MiB) as
well as the asymptotic peak. Record the table in [`RESULTS.md`](RESULTS.md) with image tags,
date, and any config deltas. Eyeball one real launcher log against the parser before trusting it.

## Caveats

- **NCCL is a reference, not an equal.** `alltoall_perf` busbw is pure transport throughput; the
  EP dispatch/combine numbers carry routing + reduction overhead, so they should sit *below* the
  NCCL ceiling. Compare against the **matched-size** busbw, not the asymptotic peak.
- **Internode = RDMA leg.** DeepEP/UCCL print both an RDMA (cross-node) and an NVL (intra-node)
  bandwidth on the same line; only the RDMA number reflects the inter-node transport being
  compared.
- **`num-experts` must divide the world size.** Both tests assert `num_experts % num_ranks == 0`.
  At 8 nodes (64 ranks) the comparison uses 256 (= 4/rank). The DeepEP low-latency default (288)
  is not divisible by 64 and must be overridden (see the run-order note).
- **CUDA skew.** NVSHMEM/NCCL (DeepEP image) are CUDA 13; the UCCL image is CUDA 12.8.1. Separate
  images/pods, so it does not affect a single run, but note it for fairness.
- **UCCL bench scripts** are pulled from upstream `uccl/ep/bench` at image-build time and pinned
  via `UCCL_COMMIT`. If upstream renames CLI flags, adjust the `torchrun` args in the UCCL
  manifests.
