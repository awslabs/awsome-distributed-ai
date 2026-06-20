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
- The three container images built and pushed to ECR:
  - NVSHMEM: `../deepep-benchmark/deepep.Dockerfile` (CUDA 13, `sm_90`+`sm_100`)
  - UCCL: `../uccl-ep-benchmark/uccl-ep.Dockerfile` (pinned UCCL commit; single arch via `GPU_SM`, default `sm_100` for B300)
  - NCCL: `../../nccl-tests/nccl-tests.Dockerfile` (CUDA 13.0.2, ships `alltoall_perf`, `sm_100`/`sm_103`)

## Account / cluster safety (run first)

```bash
aws sts get-caller-identity                          # confirm the target account
kubectl config current-context                       # confirm the target cluster
kubectl get nodes -l node.kubernetes.io/instance-type=p6-b300.48xlarge   # confirm 8 schedulable
kubectl get crd mpijobs.kubeflow.org                 # confirm MPI Operator
```

## Run order (serial — each config needs all 8 nodes)

```bash
cp env_vars.example env_vars   # then edit image URIs / topology
source env_vars

# 1) NVSHMEM (DeepEP)
( cd ../deepep-benchmark/kubernetes
  IMAGE_URI=$NVSHMEM_IMAGE_URI NUM_NODES=$NUM_NODES \
  envsubst '$IMAGE_URI $INSTANCE_TYPE $GPU_PER_NODE $EFA_PER_NODE $NUM_NODES' < test-internode.yaml   | kubectl apply -f -
  # ...wait for completion, save logs, delete, then test-low-latency.yaml )

# 2) UCCL (UCCL-EP)
( cd ../uccl-ep-benchmark/kubernetes
  IMAGE_URI=$UCCL_IMAGE_URI NUM_NODES=$NUM_NODES \
  envsubst '$IMAGE_URI $INSTANCE_TYPE $GPU_PER_NODE $EFA_PER_NODE $NUM_NODES' < test-internode.yaml   | kubectl apply -f -
  # ...then test-low-latency.yaml )

# 3) NCCL baseline
IMAGE_URI=$NCCL_IMAGE_URI \
envsubst '$IMAGE_URI $INSTANCE_TYPE $GPU_PER_NODE $EFA_PER_NODE $NUM_NODES $NP' < nccl-alltoall.yaml | kubectl apply -f -
```

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

This prints the comparison table; record it in [`RESULTS.md`](RESULTS.md) along with the image
tags, date, and any config deltas.

## Caveats

- **NCCL is a reference, not an equal.** `alltoall_perf` busbw is pure transport throughput; the
  EP dispatch/combine numbers carry routing + reduction overhead, so they should sit *below* the
  NCCL ceiling.
- **CUDA skew.** NVSHMEM/NCCL images are CUDA 13; the UCCL image is CUDA 12.8.1. Separate
  images/pods, so it does not affect a single run, but note it for fairness.
- **UCCL bench scripts** are pulled from upstream `uccl/ep/bench` at image-build time and pinned
  via `UCCL_COMMIT`. If upstream renames CLI flags, adjust the `torchrun` args in the UCCL
  manifests.
