<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# PointWorld on Kubeflow Trainer v2 (TrainJob)

This directory provides PointWorld manifests for clusters running the
[Kubeflow Trainer v2](https://www.kubeflow.org/docs/components/trainer/) controller
(`trainer.kubeflow.org` — `TrainJob` / `ClusterTrainingRuntime` CRDs).

The manifests in the parent [`../`](../) directory use the older Kubeflow
**PyTorchJob v1** API (`kubeflow.org/v1`), which matches several existing test
cases in this repository. If your cluster runs Trainer v2 instead (the
`pytorchjobs.kubeflow.org` CRD is absent and you have `trainjobs.trainer.kubeflow.org`),
use the manifests here.

| File | Kind | Purpose |
|---|---|---|
| [`pointworld-runtime.yaml`](./pointworld-runtime.yaml) | `ClusterTrainingRuntime` | Reusable torch-distributed runtime (apply once) |
| [`pointworld-trainjob.yaml`](./pointworld-trainjob.yaml) | `TrainJob` | Pre-training job referencing the runtime |

## Usage

```bash
# 1. Build + push the image and stage data/DINOv3 (see ../../README.md).
# 2. Configure and render with the shared env_vars + deploy.sh (from the test
#    case root). deploy.sh substitutes IMAGE_URI/NAMESPACE/NUM_NODES/etc.
source ../../env_vars
../deploy.sh trainer-v2/pointworld-runtime.yaml    # apply the runtime once
../deploy.sh trainer-v2/pointworld-trainjob.yaml
kubectl get trainjob pointworld-pretrain -n ${NAMESPACE}
kubectl logs -f pointworld-pretrain-node-0-0-<suffix> -n ${NAMESPACE}
```

## Validation notes (Amazon EKS, Trainer v2.0.0, p5en.48xlarge / H200)

This test case was exercised end-to-end on an EKS cluster running Kubeflow
Trainer v2.0.0. The following behaviors were confirmed and are baked into the
manifests:

- **`numNodes` -> JobSet parallelism** only propagates when the runtime's
  `replicatedJobs[].template` carries the label
  `trainer.kubeflow.org/trainjob-ancestor-step: trainer`. Without it the job runs
  on a single node regardless of `numNodes`.
- **`torchrun` is required** as the entrypoint. The torch plugin injects the
  `PET_*` rendezvous env; launching `python train.py` directly fails with
  `ValueError: Error initializing torch.distributed using env:// rendezvous:
  environment variable RANK expected, but not set`. PointWorld's `Trainer` then
  initializes with `init_method="env://"` and reads `LOCAL_RANK`.
- **EFA on p5en.48xlarge** advertises `vpc.amazonaws.com/efa: 16` (not 32).
- **DINOv3 weights** must keep the canonical filename
  `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` (see `../../scripts/2.download_dinov3.sh`).

A 1-node / 8-GPU smoke run (BEHAVIOR, `--ptv3_size=large`, `--max_train_steps=2`)
trained end-to-end (loss decreased from a cold start over two optimizer steps),
and `eval.py` loaded a released checkpoint and produced metrics on the BEHAVIOR
test split.
