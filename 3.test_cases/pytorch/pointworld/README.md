<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# PointWorld: Distributed 3D World Model Pre-training

This test case demonstrates distributed pre-training and evaluation of
[PointWorld](https://github.com/NVlabs/PointWorld) (NVIDIA + Stanford) on AWS GPU
clusters orchestrated by Amazon EKS / SageMaker HyperPod EKS.

PointWorld is a large pre-trained **3D world model** for robotic manipulation. It
forecasts full-scene 3D **point flow** (per-point 3D displacements over ~1 second)
from one or a few RGB-D images plus a sequence of robot actions. Crucially, the
robot action is itself represented as 3D point flow rather than an
embodiment-specific action space (e.g. joint angles), so a single model can learn
jointly across embodiments (a single-arm Franka and a bimanual humanoid) and
condition directly on physical geometry.

We pre-train the **large PTv3** variant on the multi-domain **DROID + BEHAVIOR**
corpus across **8 x p5en.48xlarge** instances (64 x NVIDIA H200) using PyTorch
DDP, then evaluate the resulting checkpoint with PointWorld's point-flow metrics.

| | |
|---|---|
| **Model** | PointWorld (PTv3 backbone + DINOv3 scene featurizer) |
| **Framework** | PyTorch + DDP (`torchrun`, `init_method="env://"`) |
| **Precision** | BF16 (AMP) |
| **Data** | DROID (real) + BEHAVIOR (sim), WebDataset shards |
| **Paper** | [arXiv:2601.03782](https://arxiv.org/abs/2601.03782) (CVPR 2026 Highlight) |
| **Code** | [NVlabs/PointWorld](https://github.com/NVlabs/PointWorld) (Apache-2.0) |
| **Datasets** | [PointWorld-DROID](https://huggingface.co/datasets/nvidia/PointWorld-DROID), [PointWorld-BEHAVIOR](https://huggingface.co/datasets/nvidia/PointWorld-BEHAVIOR) |
| **Checkpoints** | [nvidia/PointWorld_models](https://huggingface.co/nvidia/PointWorld_models) |

## Agenda: pre-train, then evaluate

PointWorld's thesis is **"pre-train once, no post-training."** A single
pre-trained checkpoint drives a real robot via model-predictive control (MPC)
with no demonstrations or task-specific fine-tuning. Accordingly, this test case
covers the two stages that run on a training cluster:

1. **Pre-train** the 3D world model (multi-node DDP) on DROID + BEHAVIOR.
2. **Evaluate** the resulting checkpoint (point-flow L2 metrics; viser viz optional).

There is **no fine-tuning stage** — it is intentionally absent from the model
design. Real-robot MPC deployment happens off-cluster on physical hardware and is
out of scope for this repository.

> [!info] Kubernetes-only
> This test case targets **Amazon EKS / SageMaker HyperPod EKS**. There is no
> Slurm variant.

> [!note] Kubeflow Trainer v1 vs v2
> The manifests in [`kubernetes/`](./kubernetes/) use the Kubeflow **PyTorchJob v1**
> API (`kubeflow.org/v1`). If your cluster runs the newer **Kubeflow Trainer v2**
> controller (`trainer.kubeflow.org` — `TrainJob`/`ClusterTrainingRuntime`, and the
> `pytorchjobs.kubeflow.org` CRD is absent), use the equivalent manifests in
> [`kubernetes/trainer-v2/`](./kubernetes/trainer-v2/) instead.

## Prerequisites

- An EKS or SageMaker HyperPod EKS cluster with **8 x p5en.48xlarge** (64 x H200)
- [Kubeflow Training Operator](https://www.kubeflow.org/docs/components/training/)
  (provides the `PyTorchJob` CRD)
- [NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin) and
  [EFA device plugin](https://github.com/aws-samples/aws-efa-eks)
- [FSx for Lustre](https://docs.aws.amazon.com/fsx/latest/LustreGuide/what-is.html)
  PVC named `fsx-claim` mounted at `/fsx`
- Docker + Amazon ECR for building and hosting the container image
- **DINOv3 access** (gated — see below)

## 1. Clone this repository

```bash
git clone https://github.com/awslabs/awsome-distributed-ai.git
cd awsome-distributed-ai/3.test_cases/pytorch/pointworld
```

## 2. Build and push the container

The pre-training, evaluation, **and** data-staging jobs all run inside this image,
so build and push it to Amazon ECR before anything else.

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1   # adjust to your cluster region
REPO=pointworld
TAG=05484826        # matches POINTWORLD_COMMIT in pointworld.Dockerfile

aws ecr create-repository --repository-name ${REPO} --region ${REGION} || true
aws ecr get-login-password --region ${REGION} \
  | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

docker build -t ${REPO}:${TAG} -f pointworld.Dockerfile .
docker tag ${REPO}:${TAG} ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${TAG}
docker push ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${TAG}
```

The image tag matches `POINTWORLD_COMMIT` in the Dockerfile so the running image
is traceable to an exact upstream commit. Then replace `<ACCOUNT_ID>` and
`<REGION>` in the manifests under [`kubernetes/`](./kubernetes/) with your values.

## 3. Stage data and weights onto FSx (in-cluster)

> [!important] FSx is not mounted on your laptop
> FSx for Lustre is a VPC-internal filesystem. The `/fsx` paths in this test case
> only exist **inside the cluster**, so all downloading, restoring, and WDS
> conversion runs in a Kubernetes Job that mounts the `fsx-claim` PVC — not from
> your local terminal. The `scripts/*` in this repo are the same steps expressed
> as standalone helpers; the Job runs them where `/fsx` actually exists.

PointWorld needs two things staged onto FSx before training:

1. **DINOv3 ViT-L/16 weights** (gated by Meta) — the scene featurizer backbone.
   Request access at <https://github.com/facebookresearch/dinov3>, then copy the
   `dinov3_vitl16` pre-train checkpoint URL from the access email. The weights are
   **never baked into the image**; the training/eval manifests symlink them from
   FSx at pod start.
2. **WebDataset (WDS) shards** — PointWorld ships datasets as packaged Hugging
   Face archives that you download, restore (`recover_dataset_from_parts.sh`), and
   convert to WDS.

[`kubernetes/pointworld-data-prep.yaml`](./kubernetes/pointworld-data-prep.yaml)
does all of this in one Job. Because the Job runs inside the PointWorld container,
**build and push the image first (Section 2)**, then edit the Job:

- Replace `<ACCOUNT_ID>` / `<REGION>` (your ECR image — built in Section 2).
- Paste your gated DINOv3 URL into `DINOV3_URL` (treat it as a secret; do not
  commit it).
- Set `BEHAVIOR_TASKS` / `MAX_CLIPS`: the defaults stage a small BEHAVIOR subset
  for a quick smoke run; use `BEHAVIOR_TASKS=all` and `MAX_CLIPS=-1` for the full
  dataset.

```bash
kubectl apply -f kubernetes/pointworld-data-prep.yaml
kubectl logs -f job/pointworld-data-prep -n kubeflow
```

The Job writes the DINOv3 weights and WDS shards into the layout the
training/eval manifests expect on the shared filesystem:

```text
/fsx/pointworld/
├── dataset/
│   ├── behavior/wds/{train,test}/...          # from the data-prep Job
│   │   └── metadata_rank0.json                # written by convert_wds on completion
│   └── droid/wds/{train,test}/...             # full-scale only (see note below)
│       └── test/expert_confidence-seed=42.h5  # for DROID filtered metrics
├── dinov3/checkpoints/
│   └── dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth   # gated; canonical name required
├── train_logs/                                # written during pre-training
└── checkpoints/                               # written during pre-training
```

The pre-training pod mounts the WDS root at `/dataset` via
`subPath: pointworld/dataset`, so PointWorld's `LOCAL_DATASET_DIR=/dataset`
resolves `/dataset/behavior/wds` (and `/dataset/droid/wds`).

> [!note] DROID vs BEHAVIOR
> DROID is distributed as a single multi-terabyte split archive that does not
> subset cleanly, so the Job stages **BEHAVIOR** (sharded per task,
> ~1.6–14 GB/task), which is the practical choice for development and smoke
> tests. To add DROID at full scale, extend the Job to download the DROID package
> and run the same restore → convert steps with `--domain droid`, then copy the
> released expert-confidence artifact into the generated test split for filtered
> metrics:
>
> ```text
> droid/confidence/expert_confidence-seed=42.h5
>     -> /fsx/pointworld/dataset/droid/wds/test/expert_confidence-seed=42.h5
> ```

### Running the staging steps by hand

If you prefer to stage manually (for example, from a HyperPod login pod or any
pod that mounts `fsx-claim`), the `scripts/` helpers are the equivalent steps:
`scripts/2.download_dinov3.sh` for the weights, and
`scripts/0.download_dataset.py` → `recover_dataset_from_parts.sh` →
`scripts/1.convert_wds.py` for the data. They take the same `/fsx/...` paths and
must run inside a pod with FSx mounted.

## 4. Pre-train (Kubernetes PyTorchJob)

```bash
# Update <ACCOUNT_ID> and <REGION> in kubernetes/pointworld-pretrain.yaml first.
kubectl apply -f kubernetes/pointworld-pretrain.yaml
kubectl logs -f pointworld-pretrain-worker-0 -n kubeflow
```

This launches 8 worker pods (one per node), each running `torchrun` with 8
processes (one per H200) for 64-way data-parallel pre-training. The flagship flag
set (large PTv3, DROID + BEHAVIOR) is encoded in the manifest and mirrored in
[`configs/train-large-droid-behavior.env`](./configs/train-large-droid-behavior.env).

The PyTorchJob operator injects `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, and
`RANK`; PointWorld's `Trainer` initializes the process group with
`init_method="env://"` and reads `LOCAL_RANK` directly, so no extra launcher is
needed.

### Launch-pattern detail

The container `command` is a small `bash -c` wrapper that:

1. Symlinks the gated DINOv3 weights from `/fsx/pointworld/dinov3/checkpoints`
   into `/pointworld/third_party/dinov3/checkpoints` (keeping gated weights out of
   the image), then
2. `exec`s `torchrun ... train.py` with the flagship DROID + BEHAVIOR flags.

Doing the symlink in the main container (rather than an initContainer) ensures it
lives in the same filesystem namespace as the training process.

> [!note] Kubeflow Trainer v2
> On a Trainer v2 cluster, apply the manifests in
> [`kubernetes/trainer-v2/`](./kubernetes/trainer-v2/) instead — same image and
> FSx layout, expressed as a `TrainJob` + `ClusterTrainingRuntime`.

## 5. Evaluate (Kubernetes Job)

```bash
# Point MODEL_PATH in kubernetes/pointworld-eval.yaml at your trained checkpoint
# (or a released checkpoint from nvidia/PointWorld_models).
kubectl apply -f kubernetes/pointworld-eval.yaml
kubectl logs -f job/pointworld-eval -n kubeflow
```

- **DROID** (annotation-aware): the main metric is
  `full_eval/test/filtered_l2_moved/mean`, which uses the expert-confidence
  artifact to focus on reliable moving-point regions.
- **BEHAVIOR** (simulation): unfiltered metrics (data is noiseless). Set
  `--domains=behavior`, `--data_dirs=/dataset/behavior/wds`, and drop
  `--confidence_thres`.

For a quick smoke test, set `EVAL_NUM_BATCHES` to e.g. `100` instead of `-1`.

## 6. Parse throughput

After capturing training logs, compute steady-state throughput:

```bash
kubectl logs pointworld-pretrain-worker-0 -n kubeflow > run.log
python scripts/parse_benchmark.py \
    --log_file run.log \
    --warmup_steps 20 \
    --global_batch_size 1408 \
    --num_gpus 64 \
    --gpu_type h200
```

> `--global_batch_size` is `per_gpu_batch_size * num_gpus` (e.g. `22 * 64 = 1408`).
> The step/loss/time regexes are configurable; adjust them if your build's log
> format differs from the release default.

## Tuning notes

- **Batch size**: `--batch_size=22` is the upstream default. H200 has 141 GB HBM;
  raise it if memory allows, and update `--global_batch_size` accordingly when
  parsing throughput with [`scripts/parse_benchmark.py`](./scripts/parse_benchmark.py).
- **EFA**: `vpc.amazonaws.com/efa: 16` matches p5en.48xlarge (16 EFA network cards
  per node, as advertised by the EFA device plugin). Adjust for other instance
  types — for example p5.48xlarge advertises 32.
- **Shared memory**: the `shmem` `emptyDir` backs the PyTorch DataLoader workers;
  reduce its `sizeLimit` for smaller instances.
- **B200**: this image targets H200. B200 EFA networking needs NCCL >= 2.29 — see
  "Known limitations" below.

## Directory structure

```text
pointworld/
├── README.md                          # this file
├── pointworld.Dockerfile              # CUDA 12.4; clones PointWorld @ pinned commit
├── .gitignore
├── configs/
│   └── train-large-droid-behavior.env # canonical flag set for the flagship run
├── scripts/
│   ├── 0.download_dataset.py          # download DROID + BEHAVIOR HF packages
│   ├── 1.convert_wds.py               # restore/H5 -> WDS (wraps upstream data branch)
│   ├── 2.download_dinov3.sh           # fetch gated DINOv3 weights
│   └── parse_benchmark.py             # throughput / loss from logs
└── kubernetes/
    ├── README.md                      # manifest index (points back here)
    ├── pointworld-data-prep.yaml       # Job: stage DINOv3 + WDS onto FSx (in-cluster)
    ├── pointworld-pretrain.yaml        # PyTorchJob: 8x DDP pre-training
    ├── pointworld-eval.yaml            # Job: single-GPU evaluation
    └── trainer-v2/                     # Kubeflow Trainer v2 variants (TrainJob)
```

## Known limitations

- **Non-deterministic GPU eval**: small run-to-run variation is expected even
  with fixed seeds (upstream behavior).
- **Partial-batch eval sensitivity**: `--eval_num_batches < full dataset` is
  sensitive to `num_workers` / `eval_num_workers`; match these settings when
  comparing runs.
- **B200**: the container targets H200 (p5en.48xlarge). B200 EFA networking needs
  NCCL >= 2.29, which the base image predates; use a NeMo container with NCCL
  >= 2.29 for B200.
- **Gated DINOv3**: training and evaluation cannot run until the gated DINOv3
  weights are present on FSx (Section 3).

## Validation status

This test case was exercised end-to-end on Amazon EKS (Kubeflow Trainer v2.0.0)
with p5en.48xlarge (H200) nodes:

- **Container**: built from `pointworld.Dockerfile` and imported cleanly on an
  H200 (torch 2.5.1+cu124, flash-attn 2.7.4.post1, PTv3, DINOv3).
- **Pre-training**: a 1-node / 8-GPU run (BEHAVIOR, `--ptv3_size=large`,
  `--max_train_steps=2`) initialized distributed training, loaded the DINOv3
  backbone, streamed the WebDataset dataloader, and ran forward/backward with a
  decreasing loss before stopping cleanly.
- **Evaluation**: `eval.py` loaded a released checkpoint
  (`nvidia/PointWorld_models` `large-droid+behavior`) and produced metrics on the
  BEHAVIOR test split.

> [!note] Small-scale data
> DROID is distributed as a single ~3.9 TB split archive that does not subset
> cleanly. For small-scale validation, BEHAVIOR is sharded by task
> (~1.6-14 GB/task) and restores independently, making it the practical choice
> for smoke tests on a modest filesystem.

## References and attribution

- **Paper**: [PointWorld: Scaling 3D World Models for In-The-Wild Robotic
  Manipulation](https://arxiv.org/abs/2601.03782) (CVPR 2026 Highlight)
- **Project site**: <https://point-world.github.io/>
- **Code**: [github.com/NVlabs/PointWorld](https://github.com/NVlabs/PointWorld)
  (Apache-2.0)

PointWorld and its datasets/checkpoints are released by NVIDIA under their
respective licenses. DINOv3 weights are gated by Meta under the DINOv3 license.
The scripts and manifests in this directory are authored by AWS and licensed
MIT-0. If you use PointWorld in research, please cite:

```bibtex
@article{huang2026pointworld,
  title={PointWorld: Scaling 3D World Models for In-The-Wild Robotic Manipulation},
  author={Huang, Wenlong and Chao, Yu-Wei and Mousavian, Arsalan and Liu, Ming-Yu
          and Fox, Dieter and Mo, Kaichun and Li, Fei-Fei},
  journal={arXiv preprint arXiv:2601.03782},
  year={2026}
}
```
