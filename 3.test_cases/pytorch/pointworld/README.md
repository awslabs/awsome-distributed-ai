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

> This test case is **Kubernetes-only** (EKS / HyperPod EKS). Slurm and other
> orchestrators are not included here.

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

## 2. DINOv3 (gated dependency)

PointWorld's scene featurizer uses a **DINOv3 ViT-L/16** backbone whose weights
are gated by Meta. You must request access and obtain a personal download URL:

1. Request access at <https://github.com/facebookresearch/dinov3>.
2. From the access email, copy the `dinov3_vitl16` pre-train checkpoint URL.
3. Download it onto the shared filesystem:

   ```bash
   ./scripts/2.download_dinov3.sh "<DINOV3_DOWNLOAD_URL>" \
       /fsx/$USER/pointworld/dinov3/checkpoints
   ```

The DINOv3 weights are **never baked into the container image**. The Kubernetes
manifests symlink them from FSx into the path PointWorld expects at pod start.

## 3. Data pipeline (DROID + BEHAVIOR)

PointWorld distributes generated datasets as packaged archives on Hugging Face,
which you restore and convert to WebDataset (WDS) shards before training.

### 3.1 Download

```bash
python scripts/0.download_dataset.py \
    --output_dir /fsx/$USER/pointworld/downloads \
    --datasets droid behavior
```

> DROID full-dataset restoration is multi-terabyte. The DROID flow package is
> split into independent shards, so you can use `--allow_patterns` to fetch a
> subset for development before committing to the full download.

### 3.2 Restore

Each dataset repo ships `recover_dataset_from_parts.sh`. Run it inside each
downloaded dataset directory to reassemble the restored dataset root (see the
upstream [PointWorld README](https://github.com/NVlabs/PointWorld#datasets-and-checkpoints)).

### 3.3 Convert to WebDataset

The H5-to-WDS conversion + integrity tooling lives on the PointWorld `data`
branch. Check it out, then use the wrapper:

```bash
git clone https://github.com/NVlabs/PointWorld.git /fsx/$USER/pointworld/PointWorld-data
cd /fsx/$USER/pointworld/PointWorld-data && git checkout data && cd -

python scripts/1.convert_wds.py \
    --pointworld_data_branch /fsx/$USER/pointworld/PointWorld-data \
    --restored_root /fsx/$USER/pointworld/downloads/droid/pointworld_droid_restored \
    --domain droid \
    --output_root /fsx/$USER/pointworld/dataset
# repeat with --domain behavior
```

Resulting layout (consumed by training via `LOCAL_DATASET_DIR`):

```text
/fsx/$USER/pointworld/dataset/
├── droid/wds/{train,test}/...
└── behavior/wds/{train,test}/...
```

For **DROID filtered metrics**, copy the released expert-confidence artifact into
the generated WDS test split:

```bash
cp .../droid/confidence/expert_confidence-seed=42.h5 \
   /fsx/$USER/pointworld/dataset/droid/wds/test/expert_confidence-seed=42.h5
```

## 4. Build and push the container

```bash
docker build -t pointworld:05484826 -f pointworld.Dockerfile .
```

Then push to ECR (see [`kubernetes/README.md`](./kubernetes/README.md#push-the-image-to-ecr)
for the full login/tag/push sequence). The image tag matches `POINTWORLD_COMMIT`
in the Dockerfile so the running image is traceable to an exact upstream commit.

## 5. Pre-train (Kubernetes PyTorchJob)

```bash
# Update <ACCOUNT_ID> and <REGION> in kubernetes/pointworld-pretrain.yaml first.
kubectl apply -f kubernetes/pointworld-pretrain.yaml
kubectl logs -f pointworld-pretrain-worker-0 -n kubeflow
```

This launches 8 worker pods (one per node), each running `torchrun` with 8
processes (one per H200) for 64-way data-parallel pre-training. The flagship flag
set (large PTv3, DROID + BEHAVIOR) is encoded in the manifest and mirrored in
[`configs/train-large-droid-behavior.env`](./configs/train-large-droid-behavior.env).

See [`kubernetes/README.md`](./kubernetes/README.md) for the launch-pattern
details, FSx layout, and tuning notes.

## 6. Evaluate (Kubernetes Job)

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

## 7. Parse throughput

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
    ├── README.md                      # EKS/HyperPod-EKS specifics
    ├── pointworld-pretrain.yaml       # PyTorchJob: 8x DDP pre-training
    └── pointworld-eval.yaml           # Job: single-GPU evaluation
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
  weights are present on FSx (Section 2).

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
