# DETR-ResNet50 Object Detection Fine-tuning

Fine-tune a [DETR (DEtection TRansformer)](https://arxiv.org/abs/2005.12872) ResNet-50
model for object detection using PyTorch Distributed Data Parallel (DDP) on
Amazon SageMaker HyperPod with EKS orchestration.

This test case demonstrates distributed training of a computer vision object
detection model on a custom dataset (supermarket shelf images), using
[Qualcomm AI Hub](https://aihub.qualcomm.com/) pre-trained weights. The trained
model can subsequently be deployed to edge devices via Qualcomm AI Hub.

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Dataset](#dataset)
- [Training](#training)
  - [Basic Usage](#basic-usage)
  - [Command Line Arguments](#command-line-arguments)
- [Deployment](#deployment)
- [Architecture](#architecture)
  - [Model](#model)
  - [Training Configuration](#training-configuration)
  - [Distributed Training](#distributed-training)
- [Expected Results](#expected-results)
- [Customization](#customization)
- [References](#references)

## Overview

This test case fine-tunes a DETR-ResNet50 pre-trained on COCO to detect two
classes on supermarket shelf images:

- **Price** -- price tags and labels
- **Product** -- products on shelves

The pre-trained weights are loaded from
[facebook/detr-resnet-50](https://huggingface.co/facebook/detr-resnet-50) on
HuggingFace Hub via the
[Qualcomm AI Hub](https://aihub.qualcomm.com/models/detr_resnet50) model wrapper.
The trained model can subsequently be deployed to edge devices via Qualcomm AI Hub.

The training uses PyTorch DDP via Kubeflow PyTorchJob for distributed training
across multiple GPU nodes connected with EFA networking.

## Prerequisites

- An Amazon SageMaker HyperPod EKS cluster or Amazon EKS cluster with GPU nodes
  (e.g., `ml.g5.8xlarge`), accessible via `kubectl`. We recommend setting up the
  cluster using the templates in [architectures](../../../architectures).
- An Amazon FSx for Lustre persistent volume claim (default name: `fsx-pvc`; see
  [kubernetes/README.md](kubernetes/README.md) if your cluster uses a different
  PVC name).
- [Kubeflow Training Operator](https://www.kubeflow.org/docs/components/training/pytorch/)
  deployed to your cluster (pre-installed on SageMaker HyperPod EKS).
- Docker installed on a build machine with internet access (the Docker build
  downloads model weights from HuggingFace Hub).
- AWS CLI configured with ECR access.

## Dataset

This test case uses the **Supermarket Shelves** dataset (45 images, 2 classes,
CC0 license). See [data/README.md](data/README.md) for download and preparation
instructions.

## Training

### Basic Usage

To run training locally with a single GPU:

```bash
python detr_main.py /path/to/data --epochs 50 --batch-size 4 --lr 1e-4 --num-classes 2
```

To run distributed training with `torchrun`:

```bash
torchrun --nproc_per_node=1 --nnodes=2 detr_main.py /path/to/data \
    --epochs 50 \
    --batch-size 4 \
    --lr 1e-4 \
    --num-classes 2
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `data` | `data` | Path to dataset directory |
| `--arch` | `detr-resnet50` | Model architecture |
| `--epochs` | `50` | Number of training epochs |
| `--batch-size` | `8` | Mini-batch size per GPU (YAML template uses 4) |
| `--lr` | `1e-4` | Initial learning rate |
| `--weight-decay` | `1e-4` | Weight decay |
| `--num-classes` | `2` | Number of object classes |
| `--workers` | `4` | Data loading workers |
| `--resume` | | Path to checkpoint for resuming |
| `--evaluate` | `false` | Evaluate only (no training) |
| `--seed` | | Random seed for reproducibility |
| `--print-freq` | `10` | Print frequency (batches) |

## Deployment

We provide a guide for Kubernetes (EKS). See the [kubernetes](kubernetes)
subdirectory for detailed deployment instructions including container build,
ECR push, and PyTorchJob submission.

## Architecture

### Model

The model is based on [DETR (End-to-End Object Detection with Transformers)](https://arxiv.org/abs/2005.12872):

1. **Backbone**: ResNet-50 feature extractor (pre-trained on ImageNet)
2. **Transformer**: DETR encoder-decoder with 100 object queries
3. **Detection Heads**: Custom classification head (num_classes + 1 for
   background) and 3-layer MLP bounding box regression head

Pre-trained DETR-ResNet50 weights are loaded from
[facebook/detr-resnet-50](https://huggingface.co/facebook/detr-resnet-50) on
HuggingFace Hub via the Qualcomm AI Hub model wrapper. The weights are baked
into the Docker image at build time so that training nodes do not require
internet access. The model is then wrapped with custom detection heads
(`QAIHubDETRWrapper`) that replace the original 91-class COCO heads.

### Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | AdamW | With weight decay |
| Learning Rate | 1e-4 | StepLR decay (step=30, gamma=0.1) |
| Batch Size | 4 per GPU | Optimized for DETR memory requirements |
| Image Size | 800x800 | DETR standard input resolution |
| Loss | CrossEntropy + 5x L1 bbox | Classification + weighted bbox regression |
| Augmentation | ColorJitter | HFlip omitted -- standard transform doesn't flip box coords |
| Evaluation | torchmetrics mAP | COCO-style mean average precision |

### Distributed Training

- **Strategy**: PyTorch DistributedDataParallel (DDP)
- **Backend**: NCCL (GPU-to-GPU communication)
- **Networking**: EFA (Elastic Fabric Adapter) for high-bandwidth inter-node
  communication
- **Orchestration**: Kubeflow PyTorchJob with elastic scaling (2-36 replicas)
- **Storage**: FSx for Lustre shared filesystem for data and checkpoints

## Expected Results

With the default configuration (50 epochs, 2 workers, batch size 4, lr 1e-4)
on `ml.g5.8xlarge` instances:

| Metric | Value |
|--------|-------|
| Final Validation Loss | ~1.24 |
| Dataset | 36 train / 9 val images |

**Note**: The small dataset size (45 images) is intentional for workshop/demo
purposes. The training loss uses simplified positional matching rather than
DETR's standard Hungarian matching, which is sufficient for this demo. For
production use cases, a larger dataset and the Hungarian matching loss from the
DETR paper (or `DetrForObjectDetection`'s built-in loss) are recommended.

### Output Files

Checkpoints are saved to the directory specified by `CHECKPOINT_DIR` environment
variable (script default: `/tmp/checkpoints`; the Kubernetes deployment sets
this to `/fsx/checkpoint`):

- `checkpoint.pth.tar` -- Latest checkpoint (for resuming training)
- `model_best.pth.tar` -- Best checkpoint by validation loss
- `training_stats.txt` -- Training summary statistics

## Customization

### Different Classes

Edit `meta.json` (place at `<data-dir>/meta.json` -- see [data/README.md](data/README.md)) to define your own classes:

```json
{
    "classes": [
        {"title": "YourClass1", "id": 1},
        {"title": "YourClass2", "id": 2}
    ]
}
```

Update `--num-classes` accordingly.

### Training Parameters

Adjust via command line arguments or modify the YAML template:

```bash
--epochs=100       # More training epochs
--batch-size=8     # Larger batches (if GPU memory allows)
--lr=5e-5          # Lower learning rate
--num-classes=3    # More classes
```

### Resume Training

```bash
python detr_main.py /path/to/data --resume /path/to/checkpoint.pth.tar
```

## References

- [DETR: End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872) (Carion et al., 2020)
- [Qualcomm AI Hub - DETR-ResNet50](https://aihub.qualcomm.com/models/detr_resnet50)
- [Amazon SageMaker HyperPod](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod.html)
- [Kubeflow PyTorchJob](https://www.kubeflow.org/docs/components/training/pytorch/)
- [PyTorch Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Supermarket Shelves Dataset](https://humansintheloop.org/resources/datasets/supermarket-shelves-dataset/)
