# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""DETR-ResNet50 Object Detection Training Script

Fine-tunes a DETR-ResNet50 model from Qualcomm AI Hub on a custom object detection
dataset using PyTorch Distributed Data Parallel (DDP). Designed for distributed
training on Amazon SageMaker HyperPod with EKS orchestration.

Based on the PyTorch ImageNet training example structure, adapted for DETR
object detection with Qualcomm AI Hub model integration.

Usage:
    # Single GPU
    python detr_main.py /path/to/data --epochs 50 --batch-size 4

    # Distributed (via torchrun)
    torchrun --nproc_per_node=1 --nnodes=2 detr_main.py /path/to/data \
        --epochs 50 --batch-size 4 --lr 1e-4 --num-classes 2

Reference:
    - DETR Paper: https://arxiv.org/abs/2005.12872
    - Qualcomm AI Hub: https://aihub.qualcomm.com/
"""

import argparse
import os
import random
import shutil
import time
import json
from pathlib import Path
from typing import List
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
from PIL import Image

try:
    from torchmetrics.detection import MeanAveragePrecision
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    print("[WARN] torchmetrics not available. mAP evaluation will be disabled.")

from qai_hub_models.models.detr_resnet50 import Model as DETRModel

print("[INFO] Using QAI Hub DETR-ResNet50 model")

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="DETR Object Detection Training")
parser.add_argument(
    "data", metavar="DIR", nargs="?", default="data",
    help="path to dataset (default: data)",
)
parser.add_argument(
    "--arch", metavar="ARCH", default="detr-resnet50",
    help="model architecture (default: detr-resnet50)",
)
parser.add_argument(
    "-j", "--workers", default=4, type=int, metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "--epochs", default=50, type=int, metavar="N",
    help="number of total epochs to run",
)
parser.add_argument(
    "--start-epoch", default=0, type=int, metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b", "--batch-size", default=8, type=int, metavar="N",
    help="mini-batch size per GPU (default: 8)",
)
parser.add_argument(
    "--lr", "--learning-rate", default=1e-4, type=float, metavar="LR",
    help="initial learning rate", dest="lr",
)
parser.add_argument(
    "--weight-decay", default=1e-4, type=float, metavar="W",
    help="weight decay (default: 1e-4)",
)
parser.add_argument(
    "-p", "--print-freq", default=10, type=int, metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--resume", default="", type=str, metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "-e", "--evaluate", dest="evaluate", action="store_true",
    help="evaluate model on validation set",
)
parser.add_argument(
    "--world-size", default=-1, type=int,
    help="number of nodes for distributed training",
)
parser.add_argument(
    "--rank", default=-1, type=int,
    help="node rank for distributed training",
)
parser.add_argument(
    "--dist-url", default="env://", type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str,
    help="distributed backend",
)
parser.add_argument(
    "--seed", default=None, type=int,
    help="seed for initializing training",
)
parser.add_argument(
    "--gpu", default=None, type=int,
    help="GPU id to use",
)
parser.add_argument(
    "--multiprocessing-distributed", action="store_true",
    help="use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs",
)
parser.add_argument(
    "--num-classes", default=2, type=int,
    help="number of object classes (default: 2 for Price and Product)",
)

best_loss = float("inf")
training_start_time = None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SupermarketDataset(torch.utils.data.Dataset):
    """Dataset for supermarket shelf object detection.

    Reads Supervisely-format JSON annotations and corresponding images.
    Performs an 80/20 train/validation split based on sorted file order.
    """

    def __init__(self, data_dir: str, split: str = "train", transforms=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transforms = transforms

        self.images_dir = self.data_dir / "Supermarket shelves" / "images"
        self.annotations_dir = self.data_dir / "Supermarket shelves" / "annotations"

        # Load class definitions from meta.json
        meta_path = self.data_dir / "meta.json"
        with open(meta_path, "r") as f:
            meta = json.load(f)

        # Class mapping: +1 to reserve index 0 for background (no-object)
        self.classes = {cls["title"]: idx + 1 for idx, cls in enumerate(meta["classes"])}
        self.id_to_class = {cls["id"]: cls["title"] for cls in meta["classes"]}
        self.num_classes = len(self.classes)

        print(f"[INFO] Classes: {self.classes}")

        # Sort for reproducible 80/20 split
        annotation_files = sorted(self.annotations_dir.glob("*.json"))
        split_idx = int(0.8 * len(annotation_files))

        if split == "train":
            self.annotation_files = annotation_files[:split_idx]
        else:
            self.annotation_files = annotation_files[split_idx:]

        print(f"[INFO] Loaded {len(self.annotation_files)} {split} samples")

    def __len__(self):
        return len(self.annotation_files)

    def __getitem__(self, idx):
        ann_file = self.annotation_files[idx]
        with open(ann_file, "r") as f:
            annotation = json.load(f)

        # Resolve image path from annotation filename (e.g. "001.jpg.json" -> "001.jpg")
        img_name = ann_file.name.removesuffix(".json")
        img_path = self.images_dir / img_name
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        original_size = image.size  # (width, height)

        # Parse bounding boxes and labels from Supervisely format
        boxes: List[list] = []
        labels: List[int] = []

        for obj in annotation["objects"]:
            class_title = obj["classTitle"]
            if class_title in self.classes:
                points = obj["points"]["exterior"]
                x1, y1 = points[0]
                x2, y2 = points[1]

                # Ensure correct ordering
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)

                boxes.append([x1, y1, x2, y2])
                labels.append(self.classes[class_title])

        boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
        labels_t = torch.as_tensor(labels, dtype=torch.int64)

        # Normalize to [0, 1] and convert to DETR center format [cx, cy, w, h]
        if len(boxes_t) > 0:
            img_w, img_h = original_size
            boxes_t[:, [0, 2]] /= img_w
            boxes_t[:, [1, 3]] /= img_h

            x1, y1, x2, y2 = boxes_t.unbind(-1)
            boxes_t = torch.stack(
                [
                    (x1 + x2) / 2,  # center_x
                    (y1 + y2) / 2,  # center_y
                    x2 - x1,  # width
                    y2 - y1,  # height
                ],
                dim=-1,
            )

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx]),
            "area": boxes_t[:, 2] * boxes_t[:, 3] if len(boxes_t) > 0 else torch.tensor([]),
            "iscrowd": torch.zeros((len(boxes_t),), dtype=torch.int64),
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target


def collate_fn(batch):
    """Custom collate for variable-size target dicts."""
    images, targets = tuple(zip(*batch))
    images = torch.stack(images, dim=0)
    return images, targets


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """Simple multi-layer perceptron (feed-forward network)."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = torch.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class QAIHubDETRWrapper(nn.Module):
    """Wrapper around QAI Hub DETR-ResNet50 with custom detection heads.

    Replaces the original 91-class COCO classification head with a custom
    head for the target number of classes. Keeps the ResNet-50 backbone and
    DETR transformer from the pre-trained model.

    Args:
        qai_model: Pre-trained QAI Hub DETR model instance.
        num_classes: Number of object classes (excluding background).
    """

    def __init__(self, qai_model, num_classes):
        super().__init__()
        self.qai_model = qai_model
        self.num_classes = num_classes

        # DETR uses 256 hidden dimensions
        hidden_dim = 256

        # Custom classification head: num_classes + 1 (background / no-object)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        # Custom bounding box head: 3-layer MLP -> 4 coords
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        nn.init.normal_(self.class_embed.weight, std=0.01)
        nn.init.constant_(self.class_embed.bias, 0)

    def forward(self, images):
        device = next(self.parameters()).device
        if images.device != device:
            images = images.to(device)

        # Access the underlying HuggingFace DETR model inside the QAI Hub wrapper
        if hasattr(self.qai_model, "model"):
            outputs = self.qai_model.model(pixel_values=images)
        else:
            outputs = self.qai_model(images)

        # Extract hidden states from transformer decoder output.
        # DetrForObjectDetection returns DetrObjectDetectionOutput with
        # last_hidden_state as the 256-dim decoder hidden states.
        hidden_states = outputs.last_hidden_state
        assert hidden_states.shape[-1] == 256, (
            f"Expected 256-dim decoder hidden states, got {hidden_states.shape[-1]}. "
            f"Output type: {type(outputs)}"
        )

        pred_logits = self.class_embed(hidden_states)
        pred_boxes = self.bbox_embed(hidden_states)

        return {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
        }


def create_model(num_classes: int):
    """Create a DETR model with custom detection heads.

    Loads the pre-trained DETR-ResNet50 from Qualcomm AI Hub and wraps it
    with custom classification and bounding box heads for the specified
    number of classes.

    Args:
        num_classes: Number of target object classes (excluding background).

    Returns:
        QAIHubDETRWrapper model instance.
    """
    print("[INFO] Loading QAI Hub DETR-ResNet50 model...")
    qai_model = DETRModel.from_pretrained()
    print("[OK] QAI Hub DETR-ResNet50 model loaded successfully")

    model = QAIHubDETRWrapper(qai_model, num_classes)
    print(f"[OK] DETR model wrapped for {num_classes} classes")
    return model


# ---------------------------------------------------------------------------
# Evaluation with torchmetrics
# ---------------------------------------------------------------------------


def cxcywh_to_xyxy(boxes):
    """Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2] format."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def evaluate_map(model, val_loader, args):
    """Evaluate model using torchmetrics MeanAveragePrecision.

    Collects predictions and targets across the full validation set and
    computes COCO-style mAP metrics. Note: with DistributedSampler, each
    rank evaluates its own shard, so the reported mAP is rank-local.

    Args:
        model: Trained model.
        val_loader: Validation DataLoader.
        args: Parsed arguments.

    Returns:
        Dictionary of mAP metrics (map, map_50, map_75, etc.)
    """
    if not TORCHMETRICS_AVAILABLE:
        print("[WARN] torchmetrics not available, skipping mAP evaluation")
        return {}

    metric = MeanAveragePrecision(iou_type="bbox")
    model.eval()

    with torch.no_grad():
        for images, targets in val_loader:
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            outputs = model(images)
            pred_logits = outputs["pred_logits"]  # [B, num_queries, num_classes+1]
            pred_boxes = outputs["pred_boxes"]  # [B, num_queries, 4]

            batch_size = pred_logits.shape[0]

            for i in range(batch_size):
                logits = pred_logits[i].cpu()  # [num_queries, num_classes+1]
                boxes = pred_boxes[i].cpu()  # [num_queries, 4]

                # Get class probabilities (excluding background class 0)
                probs = torch.softmax(logits, dim=-1)
                # Background is index 0; object classes are 1..num_classes
                obj_probs = probs[:, 1:]  # [num_queries, num_classes]
                scores, pred_labels = obj_probs.max(dim=-1)

                # Filter by confidence threshold
                keep = scores > 0.1
                pred_boxes_xyxy = cxcywh_to_xyxy(boxes[keep])
                pred_scores = scores[keep]
                pred_cls = pred_labels[keep] + 1  # shift back to 1-indexed

                # Target boxes: convert from cxcywh to xyxy
                target = targets[i]
                gt_boxes = target["boxes"]
                gt_labels = target["labels"]
                if len(gt_boxes) > 0:
                    gt_boxes_xyxy = cxcywh_to_xyxy(gt_boxes)
                else:
                    gt_boxes_xyxy = torch.zeros((0, 4))

                preds = [
                    {
                        "boxes": pred_boxes_xyxy,
                        "scores": pred_scores,
                        "labels": pred_cls,
                    }
                ]
                tgts = [
                    {
                        "boxes": gt_boxes_xyxy,
                        "labels": gt_labels,
                    }
                ]

                metric.update(preds, tgts)

    results = metric.compute()
    is_rank_zero = (not args.distributed) or args.rank == 0
    if is_rank_zero:
        print("[INFO] mAP Evaluation Results:")
        for key in ["map", "map_50", "map_75", "map_small", "map_medium", "map_large"]:
            val = results.get(key, torch.tensor(float("nan")))
            print(f"  {key}: {val.item():.4f}")

    return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in results.items()}


# ---------------------------------------------------------------------------
# Training and validation loops
# ---------------------------------------------------------------------------


def train(train_loader, model, criterion, optimizer, epoch, args):
    """Run one training epoch."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch [{}/{}] Train: ".format(epoch + 1, args.epochs),
    )

    model.train()

    end = time.time()
    for i, (images, targets) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        outputs = model(images)

        # Compute loss over all annotated objects per sample.
        # NOTE: This uses simplified positional matching (first N queries vs first N GT
        # objects) rather than DETR's standard Hungarian matching. This is acceptable for
        # this workshop/demo; for production use cases, consider using the bipartite
        # matching loss from the DETR paper or DetrForObjectDetection's built-in loss.
        total_loss = 0.0
        valid_samples = 0

        pred_logits = outputs["pred_logits"]  # [B, num_queries, num_classes+1]
        pred_boxes = outputs["pred_boxes"]  # [B, num_queries, 4]

        for batch_idx, target in enumerate(targets):
            if len(target["labels"]) == 0:
                continue

            num_objects = min(len(target["labels"]), pred_logits.shape[1])

            # Classification loss
            sample_logits = pred_logits[batch_idx, :num_objects, :]
            sample_labels = target["labels"][:num_objects].cuda(args.gpu)
            class_loss = criterion(sample_logits, sample_labels)

            # Bounding box L1 loss
            sample_boxes = pred_boxes[batch_idx, :num_objects, :]
            target_boxes = target["boxes"][:num_objects, :].cuda(args.gpu)
            bbox_loss = nn.functional.l1_loss(sample_boxes, target_boxes)

            # Combined loss (bbox weighted higher for detection)
            combined_loss = class_loss + 5.0 * bbox_loss
            total_loss += combined_loss
            valid_samples += 1

        # Skip batch if no valid targets (avoids backward on empty graph)
        if valid_samples == 0:
            continue
        loss = total_loss / valid_samples

        losses.update(loss.item(), images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args, epoch=None):
    """Run validation and return average loss."""
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")

    epoch_prefix = f"Epoch {epoch + 1} Validation: " if epoch is not None else "Test: "
    progress = ProgressMeter(len(val_loader), [batch_time, losses], prefix=epoch_prefix)

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, targets) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            outputs = model(images)

            # Compute validation loss (same as training)
            total_loss = 0.0
            valid_samples = 0

            pred_logits = outputs["pred_logits"]
            pred_boxes = outputs["pred_boxes"]

            for batch_idx, target in enumerate(targets):
                if len(target["labels"]) == 0:
                    continue

                num_objects = min(len(target["labels"]), pred_logits.shape[1])

                sample_logits = pred_logits[batch_idx, :num_objects, :]
                sample_labels = target["labels"][:num_objects].cuda(args.gpu)
                class_loss = criterion(sample_logits, sample_labels)

                sample_boxes = pred_boxes[batch_idx, :num_objects, :]
                target_boxes = target["boxes"][:num_objects, :].cuda(args.gpu)
                bbox_loss = nn.functional.l1_loss(sample_boxes, target_boxes)

                combined_loss = class_loss + 5.0 * bbox_loss
                total_loss += combined_loss
                valid_samples += 1

            if valid_samples > 0:
                loss = total_loss / valid_samples
            else:
                loss = torch.tensor(0.0).cuda(args.gpu)

            losses.update(loss.item(), images.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(f" * Validation Loss {losses.avg:.4f}")

    return losses.avg


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(state, is_best, filename=None):
    """Save training checkpoint with cleaned state dict for inference.

    Saves both the original state_dict (with DDP/wrapper prefixes for
    training resume) and a cleaned version (prefixes stripped, for
    standalone inference).
    """
    if filename is None:
        checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "/tmp/checkpoints")
        filename = os.path.join(checkpoint_dir, "checkpoint.pth.tar")

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if "state_dict" in state:
        # state_dict keeps DDP/wrapper prefixes for direct resume (standard convention).
        # inference_state_dict strips prefixes for standalone export/inference.
        original_state_dict = state["state_dict"]
        cleaned_state_dict = {}
        for key, value in original_state_dict.items():
            new_key = key.replace("module.", "")  # Remove DDP prefix
            new_key = new_key.replace("qai_model.", "")  # Remove wrapper prefix
            cleaned_state_dict[new_key] = value

        state_to_save = state.copy()
        state_to_save["state_dict"] = original_state_dict
        state_to_save["inference_state_dict"] = cleaned_state_dict

        torch.save(state_to_save, filename)
    else:
        torch.save(state, filename)

    print(f"[OK] Checkpoint saved to: {filename}")

    if is_best:
        best_filename = os.path.join(os.path.dirname(filename), "model_best.pth.tar")
        shutil.copyfile(filename, best_filename)
        print(f"[OK] Best model saved to: {best_filename}")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def format_time(seconds):
    """Format seconds as HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}h:{minutes:02d}m:{secs:02d}s"


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """Displays training progress."""

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 1

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_loss
    args.gpu = gpu

    # When launched via torchrun, LOCAL_RANK is set automatically
    if args.gpu is None and "LOCAL_RANK" in os.environ:
        args.gpu = int(os.environ["LOCAL_RANK"])

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    # Create model
    print(f"[INFO] Creating model '{args.arch}' with {args.num_classes} classes")
    model = create_model(args.num_classes)

    if not torch.cuda.is_available():
        print("[WARN] Using CPU, this will be slow")
    elif args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            # find_unused_parameters=True is required because the QAI Hub DETR wrapper
            # replaces the original 92-class COCO detection heads with custom heads, but
            # the original heads (class_labels_classifier, bbox_predictor) remain in the
            # model's parameter list. They are never used in forward(), so DDP would
            # otherwise error with "Expected to have finished reduction". This flag has
            # no effect on accuracy -- it only tells DDP to skip unused params in the
            # gradient all-reduce.
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # See comment above for find_unused_parameters rationale.
            model = torch.nn.parallel.DistributedDataParallel(
                model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # Resume from checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"[INFO] Loading checkpoint '{args.resume}'")
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                checkpoint = torch.load(args.resume, map_location=f"cuda:{args.gpu}")
            args.start_epoch = checkpoint["epoch"]
            best_loss = checkpoint.get("best_loss", checkpoint.get("best_map", float("inf")))
            model.load_state_dict(checkpoint["state_dict"])

            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            print(f"[OK] Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"[WARN] No checkpoint found at '{args.resume}'")

    # Data loading
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transforms = transforms.Compose(
        [
            transforms.Resize((800, 800)),
            # Note: RandomHorizontalFlip is intentionally omitted because the
            # standard torchvision transform flips images but not bounding box
            # coordinates, leading to misaligned image-box pairs.
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            normalize,
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize((800, 800)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = SupermarketDataset(args.data, "train", train_transforms)
    val_dataset = SupermarketDataset(args.data, "val", val_transforms)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        collate_fn=collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
        collate_fn=collate_fn,
    )

    if args.evaluate:
        validate(val_loader, model, criterion, args, epoch=0)
        evaluate_map(model, val_loader, args)
        return

    # Training loop
    global training_start_time
    training_start_time = time.time()
    print(f"[INFO] Starting training for {args.epochs} epochs...")
    print(f"[INFO] Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")
    print(f"[INFO] Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    final_val_loss = float("inf")

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train(train_loader, model, criterion, optimizer, epoch, args)
        final_val_loss = validate(val_loader, model, criterion, args, epoch)

        # All-reduce validation loss across ranks so all ranks agree on is_best.
        # Each rank only sees its DistributedSampler shard of the val set.
        if args.distributed:
            loss_tensor = torch.tensor([final_val_loss], device=f"cuda:{args.gpu}")
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            final_val_loss = loss_tensor.item()

        scheduler.step()

        # Track best validation loss
        is_best = final_val_loss < best_loss
        best_loss = min(final_val_loss, best_loss)

        is_rank_zero = (not args.distributed) or args.rank == 0
        if is_rank_zero:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "best_loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                is_best,
            )

    # Training complete -- final evaluation (rank 0 only for printing)
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time

    is_rank_zero = (not args.distributed) or args.rank == 0

    if is_rank_zero:
        print("=" * 80)
        print("TRAINING COMPLETED")
        print("=" * 80)

    # Compute proper mAP using torchmetrics (runs on all ranks, prints on rank 0)
    print("[INFO] Computing final mAP on validation set...")
    map_results = evaluate_map(model, val_loader, args)
    final_map = map_results.get("map", 0.0) if map_results else 0.0
    final_map_50 = map_results.get("map_50", 0.0) if map_results else 0.0

    # Print summary and save stats (rank 0 only)
    if is_rank_zero:
        print(f"FINAL TRAINING STATISTICS:")
        print(f"  Total Training Time: {format_time(total_training_time)}")
        print(f"  Training Epochs: {args.epochs}")
        print(f"  Final Validation Loss: {final_val_loss:.4f}")
        print(f"  Best Validation Loss: {best_loss:.4f}")
        print(f"  mAP: {final_map:.4f}")
        print(f"  mAP@0.5: {final_map_50:.4f}")
        print(f"  Dataset Size: {len(train_dataset)} train, {len(val_dataset)} val")
        print(f"  Model: QAI Hub DETR-ResNet50")
        print(f"  Classes: {args.num_classes}")
        print(f"  Batch Size: {args.batch_size}, LR: {args.lr}")
        print(f"  Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # Save statistics to file
        stats_file = os.path.join(
            os.environ.get("CHECKPOINT_DIR", "/tmp/checkpoints"), "training_stats.txt"
        )
        try:
            os.makedirs(os.path.dirname(stats_file), exist_ok=True)
            with open(stats_file, "w") as f:
                f.write("DETR-ResNet50 Training Statistics\n")
                f.write("=" * 50 + "\n")
                f.write(f"Total Training Time: {format_time(total_training_time)}\n")
                f.write(f"Training Epochs: {args.epochs}\n")
                f.write(f"Final Validation Loss: {final_val_loss:.4f}\n")
                f.write(f"Best Validation Loss: {best_loss:.4f}\n")
                f.write(f"mAP: {final_map:.4f}\n")
                f.write(f"mAP@0.5: {final_map_50:.4f}\n")
                f.write(f"Dataset Size: {len(train_dataset)} train, {len(val_dataset)} val\n")
                f.write(f"Model: QAI Hub DETR-ResNet50\n")
                f.write(f"Classes: {args.num_classes}\n")
                f.write(f"Batch Size: {args.batch_size}, LR: {args.lr}\n")
                f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d')}\n")
            print(f"[OK] Training statistics saved to: {stats_file}")
        except Exception as e:
            print(f"[WARN] Could not save statistics file: {e}")

    # Clean up distributed process group
    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
