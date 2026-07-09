"""
Evaluate fine-tuned π0 against the base π0 on a held-out test set.

Supports DROID and LIBERO datasets. Use --dataset to select.

Usage (defaults to droid):
    python evaluate_pi0.py

Evaluate on LIBERO:
    python evaluate_pi0.py --dataset libero

With explicit paths:
    python evaluate_pi0.py \
        --dataset droid \
        --finetuned-path /path/to/pretrained_model \
        --num-trajectories 5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


# -------------------------------------------------------------------------
# Per-dataset configuration (mirrors 02_training_job.ipynb)
# -------------------------------------------------------------------------
DATASET_CONFIGS = {
    "droid": {
        "rename_map": {
            "observation.images.exterior_image_1_left": "observation.images.base_0_rgb",
            "observation.images.wrist_image_left": "observation.images.wrist_0_rgb",
        },
        "empty_cameras": 1,
        "s3_suffix": "droid_100_test",
        "repo_id": "droid_local_test",
        "action_dim": 7,
        "description": "DROID (7-DoF real joint actions)",
    },
    "libero": {
        "rename_map": {
            # DROID-style keys (from some re-encoded LIBERO variants)
            "observation.images.exterior_image_1_left": "observation.images.base_0_rgb",
            "observation.images.wrist_image_left": "observation.images.left_wrist_0_rgb",
            # LIBERO native keys (from lerobot/libero_10 direct download)
            "observation.images.image": "observation.images.base_0_rgb",
            "observation.images.wrist_image": "observation.images.left_wrist_0_rgb",
        },
        "empty_cameras": 1,
        "s3_suffix": "libero_10_test",
        "repo_id": "libero_local_test",
        "action_dim": 7,
        "description": "LIBERO-10 (7-DoF tabletop manipulation)",
    },
}


# -------------------------------------------------------------------------
# Hub-skip monkey patch
# -------------------------------------------------------------------------
def apply_local_dataset_patch():
    """Stop LeRobot from pinging the Hub for our local-only repo_ids."""
    from lerobot.datasets import utils as lr_utils
    try:
        from lerobot.datasets import dataset_metadata as lr_meta
    except ImportError:
        lr_meta = None

    orig_safe = lr_utils.get_safe_version
    orig_repo = lr_utils.get_repo_versions

    def _is_local(repo_id):
        if os.environ.get("HF_HUB_OFFLINE", "0") in ("1", "true", "True"):
            return True
        return "/" not in repo_id

    def patched_safe(repo_id, version):
        if _is_local(repo_id):
            v = str(version)
            return v if v.startswith("v") else f"v{v}"
        return orig_safe(repo_id, version)

    def patched_repo(repo_id):
        if _is_local(repo_id):
            return []
        return orig_repo(repo_id)

    lr_utils.get_safe_version = patched_safe
    lr_utils.get_repo_versions = patched_repo
    if lr_meta is not None and hasattr(lr_meta, "get_safe_version"):
        lr_meta.get_safe_version = patched_safe

    import lerobot.datasets.lerobot_dataset as _lr_ds
    if hasattr(_lr_ds, "get_safe_version"):
        _lr_ds.get_safe_version = patched_safe
    if hasattr(_lr_ds, "get_repo_versions"):
        _lr_ds.get_repo_versions = patched_repo


# -------------------------------------------------------------------------
# Test dataset download + prep
# -------------------------------------------------------------------------
def ensure_test_dataset_local(s3_uri: str, local_dir: Path) -> Path:
    if (local_dir / "meta" / "info.json").exists():
        print(f"  test dataset already present at {local_dir}")
        return local_dir
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"  syncing {s3_uri} -> {local_dir}")
    subprocess.run(["aws", "s3", "sync", s3_uri, str(local_dir), "--quiet"], check=True)
    return local_dir


def setup_lerobot_cache(test_dir: Path, repo_id: str) -> Path:
    cache_root = Path(os.environ.get("HF_LEROBOT_HOME") or
                      (Path.home() / ".lerobot_cache"))
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_LEROBOT_HOME"] = str(cache_root)
    link = cache_root / repo_id
    if link.is_symlink() or link.exists():
        link.unlink() if link.is_symlink() else None
    link.symlink_to(test_dir.resolve())
    return link


# -------------------------------------------------------------------------
# Backfill empty FSDP-frozen params from the base checkpoint
# -------------------------------------------------------------------------
def backfill_empty_params(policy, source_path: str, base_repo_id: str = "lerobot/pi0_base"):
    empty_keys = [k for k, v in policy.state_dict().items() if v.numel() == 0]
    if not empty_keys:
        return
    print(f"    Backfilling {len(empty_keys)} empty FSDP-frozen params from base...")
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    base_dir = Path(snapshot_download(base_repo_id, allow_patterns=["*.safetensors", "*.json"]))
    base_st_files = list(base_dir.glob("*.safetensors")) + list(base_dir.glob("model-*.safetensors"))
    if not base_st_files:
        raise RuntimeError(f"Could not find base model safetensors in {base_dir}")

    base_state = {}
    for sf in base_st_files:
        base_state.update(load_file(sf, device="cpu"))

    cur_state = policy.state_dict()
    new_state = {}
    n_filled = 0
    for k, v in cur_state.items():
        if v.numel() == 0 and k in base_state:
            new_state[k] = base_state[k].to(v.dtype)
            n_filled += 1
        else:
            new_state[k] = v
    policy.load_state_dict(new_state, strict=False, assign=True)
    print(f"    Backfilled {n_filled}/{len(empty_keys)} empty params from base")


# -------------------------------------------------------------------------
# Build batches and run inference
# -------------------------------------------------------------------------
def build_batch_for_policy(
    frame: dict,
    rename_map: dict,
    n_empty_cameras: int = 1,
    device: str = "cuda",
    tokenizer=None,
    tokenizer_max_length: int = 48,
):
    """Convert a LeRobot dataset frame into the batch format π0 expects."""
    batch = {}
    for k, v in frame.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0).to(device)
        else:
            batch[k] = v

    # Remove action keys to prevent ground-truth leakage
    for key in list(batch.keys()):
        if key.startswith("action"):
            del batch[key]

    # Apply rename map
    for src_key, dst_key in rename_map.items():
        if src_key in batch:
            batch[dst_key] = batch.pop(src_key)

    # Fill empty camera slots
    real_img_keys = [k for k in batch if k.startswith("observation.images.")]
    if real_img_keys and n_empty_cameras > 0:
        ref_shape = batch[real_img_keys[0]].shape
        ref_dtype = batch[real_img_keys[0]].dtype
        for i in range(n_empty_cameras):
            batch[f"observation.images.empty_camera_{i}"] = torch.zeros(
                ref_shape, dtype=ref_dtype, device=device
            )

    # Task string
    task = batch.get("task")
    if task is None:
        task = "move object to target"
        batch["task"] = task
    if isinstance(task, list):
        task = task[0] if task else "move object to target"

    # Tokenize language
    if tokenizer is not None and "observation.language.tokens" not in batch:
        enc = tokenizer(
            task,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=tokenizer_max_length,
        )
        batch["observation.language.tokens"] = enc["input_ids"].to(device)
        batch["observation.language.attention_mask"] = enc["attention_mask"].to(
            device=device, dtype=torch.bool
        )

    return batch


def evaluate_policy_on_trajectory(
    policy,
    dataset,
    episode_index: int,
    rename_map: dict,
    n_empty_cameras: int = 1,
    action_horizon: int = 50,
    device: str = "cuda",
    tokenizer=None,
    preprocess=None,
    postprocess=None,
):
    """Predict action chunks for one episode and compare to ground truth."""
    from_idx = int(dataset.meta.episodes["dataset_from_index"][episode_index])
    to_idx = int(dataset.meta.episodes["dataset_to_index"][episode_index])
    T = to_idx - from_idx
    if T < action_horizon:
        return None

    all_pred = []
    all_gt = []
    inference_times = []

    for chunk_start in range(0, T - action_horizon + 1, action_horizon):
        global_idx = from_idx + chunk_start
        frame = dataset[global_idx]
        batch = build_batch_for_policy(
            frame, rename_map, n_empty_cameras, device, tokenizer=tokenizer,
        )

        if preprocess is not None:
            try:
                batch = preprocess(batch)
            except Exception as e:
                if chunk_start == 0:
                    print(f"    [warn] preprocess() failed: {e}; using manual batch")

        gt_chunk = []
        for offset in range(action_horizon):
            gt_chunk.append(dataset[from_idx + chunk_start + offset]["action"])
        gt_chunk = torch.stack(gt_chunk).cpu().numpy()
        gt_action_dim = gt_chunk.shape[-1]

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            pred_action = policy.predict_action_chunk(batch)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.time() - t0) * 1000
        inference_times.append(elapsed_ms)

        if postprocess is not None:
            try:
                pred_action = postprocess(pred_action)
            except Exception as e:
                if chunk_start == 0:
                    print(f"    [warn] postprocess() failed: {e}; using raw predictions")

        pred = pred_action.cpu().numpy()
        if pred.ndim == 3:
            pred = pred[0]
        elif pred.ndim == 1:
            pred = pred[None, :]
        pred = pred[:, :gt_action_dim]

        n = min(len(pred), len(gt_chunk))
        all_pred.append(pred[:n])
        all_gt.append(gt_chunk[:n])

    if not all_pred:
        return None

    pred_all = np.concatenate(all_pred, axis=0)
    gt_all = np.concatenate(all_gt, axis=0)
    return {
        "mse": float(np.mean((pred_all - gt_all) ** 2)),
        "mae": float(np.mean(np.abs(pred_all - gt_all))),
        "time_ms_per_chunk": float(np.mean(inference_times)),
        "num_chunks": len(all_pred),
    }


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned π0 against base on DROID/LIBERO test sets."
    )
    parser.add_argument(
        "--dataset", choices=list(DATASET_CONFIGS.keys()), default="droid",
        help="Which dataset to evaluate on (default: droid).",
    )
    parser.add_argument(
        "--finetuned-path", type=Path, default=None,
        help="Local path to the fine-tuned pretrained_model/ directory.",
    )
    parser.add_argument(
        "--base-policy-id", default="lerobot/pi0_base",
    )
    parser.add_argument(
        "--test-dataset-s3", default=None,
    )
    parser.add_argument(
        "--test-dataset-local", type=Path, default=None,
    )
    parser.add_argument(
        "--num-trajectories", type=int, default=5,
    )
    parser.add_argument(
        "--action-horizon", type=int, default=50,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--results-out", type=Path, default=None,
    )
    parser.add_argument(
        "--num-inference-steps", type=int, nargs="+", default=[10],
        help="ODE step counts to evaluate (sweep). Try [10 5 3 1].",
    )
    args = parser.parse_args()

    ds_cfg = DATASET_CONFIGS[args.dataset]

    # Auto-discover checkpoint if not provided
    if args.finetuned_path is None:
        job_prefix = f"pi0-finetune-{args.dataset}"
        search_roots = [
            Path.cwd() / "model_artifacts",
            Path.home() / "pi0_eval" / "checkpoint",
            Path(f"/home/sagemaker-user/Pi0_SMTJ/model_artifacts"),
        ]
        import re as _re
        _step_pat = _re.compile(r"(\d+)")
        candidates = []
        for root in search_roots:
            if root.is_dir():
                for p in root.rglob("pretrained_model"):
                    if p.is_dir() and any(part.startswith(job_prefix) for part in p.parts):
                        candidates.append(p.resolve())
        candidates = list(dict.fromkeys(candidates))
        if candidates:
            def _ckpt_score(p):
                has_cfg = (p / "config.json").is_file()
                step = -1
                for part in reversed(p.parts):
                    m = _step_pat.fullmatch(part)
                    if m:
                        step = int(m.group(1))
                        break
                return (has_cfg, step, p.stat().st_mtime)
            candidates.sort(key=_ckpt_score, reverse=True)
            args.finetuned_path = candidates[0]
            print(f"  Auto-discovered checkpoint: {args.finetuned_path}")
        else:
            args.finetuned_path = Path(
                f"/home/sagemaker-user/pi0_checkpoint/"
                f"pi0-finetune-{args.dataset}/extracted/"
                f"training/checkpoints/002000/pretrained_model"
            )

    if args.test_dataset_local is None:
        args.test_dataset_local = Path(f"/home/sagemaker-user/pi0_test_dataset_{args.dataset}")

    if args.results_out is None:
        args.results_out = Path(f"./pi0_eval_results_{args.dataset}.json")

    print("=" * 70)
    print(f"  π0 Evaluation: Fine-Tuned vs Base on {ds_cfg['description']}")
    print("=" * 70)

    # Auto-detect S3 path
    if args.test_dataset_s3 is None:
        if args.test_dataset_local and args.test_dataset_local.exists():
            args.test_dataset_s3 = "local"  # skip S3, data already on disk
        else:
            import boto3
            sm_sess = boto3.Session()
            sts = sm_sess.client("sts")
            region = sm_sess.region_name or "us-east-1"
            account = sts.get_caller_identity()["Account"]
            bucket = f"sagemaker-{region}-{account}"
            args.test_dataset_s3 = f"s3://{bucket}/pi0-finetuning/datasets/{ds_cfg['s3_suffix']}"
            print(f"  Test S3: {args.test_dataset_s3}")

    # Apply patches
    print("\n[1/5] Applying LeRobot local-dataset patch...")
    apply_local_dataset_patch()

    # Test dataset
    print(f"\n[2/5] Preparing test dataset...")
    test_dir = ensure_test_dataset_local(args.test_dataset_s3, args.test_dataset_local)
    setup_lerobot_cache(test_dir, repo_id=ds_cfg["repo_id"])

    # Load dataset
    print(f"\n[3/5] Loading test dataset...")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(
        repo_id=ds_cfg["repo_id"],
        root=str(test_dir),
        video_backend="pyav",
    )
    n_eps = len(dataset.meta.episodes["dataset_from_index"])
    print(f"  Loaded {n_eps} test episodes, {len(dataset)} total frames")

    # Load policies
    print(f"\n[4/5] Loading policies...")
    from lerobot.policies.pi0 import PI0Policy

    rename_map = ds_cfg["rename_map"]
    n_empty_cameras = ds_cfg["empty_cameras"]

    def load_policy(source: str, label: str):
        print(f"  Loading {label} from {source}")
        p = PI0Policy.from_pretrained(source, torch_dtype=torch.bfloat16)
        backfill_empty_params(p, source)
        p = p.to(args.device).eval()

        from lerobot.policies.factory import make_pre_post_processors
        try:
            preprocess, postprocess = make_pre_post_processors(
                p.config, source,
                preprocessor_overrides={"device_processor": {"device": str(args.device)}},
            )
        except Exception as e:
            print(f"    WARNING: couldn't load processors: {e}")
            preprocess, postprocess = None, None

        return p, preprocess, postprocess

    # Evaluate
    print(f"\n[5/5] Evaluating {args.num_trajectories} trajectories per model...")

    def eval_model(label, source, num_inference_steps=None):
        suffix = f" [steps={num_inference_steps}]" if num_inference_steps is not None else ""
        print(f"\n  --- {label}{suffix} ---")
        if args.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        policy, preprocess, postprocess = load_policy(source, label)

        if num_inference_steps is not None:
            policy.config.num_inference_steps = num_inference_steps

        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
        except Exception:
            tokenizer = None

        results = []
        for ep in range(args.num_trajectories):
            r = evaluate_policy_on_trajectory(
                policy, dataset, ep, rename_map,
                n_empty_cameras=n_empty_cameras,
                action_horizon=args.action_horizon,
                device=args.device,
                tokenizer=tokenizer,
                preprocess=preprocess,
                postprocess=postprocess,
            )
            if r is None:
                print(f"    Trajectory {ep}: SKIPPED (too short)")
                continue
            r["trajectory"] = ep
            results.append(r)
            print(f"    Trajectory {ep}: MSE={r['mse']:.6e}  MAE={r['mae']:.4e}  "
                  f"chunks={r['num_chunks']}  time={r['time_ms_per_chunk']:.1f}ms")

        peak_vram_mb = (torch.cuda.max_memory_allocated() / 1024 / 1024
                        if args.device == "cuda" else 0)
        del policy
        if args.device == "cuda":
            torch.cuda.empty_cache()

        if not results:
            return None
        avg_mse = float(np.mean([r["mse"] for r in results]))
        avg_mae = float(np.mean([r["mae"] for r in results]))
        avg_time = float(np.mean([r["time_ms_per_chunk"] for r in results]))
        print(f"    AVG: MSE={avg_mse:.6e}  MAE={avg_mae:.4e}  "
              f"time={avg_time:.1f}ms  VRAM={peak_vram_mb:.0f}MB")
        return {
            "per_trajectory": results,
            "avg_mse": avg_mse,
            "avg_mae": avg_mae,
            "avg_time_ms_per_chunk": avg_time,
            "peak_vram_mb": peak_vram_mb,
        }

    # Run base
    base_results = eval_model("BASE π0", args.base_policy_id,
                              num_inference_steps=args.num_inference_steps[0])

    # Sweep fine-tuned
    finetuned_sweep = {}
    for n_steps in args.num_inference_steps:
        finetuned_sweep[n_steps] = eval_model(
            "FINE-TUNED π0", str(args.finetuned_path), num_inference_steps=n_steps,
        )

    finetuned_results = finetuned_sweep[max(finetuned_sweep.keys())]

    # Summary
    print("\n" + "=" * 70)
    print(f"  COMPARISON SUMMARY — {ds_cfg['description']}")
    print("=" * 70)
    print(f"  {'Metric':<30} {'BASE':>18} {'FINE-TUNED':>18}")
    print("  " + "-" * 66)
    print(f"  {'Avg MSE':<30} {base_results['avg_mse']:>18.6e} "
          f"{finetuned_results['avg_mse']:>18.6e}")
    print(f"  {'Avg MAE':<30} {base_results['avg_mae']:>18.4e} "
          f"{finetuned_results['avg_mae']:>18.4e}")
    print(f"  {'Inference time / chunk (ms)':<30} "
          f"{base_results['avg_time_ms_per_chunk']:>18.1f} "
          f"{finetuned_results['avg_time_ms_per_chunk']:>18.1f}")
    print(f"  {'Peak VRAM (MB)':<30} "
          f"{base_results['peak_vram_mb']:>18.0f} "
          f"{finetuned_results['peak_vram_mb']:>18.0f}")
    print("=" * 70)

    # ODE sweep table
    if len(finetuned_sweep) > 1:
        print("\n" + "=" * 70)
        print("  ODE STEP-COUNT SWEEP (fine-tuned π0)")
        print("=" * 70)
        print(f"  {'ODE steps':<12} {'Avg MSE':>15} {'Avg MAE':>13} "
              f"{'Time/chunk (ms)':>18} {'VRAM (MB)':>12}")
        print("  " + "-" * 72)
        for n_steps in sorted(finetuned_sweep.keys()):
            r = finetuned_sweep[n_steps]
            print(f"  {n_steps:<12} {r['avg_mse']:>15.4e} {r['avg_mae']:>13.4e} "
                  f"{r['avg_time_ms_per_chunk']:>18.1f} {r['peak_vram_mb']:>12.0f}")
        print("=" * 70)

    # Save JSON
    out = {
        "dataset": args.dataset,
        "dataset_description": ds_cfg["description"],
        "finetuned_path": str(args.finetuned_path),
        "base_policy_id": args.base_policy_id,
        "test_dataset_s3": args.test_dataset_s3,
        "num_trajectories": args.num_trajectories,
        "action_horizon": args.action_horizon,
        "action_dim": ds_cfg["action_dim"],
        "base": base_results,
        "finetuned": finetuned_results,
        "finetuned_step_sweep": {str(n): r for n, r in finetuned_sweep.items()},
    }
    args.results_out.parent.mkdir(parents=True, exist_ok=True)
    args.results_out.write_text(json.dumps(out, indent=2))
    print(f"\n  Results saved to: {args.results_out}")


if __name__ == "__main__":
    main()
