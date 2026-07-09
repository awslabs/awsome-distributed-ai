"""
Monkey-patch LeRobot to skip Hub API lookups for local-only datasets.

This is applied at container startup so that lerobot-train and inference
scripts can load datasets from FSx without network calls to huggingface.co.

Usage:
    python lerobot_local_patch.py

Or import and call apply_patch() from another script.
"""

import os


def apply_patch():
    """Patch LeRobot's version-checking functions to work offline."""
    try:
        from lerobot.datasets import utils as lr_utils
    except ImportError:
        return  # LeRobot not installed

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

    try:
        import lerobot.datasets.lerobot_dataset as _lr_ds
        if hasattr(_lr_ds, "get_safe_version"):
            _lr_ds.get_safe_version = patched_safe
        if hasattr(_lr_ds, "get_repo_versions"):
            _lr_ds.get_repo_versions = patched_repo
    except ImportError:
        pass

    print("[patch] LeRobot local-dataset patch applied")


if __name__ == "__main__":
    apply_patch()
