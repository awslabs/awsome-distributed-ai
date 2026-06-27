# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Integration tests for LeRobotV3ActionDataset against a real LeRobot v3 dataset.

Requires the public ``lerobot/droid_100`` (v3.0) staged locally and FFmpeg
available for torchcodec. These run on the dev pod, not in CPU-only CI.

The wrapper delegates v3 chunk loading + video decode + windowing to the
official ``lerobot.datasets.LeRobotDataset`` (so it works on ANY conformant v3
dataset: droid_100, BridgeData2, LIBERO), then maps each window into the sample
dict that the framework's ActionTransformPipeline consumes. Action is kept
NATIVE (no fabricated joint->cartesian FK).
"""

from __future__ import annotations

import os

import torch

from cosmos3_aws.action.lerobot_v3_action_dataset import LeRobotV3ActionDataset

_ROOT = os.environ.get("DROID_DATASET_PATH", "/fsx/datasets/droid_lerobot_v3")
_REPO = "lerobot/droid_100"


def _make(mode: str = "policy", chunk_length: int = 16) -> LeRobotV3ActionDataset:
    return LeRobotV3ActionDataset(
        repo_id=_REPO,
        root=_ROOT,
        chunk_length=chunk_length,
        fps=15.0,
        mode=mode,
        camera_keys=[
            "observation.images.wrist_image_left",
            "observation.images.exterior_image_1_left",
            "observation.images.exterior_image_2_left",
        ],
    )


def test_len_is_positive() -> None:
    ds = _make()
    assert len(ds) > 0


def test_sample_has_required_keys() -> None:
    ds = _make()
    s = ds[0]
    for key in ("ai_caption", "video", "action", "conditioning_fps", "mode", "domain_id", "viewpoint"):
        assert key in s, f"missing key {key}"


def test_video_is_channels_first_uint8_window() -> None:
    ds = _make(chunk_length=16)
    s = ds[0]
    v = s["video"]
    # [C, T, H, W], uint8, T = chunk_length + 1 = 17
    assert v.ndim == 4
    assert v.shape[0] == 3
    assert v.shape[1] == 17
    assert v.dtype == torch.uint8


def test_action_is_native_7d_window() -> None:
    ds = _make(chunk_length=16)
    s = ds[0]
    a = s["action"]
    # [T, D] with D = native 7 (no FK expansion); T = chunk_length = 16
    assert a.ndim == 2
    assert a.shape[0] == 16
    assert a.shape[1] == 7
    assert a.dtype == torch.float32


def test_caption_is_nonempty_string() -> None:
    ds = _make()
    s = ds[0]
    assert isinstance(s["ai_caption"], str) and len(s["ai_caption"]) > 0


def test_mode_and_scalars_typed() -> None:
    ds = _make(mode="policy")
    s = ds[0]
    assert s["mode"] == "policy"
    assert torch.is_tensor(s["conditioning_fps"]) and s["conditioning_fps"].dtype == torch.long
    assert torch.is_tensor(s["domain_id"]) and s["domain_id"].dtype == torch.long


def test_get_shuffle_blocks_from_episode_boundaries() -> None:
    """get_shuffle_blocks() returns per-episode (start, length) flat-index blocks,
    dropping the last chunk_length frames of each episode (cannot start a full window)."""
    from cosmos3_aws.action.lerobot_v3_action_dataset import LeRobotV3ActionDataset

    ds = LeRobotV3ActionDataset.__new__(LeRobotV3ActionDataset)  # bypass __init__ (no lerobot)
    ds._chunk_length = 2
    # Two episodes: episode 0 has 5 frames [0..4], episode 1 has 4 frames [5..8].
    ds._episode_frame_counts = [5, 4]
    # Expected: ep0 -> (0, 5-2=3); ep1 -> (5, 4-2=2)
    blocks = ds.get_shuffle_blocks()
    assert blocks == [(0, 3), (5, 2)]


def test_get_shuffle_blocks_drops_short_episodes() -> None:
    """Episodes shorter than or equal to chunk_length contribute no windows."""
    from cosmos3_aws.action.lerobot_v3_action_dataset import LeRobotV3ActionDataset

    ds = LeRobotV3ActionDataset.__new__(LeRobotV3ActionDataset)
    ds._chunk_length = 4
    ds._episode_frame_counts = [3, 6]  # ep0 (3 <= 4) drops; ep1 -> (3, 6-4=2)
    blocks = ds.get_shuffle_blocks()
    assert blocks == [(3, 2)]
