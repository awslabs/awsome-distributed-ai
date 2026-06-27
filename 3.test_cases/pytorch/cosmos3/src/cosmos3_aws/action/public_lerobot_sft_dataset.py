# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""get_action_public_lerobot_sft_dataset — SFT dataset over TURNKEY public LeRobot v3.

Sample-side deliverable for the Cosmos 3 AWS sample. The framework ships
``get_action_droid_sft_dataset`` (action_sft_dataset.py), but it wraps the in-tree
``DROIDLeRobotDataset``, which requires a prepared DROID v3 "success split" (the
v2->v3 conversion + success filtering is run out-of-band and is not publicly
released) with hardcoded NVIDIA column names. This factory mirrors it but wraps the
sample's ``LeRobotV3ActionDataset`` so action-policy post-training runs on a TURNKEY
public LeRobot v3 Hub dataset (e.g. ``lerobot/droid_100``, ``nvidia/BridgeData2_LeRobot_v3``,
``nvidia/LIBERO_LeRobot_v3``) with native action vectors and no out-of-band prep.

It reuses the framework's own ``ActionSFTDataset`` + ``ActionTransformPipeline`` so
the per-sample transform, text tokenization, action padding, and sequence-plan
construction are identical to the in-tree DROID recipe.
"""
from __future__ import annotations

from torch.utils.data import Dataset

from cosmos_framework.data.vfm.action.datasets.action_sft_dataset import (
    ActionSFTDataset,
    ActionIterableShuffleDataset,
)
from cosmos_framework.data.vfm.action.transforms import ActionTransformPipeline

from cosmos3_aws.action.lerobot_v3_action_dataset import LeRobotV3ActionDataset


def get_action_public_lerobot_sft_dataset(
    *,
    repo_id: str = "lerobot/droid_100",
    root: str | None = None,
    fps: float = 15.0,
    chunk_length: int = 32,
    mode: str = "policy",
    camera_keys: list[str] | None = None,
    domain_name: str = "droid_lerobot",
    resolution: str | int = "480",
    max_action_dim: int = 64,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.1,
    append_viewpoint_info: bool = True,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    append_idle_frames: bool = False,
    iterable_shuffle: bool = False,
    episode_shuffle_seed: int = 42,
) -> Dataset:
    """Build a turnkey-public LeRobot v3 action SFT dataset.

    Mirrors ``get_action_droid_sft_dataset`` but swaps the dataset for
    ``LeRobotV3ActionDataset`` (wraps the official ``lerobot.LeRobotDataset``).
    ``append_idle_frames`` defaults to ``False`` to match
    ``get_action_droid_sft_dataset``; the public wrapper also does not emit an
    ``idle_frames`` key, so the pipeline never requires one.

    When ``iterable_shuffle`` is ``True``, the map-style ``ActionSFTDataset`` is
    wrapped in the framework's ``ActionIterableShuffleDataset``, which yields an
    episode-shuffled stream (seeded by ``episode_shuffle_seed``) so batches are
    decorrelated across ranks — the upstream grad-norm fix. The default
    (``False``) returns the plain map-style ``ActionSFTDataset``.
    """
    dataset = LeRobotV3ActionDataset(
        repo_id=repo_id,
        root=root,
        fps=fps,
        chunk_length=chunk_length,
        mode=mode,
        camera_keys=camera_keys,
        domain_name=domain_name,
    )
    transform = ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_action_dim=max_action_dim,
        append_viewpoint_info=append_viewpoint_info,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        append_idle_frames=append_idle_frames,
    )
    sft = ActionSFTDataset(dataset, transform, resolution)
    if iterable_shuffle:
        return ActionIterableShuffleDataset(sft, seed=episode_shuffle_seed)
    return sft
