# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Unit test for get_action_public_lerobot_sft_dataset (GPU-free, monkeypatched)."""
from __future__ import annotations

import pytest


def test_factory_wraps_our_dataset_in_action_sft_dataset(monkeypatch):
    captured = {}

    class _FakeATP:
        def __init__(self, **kw):
            captured["atp_kwargs"] = kw

    class _FakeActionSFTDataset:
        def __init__(self, dataset, transform, resolution):
            captured["dataset"] = dataset
            captured["transform"] = transform
            captured["resolution"] = resolution

    class _FakeLeRobot:
        def __init__(self, **kw):
            captured["lerobot_kwargs"] = kw

    monkeypatch.setattr(
        "cosmos3_aws.action.public_lerobot_sft_dataset.ActionSFTDataset",
        _FakeActionSFTDataset,
    )
    monkeypatch.setattr(
        "cosmos3_aws.action.public_lerobot_sft_dataset.ActionTransformPipeline",
        _FakeATP,
    )
    monkeypatch.setattr(
        "cosmos3_aws.action.public_lerobot_sft_dataset.LeRobotV3ActionDataset",
        _FakeLeRobot,
    )

    from cosmos3_aws.action.public_lerobot_sft_dataset import (
        get_action_public_lerobot_sft_dataset,
    )

    out = get_action_public_lerobot_sft_dataset(
        repo_id="lerobot/droid_100",
        root="/data/droid",
        fps=15.0,
        chunk_length=32,
        resolution="480",
        max_action_dim=64,
        tokenizer_config={"x": 1},
    )

    assert isinstance(out, _FakeActionSFTDataset)
    assert isinstance(captured["dataset"], _FakeLeRobot)
    assert isinstance(captured["transform"], _FakeATP)
    assert captured["resolution"] == "480"
    assert captured["lerobot_kwargs"]["repo_id"] == "lerobot/droid_100"
    assert captured["lerobot_kwargs"]["chunk_length"] == 32
    assert captured["atp_kwargs"]["max_action_dim"] == 64
    assert captured["atp_kwargs"]["tokenizer_config"] == {"x": 1}
    assert captured["atp_kwargs"]["append_idle_frames"] is False
