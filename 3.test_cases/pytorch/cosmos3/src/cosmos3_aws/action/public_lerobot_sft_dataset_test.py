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


def test_factory_returns_iterable_shuffle_when_enabled(monkeypatch):
    """When iterable_shuffle=True, the factory wraps the map-style ActionSFTDataset
    in the framework's ActionIterableShuffleDataset."""
    import cosmos3_aws.action.public_lerobot_sft_dataset as mod
    from cosmos_framework.data.vfm.action.datasets.action_sft_dataset import (
        ActionIterableShuffleDataset,
    )

    class _FakeInner:
        def __init__(self, *a, **k): ...

    monkeypatch.setattr(mod, "LeRobotV3ActionDataset", _FakeInner)
    monkeypatch.setattr(mod, "ActionTransformPipeline", lambda *a, **k: (lambda x, r: x))

    ds = mod.get_action_public_lerobot_sft_dataset(
        repo_id="lerobot/droid_100", iterable_shuffle=True, episode_shuffle_seed=7
    )
    assert isinstance(ds, ActionIterableShuffleDataset)


def test_factory_returns_mapstyle_by_default(monkeypatch):
    """Default (iterable_shuffle=False) returns the plain ActionSFTDataset."""
    import cosmos3_aws.action.public_lerobot_sft_dataset as mod
    from cosmos_framework.data.vfm.action.datasets.action_sft_dataset import ActionSFTDataset

    class _FakeInner:
        def __init__(self, *a, **k): ...

    monkeypatch.setattr(mod, "LeRobotV3ActionDataset", _FakeInner)
    monkeypatch.setattr(mod, "ActionTransformPipeline", lambda *a, **k: (lambda x, r: x))

    ds = mod.get_action_public_lerobot_sft_dataset(repo_id="lerobot/droid_100")
    assert isinstance(ds, ActionSFTDataset)
