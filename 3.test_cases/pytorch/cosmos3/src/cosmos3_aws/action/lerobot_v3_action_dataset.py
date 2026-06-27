# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""LeRobotV3ActionDataset — generic LeRobot v3 action dataset for Cosmos 3 policy.

Sample-side deliverable for the Cosmos 3 AWS sample. Wraps the official
``lerobot.datasets.LeRobotDataset`` (a hard dependency of cosmos-framework) so
all v3 heavy lifting — chunked Parquet loading, video decode via torchcodec,
and windowing via ``delta_timestamps`` — is handled by the upstream library.
Works on ANY conformant LeRobot v3.0 dataset (e.g. ``lerobot/droid_100``,
``nvidia/BridgeData2_LeRobot_v3``, ``nvidia/LIBERO_LeRobot_v3``) without the
hardcoded column names of the in-tree ``DROIDLeRobotDataset`` (which targets
NVIDIA's bespoke ``droid_plus_lerobot`` cartesian export).

Action is kept NATIVE (the dataset's own action vector, e.g. 7D for DROID/LIBERO
EEF-delta+gripper). No joint->cartesian forward kinematics is fabricated; the
model's domain-aware action projection + ``max_action_dim`` padding handle
heterogeneous action widths.

The emitted sample dict matches what the framework's ``ActionTransformPipeline``
consumes (see ``get_action_public_lerobot_sft_dataset``), mirroring
``DROIDLeRobotDataset`` output.
"""

from __future__ import annotations

import random
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from cosmos_framework.data.vfm.action.domain_utils import get_domain_id

_CONCAT_VIEW_DESCRIPTION = (
    "The top row is from the wrist-mounted camera. The bottom row contains two "
    "horizontally concatenated third-person perspective views of the scene from "
    "opposite sides, with the robot visible."
)


class LeRobotV3ActionDataset(Dataset):
    """Generic LeRobot v3 action dataset feeding the Cosmos 3 action pipeline.

    Parameters
    ----------
    repo_id:
        HuggingFace dataset id (e.g. ``"lerobot/droid_100"``).
    root:
        Local path to the staged v3 dataset.
    chunk_length:
        Number of action steps per sample. The video window is
        ``chunk_length + 1`` frames (one extra observation frame), matching the
        DROID convention where action ``a_t`` bridges ``v_{t-1} -> v_t``.
    fps:
        Sampling rate used to build the ``delta_timestamps`` window.
    mode:
        Action generation mode: ``"policy"`` (default), ``"forward_dynamics"``,
        ``"inverse_dynamics"``, or ``"image2video"``. ``"joint"`` randomizes per
        sample across the three action modes (matches DROIDLeRobotDataset).
    camera_keys:
        Ordered camera feature keys. The first is treated as the wrist view
        (top row); the remaining two are concatenated side-by-side on the bottom
        row, matching ``DROIDLeRobotDataset`` concat layout. If ``None``, all
        dataset cameras are used in declared order.
    domain_name:
        Embodiment domain id key for ``get_domain_id`` (default
        ``"droid_lerobot"``).
    """

    _MODE_CHOICES = ("forward_dynamics", "inverse_dynamics", "policy")

    def __init__(
        self,
        repo_id: str,
        root: str | None = None,
        chunk_length: int = 16,
        fps: float = 15.0,
        mode: str = "policy",
        camera_keys: list[str] | None = None,
        domain_name: str = "droid_lerobot",
    ) -> None:
        super().__init__()
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self._chunk_length = int(chunk_length)
        self._fps = float(fps)
        self._mode = mode
        self._domain_id = get_domain_id(domain_name)

        n = self._chunk_length + 1
        dt = [i / self._fps for i in range(n)]

        # Probe cameras if not provided.
        probe = LeRobotDataset(repo_id, root=root)
        self._camera_keys = camera_keys if camera_keys is not None else list(probe.meta.camera_keys)

        # Window state, action, and all cameras over the same delta_timestamps.
        delta = {"observation.state": dt, "action": dt}
        for cam in self._camera_keys:
            delta[cam] = dt
        self._ds = LeRobotDataset(repo_id, root=root, delta_timestamps=delta)

        # Per-episode frame counts (ordered by episode index) for shuffle-block
        # construction used by the framework's ActionIterableShuffleDataset.
        meta = self._ds.meta
        try:
            episodes = meta.episodes
            counts = [int(episodes[i]["length"]) for i in range(len(episodes))]
        except Exception:
            import numpy as np
            ep_idx = np.asarray(self._ds.hf_dataset["episode_index"])
            counts = np.bincount(ep_idx).tolist()
        self._episode_frame_counts = counts

    def __len__(self) -> int:
        return len(self._ds)

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        """Per-episode flat-index blocks ``(start, length)`` for the framework's
        ``ActionIterableShuffleDataset``. The iterable shuffles the ORDER of these
        blocks and shards them disjointly across ranks while streaming the windows
        WITHIN each block sequentially (decorrelated batches, sequential reads).

        A window needs ``chunk_length + 1`` frames, so the last ``chunk_length``
        frames of each episode cannot start a full window -> usable length is
        ``max(0, frame_count - chunk_length)``. Episodes with no usable windows
        are dropped. Mirrors the in-tree ``DROIDLeRobotDataset.get_shuffle_blocks``.
        """
        blocks: list[tuple[int, int]] = []
        start = 0
        for count in self._episode_frame_counts:
            usable = max(0, int(count) - self._chunk_length)
            if usable > 0:
                blocks.append((start, usable))
            start += int(count)
        return blocks

    def _choose_mode(self) -> str:
        if self._mode == "joint":
            return random.choice(self._MODE_CHOICES)
        return self._mode

    def _concat_views(self, item: dict) -> torch.Tensor:
        """Concat cameras into one frame stack, mirroring DROIDLeRobotDataset.

        Each camera tensor is [T, C, H, W] float in [0, 1] (lerobot output).
        Wrist (cam 0) forms the top row; remaining cams are resized to half
        height/width and concatenated horizontally to form the bottom row.
        Returns uint8 [C, T, H, W].
        """
        cams = [item[k] for k in self._camera_keys]  # each [T,C,H,W]
        wrist = cams[0]
        _, _, h, w = wrist.shape
        if len(cams) >= 3:
            half_h, half_w = h // 2, w // 2
            left = F.interpolate(cams[1], size=(half_h, half_w), mode="bilinear", align_corners=False)
            right = F.interpolate(cams[2], size=(half_h, half_w), mode="bilinear", align_corners=False)
            bottom = torch.cat([left, right], dim=-1)  # [T,C,half_h,w]
            frames = torch.cat([wrist, bottom], dim=-2)  # [T,C,h+half_h,w]
        elif len(cams) == 2:
            other = F.interpolate(cams[1], size=(h, w), mode="bilinear", align_corners=False)
            frames = torch.cat([wrist, other], dim=-2)  # stack vertically
        else:
            frames = wrist
        frames = (frames * 255.0).clamp(0.0, 255.0).to(torch.uint8)
        return frames.permute(1, 0, 2, 3).contiguous()  # [T,C,H,W] -> [C,T,H,W]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self._ds[int(idx)]
        mode = self._choose_mode()

        video = self._concat_views(item)  # [C,T,H,W] uint8
        action = item["action"][: self._chunk_length].to(torch.float32)  # [chunk_length, D] native
        caption = item.get("task")
        if not isinstance(caption, str) or not caption:
            caption = "perform the manipulation task"

        return {
            "ai_caption": caption,
            "video": video,
            "action": action,
            "conditioning_fps": torch.tensor(int(self._fps), dtype=torch.long),
            "mode": mode,
            "domain_id": torch.tensor(self._domain_id, dtype=torch.long),
            "viewpoint": "concat_view",
            "additional_view_description": _CONCAT_VIEW_DESCRIPTION,
        }
