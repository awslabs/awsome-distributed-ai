# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Import-launcher for the sample-side DROID policy experiment.

The stock cosmos-framework registers experiments via import side-effects in
``make_config()`` and exposes no plugin hook. This launcher imports our
experiment module first (firing its ``cs.store`` into the process-wide
ConfigStore), then runs the framework's normal training flow. Because the
ConfigStore is populated before Hydra ``compose`` resolves
``[job].experiment``, ``action_policy_public_lerobot`` resolves to our sample-side
node via the framework's standard composition path.

Usage (1 node, 8 GPU smoke)::

    PYTHONPATH=/path/to/cosmos3-aws \\
    torchrun --nproc_per_node=8 -m cosmos3_aws.action.launch_droid_policy \\
        --sft-toml /path/to/droid_policy_smoke.toml -- \\
        trainer.max_iter=10 ckpt_type=dummy job.wandb_mode=disabled
"""

from __future__ import annotations

import argparse
import os
import traceback

from loguru import logger as logging

# Side-effect import: registers `action_policy_public_lerobot` into the ConfigStore.
import cosmos3_aws.action.action_policy_public_lerobot_experiment  # noqa: F401

from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.scripts.train import (
    _setup_deterministic_env_and_backends,
    launch,
)
from cosmos_framework.utils.lazy_config import LazyConfig
from cosmos_framework.utils.serialization import to_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="DROID policy SFT (sample-side experiment)")
    parser.add_argument("--sft-toml", required=True, help="Path to the SFT structured TOML.")
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=[], help="Hydra dotted-path overrides.")
    parser.add_argument("--dryrun", action="store_true", help="Build/print config without training.")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic mode.")
    parser.add_argument(
        "--attach_vscode_debugger",
        action="store_true",
        help="Start a debugpy server (mirrors framework train.py; read by launch()).",
    )
    args = parser.parse_args()

    if args.deterministic:
        _setup_deterministic_env_and_backends()

    config = load_experiment_from_toml(args.sft_toml, extra_overrides=args.opts)
    args.config = args.sft_toml  # telemetry alias (mirrors framework train.py)

    if args.dryrun:
        logging.info("Config:\n" + config.pretty_print(use_color=True))
        os.makedirs(config.job.path_local, exist_ok=True)
        try:
            to_yaml(config, f"{config.job.path_local}/config.yaml")
        except Exception:
            logging.error(f"to_yaml failed: {traceback.format_exc()}")
            LazyConfig.save_yaml(config, f"{config.job.path_local}/config.yaml")
        print(f"{config.job.path_local}/config.yaml")
    else:
        launch(config, args)


if __name__ == "__main__":
    main()
