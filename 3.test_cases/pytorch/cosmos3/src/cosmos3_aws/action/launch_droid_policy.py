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

    # Optional native-observability bridge (no-op when its env var is absent, so
    # the default path is unchanged). Setting OTEL_EXPORTER_OTLP_ENDPOINT lands
    # the cosmos-framework training metrics in Amazon Managed Prometheus via the
    # HyperPod observability addon's OTLP receiver, unified with the addon's
    # GPU/DCGM metrics in Grafana. Enabling it does three things in lockstep:
    #   - attaches OTLPCallback (loss / step time / iteration straight to the
    #     addon collector's OTLP receiver; no Pushgateway, no collector edit);
    #   - installs the wandb->OTLP bridge so EVERY numeric scalar the framework
    #     callbacks log via wandb (MFU, throughput, grad-norm, packing, ...) is
    #     mirrored to OTLP gauges too;
    #   - enables the MFU + sequence-packing framework callbacks that produce
    #     those richer metrics (iter_speed and grad_clip are already enabled in
    #     the experiment, so they are NOT re-added here).
    # See observability/README.md for details.
    _otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if _otlp_endpoint:
        from omegaconf import OmegaConf

        _job_name = os.environ.get("PROMETHEUS_JOB_NAME", "cosmos3")
        _every_n = int(os.environ.get("PROMETHEUS_EVERY_N", "1"))
        _protocol = os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
        _peak_tflops = os.environ.get("COSMOS3_PEAK_TFLOPS")

        _extra_callbacks: dict[str, dict] = {}
        _extra_callbacks["otlp"] = dict(
            _target_="cosmos3_aws.observability.otlp_callback.OTLPCallback",
            endpoint=_otlp_endpoint,
            job_name=_job_name,
            every_n=_every_n,
            protocol=_protocol,
        )

        # Enabling MFUCallback adds minor per-step FLOPs-accounting overhead; it
        # is gated on observability being on, so the default path pays nothing.
        # OmegaConf interpolation strings (e.g. "${trainer.logging_iter}") can't
        # be injected into a plain dict post-load, so read the ints directly.
        _logging_iter = int(config.trainer.logging_iter)
        _mfu_cb = dict(
            _target_="cosmos3_aws.observability.cosmos3_mfu_callback.Cosmos3MFUCallback",
            every_n=_logging_iter,
            grad_accum_iter=int(getattr(config.trainer, "grad_accum_iter", 1)),
        )
        if _peak_tflops is not None:
            _mfu_cb["peak_tflops"] = float(_peak_tflops)
        _extra_callbacks["mfu"] = _mfu_cb
        _extra_callbacks["sequence_packing_padding"] = dict(
            _target_="cosmos_framework.callbacks.sequence_packing_padding.SequencePackingPadding",
            every_n=_logging_iter,
        )

        # The loaded config is an OmegaConf node in struct mode (new keys rejected);
        # open it just long enough to attach the observability callback(s).
        OmegaConf.set_struct(config.trainer.callbacks, False)
        for _name, _cb in _extra_callbacks.items():
            config.trainer.callbacks[_name] = _cb
        OmegaConf.set_struct(config.trainer.callbacks, True)

        # Mirror the framework's wandb-logged scalars to OTLP gauges too.
        from cosmos3_aws.observability.wandb_otlp_bridge import install_wandb_otlp_bridge

        install_wandb_otlp_bridge(endpoint=_otlp_endpoint, job_name=_job_name, protocol=_protocol)
        logging.info(f"Observability callbacks enabled: {list(_extra_callbacks)}")

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
