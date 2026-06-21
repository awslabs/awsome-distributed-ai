# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Auto-loaded at interpreter startup (Python imports ``sitecustomize`` if found on
the path). Installs the sample-side NormMonitor empty-shard guard for every rank,
including runs that invoke ``cosmos_framework.scripts.train`` directly (no sample
launcher). Safe no-op if the framework/callback is unavailable.

See ``cosmos3_aws/norm_monitor_guard.py`` for the rationale (LoRA empty-shard
``numel()==0`` crash in NormMonitor at high rank counts).
"""
try:
    import cosmos3_aws.norm_monitor_guard  # noqa: F401  (import side-effect installs the guard)
except Exception:
    pass
