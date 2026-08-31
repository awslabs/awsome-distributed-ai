#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# ============================================================
# SLIME GRPO training entrypoint (runs on the Ray worker).
#
# This is the single shell that the Ray job entrypoint executes. It is
# uploaded to the Ray cluster via `ray job submit --working-dir <this dir>`
# and invoked as:  ... -- bash grpo_launch.sh <train.py flags...>
#
# WHY A DEDICATED LAUNCHER (design note):
#   SLIME's per-model scripts (scripts/models/*.sh) define MODEL_ARGS as a bash
#   ARRAY. `ray job submit` re-joins everything after `--` with
#   subprocess.list2cmdline and runs it through an outer `/bin/sh -c`
#   (Popen(shell=True)). If the array were referenced in the submitted string
#   (e.g. `-- bash -c "... ${MODEL_ARGS[@]} ..."`), that outer shell would
#   expand ${MODEL_ARGS[@]} BEFORE this script sources the definition, so it
#   would expand to zero elements and train.py would receive no model config.
#   Keeping the `source` and the array expansion inside this one launcher —
#   which the outer shell never expands, because the entrypoint tokens are just
#   `bash grpo_launch.sh <scalar flags>` — makes the whole class of shell
#   escaping bug impossible. This matches how SLIME's own upstream launch
#   scripts (scripts/run-*.sh) expand ${MODEL_ARGS[@]} in the same shell that
#   sourced it.
#
# Inputs:
#   $@                 : all train.py flags assembled by the recipe (scalar
#                        argv tokens, safely quoted by Ray).
#   env MODEL_SCRIPT   : the SLIME model script to source (e.g. qwen3-4B.sh),
#                        passed through the Ray runtime env by the recipe.
#   env SLIME_DIR      : SLIME install dir in the image (default /opt/slime).
# ============================================================

set -euo pipefail

SLIME_DIR="${SLIME_DIR:-/opt/slime}"

if [[ -z "${MODEL_SCRIPT:-}" ]]; then
    echo "[grpo_launch] ERROR: MODEL_SCRIPT env var is not set." >&2
    exit 1
fi

cd "${SLIME_DIR}"

MODEL_SCRIPT_PATH="scripts/models/${MODEL_SCRIPT}"
if [[ ! -f "${MODEL_SCRIPT_PATH}" ]]; then
    echo "[grpo_launch] ERROR: model script ${SLIME_DIR}/${MODEL_SCRIPT_PATH} not found." >&2
    exit 1
fi

# Sourcing defines the MODEL_ARGS bash array in THIS shell.
# shellcheck disable=SC1090
source "${MODEL_SCRIPT_PATH}"

# Fail fast if the model script did not populate MODEL_ARGS. This is the exact
# condition that previously slipped through to train.py as "hidden_size None".
if [[ "${#MODEL_ARGS[@]}" -eq 0 ]]; then
    echo "[grpo_launch] ERROR: MODEL_ARGS is empty after sourcing ${MODEL_SCRIPT_PATH}." >&2
    exit 1
fi

echo "[grpo_launch] MODEL_SCRIPT=${MODEL_SCRIPT} MODEL_ARGS count=${#MODEL_ARGS[@]}"
echo "[grpo_launch] launching: python3 train.py <${#MODEL_ARGS[@]} model args> $# recipe args"

# MODEL_ARGS is expanded here (same shell that sourced it); the recipe-provided
# flags arrive as "$@". Quote both so values containing spaces stay single tokens.
exec python3 train.py "${MODEL_ARGS[@]}" "$@"
