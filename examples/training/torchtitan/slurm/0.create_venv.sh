#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

set -euo pipefail

# torchtitan host environment for awsome-distributed-training.
#
# Replaces the previous Miniconda-based setup with a stdlib `python -m venv`
# environment, and pins all components to released versions instead of
# pulling nightly torch + torchtitan HEAD on every run.
#
# Targets CUDA 13 (cu130) wheels for native sm_103 support on P6-B300.

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-./pt_torchtitan}"
TORCHTITAN_DIR="${TORCHTITAN_DIR:-./torchtitan}"
TORCHTITAN_REF="${TORCHTITAN_REF:-v0.2.2}"
TORCH_VERSION="${TORCH_VERSION:-2.10.0+cu130}"
TORCHAO_VERSION="${TORCHAO_VERSION:-0.17.0+cu130}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Error: ${PYTHON_BIN} not found on PATH." >&2
    echo "Install Python 3.11 (e.g. via deadsnakes PPA) or set PYTHON_BIN." >&2
    exit 1
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip

pip install \
    --index-url "${PYTORCH_INDEX_URL}" \
    "torch==${TORCH_VERSION}" \
    "torchao==${TORCHAO_VERSION}"

if [ ! -d "${TORCHTITAN_DIR}" ]; then
    git clone --depth 1 --branch "${TORCHTITAN_REF}" \
        https://github.com/pytorch/torchtitan "${TORCHTITAN_DIR}"
else
    echo "Reusing existing ${TORCHTITAN_DIR} (skipping clone)."
fi

pip install -r "${TORCHTITAN_DIR}/requirements.txt"
pip install -e "${TORCHTITAN_DIR}"

echo
echo "torchtitan venv ready:"
echo "  venv:       ${VENV_DIR}"
echo "  torchtitan: ${TORCHTITAN_DIR} @ ${TORCHTITAN_REF}"
echo "  torch:      ${TORCH_VERSION}"
echo "  torchao:    ${TORCHAO_VERSION}"
