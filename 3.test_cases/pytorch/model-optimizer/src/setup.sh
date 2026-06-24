#!/usr/bin/env bash
# Set up the ModelOpt environment on the GPU Deep Learning AMI (Amazon Linux 2023).
#
# The base OSS NVIDIA-driver DLAMI ships the driver + Docker but no conda/frameworks and no
# CUDA toolkit. That is fine: the nvidia-modelopt and vllm pip wheels bring their own CUDA.
# We use the DLAMI's Python 3.12 (ModelOpt requires Python >= 3.10).
#
# Run as the ec2-user, from the user's home directory.
set -euo pipefail

MODELOPT_VERSION="0.44.0"
WORKDIR="${HOME}/model-optimizer-recipe"
REPO_DIR="${HOME}/Model-Optimizer"

echo "==> Python: $(python3.12 --version)"

echo "==> Creating ModelOpt venv"
python3.12 -m venv "${WORKDIR}/modelopt-venv"
# shellcheck disable=SC1091
. "${WORKDIR}/modelopt-venv/bin/activate"
pip install --upgrade pip -q

echo "==> Installing nvidia-modelopt + example deps (pinned)"
# requirements.txt lives next to this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -q -r "${SCRIPT_DIR}/requirements.txt"
python -c 'import modelopt; print("modelopt", modelopt.__version__)'

echo "==> Cloning Model-Optimizer examples, pinned to the matching tag ${MODELOPT_VERSION}"
# The example scripts (hf_ptq.py) are NOT in the pip wheel. Check out the tag that matches the
# installed wheel to avoid version skew (main can reference internals not in the release).
if [ ! -d "${REPO_DIR}" ]; then
  git clone https://github.com/NVIDIA/Model-Optimizer.git "${REPO_DIR}"
fi
cd "${REPO_DIR}"
git checkout "${MODELOPT_VERSION}"

echo "==> Setup complete."
echo "    venv:  ${WORKDIR}/modelopt-venv"
echo "    repo:  ${REPO_DIR} (tag ${MODELOPT_VERSION})"
echo "    next:  bash quantize_fp8.sh"
