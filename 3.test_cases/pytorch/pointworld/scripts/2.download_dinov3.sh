#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Download the gated DINOv3 scene-encoder backbone weights for PointWorld.
#
# PointWorld uses a DINOv3 ViT-L/16 backbone to featurize the RGB scene. The
# DINOv3 weights are GATED by Meta: you must first request access on the
# official release page and obtain a personal, time-limited download URL.
#
#   1. Request access:  https://github.com/facebookresearch/dinov3
#   2. From the access email, copy the download URL for the
#      dinov3_vitl16 pretrain checkpoint.
#   3. Run this script with that URL.
#
# The weights are placed on the shared filesystem so every worker pod can read
# them. The training container symlinks third_party/dinov3/checkpoints to this
# location at run time (see kubernetes/README.md), so no gated artifact is ever
# baked into the container image.
#
# Usage:
#   ./2.download_dinov3.sh <DINOV3_DOWNLOAD_URL> [DEST_DIR]
#
# Example:
#   ./2.download_dinov3.sh "https://dl.fbaipublicfiles.com/dinov3/....pth" \
#       /fsx/$USER/pointworld/dinov3/checkpoints

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "ERROR: missing DINOv3 download URL." >&2
  echo "Usage: $0 <DINOV3_DOWNLOAD_URL> [DEST_DIR]" >&2
  echo "Request access first at https://github.com/facebookresearch/dinov3" >&2
  exit 1
fi

DINOV3_URL="$1"
DEST_DIR="${2:-/fsx/${USER}/pointworld/dinov3/checkpoints}"

# Filename MUST match what PointWorld's scene_featurizer expects. The loader
# (scene_featurizer._resolve_dinov3_weights) looks for this exact canonical name
# first; a renamed file is rejected with "Unexpected weights specification for
# the ViT-L backbone". Keep the canonical released filename.
OUT_NAME="${DINOV3_WEIGHT_NAME:-dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}"

mkdir -p "${DEST_DIR}"
echo "[2.download_dinov3] Downloading DINOv3 ViT-L/16 weights -> ${DEST_DIR}/${OUT_NAME}"
wget --no-verbose -O "${DEST_DIR}/${OUT_NAME}" "${DINOV3_URL}"

echo "[2.download_dinov3] Done."
echo "  Make these weights visible to the training entrypoint by mounting"
echo "  ${DEST_DIR} into the pod and symlinking it to:"
echo "      /pointworld/third_party/dinov3/checkpoints"
echo "  (the kubernetes/ manifests do this via an initContainer)."
