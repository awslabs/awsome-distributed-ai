#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Download sample media for the Cosmos Reason example clients.
#
# Image: Unsplash (Unsplash License — free for commercial and non-commercial use)
# Video: Wikimedia Commons (CC BY 3.0)
set -euo pipefail

cd "$(dirname "$0")"

echo "Downloading sample.jpg (urban street scene from Unsplash)..."
curl -L -o sample.jpg \
  "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=640"

echo "Downloading sample video from Wikimedia Commons..."
curl -L -o sample_meteor.webm \
  "https://commons.wikimedia.org/w/index.php?title=Special:Redirect/file/2013_Russian_meteor_event_(Magnitogorsk).webm"

echo ""
echo "Downloaded:"
ls -lh sample.jpg sample_meteor.webm
echo ""
echo "Run examples:"
echo "  python3 image_vqa.py --image sample.jpg"
echo "  python3 video_qa.py --video sample_meteor.webm"
echo "  python3 auto_label.py --image-dir . --limit 1"
