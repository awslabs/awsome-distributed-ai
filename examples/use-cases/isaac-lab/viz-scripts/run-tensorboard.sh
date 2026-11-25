#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Launch TensorBoard to view training metrics
echo "Starting TensorBoard on port 6006..."
docker kill isaac-lab-tb 2>/dev/null
docker run --name isaac-lab-tb --entrypoint bash -d --rm --network=host \
  -e "ACCEPT_EULA=Y" \
  -e "PRIVACY_CONSENT=Y" \
  -v /home/ubuntu/environment/shared-efs:/workspace/isaaclab/TrainedModel \
  ${ISAAC_IMAGE:-isaaclab-sagemaker:3.0.0} \
  -c "/opt/isaaclab-venv/bin/python -m tensorboard.main --logdir /workspace/isaaclab/TrainedModel/isaaclab-logs --host 127.0.0.1 --port 6006"
echo "TensorBoard running at http://localhost:6006"
echo "Open Firefox in DCV desktop to view."
