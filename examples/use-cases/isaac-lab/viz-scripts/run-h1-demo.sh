#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Launch the H1 Locomotion Interactive Demo (uses pre-trained rsl_rl checkpoint)
# This is the workshop Module 3 demo - click robots, control with arrow keys
echo "Starting H1 Locomotion Interactive Demo..."
echo "Controls: UP=forward, DOWN=stop, LEFT=turn left, RIGHT=turn right, C=camera toggle"
cd ~/environment/IsaacLab
xhost +SI:localuser:root
docker kill isaac-lab 2>/dev/null
docker run --shm-size=60g --name isaac-lab --entrypoint bash -it --gpus all \
  -e "ACCEPT_EULA=Y" --rm --network=host \
  -v /home/ubuntu/environment/shared-efs:/workspace/isaaclab/TrainedModel \
  -e DISPLAY \
  -e "PRIVACY_CONSENT=Y" \
  ${ISAAC_IMAGE:-isaaclab-sagemaker:3.0.0} \
  -c "cd /workspace/isaaclab && ./isaaclab.sh -p examples/demos/h1_locomotion.py --viz kit"
