# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

FROM nvcr.io/nvidia/isaac-sim:5.1.0
USER root

RUN apt-get update -y && \
    apt-get install -y cmake build-essential git ncurses-term && \
    rm -rf /var/lib/apt/lists/*

# Clone Isaac Lab v2.3.2
RUN git clone --depth 1 --branch v2.3.2 \
    https://github.com/isaac-sim/IsaacLab.git /workspace/IsaacLab

WORKDIR /workspace/IsaacLab

# Fix: flatdict==4.0.1 (source-only) fails to build with setuptools>=82 (pkg_resources removed).
# Bump to 4.1.0 which ships a wheel. See: https://github.com/isaac-sim/IsaacLab/issues/5388
RUN sed -i 's/"flatdict==4.0.1"/"flatdict==4.1.0"/' source/isaaclab/setup.py

# Pin the loose/unpinned entries in isaaclab/setup.py to specific versions
# for reproducible builds.
RUN sed -i \
    -e 's/"numpy<2"/"numpy==1.26.0"/' \
    -e 's/"torch>=2.7"/"torch==2.7.0"/' \
    -e 's/"onnx>=1.18.0"/"onnx==1.21.0"/' \
    -e 's/"toml"/"toml==0.10.2"/' \
    -e 's/"trimesh"/"trimesh==4.5.1"/' \
    -e 's/"pyglet<2"/"pyglet==1.5.31"/' \
    -e 's/"einops"/"einops==0.8.2"/' \
    -e 's/"warp-lang"/"warp-lang==1.12.1"/' \
    -e 's/"pytest"/"pytest==9.0.3"/' \
    -e 's/"pytest-mock"/"pytest-mock==3.15.1"/' \
    -e 's/"junitparser"/"junitparser==5.0.0"/' \
    -e 's/"flaky"/"flaky==3.8.1"/' \
    -e 's/"packaging"/"packaging==23.0"/' \
    source/isaaclab/setup.py

# Install Isaac Lab into the Isaac Sim Python environment
RUN ln -s /isaac-sim _isaac_sim && \
    ./isaaclab.sh -i

# Pin skrl to 1.x (Isaac Lab v2.3.2 scripts were written against skrl 1.4.x;
# the newer 2.x API is incompatible).
RUN /isaac-sim/python.sh -m pip install "skrl[torch]==1.4.3"

# MLflow client (mlflow-skinny) plus the SageMaker MLflow plugin that resolves
# MLflow tracking ARNs (arn:aws:sagemaker:...:mlflow-app/... or .../mlflow-tracking-server/...)
# to a signed HTTPS endpoint. pynvml powers MLflow's GPU system-metrics sampler.
RUN /isaac-sim/python.sh -m pip install \
    "mlflow-skinny==3.4.0" \
    "sagemaker-mlflow==0.3.0" \
    "pynvml==11.5.3"

# Patch Isaac Lab's train.py to mirror TensorBoard scalars to MLflow in real
# time when MLFLOW_TRACKING_URI is set. The patch only touches train.py; if
# Isaac Lab is upgraded, regenerate it against the new file.
COPY docker/patches/mlflow-train.patch /tmp/mlflow-train.patch
RUN cd /workspace/IsaacLab && git apply /tmp/mlflow-train.patch && rm /tmp/mlflow-train.patch

# Verify isaaclab installed correctly
RUN /isaac-sim/python.sh -c "import isaaclab; print(f'isaaclab installed: {isaaclab.__file__}')"

# Copy custom AWS entrypoints and scripts
COPY scripts/sm-train-entrypoint.sh /workspace/IsaacLab/sm-train-entrypoint.sh
RUN chmod +x /workspace/IsaacLab/sm-train-entrypoint.sh

ENTRYPOINT [""]
CMD ["/bin/bash"]
