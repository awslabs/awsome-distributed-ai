# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# NVIDIA's Isaac Lab 3.0.0 image: Isaac Sim 6.2, Python 3.12, PyTorch 2.11 with
# CUDA 13.0. Isaac Lab and its RL libraries (skrl, rsl_rl, rl_games, sb3) are
# already installed in /opt/isaaclab-venv, pinned by Isaac Lab's uv.lock.
# Isaac Lab 3.0.0 fixes CVE-2026-65141, which affects Isaac Lab 0.0 to 2.3.2
# (NVIDIA security bulletin 5902). The digest pins the multi-arch index of the
# 3.0.0 tag; change the tag and the digest together when upgrading.
FROM nvcr.io/nvidia/isaac-lab:3.0.0@sha256:f70acc9ed3bc3fc67099a0c5715379c432cc9965623cabbaf54f74599997e245

# The base image ends as the non-root user "isaaclab". The entrypoints in this
# sample write to /opt/ml (SageMaker Training) and to FSx, and the 2.3.2 image
# ran them as root, so keep root.
USER root

# Install AWS EFA userspace stack (libfabric + AWS-OFI-NCCL plugin) so NCCL
# can use EFA/RDMA when running on EFA-capable instances (g6.12xlarge+, p4d,
# p5). --skip-kmod relies on the kernel module being present on the host
# (provided by SageMaker AMI / HyperPod nodes). On instances without EFA,
# NCCL falls back to TCP automatically — these libraries are inert.
ENV EFA_INSTALLER_VERSION=1.47.0
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    cd /tmp && \
    curl -sL https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz | tar xz --no-same-owner && \
    cd aws-efa-installer && \
    ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify && \
    cd / && rm -rf /tmp/aws-efa-installer && rm -rf /var/lib/apt/lists/*

# AWS-OFI-NCCL plugin discovery: NCCL looks for libnccl-net-aws-ofi.so but the
# installer ships libnccl-net-ofi.so. Without this symlink NCCL silently falls
# back to TCP sockets even when EFA is otherwise available.
RUN ln -sf /opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so \
           /opt/amazon/ofi-nccl/lib/libnccl-net-aws-ofi.so && \
    echo "/opt/amazon/ofi-nccl/lib" > /etc/ld.so.conf.d/aws-ofi-nccl.conf && \
    echo "/opt/amazon/efa/lib"      > /etc/ld.so.conf.d/efa.conf && \
    ldconfig

# AWS-OFI-NCCL also needs libcudart.so for GPU memory pinning (it dlopens
# "libcudart.so"). Point it at the CUDA 13 runtime that PyTorch and NCCL use,
# from the nvidia-cuda-runtime wheel in the Isaac Lab venv; Isaac Sim also
# bundles CUDA 12 copies, which must not be picked. Without it the plugin
# loads but its CUDA init fails and NCCL silently falls back to Socket. The
# test fails the build if a base-image upgrade moves the library.
ENV ISAACLAB_CUDA_RUNTIME=/opt/isaaclab-venv/lib/python3.12/site-packages/nvidia/cu13/lib
RUN test -f ${ISAACLAB_CUDA_RUNTIME}/libcudart.so.13 && \
    ln -sf ${ISAACLAB_CUDA_RUNTIME}/libcudart.so.13 /usr/local/lib/libcudart.so && \
    ldconfig

ENV LD_LIBRARY_PATH="/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH}"
ENV PATH="/opt/amazon/efa/bin:${PATH}"

WORKDIR /workspace/isaaclab

# MLflow client (mlflow-skinny) plus the SageMaker MLflow plugin that resolves
# MLflow tracking ARNs (arn:aws:sagemaker:...:mlflow-app/... or .../mlflow-tracking-server/...)
# to a signed HTTPS endpoint. pynvml powers MLflow's GPU system-metrics sampler.
# Installed into the Isaac Lab venv with every package that is already there
# held at its current version, so the install can't change what Isaac Lab and
# Kit run on; the build fails instead if the pins can't be satisfied.
RUN uv pip freeze --python /opt/isaaclab-venv/bin/python \
      | grep -v -e '^-e ' -e ' @ ' > /tmp/isaaclab-constraints.txt && \
    uv pip install --no-cache --python /opt/isaaclab-venv/bin/python \
      --constraint /tmp/isaaclab-constraints.txt \
      "mlflow-skinny==3.18.0" \
      "sagemaker-mlflow==0.5.0" \
      "pynvml==11.5.3" && \
    rm /tmp/isaaclab-constraints.txt

# MLflow integration: a hook module + thin entrypoint shim. The hook
# monkey-patches the TensorBoard writers' add_scalar (mirrors TensorBoard
# scalars) and argparse.ArgumentParser.parse_args (captures CLI args as run
# params) at runtime - no line-coupling to upstream train.py. Activated when
# MLFLOW_TRACKING_URI is set.
COPY scripts/mlflow_isaaclab.py /workspace/isaaclab/mlflow_isaaclab.py
COPY scripts/run_train.py       /workspace/isaaclab/run_train.py

# Verify the Isaac Lab, skrl and MLflow packages import
RUN ./isaaclab.sh -p -c "import isaaclab, skrl, mlflow, sagemaker_mlflow; print(f'isaaclab {isaaclab.__file__}, skrl {skrl.__version__}, mlflow {mlflow.__version__}')"

# Copy custom AWS entrypoints and scripts
COPY scripts/sm-train-entrypoint.sh /workspace/isaaclab/sm-train-entrypoint.sh
RUN chmod +x /workspace/isaaclab/sm-train-entrypoint.sh

ENTRYPOINT [""]
CMD ["/bin/bash"]
