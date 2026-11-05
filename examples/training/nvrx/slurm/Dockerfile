# Build from examples/training/nvrx: docker build -f slurm/Dockerfile -t nvrx-slurm:0.7.0 .
#
# Runtime image for the Slurm variant only. It pins CUDA 13.0.3 on Ubuntu 24.04
# (NVRx 0.7.0's wheel needs glibc 2.39), PyTorch 2.9.0 cu130 and the versions in
# slurm/requirements.txt. The legacy Kubernetes image is a different build.
FROM nvidia/cuda:13.0.3-devel-ubuntu24.04
ARG EFA_INSTALLER_VERSION=1.50.0
ENV DEBIAN_FRONTEND=noninteractive
# Short writable node-local scratch for build steps and NVRx UNIX sockets.
ENV TMPDIR=/opt/build-tmp
RUN mkdir -p /opt/build-tmp && apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev curl ca-certificates git build-essential \
    autoconf automake libtool pkg-config libnuma-dev kmod && rm -rf /var/lib/apt/lists/*
# Container userspace only: never install/reload a host kernel module.
# --skip-kmod and --skip-limit-conf leave the host's EFA driver and ulimits alone;
# the libfabric/OpenMPI userspace is what NCCL's OFI plugin needs.
RUN apt-get update && cd /opt/build-tmp && curl -fsSLO https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && tar -xf aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify --disable-ngc
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:/opt/amazon/efa/bin:/opt/amazon/openmpi/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/ofi-nccl/lib:/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
RUN pip install --no-cache-dir torch==2.9.0 --index-url https://download.pytorch.org/whl/cu130
COPY slurm/requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir -r /opt/requirements.txt && pip check
WORKDIR /app
COPY src/train_gpt2.py src/training_checkpoint.py /app/src/
COPY slurm/ /app/slurm/
COPY tests/ /app/tests/
ENV PYTHONPATH=/app/src
ENV NCCL_DEBUG=INFO
# Fail the build early if the pinned stack does not import or ft_launcher is missing.
RUN python -c "from nvidia_resiliency_ext.fault_tolerance import RankMonitorClient; import transformers; import training_checkpoint" \
    && ft_launcher --help >/dev/null
CMD ["bash"]
# Isolated image scratch (no host path); Pyxis may run under the submitting UID.
RUN chmod 1777 /opt/build-tmp
