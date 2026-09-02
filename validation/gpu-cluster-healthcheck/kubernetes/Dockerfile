# Dockerfile -- GPU Cluster Health Check Agent for Kubernetes
#
# Base image bundles dcgmi + nv-hostengine. nvidia-smi is NOT included;
# it is injected at runtime by the NVIDIA device plugin / GPU Operator.

FROM nvcr.io/nvidia/cloud-native/dcgm:3.3.7-1-ubuntu22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        pciutils \
        kmod \
        curl \
        jq \
        python3 \
        procps \
        iproute2 \
        ca-certificates \
        gnupg \
        lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Install kubectl
RUN curl -fsSL "https://dl.k8s.io/release/$(curl -fsSL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
        -o /usr/local/bin/kubectl \
    && chmod +x /usr/local/bin/kubectl

# Install EFA userspace libraries (skip kernel modules -- host provides them)
# Re-populate apt lists (cleaned in prior layer) so efa_installer.sh can resolve deps
RUN apt-get update \
    && cd /tmp \
    && curl -fsSL https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz \
        -o efa-installer.tar.gz \
    && tar xzf efa-installer.tar.gz \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify \
    && cd / \
    && rm -rf /tmp/aws-efa-installer /tmp/efa-installer.tar.gz \
    && rm -rf /var/lib/apt/lists/*

# Copy health check suite
COPY . /opt/gpu-healthcheck
RUN chmod +x /opt/gpu-healthcheck/gpu-healthcheck.sh \
              /opt/gpu-healthcheck/checks/*.sh \
              /opt/gpu-healthcheck/kubernetes/agent.sh \
              /opt/gpu-healthcheck/kubernetes/sweeper.sh

WORKDIR /opt/gpu-healthcheck

ENTRYPOINT ["/opt/gpu-healthcheck/kubernetes/agent.sh"]
