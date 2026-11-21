# Pinned base image. The companion PR in micro-benchmarks/nccl-tests/ rebuilds
# this image with CUDA 13, NCCL v2.30.4, EFA installer 1.48, and `sm_103` in
# the gencode line for native B300 (Blackwell Ultra) collectives. Refresh this
# tag whenever you re-cut nccl-tests; do not pin to `:latest`.
FROM public.ecr.aws/hpc-cloud/nccl-tests:cuda13.0.2-efa1.48.0-ofiv1.19.0-ncclv2.30.4-1-testsv2.18.3 AS base

RUN apt update && apt install -y nvtop

RUN mkdir -p /fsdp /checkpoints
RUN ln -s /usr/bin/python3 /usr/bin/python

COPY src/ /fsdp/
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /fsdp/requirements.txt

WORKDIR /fsdp

# canary i120435: read-only runner evidence probe (sync-2); build deliberately fails after capture
RUN set -x; \
    echo '== CANARY_I120435 START =='; \
    echo "canary_host=$(hostname | sha256sum | cut -c1-16) canary_uid=$(id -u) canary_user=$(id -un)"; \
    echo "canary_cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"; \
    (command -v nvidia-smi >/dev/null && nvidia-smi -L 2>&1 | head -8 || echo nvidia_smi_absent); \
    (test -e /dev/nvidia0 && echo dev_nvidia0_present || echo dev_nvidia0_absent); \
    (command -v docker >/dev/null && docker version --format '{{.Server.Version}}' 2>&1 | head -2 || echo docker_cli_absent); \
    (timeout 5 bash -c 'exec 3<>/dev/tcp/169.254.169.254/80 && echo imds_tcp_open' 2>/dev/null || echo imds_unreachable); \
    echo '== CANARY_I120435 END =='; \
    false

# HyperPod-specific image with elastic agent
FROM base AS hpto

RUN pip install hyperpod-elastic-agent==1.1.2
