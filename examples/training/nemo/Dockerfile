FROM nvcr.io/nvidia/nemo:26.02
ARG GDRCOPY_VERSION=v2.5.2
ARG EFA_INSTALLER_VERSION=1.48.0
# AWS OFI NCCL is bundled into the EFA installation (/opt/amazon/ofi-nccl) for
# EFA installer >=1.47.0. Cf. https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-changelog.html
ARG AWS_OFI_NCCL_VERSION=v1.19.0
ARG NCCL_VERSION=v2.30.4-1
ARG NCCL_TESTS_VERSION=v2.18.3
ARG TRANSFORMERS_VERSION=4.57.6
# Pin megatron-core to the version that NeMo 2.7.x is API-compatible with.
# nemo:26.02 ships megatron-core 0.16.1 at /opt/Megatron-Bridge/3rdparty/Megatron-LM/
# but bundled NeMo 2.7.1 calls APIs 0.16.x removed (get_megatron_optimizer
# kwargs no_weight_decay_cond/scale_lr_cond/lr_mult) and imports submodules
# 0.16.x dropped (megatron.core.dist_checkpointing.strategies.tensorstore).
# 0.15.3 is the latest 0.15.x release on PyPI / GitHub.
ARG MEGATRON_CORE_VERSION=core_v0.15.3


ARG OPEN_MPI_PATH=/opt/amazon/openmpi    # Open MPI already packaged into EFA installation (/opt/amazon/openmpi)

######################
# Update and remove the IB libverbs
######################
RUN apt-get update -y && apt-get upgrade -y
RUN apt-get remove -y --allow-change-held-packages \
    ibverbs-utils \
    libibverbs-dev \
    libibverbs1 \
    libmlx5-1

RUN rm -rf /opt/hpcx/ompi \
    && rm -rf /usr/local/mpi \
    && rm -rf /usr/local/ucx \
    && ldconfig

RUN DEBIAN_FRONTEND=noninteractive apt install -y --allow-unauthenticated \
    apt-utils \
    autoconf \
    automake \
    build-essential \
    cmake \
    curl \
    gcc \
    gdb \
    git \
    kmod \
    libtool \
    openssh-client \
    openssh-server \
    vim
# NOTE: deliberately no `apt autoremove` here. The nemo:26.02 base image
# ships several CUDA add-on libraries (libcusparseLt0, libcudnn-frontend,
# etc.) that apt sees as orphaned because no installed Debian package
# Depends: them — they're loaded via dlopen by torch / transformer_engine
# at runtime. autoremove deletes them and torch.import then crashes with
# `libcusparseLt.so.0: cannot open shared object file`.

RUN mkdir -p /var/run/sshd && \
    sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config && \
    echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config && \
    sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

RUN rm -rf /root/.ssh/ \
 && mkdir -p /root/.ssh/ \
 && ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa \
 && cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys \
 && printf "Host *\n  StrictHostKeyChecking no\n" >> /root/.ssh/config

# NGC images install the OFI NCCL plugin via the libnccl-ofi-ngc-v2 package
# from the EFA installer, which lands at /opt/amazon/aws-ofi-nccl/ rather than
# /opt/amazon/ofi-nccl/ used on stock Ubuntu. Cover both for portability.
ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/opt/amazon/openmpi/lib:/opt/nccl/build/lib:/opt/amazon/efa/lib:/opt/amazon/aws-ofi-nccl/lib:/opt/amazon/ofi-nccl/lib:$LD_LIBRARY_PATH
# Prepend /opt/venv/bin so every `python`/`pip` resolves to the uv-managed
# venv where nemo, megatron, nemo_run, etc. live. The nemo:26.02 image's
# default PATH puts /usr/bin BEFORE /opt/venv/bin, which breaks torchelastic
# worker spawn: `ft_launcher` finds /usr/bin/python (no nemo_run) instead of
# /opt/venv/bin/python and crashes with `ModuleNotFoundError: nemo_run`.
ENV PATH=/opt/venv/bin:/opt/amazon/openmpi/bin/:/opt/amazon/efa/bin:/usr/bin:/usr/local/bin:$PATH

# nemo:26.02 ships libcusparseLt0 (libcusparselt0-cuda-13) into the nested
# /usr/lib/x86_64-linux-gnu/libcusparseLt/13/ directory but does NOT create
# the libcusparseLt.so.0 SONAME symlink and does NOT add an ld.so.conf.d
# entry. torch dlopens libcusparseLt.so.0 at import time and crashes:
#   ImportError: libcusparseLt.so.0: cannot open shared object file
# Register the dir with ld.so AND let ldconfig create the SONAME symlink.
RUN echo /usr/lib/x86_64-linux-gnu/libcusparseLt/13 > /etc/ld.so.conf.d/000_libcusparselt.conf \
 && ldconfig

# Pin the bundled megatron-core to MEGATRON_CORE_VERSION (see ARG above for
# rationale). Replace the in-place 0.16.1 source tree at
# /opt/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/ with the 0.15.3
# release. Pip's site-packages copy is updated by venv.sh on the host; this
# step covers the copy that runs inside the container.
RUN git clone -b ${MEGATRON_CORE_VERSION} --depth 1 \
        https://github.com/NVIDIA/Megatron-LM.git /tmp/megatron-lm \
 && rm -rf /opt/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core \
 && cp -r /tmp/megatron-lm/megatron/core \
        /opt/Megatron-Bridge/3rdparty/Megatron-LM/megatron/ \
 && rm -rf /tmp/megatron-lm

#################################################
## Install NVIDIA GDRCopy
##
## NOTE: if `nccl-tests` or `/opt/gdrcopy/bin/sanity -v` crashes with incompatible version, ensure
## that the cuda-compat-xx-x package is the latest.
RUN git clone -b ${GDRCOPY_VERSION} https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy \
    && cd /tmp/gdrcopy \
    && make prefix=/opt/gdrcopy install

ENV LD_LIBRARY_PATH /opt/gdrcopy/lib:/usr/local/cuda/compat:$LD_LIBRARY_PATH
ENV LIBRARY_PATH /opt/gdrcopy/lib:/usr/local/cuda/compat/:$LIBRARY_PATH
ENV CPATH /opt/gdrcopy/include:$CPATH
ENV PATH /opt/gdrcopy/bin:$PATH

#################################################
## Install EFA installer
##
## The base nemo:26.02 image ships partial EFA components (efa-profile,
## libfabric1-aws, openmpi*-aws, libnccl-ofi-ngc-v2). Their dpkg state
## doesn't match the on-disk files, so the EFA installer's verify step
## refuses to upgrade with "ld.so.conf.d/000_efa.conf is installed by
## efa-profile package but doesn't exist". Purge them first.
RUN dpkg --purge --force-all \
    efa-profile libfabric1-aws libfabric1-aws-dbg \
    openmpi40-aws openmpi50-aws \
    libnccl-ofi-ngc-v2 libnccl-ofi-ngc-v2-dbgsym 2>/dev/null || true

RUN cd $HOME \
    && curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && tar -xf $HOME/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify \
    && rm -rf $HOME/aws-efa-installer


###################################################
## Install AWS-OFI-NCCL plugin
# RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libhwloc-dev
##Switch from sh to bash to allow parameter expansion
# SHELL ["/bin/bash", "-c"]
# RUN curl -OL https://github.com/aws/aws-ofi-nccl/releases/download/${AWS_OFI_NCCL_VERSION}/aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v}.tar.gz \
#     && tar -xf aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v}.tar.gz \
#     && cd aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v} \
#     && ./configure --prefix=/opt/aws-ofi-nccl/install \
#         --with-mpi=/opt/amazon/openmpi \
#         --with-libfabric=/opt/amazon/efa \
#         --with-cuda=/usr/local/cuda \
#        --enable-platform-aws \
#     && make -j $(nproc) \
#     && make install \
#     && cd .. \
#     && rm -rf aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v} \
#     && rm aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v}.tar.gz

# SHELL ["/bin/sh", "-c"]

###################################################
RUN rm -rf /var/lib/apt/lists/*

RUN echo "hwloc_base_binding_policy = none" >> /opt/amazon/openmpi/etc/openmpi-mca-params.conf \
 && echo "rmaps_base_mapping_policy = slot" >> /opt/amazon/openmpi/etc/openmpi-mca-params.conf

# Install additional dependencies required for perf testing
RUN pip3 install --upgrade huggingface_hub>=0.34.0

RUN pip3 install awscli pynvml wandb

RUN mv $OPEN_MPI_PATH/bin/mpirun $OPEN_MPI_PATH/bin/mpirun.real \
 && echo '#!/bin/bash' > $OPEN_MPI_PATH/bin/mpirun \
 && echo '/opt/amazon/openmpi/bin/mpirun.real "$@"' >> $OPEN_MPI_PATH/bin/mpirun \
 && chmod a+x $OPEN_MPI_PATH/bin/mpirun

######################
# Transformers dependencies used in the model
######################
RUN pip install transformers==${TRANSFORMERS_VERSION} sentencepiece python-etcd

ENV OPAL_PREFIX=/opt/amazon/openmpi \
    NCCL_SOCKET_IFNAME=^docker,lo,veth

## Turn off PMIx Error https://github.com/open-mpi/ompi/issues/7516
ENV PMIX_MCA_gds=hash

# Verify EFA / OFI NCCL / OpenMPI installation. NGC's libnccl-ofi-ngc-v2
# package installs to /opt/amazon/aws-ofi-nccl, while stock EFA installs
# to /opt/amazon/ofi-nccl — accept either.
RUN ls -la /opt/amazon/efa/lib/ \
 && (ls -la /opt/amazon/aws-ofi-nccl/lib/ || ls -la /opt/amazon/ofi-nccl/lib/) \
 && ls -la /opt/amazon/openmpi/lib/
