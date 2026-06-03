FROM nvcr.io/nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ARG EFA_INSTALLER_VERSION=1.29.1
ARG NCCL_VERSION=v2.18.6-1
ARG AWS_OFI_NCCL_VERSION=v1.7.4-aws
ARG JAX_VERSION=0.4.18
ARG PRAXIS_VERSION=1.2.0
ARG PAXML_VERSION=1.2.0

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.10
ENV LD_LIBRARY_PATH=/opt/amazon/openmpi/lib:/opt/nccl/build/lib:/opt/aws-ofi-nccl/install/lib:/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH
ENV PATH=/opt/amazon/openmpi/bin/:/opt/amazon/efa/bin:/usr/local/cuda-12/bin:$PATH
ENV CUDA_HOME=/usr/local/cuda-12


#########################
# Packages and Pre-reqs #
RUN apt-get update -y && \
    apt-get purge -y --allow-change-held-packages libmlx5-1 ibverbs-utils libibverbs-dev libibverbs1 libnccl-dev libnccl2
RUN apt-get install -y --allow-unauthenticated \
    autoconf \
    automake \
    bash \
    build-essential \
    ca-certificates \
    curl \
    debianutils \
    dnsutils \
    g++ \
    git \
    libtool \
    libhwloc-dev \
    netcat \
    openssh-client \
    openssh-server \
    openssl \
    python3-distutils \
    python"${PYTHON_VERSION}"-dev \
    python-is-python3 \
    util-linux

RUN update-ca-certificates

###########################
# Python/Pip dependencies #
RUN curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && python"${PYTHON_VERSION}" /tmp/get-pip.py
RUN pip"${PYTHON_VERSION}" install numpy wheel build

######################################
# Install EFA Libfabric and Open MPI #
RUN cd /tmp \
    && curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && tar -xf aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y -d --skip-kmod --skip-limit-conf --no-verify

############################
# Compile and Install NCCL #
RUN git clone -b "${NCCL_VERSION}" https://github.com/NVIDIA/nccl.git /opt/nccl \
    && cd /opt/nccl \
    && make -j src.build CUDA_HOME=${CUDA_HOME} \
    && cp -R /opt/nccl/build/* /usr/

###############################
# Compile AWS OFI NCCL Plugin #
RUN git clone -b "${AWS_OFI_NCCL_VERSION}" https://github.com/aws/aws-ofi-nccl.git /opt/aws-ofi-nccl \
    && cd /opt/aws-ofi-nccl \
    && ./autogen.sh \
    && ./configure --prefix=/opt/aws-ofi-nccl/install \
       --with-libfabric=/opt/amazon/efa/ \
       --with-cuda=${CUDA_HOME} \
       --with-mpi=/opt/amazon/openmpi/ \
       --with-nccl=/opt/nccl/build \
       --enable-platform-aws \
    && make -j && make install

###############
# Install JAX #
RUN pip install --upgrade "jax[cuda12_pip]==${JAX_VERSION}" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install "orbax-checkpoint>=0.4.0,<0.5.0"

##################
# Install Praxis #
RUN pip install praxis==${PRAXIS_VERSION}

#################
# Install Paxml #
RUN pip install paxml==${PAXML_VERSION}

#####################################
# Allow unauthenticated SSH for MPI #
RUN mkdir -p /var/run/sshd \
    && sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config \
    && echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config \
    && sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config

COPY run_paxml.sh /run_paxml.sh
RUN pip install scipy==1.12.0
RUN pip install protobuf==3.20.3
RUN cp /usr/local/lib/python3.10/dist-packages/google/protobuf/internal/builder.py .
RUN pip3 install protobuf==3.19.6
RUN cp /builder.py /usr/local/lib/python3.10/dist-packages/google/protobuf/internal/builder.py
