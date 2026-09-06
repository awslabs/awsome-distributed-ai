# CUDA userspace and all inherited Python dependencies are fixed by this digest.
FROM lmsysorg/sglang:v0.5.12.post1-cu130@sha256:ceaf8b16e02d165143633ac228bbb994a05fe77d7e0526cf035ae4bbf4eacc36 AS engine
ARG EFA_VERSION=1.47.0
# The nixl stub otherwise pulls the CUDA 12 distribution; keep the CUDA 13 pair.
RUN python3 -m pip install --no-deps --force-reinstall nixl==1.1.0 nixl-cu13==1.1.0
RUN apt-get update \
 && apt-get install -y --no-install-recommends pciutils=1:3.10.0-2build1 environment-modules=5.4.0-1 tcl=8.6.14build1 \
 && cd /tmp \
 && curl -fSsLO https://s3-us-west-2.amazonaws.com/aws-efa-installer/aws-efa-installer-${EFA_VERSION}.tar.gz \
 && tar -xf aws-efa-installer-${EFA_VERSION}.tar.gz \
 && cd aws-efa-installer \
 && ./efa_installer.sh -y --skip-kmod -g \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* /tmp/aws-efa-installer*
ENV PATH="/opt/amazon/efa/bin:/opt/amazon/openmpi/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:$LD_LIBRARY_PATH"
ENV FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1
WORKDIR /lab
COPY verify_image.py /lab/verify_image.py
RUN python3 /lab/verify_image.py > /lab/image-versions.json
COPY . /lab

# The CPU router must not pull the CUDA engine image onto small system-node disks.
FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c AS router
COPY router-requirements.txt /tmp/router-requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/router-requirements.txt
ENTRYPOINT ["python3", "-m", "sglang_router.launch_router"]
