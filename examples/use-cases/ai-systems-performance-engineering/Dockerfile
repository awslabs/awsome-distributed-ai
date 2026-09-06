FROM public.ecr.aws/hpc-cloud/nccl-tests:cuda13.0.2-efa1.48.0-ofiv1.19.0-ncclv2.30.4-1-testsv2.18.3
COPY requirements.txt constraints.txt /tmp/
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt -c /tmp/constraints.txt
COPY . /opt/aim347
RUN nvcc -O3 /opt/aim347/lib/gemm.cu -lcublas -o /usr/local/bin/aim347-gemm
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
WORKDIR /opt/aim347
