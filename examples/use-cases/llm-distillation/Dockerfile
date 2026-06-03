
FROM public.ecr.aws/hpc-cloud/nccl-tests:latest

RUN apt update && apt install -y nvtop && apt-get install -y build-essential python3-dev

RUN mkdir -p /distillation /checkpoints
RUN ln -s /usr/bin/python3 /usr/bin/python

COPY src/ /distillation/
RUN pip install -r /distillation/requirements.txt
WORKDIR /distillation

WORKDIR /distillation

