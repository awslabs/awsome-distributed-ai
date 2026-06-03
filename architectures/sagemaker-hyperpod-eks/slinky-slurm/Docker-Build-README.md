# Docker Build for the Slurmd Deep Learning Container

This build includes Python 3.12.8 + PyTorch 2.6.0 + CUDA 12.6 + NCCL 2.23.4 + EFA Installer 1.38.0 (bundled with OFI NCCL plugin)

> **NOTE:** The `setup.sh` script automates image builds using AWS CodeBuild
> (default) or locally with `--local-build`. The manual instructions below
> are for standalone builds outside the automated deployment flow.

Clone the AWSome Distributed Training repo:
```
git clone https://github.com/awslabs/awsome-distributed-ai.git
cd awsome-distributed-ai/architectures/sagemaker-hyperpod-eks/slinky-slurm/

```

Build the container image: 

```

# Authenticate to DLC repo (Account 763104351884 is publicly known) 
aws ecr get-login-password --region us-east-1 \
| docker login --username AWS \
--password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com

# on a Mac
docker buildx build --platform linux/amd64 -t dlc-slurmd:25.11.1-ubuntu24.04 -f dlc-slurmd.Dockerfile .

# on Linux
# docker build -t dlc-slurmd:25.11.1-ubuntu24.04 -f dlc-slurmd.Dockerfile .

```

Test locally:

Verify Python 3.12.8 + PyTorch 2.6.0 + CUDA 12.6 + NCCL 2.23.4

```

docker run  --platform linux/amd64 -it --entrypoint=/bin/bash dlc-slurmd:25.11.1-ubuntu24.04

python3 --version
# Python 3.12.8

which python3
# /usr/local/bin/python3

nvcc --version
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2024 NVIDIA Corporation
# Built on Tue_Oct_29_23:50:19_PDT_2024
# Cuda compilation tools, release 12.6, V12.6.85
# Build cuda_12.6.r12.6/compiler.35059454_0

python3 -c "import torch; print(torch.__version__)"
# 2.6.0+cu126

python3 -c "import torch; print(torch.cuda.nccl.version())"
# (2, 23, 4)

ls -l /usr/local/lib/libnccl*
# -rwxr-xr-x 1 root root 263726576 Mar  6 23:36 /usr/local/lib/libnccl.so
# -rwxr-xr-x 1 root root 263726576 Mar  6 23:36 /usr/local/lib/libnccl.so.2
# -rwxr-xr-x 1 root root 263726576 Mar  6 23:36 /usr/local/lib/libnccl.so.2.23.4
# -rw-r--r-- 1 root root 277972056 Mar  6 23:36 /usr/local/lib/libnccl_static.a

cat /etc/nccl.conf
# NCCL_DEBUG=INFO
# NCCL_SOCKET_IFNAME=^docker0

exit
```

Create a private ECR repo:

```

aws ecr create-repository --repository-name dlc-slurmd

```

Authenticate to the repo:

```
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION=<your-region-here>

aws ecr get-login-password --region $AWS_REGION \
 | docker login --username AWS \
 --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
 
```

Tag the image: 

```

docker tag dlc-slurmd:25.11.1-ubuntu24.04 \
 ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/dlc-slurmd:25.11.1-ubuntu24.04
 
```

Push the image to an ECR repo:

```

docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/dlc-slurmd:25.11.1-ubuntu24.04

```

Test ECR access:

```

kubectl run test-pod \
 --image=${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/dlc-slurmd:25.11.1-ubuntu24.04 \
 --restart=Never \
 --image-pull-policy=Always

# verify slurm version
kubectl exec -it test-pod -- slurmd -V
 
kubectl describe pod test-pod

# verify additional requirements
kubectl exec -it test-pod -- ls /usr/local/lib/python3.12/site-packages/ \
 | egrep "datasets|fsspec|numpy|torch|torchaudio|torchvision|transformers"
 
kubectl delete pod test-pod

```

(Optional) Update the container image used by the Slinky NodeSet:

Note: this step is not required if you specify the image repository and tag in the [slurm-values.yaml.template](./slurm-values.yaml.template) file, but is useful if you want to test a new image build without redeploying the entire Slurm cluster. 

```
export NODESET_NAME=$(kubectl get nodeset -n slurm -o custom-columns=NAME:metadata.name --no-headers)

kubectl -n slurm patch nodeset.slinky.slurm.net \
  $NODESET_NAME \
  --type='json' \
  -p="[
    {\"op\": \"replace\", \"path\": \"/spec/template/spec/containers/0/image\", \"value\":\"${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/dlc-slurmd:25.11.1-ubuntu24.04\"},
    {\"op\": \"replace\", \"path\": \"/spec/template/spec/containers/0/imagePullPolicy\", \"value\":\"Always\"}
  ]"
  
```

Scale the Slinky NodeSet down and back up to trigger replacement:

```

kubectl -n slurm scale nodeset/$NODESET_NAME --replicas=0


kubectl -n slurm scale nodeset/$NODESET_NAME --replicas=4

```

