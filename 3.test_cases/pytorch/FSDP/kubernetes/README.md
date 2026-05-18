# Run Distributed Training with PyTorch FSDP on Amazon EKS

These scripts provide an easy way to get started with multinode [FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) training on EKS. It is designed to be as simple as possible, requires no data preparation, and uses a container image. If you would like to run FSDP with SLURM, please refer to [README.md](../slurm/README.md).

This document will run you through how to run Llama 3.1 8B model training with FSDP. You will also find in this folder manifests to run Llama 2(7B, 13B, 70B), Llama 3.1(8B, 70B), Llama 3.2(1B, 3B),  Mistral 8x7b and Mistral Mathstral. For SageMaker HyperPod EKS clusters, `HyperPodPyTorchJob` (HPTO) manifests are also available for Llama 3.1 8B and Llama 3.2 1B (see [section 4a](#4a-launch-llama-32-1b-training-with-hyperpod-hpto)).

## 0. Prerequisites

### 0.1. EKS Cluster
Before running this training, you'll need to create an Amazon EKS or a SageMaker HyperPod EKS cluster. Instructions can be found in [1.architectures](../../1.architectures), the [aws-do-eks](https://bit.ly/do-eks) project, or the [eks-blueprints](https://github.com/aws-ia/terraform-aws-eks-blueprints) project.

### 0.2. Connect to your EKS Cluster

Run the [aws eks update-kubeconfig](https://awscli.amazonaws.com/v2/documentation/api/latest/reference/eks/update-kubeconfig.html)  command to update your local kube config file (located at ~/.kube/config) with the credentials and configuration needed to connect to your EKS cluster using the kubectl command.

```bash
aws eks update-kubeconfig --name <EKS_CLUSTER_NAME>
```
You can verify that you are connected to the EKS cluster by running this commands:
```bash
kubectl config current-context
```
```
arn:aws:eks:us-west-1:xxxxxxxxxxxx:cluster/xxx-eks-cluster
```
### 0.3. Clone the awsome-distributed-training reposource code
Clone this repo. 

```
git clone https://github.com/awslabs/awsome-distributed-training/
cd awsome-distributed-training/3.test_cases/pytorch/FSDP/kubernetes
```

### 0.4. Envsubst
If the [envsubst](https://github.com/a8m/envsubst) utility is not available in your environment, please install it, following the instructions appropriate for your operating system.

### 0.5. Kubeflow training operator
Deploy the Kubeflow training operator

```bash
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.9.1"
```

## 1. Build container image

Build a container image for this example using the code below:

```bash
export AWS_REGION=$(aws ec2 describe-availability-zones --output text --query 'AvailabilityZones[0].[RegionName]')
export ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export REGISTRY=${ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/
pushd ../
docker build -f Dockerfile -t ${REGISTRY}fsdp:pytorch2.7.1 .
popd
```

The PyTorch FSDP container uses the [nccl-tests](https://github.com/awslabs/awsome-distributed-training/blob/main/micro-benchmarks/nccl-tests/nccl-tests.Dockerfile) container as base.

## 2. Push container image to Amazon ECR

In this step we create a container registry if one does not exist, and push the container image to it.

```bash
# Create registry if needed
REGISTRY_COUNT=$(aws ecr describe-repositories | grep \"fsdp\" | wc -l)
if [ "$REGISTRY_COUNT" == "0" ]; then
        aws ecr create-repository --repository-name fsdp
fi

# Login to registry
echo "Logging in to $REGISTRY ..."
aws ecr get-login-password | docker login --username AWS --password-stdin $REGISTRY

# Push image to registry
docker image push ${REGISTRY}fsdp:pytorch2.7.1
```

## 3. Data

For this example, we'll be using the [allenai/c4](https://huggingface.co/datasets/allenai/c4) dataset. Instead of downloading the entire dataset, the `create_streaming_dataloaders` function will stream the dataset from [HuggingFace](https://huggingface.co/datasets), so there's no data prep required for running this training.

**For this dataset, we will need a Hugging Face access token**. First, create a [Hugging Face account](https://huggingface.co/welcome). Then [generate your access token with read permissions](https://huggingface.co/docs/hub/en/security-tokens). We will use this token and set it in our environment variables in the next step.

If you'd like to instead use your own dataset, you can do so by [formatting it as a HuggingFace dataset](https://huggingface.co/docs/datasets/create_dataset), and passing its location to the `--dataset` argument.

## 4. Launch Llama 3.1 8B training job

Generate the Kubernetes manifest and apply it to the cluster.

Create environment variables:

``` bash
cat << EOF > env_vars
export IMAGE_URI=${REGISTRY}fsdp:pytorch2.7.1
export INSTANCE_TYPE=<INSTANCE TYPE>
export NUM_NODES=<NUMBER OF NODES>
export GPU_PER_NODE=<NUMBER OF GPUS PER NODE>
export EFA_PER_NODE=<NUMBER OF EFA PER NODE>
export FI_PROVIDER=efa
export HF_TOKEN=<YOUR HF ACCESS TOKEN>
EOF
```

For reference, we are running the Llama 3.1 8B model on 4 x p5.48xlarge instances and below is the configuration of our environment variables:
``` bash
cat << EOF > env_vars
export IMAGE_URI=${REGISTRY}fsdp:pytorch2.7.1
export INSTANCE_TYPE=p5.48xlarge
export NUM_NODES=4
export GPU_PER_NODE=8
export EFA_PER_NODE=32
export FI_PROVIDER=efa
export HF_TOKEN=<YOUR HF ACCESS TOKEN>
EOF
```

Fill in `env_vars` and then source variables:

``` bash
source env_vars
```

Apply yaml:
``` bash
envsubst < llama3_1_8b-fsdp.yaml | kubectl apply -f -
```

EFA level variables are available for adjustment in fsdp.yaml-template
Keep FI_* values commented out for non-efa instances (G5, G4d, P3) or P5
Uncomment FI_* values for P4d instances

You can also adjust the training parameters in `TRAINING_ARGS` (for example, to train Llama 3.1 70B). Additional parameters can be found in `src/model_utils/arguments.py`. Note that we use the same directory for both `--checkpoint_dir` and `--resume_from_checkpoint`. If there are multiple checkpoints, `--resume_from_checkpoint` will automatically select the most recent one. This way if our training is interupted for any reason, it will automatically pick up the most recent checkpoint.

## 4a. Launch Llama 3.2 1B training with HyperPod (HPTO)

This section describes how to run a lightweight Llama 3.2 1B FSDP training job using a `HyperPodPyTorchJob` (HPTO) manifest. This is suitable for smaller GPU instances (e.g., g5.8xlarge) and requires a **SageMaker HyperPod EKS cluster**.

### Prerequisites

- A SageMaker HyperPod EKS cluster (the HPTO manifest uses the `sagemaker.amazonaws.com/v1` API group which is not available on vanilla EKS)
- The container image must be built using the `hpto` target of the Dockerfile, which includes `hyperpod-elastic-agent==1.1.2`:

``` bash
pushd ../
docker build -f Dockerfile --target hpto -t ${REGISTRY}fsdp:pytorch2.7.1-hpto .
popd
```

Push this image to ECR following the same steps in section 2.

### Launch the training job

Create environment variables:

``` bash
cat << EOF > env_vars
export IMAGE_URI=${REGISTRY}fsdp:pytorch2.7.1-hpto
export NUM_NODES=<NUMBER OF NODES>
export GPU_PER_NODE=<NUMBER OF GPUS PER NODE>
export EFA_PER_NODE=<NUMBER OF EFA PER NODE>
export HF_TOKEN=<YOUR HF ACCESS TOKEN>
EOF
```

For reference, this manifest was validated on 8 x ml.g5.8xlarge instances (1 GPU, 1 EFA per node):

``` bash
cat << EOF > env_vars
export IMAGE_URI=${REGISTRY}fsdp:pytorch2.7.1-hpto
export NUM_NODES=8
export GPU_PER_NODE=1
export EFA_PER_NODE=1
export HF_TOKEN=<YOUR HF ACCESS TOKEN>
EOF
```

Source variables and apply:

``` bash
source env_vars
envsubst < llama3_2_1b-fsdp-hpto.yaml | kubectl apply -f -
```

> **Note:** The manifest sets `FI_EFA_USE_DEVICE_RDMA=0` because g5 instances do not support GPUDirect RDMA. If you adapt this manifest for p4d/p5 instances, set this value to `1`.

### Monitor and stop

Monitor the HPTO job:

``` bash
kubectl get hyperpodpytorchjob
kubectl get pods
kubectl logs -f llama3-2-1b-fsdp-hpto-pods-0
```

Stop the job:

``` bash
envsubst < llama3_2_1b-fsdp-hpto.yaml | kubectl delete -f -
```

## 5. Monitor training job

To see the status of your job, use the commands below

```bash
kubectl get pytorchjob 
kubectl get pods 
```

```log
NAME               STATE     AGE
llama3-1-8b-fsdp   Running   5m38s
NAME                        READY   STATUS    RESTARTS   AGE
llama3-1-8b-fsdp-worker-0   1/1     Running   0          5m39s
llama3-1-8b-fsdp-worker-1   1/1     Running   0          5m39s
llama3-1-8b-fsdp-worker-2   1/1     Running   0          5m39s
llama3-1-8b-fsdp-worker-3   1/1     Running   0          5m39s
```

Each of the pods produces job logs. One of the pods is elected master during job initialization. Only this pod will show the progress of the training job in its log. To find out which pod is currently the master, run the command below.

```bash
kubectl logs llama3-1-8b-fsdp-worker-0 | grep master_addr=
```

```log
I0620 14:27:39.789000 1 torch/distributed/elastic/agent/server/api.py:525]   master_addr=llama3-1-8b-fsdp-worker-0
```

This shows that the pod `llama3-1-8b-fsdp-worker-0` is currently the master. To look at the current job logs, use the command below:

```bash
kubectl logs -f llama3-1-8b-fsdp-worker-0
```

```log
...
2025-06-20 14:17:10 I [train.py:103] Batch 90 Loss: 7.24291, Speed: 9.41 samples/sec, lr: 0.000010
2025-06-20 14:17:14 I [train.py:103] Batch 91 Loss: 7.27470, Speed: 8.94 samples/sec, lr: 0.000010
2025-06-20 14:17:17 I [train.py:103] Batch 92 Loss: 7.06632, Speed: 9.42 samples/sec, lr: 0.000010
2025-06-20 14:17:21 I [train.py:103] Batch 93 Loss: 7.17624, Speed: 8.96 samples/sec, lr: 0.000010
2025-06-20 14:17:24 I [train.py:103] Batch 94 Loss: 7.24291, Speed: 9.06 samples/sec, lr: 0.000010
2025-06-20 14:17:28 I [train.py:103] Batch 95 Loss: 7.13051, Speed: 9.05 samples/sec, lr: 0.000010
2025-06-20 14:17:32 I [train.py:103] Batch 96 Loss: 7.16901, Speed: 8.30 samples/sec, lr: 0.000010
2025-06-20 14:17:36 I [train.py:103] Batch 97 Loss: 7.50217, Speed: 8.51 samples/sec, lr: 0.000010
```

## 6. Stop training job

To stop the current training job, use the following command.

```bash
kubectl delete -f ./llama3_1_8b-fsdp.yaml
```

If you wish to launch a new job, you must first stop the previous one, even if it is in `Completed` state.

## References
Llama 2 and  Llama 3.x models parameters are based on the values in the [Llama 2 paper](https://arxiv.org/abs/2307.09288) and [Llama 3 paper](https://arxiv.org/abs/2407.21783) 


| Parameter            | Llama 2 7B | Llama 2 13B | Llama 2 70B | Llama 3.1 8B | Llama 3.1 70B | Llama 3.2 1B | Llama 3.2 3B |
|----------------------|------------|-------------|-------------|--------------|---------------|--------------|--------------|
| intermediate_size    | 11008      | 13824       | 28672       | 14336        | 28672         | 8192         | 11008        |
| num_key_value_heads  | 32         | 40          | 8           | 8            | 8             | 8            | 8            |
| hidden_width         | 4096       | 5120        | 8192        | 4096         | 8192          | 2048         | 3072         |
| num_layers           | 32         | 40          | 80          | 32           | 80            | 16           | 28           |
| num_heads            | 32         | 40          | 64          | 32           | 64            | 32           | 24           |
| max_context_length   | 4096       | 4096        | 4096        | 8192         | 8192          | 8192         | 8192         |