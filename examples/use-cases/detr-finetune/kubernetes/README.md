# Kubernetes Deployment

Deploy DETR-ResNet50 fine-tuning as a distributed PyTorchJob on Amazon EKS.

## Prerequisites

The guide assumes that you have the following:

- An Amazon SageMaker HyperPod EKS cluster or Amazon EKS cluster with GPU nodes
  (e.g., `ml.g5.8xlarge`), accessible via `kubectl`.
- An Amazon FSx for Lustre persistent volume claim. The YAML template defaults
  to the name `fsx-pvc`. If your cluster uses a different name (e.g.,
  `fsx-claim`), update the `claimName` field in the generated YAML before
  applying, or use `sed`:
  ```bash
  sed -i 's/fsx-pvc/fsx-claim/g' detr-resnet50-finetune.yaml
  ```
- Docker installed on a machine with **internet access** (the build downloads
  model weights from HuggingFace Hub and bakes them into the image).
- The dataset uploaded to FSx at `/fsx/data/` (see [data/README.md](../data/README.md)).

We recommend that you setup a Kubernetes cluster using the templates in the
architectures [directory](../../../../architectures).

> **Note**: Amazon SageMaker HyperPod EKS clusters come with the Kubeflow
> Training Operator pre-installed. If you are using a vanilla EKS cluster,
> ensure the
> [Kubeflow Training Operator](https://www.kubeflow.org/docs/components/training/pytorch/)
> is deployed before proceeding.

## 1. Build Container Image

The Docker build downloads DETR-ResNet50 pre-trained weights from HuggingFace
Hub and bakes them into the image. This means the training pods do **not** need
internet access at runtime, which is important for clusters in private subnets
without a NAT gateway.

```bash
export AWS_REGION=$(aws ec2 describe-availability-zones --output text --query 'AvailabilityZones[0].[RegionName]')
export ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export REGISTRY=${ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/
export VERSION=v1
docker build -t ${REGISTRY}detr-resnet50-finetune:${VERSION} -f Dockerfile .
```

**Note**: Run the `docker build` command from the `detr-finetune/` directory (the parent of `kubernetes/`), where `Dockerfile` and `detr_main.py` are located.

## 2. Push to ECR

```bash
# Create repository if needed
aws ecr describe-repositories --repository-names detr-resnet50-finetune \
  >/dev/null 2>&1 || aws ecr create-repository --repository-name detr-resnet50-finetune

# Login to registry
echo "Logging in to $REGISTRY ..."
aws ecr get-login-password | docker login --username AWS --password-stdin $REGISTRY

# Push image
docker image push ${REGISTRY}detr-resnet50-finetune:${VERSION}
```

## 3. Submit Training Job

Set environment variables and create the manifest from the template:

```bash
export IMAGE_URI=${REGISTRY}detr-resnet50-finetune:${VERSION}
export INSTANCE_TYPE=ml.g5.8xlarge
export NUM_NODES=2
cat detr-resnet50-finetune.yaml-template | envsubst '$IMAGE_URI $NUM_NODES $INSTANCE_TYPE' > detr-resnet50-finetune.yaml

kubectl apply -f ./detr-resnet50-finetune.yaml
```

## 4. Monitor Training

Check job status:

```bash
kubectl get pytorchjobs
kubectl get pods -l app=detr-resnet50-finetune
```

```text
NAME                      STATE     AGE
detr-resnet50-finetune    Running   30s

NAME                                  READY   STATUS    RESTARTS   AGE
detr-resnet50-finetune-worker-0       1/1     Running   0          30s
detr-resnet50-finetune-worker-1       1/1     Running   0          30s
```

View training logs:

```bash
kubectl logs -f detr-resnet50-finetune-worker-0
```

```text
[INFO] Using QAI Hub DETR-ResNet50 model
[INFO] Classes: {'Price': 1, 'Product': 2}
[INFO] Loaded 36 train samples
[INFO] Loaded 9 val samples
[INFO] Starting training for 50 epochs...
Epoch [1/50] Train: [ 0/9]  Time  2.345 ( 2.345)  Data  0.123 ( 0.123)  Loss 1.4321e+00 (1.4321e+00)
...
```

Check GPU utilization:

```bash
kubectl exec -it detr-resnet50-finetune-worker-0 -- nvidia-smi
```

## 5. Retrieve Results

After training completes, checkpoints are saved to `/fsx/checkpoint/`:

```bash
# Access the FSx volume to check results
kubectl exec -it detr-resnet50-finetune-worker-0 -- ls -la /fsx/checkpoint/

# View training statistics
kubectl exec -it detr-resnet50-finetune-worker-0 -- cat /fsx/checkpoint/training_stats.txt
```

## 6. Stop the Training Job

```bash
kubectl delete -f ./detr-resnet50-finetune.yaml
```

**Note**: Prior to running a new job, please stop any currently running or
completed job.

## Configuration

### Scaling

Adjust the number of worker nodes by changing `NUM_NODES`:

```bash
export NUM_NODES=4
cat detr-resnet50-finetune.yaml-template | envsubst '$IMAGE_URI $NUM_NODES $INSTANCE_TYPE' > detr-resnet50-finetune.yaml
```

### Instance Type

Change the GPU instance type:

```bash
export INSTANCE_TYPE=ml.g5.12xlarge
```

### Training Parameters

Modify the training arguments in the YAML template command section. Key
parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--batch-size` | 4 | Per-GPU batch size |
| `--lr` | 1e-4 | Learning rate |
| `--num-classes` | 2 | Number of object classes |
| `--workers` | 2 | Data loading workers per GPU |

### EFA Networking

The template includes EFA configuration for high-performance inter-node
communication. If your cluster does not have EFA, remove the following from the
template:

- `vpc.amazonaws.com/efa: 1` from resource requests and limits
- `FI_PROVIDER` and `FI_EFA_FORK_SAFE` environment variables
