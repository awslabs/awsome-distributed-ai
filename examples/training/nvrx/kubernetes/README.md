# Run NVRx Resiliency Testing on Amazon EKS

This guide walks you through deploying NVRx resiliency experiments on an Amazon EKS cluster. Before following these steps, complete the common setup (build container, push to ECR) described in the [top-level README](../README.md).

## 0. Prerequisites

### 0.1. EKS Cluster

You need an Amazon EKS cluster with GPU nodes and EFA networking. Instructions for creating a cluster can be found in [architectures](../../../../architectures), the [aws-do-eks](https://bit.ly/do-eks) project, or [EKS Blueprints](https://github.com/aws-ia/terraform-aws-eks-blueprints).

Your cluster must have:
- GPU nodes (g5, p4de, or p5 instances) with [NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin)
- [EFA device plugin](https://github.com/aws/eks-charts/tree/master/stable/aws-efa-k8s-device-plugin) (for multi-node training on p4de/p5)
- [Amazon FSx for Lustre](https://docs.aws.amazon.com/fsx/latest/LustreGuide/what-is.html) filesystem in the same VPC/AZ as your nodes
- [FSx CSI Driver](https://docs.aws.amazon.com/eks/latest/userguide/fsx-csi.html) installed

### 0.2. Connect to Your EKS Cluster

```bash
aws eks update-kubeconfig --name <EKS_CLUSTER_NAME>
kubectl config current-context
```

### 0.3. Envsubst

If the [envsubst](https://github.com/a8m/envsubst) utility is not available in your environment, please install it following the instructions appropriate for your operating system.

### 0.4. Navigate to This Directory

All commands in this README are run from the `kubernetes/` directory:

```bash
cd examples/pytorch/nvrx/kubernetes
```

## 1. Create Namespace

```bash
kubectl apply -f namespace.yaml
```

## 2. Create HuggingFace Token Secret

A HuggingFace access token is required for downloading models (LLaMA) and datasets (C4). Create a [HuggingFace account](https://huggingface.co/welcome) and [generate an access token](https://huggingface.co/docs/hub/en/security-tokens).

```bash
kubectl create secret generic huggingface-token \
  --from-literal=token=<YOUR_HF_TOKEN> \
  -n nvrx-training
```

## 3. Configure FSx for Lustre Storage

Update `fsx-storage.yaml` with your FSx filesystem details:

```bash
# Find your FSx filesystem details
aws fsx describe-file-systems \
  --query 'FileSystems[*].{Id:FileSystemId,DNS:DNSName,Mount:LustreConfiguration.MountName}'
```

Replace the placeholder values in `fsx-storage.yaml`:
- `<YOUR-FSX-FILESYSTEM-ID>` -- e.g., `fs-0123456789abcdef0`
- `<YOUR-FSX-DNS-NAME>` -- e.g., `fs-0123456789abcdef0.fsx.us-west-2.amazonaws.com`
- `<YOUR-FSX-MOUNT-NAME>` -- e.g., `abcd1234`

Then apply:

```bash
kubectl apply -f fsx-storage.yaml
```

## 4. Configure Environment

```bash
cp ../env_vars.template env_vars
```

Edit `env_vars` for your environment. Key settings:

| Variable | Description | Example |
|----------|-------------|---------|
| `IMAGE_TAG` | Docker image tag | `latest` |
| `NODE_TYPE` | K8s node label for nodeSelector | `p5-h100` |
| `NUM_NODES` | Number of training pods | `2` |
| `GPU_PER_NODE` | GPUs per node | `8` |
| `EFA_PER_NODE` | EFA adapters per node (0 for g5) | `32` |
| `FSX_PVC_NAME` | Name of FSx PVC (from step 3) | `fsx-checkpoint-pvc` |
| `DEDICATED_TAINT_VALUE` | Node taint value (empty if no taints) | `p5` |
| `MODEL_NAME` | HuggingFace model name | `meta-llama/Llama-3.1-8B` |

Then source:

```bash
source env_vars
```

## 5. Prepare Dataset (Optional)

For fault recovery experiments with frequent restarts, pre-download the dataset to avoid HuggingFace API rate limiting:

```bash
source env_vars

kubectl run prepare-dataset -n nvrx-training \
  --image=$IMAGE_URI --restart=Never \
  --overrides='{"spec":{"nodeSelector":{"node-type":"'$NODE_TYPE'"},
    "containers":[{"name":"c","image":"'$IMAGE_URI'",
      "command":["python3","/app/prepare_dataset.py",
        "--output_path","/checkpoints/c4_subset","--num_samples","100000"],
      "env":[{"name":"HF_HOME","value":"/checkpoints/.cache"},
             {"name":"HF_TOKEN","valueFrom":{"secretKeyRef":{"name":"huggingface-token","key":"token"}}}],
      "volumeMounts":[{"name":"ck","mountPath":"/checkpoints"}],
      "resources":{"limits":{"nvidia.com/gpu":"1"}}}],
    "volumes":[{"name":"ck","persistentVolumeClaim":{"claimName":"'$FSX_PVC_NAME'"}}]}}'

# Wait for completion and clean up
kubectl logs prepare-dataset -n nvrx-training --follow
kubectl delete pod prepare-dataset -n nvrx-training
```

Fault recovery manifests already include `--dataset_path=/checkpoints/c4_subset`. For checkpoint experiments (no restarts), streaming mode is used by default.

## 6. Launch Training Jobs

Preview a manifest before deploying:

```bash
./deploy.sh --dry-run training-job-inprocess.yaml
```

Deploy:

```bash
./deploy.sh training-job-inprocess.yaml
```

### Available Manifests

| Manifest | NVRx Feature | Fault Injection |
|----------|-------------|-----------------|
| `training-job-inprocess.yaml` | In-process restart (NVRx Wrapper) | Yes (exception) |
| `training-job-inprocess-baseline.yaml` | Baseline (K8s container restart) | Yes (exception + hang) |
| `training-job-ft-launcher.yaml` | ft_launcher in-job restart | Yes (exception + hang) |
| `training-job-ft-launcher-inprocess.yaml` | Combined (ft_launcher + in-process) | Yes (exception + sigkill) |
| `training-job-async-ckpt.yaml` | Async checkpointing (NVRx) | No |
| `training-job-async-ckpt-baseline.yaml` | Sync checkpointing (torch.save) | No |
| `training-job-local-ckpt.yaml` | NVRx local checkpointing | No |
| `training-job-local-ckpt-baseline.yaml` | Standard torch.save baseline | No |

## 7. Monitor Training

```bash
# Check pod status
kubectl get pods -n nvrx-training

# Follow training logs
kubectl logs -f <pod-name> -n nvrx-training

# Check fault injection and recovery events
kubectl logs <pod-name> -n nvrx-training | grep -E "INJECTING FAULT|Recovery overhead|TRAINING SUMMARY"
```

## 8. Stop Training

```bash
./deploy.sh --delete training-job-inprocess.yaml
```

Or delete directly:

```bash
kubectl delete job <job-name> -n nvrx-training
```
