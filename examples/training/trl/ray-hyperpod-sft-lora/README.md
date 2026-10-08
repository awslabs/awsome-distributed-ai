# Ray on SageMaker HyperPod — Studio Workflow

Fine-tune [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) on medical reasoning data using Ray TorchTrainer with LoRA, then deploy the model for inference with Ray Serve — all on Amazon SageMaker HyperPod. Uses the [Ray on HyperPod](https://aws.amazon.com/blogs/machine-learning/introducing-new-ray-capabilities-on-sagemaker-hyperpod/) integration in SageMaker Studio — no kubectl, Helm, or Kubernetes YAML required.

Two variants are provided: **FSx** (shared file system) and **S3** (no shared file system needed).

## Architecture

```
SageMaker Studio
├── Create Ray Cluster (Tasks tab → RayCluster → Create)
├── Open Ray Dashboard (IAM-authenticated URL)
├── Open Grafana (4 pre-built dashboards)
└── Connect Workspace (JupyterLab / Code Editor)
        │
        ▼
HyperPod EKS Cluster
├── Ray Head Node (no GPU)
├── Ray Workers (GPU instances)
├── KubeRay Operator (manages Ray CRDs)
├── Job Monitoring Agent (hung job detection)
├── HyperPod Observability → Amazon Managed Grafana
└── Shared Storage: FSx for Lustre OR Amazon S3
```

## Files

### FSx variant (shared file system)

| File | Description |
|------|-------------|
| `notebook.ipynb` | 13-step walkthrough: cluster setup → training → model upload to S3 |
| `train_ray.py` | Training script — FSDP + LoRA fine-tuning with Ray TorchTrainer. Reads data from FSx (`/mnt/fsx/`), writes checkpoints and output to FSx |
| `args.yaml` | Training config — model ID, FSx dataset/checkpoint/output paths, LoRA parameters, SFT hyperparameters |
| `requirements.txt` | Full dependency list including Ray, PyTorch, and SageMaker integration packages |

### S3 variant (no shared file system)

| File | Description |
|------|-------------|
| `notebook-s3_withoutputs.ipynb` | 14-step walkthrough: cluster setup → training → **Ray Serve deployment** → cleanup. Includes cell outputs from a successful run |
| `train_ray_s3.py` | Training script — downloads data from S3 to local disk, trains with LoRA, uploads final model to S3. Heavy ML imports deferred to workers so the CPU head node stays lightweight |
| `args-s3.yaml` | Training config — `s3_base_uri` replaces FSx paths, local `/tmp` dirs for worker-local storage |
| `requirements-s3.txt` | Minimal dependency list — omits Ray and PyTorch (reuses the SageMaker Distribution image versions) |

### Key differences

| | FSx variant | S3 variant |
|-|-------------|------------|
| **Shared storage** | FSx for Lustre (`/mnt/fsx/`) | Amazon S3 (`s3_base_uri` in config) |
| **Data access** | Direct filesystem reads | Workers download from S3 to local `/tmp` |
| **Checkpoints** | Ray writes to FSx, async upload to S3 | Ray writes directly to S3 (`RunConfig.storage_path`) |
| **Model serving** | Not included | Ray Serve deployment on the same cluster |
| **Dependencies** | Pins Ray + PyTorch versions | Reuses image versions (no conflicts) |
| **Infrastructure** | Requires FSx PV/PVC on EKS | Only S3 bucket + EKS Pod Identity |

## Prerequisites

Four components must be installed on your HyperPod EKS cluster:

| Component | Purpose |
|-----------|---------|
| SageMaker Spaces EKS add-on | JupyterLab / Code Editor workspaces |
| HyperPod Observability EKS add-on | Metrics collection, Grafana dashboards |
| KubeRay operator | Manages RayCluster, RayJob, RayService CRDs |
| HyperPod Ray Endpoint Operator | Authenticated endpoints for dashboard and remote job submission |

Plus:
- A SageMaker Studio domain with access to the HyperPod cluster
- **FSx variant**: An FSx for Lustre file system attached to the cluster
- **S3 variant**: An S3 bucket/prefix and EKS Pod Identity associations for the Ray namespace

## Instance Types

### Default: 1 GPU (Qwen3-0.6B with LoRA)

Qwen3-0.6B with LoRA trains ~0.5% of parameters and fits on 1 GPU. Any single-GPU instance works:

| Role | Instance | Count | Why |
|------|----------|-------|-----|
| Worker | Any GPU instance (`g5.xlarge`+) | 1 | 1 GPU is enough for 0.6B + LoRA |
| Head / System | `m5.2xlarge` or similar | 1 | No GPU needed |

### Scaling Up for Larger Models

For models that don't fit on a single GPU (7B+), increase `num_workers` in the config. FSDP shards the model across workers.

| Model Size | Recommended | Config changes |
|------------|-------------|----------------|
| < 3B (e.g., Qwen3-0.6B) | 1 GPU | Default — `num_workers: 1` |
| 7B–13B | 2–4 GPUs | `num_workers: 4`, uncomment FSDP |
| 30B–70B | 8+ GPUs | `num_workers: 8`, uncomment FSDP |

Multi-GPU instance options:

| Instance | GPUs/node | Notes |
|----------|-----------|-------|
| `g5.16xlarge` | 1 A10G | Scale across multiple nodes |
| `g5.12xlarge` | 4 A10G | Multi-GPU on a single node |
| `g5.48xlarge` | 8 A10G | Full node for large models |
| `p4d.24xlarge` | 8 A100 | High-end training |

## Quick Start

1. Set up prerequisites (see above)
2. Create a Ray cluster from SageMaker Studio → HyperPod → Tasks tab → RayCluster
3. Create a JupyterLab space and attach it to the Ray cluster
4. Open the notebook in the workspace and follow the steps:
   - **FSx variant**: `notebook.ipynb`
   - **S3 variant**: `notebook-s3_withoutputs.ipynb`

## Remote Job Submission

From a laptop or CI/CD pipeline (no Studio required):

```bash
pip install toolkit-for-ray-on-sagemaker-ai
aws eks update-kubeconfig --name <eks-cluster-name> --region <region>

# FSx variant
ray job submit \
    --address sagemaker_ray://<ray-cluster-name>/<namespace> \
    --working-dir . \
    -- python train_ray.py --config args.yaml

# S3 variant
ray job submit \
    --address sagemaker_ray://<ray-cluster-name>/<namespace> \
    --working-dir . \
    --runtime-env-json '{"pip": "requirements-s3.txt"}' \
    -- python train_ray_s3.py --config args-s3.yaml
```

## Model Serving (S3 variant)

After training, the S3 notebook deploys the model on the same Ray cluster using Ray Serve:

- `serve.run()` starts the endpoint on the existing GPU workers — no new infrastructure
- The `MedicalQA` deployment downloads the merged model from S3, loads it on GPU, and serves HTTP requests
- Endpoint is available at `http://<head-svc>:8000/medical-qa` within the cluster
- Scale horizontally by increasing `num_replicas` in the `@serve.deployment` decorator

## Resilience

Three automatic layers, no code changes required:

| Layer | What happens |
|-------|-------------|
| **Node recovery** | HyperPod replaces faulty nodes → Ray reschedules pods → `FailureConfig` retries from last checkpoint |
| **Hung job detection** | Per-node Job Monitoring Agent detects stalls → alerts via CloudWatch and Grafana |
| **Tiered checkpointing** | Writes to local disk first, async uploads to S3 → faster recovery than restoring from S3 |
| **Serve health checks** | Ray Serve monitors replica health → unhealthy replicas restarted automatically |

## Key Packages

| Package | Purpose |
|---------|---------|
| `toolkit-for-ray-on-sagemaker-ai` | Remote job submission (`sagemaker_ray://` address), JumpStart model loader, custom hung job detection rules |
| `amzn-sagemaker-checkpointing` | Tiered checkpointing integration (local disk → S3) |
| SageMaker Distribution image | Default container with Ray pre-installed (AWS-maintained) |
