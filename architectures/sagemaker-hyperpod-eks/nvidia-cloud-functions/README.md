# NVIDIA Cloud Functions on Amazon SageMaker HyperPod EKS

Deploy [NVIDIA Cloud Functions (NVCF)](https://docs.nvidia.com/cloud-functions/user-guide/latest/cloud-function/overview.html) on an **existing** [Amazon SageMaker HyperPod](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-eks.html) EKS cluster using the Bring-Your-Own-Cluster (BYOC) model.

> **This guide assumes you already have a SageMaker HyperPod EKS cluster
> in `InService` state.** It does not create VPC, EKS, or SageMaker HyperPod
> infrastructure from scratch.

## What This Repo Contains

```
.
├── nvcf-config.env.template           # Configuration template (copy to nvcf-config.env)
├── infra/
│   └── scripts/
│       ├── 00-discover-cluster.sh      # Discover & validate existing SageMaker HyperPod cluster
│       ├── 01-prepare-cluster.sh       # Check tools, kubeconfig, scale GPU nodes
│       ├── 02-install-gpu-operator.sh  # NVIDIA GPU Operator (driver-skip mode)
│       ├── 03-register-nvca.sh         # Register cluster with NVCF
│       └── 04-validate-setup.sh        # Validate the full setup
├── nvcf/
│   ├── network-policy-patch.yaml       # Optional NetworkPolicy customization
│   └── sample-function/
│       ├── Dockerfile                  # NVCF-compatible echo container
│       ├── server.py                   # FastAPI echo server
│       ├── requirements.txt            # Python dependencies
│       └── deploy.sh                   # Build, push, create, deploy function
├── docs/
│   ├── COMPATIBILITY-ANALYSIS.md       # Detailed compatibility analysis
│   ├── DEPLOYMENT-GUIDE.md             # Step-by-step deployment runbook
│   └── TROUBLESHOOTING.md              # Common issues and solutions
└── tests/
    └── validate-cluster.sh             # End-to-end validation tests
```

## Architecture

```
Existing AWS VPC (private subnets + NAT Gateway)
  ├── Amazon EKS Control Plane (K8s 1.28-1.33)
  └── SageMaker HyperPod Worker Nodes (ml.g5.2xlarge+ / A10G recommended)
        ├── NVIDIA GPU Operator (driver-skip mode)
        ├── NVIDIA Cluster Agent (NVCA) --> NVCF Control Plane
        └── NVCF Function Pods (inference containers)
```

## Quick Start

```bash
# First: create your configuration file (fill in once, used by all scripts)
cp nvcf-config.env.template nvcf-config.env
vi nvcf-config.env                  # fill in your AWS, NGC, and NVCF values

# 0. Discover your existing SageMaker HyperPod EKS cluster
./infra/scripts/00-discover-cluster.sh

# 1. Prepare the cluster (tools, kubeconfig, scale GPU nodes)
./infra/scripts/01-prepare-cluster.sh --instance-group accelerated-worker-group-1 --target-count 1

# 2. Install the NVIDIA GPU Operator
./infra/scripts/02-install-gpu-operator.sh

# 3. Register with NVCF (fill in the four NVCA values in nvcf-config.env first)
./infra/scripts/03-register-nvca.sh

# 4. Validate
./infra/scripts/04-validate-setup.sh

# 5. Deploy a sample function (supports Docker or Finch)
./nvcf/sample-function/deploy.sh
```

See [docs/DEPLOYMENT-GUIDE.md](docs/DEPLOYMENT-GUIDE.md) for the full walkthrough.

## Compatibility Summary

| Capability | Status | Notes |
|------------|--------|-------|
| Kubernetes version | Compatible | K8s 1.28-1.32 documented; 1.33 may work (above documented max) |
| GPU detection | Compatible | GPU Operator with `driver.enabled=false` |
| Network policies | Partial | VPC CNI needs NetworkPolicy enabled (v1.14+) |
| Storage caching | Not supported | Disabled; not available for AWS EKS |
| Network egress | Compatible | NAT Gateway required for NVIDIA endpoints |
| Autoscaling | Partial | Queue-depth heuristic only on BYOC |
| Function logs | Not supported | Emit from containers to CloudWatch |

**No hard blockers.** See [docs/COMPATIBILITY-ANALYSIS.md](docs/COMPATIBILITY-ANALYSIS.md) for the full analysis.

## Key Decisions

- **Existing cluster**: This guide targets an existing SageMaker HyperPod EKS cluster -- no infrastructure creation from scratch.
- **Instance type**: `ml.g5.2xlarge` (8 vCPU, 32 GiB) or `ml.g5.8xlarge` (32 vCPU, 128 GiB) recommended to meet NVCA's 6 CPU + 8 GiB overhead requirement per GPU node.
- **GPU Operator**: Installed in driver-skip mode since SageMaker HyperPod nodes have pre-installed NVIDIA drivers.
- **Caching**: Disabled during NVCA registration (not supported for AWS EKS).
- **NetworkPolicy**: Enable VPC CNI NetworkPolicy (v1.14+) for workload isolation.

## References

- [NVCF Documentation](https://docs.nvidia.com/cloud-functions/user-guide/latest/cloud-function/overview.html)
- [NVCF Cluster Setup](https://docs.nvidia.com/cloud-functions/user-guide/latest/cloud-function/cluster-management.html)
- [SageMaker HyperPod EKS](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-eks.html)
- [HyperPod EKS Prerequisites](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-eks-prerequisites.html)
