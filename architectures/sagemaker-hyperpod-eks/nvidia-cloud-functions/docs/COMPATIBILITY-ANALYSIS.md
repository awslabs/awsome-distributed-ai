# Compatibility Analysis: NVIDIA Cloud Functions on SageMaker HyperPod EKS

## Executive Summary

NVIDIA Cloud Functions (NVCF) can be deployed on a SageMaker HyperPod EKS cluster
using the Bring-Your-Own-Cluster (BYOC) model. The NVIDIA Cluster Agent (NVCA) is
installed on the existing EKS cluster, registering it as a deployment target for
NVCF functions. **There are no hard blockers**, but several compatibility
considerations require specific configuration choices.

## Architecture Overview

```
+----------------------------------------------------------+
|                    Existing AWS VPC                       |
|  +----------------------------------------------------+  |
|  |              Private Subnets                        |  |
|  |  +----------------------------------------------+  |  |
|  |  |         Amazon EKS Control Plane              |  |  |
|  |  |  (K8s API Server, etcd, controllers)          |  |  |
|  |  +----------------------------------------------+  |  |
|  |                      |                              |  |
|  |  +----------------------------------------------+  |  |
|  |  |     SageMaker HyperPod Worker Nodes           |  |  |
|  |  |     (ml.g5.2xlarge+ recommended / A10G GPUs)  |  |  |
|  |  |                                                |  |  |
|  |  |  +------------------+  +-------------------+  |  |  |
|  |  |  | NVIDIA GPU       |  | NVIDIA Cluster    |  |  |  |
|  |  |  | Operator         |  | Agent (NVCA)      |  |  |  |
|  |  |  +------------------+  +-------------------+  |  |  |
|  |  |                                                |  |  |
|  |  |  +------------------------------------------+  |  |  |
|  |  |  | NVCF Function Pods                        |  |  |  |
|  |  |  | (inference containers + infra sidecars)   |  |  |  |
|  |  |  +------------------------------------------+  |  |  |
|  |  +----------------------------------------------+  |  |
|  +----------------------------------------------------+  |
|                      |                                    |
|  +----------------------------------------------------+  |
|  |  NAT Gateway (outbound to NVIDIA control plane)    |  |
|  +----------------------------------------------------+  |
+----------------------------------------------------------+
           |
           v
+----------------------------------------------------------+
|           NVIDIA Cloud Functions Control Plane            |
|  - api.ngc.nvidia.com                                    |
|  - nvcr.io (container registry)                          |
|  - helm.ngc.nvidia.com                                   |
|  - connect.pnats.nvcf.nvidia.com                         |
|  - grpc.api.nvcf.nvidia.com                              |
+----------------------------------------------------------+
```

## NVCF BYOC Requirements vs. SageMaker HyperPod EKS Capabilities

### 1. Kubernetes Version

| Requirement | NVCF | SageMaker HyperPod EKS | Compatible |
|-------------|------|---------------|------------|
| Min K8s version | v1.25.0 | 1.28+ | YES |
| Max K8s version | v1.32.x (documented) | Up to 1.33 | SEE NOTE |

**NVCF documented range**: K8s 1.25 - 1.32. Both systems overlap at 1.28 - 1.32.

> **K8s 1.33 note**: SageMaker HyperPod EKS clusters can run Kubernetes
> 1.33, which is **above** NVCF's documented maximum of v1.32.x. The NVCA
> operator is likely to work on 1.33 since there are no known breaking K8s
> API changes between 1.32 and 1.33, but it is **not officially tested by
> NVIDIA**. If you encounter issues, consider downgrading the EKS control
> plane to 1.32, or wait for NVIDIA to update their documentation. The
> `00-discover-cluster.sh` and `04-validate-setup.sh` scripts will warn
> about this automatically.

### 2. GPU Operator

| Requirement | NVCF | SageMaker HyperPod EKS | Compatible |
|-------------|------|---------------|------------|
| NVIDIA GPU Operator | Required | Not pre-installed | NEEDS INSTALL |

SageMaker HyperPod EKS uses its own device plugin setup. The NVIDIA GPU Operator must be
installed separately via Helm. The operator should be configured to:
- **Skip driver installation** (SageMaker HyperPod nodes come with pre-installed NVIDIA drivers)
- Install only the device plugin, DCGM exporter, and GPU feature discovery components

### 3. Container Network Interface (CNI)

| Requirement | NVCF | SageMaker HyperPod EKS | Compatible |
|-------------|------|---------------|------------|
| CNI with NetworkPolicy | Recommended | AWS VPC CNI only | PARTIAL |

NVCF network policies will **not be enforced** with the default AWS VPC CNI unless
you enable the NetworkPolicy feature (available in VPC CNI v1.14+). The NVCF
documentation acknowledges this: _"If your cluster uses a CNI that doesn't support
network policies, the security controls described below will not be enforced."_

**Impact**: Functions will work, but workload pods can communicate without restrictions.

**Mitigation options**:
1. Accept the risk (functions are already isolated by Kubernetes namespaces)
2. Enable AWS VPC CNI Network Policy support (v1.14+, using eBPF)
3. Install Calico in policy-only mode alongside VPC CNI

### 4. Storage and Caching

| Requirement | NVCF | SageMaker HyperPod EKS | Compatible |
|-------------|------|---------------|------------|
| Caching support | Optional | N/A | NOT SUPPORTED |

NVCF documentation explicitly states: **"Caching is currently not supported for
AWS EKS."** StorageClass examples are provided only for GCP, Azure, and Oracle Cloud.

**Impact**: Cold starts will be slower. Models, resources, and containers must be
pulled fresh for each function deployment or scale-up event.

**Mitigation**: Disable "Caching Support" feature during NVCA registration. Use
the EBS CSI driver for any custom persistent storage needs.

### 5. Network Connectivity

| Requirement | NVCF | SageMaker HyperPod EKS | Compatible |
|-------------|------|---------------|------------|
| Outbound internet | Required (NVIDIA endpoints) | Private subnets only | NEEDS NAT GW |

NVCA requires outbound access to:
- `nvcr.io` and `helm.ngc.nvidia.com` (container/chart pulls)
- `connect.pnats.nvcf.nvidia.com` (control plane messaging)
- `grpc.api.nvcf.nvidia.com` and `*.api.nvcf.nvidia.com` (API)
- `sqs.*.amazonaws.com` (AWS SQS, used by NVCF internally)
- `spot.gdn.nvidia.com`, `ess.ngc.nvidia.com`, `api.ngc.nvidia.com`

SageMaker HyperPod requires private subnets. A **NAT Gateway is mandatory** for outbound access.
The `00-discover-cluster.sh` script checks for this automatically.

### 6. RBAC and Authentication

| Requirement | NVCF | SageMaker HyperPod EKS | Compatible |
|-------------|------|---------------|------------|
| cluster-admin RBAC | Required | Supported via IAM | YES |

SageMaker HyperPod EKS supports `API` and `API_AND_CONFIG_MAP` authentication modes. An IAM
role/user can be granted `cluster-admin` via EKS access entries.

### 7. Node Resource Overhead

| Requirement | NVCF | SageMaker HyperPod EKS | Compatible |
|-------------|------|---------------|------------|
| Per-node overhead | 6 CPU + 8Gi RAM per GPU node | Depends on instance type | SEE TABLE |

**Important**: NVCF requires 6 CPU cores and 8 GiB for infrastructure containers
**per GPU node**.

| Instance Type | vCPU | Memory | Fits NVCA Overhead? |
|---------------|------|--------|---------------------|
| ml.g5.xlarge | 4 | 16 GiB | NO - only 4 vCPU, not enough |
| ml.g5.2xlarge | 8 | 32 GiB | YES - tight but workable |
| ml.g5.8xlarge | 32 | 128 GiB | YES - comfortable headroom |
| ml.p4d.24xlarge | 96 | 1152 GiB | YES - ample resources |

**Recommendation**: Use `ml.g5.2xlarge` or larger. If your cluster already has
`ml.g5.8xlarge` instance groups, those are the best choice.

### 8. SageMaker HyperPod-Specific Considerations

| Feature | Impact on NVCF |
|---------|----------------|
| Deep health checks | Temporary `NoSchedule` taint may disrupt running functions |
| Node auto-recovery | Beneficial -- failed nodes replaced automatically by SageMaker HyperPod |
| `ml.` instance type prefix | NVCA Dynamic GPU Discovery handles this via GPU Operator labels |
| Max pods per node | Depends on instance type; sufficient for typical function deployments |
| EFA networking | Not required by NVCF but beneficial for multi-node workloads |

### 9. Function Logs

| Requirement | NVCF | SageMaker HyperPod EKS (BYOC) | Compatible |
|-------------|------|----------------------|------------|
| Function-level logs | Available on NVIDIA-managed | NOT supported on BYOC | NO |

Function-level inference container logs are **not supported** on BYOC clusters.
Applications must emit logs from their containers directly (e.g., to stdout for
collection by Fluent Bit or CloudWatch agent).

### 10. Autoscaling

| Requirement | NVCF | SageMaker HyperPod EKS (BYOC) | Compatible |
|-------------|------|----------------------|------------|
| Autoscaling heuristics | Multiple on managed | Queue-depth only on BYOC | PARTIAL |

Only the **function queue depth** heuristic is supported on BYOC clusters. Other
utilization-based heuristics available on NVIDIA-managed clusters are not supported.

## Compatibility Matrix Summary

| Capability | Status | Action Required |
|------------|--------|-----------------|
| Kubernetes version | COMPATIBLE (1.28-1.32) | K8s 1.33 above documented max -- test carefully |
| GPU detection | COMPATIBLE | Install GPU Operator (driver-skip mode) |
| Network policies | PARTIAL | Accept risk or enable VPC CNI NetworkPolicy |
| Storage caching | NOT SUPPORTED | Disable caching feature |
| Network egress | COMPATIBLE | Verify NAT Gateway exists in VPC |
| RBAC | COMPATIBLE | Standard IAM setup |
| Node resources | DEPENDS ON INSTANCE | Use ml.g5.2xlarge or larger |
| Autoscaling | PARTIAL | Only queue-depth heuristic supported on BYOC |
| Function logs | NOT SUPPORTED | Emit logs from containers directly to CloudWatch |

## Key Recommendations

1. **Use ml.g5.2xlarge or larger** to meet NVCA's 6 CPU + 8 GiB overhead requirement.
2. **Verify NAT Gateway** exists in the VPC for outbound NVIDIA control plane access.
3. **Install GPU Operator in lightweight mode** -- skip driver install, use
   pre-installed drivers on SageMaker HyperPod nodes.
4. **Disable caching** during NVCA registration (not available for AWS EKS).
5. **Implement container-level logging** since function-level logs are not supported
   on non-NVIDIA-managed (BYOC) clusters.
6. **Enable Dynamic GPU Discovery** (default) to handle GPU detection automatically.
7. **Consider VPC CNI v1.14+ with NetworkPolicy** for workload isolation.
8. **Test K8s 1.33 carefully** if your cluster is running it; fall back to 1.32 if
   NVCA has issues.

## References

- [NVCF Cluster Setup & Management](https://docs.nvidia.com/cloud-functions/user-guide/latest/cloud-function/cluster-management.html)
- [NVCF Function Creation](https://docs.nvidia.com/cloud-functions/user-guide/latest/cloud-function/function-creation.html)
- [NVCF Function Deployment](https://docs.nvidia.com/cloud-functions/user-guide/latest/cloud-function/function-deployment.html)
- [SageMaker HyperPod EKS Overview](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-eks.html)
- [SageMaker HyperPod EKS Prerequisites](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-eks-prerequisites.html)
