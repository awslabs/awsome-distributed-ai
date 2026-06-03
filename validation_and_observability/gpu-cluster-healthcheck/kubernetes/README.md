# GPU Health Check Suite -- Kubernetes (EKS) Deployment

Kubernetes-native deployment of the GPU cluster health check suite for Amazon EKS clusters with NVIDIA GPUs. This deployment preserves the same check logic, severity classification, and "replace, don't repair" operational model as the Slurm deployment.

## Architecture Overview

The Kubernetes deployment consists of three components:

| Component | K8s Resource | Purpose | Checks |
|-----------|-------------|---------|--------|
| **Agent** | DaemonSet | Continuous lightweight checks on every GPU node | 0, 2, 3 (+ periodic L2) |
| **Sweeper** | CronJob | Rolling DCGM L2 sweep on idle nodes | 1 |
| **Quarantine** | Job (template) | Intensive diagnostics on suspect nodes | 4, 6 |

### Slurm-to-Kubernetes Concept Mapping

| Slurm Concept | Kubernetes Equivalent |
|---------------|----------------------|
| Prolog/epilog | DaemonSet agent (continuous) |
| `scontrol drain` | Taint `gpu-healthcheck.aws-samples.io/unhealthy=true:NoSchedule` |
| `scontrol reboot nextstate=resume` | Cordon + drain + node recycle (Karpenter/ASG) |
| Cron rolling sweep | CronJob sweeper |
| `sbatch --exclusive` quarantine job | Quarantine Job on cordoned node |
| Slurm node state | Node labels (`gpu-healthcheck.aws-samples.io/status`) |
| `squeue` / `sinfo` | `kubectl get nodes -l gpu-healthcheck.aws-samples.io/status` |

### How It Works

1. The **DaemonSet agent** runs on every GPU node (selected by `nvidia.com/gpu.present=true`). It runs fast checks (nvidia-smi, EFA, topology) every 5 minutes and DCGM L2 every 6 hours.

2. On failure, the agent:
   - Labels the node: `gpu-healthcheck.aws-samples.io/status=fail`
   - Applies a taint: `gpu-healthcheck.aws-samples.io/unhealthy=true:NoSchedule`
   - Annotates the node with a JSON summary of check results

3. On pass, the agent removes the taint and labels the node as healthy.

4. The **CronJob sweeper** runs every 6 hours, finds idle GPU nodes (no GPU-consuming pods), applies a maintenance taint, and runs DCGM L2 via per-node Jobs.

5. For suspect nodes, an operator creates a **quarantine Job** that runs intensive checks (DCGM L4, EFA loopback) and annotates the node with a recommendation.

## Prerequisites

- Amazon EKS cluster with GPU nodes (p4d, p5, p5e, p5en, p6-b200)
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/amazon-eks.html) or standalone [NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin)
- [EFA device plugin](https://github.com/aws/eks-charts/tree/master/stable/aws-efa-k8s-device-plugin) for EFA-equipped instances
- `kubectl` configured with cluster access
- Container registry (ECR, Docker Hub, etc.) for the health check image

## Quick Start

### 1. Build the Container Image

```bash
cd validation_and_observability/2.gpu-cluster-healthcheck

# Build
docker build -f kubernetes/Dockerfile -t gpu-healthcheck:latest .

# Tag and push (example with ECR)
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com
docker tag gpu-healthcheck:latest <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/gpu-healthcheck:latest
docker push <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/gpu-healthcheck:latest
```

### 2. Update Image References

Replace `YOUR_REGISTRY/gpu-healthcheck:latest` in the manifests:

```bash
export IMAGE="<ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/gpu-healthcheck:latest"
sed -i "s|YOUR_REGISTRY/gpu-healthcheck:latest|${IMAGE}|g" kubernetes/manifests/*.yaml
```

### 3. Deploy

```bash
kubectl apply -f kubernetes/manifests/
```

### 4. Verify

```bash
# Check agent pods are running on GPU nodes
kubectl get pods -n gpu-healthcheck -l app.kubernetes.io/component=agent -o wide

# Check node labels
kubectl get nodes -l nvidia.com/gpu.present=true \
    -L gpu-healthcheck.aws-samples.io/status,gpu-healthcheck.aws-samples.io/severity

# View agent logs
kubectl logs -n gpu-healthcheck -l app.kubernetes.io/component=agent --tail=50

# View detailed check results
kubectl get node <NODE> -o jsonpath='{.metadata.annotations.gpu-healthcheck\.aws-samples\.io/results}' | python3 -m json.tool
```

## Configuration Reference

All configuration is via the `gpu-healthcheck-config` ConfigMap in the `gpu-healthcheck` namespace.

| Key | Default | Description |
|-----|---------|-------------|
| `CHECK_INTERVAL` | `300` | Seconds between lightweight check cycles |
| `DCGM_L2_INTERVAL` | `21600` | Seconds between DCGM L2 runs (6 hours) |
| `CHECKS_LIGHTWEIGHT` | `0 2 3` | Space-separated check numbers for lightweight cycle |
| `ENABLE_TAINT` | `true` | Enable/disable automatic taint management |
| `ENABLE_LABEL` | `true` | Enable/disable node label updates |
| `DCGM_TIMEOUT` | `900` | DCGM diagnostic timeout in seconds |
| `CHECK_TIMEOUT` | `900` | Per-check timeout in seconds |
| `instance-profiles.conf` | (see file) | Hardware expectations per instance type |

To update configuration:

```bash
kubectl edit configmap gpu-healthcheck-config -n gpu-healthcheck
# Agent pods will pick up changes on next check cycle (no restart needed)
```

## Decision Flow (Kubernetes)

```
             Monitoring Alert / Pod Failure
                       │
                       ▼
              ┌─────────────────┐
              │ Agent detects   │  DaemonSet runs checks 0, 2, 3
              │ failure         │  (+ DCGM L2 periodically)
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Auto-taint node │  NoSchedule taint applied
              │ + label FAIL    │  New pods won't schedule here
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Check severity  │  kubectl get node -L ...status,...severity
              └────────┬────────┘
                       │
            ┌──────┬───┴────┬───────┐
            │      │        │       │
       ISOLATE  REBOOT   RESET   MONITOR
            │      │        │       │
            ▼      ▼        ▼       ▼
      ┌────────┐ ┌──────┐ ┌──────┐ ┌────────┐
      │Replace │ │Recycle│ │Cordon│ │Observe │
      │Node    │ │Node   │ │+ Job │ │+ Clear │
      └────────┘ └──┬───┘ └──┬───┘ └────────┘
                    │        │
                    ▼        ▼
              ┌───────────────────┐
              │ Quarantine Job    │  05-job-quarantine.yaml
              │ (checks 4, 6)    │
              └────────┬──────────┘
                       │
                ┌──────┴──────┐
                │             │
            PASS ▼         FAIL ▼
       ┌──────────┐    ┌──────────┐
       │ Uncordon │    │ Replace  │
       │ + remove │    │ Node     │
       │ taint    │    └──────────┘
       └──────────┘
```

## Operations Guide

### View GPU Health Status

```bash
# All GPU nodes with health status
kubectl get nodes -l nvidia.com/gpu.present=true \
    -L gpu-healthcheck.aws-samples.io/status,gpu-healthcheck.aws-samples.io/severity

# Only unhealthy nodes
kubectl get nodes -l gpu-healthcheck.aws-samples.io/status=fail

# Detailed results for a specific node
kubectl get node <NODE> -o jsonpath='{.metadata.annotations.gpu-healthcheck\.aws-samples\.io/results}' | python3 -m json.tool
```

### Quarantine a Suspect Node

```bash
# 1. Cordon the node (prevent new pods)
kubectl cordon <NODE>

# 2. Drain existing pods (optional, recommended for L4)
kubectl drain <NODE> --ignore-daemonsets --delete-emptydir-data

# 3. Run quarantine checks
sed "s/TARGET_NODE_NAME/<NODE>/g" kubernetes/manifests/05-job-quarantine.yaml | kubectl apply -f -

# 4. Monitor progress
kubectl logs -n gpu-healthcheck job/gpu-quarantine-<NODE> -f

# 5. Check result
kubectl get node <NODE> -o jsonpath='{.metadata.annotations.gpu-healthcheck\.aws-samples\.io/quarantine-result}' | python3 -m json.tool
```

### Return a Node to Service

```bash
# Remove unhealthy taint
kubectl taint node <NODE> gpu-healthcheck.aws-samples.io/unhealthy=true:NoSchedule-

# Update status label
kubectl label node <NODE> gpu-healthcheck.aws-samples.io/status=pass --overwrite

# Uncordon
kubectl uncordon <NODE>
```

### Replace a Node

For EKS managed node groups or Karpenter:

```bash
# Cordon and drain
kubectl cordon <NODE>
kubectl drain <NODE> --ignore-daemonsets --delete-emptydir-data

# For managed node groups: terminate the instance (ASG replaces it)
INSTANCE_ID=$(kubectl get node <NODE> -o jsonpath='{.spec.providerID}' | cut -d/ -f5)
aws ec2 terminate-instances --instance-ids "${INSTANCE_ID}"

# For Karpenter: delete the node (Karpenter provisions a replacement)
kubectl delete node <NODE>
```

## Pod Security and Privilege Requirements

The agent DaemonSet requires elevated privileges to access GPU hardware and host resources:

| Setting | Value | Reason |
|---------|-------|--------|
| `privileged: true` | Required | GPU device access, PCIe enumeration, kernel logs, RDMA |
| `hostNetwork: true` | Required | IMDS access for instance type detection, EFA loopback tests |
| `hostPID: true` | Required | Detect host processes (fabricmanager, nv-hostengine) via pgrep |
| No `nvidia.com/gpu` resource | Intentional | Avoids consuming GPU slots from the device plugin allocator |

The DaemonSet mounts these host paths read-only (except `/dev`):

| Mount | Purpose |
|-------|---------|
| `/dev` | GPU device files, RDMA devices |
| `/proc` (ro) | Host process information |
| `/sys` (ro) | PCIe topology, kernel parameters |
| `/dev/infiniband` | InfiniBand/EFA device files |
| `/var/log` (ro) | Kernel logs for Xid error scanning |
| `/run/log/journal` (ro) | systemd journal for kernel log access |
| `/etc/machine-id` (ro) | Required for journalctl access |

## Limitations

- **Check 5 (NCCL all_reduce)** is not included in the DaemonSet or sweeper because it requires multi-node coordination. Use an MPI Operator Job or similar for multi-node NCCL testing.
- **DCGM L4** requires exclusive node access. The quarantine Job template assumes the node has been cordoned and drained before running.
- The agent does **not** perform automatic node replacement. It taints and labels nodes; the operator (or automation built on top) decides whether to replace.
- Instance profiles in the ConfigMap must be kept in sync with the main `instance-profiles.conf` file.

## Troubleshooting

### Agent pod is CrashLoopBackOff

```bash
# Check pod logs
kubectl logs -n gpu-healthcheck <POD> --previous

# Common causes:
# - NODE_NAME not set (downward API misconfigured)
# - NVIDIA driver not loaded on host
# - Missing RBAC permissions
```

### Agent cannot detect instance type

The agent tries IMDS first (requires `hostNetwork: true`), then falls back to the Kubernetes node label `node.kubernetes.io/instance-type`. If both fail, it uses defaults.

```bash
# Verify IMDS is accessible from the node
kubectl exec -n gpu-healthcheck <POD> -- curl -s http://169.254.169.254/latest/meta-data/instance-type

# Verify node label
kubectl get node <NODE> -o jsonpath='{.metadata.labels.node\.kubernetes\.io/instance-type}'
```

### Taint not being applied/removed

```bash
# Check RBAC
kubectl auth can-i patch nodes --as=system:serviceaccount:gpu-healthcheck:gpu-healthcheck

# Check agent logs for taint errors
kubectl logs -n gpu-healthcheck <POD> | grep -i taint
```

### DCGM diagnostics timeout inside container

The DCGM timeout can be increased via the ConfigMap. L2 diagnostics on 8-GPU instances can take up to 10.5 minutes.

```bash
kubectl edit configmap gpu-healthcheck-config -n gpu-healthcheck
# Increase DCGM_TIMEOUT to 1200 (20 minutes)
```

### Sweeper jobs not creating

```bash
# Check CronJob status
kubectl get cronjob -n gpu-healthcheck

# Check sweeper logs
kubectl logs -n gpu-healthcheck -l app.kubernetes.io/component=sweeper

# Verify GPU nodes have the expected label
kubectl get nodes -l nvidia.com/gpu.present=true
```
