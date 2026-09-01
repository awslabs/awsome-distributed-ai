---
name: validate-deployment
description: Post-deployment health checks for slinky-slurm including pod status, Slurm node registration, SSH connectivity, and test job submission
---

# Validate Deployment

## Overview

Use this skill after completing all deployment phases (`deploy.sh`,
`setup.sh`/`install.sh`) to verify the Slurm cluster is healthy and
operational. This skill provides a systematic checklist that covers
infrastructure, Kubernetes workloads, Slurm services, network access, and
workload execution.

## Prerequisites

- All three deployment phases completed successfully
- `kubectl` configured to point to the HyperPod EKS cluster
- SSH key available at `~/.ssh/id_ed25519_slurm`
- NLB hostname available (printed by `install.sh`)

## Steps

### Step 1: Verify Kubernetes Pod Health

Check that all pods in the key namespaces are Running and Ready:

```bash
echo "=== cert-manager namespace ==="
kubectl -n cert-manager get pods -o wide

echo "=== kube-system (LB Controller) ==="
kubectl -n kube-system get pods -l app.kubernetes.io/name=aws-load-balancer-controller -o wide

echo "=== slurm namespace ==="
kubectl -n slurm get pods -o wide

echo "=== slinky namespace ==="
kubectl -n slinky get pods -o wide

echo "=== mariadb namespace ==="
kubectl -n mariadb get pods -o wide
```

**Expected state:**

| Namespace | Key Pods | Status |
|-----------|----------|--------|
| `cert-manager` | `cert-manager-*`, `cert-manager-webhook-*`, `cert-manager-cainjector-*` | Running |
| `kube-system` | `aws-load-balancer-controller-*` (x2) | Running |
| `slurm` | `slurm-controller-*`, `slurm-slurmd-slinky-*` (x replicas), `slurm-login-slinky-*`, `slurm-restapi-*`, `slurm-accounting-*` | Running |
| `slinky` | `slurm-operator-*` | Running |
| `mariadb` | `mariadb-operator-*`, `mariadb-*` | Running |

The number of `slurm-slurmd-slinky-*` pods should match the
`--instance-count` used during deployment (default: 4 for
`ml.g5.8xlarge`, 2 for `ml.p5.48xlarge`).

**If any pods are not Running:**

```bash
# Get detailed pod status
kubectl -n slurm describe pod <pod-name>

# Check pod logs
kubectl -n slurm logs <pod-name> --tail=50

# Check events
kubectl -n slurm get events --sort-by='.lastTimestamp' | tail -20
```

### Step 2: Verify Kubernetes Node Health

```bash
kubectl get nodes -o wide
```

**Expected:**
- Management nodes: `ml.m5.4xlarge`, STATUS=Ready
- Accelerated nodes: your chosen instance type (e.g., `ml.g5.8xlarge`),
  STATUS=Ready

**If nodes show NotReady:**

```bash
kubectl describe node <node-name>
# Check the Conditions section for issues
```

### Step 3: Verify Login Service and NLB

```bash
kubectl get svc slurm-login-slinky -n slurm
```

**Expected:** An external hostname under `EXTERNAL-IP`, type `LoadBalancer`.

```bash
# Get the NLB hostname
SLURM_LOGIN=$(kubectl get svc slurm-login-slinky -n slurm \
    -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
echo "Login endpoint: ${SLURM_LOGIN}"

# Test DNS resolution
nslookup "${SLURM_LOGIN}"

# Test SSH port connectivity (port 22)
nc -zv "${SLURM_LOGIN}" 22
```

### Step 4: SSH to Login Node

```bash
ssh -i ~/.ssh/id_ed25519_slurm root@<NLB_HOSTNAME>
```

**Expected:** Successful SSH login as root to the Slurm login node.

If SSH fails with "Connection refused" or timeout, the NLB may still be
provisioning. Wait 2-3 minutes and retry.

### Step 5: Verify Slurm Services (from login node)

After SSH'ing into the login node:

```bash
# Check Slurm controller status
scontrol show config | head -5

# Check Slurm version
sinfo --version

# Check registered nodes
sinfo
# Expected: Nodes in "idle" state under the correct partition

# Check partitions
scontrol show partitions

# Check node details
scontrol show nodes

# Verify Slurm accounting
sacctmgr show clusters
```

**Expected `sinfo` output (example for `ml.g5.8xlarge` with 4 nodes):**

```
PARTITION  AVAIL  TIMELIMIT  NODES  STATE  NODELIST
slinky*    up     infinite   4      idle   slurm-slurmd-slinky-[0-3]
```

Node count should match `--instance-count` used during deployment.

### Step 6: Submit a Test Job

From the login node, submit a basic test job:

```bash
# Simple test job
srun hostname

# Or submit a batch job
sbatch --wrap="hostname && nvidia-smi" --output=/tmp/test-%j.out
squeue  # Monitor job status
cat /tmp/test-*.out  # Check output after completion
```

### Step 7: Submit a Training Job (Optional)

For a full end-to-end validation, submit the Llama2 7B training job.
Only the `ml.g5.8xlarge` sbatch has been validated end-to-end:

**For `ml.g5.8xlarge`:**
```bash
sbatch sbatch/fsdp/g5-llama2_7b-training.sbatch
squeue  # Monitor
```

**For `ml.p5.48xlarge`:**
```bash
sbatch sbatch/fsdp/p5-llama2_7b-training.sbatch
squeue  # Monitor
```

These jobs run a distributed training workload using PyTorch FSDP and
validate GPU compute, inter-node communication (EFA), and the shared
filesystem (FSx Lustre).

## Quick Validation Script

Run this from your local machine (not the login node) for a rapid health
check:

```bash
#!/bin/bash
echo "=== cert-manager ==="
kubectl -n cert-manager get pods --no-headers | awk '{print $1, $3}'

echo ""
echo "=== LB Controller ==="
kubectl -n kube-system get pods -l app.kubernetes.io/name=aws-load-balancer-controller \
    --no-headers | awk '{print $1, $3}'

echo ""
echo "=== Pod Status ==="
kubectl -n slurm get pods -o custom-columns=\
'NAME:.metadata.name,STATUS:.status.phase,READY:.status.conditions[?(@.type=="Ready")].status'

echo ""
echo "=== Node Status ==="
kubectl get nodes -o custom-columns=\
'NAME:.metadata.name,STATUS:.status.conditions[?(@.type=="Ready")].status,TYPE:.metadata.labels.node\.kubernetes\.io/instance-type'

echo ""
echo "=== Login Service ==="
kubectl get svc slurm-login-slinky -n slurm \
    -o custom-columns='NAME:.metadata.name,TYPE:.spec.type,EXTERNAL:.status.loadBalancer.ingress[0].hostname'

echo ""
echo "=== Slurm Pods Count ==="
echo "Controller: $(kubectl -n slurm get pods -l app.kubernetes.io/name=controller --no-headers | wc -l)"
echo "Compute:    $(kubectl -n slurm get pods -l app.kubernetes.io/name=slurmd --no-headers | wc -l)"
echo "Login:      $(kubectl -n slurm get pods -l app.kubernetes.io/name=login --no-headers | wc -l)"
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Pods in `CrashLoopBackOff` | Configuration error in values file or missing dependencies | Check logs: `kubectl -n slurm logs <pod> --previous` |
| Pods in `Pending` | Insufficient node resources or node selector mismatch | Check: `kubectl -n slurm describe pod <pod>` -- look at Events |
| Nodes in `drain` state | Slurm put nodes in drain due to health check failure | On login node: `scontrol update nodename=<node> state=idle reason=""` |
| Nodes in `down` state | Slurm controller cannot reach slurmd | Check slurmd logs: `kubectl -n slurm logs slurm-slurmd-slinky-<n>` |
| `sinfo` shows no nodes | Slurm controller not yet synchronized | Wait 1-2 minutes; check controller logs: `kubectl -n slurm logs <controller-pod>` |
| NLB not resolving | DNS propagation delay | Wait 2-5 minutes; verify with `nslookup` |
| SSH "Connection refused" | NLB target not yet registered or SSH not started | Check NLB targets in AWS console; verify login pod is Running |
| SSH "Permission denied" | Wrong SSH key | Verify key matches: `ssh -i ~/.ssh/id_ed25519_slurm -v root@<host>` |
| `nvidia-smi` fails in job | GPU not available to pod | Check pod resource requests and node GPU capacity |
| Training job fails immediately | Missing NCCL/EFA libraries | Check container image has required libraries; review Dockerfile |
| `squeue` shows job stuck in PD | Resources unavailable or partition misconfigured | Check: `scontrol show job <jobid>` for Reason field |

## Key Logs to Investigate

When issues arise, these are the most useful log sources:

```bash
# Slurm controller (manages scheduling)
kubectl -n slurm logs -l app.kubernetes.io/name=controller --tail=100

# Slurm compute nodes (slurmd)
kubectl -n slurm logs slurm-slurmd-slinky-0 --tail=100

# Slurm operator (manages CRDs)
kubectl -n slinky logs -l app.kubernetes.io/name=slurm-operator --tail=100

# MariaDB (accounting database)
kubectl -n slurm logs mariadb-0 --tail=50
```

## References

- `install.sh` -- Cluster installation script (produced the deployment)
- `sbatch/fsdp/g5-llama2_7b-training.sbatch` -- g5 training job (4 nodes, 1 GPU each)
- `sbatch/fsdp/p5-llama2_7b-training.sbatch` -- p5 training job (4 nodes, 8 GPUs each)
- `slurm-values.yaml` -- Generated Helm values (runtime config)
- `slurm-login-service-patch.yaml` -- Generated NLB service patch
