# Troubleshooting: NVCF on SageMaker HyperPod EKS

Common issues and solutions when running NVIDIA Cloud Functions on an
existing SageMaker HyperPod EKS cluster.

---

## 1. Cluster Discovery Issues

### a. No SageMaker HyperPod clusters found
**Symptoms**: `00-discover-cluster.sh` reports no clusters in the region.

```bash
[ERROR] No SageMaker HyperPod clusters found in us-west-2.
```

**Fixes**:
- Verify the correct AWS region: `export AWS_REGION="<your-aws-region>"`
- Verify the correct AWS profile: `export AWS_PROFILE="<your-aws-profile>"`
- Check that the SageMaker HyperPod cluster exists: `aws sagemaker list-clusters --region ${AWS_REGION}`

### b. Cluster not in InService state
**Symptoms**: Cluster is in `Creating`, `Updating`, or `Failed` state.

**Fixes**:
- `Creating`/`Updating`: Wait for the operation to complete.
- `Failed`: Check the cluster events in the AWS console. Common causes include
  IAM permission issues, subnet configuration problems, or instance quota limits.

### c. NAT Gateway not found
**Symptoms**: `00-discover-cluster.sh` warns about missing NAT Gateway.

**Fix**: NVCF requires outbound internet access to NVIDIA endpoints. Add a NAT
Gateway to the VPC and update the private subnet route table with a `0.0.0.0/0`
route pointing to it.

---

## 2. GPU Instance Group Scaling Issues

### a. Scale-up request fails
**Symptoms**: `01-prepare-cluster.sh` fails when scaling up the GPU instance group.

**Possible causes**:
- **Insufficient quota**: Check your SageMaker HyperPod quota for the instance type:
  ```bash
  aws service-quotas get-service-quota \
       --service-code sagemaker \
       --quota-code <quota-code> \
       --region ${AWS_REGION}
  ```
  Request a quota increase via the AWS Service Quotas console if needed.

- **Capacity unavailable**: The instance type may not have capacity in the AZ.
  Try a different instance type or wait.

### b. GPU nodes not joining the EKS cluster
**Symptoms**: `01-prepare-cluster.sh` scaled the group, but `kubectl get nodes` doesn't
show new nodes after 15+ minutes.

**Fixes**:
- Check the instance group status:
  ```bash
  aws sagemaker describe-cluster --cluster-name <your-hyperpod-cluster-name> --region ${AWS_REGION} \
       --query 'InstanceGroups[?contains(InstanceGroupName, `gpu`)]'
  ```
- Check EKS for node join issues:
  ```bash
  kubectl get events --sort-by='.metadata.creationTimestamp' -A | tail -20
  ```
- The SageMaker HyperPod lifecycle script on the nodes may have failed. Check S3 for logs
  if the lifecycle config writes to an S3 bucket.

---

## 3. NVCA Backend Shows "Not Ready" or "unhealthy"

**Symptoms**:
```
kubectl get nvcfbackend -n nvca-operator
# NAME              AGE   VERSION   HEALTH
# nvcf-hyperpod     5m              unhealthy
```

**Possible causes and fixes**:

### a. No outbound internet connectivity
NVCA needs to reach NVIDIA control plane endpoints. SageMaker HyperPod uses private
subnets, so a NAT Gateway is required.

```bash
# Check if pods can resolve NVIDIA endpoints
kubectl run dns-test --image=busybox:1.36 --restart=Never --rm -i -- \
    nslookup api.ngc.nvidia.com

# If DNS fails, check NAT Gateway and route tables
aws ec2 describe-nat-gateways --filter "Name=state,Values=available" --region ${AWS_REGION}
```

**Fix**: Ensure the VPC has a NAT Gateway and the private subnet route table
has a `0.0.0.0/0` route pointing to it.

### b. Invalid NGC Cluster Key
The cluster key may have expired (default: 90-day expiration).

```bash
# Check NVCA operator logs for authentication errors
kubectl logs -l app.kubernetes.io/instance=nvca-operator -n nvca-operator --tail 50
```

**Fix**: Rotate the cluster key in the NVCF UI at **Settings** > **Clusters** >
**Rotate Key**, then re-run the Helm install with the new key.

### c. GPU Operator not installed
NVCA with Dynamic GPU Discovery requires the GPU Operator.

```bash
kubectl get namespace gpu-operator
kubectl get pods -n gpu-operator
```

**Fix**: Run `./infra/scripts/02-install-gpu-operator.sh`.

### d. Kubernetes version mismatch
If running K8s 1.33 (above NVCF's documented max of 1.32.x), the NVCA
operator may encounter unexpected API behavior.

```bash
kubectl version -o json | python3 -c "import sys,json; print(json.load(sys.stdin)['serverVersion']['gitVersion'])"
```

**Fix**: If NVCA shows issues on 1.33, consider downgrading the EKS control
plane to 1.32:
```bash
# Note: EKS does not support downgrading. You would need to create a new
# cluster with 1.32 and migrate. Alternatively, wait for NVIDIA to add
# 1.33 support to NVCA.
```

---

## 4. GPU Nodes Not Discovered by NVCA

**Symptoms**: The NVCF UI shows the cluster as Ready but with 0 GPUs.

**Possible causes**:

### a. GPU Operator not labeling nodes
```bash
# Check for GPU labels
kubectl get nodes -l nvidia.com/gpu.present=true

# If empty, check GPU Operator pods
kubectl get pods -n gpu-operator
kubectl logs -l app=gpu-feature-discovery -n gpu-operator --tail 30
```

**Fix**: The GPU Operator may need the nodes to be ready first. Wait for
SageMaker HyperPod nodes to fully join, then restart the GPU Feature Discovery pods:
```bash
kubectl rollout restart daemonset -n gpu-operator gpu-feature-discovery
```

### b. GPU Operator driver conflict
SageMaker HyperPod nodes have pre-installed NVIDIA drivers. If the GPU Operator tries
to install its own drivers, there will be conflicts.

```bash
# Check if driver pods are crashlooping
kubectl get pods -n gpu-operator -l app=nvidia-driver-daemonset
```

**Fix**: Ensure the GPU Operator was installed with `driver.enabled=false`:
```bash
helm upgrade gpu-operator nvidia/gpu-operator \
    -n gpu-operator \
    --set driver.enabled=false \
    --reuse-values
```

### c. DCGM Exporter crash (profiling error)
The `nvidia-dcgm-exporter` pod may show `Error` or `CrashLoopBackOff` with:
```
Failed to watch metrics: Error watching fields: The third-party Profiling module returned an unrecoverable error
```

**Impact**: This is a **non-blocking** issue. DCGM Exporter provides GPU metrics
(utilization, temperature) but is not required for NVCF function deployment. The
critical components (device plugin, GFD, toolkit) will still work.

**Cause**: The DCGM profiling module is not supported on some GPU types (e.g., A10G)
or when specific driver features are disabled.

**Fix**: Either ignore the error or disable DCGM Exporter:
```bash
helm upgrade gpu-operator nvidia/gpu-operator \
    -n gpu-operator \
    --set dcgmExporter.enabled=false \
    --reuse-values
```

### d. Nodes not labeled for scheduling
If you previously set the `nvca.nvcf.nvidia.io/schedule=true` label on some
nodes but not all, NVCA will only use labeled nodes.

```bash
# Check which nodes have the schedule label
kubectl get nodes -l nvca.nvcf.nvidia.io/schedule=true
```

**Fix**: Label all GPU nodes or remove the label entirely (NVCA defaults
to using all GPU nodes when no labels are set):
```bash
# Option 1: Label all GPU nodes
kubectl get nodes -l nvidia.com/gpu.present=true -o name | \
    xargs -I {} kubectl label {} nvca.nvcf.nvidia.io/schedule=true --overwrite

# Option 2: Remove labels to use default behavior
kubectl get nodes -l nvca.nvcf.nvidia.io/schedule=true -o name | \
    xargs -I {} kubectl label {} nvca.nvcf.nvidia.io/schedule-
```

---

## 5. Function Deployment Fails

**Symptoms**: Function status stays at `DEPLOYING` or moves to `ERROR`.

### a. Missing `instanceType` in deployment API call
**Symptoms**: The NVCF deployment API returns:
```json
{"statusCode": "INVALID_REQUEST", "statusDescription": "Missing required Instance type in deployment spec."}
```

**Cause**: The NVCF deployment API requires an `instanceType` field that matches
the instance types registered by the NVCA operator on your cluster.

**Fix**: Query the cluster group API to find the correct instance type:
```bash
curl -s https://api.ngc.nvidia.com/v2/nvcf/clusterGroups \
  -H "Authorization: Bearer ${NGC_API_KEY}" | python3 -m json.tool
```
Look for your cluster group and use the `instanceTypes[].name` value. For
SageMaker HyperPod with A10G GPUs, the typical value is `AWS.GPU.A10G_1x`.

### b. Insufficient node resources
NVCF requires 6 CPU + 8 GiB per GPU node for infrastructure containers.

```bash
# Check node allocatable resources
kubectl describe nodes | grep -A 5 "Allocatable:"
```

**Fix**: Use `ml.g5.2xlarge` (8 vCPU, 32 GiB) or larger instead of
`ml.g5.xlarge` (4 vCPU, 16 GiB). If you need to change instance types,
you may need to create a new instance group in the SageMaker HyperPod cluster.

### c. Container image pull failure
The function container must be in the NGC Private Registry.

```bash
# Check NVCA backend logs for pull errors
kubectl logs -l app.kubernetes.io/instance=nvca -n nvca-system --tail 50
```

**Fix**: Verify the image exists in NGC:
```bash
ngc registry image list --format_type csv | grep "<your-image-name>"
```

### d. Health check failing
The function container's health endpoint must return HTTP 200.

**Fix**: Test locally before deploying:
```bash
docker run -p 8000:8000 your-image:tag
curl http://localhost:8000/health
# Should return {"status": "OK"} with HTTP 200
```

### e. Port conflict
NVCF reserves ports 8080 and 8010. Your container must not use them.

**Fix**: Change your container's inference port to something else (e.g., 8000).

### f. Docker Desktop SSO / corporate policy blocking builds
**Symptoms**: `docker build` or `docker buildx build` fails with:
```
Error response from daemon: Sign in to continue using Docker Desktop.
Membership in the [org] organization is required.
```

**Cause**: Docker Desktop may enforce corporate SSO sign-in before allowing
any daemon operations.

**Fix**: Use [Finch](https://runfinch.com/) as an alternative container
runtime. The `deploy.sh` script auto-detects Finch when Docker is unavailable:
```bash
# Install Finch (macOS)
brew install --cask finch
finch vm init

# Or use it directly if already installed at /usr/local/bin/finch
./nvcf/sample-function/deploy.sh
```

---

## 6. Network Policy Issues

**Symptoms**: Function pods cannot reach external endpoints or each other.

### a. VPC CNI NetworkPolicy not enabled
AWS VPC CNI doesn't enforce NetworkPolicy by default.

```bash
# Check if network policy agent is running
kubectl get pods -n kube-system -l app.kubernetes.io/name=aws-network-policy-agent
```

**Fix**: Enable it via the VPC CNI add-on configuration:
```bash
aws eks update-addon --cluster-name <eks-cluster-name> \
    --addon-name vpc-cni \
    --configuration-values '{"enableNetworkPolicy": "true"}' \
    --resolve-conflicts OVERWRITE \
    --region ${AWS_REGION}
```

### b. NVCF default policies too restrictive
NVCF installs network policies that block internal traffic to common private
IP ranges. If your VPC CIDR falls in these ranges, adjust the policies.

**Fix**: Apply the custom network policy patch:
```bash
kubectl apply -f nvcf/network-policy-patch.yaml
```

---

## 7. SageMaker HyperPod Deep Health Check Disruptions

**Symptoms**: Running function instances get disrupted during health checks.

SageMaker HyperPod applies `NoSchedule` taint during deep health checks:
```
sagemaker.amazonaws.com/node-health-status=Unschedulable:NoSchedule
```

**Impact**: Existing pods continue running, but new pods cannot be scheduled
on the tainted node. This affects NVCF autoscaling.

**Mitigations**:
1. Use `minInstances > 1` so functions have redundancy during health checks
2. Coordinate health check windows with low-traffic periods
3. Consider disabling deep health checks if NVCF availability is critical:
   ```bash
   # Update the SageMaker HyperPod cluster to disable deep health checks
   aws sagemaker update-cluster \
       --cluster-name <your-cluster-name> \
       --instance-groups '[{"InstanceGroupName":"gpu-workers",...}]'
   ```

---

## 8. Cluster Key Expiration

**Symptoms**: NVCA stops communicating with the NVCF control plane.

```bash
kubectl logs -l app.kubernetes.io/instance=nvca-operator -n nvca-operator --tail 20
# Look for authentication/authorization errors
```

**Fix**: Rotate the key in the NVCF UI and update the Helm installation:
```bash
helm upgrade nvca-operator -n nvca-operator --reuse-values --wait \
    "${NVCA_HELM_URL}" \
    --username='$oauthtoken' \
    --password="${NEW_NGC_CLUSTER_KEY}"
```

---

## 9. Function Logs Not Available

**Expected behavior**: Function-level inference container logs are **not
supported** for BYOC (non-NVIDIA-managed) clusters. This is documented
by NVIDIA.

**Workaround**: Emit logs from your containers directly:

```python
import logging
import json

# Configure structured logging for CloudWatch
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

@app.post("/echo")
async def echo(request: EchoRequest):
    logger.info(json.dumps({"event": "inference", "message": request.message}))
    # ... inference logic
```

Then use a log forwarder (e.g., Fluent Bit) to send logs to CloudWatch:
```bash
# Install Fluent Bit via Helm
helm repo add fluent https://fluent.github.io/helm-charts
helm install fluent-bit fluent/fluent-bit \
    -n logging --create-namespace \
    --set config.outputs="[OUTPUT]\n    Name cloudwatch_logs\n    ..."
```

---

## 10. Autoscaling Not Working as Expected

**Expected behavior**: BYOC clusters only support autoscaling via the
**function queue depth** heuristic. Other utilization-based heuristics
available on NVIDIA-managed clusters are not supported.

**What this means**:
- Scaling up happens when requests queue (max concurrency exceeded)
- Scaling down happens after an idle period with no queued requests
- Set `minInstances=0` to scale to zero when idle (cost savings)
- Set `maxInstances > minInstances` to enable autoscaling

---

## Diagnostic Commands Quick Reference

```bash
# Cluster discovery
./infra/scripts/00-discover-cluster.sh

# Full validation
./infra/scripts/04-validate-setup.sh

# Cluster overview
kubectl get nodes -o wide
kubectl get pods --all-namespaces

# GPU Operator status
kubectl get pods -n gpu-operator
kubectl get nodes -l nvidia.com/gpu.present=true -o custom-columns='NAME:.metadata.name,GPU:.metadata.labels.nvidia\.com/gpu\.product'

# NVCA status
kubectl get nvcfbackend -n nvca-operator
kubectl get pods -n nvca-operator
kubectl get pods -n nvca-system

# NVCA operator logs
kubectl logs -l app.kubernetes.io/instance=nvca-operator -n nvca-operator --tail 50

# NVCA agent logs
kubectl logs -l app.kubernetes.io/instance=nvca -n nvca-system --tail 50

# Function pods (deployed by NVCF)
kubectl get pods -n nvcf-backend
kubectl describe pods -n nvcf-backend

# Network policy check
kubectl get networkpolicies --all-namespaces

# SageMaker HyperPod cluster status
aws sagemaker describe-cluster --cluster-name <your-hyperpod-cluster-name> --region ${AWS_REGION}
```
