# DeepEP Benchmark — Kubernetes (EKS)

Run the DeepEP micro-benchmarks on Amazon EKS as [MPIJobs](https://www.kubeflow.org/docs/components/training/user-guides/mpi/).
These are the Kubernetes equivalents of the [Slurm](../slurm/) `sbatch` scripts.

Each job launches **one MPI process per node** (`slotsPerWorker: 1`); that process spawns 8
local GPU ranks itself via `torch.multiprocessing`. The launcher passes the EFA / NVSHMEM / NCCL
environment (including `NCCL_SOCKET_IFNAME` and `LD_PRELOAD`) through `mpirun -x` because
SSH-launched workers do not inherit the image `ENV`.

## Prerequisites

### EKS cluster with EFA + GPUs

An EKS cluster with EFA-enabled GPU nodes (e.g. `p6-b300.48xlarge`), the
[NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin) and the
[AWS EFA device plugin](https://github.com/aws/eks-charts/tree/master/stable/aws-efa-k8s-device-plugin)
installed. See the [EKS architectures](../../../../1.architectures) in this repo.

```bash
aws eks update-kubeconfig --name <EKS_CLUSTER_NAME>
kubectl config current-context
```

### MPI Operator

The `MPIJob` CRD is provided by the standalone [Kubeflow MPI Operator](https://github.com/kubeflow/mpi-operator)
(not the Training Operator):

```bash
kubectl apply -f https://raw.githubusercontent.com/kubeflow/mpi-operator/v0.6.0/deploy/v2beta1/mpi-operator.yaml
kubectl get crd mpijobs.kubeflow.org      # confirm the CRD exists
```

## Build & push the container image

Build the image from [`../deepep.Dockerfile`](../deepep.Dockerfile) and push it to ECR. The
default image is built for **both Hopper (`sm_90`) and Blackwell (`sm_100`)**, so it runs on
`p5`/`p5en` and `p6-b300` nodes without a rebuild.

```bash
export AWS_REGION=$(aws ec2 describe-availability-zones --output text --query 'AvailabilityZones[0].[RegionName]')
export ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export REGISTRY=${ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com
export IMAGE_URI=${REGISTRY}/deepep:efa1.48.0-nvshmem3.7.0-deepep567632d

aws ecr create-repository --repository-name deepep --region ${AWS_REGION} 2>/dev/null || true
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${REGISTRY}

cd ..   # build context is the deepep-benchmark/ directory
docker build --progress=plain -f ./deepep.Dockerfile -t ${IMAGE_URI} .
docker push ${IMAGE_URI}
```

## Configure & launch

Copy the committed template to `env_vars` (gitignored) and edit it with your image URI, instance
type, and per-node device counts, then `envsubst` the manifest into `kubectl`. Restrict
substitution to the known variables so the launcher's runtime shell vars
(`$OMPI_COMM_WORLD_RANK`, `$MASTER_ADDR`, `$PATH`, …) are left intact:

```bash
cp env_vars.example env_vars   # then edit env_vars
source env_vars

# Single-node intranode (NVLink) test  (intranode is fixed at 1 node; NUM_NODES is unused)
envsubst '$IMAGE_URI $INSTANCE_TYPE $GPU_PER_NODE $EFA_PER_NODE $NUM_NODES' \
  < test-intranode.yaml | kubectl apply -f -

# Two-node internode (RDMA over EFA) test  — set NUM_NODES=2
envsubst '$IMAGE_URI $INSTANCE_TYPE $GPU_PER_NODE $EFA_PER_NODE $NUM_NODES' \
  < test-internode.yaml | kubectl apply -f -

# Two-node low-latency (decode-path) test  — set NUM_NODES=2
envsubst '$IMAGE_URI $INSTANCE_TYPE $GPU_PER_NODE $EFA_PER_NODE $NUM_NODES' \
  < test-low-latency.yaml | kubectl apply -f -
```

## Monitor

```bash
kubectl get mpijob
watch kubectl get pods -o wide
kubectl logs -f $(kubectl get pods | grep deepep.*launcher | cut -d ' ' -f 1)
```

## Clean up

MPIJob names are fixed, so delete a run before re-applying it:

```bash
envsubst '$IMAGE_URI $INSTANCE_TYPE $GPU_PER_NODE $EFA_PER_NODE $NUM_NODES' \
  < test-internode.yaml | kubectl delete -f -
```

## Notes for `p6-b300.48xlarge` (B300 / Blackwell)

- **Image:** the default image already includes `sm_100`, so no rebuild is needed.
- **EFA count:** `EFA_PER_NODE=16` (a `p6-b300.48xlarge` exposes 16 EFA NICs). `p5.48xlarge`
  exposes 32 — adjust per instance type.
- **Tolerations:** the manifests tolerate `nvidia.com/gpu`, `workload=bench`, and
  `capacity-reservation` taints. If your nodes use different taints, edit the `tolerations`
  block; if they are untainted, the extra tolerations are harmless.
- **Multi-node DNS:** the launcher sets `MASTER_ADDR` from the first entry of the
  mpi-operator hostfile (`/etc/mpi/hostfile`), which is worker-0's cluster-resolvable name. If
  `init_process_group` hangs, confirm name resolution from a peer:
  `kubectl exec deepep-internode-worker-1 -- getent hosts <worker-0-name>`.
- **GPU profiling (internode / low-latency tuning):** the `internode` and `low_latency` tests
  profile their kernels with the Kineto/CUPTI profiler during the tuning phase (the `intranode`
  test does not). Two requirements, both already handled here for `p6-b300`:
  1. *Driver profiling access.* These nodes load the driver with
     `NVreg_RestrictProfilingToAdminUsers=1` (`grep RmProfiling /proc/driver/nvidia/params`), so
     the test pods are granted `SYS_ADMIN` (see the worker `securityContext` in the internode /
     low-latency manifests). Remove it if your nodes already permit non-admin profiling.
  2. *A CUPTI that supports the GPU.* The image is built on **CUDA 13** — its CUPTI supports
     Blackwell. CUDA ≤ 12.9 returns `CUPTI_ERROR_INVALID_DEVICE` on B300 and the tuning phase
     fails after the correctness checks pass, so do not downgrade the image's CUDA for B300.
