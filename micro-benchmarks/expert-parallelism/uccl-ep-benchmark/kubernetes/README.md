# UCCL-EP Benchmark — Kubernetes (EKS)

Run the UCCL expert-parallelism micro-benchmarks on Amazon EKS as [MPIJobs](https://www.kubeflow.org/docs/components/training/user-guides/mpi/).
These are the Kubernetes equivalents of the [Slurm](../) `sbatch` scripts and mirror the
[DeepEP benchmark's Kubernetes layout](../../deepep-benchmark/kubernetes/).

Each job launches **one MPI rank per GPU** (`slotsPerWorker: 8`): the UCCL bench runs one
process per GPU and initializes `torch.distributed` directly, so MPI (not `torchrun`) does the
spawning. The launcher maps the OMPI per-rank variables to the standard `torch.distributed`
env contract — `RANK=$OMPI_COMM_WORLD_RANK`, `WORLD_SIZE=$OMPI_COMM_WORLD_SIZE`,
`LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK`, `LOCAL_WORLD_SIZE=$OMPI_COMM_WORLD_LOCAL_SIZE`, plus
`MASTER_ADDR`/`MASTER_PORT` — and passes the EFA / NCCL environment through `mpirun -x` because
SSH-launched workers do not inherit the image `ENV`. (The DeepEP benchmark uses
`slotsPerWorker: 1` because its test spawns its 8 local ranks itself with
`torch.multiprocessing`; UCCL's does not, hence 8 slots here.)

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

The `MPIJob` CRD is provided by the standalone [Kubeflow MPI Operator](https://github.com/kubeflow/mpi-operator):

```bash
kubectl apply -f https://raw.githubusercontent.com/kubeflow/mpi-operator/v0.6.0/deploy/v2beta1/mpi-operator.yaml
kubectl get crd mpijobs.kubeflow.org      # confirm the CRD exists
```

## Build & push the container image

Build the image from [`../uccl-ep.Dockerfile`](../uccl-ep.Dockerfile) and push it to ECR. The
image pins a UCCL commit (`UCCL_COMMIT` build arg) for reproducibility and builds the UCCL-EP
kernels for **both Hopper (`sm_90`) and Blackwell (`sm_100`/`sm_103`)** via an explicit
`TORCH_CUDA_ARCH_LIST` with PTX fallback (the `python3 setup.py install` path, following the
validated dsv3-uccl-nixl recipe), so one image runs on `p5`/`p5en` and `p6-b300` without a
rebuild.

```bash
export AWS_REGION=$(aws ec2 describe-availability-zones --output text --query 'AvailabilityZones[0].[RegionName]')
export ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export REGISTRY=${ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com
export IMAGE_URI=${REGISTRY}/uccl-ep:efa1.48.0-uccl0dc87eb-cu13

aws ecr create-repository --repository-name uccl-ep --region ${AWS_REGION} 2>/dev/null || true
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${REGISTRY}

cd ..   # build context is the uccl-ep-benchmark/ directory
docker build --progress=plain -f ./uccl-ep.Dockerfile -t ${IMAGE_URI} .
docker push ${IMAGE_URI}
```

## Configure & launch

Copy the committed template to `env_vars` (gitignored) and edit it with your image URI,
instance type, and per-node device counts, then `envsubst` the manifest into `kubectl`.
Set `NP` = `NUM_NODES * GPU_PER_NODE` (the total rank count). Restrict substitution to the
known variables so the launcher's runtime shell vars (`$OMPI_COMM_WORLD_RANK`, `$MASTER_ADDR`,
`$PATH`, …) are left intact:

```bash
cp env_vars.example env_vars   # then edit env_vars
source env_vars

# Single-node intranode (NVLink) test  — set NUM_NODES=1, NP=GPU_PER_NODE
envsubst '$IMAGE_URI $INSTANCE_TYPE $GPU_PER_NODE $EFA_PER_NODE $NUM_NODES $NP' \
  < test-intranode.yaml | kubectl apply -f -

# Internode (RDMA over EFA) test  — e.g. NUM_NODES=8, NP=64
envsubst '$IMAGE_URI $INSTANCE_TYPE $GPU_PER_NODE $EFA_PER_NODE $NUM_NODES $NP' \
  < test-internode.yaml | kubectl apply -f -

# Low-latency (decode-path) test  — e.g. NUM_NODES=8, NP=64
envsubst '$IMAGE_URI $INSTANCE_TYPE $GPU_PER_NODE $EFA_PER_NODE $NUM_NODES $NP' \
  < test-low-latency.yaml | kubectl apply -f -
```

### EP problem size (matched for backend comparison)

The EP shape is baked into each manifest's `python3 bench/test_*.py` invocation to a
DeepSeek-V3-class configuration: `--num-tokens` 4096 (128 for low-latency), `--hidden 7168`, `--num-topk 8`,
`--num-experts 256`. `num-experts=256` divides evenly across 64 ranks (8 nodes × 8 GPU) and
**matches the DeepEP NVSHMEM benchmark**, so the dispatch/combine bandwidths are comparable.
The upstream Slurm scripts use `--num-experts=288` for internode/low-latency; that is not
divisible by 64, so the Kubernetes manifests override it to 256. If you compare against a
DeepEP image whose hard-coded config differs, edit these args to match.

## Monitor

```bash
kubectl get mpijob
watch kubectl get pods -o wide
kubectl logs -f $(kubectl get pods | grep uccl-ep.*launcher | cut -d ' ' -f 1)
```

## Clean up

MPIJob names are fixed, so delete a run before re-applying it:

```bash
envsubst '$IMAGE_URI $INSTANCE_TYPE $GPU_PER_NODE $EFA_PER_NODE $NUM_NODES $NP' \
  < test-internode.yaml | kubectl delete -f -
```

## Notes for `p6-b300.48xlarge` (B300 / Blackwell)

- **Image:** the default image is built for `sm_90` + `sm_100`/`sm_103` (PTX fallback), so it
  runs on `p5`/`p5en` (Hopper) and `p6-b300` (Blackwell) without a rebuild.
- **EFA count:** `EFA_PER_NODE=16` (a `p6-b300.48xlarge` exposes 16 EFA NICs). `p5.48xlarge`
  exposes 32 — adjust per instance type.
- **Tolerations:** the manifests tolerate the `nvidia.com/gpu` and `capacity-reservation`
  taints. If your GPU nodes carry additional taints, add matching tolerations to the
  `tolerations` block; if they are untainted, the existing ones are harmless.
- **GPU profiling (internode / low-latency tuning):** the `internode` and `low_latency` tests
  profile their kernels during the tuning phase (the `intranode` test does not). The worker
  `securityContext` grants `SYS_ADMIN` because these nodes load the driver with
  `NVreg_RestrictProfilingToAdminUsers=1` (`grep RmProfiling /proc/driver/nvidia/params`).
  Remove it if your nodes already permit non-admin profiling.
