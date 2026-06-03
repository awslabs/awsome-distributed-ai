# Profile Distributed Training Applications with Nsight 

[Nsight Systems](https://developer.nvidia.com/nsight-systems) is a system-wide performance analysis tool designed to profile and visualize multi-node CPU and GPU workloads such as distributed training and inference to identify the largest opportunities to optimize, and tune to scale efficiently across the cluster. It also enables researchers to add their own markers into their code to surface application-level metrics into the profiler and gain further observability.

We will show how to profile and analyze:

1. [NCCL Tests](https://github.com/awslabs/awsome-distributed-training/tree/main/micro-benchmarks/nccl-tests/slurm)
2. [Distributed training run with NeMo](https://github.com/awslabs/awsome-distributed-training/tree/main/examples/2.nemo-launcher)
3. [Distributed training run with FSDP](https://github.com/awslabs/awsome-distributed-training/tree/main/examples/10.FSDP)
4. Setup Nsight on an EKS cluster

# 0. Prerequisities

1. A cluster created with P4de or P5 nodes with AWS ParallelCluster or EKS
2. Before profiling the above workloads, make sure you can run them on your cluster.
3. For EKS, we will be using a 2 node P4de cluster with EFA enabled and FSx for Lustre mounted on the cluster

# 1. Export Environment Variables

Export the following variables to setup the profiling:

```bash
export Nsight_version=2024.4.1 # Nsight Version
export Nsight_download_url=https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_4/NsightSystems-linux-cli-public-2024.4.1.61-3431596.deb
export Nsight_cli_installer=$(basename "$Nsight_download_url")
export Nsight_Path=/fsx/nsight-efa
export Nsight_Report_Path=/fsx/nsight-reports
mkdir -p ${Nsight_Report_Path}
```

# 2. Installation

If you created the cluster with DLAMI or are using the default ParallelCluster base image, Nsight comes pre-installed. You can check the version in the `/usr/local/cuda/` folder you should see `nsight-systems-202x.x.x` folder. ParallelCluster 3.8.0 has the version 2023.2 version pre-installed. 

To get the latest Nsight 2024.4 version from [here](https://developer.nvidia.com/nsight-systems/get-started). If you are installing it on a remote cluster, then the CLI version would suffice. To install it on a Ubuntu based OS node:

```bash
# Download Nsight CLI
wget ${Nsight_download_url}

# Install
sudo dpkg -i ${Nsight_cli_installer}

# This would place the nsys binay at /opt/nvidia/nsight-systems-cli/2024.3.1/target-linux-x64/nsys
# Move to FSx filesystem
mkdir -p ${Nsight_Path}
cp -r /opt/nvidia/nsight-systems-cli/${Nsight_version}/* ${Nsight_Path}
```

The `nsight-efa`folder will have the necessary dependencies for the `host` which is the head node in a Slurm cluster from which the user works and controls the profiling session and `target` which refers to the GPU on which profiling happens. This latest version also has the `nic_sampler` in `/nsight-efa/target-linux-x64/plugins/` which collects the EFA metrics.

# 3. How to generate reports

Nsight 2024.4 version supports EFA metrics. In this section we will walkthrough how to generate reports with EFA metrics on a Slurm cluster. 

## 3.1 Modify slurm submission script

For a containerized distributed training run, the srun command in the slurm submission script looks like:

```bash
srun -u -l --container-image <enroot-image.sqsh> --container-mounts /fsx:/fsx,<other-mounts> <training-cli> <training-args>
```

In the above schema <training-cli> could be `torchrun`, `python3`, `mpirun` etc

We need to inject a `nsys-slurm-exec` executable like below assuming `nsys-slurm-exec` resides in `/fsx`:

```bash
srun -u -l --container-image <enroot-image.sqsh> --container-mounts /fsx:/fsx,<other-mounts> /fsx/nccl-slurm-exec <training-cli> <training-args>
```

## 3.2 nsys-slurm-exec

Here is a template for `nsys-slurm-exec`:

```bash
#! /bin/bash -x

NSYS_EXTRAS=""
if [ "$SLURM_LOCALID" == "0" ]; then
        NSYS_EXTRAS="--enable efa_metrics"
fi

${Nsight_Path}/target-linux-x64/nsys profile $NSYS_EXTRAS --sample none --output ${Nsight_Report_Path}/profile_%q{SLURM_JOB_ID}_node_%q{SLURM_NODEID}_rank_%q{SLURM_PROCID}_on_%q{HOSTNAME}.nsys-rep --force-overwrite true \
   "$@"
```

A few key points:
1. This slurm executable will generate 1 report for each GPU if SLURM_NTASKS_PER_NODE is equal to the number of GPUs. If SLURM_NTASKS_PER_NODE=1, one report will get generated for all 8 GPUs.
2. --sample none argument disables CPU sampling. For a detailed list of CLI switches see [here](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cli-command-switches)
3. EFA metrics are shared across all GPUs on the same node so it needs to be enabled only on 1 GPU

> [!TIP]
> To include any SLURM environment variables in the report name, you can include them with %q{SLURM_ENV_VAR}

> [!TIP]
> Make sure to run chmod 777 /fsx/nsys-slurm-exec

## 3.3 Tailoring the report with --duration and --delay

You can control the training period where you want the report to focus on with `--duration` and `--delay` parameters in seconds. The `--delay` parameter specifies when to start metrics collection from all kernels and `--duration` parameter specifies how long to collect data. These variables are typically collected in an ad-hoc manner.

## 3.4 Controlling number of reports with SLURM environment variables

When running training with 100's of nodes, it is not often desirable to generate a report for each node let along each GPU. You can control it as follows:

```bash
#! /bin/bash -x

NSYS_EXTRAS=""
if [ "$SLURM_LOCALID" == "0" ]; then
        NSYS_EXTRAS="--enable efa_metrics"
fi

if [ "$SLURM_PROCID" == "0" ]; then
        ${Nsight_Path}/target-linux-x64/nsys profile $NSYS_EXTRAS --sample none --delay 330 --duration 50 -o ${Nsight_Report_Path}/profile_%q{SLURM_JOB_ID}_node_%q{SLURM_NODEID}_rank_%q{SLURM_PROCID}_on_%q{HOSTNAME}.nsys-rep --force-overwrite true \
   "$@"
else
        "$@"
fi
```

## 3.5 Controlling the report with start and stop profiling calls

We need a more convinient way to generate reports with start and stop training step user inputs. You can:


1. Add `nsys_start_step` and `nsys_end_step` as input arguments to your train.py
2. Add the following in the training loop to start collecting data from Cuda and OSRT traces:
```python
if batch_idx == args.nsys_start_step and global_rank == 0:
    logger.info("====== Start nsys profiling ======")
    torch.cuda.cudart().cudaProfilerStart()
```
3. Add to stop collection:
```python
if batch_idx == args.nsys_end_step and global_rank == 0:
    logger.info("====== Stop nsys profiling ======")
    torch.cuda.cudart().cudaProfilerStop()
```
4. Add `--capture-range=cudaProfilerApi --capture-range-end=stop` to the `nsys profile ...` command.




# 4. Profiling NCCL tests
In this section we will show how to generate Nsight reports for NCCL tests. Follow the instructions [here](https://github.com/awslabs/awsome-distributed-training/tree/main/validation_and_observability/0.nccl-tests) to setup NCCL tests and generate the Enroot image `nccl.sqsh`. The `0.nsight_nccl.sbatch` script shows an example on how to profile the NCCL run with Nsight and collect EFA metrics. Key differences between `0.nsight_nccl.sbatch` and [this](https://github.com/awslabs/awsome-distributed-training/blob/main/validation_and_observability/0.nccl-tests/1.nccl-tests.sbatch) are:

1. `/fsx` needs to be mounted to the container as this is where our Nsight binaries are located.
2. The `0.nsight_nccl.sbatch` script references the executable `nsys-slurm-exec` which is given below and should exist in `/fsx`

```bash
#! /bin/bash -x

NSYS_EXTRAS=""
if [ "$SLURM_LOCALID" == "0" ]; then
NSYS_EXTRAS="--enable efa_metrics"
fi

/fsx/nsight-efa/target-linux-x64/nsys profile $NSYS_EXTRAS --sample none --delay <DELAY-PERIOD> \
    --force-overwrite true --output <PATH-TO-SAVE-REPORT>/report_<REPORT-NAME-TAG>_job%q{SLURM_JOB_ID}_rank%q{SLURM_PROCID}_on_%q{HOSTNAME}.nsys-rep \
   "$@"
```
The above executable needs the following:

```bash
1. DELAY-PERIOD: Collection start delay in seconds. Typically the multi-node workload takes a few seconds before collection of relevant metrics start. Typically for distributed training applications delaying by ~30sec avoids having empty gaps in the timeline view of the Nsight report. For the NCCL test a delay of less than 5 seconds works. You can also specify --duration in seconds to collect metrics.

2. PATH-TO-SAVE-REPORT: One report is generated per GPU. Provide a path to save all reports.
3. REPORT-NAME-TAG: Unique name tag to group all reports. Use %q{} to include environment variables in report names.
```
Here, we are running the Nsight profile with 2 p4de nodes where each node has 4 EFA devices and 8 GPUs. The `nic sampler` metrics from all 4 EFA devices show up in every report so it is okay to collect these metrics only for 1 rank.

Below is a screenshot of the generated Nsight report:

<center><img src="nccl/NCCL_Scatter_Perf.png" width="80%"/> </br>
</center>

Here there are the following things to note:

•   The RDMA read bytes per second shown in green are from the EFA NIC samplers. You can see there are 4 `rdma*` rows in the report, one corresponding to each of the EFA devices one 1 node. For a P5.48xlarge node, you will see 32 rows.
•   This report is generated for the [Scatter Performance NCCL test](https://github.com/NVIDIA/nccl-tests/blob/2cbb968101e2bfc7d3a7f0f1826c0189355de6fe/src/scatter.cu#L34), which essentially calls the ncclSendRecv kernels again and again which is why ncclDevKernel_SendRecv takes 99.3% utilization among all kernels.
•   You can right click on any row to see the meta-data over time in the Events View which shows start times, durations and other meta-data for each kernel

> [!TIP]
> The *.qdstrm files are temporarily generated first using the nsys binaries in `.../target-linux-x64` while the `*.nsys-rep` report file is generated using the `/host-linux-x64/QdstrmImporter` binary. If for some reason, only `*.qdstrm` files are generated, use the above importer like below to generate a `*.nsys-rep report`
```bash
<Path-to-host-linux-x64>/host-linux-x64/QdstrmImporter –input-file <file-name>.qdstrm
```


## 4.1 NCCL All Reduce Test

Following the steps above, you can generate a similar result for NCCL All Reduce Test also see NCCL test output in the logs. Here we will visualize the spread in NCCL All Reduce communication for 1GB and 2GB message sizes. To do so you can:
1. Run NCCL test and generate report. Save the result for 1GB and 2GB message sizes.
2. Right click on `all_reduce_perf > NCCL` row to show in Events View. This Events View shows NCCL Kernel API calls on the CPU. You can see the NCCL Message Size for each call. Note row numbers where NCCL Message Sizes change.
3. Right click on `ncclDevKernel_AllReduce_Sum_f32_TREE_LL(ncclDevComm *, unsigned long, ncclWork *)` row and show in Events View. This Events View shows NCCL Kernel calls executed on the GPU, it start time and duration. Copy paste the entire table in a csv.
4. You should see 1-on-1 correlation between 3 and 4. Meaning for each NCCL call on the CPU there is a call executed on the GPU. Or in other words, the number of rows in Events View from 3 and 4 should exactly be the same.
5. Add NCCL Message Sizes from Step 3 to csv from Step 4. Save the csv as `all_reduce.csv` which should look like below:

<center><img src="nccl/all_reduce_csv_screenshot.png" width="80%"/> </br>
</center>

You can generate the plot below using the python script `/nccl/plot_nccl.py`

<center><img src="nccl/all_reduce_sum.png" width="80%"/> </br>
</center>


# 5. Working with the report

1. The Nsight Systems GUI offers to view the following. You can see these options by clicking on the Timeline View menu button.
a.  Output and error logfiles from the training run
b.  Analysis summary that gives a summary of the profiling session.
c.  Timeline view of the report
d.  Diagnostics summary view
2.  You can right click and pin any row at the top. This helps in analyzing multiple rows simultaneously.
3.  You can view start and execution times of any kernel by viewing them in the Events view.
4.  From the Events View, you can zoom to that specific kernel event by right clicking. This provides an easy way to look into kernel events preceding and following a specific kernel even if their durations are in nanoseconds.
5.  You can export the report in different formats such as sqllite and others as well for custom analysis.

# 6. Nsight Recipes

Once the report is generated, we can generate [recipes](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#available-multi-report-recipes) to analyze the data in the report. We provide the script ` 2.generate_recipes.sh` which will generate multiple recipes for the report and upload to S3. Each recipe run will summarize the relevant data from the report and provide python scripts and jupyter notebooks to analyze the data.

Next, we will show what kind of analysis can be generated from the recipes.

To install requirements to generate recipes:

```bash
pip3 install -r ${Nsight_Path}/target-linux-x64/python/packages/nsys_recipe/requirements/common.txt
pip3 install -r ${Nsight_Path}/target-linux-x64/python/packages/nsys_recipe/requirements/dask.txt

```
With Nsight 2024.4, the following recipes are available:

```bash
The following built-in recipes are available:

  cuda_api_sum -- CUDA API Summary
  cuda_api_sync -- CUDA Synchronization APIs
  cuda_gpu_kern_pace -- CUDA GPU Kernel Pacing
  cuda_gpu_kern_sum -- CUDA GPU Kernel Summary
  cuda_gpu_mem_size_sum -- CUDA GPU MemOps Summary (by Size)
  cuda_gpu_mem_time_sum -- CUDA GPU MemOps Summary (by Time)
  cuda_gpu_time_util_map -- CUDA GPU Time Utilization Heatmap
  cuda_memcpy_async -- CUDA Async Memcpy with Pageable Memory
  cuda_memcpy_sync -- CUDA Synchronous Memcpy
  cuda_memset_sync -- CUDA Synchronous Memset
  diff -- Statistics Diff
  dx12_mem_ops -- DX12 Memory Operations
  gpu_gaps -- GPU Gaps
  gpu_metric_util_map -- GPU Metric Utilization Heatmap
  gpu_time_util -- GPU Time Utilization
  mpi_gpu_time_util_map -- MPI and GPU Time Utilization Heatmap
  mpi_sum -- MPI Summary
  nccl_gpu_overlap_trace -- NCCL GPU Overlap Trace
  nccl_gpu_proj_sum -- NCCL GPU Projection Summary
  nccl_gpu_time_util_map -- NCCL GPU Time Utilization Heatmap
  nccl_sum -- NCCL Summary
  network_traffic_map -- Network Devices Traffic Heatmap
  nvtx_gpu_proj_pace -- NVTX GPU Projection Pacing
  nvtx_gpu_proj_sum -- NVTX GPU Projection Summary
  nvtx_gpu_proj_trace -- NVTX GPU Projection Trace
  nvtx_pace -- NVTX Pacing
  nvtx_sum -- NVTX Range Summary
  osrt_sum -- OS Runtime Summary
  ucx_gpu_time_util_map -- UCX and GPU Time Utilization Heatmap
```

Please see the `2.generate_recipes.sh` to generate multiple recipes from a given report as below:

```bash
# Do not include nsys-rep extension
export NSIGHT_REPORT_NAME=
./2.generate_recipes.sh
```

# 7. Nsight on EKS

We will use the [Nvidia Devtools Sidecar Injector](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/devtools/helm-charts/devtools-sidecar-injector) to profile containerized applications.

Pull the Nsight docker image - This step will not be needed once the 2024.4 version is released.

```bash
docker pull nvcr.io/nvstaging/devtools/nsight-systems-cli:2024.4.1-ubuntu22.04

# Push image to ECR
```

## 7.1 Make changes to the `custom_values.yaml`

```bash
# If we dont specify the Nsight image, 2024.2 version is used by default.
# Will use 2024.4 version which is planned to be released by 5/24/2024
devtoolBinariesImage:
  image: ${REGISTRY}.dkr.ecr.${REGION}.amazonaws.com/nsight-systems-cli:2024.4.1-ubuntu22.04
  imagePullPolicy: Always

# Assuming EKS cluster has a FSx for Lustre filesystem mounted on it. Nsight reports will be saved in /fsx_shared
profile:
  volumes:
    [
      {
        "name": "nsys-output-volume",
        "persistentVolumeClaim": { "claimName": "fsx-pvc" }
      }
    ]
  volumeMounts:
    [
      {
        "name": "nsys-output-volume",
        "mountPath": "/fsx_shared"
      }
    ]

  # CLI options: https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cli-command-switches
  # delay and duration values in secs

  # Use %{} to include environment variables in the Nsight report filename

  # The arguments for the Nsight Systems. The placeholders will be replaced with the actual values.
  devtoolArgs: "profile --force-overwrite true --trace nvtx,cuda  --delay 150 --duration 60 \
  -o /fsx_shared/fsdp/auto_{PROCESS_NAME}_%{POD_FULLNAME}_%{CONTAINER_NAME}_{TIMESTAMP}_{UID}.nsys-rep"

  injectionMatch: "^/usr/bin/python3 /usr/local/bin/torchrun.*$"
  #injectionMatch: "^.*torchrun.*$"
```

## 7.2 Install injector

Install helm chart for the sidecar injector as below:
```bash
helm install -f custom_values.yaml \
    devtools-sidecar-injector https://helm.ngc.nvidia.com/nvidia/devtools/charts/devtools-sidecar-injector-1.0.0.tgz
```

## 7.3 Add label to training job manifest

Add the following label:

```bash
pytorchReplicaSpecs:
    Worker:
      replicas: 2
      restartPolicy: OnFailure
      template:
        metadata:
          labels:
            app: fsdp
            nvidia-devtools-sidecar-injector: enabled
```

## 7.4 Run Training job

Run the training job as:

```bash
kubectl apply -f fsdp.yaml
```
The report will get saved to `/fsx_shared`

Below is a screenshot of the generated Nsight report:

<center><img src="EKS/fsdp_eks_report_screenshot.png" width="80%"/> </br>
</center>

## 7.5 Uninstall injector

To uninstall injector:

```bash
helm uninstall devtools-sidecar-injector

kubectl delete namespace nvidia-devtools-sidecar-injector

kubectl delete mutatingwebhookconfigurations sidecar-injector-webhook
kubectl delete mutatingwebhookconfiguration nvidia-devtools-sidecar-injector-webhook

kubectl delete cm -n example-ns nvidia-devtools-sidecar-injector
kubectl delete cm -n example-ns nvidia-devtools-sidecar-injector-custom

kubectl delete cm nvidia-devtools-sidecar-injector
kubectl delete cm nvidia-devtools-sidecar-injector-custom
```

# 8. Nsight on EKS with Host-Mount Approach (HyperPod / DLAMI)

The sidecar injector approach in Section 7 requires pulling an Nsight Docker image and installing a Helm chart. An alternative, simpler approach is available when nsys is pre-installed on the cluster nodes (as it is on **SageMaker HyperPod EKS** nodes and **DLAMI-based** EKS nodes). This approach mounts the host's nsight-systems directory directly into the training pods.

**Quick start:**
1. Verify nsys is on nodes: `ls /opt/nvidia/nsight-systems/`
2. Create ConfigMap: `kubectl create configmap nsight-scripts --from-file=nsys-profile.sh=EKS/nsys-profile.sh`
3. Apply the profiled job: `kubectl apply -f EKS/llama3_2_1b-fsdp-nsight.yaml`
4. Copy reports and run analysis: `python3 EKS/nsys_analyze.py --reports /tmp/nsight-reports/`

**Prerequisites:**
- An EKS cluster with GPU nodes where nsys is pre-installed (HyperPod EKS, or DLAMI-based nodes)
- Kubeflow Training Operator installed (for PyTorchJob CRD)
- nsys >= 2024.5 for `--pytorch=autograd-shapes-nvtx` support (HyperPod ships 2025.6.1)

**Advantages over the sidecar approach:**
- No Helm chart, no webhook, no sidecar image to manage
- Zero overhead for non-profiled ranks (selective profiling)
- Auto-detects nsys version on the host
- Modern nsys features: PyTorch NVTX annotations, Python call stack sampling, CUDA memory tracking
- Generates stats and SQLite export automatically

## 8.1 Verify nsys is available on your nodes

On HyperPod EKS nodes, nsys is pre-installed:

```bash
# Check from a pod on the node:
ls /opt/nvidia/nsight-systems/
# Should show a version directory like 2025.6.1/
```

## 8.2 Deploy the profiling wrapper script

The `EKS/nsys-profile.sh` script wraps your training command with `nsys profile`. Deploy it as a ConfigMap:

```bash
kubectl create configmap nsight-scripts \
  --from-file=nsys-profile.sh=EKS/nsys-profile.sh
```

## 8.3 Configure and run a profiled training job

A reference PyTorchJob manifest is provided at `EKS/llama3_2_1b-fsdp-nsight.yaml`. The key additions compared to a standard training job are:

1. **Volumes**: Mount nsight from host and the ConfigMap script

```yaml
volumes:
  - name: nsight
    hostPath:
      path: /opt/nvidia/nsight-systems
      type: Directory
  - name: scripts
    configMap:
      name: nsight-scripts
      defaultMode: 0755
```

2. **Environment variables** to configure profiling:

```yaml
env:
  - name: NSYS_DELAY
    value: "30"            # Skip startup warmup
  - name: NSYS_DURATION
    value: "120"           # Capture 120s of steady-state training
  - name: NSYS_RANKS_TO_PROFILE
    value: "0"             # Only profile rank 0 (set "all" for small scale)
  - name: NSYS_PYTORCH_MODE
    value: "autograd-shapes-nvtx"  # Auto-annotate PyTorch ops
  - name: NSYS_PYTHON_SAMPLE
    value: "true"          # Sample Python call stacks at 1kHz
  - name: NSYS_GPU_METRICS
    value: "none"          # "all" for A100/H100/H200, "none" for A10G (g5)
  - name: NSYS_CUDA_MEMORY
    value: "true"          # Track CUDA memory allocations
```

3. **Command**: Use the wrapper script instead of calling torchrun directly

```yaml
command:
  - /bin/bash
  - /scripts/nsys-profile.sh
  - --
  - /usr/local/bin/torchrun
  - --nproc_per_node=8
  - --nnodes=2
  - /fsdp/train.py
  - ...training args...
```

4. **Volume mounts**: Include nsight and scripts

```yaml
volumeMounts:
  - name: nsight
    mountPath: /nsight
    readOnly: true
  - name: scripts
    mountPath: /scripts
```

Run the job:

```bash
# Edit the manifest to set your image, node count, and GPU count, then:
kubectl apply -f EKS/llama3_2_1b-fsdp-nsight.yaml
```

The profiling wrapper will:
- Auto-detect the nsys binary from the mounted volume
- Skip profiling for ranks not in `NSYS_RANKS_TO_PROFILE` (zero overhead)
- Collect with `--delay` and `--duration` to capture steady-state training
- Use `--kill=none` so training continues after the profiling window ends
- Auto-generate `.nsys-rep`, `.sqlite`, and summary stats

## 8.4 Copy reports from completed pods

If the pods are still running:

```bash
kubectl cp <pod-name>:/local/nsight-reports/ /tmp/nsight-reports/
```

If the pods have completed, create a helper pod with the hostPath mounted:

```bash
# First, find the node name where the pod ran:
kubectl get pods -o wide | grep llama3-2-1b-fsdp-nsight

# Then create a helper pod on that node:
kubectl run nsys-copy --image=busybox --restart=Never --overrides='
{
  "spec": {
    "nodeSelector": {"kubernetes.io/hostname": "<node-name>"},
    "volumes": [{"name": "local", "hostPath": {"path": "/mnt/k8s-disks/0"}}],
    "containers": [{"name": "copy", "image": "busybox", "command": ["sleep", "300"],
      "volumeMounts": [{"name": "local", "mountPath": "/local"}]}]
  }
}'
kubectl cp nsys-copy:/local/nsight-reports/ /tmp/nsight-reports/
kubectl delete pod nsys-copy
```

## 8.5 Automated bottleneck analysis

The `EKS/nsys_analyze.py` script parses `.nsys-rep` files and generates a structured bottleneck report. It classifies GPU kernels into categories (NCCL, GEMM, Flash Attention, etc.), identifies the dominant bottleneck type, and provides actionable recommendations.

Run analysis on-cluster (where nsys is available) or locally (if nsys is installed):

```bash
# Analyze a directory of reports
python3 EKS/nsys_analyze.py --reports /tmp/nsight-reports/ --output /tmp/analysis.md

# JSON output for programmatic use
python3 EKS/nsys_analyze.py --reports /tmp/nsight-reports/ --format json --output /tmp/analysis.json

# Specify nsys binary explicitly (version depends on your node)
python3 EKS/nsys_analyze.py --reports /tmp/nsight-reports/ \
  --nsys-bin /opt/nvidia/nsight-systems/<version>/target-linux-x64/nsys
```

The analysis output includes:
- **GPU Kernel Time Breakdown** by category (NCCL, GEMM, Attention, Optimizer, etc.)
- **Bottleneck Classification** (Communication Bound, Compute Bound, Sync Bound, etc.)
- **CUDA API Time** (identifies synchronization overhead)
- **Memory Transfer Summary** (detects activation offloading overhead)
- **PyTorch Operation Breakdown** (from NVTX annotations)
- **Cross-Worker Comparison** (when analyzing multiple reports)
- **Recommendations** (e.g., enable EFA, disable activation offloading, increase batch size)

## 8.6 Key notes and gotchas

1. **`--gpu-metrics-devices` is NOT supported on A10G (g5 instances)** — Requires elevated privileges (`ERR_NVGPUCTRPERM`). Set `NSYS_GPU_METRICS=none`. Works on A100/H100/H200.

2. **`--kill=none` is critical** — Without it, nsys sends SIGTERM to the training process when `--duration` expires. With `--kill=none`, training continues and the report is written when the profiling window ends.

3. **`--pytorch=autograd-shapes-nvtx`** requires nsys >= 2024.5. HyperPod nodes with nsys 2025.6.1 support it.

4. **Report sizes** — With PyTorch NVTX + Python sampling + CUDA memory tracking, expect ~80-90 MB per rank per 120s window. Plan storage accordingly.

5. **Selective profiling** — For runs with > 8 GPUs, set `NSYS_RANKS_TO_PROFILE=0` to profile only rank 0. Non-profiled ranks run the training command directly with zero overhead.

6. **`restartPolicy: Never`** is recommended for profiling jobs to avoid restart loops.

7. **EFA for production profiling** — The reference manifest disables EFA (for g5/A10G testing). For production profiling on P4/P5 instances, uncomment the EFA environment variables in the manifest and ensure the container image includes the [OFI-NCCL plugin](https://github.com/aws/aws-ofi-nccl). Without EFA, NCCL falls back to TCP, which will show as communication-bound in the profiling results. Use `NCCL_TUNER_PLUGIN` to auto-select optimal NCCL protocols for your instance type.

## 8.7 Running recipes for deeper analysis

After copying reports locally, you can run nsys recipes for specialized analysis.
These require nsys to be available (run on-cluster via `kubectl exec`, or locally if nsys is installed):

```bash
# Set NSYS to the version available on your node, e.g.:
NSYS=$(ls -d /opt/nvidia/nsight-systems/*/target-linux-x64/nsys | sort -rV | head -1)
REPORT=/tmp/nsight-reports/report_rank0_hostname_20250301_120000.nsys-rep

# NCCL communication summary
$NSYS recipe nccl_sum --input $REPORT

# NCCL + GPU compute overlap (are comms hidden behind compute?)
$NSYS recipe nccl_gpu_overlap_trace --input $REPORT

# GPU idle gaps (where is the GPU waiting?)
$NSYS recipe gpu_gaps --input $REPORT

# CUDA kernel pacing (identifies stragglers)
$NSYS recipe cuda_gpu_kern_pace --input $REPORT

# GPU time utilization heatmap
$NSYS recipe cuda_gpu_time_util_map --input $REPORT
```
