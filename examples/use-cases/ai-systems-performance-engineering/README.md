# AI systems performance engineering on AWS

AIM347 is a workshop of 120 minutes at level 300 for Keita Watanabe and Aravind Neelakantan. This draft implements one FSDP training job with cumulative network, host and storage configuration changes. Production uses AWS PCS with Slurm and two `g7e.12xlarge` instances per attendee, for four GPUs total. The metric choice remains open for co-speaker review: the dashboard exposes training throughput, MFU against both dense denominators, and useful tokens per allocated GPU-hour.

**This is runnable draft content, not a calibrated workshop.** The production training ladder, its monotonic improvement, and the session timing are UNVALIDATED on `g7e.12xlarge`. The [validation record](VALIDATION.md) distinguishes successful Seoul execution checks from the remaining production rehearsal. Historical hardware facts below are attributed to the PI's Oregon measurements; they are not new results from this implementation. Seoul uses `p6-b300.48xlarge` on EKS. That platform can validate code execution and transport selection, but cannot validate PCS core availability, absence of NVLink, the production dense denominator or production performance.

## Hardware and evidence boundaries

| Status and provenance | Observation or limitation |
|---|---|
| VALIDATED, PI's Oregon `g7e.12xlarge` run, 2026-08-15 | Two nodes joined Slurm and completed a job with two GPUs and one EFA interface per node. |
| VALIDATED, PI's Oregon `g7e.12xlarge` run | Inside an exclusive PCS job, `nproc` and `SLURM_CPUS_ON_NODE` both reported 24 usable cores. `lscpu` still displayed 48 logical CPUs. PCS disables SMT at bootstrap; this is not configurable. |
| VALIDATED, PI's Oregon `g7e.12xlarge` run | `nvidia-smi nvlink --status` reported `Device does not have or support Nvlink`; the GPU pair's topology was `PIX`. NVLink counter panels read zero rather than erroring. Use PCIe fields for this instance. |
| VALIDATED, PI's Oregon `g7e.12xlarge` run | Dense BF16 cuBLAS GEMM reached 429.1 TFLOPS per GPU with FP32 accumulation, a square matrix dimension of 8192 elements, and the best of 30 measured trials. |
| VALIDATED, PI's Oregon `g7e.12xlarge` run | DCGM returned graphics-engine activity, SM activity, SM occupancy, tensor activity, DRAM activity, PCIe transmit and PCIe receive fields. Lustre client `open`, `getattr`, `read_bytes` and `write_bytes` counters were present and changed with work. |
| UNVALIDATED, production `g7e.12xlarge` | The container stack as a whole, Slurm/Pyxis launch of this implementation, NCCL TCP versus EFA bandwidth, production model and data sizing, each MFU increase, and serving latency. Historical device visibility is not evidence of collective performance. |

The production launch template must differ from a p5en template in these ways: configure one EFA interface, use `ONDEMAND` with a standard ODCR target, and omit the placement group. The PI verified those changes in Oregon. Place FSx for Lustre in the same Availability Zone as the compute reservation. The historical FSx mount crossed Availability Zones and does not establish production storage bandwidth. Do not substitute `g6e.12xlarge`: the PI deliberately excluded it because GPUDirect RDMA was not documented on its product or accelerated-computing specification pages, and its L40S GPUs with 48 GB per GPU did not fit the PI's measured configuration. Those are the hardware-selection findings from the supplied spec, not a new memory-capacity test of this draft.

## Prerequisites and preparation

Use a pre-provisioned PCS cluster. The repository's [PCS architecture](../../../architectures/aws-pcs/README.md) provides the infrastructure starting point; this example deploys the lab onto assigned nodes, rather than creating participant accounts or procuring capacity. Confirm the launch-template changes above with the facilitator before provisioning. A provisioned cluster needs:

- An assigned Slurm partition and two exclusive `g7e.12xlarge` nodes with `--gres=gpu:2`, the EFA driver, NVIDIA driver compatible with CUDA version 13.0.2, and a same-AZ FSx mount shared with the login node.
- Pyxis and Enroot installed on compute nodes, Slurm PMIx support for the container's MPI, and Docker with the Compose plugin on the preparation/login host. The compute observability containers require Docker, NVIDIA Container Toolkit and access to the GPU devices. These host packages belong in the validated PCS AMI; this draft does not provision or pin a replacement AMI.
- Host Python and Lustre's `lctl` on compute nodes. The collector runs on the host so it can use the installed Lustre libraries.
- Private connectivity between the assigned nodes and login node. Allow Prometheus to reach TCP ports 9400, 9100, 9109 and 8000 on assigned compute nodes, and compute nodes to reach Pushgateway on TCP port 9091. EFA needs the cluster's established security-group configuration. Grafana and the Prometheus UI bind to loopback and are accessed through SSH forwarding.
- Download access during preparation, then pre-staged container images and model files. There is no Hugging Face token requirement for the selected model. Pre-stage outside the attendee time budget.

Copy this directory to a shared FSx path and set the assigned hostnames and paths. Set `NCCL_SOCKET_IFNAME` to the private interface returned by `ip route get` for the peer node; the example interface name must be checked on the actual AMI:

```bash
cp lab.env.example .env
# Edit .env before continuing. Every participant needs a separate DATA_DIR and RUN_ID.
./0.deploy-observability.sh login
./1.prepare.sh
```

Run the compute deployment on each assigned compute node, using the same lab directory:

```bash
./0.deploy-observability.sh compute
```

The compute collector writes into the node-local `/var/lib/aim347/textfile` directory. It invokes `lctl get_param 'llite.*.stats'` and atomically replaces a Prometheus textfile. Missing statistics produce a collection-failure metric; they are never fabricated as zero traffic. The EFA exporter reads existing sysfs counters and omits fields absent from the driver. No external source patches are needed.

The generated Grafana password is in `observability/runtime/grafana-password` on the login host. Forward local TCP port 3000 to login-node TCP port 3000, sign in as `admin`, and open the provisioned AIM347 dashboard. Forward TCP port 9090 for the Prometheus UI. On the login node, check actual scrape health:

```bash
curl -fsS http://127.0.0.1:9090/api/v1/targets
```

**UNVALIDATED on production `g7e.12xlarge`:** each compute exporter's target should be healthy after deployment, and the vLLM targets should become healthy when serving starts. A healthy scrape does not prove that every optional profiling field is available. Inspect the actual `/metrics` payload and the Lustre collection-success panel. The serving targets being down before the pivot is normal.

### Software pins

| Component | Pin |
|---|---|
| Training and nccl-tests base | `public.ecr.aws/hpc-cloud/nccl-tests:cuda13.0.2-efa1.48.0-ofiv1.19.0-ncclv2.30.4-1-testsv2.18.3` |
| Python training dependencies | `requirements.txt` and transitive pins in `constraints.txt`: PyTorch version 2.9.1 with CUDA version 13.0 wheels, Transformers version 4.57.6, NumPy version 2.2.6, Hugging Face Hub version 0.36.0, safetensors version 0.6.2, tokenizers version 0.22.1 |
| Model configuration and pretrained serving weights | `Qwen/Qwen2.5-3B`, commit `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` |
| Prometheus | `prom/prometheus:v3.5.0` |
| Grafana | `grafana/grafana:12.1.1` |
| Pushgateway | `prom/pushgateway:v1.11.1` |
| DCGM exporter | `nvcr.io/nvidia/k8s/dcgm-exporter:4.6.0-4.8.3-distroless` |
| Node exporter | `prom/node-exporter:v1.9.1` |
| EFA collector runtime | `python:3.12.10-slim-bookworm` |
| vLLM | `vllm/vllm-openai:v0.20.2` |

Prometheus, Grafana and Pushgateway are self-hosted containers. Amazon Managed Grafana is unsupported in Workshop Studio according to the PI's confirmed workshop constraint, so this lab does not use it.

## Run the same workload through the configurations

Preparation generates deterministic synthetic token records in both small-file and packed-shard layouts. Both layouts have identical token contents, recorded by a dataset SHA-256 digest. The training architecture comes from the pinned Qwen model configuration and starts from the same random initialization for each run. This exercises systems behavior; it does not measure model quality or convergence. The initial dataset has 32768 records, each containing 512 input tokens and one following target token, packed into shards of 2048 records. These are configuration dimensions, not hardware measurements.

| Configuration | NCCL network | DataLoader workers | Storage |
|---|---|---|---|
| `v0` | Force `Socket` and disable plugin loading | 48 workers per rank, hence 96 workers per production node | One record per file, with repeated real metadata checks |
| `v1` | Require `AWS Libfabric`, with `FI_PROVIDER=efa` | Same 48 workers per rank | Same small files |
| `v2` | Same EFA configuration | 8 workers per rank, hence 16 workers per production node | Same small files |
| `v3` | Same EFA configuration | Same 8 workers per rank | Memory-mapped packed shards, without per-record file opens |

The fixed worker budget leaves eight of the 24 usable production cores for rank processes, telemetry, Slurm and the operating system. This is a starting allocation from the PI's spec, not a measured optimal worker count. CPU preprocessing performs the same configurable hashing work in every configuration. Tune `CPU_ROUNDS`, dataset size and the metadata-check count during the production rehearsal; never insert sleeps to manufacture the desired ladder.

Execute the numbered scripts in order. Each training submission waits for completion and returns failure if the job fails. Use the historical production reference of 429.1 TFLOPS per GPU in `.env` for the first baseline. The later metric-computation step recomputes all completed runs against one chosen denominator.

```bash
./2.run-v0.sh
./3.gemm-ceiling.sh
./4.run-v1.sh
./5.run-v2.sh
./6.run-v3.sh
python3 7.compute-metrics.py results/participant-trial-a \
  --gemm results/participant-trial-a/gemm-node-0.json \
  --pushgateway http://10.0.0.10:9091
```

Replace the example run identifier and private address with `.env` values. Inspect both node GEMM measurements before selecting a denominator. If the GPUs differ materially, investigate clocks, contention and placement; using one best result as the common reference is only defensible for a comparable homogeneous allocation. This script measures the first visible GPU on each node, not every GPU in the allocation.

Both network configurations run `nccl-tests` with correctness checking before training. Keep their `algbw` and `busbw` values with the instance type and GPU count printed in the job record. Require the NCCL logs to identify `Socket` for `v0` and the OFI/libfabric plugin using EFA for the fixed configurations. [NCCL's network selection settings](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-net) distinguish the network name from the libfabric provider: `NCCL_NET=efa` is not the network name used here. Merely unsetting environment variables does not reliably cause TCP fallback because plugin discovery can succeed automatically.

**UNVALIDATED on production `g7e.12xlarge`:** the intended sequence is network limitation, then CPU contention, then storage metadata limitation. EFA counter silence alone does not prove TCP fallback: an absent metric, idle job or other workload can produce misleading observations. Pair counter deltas with NCCL logs and a controlled collective benchmark. An increase in every MFU step is an untested hypothesis. If a step does not improve, retain the measurement and identify the current limiting resource. Do not replace it with an expected value.

## Dense GEMM ceiling and MFU arithmetic

The benchmark calls dense `cublasGemmEx` with BF16 inputs, FP32 accumulation and FP32 output. Its default matrix dimension is 8192 elements, with ten warmup trials and 30 measured trials. It records each CUDA event duration, the median and the best rate, software versions, GPU name and the supplied instance type. Inputs are initialized to ones, and sampled output elements must match the matrix dimension. There is no sparse kernel or structured sparsity enabled.

For a square GEMM with dimension `n` elements and duration `t` seconds:

```text
GEMM FLOP count = 2 × n³
GEMM TFLOPS per GPU = 2 × n³ / t / 10¹²
model FLOP count per token ≈ 6 × nonembedding parameter count
training TFLOPS per GPU = tokens per second × 6 × parameter count / GPU count / 10¹²
MFU against measured dense GEMM = training TFLOPS per GPU / dense GEMM TFLOPS per GPU
```

**VALIDATED, PI's Oregon `g7e.12xlarge` run:** the measured achievable reference is 429.1 TFLOPS per GPU. The alternative dense theoretical reference of 480 TFLOPS per GPU is **DERIVED**, using the spec's FP32 rate of 120 TFLOPS per GPU and a dimensionless multiplier of four. The PI did not find a vendor-stated dense BF16 value of 480 TFLOPS per GPU. The measured-to-derived ratio is approximately 89.4 percent, above the spec's initial expected range; this is evidence to retain, not a reason to alter the observation.

The NVIDIA headline of 1 PFLOP per second for BF16 on the RTX PRO 6000 Blackwell Server Edition includes structured sparsity. A dense training workload needs a dense denominator. The arithmetic direction matters: using 1000 TFLOPS per GPU in place of 429.1 TFLOPS per GPU makes the reported MFU smaller by a dimensionless factor of approximately 2.33. It overstates available compute, not achieved utilization. Both source outlines contain wording that reverses or obscures that direction; this implementation follows the formula.

For the production allocation of four GPUs, the historical achievable denominator is 1.7164 PFLOPS for the allocation, and the derived theoretical denominator is 1.92 PFLOPS for the allocation. These are arithmetic totals from the PI's `g7e.12xlarge` reference, not new measurements. The dashboard labels these ratios separately. A value above a measured GEMM reference calls for scrutiny of the numerator and benchmark conditions; it is not clamped away.

The `6N` numerator is a short-sequence approximation. It omits attention's sequence-dependent term and is not a kernel-level FLOP measurement. Training timing includes data wait, host-to-device copies, forward, backward and optimizer execution, using the slowest rank's duration. Warmup updates are excluded from steady-state MFU timing and included in useful completed work. The allocation ledger includes failed attempts with no useful-work credit and successful updates with their token count. Its scope is the training job body, including startup and the network microbenchmark; separately scheduled GEMM and serving allocations are excluded. It is therefore not an account-wide goodput measure or a recovery demonstration. A full-session goodput metric must include those extra allocations and any lost or repeated work in its accounting.

## Read the dashboard

| Layer | Fields and interpretation |
|---|---|
| GPU compute | `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`, `DCGM_FI_PROF_GR_ENGINE_ACTIVE`, `DCGM_FI_PROF_SM_ACTIVE`, `DCGM_FI_PROF_SM_OCCUPANCY`: dimensionless activity ratios. The supplied CSV explicitly requests the profiling fields. |
| GPU memory | `DCGM_FI_PROF_DRAM_ACTIVE`: dimensionless activity ratio; `DCGM_FI_DEV_FB_USED`: MiB of framebuffer memory. |
| GPU interconnect | `DCGM_FI_PROF_PCIE_TX_BYTES` and `DCGM_FI_PROF_PCIE_RX_BYTES`: bytes per second, already rates. They include GPU P2P and host transfers, so they cannot uniquely attribute communication to either. Production `g7e.12xlarge` has no NVLink; an NVLink-only panel would teach nothing about its actual link. |
| EFA | `node_amazonefa_tx_bytes`, `node_amazonefa_rx_bytes`, `node_amazonefa_rdma_write_bytes`, `node_amazonefa_retrans_bytes`: cumulative bytes. `node_amazonefa_rx_drops`, `node_amazonefa_retrans_timeout_events`, `node_amazonefa_unresponsive_remote_events`: cumulative event counts. Apply `rate()` to these counters. |
| Lustre | `lustre_read_bytes_total`, `lustre_write_bytes_total`: byte sums from `llite` histograms. `lustre_open_operations_total`, `lustre_getattr_operations_total`: operation counts. Compare their rates; high metadata operations with low byte throughput can suggest metadata limitation. |
| Host | `node_cpu_seconds_total`, `node_pressure_io_waiting_seconds_total`, `node_memory_*`. Interpret CPU utilization against PCS's 24 usable cores per `g7e.12xlarge` node. |
| Training | `aim347_tokens_per_second`, `aim347_step_duration_ms`, `aim347_tflops_per_gpu`, `aim347_mfu_dense_gemm_ratio`, `aim347_mfu_dense_theory_ratio`. Rank zero pushes these gauges with configuration and instance labels; local JSON remains the measurement record if telemetry fails. |
| Accounting | `aim347_useful_tokens_per_gpu_hour`, published by the metric-computation script from the allocation ledger. |
| Serving | `vllm:generation_tokens_total`, `vllm:time_to_first_token_seconds_bucket`, `vllm:inter_token_latency_seconds_bucket`, `vllm:num_requests_running`. |

Profiling requires the DCGM container's `SYS_ADMIN` capability. [DCGM's profiling documentation](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html) explains activity metrics and collection limitations. Missing or sentinel-valued profiling fields require investigation before interpreting them as utilization. Do not run an additional profiler that conflicts with an existing DCGM profiling session on a shared node.

## Serving pivot

The serving script reserves the same assigned nodes and GPUs after training completes. It starts one tensor-parallel vLLM replica on each node, each using two GPUs. This uses the same model architecture with pinned pretrained weights, not the randomly initialized training run's weights. It avoids a distributed serving runtime dependency between the replicas. The common dashboard scrapes both endpoints.

```bash
./8.serve-vllm.sh
# In another terminal, after both /health endpoints succeed:
python3 9.load-serving.py http://aim347-g7e-1:8000 http://aim347-g7e-2:8000 \
  --instance-type g7e.12xlarge --output results/participant-trial-a/serving.json
```

**UNVALIDATED on production `g7e.12xlarge`:** serving start, throughput, TTFT and ITL for this configuration. The load generator records request TTFT and output-token counts from real streamed responses. Client chunk intervals are not token-level ITL; use vLLM's server histogram for ITL. Decode can be memory-bandwidth limited, so MFU alone is insufficient. No MBU value is computed here because a defensible byte-traffic numerator and measured bandwidth denominator have not been established for this model. GPU DRAM activity is not MBU.

The serving job has a time limit of 20 minutes. Cancel its specific Slurm job when the load exercise completes. Failure injection, automatic checkpoint recovery and prerecorded Nsight timelines remain outside this draft; they need an explicit decision and real evidence before being included in the agenda.

## Rehearsal and cleanup

The proposed module budget is framing for ten minutes, observability for 25 minutes, baseline plus GEMM for 20 minutes, network for 15 minutes, data/host for 15 minutes, storage for 15 minutes, ceiling comparison plus serving for ten minutes, and a buffer of ten minutes. **UNVALIDATED on production `g7e.12xlarge`:** all timing, including the target training duration of at most three minutes per configuration. Model initialization, worker startup and container import can dominate a short run. Pre-staging reduces preparation work but does not establish the timing target.

Retain `results/` locally. It contains logs, raw GEMM timings, training summaries and failed-attempt accounting. The [dashboard JSON](observability/dashboard.json) is the takeaway artifact. Do not add datasets, model weights, container images or result binaries to Git.

```bash
# On each assigned compute node:
./10.cleanup.sh compute
# On the login node:
./10.cleanup.sh login
```

Cleanup stops only this lab's Compose projects and the `aim347-lustre` service. It keeps local metrics volumes and staged data for review. It does not delete the PCS cluster, FSx filesystem or another participant's jobs. Remove this participant's staged files and named volumes only after preserving the required evidence.

Local checks use `python3 -m unittest discover -s tests -v`, `bash -n` for the shell entry points and `promtool check config` for the generated Prometheus configuration. Arithmetic test fixtures are synthetic software checks, never hardware benchmark results.
