# AI systems performance engineering on AWS

AIM347 is a workshop of 120 minutes at level 300 for Keita Watanabe and Aravind Neelakantan. This example implements one FSDP training job with cumulative network, host and storage configuration changes. AWS PCS with Slurm allocates two assigned nodes exclusively and uses their full GPU count. The dashboard exposes training throughput, MFU against both dense denominators, and useful tokens per allocated GPU-hour. The serving exercise adds weights-only MBU against a measured DRAM read-bandwidth ceiling.

The primary allocation is `g7.48xlarge`; the secondary allocation is `g7.24xlarge`. Each uses the same companion code and records its actual GPU, EFA and processor counts. The [validation record](VALIDATION.md) labels the full G7 run, the g7.24xlarge stand-in and historical g7e/Seoul observations separately. Throughput and utilization conclusions are limited to those measured allocations.

## Hardware and evidence boundaries

| Status and provenance | Observation or limitation |
|---|---|
| VALIDATED, PI's Oregon `g7e.12xlarge` run, 2026-08-15 | Two nodes joined Slurm and completed a job with two GPUs and one EFA interface per node. |
| VALIDATED, PI's Oregon `g7e.12xlarge` run | Inside an exclusive PCS job, `nproc` and `SLURM_CPUS_ON_NODE` both reported 24 usable cores. `lscpu` still displayed 48 logical CPUs. This is the observed scheduler-visible shape for that historical allocation; the Spain G7 full allocation exposes all 192 logical processors per node. |
| VALIDATED, PI's Oregon `g7e.12xlarge` run | `nvidia-smi nvlink --status` reported `Device does not have or support Nvlink`; the GPU pair's topology was `PIX`. NVLink counter panels read zero rather than erroring. Use PCIe fields for this instance. |
| VALIDATED, PI's Oregon `g7e.12xlarge` run | Dense BF16 cuBLAS GEMM reached 429.1 TFLOPS per GPU with FP32 accumulation, a square matrix dimension of 8192 elements, and the best of 30 measured trials. |
| VALIDATED, PI's Oregon `g7e.12xlarge` run | DCGM returned graphics-engine activity, SM activity, SM occupancy, tensor activity, DRAM activity, PCIe transmit and PCIe receive fields. Lustre client `open`, `getattr`, `read_bytes` and `write_bytes` counters were present and changed with work. |
| UNVALIDATED, tertiary `p4d.24xlarge` / `p4de.24xlarge` | The container stack as a whole, Slurm/Pyxis launch of this implementation, NCCL TCP versus EFA bandwidth, production model and data sizing, each MFU increase, and serving latency. Historical device visibility is not evidence of collective performance. |

The historical g7e rehearsal launch template differed from a p5en template in these ways: configure one EFA interface, use `ONDEMAND` with a standard ODCR target, and omit the placement group. The PI verified those g7e-specific changes in Oregon. They are not a p4d/p4de launch configuration. The current Workshop Studio wrapper selects the G7-specific node template and retains the existing P4 and P5/P6 alternatives. Place FSx for Lustre in the same Availability Zone as the compute reservation. The historical FSx mount crossed Availability Zones and does not establish production storage bandwidth. Do not substitute `g6e.12xlarge`: the PI deliberately excluded it because GPUDirect RDMA was not documented on its product or accelerated-computing specification pages, and its L40S GPUs with 48 GB per GPU did not fit the PI's measured configuration. Those are the hardware-selection findings from the supplied spec, not a new memory-capacity test of this draft.

## Prerequisites and preparation

Use a pre-provisioned PCS cluster. The repository's [PCS architecture](../../../architectures/aws-pcs/README.md) provides the infrastructure starting point; this example deploys the lab onto assigned nodes, rather than creating participant accounts or procuring capacity. Select the G7 shape in the wrapper and verify the interface list read back from EC2. A provisioned cluster needs:

- An assigned Slurm partition and two exclusive homogeneous GPU nodes, the EFA driver, NVIDIA driver compatible with CUDA version 13.0.2, and a same-AZ FSx mount shared with the login node.
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

The generated Grafana password is in `observability/runtime/grafana-password` on the login host. Forward local TCP port 3000 to login-node TCP port 3000, sign in as `admin`, and open the provisioned AIM347 dashboard. Forward the configured `PROMETHEUS_PORT` for the Prometheus UI; the rehearsal uses TCP port `9092` because an existing agent owns TCP port `9090`. On the login node, check actual scrape health:

```bash
source .env
curl -fsS "http://127.0.0.1:${PROMETHEUS_PORT:-9090}/api/v1/targets"
```

Inspect the requested profiling and EFA diagnostic fields on each assigned compute node:

```bash
curl -s localhost:9400/metrics | grep PIPE_TENSOR_ACTIVE
curl -s localhost:9109/metrics | grep -E 'retrans|rx_drops'
```

**Hardware-dependent expectation:** each compute exporter's target should be healthy after deployment, and the vLLM targets should become healthy when serving starts. A healthy scrape does not prove that every optional profiling field is available. Inspect the actual `/metrics` payload and the Lustre collection-success panel. The serving targets being down before the pivot is normal.

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

The Dockerfile rebuilds NCCL and the nccl-tests CUDA kernels from the same sources supplied by the base image with native kernels for GPU architecture codes `sm_80`, `sm_90`, `sm_100`, `sm_103` and `sm_120`. The original binary lacked `sm_120` and PTX and failed on the observed RTX PRO Blackwell GPUs. No upstream source patch or NCCL version substitution is applied.

Prometheus, Grafana and Pushgateway are self-hosted containers. Amazon Managed Grafana is unsupported in Workshop Studio according to the PI's confirmed workshop constraint, so this lab does not use it.

## Run the same workload through the configurations

Preparation generates deterministic synthetic token records in both small-file and packed-shard layouts. Both layouts have identical token contents, recorded by a dataset SHA-256 digest. The training architecture comes from the pinned Qwen model configuration and starts from the same random initialization for each run. This exercises systems behavior; it does not measure model quality or convergence. The initial dataset has 32768 records, each containing 512 input tokens and one following target token, packed into shards of 2048 records. These are configuration dimensions, not hardware measurements.

| Configuration | NCCL network | DataLoader workers | Storage |
|---|---|---|---|
| `v0` | Force `Socket` and disable plugin loading | 48 workers per rank × detected GPUs per node | One record per file, with repeated real metadata checks |
| `v1` | Require `AWS Libfabric`, with `FI_PROVIDER=efa` | Same 48 workers per rank | Same small files |
| `v2` | Same EFA configuration | 8 workers per rank × detected GPUs per node | Same small files |
| `v3` | Same EFA configuration | Same 8 workers per rank | Memory-mapped packed shards, without per-record file opens |

The launcher records GPUs per node, total GPUs and available processors in `results/$RUN_ID/<action>-resources.json`. It uses a positive numeric `SLURM_GPUS_ON_NODE` when supplied, otherwise counts devices from `nvidia-smi -L` inside the allocation. Slurm sets that variable from the allocated GPU bitmap in its [GRES environment implementation](https://github.com/SchedMD/slurm/blob/slurm-24-11-0-1/src/plugins/gres/common/gres_common.c#L278). It clears OpenMP overrides before running `nproc`; processor availability is a recorded measurement, not a fixed-core gate. The inventory must contain one record per assigned node with the same GPU count. `torchrun` starts that many processes per node, and MPI collective tests use the total GPU count. By default, the [Slurm exclusive allocation](https://slurm.schedmd.com/sbatch.html#OPT_exclusive) requests all CPUs and GRES on the assigned nodes. Optional `LAB_GPUS_PER_NODE`, `LAB_CPUS_PER_TASK` and `LAB_MEMORY_PER_NODE` values select an explicit GPU, CPU and memory budget and omit the exclusive flag. Whole-node launch steps use the CPUs granted to the allocation; MPI steps use the configured per-rank CPU count.

Multiply workers per rank by the detected GPUs per node, then compare that worker count with the recorded available processors. On the historical `g7e.12xlarge` allocation, two GPUs per node made the reduced configuration 16 workers per node against 24 available cores. Other allocations produce different totals. The fixed per-rank settings are an experiment in host contention, not a measured optimal worker budget. CPU preprocessing performs the same configurable hashing work in every configuration. Tune `CPU_ROUNDS`, dataset size and the metadata-check count during the production rehearsal; never insert sleeps to manufacture the desired ladder.

Execute the numbered scripts in order. Each training submission waits for completion and returns failure if the job fails. Run the GEMM ceiling first and set `DENSE_TFLOPS` in `.env` to its recorded `best_tflops_per_gpu` value before the baseline. The historical value of 429.1 TFLOPS per GPU belongs to `g7e.12xlarge`; the example environment leaves the denominator empty until it is measured. The later metric-computation step recomputes all completed runs against one chosen denominator.

```bash
./3.gemm-ceiling.sh
source .env
cat "results/$RUN_ID/gemm-node-0.json" "results/$RUN_ID/gemm-node-1.json"
./3b.bandwidth-ceiling.sh
# Set DENSE_TFLOPS in .env to the measured best_tflops_per_gpu value.
./2.run-v0.sh
cat "results/$RUN_ID/v0-resources.json"
./4.run-v1.sh
./5.run-v2.sh
./6.run-v3.sh
source .env
python3 7.compute-metrics.py "results/$RUN_ID" \
  --gemm "results/$RUN_ID/gemm-node-0.json" \
  --pushgateway "$PUSHGATEWAY_URL"
```

Load the prepared `.env` in each shell. Inspect both node GEMM measurements before selecting a denominator. If the GPUs differ materially, investigate clocks, contention and placement; using one best result as the common reference is only defensible for a comparable homogeneous allocation. This script measures the first visible GPU on each node, not every GPU in the allocation.

Both network configurations run `nccl-tests` with correctness checking before training. Keep their `algbw` and `busbw` values with the instance type and GPU count printed in the job record. Require the NCCL logs to identify `Socket` for `v0` and the OFI/libfabric plugin using EFA for the fixed configurations. [NCCL's network selection settings](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-net) distinguish the network name from the libfabric provider: `NCCL_NET=efa` is not the network name used here. Merely unsetting environment variables does not reliably cause TCP fallback because plugin discovery can succeed automatically.

**Hardware-dependent expectation:** the intended sequence is network limitation, then CPU contention, then storage metadata limitation. EFA counter silence alone does not prove TCP fallback: an absent metric, idle job or other workload can produce misleading observations. Pair counter deltas with NCCL logs and a controlled collective benchmark. An increase in every MFU step is an untested hypothesis. If a step does not improve, retain the measurement and identify the current limiting resource. Do not replace it with an expected value.

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

For the historical `g7e.12xlarge` allocation of four GPUs, the historical achievable denominator is 1.7164 PFLOPS for the allocation, and the derived theoretical denominator is 1.92 PFLOPS for the allocation. These are arithmetic totals from the PI's `g7e.12xlarge` reference, not new measurements. The dashboard labels these ratios separately. A value above a measured GEMM reference calls for scrutiny of the numerator and benchmark conditions; it is not clamped away.

The `6N` numerator is a short-sequence approximation. It omits attention's sequence-dependent term and is not a kernel-level FLOP measurement. Training timing includes data wait, host-to-device copies, forward, backward and optimizer execution, using the slowest rank's duration. Warmup updates are excluded from steady-state MFU timing and included in useful completed work. The allocation ledger includes failed attempts with no useful-work credit and successful updates with their token count. Its scope is the training job body after initial allocation-evidence collection, including startup and the network microbenchmark; scheduler polling and separately scheduled GEMM, DRAM bandwidth and serving allocations are excluded. It is therefore not an account-wide goodput measure or a recovery demonstration. A full-session goodput metric must include those extra allocations and any lost or repeated work in its accounting.

## Read the dashboard

| Layer | Fields and interpretation |
|---|---|
| GPU compute | `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`, `DCGM_FI_PROF_GR_ENGINE_ACTIVE`, `DCGM_FI_PROF_SM_ACTIVE`, `DCGM_FI_PROF_SM_OCCUPANCY`: dimensionless activity ratios. The supplied CSV explicitly requests the profiling fields. |
| GPU memory | `DCGM_FI_PROF_DRAM_ACTIVE`: dimensionless activity ratio; `DCGM_FI_DEV_FB_USED`: MiB of framebuffer memory. |
| GPU interconnect | `DCGM_FI_PROF_PCIE_TX_BYTES` and `DCGM_FI_PROF_PCIE_RX_BYTES`: bytes per second, already rates. They include GPU P2P and host transfers, so they cannot uniquely attribute communication to either. The rehearsal `g7e.12xlarge` pair has no NVLink. The current G7 allocation also has no NVLink; record the actual topology. |
| EFA | `node_amazonefa_tx_bytes`, `node_amazonefa_rx_bytes`, `node_amazonefa_rdma_write_bytes`, `node_amazonefa_retrans_bytes`: cumulative bytes. `node_amazonefa_rx_drops`, `node_amazonefa_retrans_timeout_events`, `node_amazonefa_unresponsive_remote_events`: cumulative event counts. Apply `rate()` to these counters. |
| Lustre | `lustre_read_bytes_total`, `lustre_write_bytes_total`: byte sums from `llite` histograms. `lustre_open_operations_total`, `lustre_getattr_operations_total`: operation counts. Compare their rates; high metadata operations with low byte throughput can suggest metadata limitation. |
| Host | `node_cpu_seconds_total`, `node_pressure_io_waiting_seconds_total`, `node_memory_*`. Interpret CPU utilization against the available processors recorded for each allocated node. |
| Training | `aim347_tokens_per_second`, `aim347_step_duration_ms`, `aim347_tflops_per_gpu`, `aim347_mfu_dense_gemm_ratio`, `aim347_mfu_dense_theory_ratio`. Rank zero pushes these gauges with configuration and instance labels; local JSON remains the measurement record if telemetry fails. |
| Accounting | `aim347_useful_tokens_per_gpu_hour`, published by the metric-computation script from the allocation ledger. |
| Serving | `vllm:generation_tokens_total`, `vllm:time_to_first_token_seconds_bucket`, `vllm:inter_token_latency_seconds_bucket`, `vllm:num_requests_running`. |

Profiling requires the DCGM container's `SYS_ADMIN` capability. [DCGM's profiling documentation](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html) explains activity metrics and collection limitations. Missing or sentinel-valued profiling fields require investigation before interpreting them as utilization. Do not run an additional profiler that conflicts with an existing DCGM profiling session on a shared node.

## Serving pivot and measured weights-only MBU

The serving script reserves the same assigned nodes after training completes. It starts one tensor-parallel vLLM replica per node, using the detected GPU count on that node. It loads the pinned pretrained Qwen checkpoint with the training model's architecture. The common dashboard scrapes both endpoints.

First measure DRAM read bandwidth on every allocated GPU. The script writes raw per-GPU trials, prints each node's aggregate median bandwidth in bytes/s, and creates an endpoint-to-GPU measurement map:

```bash
source .env
./3b.bandwidth-ceiling.sh
cat "results/$RUN_ID/bandwidth-map.json"
```

The CUDA microbenchmark in [lib/dram.cu](lib/dram.cu) reads a buffer of 1 GiB per trial, requires that buffer to exceed four times the GPU's L2 cache size, uses loads that bypass L1, and times ten warmup trials followed by 30 measured trials with CUDA events. It checks the reduction result against the initialized input. Its reported byte count covers the input reads; the small reduction output is excluded. Record `median_bytes_per_second` for every GPU. The denominator for each replica is the sum of those measured rates multiplied by its measured server decode duration.

Read the numerator from the actual staged checkpoint, using the GPU count that will become the replica's tensor-parallel size:

```bash
python3 11.model-bytes.py "$DATA_DIR/serving-model" \
  --tensor-parallel-size "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["gpus_per_node"])' "results/$RUN_ID/bandwidth-resources.json")" \
  --output "results/$RUN_ID/model-bytes.json"
./8.serve-vllm.sh
```

[lib/serving_metrics.py](lib/serving_metrics.py) reads BF16 tensor shapes and byte offsets from the safetensors headers and the served `config.json`. It counts 2 bytes per element, the tied embedding/output matrix once, replicated normalization parameters once per tensor-parallel rank, and replicated K/V projection parameters when the rank count exceeds the model's KV-head count. The pinned [vLLM Qwen2 implementation](https://github.com/vllm-project/vllm/blob/v0.20.2/vllm/model_executor/models/qwen2.py) defines those parameter uses and partitions. `weights_bytes_per_decode_step` is a logical full-weight-read model. KV-cache traffic, activations, input embedding lookups, padding, and physical cache effects are outside this weights-only numerator.

In another terminal, load the same environment and endpoint map, wait for both health checks, and run the normal load pass followed by a serialized decode measurement:

```bash
source .env
mapfile -t SERVING_ENDPOINTS < <(python3 -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))))' "results/$RUN_ID/bandwidth-map.json")
curl -fsS "${SERVING_ENDPOINTS[0]}/health"
curl -fsS "${SERVING_ENDPOINTS[1]}/health"
python3 9.load-serving.py "${SERVING_ENDPOINTS[@]}" \
  --instance-type "$INSTANCE_TYPE" --output "results/$RUN_ID/serving.json"
python3 9.load-serving.py "${SERVING_ENDPOINTS[@]}" --concurrency 1 --measure-decode \
  --instance-type "$INSTANCE_TYPE" --output "results/$RUN_ID/serving-decode.json"
python3 12.compute-mbu.py "results/$RUN_ID/serving-decode.json" \
  --model-bytes "results/$RUN_ID/model-bytes.json" --bandwidth-map "results/$RUN_ID/bandwidth-map.json" \
  --output "results/$RUN_ID/serving-mbu.json"
```

Keep the endpoints idle between these passes and exclude other clients during decode measurement. The normal pass uses concurrency of four requests; the MBU pass uses one request at a time, preventing decode batching. The client rejects counters that disagree with its completed requests and token usage. Speculative decoding is outside this measurement mode. In pinned vLLM, the first output token belongs to prefill; subsequent output events contribute server ITL samples. The [request statistics implementation](https://github.com/vllm-project/vllm/blob/v0.20.2/vllm/v1/metrics/stats.py) supplies these intervals, and the [Prometheus logger](https://github.com/vllm-project/vllm/blob/v0.20.2/vllm/v1/metrics/loggers.py) exports the generation-token and successful-request counters and ITL histogram.

```text
decode steps = generated output tokens - completed requests
modeled weight-read bytes = weights bytes per decode step × decode steps
measured decode capacity bytes = sum of replica GPU median bandwidths in bytes/s × server decode seconds
weights-only MBU ratio = modeled weight-read bytes / measured decode capacity bytes
```

For multiple replicas, sum the numerator and capacity bytes over each replica's own decode interval before dividing. This measures the modeled weights traffic during active server decode, not whole-session allocation utilization or physical DRAM counter utilization. The output keeps MBU as a dimensionless ratio beside output tokens/s, median client TTFT in seconds, and mean server ITL in seconds. Client chunk spacing and GPU DRAM activity remain separate observations. MBU values are not clamped. Compare measurements made with the same byte model, batch policy and bandwidth method.

The serving job has a time limit of 20 minutes. Cancel its specific Slurm job when the exercise completes. Hardware observations and the PCS/Slurm validation boundary are recorded in [VALIDATION.md](VALIDATION.md).

## Rehearsal and cleanup

The proposed module budget is framing for ten minutes, observability for 25 minutes, baseline plus GEMM for 20 minutes, network for 15 minutes, data/host for 15 minutes, storage for 15 minutes, ceiling comparison plus serving for ten minutes, and a buffer of ten minutes. **Hardware-dependent expectation:** all timing, including the target training duration of at most three minutes per configuration. Model initialization, worker startup and container import can dominate a short run. Pre-staging reduces preparation work but does not establish the timing target.

Retain `results/` locally. It contains logs, raw GEMM timings, training summaries and failed-attempt accounting. The [dashboard JSON](observability/dashboard.json) is the takeaway artifact. Do not add datasets, model weights, container images or result binaries to Git.

```bash
# On each assigned compute node:
./10.cleanup.sh compute
# On the login node:
./10.cleanup.sh login
```

Cleanup stops only this lab's Compose projects and the `aim347-lustre` service. It keeps local metrics volumes and staged data for review. It does not delete the PCS cluster, FSx filesystem or another participant's jobs. Remove this participant's staged files and named volumes only after preserving the required evidence.

Local checks in the pinned training container use `python3 -m unittest discover -s tests -v`, `bash -n` for the shell entry points and `promtool check config` for the generated Prometheus configuration. Arithmetic test fixtures are synthetic software checks, never hardware benchmark results.

## Facilitator preparation helper

`facilitator/prepare-login.sh` stages the login prerequisites and is also the entry point for the Workshop Studio SSM preparation document. Set `AIM347_PARTITION` and, after checking the peer route, `AIM347_SOCKET_INTERFACE`. The helper reads the instance type from the assigned compute hosts. The helper writes a new `.env` only when one is absent, uses Slurm's `NodeAddr` for the private login route, runs `1.prepare.sh`, starts login monitoring and checks its ready endpoints. A new environment leaves `DENSE_TFLOPS` empty until the GEMM measurement. Existing `.env` files are preserved.

```bash
./facilitator/prepare-login.sh
```

The PCS rehearsal used TCP port `9092` for Prometheus because the existing login agent occupied TCP port `9090`. The full upstream monitoring stack also binds TCP ports `3000` and `9091`, so the AIM347 Workshop Studio wrapper defaults to `MonitoringStack=none`. The compute exporter deployment remains a separate step after exclusive health diagnostics. The helper accepts a digest-pinned `PREBUILT_LAB_IMAGE` to avoid rebuilding the same training image on each login host. The staged vLLM image is pulled by its immutable platform digest with Docker and then imported through `dockerd://`; direct registry import failed on the observed Enroot version `3.5.0`.

The PCS launcher explicitly preserves the configured `NCCL_SOCKET_IFNAME` through Pyxis. Image-defined environment values otherwise take precedence over the host. MPI ranks use the participant UID with native PMIx authentication, select the compatible `hash` data store inside only MPI tasks, and bind MPI TCP bootstrap after the container environment loads. The configured NCCL network remains the difference between the Socket baseline and OFI configurations.

## PCS g7e rehearsal outcome, 2026-09-09

The facilitator helpers and complete participant path ran on two `g7e.12xlarge` PCS nodes with four GPUs total and twenty-four available CPU cores per node. The current dense ceiling was `427.142119 TFLOP/s/GPU`. Training throughput changed from `1383.301689 tokens/s` on Socket to `4094.242545 tokens/s` on EFA; the worker and packed-shard changes then measured `4082.912026 tokens/s` and `4086.011788 tokens/s`. This run did not demonstrate a monotonic ladder or CPU/storage saturation. Both serving replicas completed the supplied client, with serialized weights-only MBU of dimensionless ratio `0.662560288067`. All nine monitoring targets were healthy during serving, and the collector read real FSx counters on both nodes. See [VALIDATION.md](VALIDATION.md) for exact images, wall times, counter windows, failed attempts and scope. The allocation had no common cluster placement group and accessed FSx across AZs.

## Spain resource and storage configuration, 2026-09-10

The constrained G7 rehearsal used the following optional `.env` settings. Their numeric values are four GPUs per node, 24 vCPUs per GPU rank and 384 GiB per node. PCS also needs device, CPU and RAM cgroup enforcement and memory-aware scheduling. `AIM_ENFORCE_STANDIN=1` makes the allocation-evidence script reject an incorrect GPU count, excess available processors or an absent/excess memory ceiling.

```bash
LAB_GPUS_PER_NODE=4
LAB_CPUS_PER_TASK=24
LAB_MEMORY_PER_NODE=384G
LAB_NODE_LOCAL=1
AIM_ENFORCE_STANDIN=1
DCGM_EXPORTER_DEVICES=g:0-3
FI_EFA_IFACE=rdmap83s0
OFI_NCCL_PROTOCOL=RDMA
```

Select the EFA and exporter GPU indices from the actual assignment. The G7-specific RDMA protocol improved the measured NCCL baseline; do not make it a universal platform default. The DCGM filter must match the GPU UUIDs inside the job. It filters monitored devices rather than granting a GPU allocation.

Set `AIM347_NODE_LOCAL=1` for `facilitator/prepare-login.sh` when using an existing `/opt/aim347` NVMe stage instead of Lustre. Prepare on a compute node, then copy the checkout, `.env`, images, data and pretrained model to the identical path on its peer. `LAB_NODE_LOCAL=1` gathers per-node ceiling results through Slurm steps because those files are not shared. Verify image checksums and dataset fingerprints on both hosts before starting. The default preparation path still requires Lustre. In Spain, Prometheus, Grafana and Pushgateway ran on the first compute node; their private address and ports were written into the prepared environment and dashboard configuration.

Without a Lustre mount, `aim347_lustre_collection_success` remains a dimensionless value of zero. The collector publishes no substitute filesystem counters, and the participant Lustre completion conditions remain unmet. This rehearsal measured the local-NVMe workload and preserved that limitation. Best dense GEMM was 171.702538 TFLOP/s/GPU. Training throughput progressed through 1715.936407, 5602.045734, 5713.913125 and 5655.706739 tokens/s, so the packed-shard change was slower. Serialized weights-only MBU was a dimensionless ratio of 0.552492958628. See [VALIDATION.md](VALIDATION.md) for allocations, full metrics, cold serving latency, per-module times and remaining boundaries.

## G7 allocation and automatic protocol selection

Use two `g7.48xlarge` nodes as the primary allocation or two `g7.24xlarge` nodes as the secondary allocation. The infrastructure parameter selects the shape. Preparation discovers the compute instance type; each job reads it again from DMI, falling back to IMDSv2, and detects its allocated GPU count and available processors. Record the host inventory during the environment check:

```bash
bash lib/resources.sh
```

`lib/job.sh` applies `OFI_NCCL_PROTOCOL=RDMA` only when `lib/instance-type.sh` detects `g7.*`, and preserves it through Pyxis. No participant-set instance variable controls the gate. The Spain `g7.24xlarge-stand-in` collective measured 5.2 GB/s with SENDRECV and 20.3 GB/s with RDMA at 2 GiB, four ranks per node and one EFA; [VALIDATION.md](VALIDATION.md) preserves that evidence. Leave `FI_EFA_IFACE` unset to expose all EFA devices in a whole-node run.

The default exclusive submission derives ranks from the allocation. Explicit budgets can additionally set `LAB_EXCLUSIVE=1`; `LAB_MEMORY_PER_NODE=0` selects all scheduler-visible memory. A stand-in rehearsal sets an explicit device, CPU and memory budget without exclusivity. The hardware inventory records physical EFA devices, while `FI_EFA_IFACE` records any deliberately restricted provider selection.

| Allocation | Initial worker request | Reduced worker request | CPU budget per node |
|---|---|---|---|
| g7.48xlarge | 48 workers/rank × 8 ranks/node = 384 workers/node | 8 workers/rank × 8 ranks/node = 64 workers/node | 192 vCPUs |
| g7.24xlarge-stand-in | 48 workers/rank × 4 ranks/node = 192 workers/node | 8 workers/rank × 4 ranks/node = 32 workers/node | 96 vCPUs |

These are configured worker requests, not a claim of saturation or an optimal worker count. Use the recorded job inventory and measured result for each comparison. G7 has no NVLink; inspect PCIe and EFA fields.
