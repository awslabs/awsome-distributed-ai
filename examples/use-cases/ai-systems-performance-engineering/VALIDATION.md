# Validation record

The first sections preserve historical observations from 2026-09-06. The final sections record new Oregon checks from 2026-09-09. The historical GPU tests ran on `p6-b300.48xlarge` in Seoul using EKS. Production uses `g7e.12xlarge` with AWS PCS and Slurm. A successful test below establishes only the stated scope. None establishes production performance.

## Observed scope

| Status | Hardware and test | Result |
|---|---|---|
| VALIDATED | Seoul `p6-b300.48xlarge`, two existing EKS nodes | Used an isolated namespace with nonpreempting Pods and no additional EC2 instances. The reservation had 32 occupied instance seats and zero available instance seats. Existing node groups and workloads were preserved. |
| VALIDATED | Seoul `p6-b300.48xlarge`, cuBLAS GEMM on one visible GPU per node | Dense BF16 input, FP32 accumulation/output, matrix dimension of 8192 elements, ten warmup trials and 30 measured trials. Best rates were 2204.525761 TFLOPS per GPU and 2204.950141 TFLOPS per GPU; median rates were 2199.164036 TFLOPS per GPU and 2197.968370 TFLOPS per GPU. |
| VALIDATED | Seoul `p6-b300.48xlarge`, topology | Each visible GPU pair reported `NV18`. This confirms that Seoul cannot reproduce the production instance's absence of NVLink. |
| VALIDATED | Seoul `p6-b300.48xlarge`, nccl-tests across four GPUs on two nodes | Both corrected runs passed with zero out-of-bounds values. The reported average bus bandwidth was 5.89254 GB/s using Socket and 39.3278 GB/s using EFA. These are observations from one run per transport, not calibrated fabric baselines. |
| VALIDATED | Seoul `p6-b300.48xlarge`, reduced FSDP workload | All configuration files executed with their original worker counts and layouts. NCCL logs selected Socket for `v0` and AWS OFI NCCL with libfabric/EFA for the fixed configurations. The measured ladder was not monotonic. |
| VALIDATED | Seoul `p6-b300.48xlarge`, full Qwen model configuration | The `v3` configuration completed four update steps across four GPUs, with one warmup step and three measured steps. This is an execution smoke test. |
| VALIDATED | Seoul `p6-b300.48xlarge`, self-hosted telemetry | Prometheus, Grafana and Pushgateway started. The dashboard loaded with 25 panels. Both node-exporter targets, both EFA targets, both existing DCGM targets and Pushgateway were healthy. All four configuration metrics could be queried simultaneously. |
| VALIDATED | Seoul `p6-b300.48xlarge`, serving | One vLLM replica with tensor parallelism across two GPUs served 16 requests from the supplied client and exported token and latency metrics to the same Prometheus instance. |
| VALIDATED, software test only | Pinned training container, CPU execution | Six regression tests passed, covering dense-denominator arithmetic, input validation, separate Pushgateway groups for each configuration, Lustre histogram parsing, missing EFA fields, and identical data across storage layouts. These are synthetic fixtures, not GPU measurements. |
| UNVALIDATED | Production `g7e.12xlarge` with PCS/Slurm | This implementation's complete launch path, production model/data sizing, CPU saturation, metadata limitation, monotonic MFU increases, production collective performance, and the workshop time budget. |
| UNVALIDATED | Production `g7e.12xlarge` and Seoul `p6-b300.48xlarge` | Live Lustre collection by this new script. Parsing was tested against explicit fixtures; the Seoul dataset lived on local container storage, not FSx. Historical Oregon Lustre visibility remains PI-supplied evidence. |
| UNVALIDATED | Production `g7e.12xlarge` | The new compute Compose deployment with its custom DCGM CSV. Seoul reused the existing GPU Operator DCGM exporters read-only rather than starting competing profiling sessions. |
| UNVALIDATED | Production `g7e.12xlarge` | Both serving replicas together under PCS/Slurm. The single-replica EKS MBU observation is recorded separately below; failure/recovery goodput and Nsight timelines are not part of the selected exercise. |

## Reduced training results

All measurements in this table came from Seoul `p6-b300.48xlarge`, with four GPUs across two EKS nodes. The test model had two decoder layers, a hidden dimension of 128 elements, an intermediate dimension of 512 elements, four attention heads, two key/value heads and a vocabulary of 1024 tokens. It used sequences of 128 tokens, one sequence per rank per step, 12 update steps with two warmup steps, and 100 CPU hashing rounds per record. The unmodified configuration files selected 48 workers per rank for `v0` and `v1`, then eight workers per rank for `v2` and `v3`. The EKS Pods had a CPU quota equivalent to 24 cores, while process affinity exposed 192 logical CPUs; that is not PCS's physical-core behavior.

| Configuration | Observed throughput on `p6-b300.48xlarge` | MFU / measured B300 GEMM, dimensionless ratio |
|---|---|---|
| `v0` | 28192.89 tokens/s | 0.000009450908 |
| `v1` | 30490.87 tokens/s | 0.000010221246 |
| `v2` | 29972.32 tokens/s | 0.000010047413 |
| `v3` | 28581.45 tokens/s | 0.000009581163 |

The common denominator was the observed 2204.525761 TFLOPS per GPU on Seoul `p6-b300.48xlarge`. The model is deliberately too small for these MFU values to represent training efficiency at the workshop's intended scale. The full model's short `v3` execution check on the same Seoul hardware observed 5891.33 tokens/s and a dimensionless MFU/GEMM ratio of 0.011122877, using sequences of 512 tokens and 2774773760 nonembedding parameters. Neither result is a production target.

Seoul training calls used the shipped `train.py` through `torchrun` directly, with an explicit node rank, two processes per node, and two nodes. They bypassed the PCS submission wrapper. They allocated one EFA device per Pod and disabled huge-page use for these container checks. NCCL used the routable private interface. No source patches were applied. The Slurm allocation ledger was not exercised on EKS, so hardware goodput accounting remains UNVALIDATED.

## Serving and telemetry

The serving test used `vllm/vllm-openai:v0.20.2`, the pinned Qwen pretrained weights, BF16 precision and tensor parallelism across two GPUs on one Seoul `p6-b300.48xlarge` node. Validation used a GPU-memory fraction of 0.2, a maximum of eight concurrent sequences, eager execution and a model context limit of 2048 tokens. Production scripts use a memory fraction of 0.8 and the normal execution mode; the Oregon single-replica check below subsequently exercised those settings on EKS. The PCS path remains UNVALIDATED.

The client ran on the serving node against loopback, with 16 requests, concurrency of four requests and 128 generated tokens per request. It observed 168.36 output tokens/s across a wall duration of 12.1645 seconds and a median TTFT of 44.16 milliseconds on Seoul `p6-b300.48xlarge`. These include the actual request scheduling in this single client run and are not production latency targets. Prometheus received vLLM token totals and TTFT histogram samples. Only the running replica's target was healthy; the absent second replica was not counted as validated.

The production Docker Compose login stack also passed a local CPU-only startup check: Prometheus accepted the configuration and Grafana loaded the dashboard. The new EFA collector ran in the pinned Python container on Seoul. PCIe fields were present through the existing DCGM exporters. The scope of that telemetry check is narrower than a complete production deployment with the supplied custom CSV.

## Failures retained during validation

- NCCL's automatic interface selection chose an EKS link-local interface and the first training attempt could not connect. Selecting the actual private interface fixed the run. The production environment example requires that interface to be verified.
- Setting `OMP_NUM_THREADS` to a dimensionless thread count of one made an unqualified `nproc` invocation report one usable processor. The PCS preflight now clears the OpenMP overrides before checking core availability.
- The first Socket nccl-tests attempt left OFI plugin loading enabled and hit an OFI parameter-initialization assertion. Disabling plugin loading for the Socket baseline resolved the failure. The shipped baseline already uses that setting; no patch or upstream merge is required for this path.
- A system node could not finish pulling Grafana because its filesystem was full. Only the new observability Pod was moved to a validation node. Existing images and workloads were left in place.
- The first replacement EFA-exporter Pods started before the previous training Pods had released their listener ports. Their initial starts failed with an address-in-use error; recreating only those exporters after the old Pods exited succeeded.
- The earliest training pushes ran before Pushgateway became ready. Local JSON summaries survived and were later published successfully. A subsequent check caught configuration groups overwriting one another; the exporter now includes the configuration in the Pushgateway grouping path, with a regression test and a successful query of all four configurations.

The complete commands and unedited logs are retained in the PI's local handoff report. The submitted speaker outline was not changed. New validation resources were scoped to this task and are removed after evidence collection.

## Oregon serving and weights-only MBU: VALIDATED, 2026-09-09

The new checks used two existing `g7e.12xlarge` EKS nodes in `us-west-2`, each with two NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs and one EFA device. Node `ip-10-3-132-239.us-west-2.compute.internal` ran training checks; node `ip-10-3-133-250.us-west-2.compute.internal` ran the DRAM read benchmark and then serving on the same GPU pair. The new namespace was `aim347-pi-20260909`, with a dedicated nonpreempting PriorityClass. Each pod requested 8 CPU cores and 128 GiB memory. Process affinity exposed 48 available processors per node; the pod CPU quota was 8 CPU cores. These are EKS measurements, not PCS core observations. The NVIDIA driver was version `580.159.03`.

The benchmark used the revised training Dockerfile at ECR manifest-list digest `sha256:9b06854610d1ba5ef688f7dc1e350866a0cb4b50880167229e746a762358cbe7`. This image compiled both `aim347-gemm` and `aim347-dram`. The current controller and Python measurement sources were copied into the owned pods; the raw CUDA benchmark source was unchanged after the image build. The serving pod used vLLM version `0.20.2`, pinned to the public AMD64 manifest digest `sha256:68b773151407ca28c05479c02c0c08a573285ba197c8ff35d2103acd97aff78c`.

Both `aim347-dram` runs read 1073741824 bytes per trial, exceeding the observed L2 cache size of 134217728 bytes by a dimensionless factor of eight. Each used ten warmup trials and 30 measured trials, with correctness checks passed. GPU UUIDs were `5f3b8e9322760954f672dc9423fcfb30` and `d2aad840a431fa5a9220e8dfc21d2b78`. Median bandwidths were 1332450366654.801 bytes/s and 1329994553557.416 bytes/s, giving a replica denominator rate of 2662444920212.217 bytes/s. This is a streaming read microbenchmark measurement; it is not a datasheet rate or a DCGM activity ratio.

The model was the actual staged `Qwen/Qwen2.5-3B` checkpoint at revision `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`. Its `config.json` specifies BF16 precision, 36 decoder layers, a hidden dimension of 2048 elements, an intermediate dimension of 11008 elements, 16 attention heads, two KV heads, a vocabulary of 151936 tokens, and tied word embeddings. `11.model-bytes.py` read the safetensors headers and recorded 6171877376 checkpoint weight bytes. With tensor parallelism across two GPUs, replicated normalization parameters added 299008 logical read bytes per decode step, yielding 6172176384 weight bytes per decode step. The actual served `config.json:9` gives the hidden dimension, `config.json:16` gives the layer count, `config.json:17` gives the KV-head count, and `config.json:21` and `config.json:22` specify tied embeddings and BF16 precision. The header-byte formula and replication adjustment are in `lib/serving_metrics.py:34` and `lib/serving_metrics.py:39`. The input reads, CUDA event timing and bytes/s formula are in `lib/dram.cu:16`, `lib/dram.cu:41` and `lib/dram.cu:53`. The local report retains the served configuration and tensor inventory. KV-cache traffic is omitted. The value models full logical weight reads and does not measure physical cache behavior.

vLLM started with the companion's BF16 precision, tensor-parallel size of two GPUs, context limit of 2048 tokens, GPU-memory fraction of 0.8, and normal graph execution. No eager-execution override or lower memory fraction was used. The standard client first completed 16 requests at concurrency of four requests, each producing 128 output tokens. This first pass took 34.206296 seconds and includes cold execution effects; its 59.872020 output tokens/s is not a calibrated concurrency comparison.

The same `9.load-serving.py` then ran with `--concurrency 1 --measure-decode` against the replica's loopback endpoint. It completed 16 requests and 2048 output tokens over 7.994151 seconds. Server counter deltas exactly matched 16 successful requests, 2048 generated tokens and 2032 ITL samples. Their summed server decode duration was 7.843961 seconds. The first output token per request is excluded from decode steps, consistent with pinned vLLM's `v1/metrics/stats.py` implementation. The client preserved the before counters and per-endpoint deltas.

| Recorded quantity | Oregon EKS observation |
|---|---|
| Modeled weight-read numerator | 12541862412288 bytes |
| Measured bandwidth × server decode duration | 20884114246430.234 bytes |
| Weights-only MBU | 0.600545576 dimensionless ratio, or 60.054558 percent |
| Output throughput in the serialized pass | 256.187315 output tokens/s |
| Median client TTFT | 8.807258 ms |
| Mean server ITL | 3.860217 ms |

`12.compute-mbu.py` produced those values together in `serving-mbu.json`, retaining each GPU's raw timings and the model tensor inventory. Both the numerator and denominator belong to the same replica and active decode intervals. This validates the direct EKS client and arithmetic on two GPUs. It does not establish physical DRAM-counter utilization, concurrent/batched MBU, KV-cache MBU, multiple serving replicas together, another model or precision, another instance type, or the PCS/Slurm serving and bandwidth launch path. Those scopes remain UNVALIDATED.

## Allocation detection and training checks, 2026-09-09

The shell-level tests exercised `lib/resources.sh` with both `nvidia-smi` and `SLURM_GPUS_ON_NODE` stubs for allocations of two GPUs, four GPUs and eight GPUs per node. They executed the actual `lib/job.sh` through stubbed `srun`, `scontrol` and `torchrun`, inspecting the expanded MPI task totals and `torchrun --nproc-per-node` values for two allocated nodes. Failed stub launches retained the detected total GPU count in the ledger with zero useful tokens. These are software tests, not Slurm hardware observations. All 11 tests in the training container passed before the hardware run.

The real node inventory detected two GPUs from `nvidia-smi`. Dense BF16 GEMM on the first visible training GPU passed correctness checks and measured a best rate of 423.906461 TFLOP/s/GPU and a median rate of 420.696274 TFLOP/s/GPU, using the default 8192-element matrix dimension, ten warmup trials and 30 measured trials.

The initial two-process `torchrun` used the full shipped model configuration, `configs/v3.json`, and the pinned NCCL version `2.30.4`. It failed at the first barrier with CUDA error `named symbol not found`. The same image's two-GPU `all_reduce_perf` reproduced the failure. `cuobjdump` found native architecture images through `sm_103`, no `sm_120` image and no PTX in `/opt/nccl/build/lib/libnccl.so`; the GPUs reported compute capability version `12.0`. The Dockerfile now rebuilds the same supplied NCCL source with `sm_120` included alongside the other documented architecture targets. No library source patch or version replacement was applied.

The full PCS/Slurm path, a real four-GPU-per-node or eight-GPU-per-node launch, cross-node collectives in this tab, and allocation-ledger timing on real Slurm remain UNVALIDATED. No competing profiler or DCGM exporter was started. Dedicated staged data and model files remain under `/mnt/aim347-decisions-20260909` on the assigned nodes. Exact command transcripts and the cleanup result are in `/tmp/companion-decisions-report.md`.

### Two-GPU torchrun: VALIDATED

With the rebuilt NCCL library, the same full `lib/train.py` completed four update steps with one warmup step and three measured steps on the training node's two GPUs. The image manifest-list digest was `sha256:fb18b91b96daa1b7b2c2530ad3d817a15446b69f3b140dde7dcfd1f49ee443db`. The runtime NCCL library SHA-256 was `5ee5f392fbc91f24d5ab8f274f1f5a8c71e29d9ef486482027c4137dbf49527b`, and `cuobjdump` showed native `sm_120` images. The source overlay contained the revised detection and accounting scripts; the training implementation itself was unchanged.

The command used `--standalone --nnodes=1` and `--nproc-per-node` from the actual `lib/resources.sh` inventory. It used the full model configuration, the original `configs/v3.json` worker/layout settings, 512 input tokens per sequence, one sequence per rank, and 100 CPU hashing rounds per record. The generated dataset retained 32768 records and SHA-256 `cbbf272e4c2f6ebf0a502f384425a4d5cbafcc83823ffe00dce86aa8babf4807`. This single-node EKS execution selected Socket and disabled plugin loading; the configuration file's cross-node EFA path was not exercised.

The observed throughput was 2308.891474 tokens/s, mean step duration was 443.502872 ms, and estimated computation rate was 19.219954 TFLOP/s/GPU. Against the measured dense GEMM ceiling of 423.906461 TFLOP/s/GPU, MFU was a dimensionless ratio of 0.045340084. The run completed 4096 useful training tokens. These are short execution observations, not a calibrated training ladder or production targets.

The first diagnostic after replacing NCCL still failed in the original nccl-tests correctness kernel with `no kernel image is available for execution on the device`. The Dockerfile therefore rebuilds nccl-tests with the same architecture list and MPI enabled. A command launched during the earlier container image transition also reached the old binary before Kubernetes restarted the container; that log was preserved separately. Successful results above were taken only after checking the live image ID.

### Final image collective check: VALIDATED

The final Dockerfile image, manifest-list digest `sha256:e714b9e6c38bafccc0ce5b08561a0afb83c9dc1a3af57f80d5228193a00dd81d` in the task ECR repository, rebuilt both NCCL and nccl-tests with native `sm_120` kernels and MPI support. After checking the live image ID, `all_reduce_perf -b 8M -e 8M -g 2 -w 1 -n 1 -c 1` completed with zero out-of-bounds values and average bus bandwidth of 26.6249 GB/s. This short two-GPU, same-node diagnostic used P2P/direct-pointer channels; it is not cross-node EFA performance. Runtime versions remained nccl-tests version `2.18.3` and NCCL version `2.30.4`. The final NCCL library SHA-256 was `d0db16f797021367ba935f120e3f837097f22912c6b006868c53dfe9abd0417f`; the all-reduce executable SHA-256 was `ae7ae1ad8f61ff9f8bb3996b1f6432aa50232f532b5ae40fd7a94dc4b18cb0e0`.

The owned bandwidth and serving pods were removed after evidence collection. The remaining training pod, namespace and dedicated PriorityClass were deleted at the end of the run. Existing namespaces, Deployments, nodes and workloads were not modified. The PI-managed `leahtuck-llama70b` namespace and Deployment were preserved with the same Deployment spec and zero desired replicas.
