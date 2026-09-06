# Validation record

These are observed results from 2026-09-06. The new GPU tests ran on `p6-b300.48xlarge` in Seoul using EKS. Production uses `g7e.12xlarge` with AWS PCS and Slurm. A successful test below establishes only the stated scope. None establishes production performance.

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
| UNVALIDATED | Production `g7e.12xlarge` | Both serving replicas together with the production memory fraction and default graph execution, measured MBU, failure/recovery goodput and prerecorded Nsight timelines. |

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

The serving test used `vllm/vllm-openai:v0.20.2`, the pinned Qwen pretrained weights, BF16 precision and tensor parallelism across two GPUs on one Seoul `p6-b300.48xlarge` node. Validation used a GPU-memory fraction of 0.2, a maximum of eight concurrent sequences, eager execution and a model context limit of 2048 tokens. Production scripts use a memory fraction of 0.8 and the normal execution mode; that larger configuration remains UNVALIDATED.

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
