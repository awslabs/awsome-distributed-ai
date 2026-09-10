# Validation record

The first sections preserve historical observations from 2026-09-06. Later sections record Oregon checks from 2026-09-09 and Spain checks from 2026-09-10. The historical GPU tests ran on `p6-b300.48xlarge` in Seoul using EKS. The current participant guide targets `p4d.24xlarge` or `p4de.24xlarge` with AWS PCS and Slurm; the Oregon rehearsal uses `g7e.12xlarge`. Historical sections retain their original hardware labels. A successful test below establishes only the stated scope. None establishes production performance.

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
| VALIDATED, 2026-09-09 PCS rehearsal | Two Oregon `g7e.12xlarge` nodes | Full Slurm/Pyxis/PMIx training ladder, measured ceilings, real-FSx collector, custom compute exporters, allocation ledger and both serving replicas. Exact scope and results appear in the final sections. |
| UNVALIDATED | Production `p4d.24xlarge` / `p4de.24xlarge` | Full deployment and runtime, A100 performance, CPU/storage bottleneck progression and workshop time budget. |
| NOT DEMONSTRATED | Oregon `g7e.12xlarge` | Monotonic MFU improvement, CPU saturation and a storage metadata bottleneck; the recorded full ladder did not establish these hypotheses. |

## Reduced training results

All measurements in this table came from Seoul `p6-b300.48xlarge`, with four GPUs across two EKS nodes. The test model had two decoder layers, a hidden dimension of 128 elements, an intermediate dimension of 512 elements, four attention heads, two key/value heads and a vocabulary of 1024 tokens. It used sequences of 128 tokens, one sequence per rank per step, 12 update steps with two warmup steps, and 100 CPU hashing rounds per record. The unmodified configuration files selected 48 workers per rank for `v0` and `v1`, then eight workers per rank for `v2` and `v3`. The EKS Pods had a CPU quota equivalent to 24 cores, while process affinity exposed 192 logical CPUs; that is not PCS's physical-core behavior.

| Configuration | Observed throughput on `p6-b300.48xlarge` | MFU / measured B300 GEMM, dimensionless ratio |
|---|---|---|
| `v0` | 28192.89 tokens/s | 0.000009450908 |
| `v1` | 30490.87 tokens/s | 0.000010221246 |
| `v2` | 29972.32 tokens/s | 0.000010047413 |
| `v3` | 28581.45 tokens/s | 0.000009581163 |

The common denominator was the observed 2204.525761 TFLOPS per GPU on Seoul `p6-b300.48xlarge`. The model is deliberately too small for these MFU values to represent training efficiency at the workshop's intended scale. The full model's short `v3` execution check on the same Seoul hardware observed 5891.33 tokens/s and a dimensionless MFU/GEMM ratio of 0.011122877, using sequences of 512 tokens and 2774773760 nonembedding parameters. Neither result is a production target.

Seoul training calls used the shipped `train.py` through `torchrun` directly, with an explicit node rank, two processes per node, and two nodes. They bypassed the PCS submission wrapper. They allocated one EFA device per Pod and disabled huge-page use for these container checks. NCCL used the routable private interface. No source patches were applied. The Slurm allocation ledger was not exercised on EKS, so that historical run did not validate Slurm accounting. The later PCS job-body ledger is recorded below.

## Serving and telemetry

The serving test used `vllm/vllm-openai:v0.20.2`, the pinned Qwen pretrained weights, BF16 precision and tensor parallelism across two GPUs on one Seoul `p6-b300.48xlarge` node. Validation used a GPU-memory fraction of 0.2, a maximum of eight concurrent sequences, eager execution and a model context limit of 2048 tokens. Production scripts use a memory fraction of 0.8 and the normal execution mode; the Oregon single-replica check below subsequently exercised those settings on EKS. That historical check did not exercise PCS; the later PCS result appears below.

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

`12.compute-mbu.py` produced those values together in `serving-mbu.json`, retaining each GPU's raw timings and the model tensor inventory. Both the numerator and denominator belong to the same replica and active decode intervals. This validates the direct EKS client and arithmetic on two GPUs. It does not establish physical DRAM-counter utilization, concurrent/batched MBU, KV-cache MBU, multiple serving replicas together, another model or precision, another instance type, or the PCS/Slurm serving and bandwidth launch path. The later PCS section establishes the two-replica launch and arithmetic; the other listed limits remain unvalidated.

## Allocation detection and training checks, 2026-09-09

The shell-level tests exercised `lib/resources.sh` with both `nvidia-smi` and `SLURM_GPUS_ON_NODE` stubs for allocations of two GPUs, four GPUs and eight GPUs per node. They executed the actual `lib/job.sh` through stubbed `srun`, `scontrol` and `torchrun`, inspecting the expanded MPI task totals and `torchrun --nproc-per-node` values for two allocated nodes. Failed stub launches retained the detected total GPU count in the ledger with zero useful tokens. These are software tests, not Slurm hardware observations. All 11 tests in the training container passed before the hardware run.

The real node inventory detected two GPUs from `nvidia-smi`. Dense BF16 GEMM on the first visible training GPU passed correctness checks and measured a best rate of 423.906461 TFLOP/s/GPU and a median rate of 420.696274 TFLOP/s/GPU, using the default 8192-element matrix dimension, ten warmup trials and 30 measured trials.

The initial two-process `torchrun` used the full shipped model configuration, `configs/v3.json`, and the pinned NCCL version `2.30.4`. It failed at the first barrier with CUDA error `named symbol not found`. The same image's two-GPU `all_reduce_perf` reproduced the failure. `cuobjdump` found native architecture images through `sm_103`, no `sm_120` image and no PTX in `/opt/nccl/build/lib/libnccl.so`; the GPUs reported compute capability version `12.0`. The Dockerfile now rebuilds the same supplied NCCL source with `sm_120` included alongside the other documented architecture targets. No library source patch or version replacement was applied.

At this historical EKS checkpoint, the full PCS path, real four-GPU-per-node or eight-GPU-per-node launch, cross-node collectives and real Slurm accounting had not been observed. The later PCS sections establish the two-GPU-per-node launch, cross-node collectives and job-body ledger; larger per-node GPU counts remain unvalidated. No competing profiler or DCGM exporter was started. Dedicated staged data and model files remain under `/mnt/aim347-decisions-20260909` on the assigned nodes. Exact command transcripts and the cleanup result are in `/tmp/companion-decisions-report.md`.

### Two-GPU torchrun: VALIDATED

With the rebuilt NCCL library, the same full `lib/train.py` completed four update steps with one warmup step and three measured steps on the training node's two GPUs. The image manifest-list digest was `sha256:fb18b91b96daa1b7b2c2530ad3d817a15446b69f3b140dde7dcfd1f49ee443db`. The runtime NCCL library SHA-256 was `5ee5f392fbc91f24d5ab8f274f1f5a8c71e29d9ef486482027c4137dbf49527b`, and `cuobjdump` showed native `sm_120` images. The source overlay contained the revised detection and accounting scripts; the training implementation itself was unchanged.

The command used `--standalone --nnodes=1` and `--nproc-per-node` from the actual `lib/resources.sh` inventory. It used the full model configuration, the original `configs/v3.json` worker/layout settings, 512 input tokens per sequence, one sequence per rank, and 100 CPU hashing rounds per record. The generated dataset retained 32768 records and SHA-256 `cbbf272e4c2f6ebf0a502f384425a4d5cbafcc83823ffe00dce86aa8babf4807`. This single-node EKS execution selected Socket and disabled plugin loading; the configuration file's cross-node EFA path was not exercised.

The observed throughput was 2308.891474 tokens/s, mean step duration was 443.502872 ms, and estimated computation rate was 19.219954 TFLOP/s/GPU. Against the measured dense GEMM ceiling of 423.906461 TFLOP/s/GPU, MFU was a dimensionless ratio of 0.045340084. The run completed 4096 useful training tokens. These are short execution observations, not a calibrated training ladder or production targets.

The first diagnostic after replacing NCCL still failed in the original nccl-tests correctness kernel with `no kernel image is available for execution on the device`. The Dockerfile therefore rebuilds nccl-tests with the same architecture list and MPI enabled. A command launched during the earlier container image transition also reached the old binary before Kubernetes restarted the container; that log was preserved separately. Successful results above were taken only after checking the live image ID.

### Final image collective check: VALIDATED

The final Dockerfile image, manifest-list digest `sha256:e714b9e6c38bafccc0ce5b08561a0afb83c9dc1a3af57f80d5228193a00dd81d` in the task ECR repository, rebuilt both NCCL and nccl-tests with native `sm_120` kernels and MPI support. After checking the live image ID, `all_reduce_perf -b 8M -e 8M -g 2 -w 1 -n 1 -c 1` completed with zero out-of-bounds values and average bus bandwidth of 26.6249 GB/s. This short two-GPU, same-node diagnostic used P2P/direct-pointer channels; it is not cross-node EFA performance. Runtime versions remained nccl-tests version `2.18.3` and NCCL version `2.30.4`. The final NCCL library SHA-256 was `d0db16f797021367ba935f120e3f837097f22912c6b006868c53dfe9abd0417f`; the all-reduce executable SHA-256 was `ae7ae1ad8f61ff9f8bb3996b1f6432aa50232f532b5ae40fd7a94dc4b18cb0e0`.

The owned bandwidth and serving pods were removed after evidence collection. The remaining training pod, namespace and dedicated PriorityClass were deleted at the end of the run. Existing namespaces, Deployments, nodes and workloads were not modified. The PI-managed `leahtuck-llama70b` namespace and Deployment were preserved with the same Deployment spec and zero desired replicas.

### PCS MPI launch compatibility

The first participant baseline in Slurm job identifier `25` failed before NCCL initialization. The pinned image has Open MPI version `4.1.7` with embedded PMIx version `3.2.5a1`; the PCS host has PMIx version `5.0.6` and exposes the `pmix_v5` Slurm plugin. Both Enroot PMI hooks were already installed. The first error was `Framework: gds` / `Component: shmem2`; the following generic “not built with SLURM's PMI support” message did not identify the actual defect. EFA counter changes were `0 B` on both nodes.

A controlled four-rank CPU MPI probe separated data-store selection from user identity. Job identifier `27` used `PMIX_MCA_gds=hash` with container root remapping and still failed MPI initialization. Job identifier `28` used `hash` without root remapping, retained `PMIX_MCA_psec=native`, and completed a real `MPI_Allreduce`: all four ranks reported `size=4` and `sum=6`, both dimensionless values. The controller command took `44.266194 s`, including batch submission and polling. No PMIx security mode or Slurm daemon configuration was changed. The failed jobs left their MPI step daemons stuck in completion; only the identified daemons for job identifiers `25` and `27` were killed on the two rehearsal nodes. The scheduler then released the nodes. Probe job identifier `26` had a C-source quoting error and is not MPI validation evidence.

A follow-up baseline, job identifier `29`, exposed a second invocation issue: exporting a `PMIX_` variable around `--mpi=none` steps makes Enroot version `3.5.0`'s hook select its PMIx branch despite the absent server paths. It failed with `enroot-mount: failed to mount` for `/var/spool/slurmd/pmix.29.3`. The final launchers set `PMIX_MCA_gds=hash` inside only the MPI rank, after container environment loading, and run MPI ranks without root remapping. AIM344's plugin mutation commands retain remapping inside their private writable container. AIM347 also applies MPI TCP bootstrap settings inside its rank wrapper so the image's environment cannot override them. The configured NCCL transport remains the experimental variable.

Evidence: `/tmp/g7e-e2e/pcs/participant-25/`, `pcs/pmix-host-first.log`, `pcs/pmix-host-second.log`, `pcs/pmix-probe-27.log`, `pcs/pmix-probe-28.log`, `pmix-probe-uid-submit.command.json`, and the individual cleanup SSM records. NVIDIA documents the extra PMI hooks and the `hash` compatibility setting in [Pyxis setup](https://github.com/NVIDIA/pyxis/wiki/Setup). The observed native-authentication success supplies the basis for preserving that configuration.

### AIM347 compute preparation and live observability

Slurm job identifier `31` executed the facilitator compute preflight and both staged-image imports on the same two g7e nodes after the original PCS Prolog settings were restored. Its controller wall time was `76.390334 s`, with exit code dimensionless value `0`. Allocated inventory again showed two GPUs and twenty-four available CPU cores on each node. Docker Compose version `5.1.4`, NVIDIA Container Toolkit version `1.20.0`, the NVIDIA Docker runtime and host `lctl` were present. The original node-exporter services on TCP port `9100` were stopped only on these nodes; their enabled state was recorded. Inherited DCGM exporter containers remained stopped. The companion started its own compute exporters and `aim347-lustre` service.

At `16:47:21.860115 UTC`, the login-side scrape verification found all seven non-serving Prometheus targets healthy: two DCGM exporters, two node exporters, two EFA exporters and Pushgateway. Both future vLLM targets were down with connection refused because serving had not started. Five required DCGM profiling families were present for all four GPUs, with valid dimensionless idle values of zero: tensor pipeline, graphics engine, SM activity, SM occupancy and DRAM activity. Both EFA exporters exposed transmit/receive bytes, retransmission bytes and receive-drop event counts. Retransmission counters were `0 B` and drop counters were zero events at this snapshot.

For the first real FSx validation, both node exporters exposed `aim347_lustre_collection_success` with dimensionless value `1`, plus byte and metadata-operation counters for the actual mounted Lustre filesystem. Read counters were `194420178488 B` and `194419419795 B`; write counters were `387504 B` and `72079 B`; open counters were `599 operations` and `368 operations` on the first and second nodes, respectively. These are cumulative mount counters, not per-workload deltas or same-AZ performance measurements. The FSx mount remains in `us-west-2a`, while the compute pair is in `us-west-2d`.

Exact checks and raw scrapes are in `/tmp/g7e-e2e/pcs/observability-before-training/`, `pcs/compute-preflight-31/`, and `aim347-observability-before-training.*`. The verification command took `2.659874 s` including SSH and returned exit code dimensionless value `0`.

### AIM347 image environment precedence correction

The dense ceiling measurements completed in jobs `32` and `33`. The first training attempt, job identifier `34`, failed before NCCL data transfer. The configured host value was `NCCL_SOCKET_IFNAME==enp39s0`, but the image's `NCCL_SOCKET_IFNAME=^docker,lo,veth` took precedence. MPI then received `^docker` as its interface include value and reported `Invalid specification (missing "/")`. Pyxis version `0.20.0` documents image environment precedence and the host override option in [README.md lines 98-105](https://github.com/NVIDIA/pyxis/blob/v0.20.0/README.md#L98-L105).

`lib/job.sh` now passes `--container-env=NCCL_SOCKET_IFNAME` for training and serving. The rank wrapper still sets MPI bootstrap variables after image environment loading. This preserves the configured private NIC in both the NCCL microbenchmark and Torch training. The failed attempt took `10.052539 s` at the module controller and remains in `v0/allocation.jsonl` with zero useful-work credit. Its raw job log is `results/participant-trial-a/v0-34.log`; the retry uses a distinct module log and preserves the original attempt. The first retried microbenchmark selected `NET/Socket` on `enp39s0` and passed correctness through `256 MiB`.

### AIM347 dense compute and DRAM ceilings

The real Slurm GEMM job identifier `32` measured the first visible GPU on each g7e node with dense BF16 inputs, FP32 accumulation, square dimension `8192 elements`, ten warmup trials and thirty measured trials. The command `bash 3.gemm-ceiling.sh` took `42.061678 s` including scheduler submission and polling. Both output files identify `g7e.12xlarge`, the RTX PRO 6000 Blackwell Server Edition GPU, and `sparse=false`. The first node measured best/median rates of `427.142119 TFLOP/s/GPU` and `423.311087 TFLOP/s/GPU`, with best/median trial times of `2.574112 ms` and `2.597408 ms`. The second measured `424.934011 TFLOP/s/GPU` and `422.868241 TFLOP/s/GPU`, with times of `2.587488 ms` and `2.600128 ms`. The first node's best rate, `427.142119 TFLOP/s/GPU`, was written to `.env` and used consistently by training and recomputed metrics.

The command `bash 3b.bandwidth-ceiling.sh`, Slurm job identifier `33`, took `10.109148 s`. Each GPU passed the DRAM benchmark's reduction check, using a `1 GiB` read buffer against a measured `128 MiB` L2 cache, ten warmup trials and thirty measured trials. These are same-allocation empirical bandwidth denominators for the serving calculation.

| g7e node and GPU index | Median read bandwidth | Median trial time |
| --- | --- | --- |
| First node, GPU index `0` | `1332873798457.720 B/s` | `0.805584 ms` |
| First node, GPU index `1` | `1333165082465.987 B/s` | `0.805408 ms` |
| Second node, GPU index `0` | `1334251837054.080 B/s` | `0.804752 ms` |
| Second node, GPU index `1` | `1332662098215.186 B/s` | `0.805712 ms` |

The first and second replica bandwidth sums are `2666038880923.707 B/s` and `2666913935269.266 B/s`. `bandwidth-map.json` associates each physical hostname endpoint with its two measured GPU files. Raw trial arrays, GPU identities and software versions are retained in `results/participant-trial-a/gemm-node-*.json` and `dram-node-*-gpu-*.json` under the staged companion, plus the final evidence copy. No theoretical memory bandwidth was substituted for these measurements.

### AIM347 full training ladder on PCS g7e

All four configurations completed through the shipped `lib/job.sh`, real Slurm batch submission, Pyxis and PMIx on the same two `g7e.12xlarge` nodes in `us-west-2d`. Each node supplied two RTX PRO 6000 Blackwell Server Edition GPUs and twenty-four allocated CPU cores. PCS Slurm was version `25.05.7`, AMI was `ami-0aa5c69145b3bda5e`, and driver was version `595.71.05`. The nodes had one EFA device each, no NVLink, no common cluster placement group, and an FSx mount across AZs. The training image digest was `sha256:cc952d627df9d4a8b306117f152bd8bf04156f72b7b7dc517cd46c78ee9a2e17`, with PyTorch version `2.9.1+cu130` and runtime NCCL version `2.30.4`. Its imported-image SHA-256 was `f39fef48cd9e14d8bf02998fa62d76d4106d2a96402a5cea12f41e74821de14c`.

The full Qwen architecture used thirty-six layers, a hidden dimension of `2048 elements`, an intermediate dimension of `11008 elements`, a vocabulary of `151936 tokens`, and `2774773760 nonembedding parameters`. Training used randomly initialized weights, `512 tokens` per sequence, one sequence per GPU rank, `100 update steps`, ten warmup steps, ninety measured steps and `2000 CPU hashing rounds` per record. The deterministic dataset contained `32768 records`; its logical data SHA-256 was `cbbf272e4c2f6ebf0a502f384425a4d5cbafcc83823ffe00dce86aa8babf4807`. Every successful configuration completed `204800 useful tokens`. Each final finite loss was a dimensionless value of `12.3141002655`. The common measured dense denominator was `427.142119 TFLOP/s/GPU`.

| g7e configuration, job identifier and command | Module wall time | Training throughput | Mean measured step | MFU/GEMM, dimensionless ratio | Completion check |
| --- | --- | --- | --- | --- | --- |
| `v0`, job `35`, `bash 2.run-v0.sh` | `234.072739 s` | `1383.301689 tokens/s` | `1480.515795 ms` | `0.013479176104` | PASS: completed, same allocation and denominator, actual counters and comparison retained |
| `v1`, job `36`, `bash 4.run-v1.sh` | `138.061675 s` | `4094.242545 tokens/s` | `500.214625 ms` | `0.039895141248` | PASS: completed, same allocation and denominator, actual counters and comparison retained |
| `v2`, job `37`, `bash 5.run-v2.sh` | `106.069815 s` | `4082.912026 tokens/s` | `501.602774 ms` | `0.039784734347` | PASS: completed, same allocation and denominator, actual counters and comparison retained |
| `v3`, job `38`, `bash 6.run-v3.sh` | `106.063282 s` | `4086.011788 tokens/s` | `501.222245 ms` | `0.039814939058` | PASS: completed, same allocation and denominator, actual counters and comparison retained |

At `256 MiB`, the correctness-enabled Socket microbenchmark in job identifier `35` reported `13.08 GB/s` out-of-place bus bandwidth and `12.71 GB/s` in-place bus bandwidth. The OFI microbenchmark in job identifier `36` reported `32.20 GB/s` and `32.31 GB/s`, respectively. Both reported `Out of bounds values : 0 OK`. Logs selected `NET/Socket` for the initial configuration and `AWS Libfabric` with provider `efa`, GPU Direct RDMA and RDMA protocol for the corrected network path. The OFI tuner reported no dedicated g7e tuning profile and fell back to NCCL's tuner; successful execution does not establish platform tuning.

The host change reduced the requested worker population from ninety-six workers per node to sixteen workers per node, against twenty-four allocated CPU cores. Throughput did not increase from `v1` to `v2`, and the storage-path change in `v3` produced a similar steady-state result. `7.compute-metrics.py` returned `monotonic_increase_observed=false`. Thus the transport exercise demonstrated an improvement, while CPU saturation and a storage metadata bottleneck were not established by this rehearsal. No production A100 timing or monotonic progression is claimed. The initial successful module took longer than the proposed `180 s` per-training-command target; the subsequent three modules fit that target.

The Prometheus history was sampled every `5 s`. The following changes use the first and last samples inside each command interval. They include image/startup and other job-body activity, omit up to one sample interval at each boundary, and are not isolated training-kernel counters. Actual first/last timestamps, values, query strings and the derivation are in `pcs/aim347-training-counter-summary.json`, `pcs/prometheus-history/` and `/tmp/g7e-e2e/analyze-aim347.py`.

| g7e configuration | First / second node EFA transmit change | First / second node Lustre opens | First / second node Lustre getattr | First / second node Lustre reads |
| --- | --- | --- | --- | --- |
| `v0` | 0 B / 0 B | 1420 operations / 1389 operations | 43590 operations / 43434 operations | 27311455218 B / 27311443838 B |
| `v1` | 1380895249792 B / 1380895246592 B | 1406 operations / 1393 operations | 43707 operations / 43689 operations | 40965347384 B / 40965343273 B |
| `v2` | 1342050289984 B / 1342050286784 B | 453 operations / 426 operations | 9709 operations / 9563 operations | 27308811532 B / 27308799704 B |
| `v3` | 1342050289984 B / 1342050286784 B | 221 operations / 210 operations | 8056 operations / 8072 operations | 25712021845 B / 27308323642 B |

EFA receive changes mirrored the peer transmit changes, and retransmission-byte changes were `0 B` on both nodes throughout all four command windows. The zero transmit/receive changes in the Socket run were actual exposed counter values. Positive EFA changes in the corrected runs corroborate the loaded transport. Lustre collection succeeded on both clients. Fewer metadata operations in the packed-shard command are observable, but the similar training throughput does not establish that metadata dominated the measured steps. Startup reads from the shared SquashFS images are included in the Lustre byte totals.

The corresponding whole-command sampled rates for `v2` were `4.314286 operations/s` and `4.057143 operations/s` for opens, `92.466667 operations/s` and `91.076190 operations/s` for getattr, and `260083919.352381 B/s` and `260083806.704762 B/s` for reads. For `v3`, they were `2.210000 operations/s` and `2.100000 operations/s`, `80.560000 operations/s` and `80.720000 operations/s`, and `257120218.450000 B/s` and `273083236.420000 B/s`, respectively. These use sampled spans of `105 s` for `v2` and `100 s` for `v3`, retaining the whole-command scope described above.

| g7e configuration | First / second node mean user-plus-system CPU use | GPU tensor-activity sample range, dimensionless ratio |
| --- | --- | --- |
| `v0` | 6.884613 cores / 6.858655 cores | 0.000000 to 0.016277 |
| `v1` | 4.222859 cores / 4.270341 cores | 0.000000 to 0.045385 |
| `v2` | 2.884291 cores / 2.816509 cores | 0.000000 to 0.045789 |
| `v3` | 3.161352 cores / 3.157029 cores | 0.000000 to 0.045251 |

CPU values are command-window averages of host `rate(node_cpu_seconds_total[30s])` summed across CPUs, with user and system modes combined. They are below the allocated core count and cannot by themselves establish CPU saturation. Tensor ranges include idle/startup samples; valid positive profiling samples were present for all four GPUs during training. The original raw series also preserve SM and DRAM activity.

| g7e configuration | Training job-body allocation | Useful-token goodput |
| --- | --- | --- |
| `v0` | `0.243642225001 GPU-hours` | `840576.792465 tokens/GPU-hour` |
| `v1` | `0.129684919251 GPU-hours` | `1579212.148817 tokens/GPU-hour` |
| `v2` | `0.113590705130 GPU-hours` | `1802964.421833 tokens/GPU-hour` |
| `v3` | `0.110814251370 GPU-hours` | `1848137.739219 tokens/GPU-hour` |

The ledger includes the failed initial Socket attempt with zero useful-work credit and the successful retry under the same run identifier. It covers each training job body, including startup and its collective benchmark, and excludes scheduler polling, separate ceiling jobs and serving. Shorter job bodies and unequal cold-start costs explain why this accounting measure can improve while steady-state MFU does not. It is not whole-session goodput or a recovery measurement. Metric recomputation after `v0`, `v1`, `v2` and `v3` took `0.090453 s`, `0.091434 s`, `0.097976 s` and `0.097417 s`, respectively, each with exit code dimensionless value `0`.

Exact command argv, UTC timestamps, wall times and exit codes are retained in `/tmp/g7e-e2e/pcs/aim347-evidence-final/participant-modules/`. Full job logs, resource inventories, step samples and ledgers are in `/tmp/g7e-e2e/pcs/aim347-results-final/`, mirrored by the shared `results/participant-trial-a/` directory. The corrected ladder controller took `587.292499 s` from `16:50:15.244479 UTC` to `17:00:02.536969 UTC`, including the metric calls and model-byte calculation.

### AIM347 two-replica serving and weights-only MBU on PCS g7e

Job identifier `39` launched one vLLM version `0.20.2` replica per node through `8.serve-vllm.sh` and the real `lib/job.sh`, with tensor parallelism across both local GPUs. The platform image digest was `sha256:68b773151407ca28c05479c02c0c08a573285ba197c8ff35d2103acd97aff78c`; the SquashFS SHA-256 was `e12c99916a0cd378285b4f93594651192fb68b7825ac811fefbb48dca5fb0fd4`. vLLM logged runtime NCCL version `2.28.9`. It loaded `Qwen/Qwen2.5-3B` revision `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` using BF16, a context limit of `2048 tokens`, a dimensionless GPU-memory fraction of `0.8`, normal compilation and graph execution. No eager-mode override was used.

Both health probes returned HTTP status code `200` at `17:06:19 UTC`, before load generation. This is the recorded readiness check, not an exact engine-startup duration. The normal pass completed sixteen requests with concurrency of four requests, each producing `128 output tokens`. Its client wall time was `35.381788 s`, throughput was `57.882885 output tokens/s`, and median TTFT was `14.444335 ms`. The first four requests each waited approximately `32.496 s` to `33.375 s` for the first token. Those cold effects dominate the pass; its median alone does not describe that tail, and the two passes are not a calibrated concurrency comparison.

The serialized pass used `--concurrency 1 --measure-decode` with no other client traffic. Each replica completed eight requests, generated `1024 tokens` and contributed `1016 ITL samples`, exactly matching the client. The pair completed sixteen requests and `2048 output tokens`; the first token from each request was excluded from decode steps. The first replica's summed server decode duration was `3.552894640 s`; the second's was `3.546134395 s`.

| Recorded quantity | PCS g7e observation |
| --- | --- |
| Logical weight reads per decode step | `6172176384 B/decode-step` |
| Total modeled weight-read numerator | `12541862412288 B` |
| Sum of per-replica measured bandwidth × server decode time | `18929390484419.04 B` |
| Weights-only MBU | `0.662560288067` dimensionless ratio |
| Serialized client duration | `7.267624 s` |
| Serialized output throughput | `281.797731 output tokens/s` |
| Median serialized client TTFT | `10.285840 ms` |
| Mean server ITL | `3.493617 ms` |

The byte reader observed `6171877376 checkpoint weight bytes` and added `299008 logical read bytes/decode-step` for replicated normalization parameters at tensor-parallel size of two GPUs. The numerator is `6172176384 B/decode-step × 2032 decode steps`. The denominator uses the two local GPU bandwidth sums of `2666038880923.707 B/s` and `2666913935269.266 B/s` multiplied by their matching server decode durations. All four DRAM correctness checks passed. The weights-only model excludes KV-cache traffic, activations, input embedding lookups, padding and physical cache effects. It is neither measured physical DRAM utilization nor whole-allocation utilization.

| Serving module command | Command wall time | Exit code, dimensionless | Completion check |
| --- | --- | --- | --- |
| `python3 11.model-bytes.py` with the staged model and detected tensor-parallel size | `0.089039 s` | `0` | PASS: actual checkpoint inventory retained |
| `bash 8.serve-vllm.sh` | `426.089680 s` | `1` | PASS for startup and requests; this duration ends at intentional cancellation after evidence capture |
| First and second `curl -fsS .../health` | `0.012481 s` / `0.010075 s` | `0` / `0` | PASS: both HTTP status codes were `200` |
| `python3 9.load-serving.py` normal pass | `35.462767 s` | `0` | PASS: sixteen complete responses |
| `python3 9.load-serving.py --concurrency 1 --measure-decode` | `7.392726 s` | `0` | PASS: matching per-replica server/client counts |
| `python3 12.compute-mbu.py` with model bytes and bandwidth map | `0.065594 s` | `0` | PASS: matching numerator, denominator and serving metrics |

All nine Prometheus targets, including both vLLM endpoints, were healthy at `17:07:16 UTC`; all four training configuration groups remained queryable. The source is `aim347-serving-targets.log` and `pcs/aim347-evidence-final/prometheus-snapshots/serving-complete.json`. `scancel 39` was issued at `17:07:32 UTC` after load and telemetry capture. The waiting batch command returned its intentional-cancellation status at `17:07:38 UTC`; it was not a startup or load failure. Both nodes subsequently had no compute processes and no queued g7e jobs. The module records include the complete argv omitted from the compact table.

### Remaining AIM347 validation limits at 2026-09-09

Production `p4d.24xlarge` and `p4de.24xlarge` execution, the four-EFA/NVLink topology, A100 ceilings and serving results remain UNVALIDATED because no A100 allocation was used. CPU saturation, a storage metadata bottleneck, monotonic MFU improvement, same-AZ FSx performance, a common placement-group fabric baseline, calibrated sustained serving, cold-start timing across repeats and the complete participant session budget remain UNVALIDATED. Physical DRAM-counter MBU, concurrent/batched MBU, KV-cache MBU, other models/precisions, failure/recovery goodput and Nsight timelines were not measured. A real four-GPU-per-node or eight-GPU-per-node launch remains UNVALIDATED beyond software allocation tests. Workshop Studio deployment, unpublished-commit fetching, participant-role handoff and replacement-node automation were not executed in a vended workshop account.

## Spain g7.48xlarge as a g7.24xlarge resource stand-in, 2026-09-10

The physical allocation was two `g7.48xlarge` nodes in `eu-south-2a`, each with eight RTX PRO 4500 Blackwell Server Edition GPUs reporting `32623 MiB/GPU`, compute capability version `12.0` (SM120), `192 vCPUs`, `768 GiB RAM` and two EFA interfaces. Every result in the constrained PCS tables used `--gres=gpu:4 --ntasks-per-node=4 --cpus-per-task=24 --mem=384G` on each node, with `FI_EFA_IFACE=rdmap83s0`. Job logs contain the requested resources, actual GPU UUIDs, `96` available logical processors, CPU affinity indices `96-191` and a cgroup memory ceiling of `412316860416 B` (`384 GiB`). The GPUs were device indices `0-3` on both nodes. The second EFA interface, `rdmap176s0`, had zero byte-counter change. This enforces the resource budget; CPU/GPU NUMA placement, cache state and NVMe bandwidth still differ from a literal `g7.24xlarge` instance.

PCS cluster `pcs_lefqlz8o8e` used queue `gpu-g7`, Slurm version `25.05.9`, AMI identifier `ami-02165285ade1d0209`, Ubuntu version `24.04.4`, kernel version `6.17.0-1020-aws` and driver version `595.91.07`. The compute hosts were `gpu-g7-1` (`10.8.30.139`, instance identifier `i-0bbcee65f305fc7c6`) and `gpu-g7-2` (`10.8.17.100`, instance identifier `i-0af87d1628a5f761b`). PCS was configured with `ConstrainDevices=yes`, `ConstrainCores=yes`, `ConstrainRAMSpace=yes` and `SelectTypeParameters=CR_CPU_Memory`; GRES requests alone had not enforced memory. Existing NVMe was bind-mounted under the lab's `/opt` path and staged identically on both nodes. No FSx or EFS filesystem was created.

### Preparation, images and observability

The lab was staged under `/opt/aim347`, backed by existing NVMe. Compute node `gpu-g7-1` performed `1.prepare.sh` and hosted Prometheus version `3.5.0` on TCP port `9092`, Grafana version `12.1.1` on TCP port `3000`, and Pushgateway version `1.11.1` on private TCP port `9091`. The `m5.xlarge` login node did not build images or host monitoring. Preparation, pinned-image tests and upload took `957.820361 s` at the controller. Overlapping initial staging to the second node took `204.754627 s`; the final synchronization and checksum check took `35.982101 s`. Both nodes had `32768` token-record files and `16` shards. The dataset manifest SHA-256 was `9151e3e6456fe85b894fa497b82eafec55eb0b53ffc8916ac7a6f9eeab234bc4`.

The Spain ECR training digest was `sha256:cc952d627df9d4a8b306117f152bd8bf04156f72b7b7dc517cd46c78ee9a2e17`, with Torch version `2.9.1+cu130`, runtime NCCL version `2.30.4+cuda13.0`, aws-ofi-nccl version `1.19.0`, EFA installer version `1.48.0` and libfabric version `2.4.0`. The vLLM version `0.20.2` platform digest was `sha256:68b773151407ca28c05479c02c0c08a573285ba197c8ff35d2103acd97aff78c`. Their imported-image SHA-256 values were `4624e0ccf1e5dec07be1eea75eefcdeafa8646035591c13165dacec06b6f07cc` and `ea16c095d375012bc608d41eb7db8f8ff5f31e685b55c8b18021017fa5070d79`, identical on both nodes. All eleven tests passed in the pinned training image; local checks passed ten tests with one dependency-related skip, resolved by the container run. The new node-local result collector was exercised by both ceiling jobs.

Both compute exporter deployments succeeded. Their initial capture wrappers checked for `lustre.prom` before the first sample existed; later checks passed and the capture procedure now waits for that sample. DCGM exporter version `4.6.0-4.8.3` was configured with `DCGM_EXPORTER_DEVICES=g:0-3`, and its four UUIDs per node matched the job allocation. All seven pre-serving targets and all nine targets during serving were healthy. Grafana's provisioned dashboard and datasource were checked through its API. The Lustre collection-success metric was dimensionless value `0` throughout, with no Lustre byte or metadata series. Thus participant modules with identifiers `010` and `050` did not meet their Lustre-specific completion conditions. Healthy exporters do not supply missing filesystem measurements.

### Dense ceilings and complete training ladder

Job identifiers `6` and `7` ran GEMM and DRAM measurements, each with an allocated runtime of `11 s`. Their participant command wall times were `42.052893 s` and `42.068428 s`, including Slurm polling. GEMM used a square dimension of `8192 elements`, dense BF16 inputs, FP32 accumulation, ten warmup trials and thirty measured trials. Best/median rates were `171.702538 / 150.764417 TFLOP/s/GPU` on the first node and `171.923336 / 152.464990 TFLOP/s/GPU` on the second. The first best rate was used consistently as the denominator.

The DRAM probe read `1 GiB/trial` against a `64 MiB` L2 cache, with ten warmup trials and thirty measured trials. Every allocated GPU passed its reduction check.

| Measured GPU | Median read bandwidth | Correctness |
|---|---|---|
| Node index `0`, GPU index `0` | 662482967918.425 B/s | PASS |
| Node index `0`, GPU index `1` | 662273805061.624 B/s | PASS |
| Node index `0`, GPU index `2` | 662738097373.843 B/s | PASS |
| Node index `0`, GPU index `3` | 663059018436.784 B/s | PASS |
| Node index `1`, GPU index `0` | 662685827116.986 B/s | PASS |
| Node index `1`, GPU index `1` | 662653064891.405 B/s | PASS |
| Node index `1`, GPU index `2` | 663314591857.428 B/s | PASS |
| Node index `1`, GPU index `3` | 662666130390.636 B/s | PASS |

All configurations trained the full randomly initialized Qwen architecture with `2774773760 nonembedding parameters`, `512 tokens/sequence`, one sequence per GPU rank, `4096 tokens/step`, `100 update steps`, ten warmup steps, ninety measured steps and `2000 CPU hashing rounds/record`. Each completed `409600 useful tokens`. The common logical-data SHA-256 was `cbbf272e4c2f6ebf0a502f384425a4d5cbafcc83823ffe00dce86aa8babf4807`. Initial and reduced worker requests were `192 workers/node` and `32 workers/node`, against `96` available logical processors per node.

| Configuration and allocation | Command wall time | Training throughput | Mean measured step | MFU/GEMM, dimensionless ratio | Companion-ledger goodput |
|---|---|---|---|---|---|
| `v0`, job identifier `8` | 330.073130 s | 1715.936407 tokens/s | 2387.034848 ms | 0.020797604554 | 603953.105337 tokens/GPU-hour |
| `v1`, job identifier `9` | 170.063729 s | 5602.045734 tokens/s | 731.161471 ms | 0.067898280738 | 1257976.404918 tokens/GPU-hour |
| `v2`, job identifier `10` | 138.055873 s | 5713.913125 tokens/s | 716.846741 ms | 0.069254143198 | 1447655.655079 tokens/GPU-hour |
| `v3`, job identifier `11` | 138.051899 s | 5655.706739 tokens/s | 724.224255 ms | 0.068548666364 | 1429593.640045 tokens/GPU-hour |

All four jobs completed successfully with matching hardware, model, token workload and data fingerprints. NCCL microbenchmarks passed correctness. The initial run selected Socket and changed EFA RDMA counters by `0 B`. The corrected runs selected OFI's RDMA protocol and only `rdmap83s0`; the unused EFA remained unchanged. The first-node RDMA-write increases were `1595981916800 B`, `1565767184000 B` and `1565767184000 B` in the corrected command windows. These sampled windows include startup and microbenchmarks and can omit up to `5 s` at each boundary. Ring logs reported `GDR 0`, so no-host-staging transport is not established.

Throughput rose with the network and worker changes, then fell by about `1.02%` with packed shards. Metric recomputation returned `monotonic_increase_observed=false`. The mean busy logical-processor counts over the assigned CPU indices were about `12.03 / 11.98`, `7.89 / 7.87`, `5.06 / 4.92` and `5.04 / 5.03` on the two hosts across the four command windows. These host samples include startup and do not establish CPU saturation. GPU tensor, SM and DRAM profiling values were present for all eight GPUs; training framebuffer usage peaked at `14276 MiB/GPU`. With no Lustre series, there is no measured metadata-rate comparison or demonstrated storage metadata bottleneck. The ledger counts the training job body after initial allocation-evidence collection, including startup and its collective benchmark, and excludes scheduler polling, ceilings and serving.

### Serving, MBU and time budget

Job identifier `12` launched two vLLM replicas with tensor parallelism across four local GPUs each, BF16 weights, a context limit of `2048 tokens` and dimensionless GPU-memory fraction `0.8`. Serving used the pretrained `Qwen/Qwen2.5-3B` revision `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`, not the short training run's weights. Both endpoints became healthy `157.319 s` after submission. A shared-memory wait warning during graph profiling cleared without intervention. Each allocated GPU used `26511 MiB` at readiness; all four unused GPUs on each node used `0 MiB`.

The first load pass completed sixteen requests and `2048 output tokens` at concurrency of four requests in `40.475088 s`. Its throughput was `50.599026 output tokens/s` and median TTFT was `17.138386 ms`, but the first four requests had TTFT between `37.432 s` and `38.083 s`. The warm serialized pass completed the same counts in `8.841459 s`. Each replica's server counters matched eight requests, `1024 output tokens` and `1016 decode steps`. The cold and warm passes are not a calibrated concurrency comparison.

| Quantity | Spain observation |
|---|---|
| Checkpoint weight bytes | 6171877376 B |
| Logical weight reads per decode step at TP across four GPUs | 6248308736 B/decode-step |
| Total modeled weight-read numerator | 12696563351552 B |
| Measured bandwidth × server decode-time denominator | 22980498037640.95 B |
| Weights-only MBU | 0.552492958628 dimensionless ratio |
| Warm serialized output throughput | 231.635971 output tokens/s |
| Median warm serialized TTFT | 10.535029 ms |
| Mean server ITL | 4.266153 ms |

The weight model accounts for replicated normalization and KV-head parameters at this tensor-parallel size. It excludes KV-cache traffic, activations, input lookups, padding and physical cache effects. MBU is modeled weights-only traffic over measured active decode capacity, not a physical DRAM byte-counter measurement.

Ceilings, training, metric calls and the model-byte calculation took `864.768077 s` at the controller. Normal load, serialized decode and MBU commands took `40.534715 s`, `8.932377 s` and `0.049278 s`. The serving allocation ran for `299 s`, then was intentionally canceled after evidence capture; its waiting command took about `330 s` and returned exit code dimensionless value `1`. The complete machine participant sequence was about `20.3 minutes`, within the `120-minute` allocation, with cold preparation taking about `20 minutes` separately. Serving's allocation interval includes load and capture, so those times must not be added twice. Human-paced instruction, discussion and recovery time were not rehearsed.

The final queue was empty, both nodes had no GPU compute processes, and no named Enroot containers remained. Monitoring, dataset, weights and images remain staged. The original inactive update timers remain inactive; the native DCGM service is stopped in favor of the companion exporter. The AIM344 Prolog setting is absent and resource enforcement remains enabled. Raw per-node results, allocations, UUIDs, all module argv and exit records, server counters, monitoring histories and final state are under `/tmp/spain-rehearsal/phase2/aim347/`.

Literal g7.24xlarge execution, production A100 execution, shared-filesystem metadata attribution, CPU saturation, a monotonic ladder, repeated-run uncertainty, sustained serving, participant-role access, Workshop Studio deployment and reboot/replacement-node persistence remain UNVALIDATED. Four-GPU-per-node launch, native SM120 ceilings, the complete constrained training ladder and TP across four local GPUs are now observed on this physical G7 allocation.

## Dual-G7 qualification on Lustre, 2026-09-10

The full g7.48xlarge allocation used two exclusive nodes, eight GPUs, both EFAs, 192 vCPUs and all 747110 MiB of schedulable RAM per node. The g7.24xlarge-stand-in repeat constrained those same physical nodes to four GPUs, one selected EFA, 96 vCPUs and 384 GiB per node. Raw DMI labels correctly remain g7.48xlarge. Both runs detected the G7 family without a participant instance-type variable and selected the documented RDMA protocol. Literal secondary hardware remains unvalidated.

Both runs read the same shared FSx for Lustre dataset at `/mnt/aim347-lustre/data`. The token-manifest file SHA-256 is `9151e3e6456fe85b894fa497b82eafec55eb0b53ffc8916ac7a6f9eeab234bc4`; the training summaries record the token-content digest `cbbf272e4c2f6ebf0a502f384425a4d5cbafcc83823ffe00dce86aa8babf4807`. The earlier NVMe-only stand-in had no Lustre evidence and failed the storage checks. These new Lustre runs close that specific gap without changing the historical record.

| Measured quantity | g7.48xlarge full | g7.24xlarge-stand-in |
|---|---|---|
| Best dense BF16 GEMM, node indices 0 / 1 | 170.284864 TFLOP/s/GPU / 169.958041 TFLOP/s/GPU | 169.630811 TFLOP/s/GPU / 170.017754 TFLOP/s/GPU |
| Measured DRAM range | 662.280330 to 663.662281 GB/s/GPU; 16 correctness passes | 661.783954 to 662.679245 GB/s/GPU; 8 correctness passes |
| Configuration v0: throughput; MFU; step duration | 3663.649623 tokens/s; 0.022387043 dimensionless; 2236.021684 ms | 1674.887242 tokens/s; 0.020548006 dimensionless; 2445.537763 ms |
| Configuration v1: throughput; MFU; step duration | 15194.399857 tokens/s; 0.092846674 dimensionless; 539.146006 ms | 5686.552654 tokens/s; 0.069764288 dimensionless; 720.295801 ms |
| Configuration v2: throughput; MFU; step duration | 15152.385589 tokens/s; 0.092589942 dimensionless; 540.640941 ms | 5649.058068 tokens/s; 0.069304294 dimensionless; 725.076632 ms |
| Configuration v3: throughput; MFU; step duration | 15167.977144 tokens/s; 0.092685216 dimensionless; 540.085202 ms | 5682.618397 tokens/s; 0.069716022 dimensionless; 720.794485 ms |
| Serialized serving throughput | 232.208423 output tokens/s | 232.508270 output tokens/s |
| Median client TTFT | 11.278417 ms | 10.560060 ms |
| Mean server ITL | 4.249024 ms/output token | 4.248957 ms/output token |
| Weights-only MBU | 0.284036 dimensionless | 0.555166 dimensionless |
| Modeled weight reads | 13005965230080.000000 B | 12696563351552.000000 B |
| Measured decode capacity | 45789788696895.617188 B | 22869834945928.429688 B |
| Complete machine interval, GEMM submission through serving cancellation | 1130.581692 s (18.843 minutes) | 1095.147185 s (18.252 minutes) |

The network change improved throughput in both allocations. Reducing workers slightly reduced throughput; packed shards recovered only a small amount. Neither sequence was monotonic. Each shape used its own measured GEMM denominator throughout. The serialized serving pass completed all 16 requests per shape and preserved the weights-only numerator and denominator. Similar serving throughput with more aggregate measured bandwidth lowers full-shape MBU; it does not establish worse physical DRAM utilization.

| Allocation | Module | Command wall time | Exit status, dimensionless |
|---|---|---|---|
| g7.48xlarge | 020-gemm | 42.050070 s | 0 |
| g7.48xlarge | 020-bandwidth | 42.067664 s | 0 |
| g7.48xlarge | 020-v0 | 330.076976 s | 0 |
| g7.48xlarge | 020-metrics | 0.071141 s | 0 |
| g7.48xlarge | 030-v1 | 170.060434 s | 0 |
| g7.48xlarge | 030-metrics | 0.074338 s | 0 |
| g7.48xlarge | 040-v2 | 138.054201 s | 0 |
| g7.48xlarge | 040-metrics | 0.072543 s | 0 |
| g7.48xlarge | 050-v3 | 138.054450 s | 0 |
| g7.48xlarge | 050-metrics | 0.080404 s | 0 |
| g7.48xlarge | 060-model-bytes | 0.076094 s | 0 |
| g7.48xlarge | 060-serving-allocation | 266.065116 s | 1 |
| g7.48xlarge | 060-health-first | 0.006201 s | 0 |
| g7.48xlarge | 060-health-second | 0.004844 s | 0 |
| g7.48xlarge | 060-serving-load | 41.051278 s | 0 |
| g7.48xlarge | 060-serving-decode | 8.909195 s | 0 |
| g7.48xlarge | 060-serving-mbu | 0.049991 s | 0 |
| g7.24xlarge-stand-in | 020-gemm | 42.051091 s | 0 |
| g7.24xlarge-stand-in | 020-bandwidth | 42.067175 s | 0 |
| g7.24xlarge-stand-in | 020-v0 | 330.073077 s | 0 |
| g7.24xlarge-stand-in | 020-metrics | 0.070202 s | 0 |
| g7.24xlarge-stand-in | 030-v1 | 170.062440 s | 0 |
| g7.24xlarge-stand-in | 030-metrics | 0.073341 s | 0 |
| g7.24xlarge-stand-in | 040-v2 | 138.055173 s | 0 |
| g7.24xlarge-stand-in | 040-metrics | 0.074592 s | 0 |
| g7.24xlarge-stand-in | 050-v3 | 138.054164 s | 0 |
| g7.24xlarge-stand-in | 050-metrics | 0.075806 s | 0 |
| g7.24xlarge-stand-in | 060-model-bytes | 0.076819 s | 0 |
| g7.24xlarge-stand-in | 060-serving-allocation | 234.068775 s | 1 |
| g7.24xlarge-stand-in | 060-health-first | 0.006440 s | 0 |
| g7.24xlarge-stand-in | 060-health-second | 0.005049 s | 0 |
| g7.24xlarge-stand-in | 060-serving-load | 40.551945 s | 0 |
| g7.24xlarge-stand-in | 060-serving-decode | 8.901738 s | 0 |
| g7.24xlarge-stand-in | 060-serving-mbu | 0.048718 s | 0 |

The serving-allocation command overlaps its health/load/MBU commands, so those durations must not be added twice. Its nonzero exit is the deliberate cancellation after capture. Startup to both healthy replicas was g7.48xlarge: 190.245355 s; g7.24xlarge-stand-in: 170.188239 s. Both complete machine intervals fit the 120-minute budget. Cold preparation and human-paced instruction are outside these intervals.

| Allocation | Configuration | Node | Mean open rate | Mean getattr rate | Mean read rate |
|---|---|---|---|---|---|
| g7.48xlarge | 040-v2 | gpu-g7-1 | 6.171429 operations/s | 42.928571 operations/s | 12560.971429 B/s |
| g7.48xlarge | 040-v2 | gpu-g7-2 | 6.411429 operations/s | 44.608571 operations/s | 13053.451429 B/s |
| g7.48xlarge | 050-v3 | gpu-g7-1 | 1.028571 operations/s | 6.928571 operations/s | 1069.771429 B/s |
| g7.48xlarge | 050-v3 | gpu-g7-2 | 1.085714 operations/s | 7.328571 operations/s | 1187.028571 B/s |
| g7.24xlarge-stand-in | 040-v2 | gpu-g7-1 | 3.177143 operations/s | 22.111429 operations/s | 6462.234286 B/s |
| g7.24xlarge-stand-in | 040-v2 | gpu-g7-2 | 3.028571 operations/s | 21.071429 operations/s | 6163.228571 B/s |
| g7.24xlarge-stand-in | 050-v3 | gpu-g7-1 | 0.554286 operations/s | 3.751429 operations/s | 622.828571 B/s |
| g7.24xlarge-stand-in | 050-v3 | gpu-g7-2 | 0.600000 operations/s | 4.071429 operations/s | 710.771429 B/s |

Both collector-success flags stayed at dimensionless value 1 through the storage windows. These are means of 30-second rates sampled every 5 seconds over complete command intervals, including startup. Metadata operations fell with packed shards, but the small throughput difference and low observed host activity do not establish a saturated metadata or CPU bottleneck. Raw ranges retain GPU tensor/SM/DRAM activity, CPU activity, EFA per-device traffic and retransmission fields for each module.

The full health matrix was performed before this ladder using the unmodified AIM344 health image. The retained upstream profile skips expected-count checks on G7; the candidate patch remains unapplied. Full and stand-in logs select Socket for v0 and AWS Libfabric for v1 through v3. The correctness-enabled microbenchmarks run with v0 and v1 and report zero out-of-bounds values. DRAM correctness records and endpoint/GPU mappings cover every allocated GPU. All executable module completion checks passed on the new Lustre runs; participant discussion and interpretation were reviewed against the evidence rather than measured with attendees.

Raw results, module commands, exit codes, allocations and Prometheus ranges are retained under `/tmp/g7-dual-instance/pcs/aim347/` and the private Spain evidence prefix `g7-dual/aim347/`.
