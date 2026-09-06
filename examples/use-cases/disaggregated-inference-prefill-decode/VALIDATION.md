# Validation evidence

The hardware observations below were collected on 2026-09-06 in `ap-northeast-2`. They validate a transport mechanism on Seoul `p6-b300.48xlarge`, not the production instance choice or the session's expected latency crossover.

## NIXL GPU-buffer transfer: VALIDATED

Two existing `p6-b300.48xlarge` bench nodes each supplied one GPU and one EFA device to an isolated test pod. The CUDA buffer size was 64 MiB per transfer, with an iteration count of 20 transfers per run. `fabric_probe.py` registered CUDA tensors, exchanged NIXL metadata, called `make_connection` before the timed loop, issued writes through LIBFABRIC, and checked every destination byte on the receiving GPU.

| Observation | Value | Hardware and scope |
|---|---|---|
| GPU model | NVIDIA B300 SXM6 AC | `p6-b300.48xlarge`, one GPU per peer |
| NVIDIA driver version | `595.91.07` | Same Seoul nodes |
| Installed serving package | SGLang version `0.5.12.post1` | Inspected inside the actual GPU pod |
| Installed transfer packages | `nixl` and `nixl-cu13`, both version `1.1.0` | Inspected distributions, not inferred from the image tag |
| Engine image used for the probe | Digest `sha256:8a77f87f2e56fd2e462b983dc957c03e952227c16a06da4929574cf874d3d6ae` | Built from the pinned engine base and EFA installer |
| Destination verification | Passed | CUDA tensor contents matched on the remote GPU |
| Sending EFA RDMA-write counter increase | 1,342,177,280 bytes | Exactly the transferred payload volume in the counter-observed run on `p6-b300.48xlarge` |
| Warm transfer bandwidth, initial run | 25.763 GiB/s | `p6-b300.48xlarge`; single GPU buffer, Python polling included |
| Warm transfer bandwidth, counter-observed repeat | 25.441 GiB/s | Same hardware and transfer settings; not aggregate instance bandwidth |

The observed counter increase equals 20 transfers × 64 MiB per transfer. The logs reported `Backend LIBFABRIC was instantiated`, `cuda dmabuf support status: 1`, EFA device selection, and `use_device_rdma=1`. Together with CUDA tensor registration and remote content verification, these establish the probe's GPU-buffer EFA path. They do not establish SGLang request scheduling, the outer transfer selector's negative controls, or a full-model KV transfer.

The probe used the same installed serving and NIXL libraries as the engine image. The diagnostic helper was copied into the running test pods and did not modify those libraries. The final Dockerfile includes that helper directly.

## SGLang positive transport path: VALIDATED

The same Seoul `p6-b300.48xlarge` pair served a streamed disaggregated request using the pinned DeepSeek-V2-Lite-Chat revision, one GPU per engine, and one EFA device per engine. Both live engine configurations included `--disaggregation-transfer-backend nixl` and `SGLANG_DISAGGREGATION_NIXL_BACKEND=LIBFABRIC`, with `FI_PROVIDER=efa` and no Mooncake-only device flag. The request used 1024 input tokens and produced 4 output tokens. It succeeded, and EFA RDMA-write counters increased by 31,852,096 bytes. These are mechanism-validation observations on `p6-b300.48xlarge`, not production session numbers.

The first-token latency of that request was 5311.262 ms on `p6-b300.48xlarge`. It was a cold smoke request and is not a steady-state performance result or an isolated handshake measurement. The logical BF16 MLA cache estimate was 31,850,496 bytes for this model and prompt. Using the single-GPU fabric probe's measured bandwidth gives an ideal wire-time estimate of 1.166 ms on that same hardware, subject to the probe's polling and layout limitations. The difference does not by itself attribute the remaining latency to NIXL; model execution, JIT compilation, and scheduling remain confounders.

Short disaggregated smoke runs also completed both traffic shapes on this `p6-b300.48xlarge` pair. Shape A completed 9 sequential calls in each warmup and measured task, with 256 output tokens per call and a minimum observed adjacent-prompt overlap of 76.762 percent. Shape B completed one call per phase with 8192 input tokens and 256 output tokens. The offered window was 1 second at 0.1 tasks/s; that yielded one task and is too short to establish rate capacity or a crossover. The harness drained the calls and the collector saved engine logs, metrics, pod identities, and CSV summaries.

The matched unified deployment also completed both shapes with the same input and output settings on the same `p6-b300.48xlarge` pair and GPU count. Both its warmup and measured agentic tasks completed 9 calls; each long-context phase completed one call. The first readiness settings used a timeout of 1 second, which intermittently marked functioning engines unready during health checks. The renderer now uses a timeout of 10 seconds for engine startup and readiness probes.

The startup-priming helper also completed its requests and cache flushes on the same hardware. Its latency benefit remains UNVALIDATED; the smoke run did not isolate connection setup from kernel warmup.

## Prefill scaling operation: VALIDATED

The scaling script added a prefill worker on a third existing Seoul `p6-b300.48xlarge` bench node, each worker using one GPU and one EFA device. The total GPU count increased from two GPUs to three GPUs. The decode pod UID remained unchanged and its restart count remained zero restarts. All workers became Ready, startup requests completed through the expanded router pool, and shape B completed one warmup call and one measured call with 8192 input tokens and 256 output tokens per call. The new prefill worker's log recorded an 8192-token prefill.

This validates adding prefill capacity and routing requests through the expanded pool. SLO recovery remains UNVALIDATED because the short smoke runs did not establish a prefill bottleneck or a failing objective before scaling.

## Remaining serving and measurement status

The installed source confirms that the outer SGLang selector defaults to Mooncake and the NIXL selector defaults to UCX. It also confirms no explicit `make_connection` or `makeConnection` call in the pinned SGLang NIXL connector. The hardware selector negative controls, eager SGLang bootstrap integration, agentic/unified TTFT ranking, load-collapse crossover, and prefill-only recovery remain **UNVALIDATED** until their own observed results are recorded.

CPU tests passed for agentic continuation semantics, input-token budgets, distinct long-context prefixes across warmup and measurement, joint SLO accounting including failures, streaming truncation handling, and equal-GPU deployment generation. A local mock also exercised the ramp and CSV collector. Mock timings are not inference measurements.

## Deployment findings

The first EKS submission exposed a PriorityClass admission requirement for `preemptionPolicy: Never`. The renderer now creates a dedicated nonpreempting PriorityClass and assigns it to its pods.

A subsequent attempt used the serving image for the CPU router. Pulling and unpacking it exhausted the small image filesystem on system nodes of type `m5.xlarge`; the router was evicted. The test Deployments were scaled to zero, and filesystem reads subsequently showed free space again. The final configuration gives the router a separate small image with pinned dependencies and adds explicit ephemeral-storage requests to both pod types. It does not change system-node storage or delete other workloads.

The EC2 capacity reservation had 32 occupied `p6-b300.48xlarge` instance seats and 0 free instance seats at the check. Normal scheduling nevertheless exposed idle GPUs on existing bench nodes. The probe used those already-running nodes without launching instances, scaling node groups, or preempting other workloads. Future scheduling availability must be checked again.
