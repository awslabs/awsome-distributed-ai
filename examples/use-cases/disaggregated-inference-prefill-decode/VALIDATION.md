# Validation evidence

The historical hardware observations in the first sections were collected on 2026-09-06 in `ap-northeast-2`. The final section records new Oregon checks from 2026-09-09. They validate a transport mechanism on Seoul `p6-b300.48xlarge`, not the production instance choice or the session's expected latency crossover.

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

## Paired resident endpoints on Oregon g7e: VALIDATED, 2026-09-09

Both `g7e.12xlarge` nodes in the shared Oregon EKS cluster supplied two GPUs and one EFA device per stack. The unified stack used `ip-10-3-132-239.us-west-2.compute.internal`; disaggregation used `ip-10-3-133-250.us-west-2.compute.internal`. The stacks occupied separate namespaces, `aim345-pi-20260909-unified` and `aim345-pi-20260909-disaggregated`, each with its own nonpreempting PriorityClass. Packed placement ran two engine processes in one pod per stack, with one GPU per process and disjoint visible-device lists. Each pod requested 8 CPU cores, 128 GiB memory and one EFA device. No other team's workload or namespace was modified.

The engine used ECR manifest-list digest `sha256:772d52067fab28c9eea5fe1fd629694218b4aa1669b257c43e689abdcca59338` from repository `aim-content-decisions-20260909` in `us-west-2`. The image was built from this asset's Dockerfile, with SGLang version `0.5.12.post1`, NIXL version `1.1.0`, and the new packed-process launcher. Package versions were inspected in the live engines. The model was DeepSeek-V2-Lite-Chat revision `85864749cd611b4353ce1decdb286193298f64c7`. The controller used the same `paired.py`, traffic generators, ramp, collector and transport verifier shipped here.

`paired.py verify` returned HTTP status code 200 for both endpoints before and after all traffic passes. The before/after records were exactly equal: unified engine pod UID `ec7c6154-271d-457b-9f42-ff8f1fbf1c02` and disaggregated engine pod UID `1f1b6f7b-2ec6-4fc6-9343-858d0a4a26ed`, the same node placement, and two requested GPUs per stack. Both engine pods retained zero restarts. No stack was redeployed to measure the other.

Each shape ran at 0.1 offered tasks/s for a 10-second offered window on each endpoint, with the harness's separate warmup pass and a measured pass. Shape A completed nine sequential calls per measured task; shape B completed one call with 8192 input tokens. Every call requested and produced 256 output tokens. These short runs establish execution and residency, not capacity or a sustained ranking.

| Stack and shape | Completed calls | Failed/skipped calls | p90 TTFT | p90 average TPOT | Useful calls/s/GPU | Joint attainment |
|---|---|---|---|---|---|---|
| Unified A | 9 calls | 0 calls | 455.316 ms | 5.673 ms | 0.289814 calls/s/GPU | 100 percent |
| Disaggregated A | 9 calls | 0 calls | 715.086 ms | 5.701 ms | 0.269568 calls/s/GPU | 100 percent |
| Unified B | 1 call | 0 calls | 779.202 ms | 5.748 ms | 0.050000 calls/s/GPU | 100 percent |
| Disaggregated B | 1 call | 0 calls | 773.852 ms | 5.770 ms | 0.050000 calls/s/GPU | 100 percent |

The collector retained logs, pod identities, image IDs, both packed engine ports' metrics and server configuration, and the combined CSV. Source inspection and live `/proc` reads confirmed the NIXL CLI selector, LIBFABRIC environment selector and EFA provider in both disaggregated engine processes. A verifier request with 1024 input tokens and four output tokens completed successfully. Its EFA RDMA byte-counter increase was 0 bytes. This validates the packed serving path and live selector inspection; **cross-node KV transfer over EFA is UNVALIDATED in this topology**. The two engine processes share a node. The historical Seoul cross-node observations above retain their original hardware and date.

The first packed routers could not schedule on the shared system nodes because of insufficient CPU, while the assigned GPU nodes were tainted. The renderer now pins each packed router to its own assigned node with the same tolerations as its engines. Only these new router Deployments were updated. Both then became ready. After the hardware run, the renderer also included explicit Namespace objects in its saved manifests; this serialization addition was checked by CPU tests, not another deployment.

The combined CPU suite passed eight tests for the existing traffic/accounting path and the paired renderer, including separate namespaces and PriorityClasses, disjoint GPU indices, headless-service ports, node pinning, mismatched-budget rejection and the preserved separate-node renderer. CPU checks do not validate the paired separate-node serving topology. The full offered-rate sweep, crossover, startup-priming effect on packed engines, larger allocations and paired cross-node engine placement remain UNVALIDATED in this tab.

`paired.py cleanup` deleted both new namespaces and their PriorityClasses. Model caches remain only under the documented `/mnt/aim345-models` path on the assigned nodes. The local report at `/tmp/companion-decisions-report.md` records exact commands, image digests and evidence paths.
