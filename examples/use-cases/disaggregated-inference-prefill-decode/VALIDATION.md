# Validation evidence

The historical hardware observations in the first sections were collected on 2026-09-06 in `ap-northeast-2`. Later sections record Oregon checks from 2026-09-09 and Spain checks from 2026-09-10. They validate a transport mechanism on Seoul `p6-b300.48xlarge`, not the production instance choice or the session's expected latency crossover.

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

The installed source confirms that the outer SGLang selector defaults to Mooncake and the NIXL selector defaults to UCX. It also confirms no explicit `make_connection` or `makeConnection` call in the pinned SGLang NIXL connector. The control omitting both selectors, eager SGLang bootstrap integration, agentic/unified TTFT ranking, load-collapse crossover, and prefill-only recovery remain **UNVALIDATED** until their own observed results are recorded.

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

## Cross-node Oregon g7e transport, 2026-09-09

The PI-authorized rehearsal used the existing EKS nodes `ip-10-3-132-239.us-west-2.compute.internal` and `ip-10-3-133-250.us-west-2.compute.internal` in `us-west-2d`, each a `g7e.12xlarge` with NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs. One disaggregated stack allocated one GPU and one EFA device per worker, with prefill and decode on different nodes, in namespace `aim345-xnode-20260909`. Each engine requested 16 CPU cores and 128 GiB memory. The engine digest was `sha256:772d52067fab28c9eea5fe1fd629694218b4aa1669b257c43e689abdcca59338` and the router digest was `sha256:670c7f0004e2d068c7219888109362980195e554fb66480955d985dd9c9b52f2`, both in account `159553542841` repository `aim-content-decisions-20260909`. SGLang version `0.5.12.post1` and both NIXL distributions at version `1.1.0` were verified in the running workers. The model revision and node cache matched the earlier packed run.

`7.verify-transport.py --config /tmp/g7e-e2e/phase-a/config.json --url http://127.0.0.1:8011 --output /tmp/g7e-e2e/phase-a/transport.json` completed a streamed request with 1024 input tokens and four output tokens. The cold request observed TTFT of 4670.617346 ms and average TPOT of 4.765288 ms. Both engine processes' `/proc/1/cmdline` and `/proc/1/environ` contained the `nixl` CLI selection, `SGLANG_DISAGGREGATION_NIXL_BACKEND=LIBFABRIC` and `FI_PROVIDER=efa`. Both logs recorded `Backend LIBFABRIC was instantiated`, `cuda dmabuf support status: 1` and `use_device_rdma=1`.

Counters were read at `/sys/class/infiniband/rdmap49s0/ports/1/hw_counters/` on both nodes. The following values surround that single request; each value is in bytes.

| Worker and counter | Before | After | Delta |
|---|---|---|---|
| Prefill `rdma_write_bytes` | 67541760 bytes | 99393856 bytes | 31852096 bytes |
| Decode `rdma_write_bytes` | 67296000 bytes | 67296000 bytes | 0 bytes |
| Prefill `rdma_read_bytes` | 193370880 bytes | 193370880 bytes | 0 bytes |
| Decode `rdma_read_bytes` | 130210560 bytes | 130210560 bytes | 0 bytes |
| Prefill `send_bytes` / decode `recv_bytes` | 7683136 bytes / 4495936 bytes | 7683427 bytes / 4496227 bytes | 291 bytes each |
| Decode `send_bytes` / prefill `recv_bytes` | 8910624 bytes / 10269520 bytes | 8910664 bytes / 10269560 bytes | 40 bytes each |

The cached model configuration specifies 27 layers, a KV latent dimension of 512 elements and a rotary-key dimension of 64 elements. The logical BF16 MLA estimate is 1024 tokens × 27 layers × (512 elements + 64 elements) × 2 bytes per element = 31850496 bytes. The sending RDMA-write delta exceeds that estimate by 1600 bytes. This establishes the configured cross-node KV/EFA path with byte volume consistent with the model; the extra transfer layout bytes were not separately attributed. The decode-side outgoing RDMA counters remained flat. Counter families are kept separate rather than added into a wire-byte total.

The companion traffic generators, `4.ramp.py` and `5.collect.py` ran each shape once with a separate warmup, 0.1 offered tasks/s and a 10-second offered window, using two GPUs per stack. The comparison below includes the earlier packed observations from this file. All measurements are on `g7e.12xlarge` in Oregon.

| Placement, stack and shape | Completed / failed calls | p90 TTFT | p90 average TPOT | Useful throughput | Joint attainment |
|---|---|---|---|---|---|
| Packed unified A, earlier run | 9 calls / 0 calls | 455.316 ms | 5.673 ms | 0.289814 calls/s/GPU | 100 percent |
| Packed disaggregated A, earlier run | 9 calls / 0 calls | 715.086 ms | 5.701 ms | 0.269568 calls/s/GPU | 100 percent |
| Cross-node disaggregated A | 9 calls / 0 calls | 285.963897 ms | 5.703108 ms | 0.299581 calls/s/GPU | 100 percent |
| Packed unified B, earlier run | 1 call / 0 calls | 779.202 ms | 5.748 ms | 0.050000 calls/s/GPU | 100 percent |
| Packed disaggregated B, earlier run | 1 call / 0 calls | 773.852 ms | 5.770 ms | 0.050000 calls/s/GPU | 100 percent |
| Cross-node disaggregated B | 1 call / 0 calls | 772.553072 ms | 5.779861 ms | 0.050000 calls/s/GPU | 100 percent |

These observations establish execution on the separate-node topology. The cross-node run used separate engine pods with larger per-engine CPU requests than the earlier packed pod and ran later. It is not a controlled estimate of placement's latency effect, a repeated ranking or a sustained-capacity result.

### Hardware selector controls: observed failures

The installed source at `/sgl-workspace/sglang/python/sglang/srt/server_args.py:798` defines `disaggregation_transfer_backend: str = "mooncake"`. `/sgl-workspace/sglang/python/sglang/srt/environ.py:243` defines `SGLANG_DISAGGREGATION_NIXL_BACKEND = EnvStr("UCX")`; `srt/disaggregation/nixl/conn.py:234` reads that variable. Each control retained the same image, model, allocation and `FI_PROVIDER=efa`, changing only the specified selector in both worker Deployments.

| Control on Oregon g7e | Engine startup | Streamed request | EFA counters and observed log |
|---|---|---|---|
| Omit CLI backend; retain `LIBFABRIC` environment | Both engines became ready with Mooncake | Did not complete before the 90-second client deadline | All observed counter deltas were 0 bytes on both nodes. Prefill logged `Failed to create QP: Operation not supported [95]`, then `Fatal Python error: Segmentation fault`. |
| Retain CLI `nixl`; omit NIXL environment selector | Both engines became ready; both logged `Backend UCX was instantiated` | Did not complete before the 90-second client deadline | All observed counter deltas were 0 bytes on both nodes. UCX tried `169.254.170.23`, reported `Destination is unreachable`, and raised `NIXL_ERR_BACKEND` while loading remote metadata. |

The UCX observation is the default interface-selection behavior of this EKS environment. It does not establish that UCX cannot serve with a separately configured routable interface. Before the recorded UCX engine request, the router rejected an attempt with HTTP status code 503 and `No available prefill workers (all circuits open or unhealthy)`. Restarting only the owned router and verifying both entries in `/workers` as healthy allowed the engine probe. A stale local port forward caused another controller connection failure before the engine probe. Both discarded attempts are retained separately. The initial Mooncake harness also had counter-mount and working-directory defects before any request was sent; its empty counter snapshots are excluded. The corrected independent counter pods mounted host `/sys` read-only at `/host-sys` and retained the per-device paths even if an engine failed.

Exact manifests, request records, command transcripts, source reads, provider logs and unmodified collector output are under `/tmp/g7e-e2e/phase-a/`, with the consolidated report at `/tmp/g7e-e2e-report.md`. The positive path and these two selector controls are now observed on cross-node `g7e.12xlarge`. The control omitting both selectors, routable-interface UCX tuning, an independent g7e CUDA-buffer probe, eager SGLang bootstrap, sustained rate calibration, repeated crossover measurements, and a simultaneous pair of separate-node stacks remain UNVALIDATED.

## Facilitator helper review, 2026-09-09

The new `facilitator/prepare-config.py` was run with `--inventory /tmp/g7e-e2e/phase-a/nodes-before.json` against the saved real EKS node List, the Phase A context and immutable image digests, and `--output /tmp/g7e-e2e/runbook-aim345-configs`. It generated matched packed configurations for the two explicitly assigned g7e nodes, with two GPUs and one EFA per stack. `check_pair` accepted the result. Output is retained in `/tmp/g7e-e2e/runbook-aim345-configs.log`. The controller environment's eight existing tests passed. This establishes offline configuration generation only; the facilitator runbook and its new Workshop Studio CodeBuild phase were not redeployed after the GPUs moved to PCS.

## Spain g7.48xlarge feasibility and constrained paired rehearsal, 2026-09-10

The two EKS nodes in `eu-south-2a` were physical `g7.48xlarge` instances with eight RTX PRO 4500 Blackwell Server Edition GPUs and two EFA interfaces each. Every GPU reported `32623 MiB`, or `31.8583984375 GiB`, and compute capability version `12.0` (SM120). Kubernetes was version `1.35.7-eks-cb19647`, AMI identifier `ami-039041ed3d883ce32`, AL2023 release `2023.12.20260831`, kernel version `6.12.103-127.188.amzn2023.x86_64` and containerd version `2.2.5`.

The inherited proprietary driver version `580.178.04` could not operate the GPUs. Installing and building the open driver version `595.91.07` restored all devices without a reboot or engine-source patch. NVIDIA device-plugin chart version `0.20.0` and EFA device-plugin chart version `v0.5.31` (application version `v0.5.21-eksbuild.9`) then exposed eight GPUs and two EFAs per node. The existing NVMe mount `/mnt/k8s-disks/0` held the weight cache; no model weights were put on the root disk.

### DeepSeek-V4-Flash memory and engine verdict

HF revision `60d8d70770c6776ff598c94bb586a859a38244f1` contains `46` weight shards totaling `159617149040 B`, or `148.655054197 GiB`. Its configuration has `43` layers, a hidden dimension of `4096 elements`, `256` routed experts, `6` active experts per token, FP4 expert weights and FP8 non-expert quantization. The model card reports `284 billion total parameters` and `13 billion active parameters`; the configured context limit is `1048576 tokens`.

Four actual GPUs provide `127.43359375 GiB`, leaving a weight-only deficit of `21.221460447 GiB` before KV cache or runtime memory. A V4 prefill or decode stack therefore cannot fit on one constrained node. A single cross-node unified stack over the eight stand-in GPUs is memory-plausible, but was not tested and would leave no GPUs for a second stack. Eight GPUs on one full physical node provide `254.8671875 GiB`, with nominal weights of `18.581881775 GiB/GPU` and `13.276516663 GiB/GPU` remaining before runtime allocations.

The following tests used all eight GPUs on one node, one EFA, `96 vCPUs` and `384 GiB RAM`. They are full-node GPU tests, outside the four-GPU stand-in. Each successful cold smoke request contained `1024 input tokens` and produced `32 output tokens`.

| Engine version and immutable amd64 digest | Context | Result | Ready GPU memory | Cold TTFT | Cold TPOT |
|---|---|---|---|---|---|
| SGLang `0.5.12.post1`, `sha256:0b9ebdd8fbb659500a4abfe8541049923dc768088e0e36cf8bffe8f3728f6f9a` | 32768 tokens | Post-load scale setup failed | No ready measurement | Not served | Not served |
| SGLang `0.5.19`, `sha256:37bbbd3444732a464bbc68dee4fb0164e0ce9e18e2f027f3fc967f1152d3c262` | 32768 tokens | Served | 29625 MiB/GPU | 11795.476 ms | 11.128198 ms/token |
| vLLM `0.29.0`, `sha256:082ca6f035279109041ffd3fe0695cb568b29bc580b35c4f297a66a08b216c1b` | 32768 tokens | Weights loaded; KV capacity rejected | No ready measurement | Not served | Not served |
| Same vLLM digest and version | 8192 tokens | Served | 28039 MiB/GPU | 18716.741 ms | 9.500323 ms/token |

The pinned SGLang failure was `tvm.error.InternalError: Assertion error (/deepgemm/csrc/apis/layout.hpp:59): Unknown SF transformation`. Installed Python call sites were `sglang/srt/models/deepseek_v4.py:1277` and `:1288`, and `deep_gemm/__init__.py:245`. The C++ location is the compiled exception's path. The vLLM failure at installed `vllm/v1/core/kv_cache_utils.py:879` required `6.36 GiB` of KV memory while `5.27 GiB` was available; it was a memory-capacity rejection, not an SM120 architecture assertion. The smaller context succeeded without source changes. Full source excerpts and engine logs are in `/tmp/spain-rehearsal/phase1/`.

An experimental image derived from stock SGLang version `0.5.19`, with NIXL version `1.4.1` and EFA installer version `1.47.0`, served V4 disaggregated on both full physical nodes: prefill across eight GPUs and decode across eight GPUs, with one EFA, `96 vCPUs` and `384 GiB RAM` per node. Its Spain ECR digest is `sha256:85db181b224fe99c8ef474953bd6ef798c68e331dfcd2b4b62dccaac3b3545fe`. The companion's strict `7.verify-transport.py` failed its old package-pin assertion; a separately recorded version-aware request succeeded. A `1024`-input-token, `4`-output-token request increased prefill RDMA-write bytes by `97235456 B` on only one EFA. Shape A completed `9` calls with p90 TTFT `717.771 ms`; shape B completed one `8192`-input-token, `256`-output-token call with p90 TTFT `1797.577 ms`. Both had dimensionless joint attainment `1.0` at `0.1 tasks/s` with a `10 s` offered window. Workload traffic added `5388893184 RDMA-write bytes`. An allocator warning during shape B warmup recovered; all recorded calls completed. This is experimental full-16-GPU feasibility, not a validated replacement for the pinned companion or a sustained-capacity result. vLLM disaggregation was not tested.

V4 weights were downloaded once to the first node's NVMe in `96.550 s` and copied privately to the second node in `41.542 s`. The successful smaller model was DeepSeek-V2-Lite-Chat, `16 billion total parameters`, revision `85864749cd611b4353ce1decdb286193298f64c7`, with `31413626576 B` of weights. It is the largest model demonstrated on both constrained companion layouts in this rehearsal; no exhaustive model-size search was performed.

### Full paired participant flow within the stand-in budget

Both packed resident stacks used four GPUs, one EFA, a combined `96 vCPU` limit and a combined `384 GiB` limit per node. The engine pod received `92 vCPUs` and `376 GiB`; its local CPU router received `4 vCPUs` and `8 GiB`. CPU quota and memory cgroup reads confirmed enforcement. CPU affinity still exposed the full host, so this is a quota-limited comparison. The first node received GPU indices `0-3` and EFA `rdmap83s0`; the second received GPU indices `4-7` and EFA `rdmap176s0`. Each engine process used two GPUs. Both engine pod UIDs were unchanged across the complete comparison.

The Spain ECR engine digest was `sha256:772d52067fab28c9eea5fe1fd629694218b4aa1669b257c43e689abdcca59338`, with SGLang version `0.5.12.post1` and NIXL version `1.1.0`; router digest was `sha256:670c7f0004e2d068c7219888109362980195e554fb66480955d985dd9c9b52f2`. The stock transport verifier passed its request and selector checks, with `0 B` of cross-node EFA traffic because each stack's workers share one node. Cross-node KV transfer is UNVALIDATED by this packed run.

Every row used a `30 s` offered window plus warmup and drain, TTFT bound `2000 ms`, TPOT bound `100 ms/token` and dimensionless attainment target `0.9`. All `928` planned calls completed, with zero failed or skipped calls, and every measured row was client-valid.

| Architecture | Shape | Offered rate | p90 TTFT | p90 TPOT | Joint attainment, dimensionless | Useful rate |
|---|---|---|---|---|---|---|
| unified | A-agentic | 0.1 tasks/s | 546.371 ms | 7.555 ms/token | 1.000000 | 0.175000 calls/s/GPU |
| unified | B-long-context | 0.25 tasks/s | 1097.993 ms | 6.703 ms/token | 1.000000 | 0.025000 calls/s/GPU |
| unified | B-long-context | 0.5 tasks/s | 1101.292 ms | 10.923 ms/token | 1.000000 | 0.084718 calls/s/GPU |
| unified | B-long-context | 1 tasks/s | 1142.445 ms | 20.015 ms/token | 1.000000 | 0.203065 calls/s/GPU |
| unified | B-long-context | 2 tasks/s | 1468.162 ms | 62.390 ms/token | 1.000000 | 0.358609 calls/s/GPU |
| unified | B-long-context | 4 tasks/s | 22154.273 ms | 126.522 ms/token | 0.000000 | 0.000000 calls/s/GPU |
| unified | B-long-context | 8 tasks/s | 91537.452 ms | 136.158 ms/token | 0.000000 | 0.000000 calls/s/GPU |
| disaggregated | A-agentic | 0.1 tasks/s | 462.429 ms | 10.730 ms/token | 1.000000 | 0.158225 calls/s/GPU |
| disaggregated | B-long-context | 0.25 tasks/s | 1880.119 ms | 7.868 ms/token | 1.000000 | 0.025000 calls/s/GPU |
| disaggregated | B-long-context | 0.5 tasks/s | 2144.867 ms | 9.443 ms/token | 0.818182 | 0.068019 calls/s/GPU |
| disaggregated | B-long-context | 1 tasks/s | 2631.804 ms | 16.966 ms/token | 0.576923 | 0.114930 calls/s/GPU |
| disaggregated | B-long-context | 2 tasks/s | 10647.037 ms | 19.208 ms/token | 0.078431 | 0.022717 calls/s/GPU |
| disaggregated | B-long-context | 4 tasks/s | 60748.368 ms | 26.812 ms/token | 0.018182 | 0.005245 calls/s/GPU |
| disaggregated | B-long-context | 8 tasks/s | 167975.096 ms | 24.582 ms/token | 0.004132 | 0.001190 calls/s/GPU |

For long-context traffic, the highest tested qualifying rate was `2 tasks/s` for unified serving and `0.25 tasks/s` for disaggregation. The corresponding useful rates were `0.358609 calls/s/GPU` and `0.025000 calls/s/GPU`. The larger disaggregated useful rate reported at `1 task/s` did not meet the joint SLO. No tested qualifying crossover favored disaggregation. Shape A had lower disaggregated TTFT but higher unified useful throughput; neither metric is replaced with the expected ranking.

The complete paired flow took `1670.738585 s`. Unified and disaggregated shape A commands took `61.690303 s` and `68.287915 s`; the shape B sweeps took `620.158170 s` and `841.868637 s`. The complete load-ramp module, including collection and final identity checks, took about `25 minutes`, exceeding the previous `15-minute` allocation. The revised introduction reserves `26 minutes` for it within the `60-minute` session. Human-paced facilitation and repeated boundary measurements remain UNVALIDATED.

Raw manifests, package versions, per-request data, counter snapshots, paired identities and CSV are under `/tmp/spain-rehearsal/phase1/`. All rehearsal namespaces and PriorityClasses were removed. Device plugins, the open driver and NVMe caches remain. Driver reboot persistence, literal g7.24xlarge execution, cross-node unified V4 on the stand-in, sustained V4 capacity, no-host-staging transport, simultaneous separate-node paired stacks, eager bootstrap and prefill-only SLO recovery remain UNVALIDATED. Engine pins were not changed and engine sources were not patched.

## Dual-G7 main-line qualification, 2026-09-10

DeepSeek-V2-Lite-Chat revision `85864749cd611b4353ce1decdb286193298f64c7` remains on the engine digest `sha256:772d52067fab28c9eea5fe1fd629694218b4aa1669b257c43e689abdcca59338`, with SGLang version `0.5.12.post1` and both NIXL distributions at version `1.1.0`. The router digest remains `sha256:670c7f0004e2d068c7219888109362980195e554fb66480955d985dd9c9b52f2`.

The full g7.48xlarge pair had one resident stack per node, eight GPUs and both EFAs per stack, with two TP groups of four ranks in each packed engine pod. Combined engine/router limits were 188 vCPUs and 672 GiB per node. The original 740 GiB request exceeded Kubernetes allocatable memory and stayed Pending; those manifests/events were retained, and the configuration generator now rejects CPU/memory budgets above live allocatable quantities. No GPU result came from the oversized attempt.

The full flow completed in 1239.817780 s, including endpoint and transport checks, shape generation, both traffic shapes, sweeps, collection and final identity checks. Both engine pod UIDs remained unchanged. The first local port-forward attempt collided with existing controller listeners; the successful flow used local TCP ports 18000 and 18001. The generator and package verifier were then exercised again with the g7.24xlarge-stand-in allocation, shape A/B at 0.1 tasks/s for 10-second offered windows, before/after identity checks and a transport check. The previous complete stand-in sweep remains the comparison below.

Packed placement shares a node between prefill and decode. The verifier passed both package/selector profiles and successful requests while preserving `UNVALIDATED: co-located prefill/decode` for cross-node EFA. Both physical EFA domains were enumerated; flat packed request counters do not establish cross-node transfer.

All rows use a TTFT bound of 2000 ms, TPOT bound of 100 ms/output token and a minimum joint-attainment fraction of 0.9. Each measured offered window lasts 30 seconds, with separate warmup and full request drain. The reported rates are finite-window observations, not sustained-capacity estimates.

| Allocation | Architecture | Shape | Offered rate | p90 TTFT | p90 TPOT | Joint attainment, dimensionless | Useful rate | Qualifies |
|---|---|---|---|---|---|---|---|---|
| g7.48xlarge | unified | A-agentic | 0.1 tasks/s | 511.761 ms | 6.485 ms/output token | 1.000000 | 0.087500 calls/s/GPU | True |
| g7.48xlarge | unified | B-long-context | 0.25 tasks/s | 801.466 ms | 6.350 ms/output token | 1.000000 | 0.012500 calls/s/GPU | True |
| g7.48xlarge | unified | B-long-context | 0.5 tasks/s | 809.603 ms | 8.711 ms/output token | 1.000000 | 0.042879 calls/s/GPU | True |
| g7.48xlarge | unified | B-long-context | 1 tasks/s | 801.474 ms | 10.853 ms/output token | 1.000000 | 0.104366 calls/s/GPU | True |
| g7.48xlarge | unified | B-long-context | 2 tasks/s | 808.791 ms | 22.011 ms/output token | 1.000000 | 0.194294 calls/s/GPU | True |
| g7.48xlarge | unified | B-long-context | 4 tasks/s | 1449.429 ms | 90.035 ms/output token | 1.000000 | 0.363865 calls/s/GPU | True |
| g7.48xlarge | unified | B-long-context | 8 tasks/s | 30289.595 ms | 133.672 ms/output token | 0.000000 | 0.000000 calls/s/GPU | False |
| g7.48xlarge | disaggregated | A-agentic | 0.1 tasks/s | 528.861 ms | 8.340 ms/output token | 1.000000 | 0.085550 calls/s/GPU | True |
| g7.48xlarge | disaggregated | B-long-context | 0.25 tasks/s | 1396.770 ms | 7.362 ms/output token | 1.000000 | 0.012500 calls/s/GPU | True |
| g7.48xlarge | disaggregated | B-long-context | 0.5 tasks/s | 1316.110 ms | 8.867 ms/output token | 1.000000 | 0.042467 calls/s/GPU | True |
| g7.48xlarge | disaggregated | B-long-context | 1 tasks/s | 1512.302 ms | 11.237 ms/output token | 1.000000 | 0.102670 calls/s/GPU | True |
| g7.48xlarge | disaggregated | B-long-context | 2 tasks/s | 2384.346 ms | 18.617 ms/output token | 0.725490 | 0.135482 calls/s/GPU | False |
| g7.48xlarge | disaggregated | B-long-context | 4 tasks/s | 26586.201 ms | 20.407 ms/output token | 0.054545 | 0.012413 calls/s/GPU | False |
| g7.48xlarge | disaggregated | B-long-context | 8 tasks/s | 96870.185 ms | 23.327 ms/output token | 0.016529 | 0.003710 calls/s/GPU | False |
| g7.24xlarge-stand-in | unified | A-agentic | 0.1 tasks/s | 546.371 ms | 7.555 ms/output token | 1.000000 | 0.175000 calls/s/GPU | True |
| g7.24xlarge-stand-in | unified | B-long-context | 0.25 tasks/s | 1097.993 ms | 6.703 ms/output token | 1.000000 | 0.025000 calls/s/GPU | True |
| g7.24xlarge-stand-in | unified | B-long-context | 0.5 tasks/s | 1101.292 ms | 10.923 ms/output token | 1.000000 | 0.084718 calls/s/GPU | True |
| g7.24xlarge-stand-in | unified | B-long-context | 1 tasks/s | 1142.445 ms | 20.015 ms/output token | 1.000000 | 0.203065 calls/s/GPU | True |
| g7.24xlarge-stand-in | unified | B-long-context | 2 tasks/s | 1468.162 ms | 62.390 ms/output token | 1.000000 | 0.358609 calls/s/GPU | True |
| g7.24xlarge-stand-in | unified | B-long-context | 4 tasks/s | 22154.273 ms | 126.522 ms/output token | 0.000000 | 0.000000 calls/s/GPU | False |
| g7.24xlarge-stand-in | unified | B-long-context | 8 tasks/s | 91537.452 ms | 136.158 ms/output token | 0.000000 | 0.000000 calls/s/GPU | False |
| g7.24xlarge-stand-in | disaggregated | A-agentic | 0.1 tasks/s | 462.429 ms | 10.730 ms/output token | 1.000000 | 0.158225 calls/s/GPU | True |
| g7.24xlarge-stand-in | disaggregated | B-long-context | 0.25 tasks/s | 1880.119 ms | 7.868 ms/output token | 1.000000 | 0.025000 calls/s/GPU | True |
| g7.24xlarge-stand-in | disaggregated | B-long-context | 0.5 tasks/s | 2144.867 ms | 9.443 ms/output token | 0.818182 | 0.068019 calls/s/GPU | False |
| g7.24xlarge-stand-in | disaggregated | B-long-context | 1 tasks/s | 2631.804 ms | 16.966 ms/output token | 0.576923 | 0.114930 calls/s/GPU | False |
| g7.24xlarge-stand-in | disaggregated | B-long-context | 2 tasks/s | 10647.037 ms | 19.208 ms/output token | 0.078431 | 0.022717 calls/s/GPU | False |
| g7.24xlarge-stand-in | disaggregated | B-long-context | 4 tasks/s | 60748.368 ms | 26.812 ms/output token | 0.018182 | 0.005245 calls/s/GPU | False |
| g7.24xlarge-stand-in | disaggregated | B-long-context | 8 tasks/s | 167975.096 ms | 24.582 ms/output token | 0.004132 | 0.001190 calls/s/GPU | False |

The highest tested qualifying long-context rates were 4 tasks/s unified and 1 task/s disaggregated on g7.48xlarge, versus 2 tasks/s and 0.25 tasks/s on the g7.24xlarge-stand-in. Neither sweep established a qualifying crossover favoring disaggregation. Preserve this result and the common workload/routing policy. The full flow and previous stand-in flow fit the 60-minute machine budget; human-paced delivery, literal secondary hardware and the production account/participant-role path remain unvalidated.

## Optional V4-Flash full-G7 qualification, 2026-09-10

This is a separate optional engine and model qualification. It does not advance the DeepSeek-V2-Lite-Chat main-line pin. `Dockerfile.v4` pins SGLang version `0.5.19`, both NIXL distributions at version `1.4.1` and EFA installer version `1.47.0`. The final amd64 manifest is `sha256:7da39d58804be6c781991dad9c099bf65c95c49eec42b3d5e927d89105b2e061`; the pushed image index is `sha256:30fd3480469d257011feba3548ee30ed866ebfb1c3fee50f35252300651967b3`. The Spain ECR tag is `aim345-v4-g7-dual-final` in repository `aim-spain-rehearsal-20260910`. The initial package-source assertion failed on the newer annotated server-argument default, was corrected in the written verifier, and the final image was rebuilt and pushed. No serving-engine source was patched.

DeepSeek-V4-Flash revision `60d8d70770c6776ff598c94bb586a859a38244f1` uses approximately 148.7 GiB of weight shards. Each full g7.48xlarge node provides approximately 254.9 GiB of GPU memory; a secondary node provides approximately 127.4 GiB and cannot hold that replica. The split arm used prefill TP size of eight ranks on the first node and decode TP size of eight ranks on the second, with both EFA devices and engine limits of 188 vCPUs and 672 GiB per node. The unified comparison used the same pair sequentially. The NVMe snapshot and persisted compilation cache were already warm; these are not cold-download or fresh-JIT timings.

The live transport verifier passed with `engine_profile=v4-optional`, a completed request and 97235456 B of positive cross-node RDMA counter deltas. This proves an EFA path for that request, not absence of host staging. The optional model is excluded from the main-line MLA bandwidth formula. GPU limits and actual node placement are retained in the allocation JSON files.

Both traffic shapes used the V4 tokenizer revision. All rows have a 30-second offered window, separate warmup and complete request drain, with TTFT bound 2000 ms, TPOT bound 100 ms/output token and required joint attainment 0.9 dimensionless. Shape A passed at 0.1 tasks/s for both architectures. Shape B in the split architecture already missed the joint SLO at 0.1 tasks/s, despite all calls completing. Unified first missed at 0.5 tasks/s; its highest tested qualifying long-context rate was 0.25 tasks/s. No qualifying crossover favored disaggregation. The lowest rates contain only three long-context calls per window and do not establish sustained capacity or a precise boundary.

| Allocation | Architecture | Shape | Offered rate | p90 TTFT | p90 TPOT | Joint attainment, dimensionless | Useful rate | Qualifies |
|---|---|---|---|---|---|---|---|---|
| g7.48xlarge | unified | A-agentic | 0.1 tasks/s | 531.570 ms | 17.855 ms/output token | 1.000000 | 0.026600 calls/s/GPU | True |
| g7.48xlarge | unified | B-long-context | 0.1 tasks/s | 1757.166 ms | 16.264 ms/output token | 1.000000 | 0.006250 calls/s/GPU | True |
| g7.48xlarge | unified | B-long-context | 0.25 tasks/s | 1808.411 ms | 16.246 ms/output token | 1.000000 | 0.006250 calls/s/GPU | True |
| g7.48xlarge | unified | B-long-context | 0.5 tasks/s | 2074.495 ms | 25.285 ms/output token | 0.818182 | 0.015425 calls/s/GPU | False |
| g7.48xlarge | unified | B-long-context | 1 tasks/s | 9615.488 ms | 36.344 ms/output token | 0.230769 | 0.008796 calls/s/GPU | False |
| g7.48xlarge | unified | B-long-context | 2 tasks/s | 33661.477 ms | 36.392 ms/output token | 0.058824 | 0.002616 calls/s/GPU | False |
| g7.48xlarge | disaggregated | A-agentic | 0.1 tasks/s | 665.327 ms | 20.504 ms/output token | 1.000000 | 0.024349 calls/s/GPU | True |
| g7.48xlarge | disaggregated | B-long-context | 0.1 tasks/s | 3095.947 ms | 17.404 ms/output token | 0.333333 | 0.002083 calls/s/GPU | False |
| g7.48xlarge | disaggregated | B-long-context | 0.25 tasks/s | 3085.620 ms | 17.447 ms/output token | 0.333333 | 0.002083 calls/s/GPU | False |
| g7.48xlarge | disaggregated | B-long-context | 0.5 tasks/s | 5438.538 ms | 19.438 ms/output token | 0.090909 | 0.001622 calls/s/GPU | False |
| g7.48xlarge | disaggregated | B-long-context | 1 tasks/s | 13294.535 ms | 18.442 ms/output token | 0.038462 | 0.001341 calls/s/GPU | False |
| g7.48xlarge | disaggregated | B-long-context | 2 tasks/s | 38496.522 ms | 15.828 ms/output token | 0.000000 | 0.000000 calls/s/GPU | False |

| Allocation | Optional command | Wall time | Exit status, dimensionless |
|---|---|---|---|
| g7.48xlarge | v4-cleanup | 47.614721 s | 0 |
| g7.48xlarge | v4-comparison | 0.039760 s | 0 |
| g7.48xlarge | v4-disaggregated-a | 109.115150 s | 0 |
| g7.48xlarge | v4-disaggregated-allocation | 1.143256 s | 0 |
| g7.48xlarge | v4-disaggregated-b | 46.097991 s | 0 |
| g7.48xlarge | v4-disaggregated-collect | 19.089330 s | 0 |
| g7.48xlarge | v4-disaggregated-deploy | 16.532956 s | 0 |
| g7.48xlarge | v4-disaggregated-sweep | 350.248565 s | 0 |
| g7.48xlarge | v4-generate-a | 1.432091 s | 0 |
| g7.48xlarge | v4-generate-b | 1.506749 s | 0 |
| g7.48xlarge | v4-source | 0.835300 s | 0 |
| g7.48xlarge | v4-transport | 35.377011 s | 0 |
| g7.48xlarge | v4-unified-a | 117.628601 s | 0 |
| g7.48xlarge | v4-unified-allocation | 1.046901 s | 0 |
| g7.48xlarge | v4-unified-b | 42.814948 s | 0 |
| g7.48xlarge | v4-unified-collect | 18.627997 s | 0 |
| g7.48xlarge | v4-unified-deploy | 9.926179 s | 0 |
| g7.48xlarge | v4-unified-sweep | 326.716856 s | 0 |

The optional completion check passes as an experiment: allocation guard, package profile, cross-node transfer, both shape records and an observed rate-dependent SLO miss are preserved. It does not require every measured rate to meet SLO. Both optional deployments were removed after evidence capture. The changed main-line generator was also rerun against full and constrained inventory; its final automatically derived full budget exactly matched the qualified deployment, excluding namespace. Ten controller tests passed.
