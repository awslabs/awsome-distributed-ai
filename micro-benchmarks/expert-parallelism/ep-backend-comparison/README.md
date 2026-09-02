# Expert-Parallelism Backend Comparison on EFA

This directory compares 3 expert-parallel dispatch/combine backends through common Decode-like and Prefill-like communication workloads:

| Backend | Implementation and transport |
|---|---|
| UCCL | DeepEP-compatible dispatch/combine over UCCL all-to-all and EFA |
| DeepEP V1 NVSHMEM | DeepEP V1 `Buffer` over NVSHMEM, libfabric, and EFA |
| DeepEP V2 NCCL GIN | DeepEP V2 `ElasticBuffer` over NCCL GIN EFA-GDA |

Each workload uses one external CUDA timing boundary and one logical payload definition. Backend-native latency and bandwidth fields remain diagnostics because they do not share one timing boundary or byte numerator.

The B200 report and backend box plots are in [RESULTS.md](RESULTS.md). EP32 means 32 GPU ranks on 4 B200 nodes, not 32 B200 nodes.

## Workload profiles

| Profile | Tokens | UCCL and DeepEP V1 API | DeepEP V2 API | Primary metric |
|---|---:|---|---|---|
| Decode-like | 128 tokens/rank | `low_latency_dispatch` and `low_latency_combine` | `ElasticBuffer.dispatch` and `ElasticBuffer.combine` | Slowest-rank latency, in ms |
| Prefill-like | 4,096 tokens/rank | Normal `Buffer.dispatch` and `Buffer.combine` | `ElasticBuffer.dispatch` and `ElasticBuffer.combine` | Slowest-rank latency, in ms |

The Prefill-like timing boundary starts with the dispatch input and exact route ready. It includes the dispatch layout required by the normal UCCL and DeepEP V1 APIs, dispatch, and combine completion. The Decode-like boundary starts with the dispatch input ready and includes dispatch and combine completion. Host-side FP8 conversion runs once before either boundary, so no arm is charged for its Python quantizer. In the Decode-like FP8 cell the low-latency UCCL and DeepEP V1 kernels still quantize internally, inside the boundary; each backend is measured from its own API's dispatch-ready entry point.

Both profiles hold the following controls constant across backends:

| Control | Rule |
|---|---|
| Input | One deterministic BF16 tensor per profile and EP size |
| Routing | One exact top-k route and one set of weights, verified by SHA-256 across all arms and starts |
| Model shape | Hidden size 7,168, 256 experts, top-k 8 experts/token |
| Operations | FP8 or BF16 dispatch followed by BF16 combine |
| Rank reduction | Maximum elapsed time across all ranks for each measured iteration |
| SM count | Each result records the communication-kernel SM count (`num_sms`) where the backend exposes one; `EP_NUM_SMS` pins it explicitly, and 0 keeps each backend's automatic choice |
| Warmup | 20 warmup iterations per dtype and process start |
| Measurement | 100 measured iterations per dtype and process start |
| Replication | `INDEPENDENT_STARTS` independent process starts per arm and workload cell, default 20; the reported campaigns used 16 and 4 |
| Order | Backend, dtype, and workload-profile order rotate across starts |
| Hardware | The same named nodes serve every arm at a given EP size |
| Runtime | Every result reports the same GPU, PyTorch, CUDA, and torch-built NCCL versions, plus the NCCL library actually loaded (consistent within each arm) |
| Correctness | Every rank passes the common identity-expert result before timing |

Each process start contributes its median of 100 slowest-rank iteration measurements. The report then takes the median across the independent process starts. Iterations within one process are not treated as independent replicates.

## Common logical throughput

Each valid expert assignment contributes the dispatch tensor, FP8 scales when FP8 dispatch is selected, and the BF16 combine tensor. Backend metadata is excluded. Scale-out logical bytes include only assignments whose destination expert is on another node.

```text
logical GB/s/rank = average logical bytes/rank / median slowest-rank latency
scale-out logical GB/s/rank = average remote logical bytes/rank / median slowest-rank latency
```

These are secondary logical efficiency metrics, not observed wire bandwidth. Within one profile, EP size, and dtype cell the byte numerator is a shared constant, so they rank backends identically to the primary latency metric. Compare across EP sizes with aggregate input tokens/s instead: the scale-out numerator counts every remote assignment as a full tensor, while a backend may send one copy per destination node and forward locally, so its over-count factor changes with node count and the scale-out column is comparable only within one EP size.

## Files

| File | Purpose |
|---|---|
| [`ep_benchmark.py`](ep_benchmark.py) | Workload profiles, backend adapters, correctness checks, CUDA timing, and logical-byte accounting |
| [`run_ep_rank.sh`](run_ep_rank.sh) | Per-node `torchrun` entry point and backend-specific transport environment |
| [`run_ep_comparison.sh`](run_ep_comparison.sh) | EKS admission, shared-Lease coordination, rotated matrix, durable harvest, and verified teardown |
| [`result_io.py`](result_io.py) | Robust result-marker parsing from interleaved native output |
| [`extract_results.py`](extract_results.py) | Canonical JSONL extraction from a rank-zero log |
| [`summarize_results.py`](summarize_results.py) | Matrix validation, per-start aggregation, paired deltas, and bootstrap intervals |
| [`plot_results.py`](plot_results.py) | Box plots of the independent-start primary values for every backend arm |
| [`RESULTS.md`](RESULTS.md) | Human-readable result, provenance, and scope limits |

## Requirements

The scored B200 matrix requires:

- enough named, Ready `p6-b200.48xlarge` nodes (`EP_INSTANCE_TYPE`) in one EKS cluster for the largest configured EP size (`EP_WORLD_SIZES`, default `16 32`, needs 4 nodes; a 2-node cluster can run `EP_WORLD_SIZES=16`); `EP_REGION` and `EP_CLUSTER_NAME` label the campaign provenance;
- 8 allocatable GPUs and 8 allocatable EFA devices on every selected node;
- no active GPU requests on the selected nodes before each arm;
- the NVIDIA and EFA Kubernetes device plugins;
- `uvm_disable_hmm=Y` or `uvm_disable_hmm=1` on every selected host;
- `/dev/gdrdrv` as a character device on every selected host;
- `aws`, `kubectl`, `jq`, `rg`, Python 3, and Bash on the launch host; and
- access to the 3 digest-pinned backend images.

DeepEP V2 receives an INFO-level Decode-like admission run at the smallest configured EP size before the scored matrix. All 3 backends then receive a Prefill-like admission run at that size. A missing HMM mitigation, GDRCopy device, GDAKI proof, or profile correctness result stops the campaign before scoring.

## Run on an exclusive node set

Use a unique namespace and durable artifact directory. `KUBECTL_CONTEXT` is required explicitly so a concurrent process changing the default context cannot redirect the campaign.

```bash
campaign_id=ep-b200-$(date -u +%Y%m%d%H%M%S)
CAMPAIGN_ID="${campaign_id}" \
EP_BENCHMARK_NODES=node-a,node-b,node-c,node-d \
PROTECTED_NODES_CSV="" \
ARTIFACT_ROOT="/shared/artifacts/${campaign_id}" \
KUBECTL_CONTEXT=aps1-shared \
LOCK_MODE=exclusive \
./run_ep_comparison.sh
```

`LOCK_MODE=exclusive` claims the configured shared Lease only when its holder is empty. The runner releases only a Lease that it still owns.

## Run beside a coordinated campaign

Observe mode is allowed only when the other campaign has a known Lease holder and a disjoint named node set:

```bash
campaign_id=ep-b200-$(date -u +%Y%m%d%H%M%S)
CAMPAIGN_ID="${campaign_id}" \
EP_BENCHMARK_NODES=node-a,node-b,node-c,node-d \
PROTECTED_NODES_CSV=foreign-node-a,foreign-node-b \
ARTIFACT_ROOT="/shared/artifacts/${campaign_id}" \
KUBECTL_CONTEXT=aps1-shared \
LOCK_MODE=observe \
EXPECTED_LOCK_HOLDER=foreign-campaign-id \
./run_ep_comparison.sh
```

Observe mode never mutates the shared Lease. It verifies the exact holder before every arm and again before aggregation. Any selected/protected node overlap or Lease-holder change stops the run.

## Execution matrix

The scored order uses 20 independent starts (`INDEPENDENT_STARTS`). Three rotation patterns cycle across the starts:

| Start index mod 3 | Backend order | Profile order | Dtype order |
|---:|---|---|---|
| 1 | UCCL, DeepEP V1, DeepEP V2 | Decode-like, Prefill-like | FP8, BF16 |
| 2 | DeepEP V2, UCCL, DeepEP V1 | Prefill-like, Decode-like | BF16, FP8 |
| 0 | DeepEP V1, DeepEP V2, UCCL | Decode-like, Prefill-like | FP8, BF16 |

The runner executes the rotation at each configured EP size in `EP_WORLD_SIZES` order, by default first at 16 ranks on 2 nodes and then at 32 ranks on 4 nodes. Arms run serially, and every StatefulSet and its GPU pods must be gone before the next arm is admitted. The default matrix contains 240 distributed process starts and 480 scored dtype results.

## Durable artifacts and teardown

The campaign writes the following layout under `ARTIFACT_ROOT`:

```text
control/
  aws-caller-identity.json
  fleet-nodes-before.json
  fleet-nodes-after.json
  fleet-pods-before.json
  fleet-pods-after.json
  provenance.json
  selected-nodes.txt
runs/
  decode/ep16/{admission,measurement}-repeat-*/<backend>/
  decode/ep32/measurement-repeat-*/<backend>/
  prefill/ep16/{admission,measurement}-repeat-*/<backend>/
  prefill/ep32/measurement-repeat-*/<backend>/
summary/
  summary.json
  summary.md
teardown/
  namespace-delete.log
  remaining-resources.json
  shared-lease-after.json
CAMPAIGN_COMPLETE
SHA256SUMS
STATUS
```

Every rank log, rendered Pod manifest, Pod description, canonical rank-zero JSONL result, case status, input/route hash, and immutable image reference is retained. `CAMPAIGN_COMPLETE` is written only after the full scored matrix succeeds and teardown verifies that the owned namespace and labeled resources are absent. `SHA256SUMS` is generated after the final status markers.

## Re-aggregate preserved logs

```bash
python3 summarize_results.py /path/to/artifacts/runs \
  --starts=20 \
  --provenance=/path/to/artifacts/control/provenance.json \
  --json=/path/to/artifacts/summary/summary.json \
  --markdown=/path/to/artifacts/summary/summary.md
```

Pass `--world-sizes` for a reduced matrix, for example `--world-sizes=16` for a 2-node EP16-only campaign. Without the flag the summarizer derives the EP sizes from the loaded logs and validates the full arm/dtype/profile/start matrix for each derived size.

If a native library appends a diagnostic to the JSON marker's physical line, use the repository parser:

```bash
python3 extract_results.py rank-zero.log results.jsonl
```

The summarizer rejects an incomplete matrix, correctness failure, mutable image tag, route/input mismatch, runtime-stack mismatch, or disagreement in common logical payload accounting.

Regenerate the box plots from the committed machine-readable summary:

```bash
python3 plot_results.py results/b200-us-east-1-2026-09-01-ep16-n16.json \
  --output=results/b200-us-east-1-2026-09-01-ep16-n16-boxplots.png
```

Plot generation requires Matplotlib. Each box uses the independent process-start medians for one backend and workload cell. The plot also shows every underlying point.

## Local validation

```bash
python3 -m pytest -q test_ep_benchmark.py test_summarize_results.py
python3 -m py_compile \
  ep_benchmark.py result_io.py extract_results.py summarize_results.py plot_results.py
bash -n run_ep_comparison.sh run_ep_rank.sh
shellcheck run_ep_comparison.sh run_ep_rank.sh
```

## Scope limits

These profiles measure synthetic dispatch-plus-combine communication. Prefill-like does not measure TTFT, and Decode-like does not measure TPOT. Neither profile measures expert compute, communication/computation overlap, end-to-end training, serving throughput, or end-to-end latency. Results apply only to the reported profile, EP size, routing distribution, hardware, and runtime stack.
