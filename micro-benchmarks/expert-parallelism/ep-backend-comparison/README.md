# Fair Expert-Parallelism Backend Comparison on EFA

This directory compares 3 expert-parallel dispatch/combine backends through one common semantic workload and one external timing boundary:

| Backend | Implementation and transport |
|---|---|
| UCCL | DeepEP-compatible dispatch/combine over UCCL all-to-all and EFA |
| DeepEP V1 NVSHMEM | DeepEP V1 `Buffer` over NVSHMEM, libfabric, and EFA |
| DeepEP V2 NCCL GIN | DeepEP V2 `ElasticBuffer` over NCCL GIN EFA-GDA |

The primary metric is slowest-rank CUDA latency from BF16 input readiness through dispatch and combine completion. Backend-native latency and bandwidth fields are not used for cross-backend rankings because they do not share one timing boundary or byte numerator.

The validated B200 result is in [RESULTS.md](RESULTS.md). EP32 means 32 GPU ranks on 4 B200 nodes, not 32 B200 nodes.

## What makes the comparison fair

The backend implementation is the intended independent variable. The harness holds these inputs and measurement rules constant:

| Control | Rule |
|---|---|
| Input | One deterministic BF16 tensor per EP size |
| Routing | One exact top-k route and one set of weights, verified by SHA-256 across all arms and starts |
| Shape | 128 tokens/rank, hidden size 7,168, 256 experts, top-k 8 experts/token |
| Operations | FP8 or BF16 dispatch followed by BF16 combine |
| Timing | One CUDA Event boundary around input preparation, dispatch, and combine |
| Rank reduction | Maximum elapsed time across all ranks for each measured iteration |
| Warmup | 20 warmup iterations per dtype and process start |
| Measurement | 100 measured iterations per dtype and process start |
| Replication | 3 independent process starts per arm and workload cell |
| Order | Backend order rotates across starts; dtype order also rotates |
| Hardware | The same named nodes serve every arm at a given EP size |
| Runtime | Every result must report the same GPU, PyTorch, CUDA, and NCCL versions |
| Correctness | Every rank must pass the common identity-expert result before timing |

Each process start contributes its median of 100 slowest-rank iteration latencies. The report then takes the median across 3 process starts. Iterations within one process are not treated as independent replicates.

### Common logical throughput

The harness derives effective logical throughput from a common useful-payload numerator. Each valid expert assignment contributes:

- the dispatch tensor;
- FP8 scales when FP8 dispatch is selected; and
- the BF16 combine tensor.

Backend metadata is excluded. Scale-out logical bytes include only assignments whose destination expert is on another node.

```text
logical GB/s/rank = average logical bytes/rank / median slowest-rank latency
scale-out logical GB/s/rank = average remote logical bytes/rank / median slowest-rank latency
```

These are logical efficiency metrics, not observed wire bandwidth. DeepEP V2 SO/SU bandwidth, DeepEP V1 native bandwidth, and UCCL native bandwidth remain useful backend diagnostics, but their numerators and aggregation boundaries differ and must not be compared directly.

## Files

| File | Purpose |
|---|---|
| [`fair_ep_benchmark.py`](fair_ep_benchmark.py) | Common workload, backend adapters, correctness check, CUDA timing, and logical-byte accounting |
| [`run_fair_ep_rank.sh`](run_fair_ep_rank.sh) | Per-node `torchrun` entry point and backend-specific transport environment |
| [`run_fair_ep_comparison.sh`](run_fair_ep_comparison.sh) | EKS admission, shared-Lease coordination, rotated matrix, durable harvest, and verified teardown |
| [`fair_result_io.py`](fair_result_io.py) | Robust result-marker parsing from interleaved native output |
| [`extract_fair_results.py`](extract_fair_results.py) | Canonical JSONL extraction from a rank-zero log |
| [`summarize_fair_results.py`](summarize_fair_results.py) | Matrix validation, per-start aggregation, paired deltas, and bootstrap intervals |
| [`results/b200-ap-south-1-2026-08-24.json`](results/b200-ap-south-1-2026-08-24.json) | Machine-readable validated result summary |
| [`RESULTS.md`](RESULTS.md) | Human-readable result, provenance, and scope limits |

## Requirements

The scored B200 matrix requires:

- 4 named, Ready `p6-b200.48xlarge` nodes in one EKS cluster;
- 8 allocatable GPUs and 8 allocatable EFA devices on every selected node;
- no active GPU requests on the selected nodes before each arm;
- the NVIDIA and EFA Kubernetes device plugins;
- `uvm_disable_hmm=Y` or `uvm_disable_hmm=1` on every selected host;
- `/dev/gdrdrv` as a character device on every selected host;
- `aws`, `kubectl`, `jq`, `rg`, Python 3, and Bash on the launch host; and
- access to the 3 digest-pinned backend images.

DeepEP V2 receives an INFO-level EP16 admission run before the scored matrix. The admission must log a GDAKI context. A missing HMM mitigation, GDRCopy device, or GDAKI proof stops the campaign before scoring.

## Run on an exclusive node set

Use a unique namespace and durable artifact directory. `KUBECTL_CONTEXT` is required explicitly so a concurrent process changing the default context cannot redirect the campaign.

```bash
campaign_id=fair-ep-b200-$(date -u +%Y%m%d%H%M%S)
CAMPAIGN_ID="${campaign_id}" \
FAIR_EP_NODES=node-a,node-b,node-c,node-d \
PROTECTED_NODES_CSV="" \
ARTIFACT_ROOT="/shared/artifacts/${campaign_id}" \
KUBECTL_CONTEXT=aps1-shared \
LOCK_MODE=exclusive \
./run_fair_ep_comparison.sh
```

`LOCK_MODE=exclusive` claims the configured shared Lease only when its holder is empty. The runner releases only a Lease that it still owns.

## Run beside a coordinated campaign

Observe mode is allowed only when the other campaign has a known Lease holder and a disjoint named node set:

```bash
campaign_id=fair-ep-b200-$(date -u +%Y%m%d%H%M%S)
CAMPAIGN_ID="${campaign_id}" \
FAIR_EP_NODES=node-a,node-b,node-c,node-d \
PROTECTED_NODES_CSV=foreign-node-a,foreign-node-b \
ARTIFACT_ROOT="/shared/artifacts/${campaign_id}" \
KUBECTL_CONTEXT=aps1-shared \
LOCK_MODE=observe \
EXPECTED_LOCK_HOLDER=foreign-campaign-id \
./run_fair_ep_comparison.sh
```

Observe mode never mutates the shared Lease. It verifies the exact holder before every arm and again before aggregation. Any selected/protected node overlap or Lease-holder change stops the run.

## Execution matrix

The scored order is a 3-start rotation:

| Start index | Backend order | Dtype order |
|---:|---|---|
| 1 | UCCL, DeepEP V1, DeepEP V2 | FP8, BF16 |
| 2 | DeepEP V2, UCCL, DeepEP V1 | BF16, FP8 |
| 3 | DeepEP V1, DeepEP V2, UCCL | FP8, BF16 |

The runner executes this rotation first at 16 ranks on 2 nodes and then at 32 ranks on 4 nodes. Arms run serially, and every StatefulSet and its GPU pods must be gone before the next arm is admitted.

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
  ep16/{admission,measurement}-repeat-*/<backend>/
  ep32/measurement-repeat-*/<backend>/
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

The normal campaign aggregates automatically. To validate a preserved artifact tree again:

```bash
python3 summarize_fair_results.py /path/to/artifacts/runs \
  --starts=3 \
  --provenance=/path/to/artifacts/control/provenance.json \
  --json=/path/to/artifacts/summary/summary.json \
  --markdown=/path/to/artifacts/summary/summary.md
```

If a native library appends a diagnostic to the JSON marker's physical line, extract the JSON object with the repository parser rather than `grep` or line splitting:

```bash
python3 extract_fair_results.py rank-zero.log results.jsonl
```

The summarizer rejects an incomplete matrix, correctness failure, mutable image tag, route/input mismatch, runtime-stack mismatch, or disagreement in the common logical payload.

## Local validation

```bash
python3 -m pytest -q test_fair_ep_benchmark.py test_summarize_fair_results.py
python3 -m py_compile \
  fair_ep_benchmark.py fair_result_io.py extract_fair_results.py \
  summarize_fair_results.py
bash -n run_fair_ep_comparison.sh run_fair_ep_rank.sh
shellcheck run_fair_ep_comparison.sh run_fair_ep_rank.sh
```

## Scope limits

This harness measures a synthetic decode dispatch-plus-combine communication workload. It does not measure prefill, expert compute, communication/computation overlap, end-to-end training, serving throughput, TTFT, TPOT, or E2E latency. A result from this harness does not establish a universal backend winner or a limit at an unmeasured EP size.
