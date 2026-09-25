# Dense LLM training, recovery and serving

The proposed participant training route uses SmolLM2-1.7B at 4,096 positions. Qwen3-4B recovery and serving remain separate modules. Smol eager/SDPA numerical qualification is unresolved; measured compute and network timings are development evidence, not an accepted cumulative optimization. The historical Qwen training sequence below is reference only.

## Prepare separate Smol and Qwen environments

The revised `1.prepare.sh --llm` accepts `LLM_CONFIG`, `LLM_PREP_RECORDS` and `LLM_MIN_DOCUMENT_TOKENS`. Defaults preserve the Qwen preparation path. In the existing assigned two-node allocation, provide the image paths/hashes and fresh data/results/package paths documented below. Select Smol explicitly:

```bash
export LLM_CONFIG=configs/llm-zero-smol.json LLM_PREP_RECORDS=160 LLM_MIN_DOCUMENT_TOKENS=4096
bash facilitator/prepare-login.sh --llm-prepare
```

Then choose distinct fresh `LLM_DATA_DIR`, `LLM_RESULTS_DIR` and `LLM_PREPARATION_PACKAGES` paths for Qwen and prepare it separately:

```bash
export LLM_CONFIG=configs/llm.json LLM_PREP_RECORDS=1024 LLM_MIN_DOCUMENT_TOKENS=0
bash facilitator/prepare-login.sh --llm-prepare
```

The helper creates a config-validated `llm-environment.sh` under each result root. Load the appropriate file before each module. Smol uses `configs/llm-zero-smol.json` explicitly in every training invocation; Qwen recovery uses `configs/llm-recovery.json`, and serving reads the Qwen pretrained model. The launcher default remains Qwen for compatibility. These selectors require this revision and are not implemented by the earlier published draft head `2cc899a38c2c1934279ede273e75307af8b22f99`.

Smol's expected160-document token/length fingerprint is `276a090b9451c92d61cdeb1ca627a92bbff1e7512ad49ad7f322fb537546a166`. Five updates with one warmup yield524,160 measured useful tokens. Qwen's original1,024-document fingerprint is `da15432db18fc483844f9f6b0b9b5142d322a3b8ee049827fe9fb5b167d921f7`. Preserve the Qwen participant checkpoint and its recording under the Qwen result root; preparation does not generate it.

For Smol, the conservative reference selects `--attention-implementation eager --activation-checkpointing --microbatch 1`; the endpoint selects `--attention-implementation sdpa --fused-optimizer --microbatch 2` without activation checkpointing. Keep EFA fixed during compute comparisons. For the network comparison, keep endpoint compute fixed and select `LLM_TRANSPORT=socket` or `LLM_TRANSPORT=efa`, preserving actual NCCL logs. No strict-deterministic or no-fill options are part of this Smol timing recipe. Fixed numerical policy remains unchanged.

## Historical Qwen training and separate Qwen module details

The remaining training comparisons document the earlier Qwen investigation, not the active Smol participant sequence. Qwen recovery and serving measurements retain their original scope.


This development recipe runs full-parameter Qwen3-4B training on pretokenized FineWeb-Edu documents, restores PyTorch distributed checkpoints, and measures quality-constrained serving. Spain PCS runs on two `g7.48xlarge` nodes with sixteen RTX PRO 4500 Blackwell Server Edition GPUs have completed training and synchronous/asynchronous recovery on shared FSx for Lustre. Performance tuning and serving qualification remain in progress. The complete three-stage training sequence and participant workshop are not yet qualified.

The older numbered `v0` through `v3` scripts and their validation records describe the previous workload. Use the `llm` entry points below for this recipe. Every comparison starts from the same pretrained revision.

## Software and data

[configs/llm.json](configs/llm.json) pins the model, tokenizer and dataset revisions. Qwen3-4B weights and tokenizer use Apache-2.0. FineWeb-Edu uses ODC-By-1.0; Common Crawl terms also apply. Preparation writes attribution, source terms and per-document source URLs alongside the token files. Keep that attribution when redistributing permitted prepared data.

Training uses PyTorch `2.9.1+cu130`, Transformers `4.57.6`, FSDP2, BF16 computation, FP32 gradient reduction and FP32 master parameters. The existing [Dockerfile](Dockerfile), [requirements.txt](requirements.txt) and [constraints.txt](constraints.txt) describe the training image. The tested SquashFS SHA-256 is `4624e0ccf1e5dec07be1eea75eefcdeafa8646035591c13165dacec06b6f07cc`. Serving requires vLLM `0.20.2`; its tested image candidate SHA-256 is `ea16c095d375012bc608d41eb7db8f8ff5f31e685b55c8b18021017fa5070d79`. A rebuilt image needs its own runtime verification.

The preparation runtime tested on Spain uses Python `3.10`. Install [requirements-data.txt](requirements-data.txt) in a separate preparation environment with the pinned training dependencies available. Its `fsspec` pin differs from the training image, so keep these environments separate. Stage once on shared storage:

```bash
python3 -m pip install --target /path/to/preparation-packages -r requirements-data.txt -c constraints-data.txt
PYTHONPATH=/path/to/preparation-packages python3 1.prepare-llm.py \
  --output /path/to/shared/qwen3 --records 1024 --download-weights
```

Use an empty output directory. Use a fresh preparation-package directory too; an existing `pip --target` directory may retain earlier modules. [constraints-data.txt](constraints-data.txt) records the resolved Python 3.10 preparation versions separately from training dependencies. The command downloads public pinned artifacts without authentication or remote model code. Documents remain separate, are right-truncated to 2,048 positions and retain EOS only when the original document end fits. Padding labels use `-100`. Useful tokens count shifted, unmasked prediction targets. The default training interval uses 640 documents across twenty updates, with a global batch of thirty-two documents. The remaining prepared documents allow separate short checks.

The local Python `3.12` preparation environment produced its manifest and then failed during PyArrow interpreter finalization. A manifest alone therefore does not establish successful preparation. Check the process exit status and file hashes; the Spain Python `3.10` preparation exited successfully.

The existing `1.prepare.sh --llm` automates preparation inside an assigned two-node Slurm allocation using already staged SquashFS images. Export `LAB_IMAGE`, `VLLM_IMAGE`, their expected `LAB_IMAGE_SHA256` and `VLLM_IMAGE_SHA256` digests, a fresh shared `LLM_DATA_DIR` and a fresh node-local `LLM_PREPARATION_PACKAGES` path. The script verifies both images against the supplied digests on both nodes, installs the constrained preparation packages on one explicit node, downloads the pinned model and attributed data, and reads the token hashes from both nodes. It preserves failed directories instead of silently reusing them. The Spain execution completed in 102 seconds with the same data manifest as the qualified training runs and all 42 installed preparation-package versions matching `constraints-data.txt`; this used existing images rather than building a new deployment.

After preparation, export the assigned `PARTITION`, comma-separated `ASSIGNED_NODES`, `GPUS_PER_NODE=8`, private `LOGIN_BIND_IP`, `PROMETHEUS_PORT`, verified `NCCL_SOCKET_IFNAME`, and an existing writable `LLM_RESULTS_DIR`. Run `bash facilitator/prepare-login.sh --llm` from the matching companion to create `$LLM_RESULTS_DIR/llm-environment.sh`. It validates the prepared inputs and refuses to overwrite an existing environment. Supply the actual companion and environment paths to participants. Image distribution, recorded participant checkpoints and event permissions remain separate provisioning responsibilities.

## Run inside an assigned PCS allocation

The launcher uses an existing Slurm allocation with Pyxis/Enroot. The tested shape grants eight GPUs and 192 CPU cores per node. Request GPUs explicitly, for example `--gres=gpu:8`, as well as the assigned node count and CPU budget. Confirm current occupancy before allocating resources. No script purchases capacity or evicts jobs.

Set paths to the prepared image, data and shared results, then run from this directory inside the allocation:

```bash
export LAB_IMAGE=/path/on/each/node/aim347-lab.sqsh
export LLM_DATA_DIR=/path/to/shared/qwen3
export LLM_RESULTS_DIR=/path/to/shared/results
export GPUS_PER_NODE=8
export NCCL_SOCKET_IFNAME='=enp71s0'  # Replace with the verified private interface.
bash 2.run-llm.sh --output /results/baseline \
  --allocation-label pcs-g7.48xlarge-2nodes-16gpu --deterministic
```

The example interface belongs to the measured Spain nodes. Discover the interface in another allocation. The result directory must not already exist. The launcher requires the AWS Libfabric NCCL network and EFA provider for every training configuration. It does not run a network microbenchmark inside the measurement.

Training writes one update log and one completion record per rank, plus `summary.json`. The continuous post-warmup window includes input wait, H2D, forward/loss, backward/collectives, optimizer, accounting, logging and the final device/rank boundary. It reports the maximum rank duration and globally summed useful tokens. GPU UUIDs, CPU affinity and peak allocated CUDA memory are recorded. The first two updates are warmup in the default configuration.

Inspect a separate short profile with `--profile`. Its trace includes phase markers and CUDA kernels; profiled runs are rejected by the performance comparison command. Candidate options include `--trim-padding`, `--no-reshard-after-forward`, `--compile-loss`, `--compile-blocks`, `--fused-optimizer`, `--microbatch`, `--delay-gradient-sync` and `--activation-checkpointing`. Each option requires numerical and performance qualification on the target allocation. The global batch and sample membership per update remain fixed when microbatch changes.

`--trim-padding` removes only suffix columns that are padding for the entire local microbatch. It preserves document positions and all useful loss targets, but failed the fixed full-model numerical policy on Spain even with deterministic algorithms enabled. Loss compilation also failed that policy with fixed input shapes, including a run with `--compile-preserve-casts`. Neither change is an accepted performance stage. Keeping weights after forward reduces re-gather work at the cost of GPU memory and is tested separately.

The tested training image lacks Python development headers needed by Triton compilation. For that exact Ubuntu `22.04` / Python package `3.10.12-1~22.04.17` image, prepare checksum-pinned headers without installing host packages:

```bash
bash prepare-compile-headers.sh /path/to/shared/python-headers
export LLM_PYTHON_HEADERS_DIR=/path/to/shared/python-headers
mkdir -p /path/on/each/node/compiler-cache
export LLM_COMPILER_CACHE_DIR=/path/on/each/node/compiler-cache
```

Prepare the cache directory on every assigned node. The launcher mounts these directories and limits compiler workers to one thread per training rank. Compilation, model loading and warmup remain part of allocation time even when they fall outside the steady measurement. The repaired compile path is still under hardware qualification.

Compare completed unprofiled runs using their recorded invariants:

```bash
python3 7.compare-llm.py /path/to/results/baseline/summary.json \
  /path/to/results/candidate/summary.json
```

One pair establishes an observation. Adoption also requires repeated comparisons, an explained profile mechanism, and numerical checks. For short correctness-only runs, add `--verify-numerics` to retain every local parameter/gradient shard and initial-to-final parameter delta. This creates large files outside the timed window. Compare them with the preregistered dimensionless limits in [configs/llm-numerics-policy.json](configs/llm-numerics-policy.json):

```bash
python3 7.compare-llm.py /path/to/numerics-baseline /path/to/numerics-candidate \
  --numerics-policy configs/llm-numerics-policy.json
```

The first full-model numerical control on Spain failed: repeating the unchanged baseline produced a gradient relative L2 difference of 0.313632 (dimensionless), despite a maximum relative loss difference of 0.000256215 (dimensionless). Those nondeterministic pilot timings remain rejected observations. Strict deterministic controls and the two accepted retention stages below were qualified separately; they do not retroactively qualify the pilots.

`--deterministic` enables PyTorch's strict deterministic algorithms and the required cuBLAS workspace setting before CUDA initialization. On the tested Spain allocation, two strict baseline runs matched exactly across all saved parameters, gradients and parameter deltas. Use it for every baseline and optimized correctness **and performance** run in the qualified comparison path. Do not compare timings across determinism policies or widen the saved numerical tolerances to accept a speedup.

With fixed shapes and strict determinism, `--no-reshard-after-forward` matched the baseline exactly across the saved parameters, gradients, parameter deltas and per-update losses in the two-update numerical check. Three separate performance pairs each processed 443,896 useful tokens across twenty updates, excluding two warmup updates. The baseline took 50.90, 51.29 and 51.22 seconds; retained weights took 49.05, 49.14 and 49.23 seconds. Paired speedup ratios were 1.038, 1.044 and 1.040. A separate strict four-update profile reduced rank-zero all-gather API calls from 592 to 296 while retaining 296 reduce-scatter calls. GPU trace events were incomplete relative to CPU calls, so their summed durations do not establish exposed communication time. This is the earlier two-node comparison; the complete same-work sequence below provides the current cumulative results.

`--forward-prefetch` uses FSDP2's explicit next-block all-gather prefetch. It matched the baseline numerically but did not improve retained-weight training in three paired performance runs: the median speedup ratio was 0.998. `--compile-preserve-casts` preserves eager low-precision cast boundaries inside compiled operators; it does not guarantee identical reduction results. Both block compilation and the narrower `--compile-pointwise` normalization/activation compilation failed the fixed numerical policy, including cast preservation. Adding `--delay-gradient-sync` to retained weights exceeded GPU memory on the tested configuration before an update completed. These candidates are not accepted cumulative stages.

`--retain-between-microbatches` uses standard FSDP2 backward reshard control to retain gathered parameters between accumulation microbatches, then reshards on the final backward before the optimizer step. It requires `--no-reshard-after-forward` and at least two microbatches per update. Gradient synchronization remains unchanged unless separately requested. The two-update Spain comparison matched all saved tensors and losses exactly.

After the expansion to 2,400 GiB completed, three fresh performance pairs on shared Lustre measured baseline-to-forward-retention ratios of 1.027730, 1.030611 and 1.028160. Adding retention between microbatches measured incremental ratios of 1.025520, 1.028118 and 1.026342. Ratios are dimensionless; every run processed 443,896 measured useful tokens with the same strict policy. Both stages met the fixed per-pair minimum of 1.01 and median minimum of 1.02. Separate four-update profiles on all sixteen ranks reduced host all-gather submissions from 296 to 148 while preserving eight microbatches, four optimizer steps and 296 reduce-scatter submissions per rank. GPU annotations are not added to host submissions when counting collectives.

A separate retained-weight run with `--workers 0` matched the baseline exactly but failed performance acceptance: the three incremental speedup ratios were 1.002674, 1.002440 and 1.006437. It is not a third cumulative stage.

Additional standard communication settings were tested on top of both retention stages. Requesting `NCCL_PROTO=Simple`, selecting eight NCCL CTAs, and using the internal NCCL tuner all passed the fixed short-run numerical policy but failed the three-pair performance rule. The Simple request reached every rank, yet the captured kernel names still identified LL; an environment setting alone does not establish the algorithm executed. `--fsdp-blocks-per-group 2` groups adjacent decoder blocks through the standard FSDP2 list-of-modules API. It also passed the numerical policy, but reduced throughput: paired speedup ratios were 0.719485, 0.723309 and 0.725037, all dimensionless. Fewer collective submissions did not establish a faster iteration. Keep the default group size of one block for the accepted path.

## Complete cumulative training comparison

These results are development evidence, not an accepted training curriculum. PI judged the roughly 1.091 dimensionless cumulative ratio and diagnostic-fill removal insufficient for the ordinary-workload teaching objective. The screening cutoffs below were selected by the implementation author and do not represent PI acceptance. A redesigned training case is under investigation; the separately validated config-only recovery improvement is retained.

The complete training sequence was remeasured with all four configurations on the same two-node, sixteen-GPU allocation and expanded shared filesystem. Each run processed 443,896 measured useful tokens across twenty updates with two warmup updates. Forward retention produced paired speedup ratios of 1.020517, 1.023343 and 1.020999; accumulation retention added 1.027549, 1.031220 and 1.033264; `--skip-uninitialized-fill` added 1.035944, 1.033514 and 1.034205. All ratios are dimensionless and pass the existing per-pair minimum of 1.01 and median minimum of 1.02. The median baseline-to-final ratio is 1.090660. The middle repetition reversed stage order.

`--skip-uninitialized-fill` disables PyTorch's diagnostic initialization of otherwise unwritten allocations, while retaining strict deterministic algorithms and the cuBLAS workspace setting. Use it only after verifying that the workload initializes storage before consumption. The exact two-update GPU comparison matched every saved parameter, gradient, parameter delta and per-update loss on all sixteen ranks. A separate four-update profile reduced rank-zero `aten::fill_` calls from 19,101 to 2,587 while preserving 1,971 zero-initialization calls. Nested CPU durations are not added or interpreted as exposed GPU time. Checkpoint runs below retain the default fill setting; this validation does not establish the combined no-fill/config-only-restore path.

```bash
bash 2.run-llm.sh --output /results/retention-and-allocation \
  --allocation-label pcs-g7.48xlarge-2nodes-16gpu --deterministic \
  --no-reshard-after-forward --retain-between-microbatches --skip-uninitialized-fill
```

## Checkpoint state and full-campaign comparison

Use shared Lustre storage for every multinode checkpoint run. Both modes save model parameters, AdamW state, scheduler, next data position and per-rank Python/NumPy/PyTorch/CUDA RNG state. They use the same checkpoint update indices. PyTorch DCP flushes checkpoint files; the recipe publishes `COMPLETED.json` only after every rank finishes its write and the sidecars and completion marker are fsynced. Pending saves receive no durable credit. A final drain is included.

An `S3_ARCHIVE.json` index at the campaign or segment root marks relocated experiment data. Completion metadata may remain there after tensor files have moved. Local checkpoint discovery and explicit restore reject such indexed sources; restore the verified archive into a new directory first. Participant-provided checkpoints must remain locally complete.

Run a fresh campaign within one retained allocation. The output path here must be the shared host path mounted by the launcher, rather than its `/results` alias:

```bash
python3 13.measure-recovery.py --output "$LLM_RESULTS_DIR/campaign-sync" \
  --allocated-gpus 16 --interrupt-after-update 5 -- \
  bash 2.run-llm.sh --allocation-label pcs-g7.48xlarge-2nodes-16gpu \
  --config /opt/aim347/configs/llm-recovery.json --checkpoint-mode sync --deterministic
```

The controller starts training, deliberately exits this job's ranks after the specified completed update, selects the latest completed checkpoint and launches a fresh training process. Recovery must pass the original interruption point and publish the final checkpoint. Use a different output directory and `--checkpoint-mode async` for the comparison. The recovery configuration matches the completed Spain pilot: six updates, checkpoint indices `[2,4,6]` and interruption after update five.

The current controller requires `controlled_interruption: true` in each rank's final update record before restarting. The trainer writes that field only on its deliberate interruption path. A nonzero launcher exit and a matching update number alone are insufficient. Keep the trainer and controller at the same revision; historical campaigns without this field were checked separately against their retained torchrun exit-status-75 logs.

`campaign.json` reports **Training goodput** as final retained useful tokens divided by the controller's full elapsed interval. The interval includes initial model loading, training, saves, failure detection, the recovery gap, child-step relaunch/wait, restore, replay, final durable completion and process exit. Replayed tokens are counted once in final retained progress. The same GPUs remain allocated throughout; initial allocation waiting and preparation before controller start are outside this campaign. Tokens per allocated GPU-hour use the same explicit interval.

A standalone `--resume /path/to/completed/checkpoint` invocation instead reports restoration, training and final flush. Its `training_goodput.scope` excludes model loading, sharding, optimizer construction, the earlier failed process and downtime. Combine a recorded earlier segment only with trustworthy interval and retained-progress evidence; an attendee resume is not a fresh complete campaign.

The Spain pilot completed both modes on shared FSx. Synchronous recovery retained 137,174 tokens in 440.50 seconds; asynchronous recovery retained the same tokens in 460.12 seconds. The asynchronous save at update four was still pending at interruption, so recovery selected update two and replayed three updates, compared with one replayed update for synchronous saving. This pair does not establish an asynchronous speedup. Reboot and replacement-node persistence were not tested.

After expansion, per-OST inspection showed that the older object-storage target was nearly full despite aggregate free space. A new task-scoped output directory was assigned one stripe on the target with sufficient space, with no change to existing files or mount defaults. Both arms inherited that identical layout. Three new synchronous campaigns took 601.731, 512.610 and 506.832 seconds; asynchronous campaigns took 572.781, 560.442 and 471.883 seconds. Every campaign retained 137,174 tokens and completed update six. The second pair regressed, so asynchronous saving remains unqualified despite the better median. All 336 resulting checkpoint files, totaling 911,516,494,575 bytes, were verified on the intended target and retained. This is an explicit single-target placement result, not an improvement attributed to striping or storage expansion.

Before another large campaign, inspect both `lfs df` and `lfs getstripe` for the actual output directory and existing files. Budget for completed checkpoints, interrupted partial writes and resumed checkpoints together. Aggregate filesystem free space does not guarantee space on the targets selected by a file layout. Use a new, scoped output directory only when placement is justified by the live inventory, and keep the same placement for both comparison arms. Preserve participant checkpoints and unrelated evidence. Authorized rolling relocation may reclaim exact inactive experimental files only after complete inventory, S3-validated checksums and size readback; retain manifests, logs and retrieval indexes. An ETag is not a SHA-256 digest. Previously verified objects need no redundant full download before reclaim.

`--checkpoint-writer-threads` controls PyTorch DCP's file-writer concurrency per rank and defaults to one thread. It preserves the saved state, update indices and file synchronization. The four-thread synchronous Spain campaign restored update four, replayed update five and completed update six, retaining 137,174 tokens in 502.13 seconds for 273.19 tokens/s of Training goodput. It did not outperform the earlier single-thread campaign. The runs occurred at different times, so this exploratory comparison does not isolate writer concurrency from storage conditions. Reserve storage for every retained checkpoint and interrupted partial save before running another campaign.

`--checkpoint-copy-ahead-bytes` exposes DCP's standard per-thread GPU-to-CPU copy-ahead budget, defaulting to 10,000,000 bytes. The synchronous single-thread writer can overlap copying and serialization; multiple writer threads use the serial loader, and asynchronous staging disables this copy-ahead path. A single-node, eight-GPU NVMe diagnostic compared the default with 67,108,864 bytes. Both restored update four and completed update six with 137,174 durable tokens, but the larger budget took 277.86 seconds versus 222.05 seconds for the default. This pair does not establish an improvement or qualify shared-FSx behavior. File synchronization and checkpoint frequency were unchanged.

## Avoid redundant pretrained loading during restore

`--resume-from-config` avoids loading pretrained weights immediately before overwriting them with a complete training checkpoint. Fresh runs still load the pinned pretrained revision. Resume constructs the pinned configuration without weight initialization, explicitly ties shared weights, shards through FSDP and restores the model, optimizer, scheduler, data position and all rank-local random-number states before training. The tested Qwen3 rotary buffer is reconstructed from its configuration. Full final model/optimizer checkpoint entries and all sixteen ranks' RNG and update logs matched exactly after the two resumed updates.

Three same-work, unprofiled shared-filesystem pairs, including an order reversal, took 522.016/427.750 seconds, 509.132/487.020 seconds and 520.697/429.052 seconds for normal/config-only resume. Their dimensionless speedup ratios are 1.220376, 1.045403 and 1.213599. Each campaign retained 137,174 useful tokens, with the same update-five interruption, update-four restore and update-six completion. This improves full-campaign Training goodput through model loading, not checkpoint write throughput. Parallel pretrained loading alone failed its reversed confirmation and remains unqualified.

```bash
python3 13.measure-recovery.py --output "$LLM_RESULTS_DIR/campaign-config-resume" \
  --allocated-gpus 16 --interrupt-after-update 5 -- \
  bash 2.run-llm.sh --allocation-label pcs-g7.48xlarge-2nodes-16gpu \
  --config /opt/aim347/configs/llm-recovery.json --checkpoint-mode sync --deterministic \
  --no-reshard-after-forward --retain-between-microbatches --resume-from-config
```

Use `--diagnostic-profile` on the campaign controller only for investigation. It records setup/restore and training/save traces separately and includes profiler overhead in the unchanged full campaign interval. Diagnostic results are marked `profiled=true` and cannot be accepted or published as performance measurements.

## Serving placement and request quality

The serving launcher splits each node's assigned GPUs into disjoint vLLM replicas. It checks the pinned model snapshot and records GPU UUIDs and exact server commands. Start it inside an assigned allocation, setting `VLLM_IMAGE`, `LLM_DATA_DIR`, `LLM_RESULTS_DIR`, `GPUS_PER_NODE`, `NCCL_SOCKET_IFNAME` and a unique `SERVING_RUN`:

```bash
export VLLM_IMAGE=/path/on/each/node/vllm-v0.20.2.sqsh
export SERVING_RUN=placement-tp8
bash 8.serve-llm.sh --tensor-parallel-size 8
```

Each node serves its first replica on port `8100`; further replicas use consecutive ports. Wait for every `/health` endpoint. The service uses BF16, a 4,096-token context, prefix caching disabled and the engine generation defaults. Training and serving use the same fixed pretrained revision; the short training exercise is not assumed to improve serving quality.

The serving wrapper and manifest collector use `SLURM_JOB_CPUS_PER_NODE`, including Slurm's `192(x2)` notation, so they can run from an allocation shell without `SLURM_CPUS_ON_NODE`. They require a homogeneous CPU allocation. After each replica writes its record, collect the actual commands and assigned GPU identities in that same shell:

```bash
python3 lib/serve_llm.py --collect-manifest \
  --output "$LLM_RESULTS_DIR/$SERVING_RUN" --gpus-per-node "$GPUS_PER_NODE" \
  --tensor-parallel-size 8 --max-num-batched-tokens 8192
```

Use the exact TP degree, token budget, sequence limit and optional model path from the launch. The collector writes `manifest.json` and `endpoints.txt` in the run directory and refuses conflicting commands or records from another Slurm job/node assignment. It does not start servers or establish endpoint health. Use a fresh directory for each placement and retain previous measurements. The older manual-manifest example below describes the historical pilot; it does not replace the collector's current provenance checks.

The pilot request files contain authored JSON tasks: integer sequence generation for short input/long output and warehouse-record extraction for long input/short output. They are finite quality fixtures, not a general model-quality benchmark. Before the first serving run, the pilot policy selected TTFT at most 2 seconds and client TPOT at most 0.05 seconds for an interactive structured-response scenario. The pilot compares 128 requests per task under 64 closed-loop client slots, with `max_tokens=512`, temperature zero, seed `347` and thinking disabled.

[9.load-llm.py](9.load-llm.py) requires a server manifest with `gpu_budget`, `placement`, `server_batch` and `workload` matching the actual launch. It warms every endpoint equally, routes inside the measured interval and preserves every response, error and timeout. **Serving goodput** counts completed exact-JSON answers meeting both latency conditions, divided by the measured duration. One-token output has undefined TPOT and remains in offered totals. Client TPOT uses output-token usage with first/last content receipt times; content chunks are not counted as tokens. Raw engine `/metrics` snapshots before and after the workload retain server ITL separately.

For the two-node TP degree eight pilot, write a manifest from the fixed configuration and query the assigned node addresses. Run the client while the server job is healthy:

```bash
python3 - <<'PYTHON'
import json
import os
from pathlib import Path
cfg = json.loads(Path('configs/llm.json').read_text())
manifest = {
    'gpu_budget': 16,
    'placement': {'tensor_parallel_size': 8, 'replicas': 2},
    'server_batch': {'max_num_seqs': 256, 'max_num_batched_tokens': 8192},
    'workload': {key: cfg[key] for key in ('model_id', 'model_revision', 'tokenizer_revision')},
}
manifest['workload'].update(precision='bfloat16', vllm_version='0.20.2',
                            prefix_cache=False, max_model_len=4096,
                            gpu_memory_utilization=0.85)
run = Path(os.environ['LLM_RESULTS_DIR']) / os.environ['SERVING_RUN']
records = [json.loads(path.read_text()) for path in run.glob('node-*/replica-*.json')]
uuids = sorted(value for record in records for value in record['gpu_uuids'])
assert len(records) == 2 and len(uuids) == len(set(uuids)) == 16
manifest['allocation'] = {'gpu_uuids': uuids, 'nodes': 2,
                          'cpus_per_node': int(os.environ['SLURM_CPUS_ON_NODE'])}
Path('serving-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
PYTHON
endpoints=()
while read -r node; do
  address=$(scontrol show node "$node" -o | tr ' ' '\n' | sed -n 's/^NodeAddr=//p')
  endpoints+=("http://$address:8100")
done < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
python3 9.load-llm.py "${endpoints[@]}" \
  --server-manifest serving-manifest.json \
  --requests-file configs/short-input-long-output.jsonl \
  --output "$LLM_RESULTS_DIR/serving-short.json" --concurrency 64 \
  --max-tokens 512 --ttft-slo-seconds 2 --tpot-slo-seconds 0.05
```

Use the actual replica count, TP degree and server settings in other configurations. A manifest declaration does not verify a server by itself; retain its command and GPU records. Include the sorted actual GPU UUIDs and node/CPU assignment in the manifest's `allocation` object. Comparisons reject different recorded allocations; absent records remain explicitly unverified.

Compare TP placement at the same GPU budget and fixed load. Then hold placement and load fixed while changing `--max-num-batched-tokens` or `--max-num-seqs`. A client-concurrency change is a separate load experiment. Stop only the serving job or launcher you started; its cleanup targets its own replica process groups. Serving hardware results are still pending qualification.

Compare saved results with the corresponding experiment type:

```bash
python3 7.compare-llm.py /path/to/placement-before.json \
  /path/to/placement-after.json --serving-experiment placement
python3 7.compare-llm.py /path/to/batch-before.json \
  /path/to/batch-after.json --serving-experiment server_batch
```

The quality check ignores object-key ordering and whitespace but preserves JSON value types. A boolean is not an integer answer.

The longer server-budget comparison used 2,048 requests per task, the same 64 client slots and three client repetitions per engine configuration. Increasing the TP-degree-two configuration's token budget from 8,192 to 16,384 tokens changed median Serving goodput from 35.425 to 35.446 requests/s for short input and from 73.622 to 73.733 requests/s for long input. These small differences do not establish a useful batching improvement. Malformed stream objects, choices, content and usage counts are retained as failed requests rather than aborting result collection.

The longer placement comparison held the sixteen-GPU budget and the 8,192-token server budget fixed. Moving from TP degree eight with two replicas to TP degree two with eight replicas increased median Serving goodput from 26.314 to 35.445 requests/s for short input and from 4.458 to 74.084 requests/s for long input. TP degree one with sixteen replicas measured 21.603 requests/s for short input and 80.808 requests/s for long input, so the best placement differed by task. Every response passed the JSON quality check; the long-input difference includes the number of requests meeting both latency constraints. These are three client repetitions within each engine deployment, not three independent deployment repetitions.

The serving and earlier checkpoint measurements above precede expansion of the rehearsal filesystem from 1,200 GiB to 2,400 GiB. The later training comparison explicitly reran both baseline and candidates after expansion completed. Capacity scaling also changes aggregate provisioned throughput. Hold new writes during the capacity transition, verify shared I/O before correctness runs, and wait for storage optimization to finish before performance qualification. An old-capacity baseline cannot establish a software-only speedup for a new-capacity candidate.

## Publish a completed measurement

[14.publish-llm.py](14.publish-llm.py) reuses the existing Pushgateway helper after the measured job. It publishes separate metric names and Pushgateway groups for full campaign goodput, process-segment goodput and serving goodput. Publishing a campaign and its resumed segment under the same run/config keeps both measurements:

```bash
python3 14.publish-llm.py "$LLM_RESULTS_DIR/campaign-sync/campaign.json" \
  --pushgateway http://LOGIN_PRIVATE_ADDRESS:9091 \
  --run-id llm-recovery --configuration sync --instance-type g7.48xlarge
```

Before starting the existing monitoring helper on the assigned login node, select the LLM dashboard and the actual replica ports. For the tested placement of four replicas per node:

```bash
export GRAFANA_DASHBOARD_FILE=./llm-dashboard.json
export AIM347_SKIP_LEGACY_ENV=1
export VLLM_METRICS_PORTS=8100,8101,8102,8103
bash 0.deploy-observability.sh login
```

Set `LOGIN_BIND_IP` and `COMPUTE_NODES` to the assigned private login address and comma-separated Slurm node names as required by the existing helper. Its default dashboard and serving port remain available for historical runs. The Lustre host collector needs the privileges used by the helper's systemd service; an unprivileged `lctl` failure is missing telemetry.

Import [observability/llm-dashboard.json](observability/llm-dashboard.json) into the lab Grafana instance with the existing `aim347-prometheus` data source. Select the recorded run and assigned nodes. Completed-result gauges remain constant after publication; the GPU/EFA/Lustre panels show live node-wide telemetry. Use the recorded UTC measurement boundaries to select the time range. UTC is not used to calculate elapsed duration. Configure the vLLM scrape targets to match the actual replica ports starting at `8100`; the historical deployment default is port `8000`.

The dashboard source is a draft until its live queries and rendered panels are checked. CPU fixtures, missing hardware provenance, incomplete results and profiled/numerical training runs are rejected by the publisher. Serving records without actual allocation evidence can be inspected locally but cannot be published as verified hardware measurements.

## Supporting checks

Run the existing local suite with the pinned CPU or CUDA dependencies available:

```bash
PYTHONPATH=lib python3 -m unittest discover -s tests -v
```

CPU fixtures check small-Qwen loss/gradient/update equivalence, DCP state restoration and request accounting. Target-hardware runs establish whether the actual example works. Preserve unsuccessful launches and their elapsed resource use alongside successful runs.
