# SmolLM2 training, recovery and serving on AWS PCS

This AIM347 companion uses `HuggingFaceTB/SmolLM2-1.7B` throughout. [configs/llm.json](configs/llm.json) pins pretrained weights and tokenizer to `effd688a12921b4cc83e3312b6feb579f70f9c71`; [configs/llm-recovery.json](configs/llm-recovery.json) extends the same model to forty updates. The model uses the Llama architecture, tied embeddings, 24 layers and 32 attention/KV heads. Serving loads the original pretrained weights rather than the short training run's updates.

## Prepare one model with training and recovery inputs

Use a pre-provisioned AWS PCS cluster with Pyxis/Enroot, an assigned pair of `g7.48xlarge` nodes, eight GPUs and 192 Slurm CPUs per node, and FSx for Lustre shared with the login node. This example creates no capacity. Verify the assigned nodes' occupancy, GPU processes, EFA interface and per-OST storage headroom before using them. The [PCS architecture](../../../architectures/aws-pcs/README.md) is an infrastructure starting point, not evidence that participant permissions or this workload have been rehearsed.

Training uses PyTorch `2.9.1+cu130`, Transformers `4.57.6`, FSDP2, BF16 computation, FP32 reductions and FP32 master parameters. The [Dockerfile](Dockerfile), [requirements.txt](requirements.txt) and [constraints.txt](constraints.txt) pin the runtime. The retained training SquashFS has SHA-256 `4624e0ccf1e5dec07be1eea75eefcdeafa8646035591c13165dacec06b6f07cc`; serving uses vLLM `0.20.2`, with retained SquashFS SHA-256 `ea16c095d375012bc608d41eb7db8f8ff5f31e685b55c8b18021017fa5070d79`. Rebuilt images need fresh validation. Preparation installs [requirements-data.txt](requirements-data.txt) with [constraints-data.txt](constraints-data.txt) separately inside the training image. Do not reuse a partially populated package directory.

From the companion directory in the assigned login-node allocation, export actual paths and assignment:

```bash
export PCS_SLURM_VERSION=YOUR_CONFIGURED_SLURM_VERSION
export PARTITION=YOUR_ASSIGNED_PARTITION ASSIGNED_NODES=NODE_A,NODE_B
export GPUS_PER_NODE=8 NCCL_SOCKET_IFNAME='=YOUR_PRIVATE_INTERFACE'
export LAB_IMAGE=/absolute/staged/aim347-lab.sqsh
export VLLM_IMAGE=/absolute/staged/vllm-v0.20.2.sqsh
export LAB_IMAGE_SHA256=4624e0ccf1e5dec07be1eea75eefcdeafa8646035591c13165dacec06b6f07cc
export VLLM_IMAGE_SHA256=ea16c095d375012bc608d41eb7db8f8ff5f31e685b55c8b18021017fa5070d79
export LLM_DATA_DIR=/absolute/shared/new-smol-data
export LLM_RESULTS_DIR=/absolute/shared/new-smol-results
export LLM_PREPARATION_PACKAGES=/absolute/node-local/new-preparation-packages
export LOGIN_BIND_IP=YOUR_PRIVATE_LOGIN_IP PROMETHEUS_PORT=9092
export LLM_CONFIG=configs/llm.json LLM_PREP_RECORDS=160 LLM_MIN_DOCUMENT_TOKENS=4096
bash facilitator/prepare-login.sh --llm-prepare
```

Replace the uppercase placeholders before execution. Both data and result directories must be new. The helper checks the exact allocation and image digests, downloads the pinned model once, and prepares `tokens/` with 160 documents and `recovery/tokens/` with 1,280 documents. Both select the first eligible documents with at least 4,096 tokens from the pinned FineWeb-Edu source, independently right-truncated to 4,096 positions. The data fingerprints are respectively `276a090b9451c92d61cdeb1ca627a92bbff1e7512ad49ad7f322fb537546a166` and `b9a7d29b22c2b6cac9455323bea87f2dd69b6a39e73d33bbaa1bbcf4faf06488`. Keep per-document attribution and source URLs: FineWeb-Edu uses ODC-By-1.0 and Common Crawl terms also apply; the model uses Apache-2.0.

The result root receives `llm-environment.sh`. Source it in the parent login shell before allocations. For already prepared data, `bash facilitator/prepare-login.sh --llm` validates assets and writes a new environment file without downloading or allocating resources. Never supply an environment from another assignment. Preparation does not generate or copy a recovery recording; stage the locally complete matching Smol recording separately before Lab 5.

## Compare fixed training work

Each invocation reloads the same pretrained model. The five-update configuration uses 32 documents per update and one warmup update. The continuous measured interval counts 524,160 shifted nonpadding prediction targets and includes batch wait, H2D, forward/loss, backward/collectives, optimizer and accounting. Rank durations use the slowest rank, not the sum.

```bash
export LLM_TRANSPORT=efa
bash 2.run-llm.sh --output /results/reference --allocation-label pcs-g7.48xlarge-2nodes-16gpu \
  --attention-implementation eager --activation-checkpointing --microbatch 1
bash 2.run-llm.sh --output /results/endpoint --allocation-label pcs-g7.48xlarge-2nodes-16gpu \
  --attention-implementation sdpa --fused-optimizer --microbatch 2
python3 7.compare-llm.py "$LLM_RESULTS_DIR/reference/summary.json" "$LLM_RESULTS_DIR/endpoint/summary.json"
```

Use fresh output names, then reverse order for the second pair. Eager attention and activation checkpointing form an explicit conservative reference, not a claim about recommended defaults. Decompose attention and checkpoint removal separately. Keep a profile (`--profile`) separate from performance runs. Check `aten::_scaled_dot_product_flash_attention` and its backward operation rather than inferring the kernel from the SDPA interface name.

**Numerical qualification is unresolved.** The recorded eager/SDPA comparison failed the unchanged [numerical policy](configs/llm-numerics-policy.json), including first-update gradient and parameter-delta relative L2 limits. `--verify-numerics` retains full local tensors outside the timer for correctness-only comparisons. Finite/decreasing loss does not establish equivalent updates. Never widen limits to accept a faster candidate.

For transport, hold the SDPA/fused-optimizer/microbatch-two implementation fixed and compare `LLM_TRANSPORT=socket` against `efa`, reversing order. Save launcher logs with exclusive creation before each run. Require actual network selection for all sixteen ranks plus attributable EFA counters. Flags alone do not establish execution. Do not multiply separate compute and transport ratios into an unmeasured cumulative gain.

## Recover durable training progress

Use shared Lustre storage. The trainer saves model parameters, AdamW, scheduler, next data position and every rank's Python/NumPy/PyTorch/CUDA RNG. `COMPLETED.json` is published only after global save completion and flushed sidecars. An archive index does not supply local checkpoint tensors. Budget the actual file stripe targets for all saves and partial writes; pending saves earn no durable credit.

A fresh campaign in an existing sixteen-GPU allocation can use:

```bash
python3 13.measure-recovery.py --output "$LLM_RESULTS_DIR/new-smol-campaign" \
  --allocated-gpus 16 --interrupt-after-update 30 -- \
  bash 2.run-llm.sh --allocation-label pcs-g7.48xlarge-2nodes-16gpu \
  --config configs/llm-recovery.json --data /data/recovery/tokens \
  --attention-implementation sdpa --fused-optimizer --microbatch 2 \
  --checkpoint-mode sync --drain-before-planned-interruption
```

To reproduce the mode comparison, run this command with fresh output roots for SYNC then PROCESS, then PROCESS then SYNC; change only `--checkpoint-mode sync` to `--checkpoint-mode process`. Preserve every `campaign.json` and all-rank logs. Pool equal retained tokens divided by summed campaign seconds, not the mean of rates. Check storage and competing I/O before each run.

The controller deliberately stops only its job's ranks, verifies their controlled-interruption records, restores the latest completed checkpoint and continues to update 40. The planned restart drains saves after useful intervening work. The recorded Smol state-audit campaign used process-based asynchronous saves, restarted at 30, restored 20 and completed 40; all sixteen restored-state audits passed. Its instrumented 237.818 seconds cannot support a performance ratio. PROCESS stages a stable snapshot into pinned/shared CPU memory, fences staging before training reuses its source tensors, and writes through the native DCP background process. Sidecar flushing and state extraction still do synchronous work. The writer must finish before the all-rank completion marker is published; only completed saves earn durable progress. This can overlap checkpoint writing with later updates, not eliminate saving or accelerate restore by itself.

For the participant continuation, use `--resume` on the supplied `update-000020` with the same recovery config, `--data /data/recovery/tokens`, SDPA, fused optimizer and microbatch two, and `--checkpoint-mode sync`. Updates 21–30 replay; 31–40 are new progress. Final durable progress is 5,241,600 targets. A standalone resume reports a process segment, excluding the earlier process, downtime and model construction before its timer. Only `campaign.json` from the full controller interval reports recovery-inclusive Training goodput. Separate uninstrumented SYNC/PROCESS pairs measured 255.121502/233.578463 seconds and, in reverse order, 254.329143/236.860013 seconds for that same durable work. PROCESS goodput rose 9.2230% and 7.3753%, pooled 8.2927%. This reproduces the direction across two short campaigns per mode, not long-run, convergence or full-threshold qualification; it is not a configuration-only restore gain. See [validation](VALIDATION.md) for the full table and scope.

## Serve the base model with text completions

Within one assigned sixteen-GPU allocation, start TP degree eight with two replicas. Change to TP degree one with sixteen replicas only after stopping and waiting for the first launcher's process groups and confirming assigned GPUs are clear. The launcher validates the Smol pin and records exact commands, GPU UUIDs and the Slurm job/node identity. Both placements use BF16, context limit 4,096, prefix caching disabled, 256 maximum sequences and an 8,192-token server budget.

```bash
export SERVING_RUN=smol-tp8
mkdir "$LLM_RESULTS_DIR/$SERVING_RUN"
bash 8.serve-llm.sh --tensor-parallel-size 8 > "$LLM_RESULTS_DIR/$SERVING_RUN/launch.log" 2>&1 &
SERVING_PID=$!
# After all replica records appear:
python3 lib/serve_llm.py --collect-manifest --output "$LLM_RESULTS_DIR/$SERVING_RUN" \
  --gpus-per-node 8 --tensor-parallel-size 8
mapfile -t endpoints < "$LLM_RESULTS_DIR/$SERVING_RUN/endpoints.txt"
( test "${#endpoints[@]}" -gt 0 || exit 1
  for endpoint in "${endpoints[@]}"; do curl --fail --connect-timeout 5 --max-time 10 "$endpoint/health" || exit; done
) && python3 9.load-llm.py "${endpoints[@]}" \
  --server-manifest "$LLM_RESULTS_DIR/$SERVING_RUN/manifest.json" \
  --requests-file configs/range64-fewshot-boundary.jsonl \
  --output "$LLM_RESULTS_DIR/$SERVING_RUN/range-repeat-1.json" --concurrency 64 \
  --warmup-per-endpoint 2 --max-tokens 512 --timeout-seconds 120 \
  --ttft-slo-seconds 2 --tpot-slo-seconds 0.05
# Repeat twice with new output paths before switching placement.
kill -TERM "$SERVING_PID"
wait "$SERVING_PID"
```

The client calls `/v1/completions` with the literal prompt, not `/v1/chat/completions`: this base-model pin does not supply an instruction chat template. The frozen `range64-fewshot-boundary` file repeats one literal request 128 times: emit `values` containing integers 1 through 64 after two short format examples. Its literal `\n\nInput:` stop is forwarded unchanged to the engine, and the entire answer must match exact JSON. This controlled fixture isolates serving placement, not general model quality or extraction ability. A response qualifies only if quality is valid, TTFT is at most 2 seconds and client TPOT is at most 0.05 seconds/token. Failures, timeouts and undefined one-token TPOT remain in offered totals. Streaming chunks are not token counts; the client uses engine usage. Preserve zero-goodput outcomes rather than treating a faster incorrect answer as an optimization.

Copy identical requests to each placement, keep load and generation fixed, and use `7.compare-llm.py BEFORE AFTER --serving-experiment placement`. A batching comparison changes only `--max-num-batched-tokens` at fixed placement and uses `--serving-experiment server_batch`. Neither is a promised win. Retain three clients per deployment and four order-balanced cycles (TP8 → TP1, TP1 → TP8, repeated once), including all startup and serial warmup costs. The completed four-cycle comparison qualified all 3,072 measured responses: pooled TP8 goodput was 52.939285 requests/s versus TP1 45.113797, or 1.173461×, with the same direction in every cycle. This result is limited to the repeated literal range fixture at 64 closed-loop slots, not a general quality, fixed-arrival-rate SLA or cold-start claim; see [validation](VALIDATION.md) for every cycle and the measured denominators. The frozen request-file SHA256 is `60dbda657eaac76d3192c4fe800b3a65a6aeea5bf733b635085bd9b540723ebb`.

## Observability and cleanup

Before participants arrive, source the generated environment and run `bash 0.deploy-observability.sh login` on the designated login node and its `compute` mode on each assigned compute node. This requires the existing Docker/host-service privileges; it is not a participant repair command. The default LLM dashboard uses Pushgateway plus live GPU/EFA/Lustre samples. Ports 8100–8107 cover every serving replica. Missing telemetry stays missing rather than becoming zero.

`14.publish-llm.py` publishes completed unprofiled hardware records to the supplied Pushgateway, distinguishing continuous training, full campaign, resumed segment and serving groups. Match result gauges to run/model/configuration and hardware panels to their UTC measurement interval. A retained gauge is not current activity or numerical acceptance.

Stop only your launcher and temporary download/forward processes, release your own allocation, and preserve the model, shared data, checkpoints and monitoring. Infrastructure teardown belongs to the environment owner.

## Verification

[VALIDATION.md](VALIDATION.md) states measured scope and open qualification. Local supporting checks use `PYTHONPATH=lib python3 -m unittest discover -s tests -v` with the pinned dependencies. Tiny Llama-architecture CPU fixtures verify loss/update accounting and checkpoint state, not GPU numerics or performance. Human-paced participant entry, scheduler/placement transitions and release integration need their own rehearsal.
