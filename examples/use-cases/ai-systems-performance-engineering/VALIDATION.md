# SmolLM2 validation

All results here identify `HuggingFaceTB/SmolLM2-1.7B` revision `effd688a12921b4cc83e3312b6feb579f70f9c71`, on the Spain PCS pair of `g7.48xlarge` nodes with sixteen RTX PRO 4500 GPUs. These measurements do not establish participant-role entry or a complete 120-minute rehearsal.

## Compute and transport

For 524,160 measured useful targets across five updates with one warmup, direct reference/endpoint pairs took 22.686856/4.915718 and 22.804385/4.917109 seconds, with reversed order. A separate decomposition measured SDPA with checkpointing at 6.119650/6.130137 seconds and without checkpointing at 5.089520/5.095201 seconds. Endpoint peak allocated memory was 29,180,033,536 bytes/GPU. These are observations, not an accepted cumulative optimization.

Full-model first-update eager/SDPA gradient relative L2 was 0.053490 against the fixed 0.02 limit; parameter-delta relative L2 was 0.224322 against 0.1. Numerical qualification remains unresolved. The policy file is unchanged.

Same-compute transport pairs measured Socket 7.003694/6.637452 seconds versus EFA 4.881799/4.890088 seconds. All sixteen rank transport selections and NIC counter deltas were retained. Separate profiles preserve equal collective counts; profile overhead and overlapping durations are not unprofiled wall time. Do not multiply this ratio by the compute ratio. The small input does not demonstrate a storage-bandwidth optimization.

## Recovery state

The 40-update recording used 1,280 real 4,096-position documents, checkpoint indices 5/20/40, a planned restart at 30 and restore from 20. All sixteen restored-state audits passed for model, optimizer, scheduler, data progress and all random-number states. Interrupted logs contain updates 1–30; resumed logs contain 21–40; final retained progress is 5,241,600 targets. The recording's 237.81822269799886-second interval contains state-signature instrumentation and is **not performance evidence**, regardless of a metadata eligibility flag. This state audit alone establishes neither a performance gain nor configuration-only faster restoration.

## Recovery-inclusive timing

Each uninstrumented campaign retained 5,241,600 useful targets on the same sixteen GPUs: 40 updates, saves at 5/20/40, a drained planned interruption at 30, restore from 20 and replay of 21–30. Model, input, update order and configuration matched except checkpoint mode; all sixteen interrupted/resumed rank sequences were checked. The controller interval includes initial loading, training, saves, interruption, relaunch, restore, replay and final durable flush, but not allocation wait or preparation.

| Pair and execution order | SYNC job: seconds | PROCESS job: seconds | PROCESS goodput gain |
|---|---:|---:|---:|
| 1: SYNC → PROCESS | 357: 255.121502 | 360: 233.578463 | 9.2230% |
| 2: PROCESS → SYNC | 363: 254.329143 | 362: 236.860013 | 7.3753% |

Pooling equal retained work gives PROCESS/SYNC goodput **1.082927× (+8.2927%)**, from 470.438476 versus 509.450645 total seconds. The direction reproduced in opposite execution orders. These are two short whole-campaign pairs, not long-run, convergence, numerical or full-threshold qualification. The separate instrumented state audit remains correctness-only; planned draining does not test arbitrary failure during an in-flight save.

PROCESS stages a stable snapshot into pinned/shared CPU memory, fences staging before training reuses its source tensors, and writes through the native DCP background process. Sidecar flushing and state extraction still do synchronous work. The writer must finish before the all-rank completion marker is published; only completed saves earn durable progress. This can overlap checkpoint writing with later updates, not eliminate saving or accelerate restore by itself.

Source: `campaign.json` in external evidence directories `smol-sync357-raw`, `smol-process360-raw`, `smol-process362-raw` and `smol-sync363-raw`, under `aim347-llm-20260922`. SYNC363 completed at 11:05:01 UTC on 2026-09-29 with exit code 0:0. Check selected-OST headroom for complete and partial saves before reproduction.

## Serving

The participant workload is the frozen `configs/range64-fewshot-boundary.jsonl` fixture, SHA256 `60dbda657eaac76d3192c4fe800b3a65a6aeea5bf733b635085bd9b540723ebb`. It contains 128 copies of one literal range request with two format examples and the `\n\nInput:` stop. The unchanged validator requires the entire JSON object, TTFT ≤2 seconds and TPOT ≤0.05 seconds/token; no output clipping or repair is used.

Clean serving job361 completed on 2026-09-29 at 10:40:54 UTC with exit code 0:0. All 3,072 measured requests in 24 client files passed exact JSON quality and both latency limits. Four prospectively order-balanced deployment cycles used TP8 → TP1, TP1 → TP8, TP8 → TP1 and TP1 → TP8, with three 128-request clients per deployment on the same sixteen physical GPUs. All twelve matched client comparisons and 72 replica identity records passed validation.

| Cycle | Deployment order | TP8 qualified requests/s | TP1 qualified requests/s | TP8 / TP1 |
|---|---|---:|---:|---:|
| 1 | TP8 → TP1 | 52.977224 | 45.148653 | 1.173395 |
| 2 | TP1 → TP8 | 52.982706 | 45.121158 | 1.174232 |
| 3 | TP8 → TP1 | 52.793050 | 45.005976 | 1.173023 |
| 4 | TP1 → TP8 | 53.004708 | 45.179784 | 1.173195 |

Pooling all measured client intervals gives **52.939285 qualified requests/s for TP8 versus 45.113797 for TP1**, a **1.173461×** ratio; the median cycle ratio is **1.173295×**. Each placement contributed 1,536 qualified requests, over 29.014370 and 34.047234 measured seconds respectively. Startup and serial warmups are retained separately and excluded from those denominators. TP8 was faster in all four cycles for this controlled repeated literal fixture at 64 closed-loop slots. This is not general model or extraction quality, a fixed-arrival-rate SLA, or cold-start throughput evidence.

The literal prompt demonstrates ranges 1–8 and 11–18, then requests 1–64; generation uses temperature 0, seed 347, at most 512 output tokens and the unchanged stop. Both placements use BF16/vLLM 0.20.2, prefix caching off, a 4,096-position context, 256 maximum sequences and an 8,192-token batch budget. Complete source evidence is `smol-clean361-final-analysis.json`, the 24 files in `smol-clean361-raw/`, and `smol-clean361-server-custody.json` with `smol-clean361-final-raw/`, under the external `aim347-llm-20260922` evidence directory. Earlier controls, failures and quality-only pilots remain there, separate from this comparison and outside the participant lesson.

## Local checks and release

The Smol/Llama CPU fixture suite covers manifest endpoint substitution, duplicate rejection and literal stop forwarding in warmups and measured requests. CPU tests are supporting checks, not target-hardware acceptance. The completed scoped serving comparison does not qualify recovery performance, updated preparation, rebuilt images or the participant workflow. The companion commit used by deployment must be replaced with the reviewed unification commit before a participant deployment; no unpublished commit is represented as remotely available.
