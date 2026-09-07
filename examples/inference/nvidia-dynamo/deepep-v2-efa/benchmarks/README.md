<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0 -->
# Benchmarks: DeepEP-V2 over EFA (measured on the sibling vLLM sample's image)

Concurrency sweep of a live DP16/EP16 serve — same pods / same day / same (earlier-revision) probe
for both modes; the non-eager mode came from applying [vLLM #52632](https://github.com/vllm-project/vllm/pull/52632)
**in-pod** (patch + engine restart in the same pods), not from a separate image. These tables were
measured on the **sibling vLLM sample's image** (`vllm serve` — no Dynamo front in the loop), **not
re-measured on this Dynamo image**; they are quoted here as the shared-substrate datapoint. The
**shipped** `recipe/benchmark.sh` + probe fire a different population (concurrency × 5 requests per
level, unique prompt prefixes, `ignore_eos` — see the methodology caveats), report aggregate output
tok/s + per-request latency percentiles, and write raw JSONL under `raw/` (gitignored; when run
in-pod, persist the raws with the `kubectl cp` step in the main README **before** deleting the pods).

The **eager** table below is the shipped, supported path (`SERVE_ENFORCE_EAGER=1`) and was measured on
the **shipped pin** (`e2f993dc4`), but on the sibling vLLM sample's image and with an earlier revision
of the probe (see the methodology caveats below). It is **not** re-measured on this Dynamo image, and
the checked-in probe measures a different population, so a stock rebuild will **not** reproduce it. The
**non-eager** table is **historical**: it needed [vLLM #52632](https://github.com/vllm-project/vllm/pull/52632) applied
as an unmerged cherry-pick (that guard is only on the 0.26 line, which regresses `deepep_v2` combine here
— see the main README "eager vs non-eager"), so a stock rebuild does **not** reproduce it either. Both
are kept for the eager-vs-non-eager comparison, not as paths this sample ships.

## Environment provenance

**Substrate numbers.** Everything below is the measured provenance of the *vLLM-image* runs; no
request in these tables traversed `dynamo.frontend` or `dynamo.vllm`. Rows record the **resolved**
values of that measurement (what was actually built and run), not the configured ones — where the
current recipe has since diverged, the row says so.

| | |
|---|---|
| Instance | 2× p5en.48xlarge (H200), cross-node EFA |
| Transport | DeepEP-V2 `ElasticBuffer`, NCCL-GIN CPU-proxy (`NCCL_GIN_TYPE=2`, `OFI_NCCL_GIN_GDAKI=0`), `efa-direct` |
| Model | `Qwen/Qwen3-30B-A3B-FP8`, DP16/EP16 (`--enable-expert-parallel --all2all-backend deepep_v2`) |
| vLLM | `0.22.1rc1.dev283+ge2f993dc4` — the merge commit of [PR#41183](https://github.com/vllm-project/vllm/pull/41183) (first, and only measured-working, `deepep_v2` backend). **This is the shipped `Dockerfile` pin** — the vLLM substrate matches; the serving front and the rest of the stack do not (see the other rows). |
| ai-dynamo | **none** — these tables predate the Dynamo front: measured via `vllm serve` on the sibling vLLM sample's image (no `dynamo.frontend`, no `dynamo.vllm`, no `--discovery-backend file`). This sample pins `ai-dynamo{,-runtime}==1.3.1`; producing Dynamo-fronted numbers is the queued re-measure (see Honest caveats). |
| Stack | torch 2.11.0+cu130, nvidia-nccl-cu13 2.30.4, DeepEP `28d1f7fb` (the **effective** tree: the configured `b306af06` + PR#612 `git merge` fast-forwarded to it), aws-ofi-nccl `9c44d34` + [#1351](https://github.com/aws/aws-ofi-nccl/pull/1351) commit 1/2 (`c2e773df`). The **current recipe differs**: DeepEP = `amazon-contributing` fork @ `97d8f9bc`, aws-ofi-nccl = released `v1.21.1` — a further reason a rebuild does not reproduce these tables. |
| Serve fingerprint | `vllm-0.22.1rc1.dev283+ge2f993dc4-dp16-ep-1f3ed125` (that run's vLLM engine fingerprint, recorded as-is; not reproducible from this sample) |
| Probe | stdlib urllib+threads, 127.0.0.1 loopback, `max_tokens=128`, `temperature=0.0` (greedy), concurrency 1/8/16/32/64, **one shot per level** (N requests at concurrency N), identical prompt every request — an **earlier revision** of the checked-in probe (see caveats) |
| QP knobs | `EP_EFA_MAX_QPS=2`, `EP_EFA_RDMA_GBS=25.0` — the serve defaults **at measurement time** (the deepseek tree's PR#612 knobs). The current recipe pins the `amazon-contributing` fork, which resolves QP count and link rate internally; neither env exists (or is set) in this sample any more. |
| Date | 2026-08-14 — the vLLM-image measurement date (this PR's own Dynamo E2E evidence dates 2026-09-04) |

**Probe methodology caveats for these tables** (the checked-in `recipe/benchmark_probe.py` has
since been upgraded — see below — so a re-run will NOT reproduce these tables exactly):

- **One shot per level**: at `conc=1` the "percentiles" are a single observation (visible above:
  p50 = wall exactly). Rows describe one batch of N concurrent requests, not a distribution.
- **Identical prompt, temperature 0**: vLLM's prefix cache serves every prompt after the first, so
  prefill is ~free — these are **decode-focused** numbers, not end-to-end serving numbers.
- **No `ignore_eos`**: the 128-token cap happened to bind for this prompt+model (4.8 × 26.91 ≈ 129),
  so the denominator was stable here, but the harness did not enforce it.

The current `recipe/benchmark_probe.py` fixes all three (5× requests per level so percentiles are
distributions; a unique prompt prefix per request so prefix-cache can't serve prefill;
`ignore_eos: true` pinning tokens = `max_tokens`; failures excluded from percentiles/throughput and
failing the run). The new probe changes the sample count, the cache behaviour and the decode length,
so the measured population differs — its numbers are **not directly comparable** to these tables;
re-measure before quoting.

## Results

**What these tables are:** one concurrency sweep of the same backend per mode — the two endpoints
of each sweep are NOT a before/after comparison. In the **eager** sweep per-stream rate stays pinned
at ~4.75 tok/s while aggregate scales ≈ concurrency × constant (4.8 → 301.8 tok/s for 64× concurrency
with wall flat at ~27 s), so no saturation point is reached. The **non-eager** sweep does show a
wall-time rise at `conc=64` (26.61 → 34.18 s; per-stream 3.75 tok/s) — the one place in either sweep
where the server is visibly under strain — while still stopping short of a measured saturation point.
Read the numbers as the **DeepEP-V2-over-EFA per-stream latency floor at each concurrency**, not as a
throughput ceiling — a saturation point was never measured (concurrency was not pushed until wall
time rose across the whole sweep).

### Eager (`--enforce-eager`; the shipped default, measured on the shipped pin `e2f993dc4` — on the sibling vLLM image, not this sample) — 121/121 HTTP 200 (sweep = 1+8+16+32+64 requests)

| conc | agg tok/s | wall s | p50 s | codes |
|---|---|---|---|---|
| 1 | 4.8 | 26.91 | 26.91 | 200 |
| 8 | 37.1 | 27.63 | 27.63 | 200 |
| 16 | 74.6 | 27.45 | 27.43 | 200 |
| 32 | 151.9 | 26.97 | 26.95 | 200 |
| 64 | 301.8 | 27.15 | 27.11 | 200 |

### Non-eager (default compilation; historical — measured with the then-unmerged upstream guard, [vLLM #52632](https://github.com/vllm-project/vllm/pull/52632)) — sweep 121/121 HTTP 200

| conc | agg tok/s | wall s | p50 s | codes |
|---|---|---|---|---|
| 1 | 5.0 | 25.48 | 25.48 | 200 |
| 8 | 39.9 | 25.69 | 25.68 | 200 |
| 16 | 76.9 | 26.64 | 26.19 | 200 |
| 32 | 153.9 | 26.61 | 26.17 | 200 |
| 64 | 239.7 | 34.18 | 33.91 | 200 |

(The non-eager run's raw log totaled 153/153 HTTP 200 = a 31-request warm-up ramp (1+2+4+8+16 at
`max_tokens=8`) + 1 coherence check + the 121-request sweep above. The table rows are the sweep
only — identical 121-request methodology to the eager table; the extra 32 requests were the
run's warm/health phases, not extra sweep samples. That log was not persisted off-pod — see
Honest caveats. Note the asymmetry it creates: the non-eager sweep started against a **warm**
server — 31 warm-up requests plus prefix caching over an identical prompt — while the eager
sweep ran with no warm-up, which favors non-eager most at `conc=1`.)

### Eager vs non-eager (agg tok/s; delta = non-eager relative to eager)

Deltas are computed from the underlying **wall times** (the same token total divides out), not from
the 1-decimal rate column — at `conc=1` the rounded rates suggest +4.2% while the walls
(26.91 s / 25.48 s) give the real **+5.6%**. Single sweep per mode, so each delta is a single
observation: the sub-10% rows are **within run-to-run variation** (and carry the warm-cache
asymmetry noted above, strongest at `conc=1`); only the −20.6% row at `conc=64` separates from noise.

| conc | eager | non-eager | delta (from walls) |
|---|---|---|---|
| 1 | 4.8 | 5.0 | +5.6% (single obs., within run-to-run variation; warm-cache-favored) |
| 8 | 37.1 | 39.9 | +7.5% (single obs., within run-to-run variation) |
| 16 | 74.6 | 76.9 | +3.1% (single obs., within run-to-run variation) |
| 32 | 151.9 | 153.9 | +1.3% (single obs., within run-to-run variation) |
| 64 | 301.8 | 239.7 | −20.6% (wall 27.15s → 34.18s — the one delta large enough to read) |

**Reading:** non-eager is indistinguishable from eager through concurrency 32 (single-observation
deltas within run-to-run variation), then falls ~21% behind at 64 on this shape (the non-eager wall
jumps 27→34 s at c=64 while eager stays flat). Mechanism not root-caused here — candidates are
CUDA-graph capture-size coverage vs per-engine batch shape at high concurrency. Guidance: **eager is
the shipped, supported, zero-patch path and is at worst ~a few percent off non-eager below c=64.**
Non-eager is not a supported flip at the shipped pin (it needs #52632, which is only on the
`deepep_v2`-combine-regressing 0.26 line — see the main README); the numbers above are the
historical comparison, not a production option this sample offers.

## Honest caveats

- Pods were ~1 day old at measurement (accumulated-state can inflate latency; a fresh-pod baseline would
  differ). Fixed 128-token greedy decode. **Single sweep per mode — no variance bars; the c=64 non-eager
  delta was reproduced once.** These are at-scale **throughput + relative-latency** datapoints, not tuned
  TTFT baselines: the ~26–34 s wall reflects 128-token generation, not a latency-optimized single token.
- Both modes: 0 errors, 0 crashes, all HTTP 200. **The per-level raw JSONLs were not persisted
  off-pod** — they were written inside the pods' `emptyDir` and deleted with the pods, so these
  tables cannot be re-audited against their raws. (That is why the main README's benchmark
  procedure now ends with a `kubectl cp` persistence step.) The queued re-measure of this Dynamo
  image on the updated recipe will produce a persisted, auditable set.
- No alternative-backend baseline was measured: every table is `--all2all-backend deepep_v2`. The sweeps show the EFA path works and how it scales — not that it outperforms vLLM's default all-to-all on this hardware.
