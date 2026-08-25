<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0 -->
# TensorRT-LLM NcclEP — MoE expert-parallel all-to-all over AWS EFA (NCCL-GIN CPU-proxy)

This test case runs **TensorRT-LLM's own MoE expert-parallel backend, `NcclEP`**, over **AWS EFA**.
Unlike the sibling DeepEP samples (vLLM, SGLang), TRT-LLM's wide-EP dispatch/combine path is not the
`deep_ep` Python package at all — it is **`nccl.ep`** (the `nccl4py` package + `libnccl_ep`), which
drives its network traffic **through NCCL**, and on EFA that means **aws-ofi-nccl with GIN
(GPU-Initiated Networking) compiled in**, running GIN's **CPU-proxy** mode. The image is built
NGC-from-scratch from public sources only; the recipe takes you from image build → transport smoke
test → a served `/v1/chat/completions` → a concurrency benchmark.

## Pins (every one justified)

| Component | Pin | Why |
|---|---|---|
| Base image | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc24` | GA v1.2.1 has **no NcclEP backend at all** (`nccl_ep_utils.py` absent at that tag), so a 1.3.0rc pin is *required*, not a preference; rc24 is the first release container whose factory ships the `NCCL_EP` arm natively. Re-test + re-pin when a GA carrying the backend appears. |
| EFA installer | 1.48.0 | The userspace the measured substrate ran (2026-08-07). Bumping is a re-measure event. |
| NCCL | pip `nvidia-nccl-cu13==2.30.4` | `is_nccl_ep_installed()` gates on ≥ 2.30.4; the NGC container bakes an older NCCL, so NcclEP is dead-on-arrival as shipped (trap 3). |
| nccl4py | 0.3.1 (ships `nccl.ep`, `libnccl_ep` 0.1.0) | The EP kernel library `NcclEP` binds to. The 0.1.x HT-kernel int64 ABI detail matters (trap 4). |
| aws-ofi-nccl | commit `9c44d34` + PR#1351 head `c2e773d` | GIN plugin with the gdrdrv-2.4 v1-fallback; PR#1351 adds `OFI_NCCL_GDRCOPY_FORCED_PCIE_COPY`. Immutable SHAs — a bare `refs/pull/N/head` is a moving ref. |
| gdrcopy | commit `c91ad9f` (= v2.5.2) | GIN **requires** gdrcopy compiled in (trap 7). Commit pin, not tag. |
| GPU arch | validated on **H200 (p5en.48xlarge)** only | H100/p5 differs only in EFA NIC count (32 vs 16) in the manifest; not re-measured there. |

## How NcclEP gets onto EFA

```
trtllm-serve ── CommunicationFactory.create_strategy()      (reads TRTLLM_FORCE_COMM_METHOD)
                    └─ NcclEP ── nccl.ep (nccl4py / libnccl_ep 0.1.0)
                                    └─ NCCL 2.30.4 (pip) ── GIN device API (NCCL_GIN_TYPE=2, CPU-proxy)
                                                              └─ aws-ofi-nccl @9c44d34 (ncclGinPlugin, gdrcopy)
                                                                    └─ libfabric ── EFA (efa-direct, SRD)
```

Selection is *forced* with `TRTLLM_FORCE_COMM_METHOD=NCCL_EP`, but only after TRT-LLM's own
feasibility gate (`_get_nccl_ep_unavailable_reason`) passes and **only when attention DP is
enabled** (trap 2). The proof that the whole chain engaged is one log line, on every rank:

```
[TRT-LLM] [RANK 5] [I] NCCL EP group created: ep_size=8, num_experts=128, ...,
  layout=RANK_MAJOR, algorithm=LOW_LATENCY
```

## The integration traps

1. **`TLLM_LOG_LEVEL=info`, not `TRTLLM_LOG_LEVEL`.** `tensorrt_llm/logger.py` reads `TLLM_LOG_LEVEL`;
   the intuitive `TRTLLM_`-prefixed spelling is **silently ignored**, the level stays at `error`, and
   the one "NCCL EP group created" line that constitutes engagement proof never prints. You then have
   a serve that answers requests while you cannot tell which all-to-all it is running.
2. **No attention DP ⇒ the env knob is never read.** `create_strategy()` returns `None` /
   `AllGatherReduceScatter` when `enable_attention_dp` is false or `dp_size == 1` — *before* it looks
   at `TRTLLM_FORCE_COMM_METHOD`. `recipe/serve.sh` passes `enable_attention_dp: true` via
   `--extra_llm_api_options`; without it the serve comes up fine on the wrong backend.
3. **The NGC container's baked NCCL is too old for NcclEP.** `is_nccl_ep_installed()` requires
   NCCL ≥ 2.30.4. The Dockerfile force-installs pip `nvidia-nccl-cu13==2.30.4` (`--no-deps`, one pin
   overridden deliberately) and every launcher puts the pip wheel's `lib/` **first** on
   `LD_LIBRARY_PATH` — if the baked copy wins the link, NcclEP silently reports unavailable.
   `recipe/verify-image.sh` asserts which `libnccl.so.2` resolves *and* its version string.
4. **Harness-green ≠ serve-green (the int64/int32 boundary).** The nccl_ep 0.1.x HIGH_THROUGHPUT
   kernel ABI hands back `recv_topk_idx` as **int64**, while the downstream consumer
   `torch.ops.trtllm.fused_moe` requires **int32** routing ids — a real model forward crashes
   ("token_selected_experts dtype is Long") even though a standalone dispatch/combine harness passes
   16/16, because the harness does its own expert math and never crosses that boundary. Under the
   upstream LOW_LATENCY/RANK_MAJOR default the buffer is already int32, so the conflict is invisible.
   [TensorRT-LLM PR#17715](https://github.com/NVIDIA/TensorRT-LLM/pull/17715) carries the
   dispatch-return narrowing with the selectability patch (Dockerfile `APPLY_HT_FLAT_PATCH=1` layer).
5. **The GA-that-lacks-the-backend trap.** Pinning "the latest GA" (v1.2.1) gives you an image where
   this entire test case is impossible — the factory modules do not exist at that tag.
   `recipe/verify-image.sh` fails loud on that with "wrong base tag?".
6. **Do not carry "LOW_LATENCY faults on EFA" as a rule.** On *this* substrate (NCCL 2.30.4 +
   nccl_ep 0.1.0 + GIN CPU-proxy) the upstream LL/RANK_MAJOR default was measured clean — a real EP8
   serve answered correctly and a 16-rank cross-node probe passed with zero illegal-memory-access.
   An LL fault seen elsewhere was a different substrate. The HT/FLAT patch layer is therefore an
   **opt-in selector** (default OFF), not a fix for a broken default.
7. **GIN needs gdrcopy at compile time and gdrdrv at run time.** An aws-ofi-nccl built without
   `gdrapi.h` carries "GDRCopy support not available at compile time" and GIN init fails at serve;
   the setup script asserts that string is *absent* from the built plugin. At run time, clusters
   without a gdrdrv device plugin need `privileged: true` (the unprivileged device cgroup blocks
   `open("/dev/gdrdrv")` with EPERM even after CAP_MKNOD — measured; the manifest header documents it).
8. **A node-spanning `trtllm-serve` does not bootstrap on EKS.** trtllm-serve uses mpirun across
   nodes, and mpirun's OOB cannot route between VPC-CNI pods (each pod has a /32 eth0, so OMPI's
   `opal_net_samenetwork()` never matches a peer). torchrun is unaffected — which is why this sample's
   cross-node proof is the torchrun probe and the served proof is single-node EP8 (Known limitations).

## Runtime requirements (exported by every launcher, repeated in the manifest)

| Env | Why |
|---|---|
| `NCCL_GIN_TYPE=2`, `NCCL_GIN_ENABLE=1` | GIN CPU-proxy — the EFA-viable GIN mode |
| `OFI_NCCL_GIN_GDAKI=0` | GPU-initiated GIN is not the shipped path on EFA |
| `OFI_NCCL_GIN_MAX_REQUESTS=512` | GIN request-ring depth used by the measured runs |
| `OFI_NCCL_GDRCOPY_FORCED_PCIE_COPY=1` | aws-ofi-nccl PR#1351 — forced-PCIe on gdrdrv-2.4-kernel nodes |
| `FI_PROVIDER=efa`, `FI_EFA_USE_DEVICE_RDMA=1` | EFA with GPU-direct RDMA |
| `FI_EFA_ENABLE_SHM_TRANSFER=0`, `FI_EFA_FORK_SAFE=1` | no SHM shortcut; fork-safe for the proxy |
| `NCCL_NET_PLUGIN=/opt/aws-ofi-nccl/lib/libnccl-net-ofi.so` | the GIN-capable plugin, explicitly |
| `NCCL_CUMEM_ENABLE=1`, `NCCL_NVLS_ENABLE=0`, `NCCL_IGNORE_DISABLED_P2P=1` | NCCL settings the measured substrate ran |
| `TRTLLM_FORCE_COMM_METHOD=NCCL_EP` | selects NcclEP in the factory (trap 2 applies) |
| `NCCL_EP_NUM_QP_PER_RANK=32` | QPs per rank — benchmark provenance knob |
| `ENABLE_CONFIGURABLE_MOE=1` | enables the configurable-MoE path the backend rides |
| `TLLM_LOG_LEVEL=info` | trap 1 — the engagement-proof line |

## Prerequisites

- An EKS cluster with p5en.48xlarge (or p5.48xlarge — adjust the EFA count in the manifest) nodes,
  the EFA device plugin, the NVIDIA device plugin, and 2Mi hugepages pre-allocated on the nodes.
- Docker + network access to `nvcr.io`, `pypi.org`, `github.com`, `efa-installer.amazonaws.com`.
- An image registry you own (ECR); **do not point at anyone's private registry**.

## Build

```bash
cp setup/env_vars.example setup/env_vars   # edit REGISTRY etc. (env_vars is gitignored)
setup/build-push.sh                        # baseline image (upstream LL/RANK_MAJOR default)
APPLY_HT_FLAT_PATCH=1 setup/build-push.sh  # opt-in: bake PR#17715 (algorithm/layout selectable)
```

Then gate the image before any model load:

```bash
recipe/verify-image.sh ${REGISTRY}/trtllm-nccl-ep-efa:v1-20260825
# ... ALL CHECKS PASS
```

## Smoke-test the transport before the model load

Deploy `kubernetes/trtllm-nccl-ep-2node.yaml` (set `image:` to your pushed image), then run the
16-rank cross-node probe — it drives TRT-LLM's real factory entrypoint and checks dispatch/combine
numerics against an oracle, in minutes, before any weights download:

```bash
kubectl -n trtllm-nccl-ep exec trtllm-nccl-ep-1 -- bash -lc \
  'nohup /opt/run-kernel-test.sh worker <pod-0-ip> 1 > /tmp/kt.log 2>&1 &'
kubectl -n trtllm-nccl-ep exec trtllm-nccl-ep-0 -- /opt/run-kernel-test.sh leader <pod-0-ip>
# ... PROBE-PASS factory-selected NcclEP LOW_LATENCY+RANK_MAJOR world=16
# ... KERNEL-TEST PASS — factory-selected NcclEP dispatch/combine over EFA verified
```

## Serve

Pod 0 of the manifest runs `recipe/serve.sh` (single-node, `tp_size=ep_size=8`,
Qwen/Qwen3-30B-A3B by default — any MoE whose routed-expert count divides by `SERVE_EP` passes the
preflight). Confirm engagement, then ask it something with a checkable answer:

```bash
kubectl -n trtllm-nccl-ep logs trtllm-nccl-ep-0 | grep "NCCL EP group created"
curl -s http://<pod-0-ip>:8000/v1/chat/completions -H 'Content-Type: application/json' -d \
 '{"model":"Qwen/Qwen3-30B-A3B","messages":[{"role":"user","content":"What is 17 multiplied by 23? Reply with only the number."}],"max_tokens":16}'
# a correct "391" pins routing correctness, not mere fluency
```

## Benchmark

```bash
kubectl -n trtllm-nccl-ep exec trtllm-nccl-ep-0 -- \
  bash -lc 'OUT_ROOT=/work/benchmarks /opt/benchmark.sh 127.0.0.1'
```

Methodology, provenance knobs, and status live in [benchmarks/README.md](benchmarks/README.md).

## Known limitations

- **The served completion is single-node (EP8).** The cross-node (EP16) proof is at the
  factory/dispatch level via the torchrun probe, not a served HTTP completion — a node-spanning
  `trtllm-serve` needs mpirun, which cannot bootstrap between VPC-CNI pods (trap 8). Both halves
  were measured; the combined "one serve spanning nodes" was not, and this sample does not claim it.
- **This exact image assembly is build-staged, not yet cluster-re-measured.** The measured E2E
  (2026-08-07: real `trtllm-serve` HTTP-200-correct EP8 + 16-rank cross-node probe 16/16 PASS,
  zero illegal-memory-access, `efa-direct` on every rank, both LL/RANK_MAJOR and patched HT/FLAT)
  ran the same component versions with the rc24 NcclEP modules grafted onto an rc9 container. This
  Dockerfile builds the cleaner equivalent — rc24 native — which is exactly what the recipe's gates
  (`verify-image.sh`, `run-kernel-test.sh`) exist to re-verify on your cluster.
- **No performance numbers are published here** (see benchmarks/README.md). Nothing in this sample
  demonstrates a speedup; it demonstrates a *working, verifiable* NcclEP-over-EFA path.
- Validated on H200/p5en only; p5/H100 differs in manifest EFA count and was not re-measured.

## References

- [NVIDIA/TensorRT-LLM PR#17715](https://github.com/NVIDIA/TensorRT-LLM/pull/17715) — NcclEP
  algorithm/layout selectability (`TRTLLM_NCCL_EP_ALGO` / `TRTLLM_NCCL_EP_LAYOUT`) + the int64→int32
  dispatch-return fix; the Dockerfile's opt-in patch layer, retired when it merges.
- [NVIDIA/TensorRT-LLM issue#17714](https://github.com/NVIDIA/TensorRT-LLM/issues/17714) — the
  companion issue documenting the HT/FLAT selection gap.
- [aws/aws-ofi-nccl PR#1351](https://github.com/aws/aws-ofi-nccl/pull/1351) —
  `OFI_NCCL_GDRCOPY_FORCED_PCIE_COPY` (cherry-picked at its immutable head SHA).
- Sibling test cases: [vLLM + DeepEP-V2 over EFA](../../vllm/deepep-v2-efa/) (same NCCL-GIN
  CPU-proxy substrate, different EP kernel package),
  [SGLang + DeepEP over EFA](../../sglang/dsr1-deepep-efa/) (NVSHMEM host-proxy substrate), and the
  [expert-parallelism micro-benchmarks](../../../../micro-benchmarks/expert-parallelism/).
