# Megatron-Bridge 26.08 EP implementation notes

All citations below refer to the locked upstream source commits in `versions.lock.yaml`. The Docker build applies the stored patches only after checking those commits and each patch SHA-256.

## Source and dispatcher decisions

- The shipped helper is `src/megatron/bridge/training/flex_dispatcher_backend.py:33-80` at Bridge `c93251151adeeadbae3ff2a2bf5ee7a1c34cff01`. It sets both the flex dispatcher and backend and clears shared-expert overlap. `bridge-deepep-v2.patch` extends its hardware validation to `deepep_v2`; the benchmark imports this path directly with no compatibility fallback.
- MCore constructs the token dispatcher with `pg_collection.expt_tp` and `pg_collection.tp_ep` at `megatron/core/transformer/moe/token_dispatcher.py:83-90`. The default collection assigns these from `get_expert_tensor_parallel_group()` and `get_expert_tensor_and_model_parallel_group()` at `megatron/core/transformer/moe/moe_utils.py:1441-1451`. Therefore Kimi ETP1 and EP32 creates a 32-rank ElasticBuffer group, not a dense TP8 by EP32 group.
- The patch centralizes V1-compatible metadata and local permutation in `_BaseDeepepManager` at patched `megatron/core/transformer/moe/token_dispatcher.py:1218-1448`, leaves V1 on `_DeepepManager` at lines 1451-1470, and selects `_DeepepV2Manager` at lines 1473-1519 and 1836-1845. The normalized runtime marker names both `_DeepepV2Manager` and `ElasticBuffer`.

## ElasticBuffer contract

- ElasticBuffer construction flags and QP semantics are defined at amazon-contributing DeepEP `02efc268a37802fc00812ede8f5ad7f535ceea0e`, `deep_ep/buffers/elastic.py:244-301`. The compatibility profile explicitly uses BF16 dispatch, hybrid mode, multiple reduction, compute-overlap preference, and automatic QPs.
- Dispatch arguments, cached-handle constraints, events, CPU sync, and expanded layout are defined at `deep_ep/buffers/elastic.py:881-930` and `931-1057`. The adapter fixes `expert_alignment=1`, `do_cpu_sync=True`, `do_expand=False`, and `use_fp8_dispatch=False`. Cached backward dispatch uses the forward handle and `do_cpu_sync=False`, as required by that API.
- Combine consumes the dispatch handle and optionally returns top-k weight gradients at `deep_ep/buffers/elastic.py:1076-1138`. Dispatch backward maps to combine with the received probability gradient. Combine backward maps to cached-handle dispatch.
- Each autograd context owns its exact handle and a reference to its ElasticBuffer. The process-level cache shares only compatible workspaces. Async calls wait their returned event on the current stream before dependent work, while distinct handles preserve multiple in-flight microbatches and recomputation.

## Replay, pipeline, and overlap

- Router Replay defines record, forward replay, and backward recomputation replay at `megatron/core/transformer/moe/router_replay.py:8-15`; global data distribution and lifecycle are at lines 28-75; exact index gathering for forward and backward is at lines 137-180. The correctness test hashes the recorded indices and demands byte-identical replay.
- MCore requires virtual pipeline parallelism for EP overlap and rejects full/MoE recomputation in that mode at `megatron/core/transformer/transformer_config.py:2685-2729`. The interleaved schedule selects combined 1F1B when overlap is enabled at `megatron/core/pipeline_parallel/schedules.py:1458-1482`. The benchmark follows these constraints and keeps recomputation enabled only in the non-overlap profile.
- MCore's fine-grained lifetime rules originally recognized backend `deepep` but not backend `deepep_v2`. Consequently `should_free_input()` could release the MoE dispatch input, and the schedule did not restore each node's detached `token_probs` and `dispatched_probs` before backward. The accepted patch adds `deepep_v2` to the backend predicates at `megatron/core/models/common/utils.py:66-74` and `megatron/core/models/gpt/fine_grained_callables.py:53-61`.
- The dedicated overlap regression proves that 2 in-flight schedule nodes retain distinct probability tensors and dispatch state. The 256-GPU overlap qualification additionally proves 7 layer-owned `MoEFlexTokenDispatcher` objects, 7 `_DeepepV2Manager` objects with 32-rank process groups, and 1 compatible shared `ElasticBuffer` workspace at runtime.

## NCCL identity

- DeepEP v2 checks `/proc/self/maps`, rejects duplicate NCCL mappings, and compares the loaded binary byte-for-byte with its linked NCCL root at `deep_ep/__init__.py:46-68`; it runs this check before importing ElasticBuffer at lines 82-95.
- NCCL `v2.31.2-1` defines EFA-GDA as GIN type 5 at `src/include/nccl_device/core.h:86-90`; plugin selection honors `NCCL_GIN_TYPE` at `src/plugin/gin.cc:137-151`. The scored V2 arm therefore sets `NCCL_GIN_TYPE=5`. NCCL's symmetric GIN kernel control is declared at `src/sym_kernels.cc:131`, so the isolation profile explicitly sets `NCCL_SYM_GIN_KERNELS_ENABLE=0`.
- DeepEP checks the communicator's direct or railed GIN capability and refuses `NCCL_GIN_TYPE_NONE` at patched `csrc/kernels/backend/nccl.cu:98-109`. In debug mode, its selected QP, GIN-context, and indexed-signal budgets are printed at lines 89-93 and 137-144. The PR 5 scale runner enables that output and preserves every node log.
- The common image replaces NCCL, aws-ofi-nccl/libfabric, and GDRCopy for every arm. Image acceptance independently records `torch.cuda.nccl.version()`, `ncclGetVersion()`, extension `ldd`, mapped paths, and the resolved library SHA-256.
- The supplied NeMo digest retained the exact PyTorch source tree at `/opt/pytorch/pytorch` even though it stripped the parent repository metadata. `docker/rebuild-pytorch-nccl.sh` hashes that source tree, identifies its packaged upstream commit as `8145d630e811c9ec098a632d9ea7f4d52a2ea16e`, restores the exact ONNX gitlink commit, and rebuilds Torch against the shared NCCL 2.31.2 installation. Final acceptance reports NCCL build version 2.31.2, runtime version 2.31.2, and exactly 1 loaded NCCL object in every qualified process.

## Scientific isolation

- `EP_ARM` is mandatory and is checked against `/opt/benchmark/backend.json`. The four accepted values are `nccl-alltoall`, `uccl`, `deepep-v1-nvshmem`, and `deepep-v2-gin-gda`.
- The three flex arms share dispatcher metadata, local permutation, GroupedMLP, the model configuration, and the common network stack. The NCCL all-to-all arm is reported as the legacy end-to-end baseline, not as a transport-only control.
- The initial V2 headline profile excludes expanded and sync-free layouts. `matrices/optimized.yaml` keeps that experiment separate.
- The campaign acquires the shared Kubernetes Lease with optimistic resource-version replacement before selecting nodes, rechecks requested GPU occupancy immediately before every pod claim, and records the concurrent vLLM namespace's named nodes as an immutable protected set for the campaign. Those nodes are excluded even if their instantaneous GPU request count falls to 0 GPUs.
- Raw-pod creation takes a single preflight snapshot of nodes and live pod occupancy, emits all named-node Pod objects into one multi-document manifest, and submits that manifest in one API operation. Each node receives exactly 8 GPU requests and the instance-type-specific EFA request. Harvest runs with bounded parallelism, verifies each node-local custody index before accepting the run, and writes the aggregate `SHA256SUMS` only after the artifact tree is frozen.
