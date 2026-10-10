<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Upstream Megatron dev with DeepEP v2 over EFA

This opt-in Kimi-K2 configuration selects upstream `moe_token_dispatcher_type="flex"` and `moe_flex_dispatcher_backend="deepepv2"`. It uses pinned NVIDIA Megatron-LM dev with the communicator initialization backport from PR #5153, and unmodified Megatron-Bridge sources. The existing `Dockerfile`, UCCL/NVSHMEM build commands and their default benchmark settings remain available.

The unpatched dev source failed at the first DeepEP buffer-size query in a full Kimi run on 32 B300 nodes. A 32-rank reproduction identified a null NCCL subgroup communicator. This image applies only the same-group initialization hunk from [Megatron-LM PR #5153 at the pinned head](https://github.com/NVIDIA/Megatron-LM/blob/9f7cd50a440b0d29625cfea9c58d55ebb4484b1f/megatron/core/transformer/moe/fused_a2a.py), before DeepEP first reads or caches the handle. The [provenance-linked patch](mcore-pr5153-comm-init.patch) contains no overlap or layout changes. This is upstream dev plus an explicit initialization backport. Full training must be verified with this image; a successful size-hint probe is insufficient.

## Source and library pins

| Component | Pin |
| --- | --- |
| NeMo base | `nvcr.io/nvidia/nemo:26.08` |
| NVIDIA/Megatron-LM, dev | `bb5dfd08f09ce06c5925af453fef06b3129f199d` |
| Communicator-init backport source | `NVIDIA/Megatron-LM#5153@9f7cd50a440b0d29625cfea9c58d55ebb4484b1f` |
| NVIDIA-NeMo/Megatron-Bridge | `281f4bebfd78eb7cc9e8282c7929e99a9746b6cc` |
| amazon-contributing/DeepEP | `874779c9ccd2294b56304bd6cc5f138f1f71d097` |
| EFA Installer | `1.50.0`, userspace installation only |
| NCCL | `nvidia-nccl-cu13==2.31.2` |
| GDRCopy userspace | `v2.5.2` |

The `26.08` release base provides Python `3.12.3`, CUDA toolkit `13.3.33`, PyTorch `2.13.0a0+8145d630e8.nv26.6.54250401`, Transformer Engine `2.17.1+4329ff84`, Transformers `5.12.1` and the legacy DeepEP build dependency NVSHMEM `3.6.5`. These were inspected in the base image. Python matches the selected Bridge source's requirement. CUDA supports the selected Hopper and Blackwell targets. The image uses the base's compiled Torch/TE stack; Core and Bridge are installed with `--no-deps` to preserve it. Import checks must pass before using the image; this does not establish full compatibility with every optional Bridge feature.

The transport follows the merged [`deepep-v2-benchmark/deepep.Dockerfile`](../../../micro-benchmarks/expert-parallelism/deepep-v2-benchmark/deepep.Dockerfile) EFA/GDRCopy recipe and directly copies its [`setup_deepep_gin.sh`](../../../micro-benchmarks/expert-parallelism/deepep-v2-benchmark/setup_deepep_gin.sh). The NeMo base supplies Torch and NVSHMEM, so the bare-CUDA bootstrap is unnecessary. MPI and verbs libraries are retained because the base's Torch depends on them. NCCL is installed into the training interpreter's environment and placed first in `LD_LIBRARY_PATH`. Torch reports build-time NCCL `2.30.5`; the verifier checks runtime NCCL `2.31.2` and rejects multiple loaded NCCL libraries. Communicator interoperability still requires hardware verification.

## Build and check imports without GPUs

Run from the repository root. Build output stays under `/tmp`; this command does not publish an image or launch infrastructure.

```bash
set -o pipefail
mkdir -p /tmp/megatron-deepep-v2-efa-verification
MAX_JOBS=4 bash examples/training/megatron-bridge/1.build-deepep-v2.sh \
  2>&1 | tee /tmp/megatron-deepep-v2-efa-verification/build.log

IMAGE=megatron-bridge:dev-bb5dfd0-pr5153-init-deepepv2-874779c
docker run --rm --runtime=runc --network=none --cpus=4 --memory=16g \
  --entrypoint python3 "$IMAGE" /opt/benchmark/verify_deepep_v2.py --cuda-stub
```

The build targets `sm_90`, `sm_100` and `sm_103`. `MAX_JOBS` defaults to a count of `4`. Override `TORCH_CUDA_ARCH_LIST=9.0` for a Hopper-only image. The verifier imports the compiled DeepEP extension, Core, Bridge, recipe and training entrypoints, verifies the Core/Bridge source paths and commit IDs, requires Core to differ only by the exact initialization patch and Bridge to remain unchanged, checks the DeepEP version contains its commit, and checks the runtime NCCL version and plugin files. No GPU is exposed by the command above. `--cuda-stub` explicitly loads the CUDA toolkit driver stub for the GPU-free import check; it is never enabled by the training entrypoint. Some bundled optional packages emit warnings without a GPU. The verifier fails on any required import or source-identity error.

## Kimi configuration and launch

The existing Kimi benchmark entrypoint recognizes `MEGATRON_VARIANT=dev-deepepv2`, which the new image sets. The dev comparison uses a learning rate of `1e-5` (dimensionless), no learning-rate warmup, FP32 routing probabilities, native cross-entropy fusion, fixed-length mock data, random weights and no EP overlap. Both `alltoall` and `deepepv2` use these settings. Checkpoint save/load are disabled for this short random-weight benchmark. The launcher places TensorBoard output under the FSx run directory. The Kimi architecture remains at `61` layers and `384` routed experts, with no MTP layers. `BENCHMARK_LEARNING_RATE` can override the learning rate for direct entrypoint launches.

Bridge's helper does not accept `deepepv2`; the entrypoint assigns the supported Core configuration fields directly. A callback checks the actual dispatcher and `_DeepepV2Manager` before training, then checks `deep_ep.ElasticBuffer` after the first completed training step. Every rank reports its local MoE layer count, including pipeline stages with no MoE layers. A successful configuration print is not proof that dispatch or EFA executed.

For a GPU-free configuration check, stage the Kimi HF configuration at a local path and supply the DeepSeek-V3 HF config in the Hugging Face cache. The selected Bridge recipe constructs its DeepSeek scaffold before replacing it with Kimi; the scaffold therefore also needs that config even though its weights are unused. Use reviewed, pinned HF revisions. `trust_remote_code=True` loads the staged Kimi configuration code.

```bash
# KIMI_CONFIG_DIR contains config.json and its configuration_deepseek.py.
# HF_CACHE_DIR contains the recipe's deepseek-ai/DeepSeek-V3 config cache.
docker run --rm --runtime=runc --cpus=4 --memory=16g --network=none \
  -v "$KIMI_CONFIG_DIR:/config:ro" -v "$HF_CACHE_DIR:/hf-cache:ro" \
  -e KIMI_K2_HF_PATH=/config -e HF_HOME=/hf-cache -e HF_HUB_OFFLINE=1 \
  -e HF_MODULES_CACHE=/tmp/hf-modules \
  --entrypoint python3 "$IMAGE" /opt/benchmark/verify_deepep_v2.py --cuda-stub --config
```

The launch wrapper reuses `run-ab-rawpods.sh`, selecting the baked Kimi entrypoint by default (`BENCH_PY` can select an explicitly staged revision), mounting `/dev/gdrdrv`, and requiring overlap to be off. For an assigned allocation, set comma-separated `NODE_NAMES` (one unique hostname per requested node), `NODE_GROUP`, and `CAPACITY_RESERVATION_LABEL` to restrict scheduling. On GPU Operator nodes, set `GDRCOPY_HOST_PATH=/run/nvidia/driver/dev/gdrdrv`; the container path remains `/dev/gdrdrv`. The launcher passes each Pod's IP explicitly (`--local-addr`), avoiding dependence on the guest hostname; with torchrun's static rendezvous the master is the node-rank-zero Pod addressed via `--master_addr`. The dev wrapper also retains each worker's stdout/stderr under `logs/torchrun-<node-rank>/`, in addition to the combined node log. Rank zero preserves the training exit code after writing `STATUS`. The launcher sets `FI_PROVIDER=efa` per pod; the container sets the OFI network/GIN plugin paths and `NCCL_GIN_TYPE=5`. The image bakes `NCCL_SYM_GIN_KERNELS_ENABLE=0`: NCCL `2.31.2` otherwise initializes optional symmetric GIN collective kernels that require strong/VA signals unsupported by this EFA GIN plugin. This disables those optional NCCL kernels while retaining DeepEP GIN type `5` and the same EFA transport. The dev wrapper still exports the same value explicitly, so direct entrypoint launches and the launcher path agree by default. Render manifests for review without contacting Kubernetes:

```bash
CTX=review-only IMG="$IMAGE" RENDER_ONLY=1 \
  bash examples/training/megatron-bridge/3.run-deepep-v2.sh deepepv2 32 \
  > /tmp/kimi-deepepv2.yaml
```

On an explicitly assigned, existing cluster allocation with this image already available to its nodes, the following submits the requested arm. The inherited launcher deletes previous pods with the same job names, so choose a fresh `ARM_LABEL` and `CAMPAIGN_ID` when retaining an earlier run.

```bash
CTX="$CLUSTER_CONTEXT" IMG="$CLUSTER_IMAGE" NS=kimi-k2-bench \
  STAGE=/fsx/kimi-k2 FSX_PVC=fsx-kimi-k2 HF_HUB_OFFLINE=1 \
  TENSOR_PARALLEL=8 PIPELINE_PARALLEL=8 EXPERT_PARALLEL=32 \
  MICRO_BATCH=1 GLOBAL_BATCH=256 SEQ_LEN=4096 TRAIN_ITERS=24 \
  bash examples/training/megatron-bridge/3.run-deepep-v2.sh deepepv2 32
```

This dev wrapper requires FSx PVC storage and rejects `STORAGE=hostpath`. If Lustre cannot mount, stop and coordinate the host client repair with the infrastructure owner before starting training.

This layout requests `32` existing `p6-b300.48xlarge` nodes, with `8` GPUs and `16` EFA devices per node, for a total of `256` GPUs. The fixed microbatch and sequence lengths are required because the pinned adapter passes the local input token count as `num_max_tokens_per_rank`, which DeepEP requires to agree across its process group. Variable-length inputs and rank-dependent token counts are outside this configuration. For the NCCL reference, run the same command with `alltoall` as the arm after the first arm releases its allocation. Do not compare against historical results as evidence for this image.

## Hardware verification

On 2026-09-16, this pinned image with the initialization backport and `NCCL_SYM_GIN_KERNELS_ENABLE=0` completed the full Kimi configuration above on `32` existing `p6-b300.48xlarge` nodes (`256` GPUs), using FSx for Lustre. All `24` training iterations and the validation/test evaluations completed, all `32` torchrun launchers exited with status `0`, and loss and gradient norms remained finite. All `256` ranks verified their actual `ElasticBuffer` after a completed training step and logged GDAKI context creation. This used random weights, mock data and overlap off. Logs and measurements are retained outside the repository.

The completed run above establishes execution for its stated configuration, not general autograd correctness or comparative performance. The four-arm comparison renderer and the post-run analysis tools are published below; the campaign-specific orchestration (cluster lock, serial run driver, harvest/normalization) is environment-specific and stays outside the repository, so the analysis tools document the harvested-run-directory schema they consume.

Additional qualification found two unresolved upstream limitations. A 16-rank autograd contract test produces a non-contiguous FP32 router-probability gradient; the pinned Core passes it through `.float()` without making it contiguous, and DeepEP rejects it during backward. A separate 2-rank test with 384 experts fails JIT compilation with `Insufficient notify threads`. Neither issue is repaired by the communicator-initialization backport. The successful Kimi execution does not establish support for those test cases.

## Host prerequisites and unresolved runtime limits

- EFA driver version `3.3.0` or newer, EFA devices exposed to the container, and loaded `gdrdrv` with `/dev/gdrdrv`. The container installs userspace libraries only. It cannot upgrade the host EFA driver.
- An allocation explicitly assigned to this work and an NVIDIA driver compatible with the base CUDA toolkit. GPU idleness alone does not establish availability.
- `MOE_A2A_OVERLAP=on` is rejected for `deepepv2`: the pinned Core `should_free_input` does not account for that backend. The launcher keeps overlap off in both comparison arms.
- The backport initializes the same NCCL subgroup before the first DeepEP handle lookup. Initializing the default group is insufficient; initializing the subgroup after DeepEP has cached a null handle is too late. No alternate communicator mode, custom Core adapter, Bridge patch, or old PR #1243 transport patch is used. The source verifier checks the base commit, patch checksum, patched file checksum, reverse application, and absence of other tracked differences.
- Kernel JIT, dispatch/combine correctness, backward execution and internode EFA-GDA performance require a hardware run. Import and configuration checks establish none of these.

Run the targeted local regressions separately from hardware verification:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s examples/training/megatron-bridge -p test_deepep_v2_config.py -v
bash -n examples/training/megatron-bridge/1.build-deepep-v2.sh \
  examples/training/megatron-bridge/3.run-deepep-v2.sh \
  examples/training/megatron-bridge/run-ab-rawpods.sh
```

## Four-arm comparison tooling

`4.render-deepep-comparison.py` renders one comparison job (NCCL all-to-all /
UCCL / DeepEP v1 NVSHMEM / DeepEP v2 GIN-GDA) through the same raw-Pod
launcher; it prints manifests and never contacts Kubernetes itself. It sets
`COMPARISON_PRIMARY=1`, which pins the scored comparison profile inside
`kimi-k2/benchmarks/bench_kimi_k2_pretrain.py` (seed 1234, evaluation off,
micro batch 4, global batch 256, sequence length 4096, LR 5e-6, TP8/PP8/EP32)
and refuses any other launch shape. Leave `COMPARISON_PRIMARY` unset for
ordinary single-arm runs.

The UCCL and DeepEP v1 arms build from `deepep-v2-backends.Dockerfile` on top
of the v2 image:

```bash
docker build -f examples/training/megatron-bridge/deepep-v2-backends.Dockerfile \
  --build-arg COMMON_IMAGE=<your v2 image> --target uccl -t kimi-uccl .
docker build -f examples/training/megatron-bridge/deepep-v2-backends.Dockerfile \
  --build-arg COMMON_IMAGE=<your v2 image> --target deepep-v1 -t kimi-v1 .
```

Render one arm (the renderer needs host PyYAML, `IMG` as an immutable
`@sha256:` digest, exactly 32 comma-separated `NODE_NAMES`, and a
`CAMPAIGN_ID`; it prints the Pod manifests to stdout):

```bash
CTX=<kubectl context> NS=<namespace> IMG=<image@sha256:digest> \
STORAGE=pvc FSX_PVC=<pvc> STAGE=/fsx/<campaign>/stage \
BENCH_PY=/fsx/<campaign>/source/short_kimi_training_with_update_norm.py \
NODE_NAMES=<32 node names, comma-separated> CAMPAIGN_ID=<campaign id> \
python3 examples/training/megatron-bridge/4.render-deepep-comparison.py \
  uccl --repeat 1 --kind performance \
  --source-dir /fsx/<campaign>/source --tmp-dir /fsx/mc/u1
```

Post-run analysis (all read-only over harvested run directories). The
harvest/normalization step itself is environment-specific and not included;
the analysis tools consume one directory per run with this schema:

- `environment.txt` — `key=value` lines with at least `ep_arm`, `cell`,
  `repeat`, `world_size`, `train_iterations`, `run_kind`,
  `benchmark_learning_rate`.
- `STATUS` — first line begins with `PASS` for a run whose processes all
  succeeded.
- `pod-logs/node-rank-<n>.log` — concatenated per-node rank stdout/stderr
  (the source of Megatron's per-iteration lines and of the
  `RUNTIME_DISPATCHER_IDENTITY` / `MODEL_INITIALIZATION_IDENTITY` markers
  that `bench_kimi_k2_pretrain.py` prints in dev mode).
- `node-<n>/runtime-manifest.json` — per-node summary derived from those
  markers (NCCL library identity, EFA device list, backend identity).
- `node-<n>/routes/` — router trace files when route tracing was enabled.
- optional `manifests/run-entrypoint.py` plus a
  `run_entrypoint_source_sha256` key in `environment.txt` — the measurement
  entrypoint copy whose hash the source-identity gate checks (a run without
  them simply fails that gate and is excluded from scored aggregation).

Post-run analysis tools:

- `kimi-k2/benchmarks/parse_runs.py` — per-run validity gates, steady
  iteration timing (first 8 iterations discarded, 32 scored), paired
  arm-vs-NCCL speedups with a fixed-seed bootstrap interval.
- `kimi-k2/benchmarks/compare_training_curves.py` — loss/gradient-norm curve
  overlays and scalar-delta tables for short correctness runs.
- `kimi-k2/benchmarks/summarize_route_trace.py` — router expert-selection
  hash comparison across arms.
- `kimi-k2/tests/` — the 8-step measurement entrypoint with full-precision
  update-norm sampling, NCCL self-repeat envelope derivation, and CPU unit
  tests for the DeepEP v2 autograd contract.

Timing numbers from runs whose pods failed after the final iteration
(DeepEP v1 teardown failures) are recorded as incomplete/exploratory and are
never substituted into scored statistics.
