# Porting notes (slime -> miles)

Record of porting the sibling [slime test case](../../slime/) to
[miles](https://github.com/radixark/miles), slime's direct fork. miles diverged from slime
at fork point `fcce96ca0` (2025-10-05) and rewrote the train loop sync -> async, but the
`train.py` CLI is compatible, so the same GRPO recipe runs with only path/branding changes
(verified on hardware).

## Why miles

- miles targets CUDA 13.0.1 / PyTorch 2.11 / Blackwell (sm_103) as first-class. Where the
  slime image reached sm_103 via an NGC base plus hand patches, the miles image already
  applies the sm_103 Transformer Engine FA2 whitelist patch.
- miles has features slime lacks (fp8 rollout, fully-async, true-on-policy) -- out of scope
  here, but future material.

## Docker strategy C: miles official image + EFA layer

miles's dependencies (the sglang-miles fork, radixark/Megatron-LM fork, prebuilt
flash_attn/TE/apex wheels) are built for the PyTorch 2.11 stable + cu130 ABI and do not
load on an NGC (nightly ABI) base. So the image takes `radixark/miles:<dated-tag>` as the
base and adds only the AWS EFA stack (`miles.Dockerfile`):

- remove IB libverbs (let the EFA installer's libfabric win)
- gdrcopy (GPUDirect RDMA)
- EFA installer 1.48.0 (`--skip-kmod`, libfabric + aws-ofi-nccl plugin)
- NCCL/EFA runtime env (`FI_PROVIDER=efa` etc.), `GLOO_SOCKET_IFNAME=eth0`
- `PYTHONPATH=/root/miles:/root/Megatron-LM` (miles does not bake this into the image)

Pin by digest (`MILES_BASE_DIGEST`), never `:latest` and not by tag alone: radixark
publishes `dev-*` as mutable snapshots and has already deleted a tag this test case once
pinned. The dated tag is kept next to the digest for readability. Per awsome-distributed-ai
CONTRIBUTING.

## slime -> miles changes (recipe / launcher)

| Item | slime | miles |
| --- | --- | --- |
| framework install dir | `/opt/slime` | `/root/miles` (editable install) |
| Megatron | `/opt/Megatron-LM` | `/root/Megatron-LM` (radixark fork) |
| recipe runtime-env PYTHONPATH | `/opt/Megatron-LM` | `/root/Megatron-LM:/root/miles` |
| launcher `SLIME_DIR` default | `/opt/slime` | `/root/miles` (variable name kept) |
| model script | `scripts/models/qwen3-4B.sh` | same (path compatible) |

The `train.py` flags are unchanged, so the GRPO recipe argv is identical to slime's apart
from the paths above. SGLang passthrough flags (`--sglang-*`) work the same way on miles
(ServerArgs auto-expose via parse_known_args).

## Pitfalls found on real hardware

All three occur on the miles base (nvidia/cuda) and NOT on slime's NGC base, and are fixed
in `miles.Dockerfile` / the manifests.

### 1. CUDA compat shadowing -> torch.cuda dies (Error 803)

Carrying slime.Dockerfile's EFA layer verbatim also carries
`ENV LD_LIBRARY_PATH=...:/usr/local/cuda/compat:...`, which is fatal on the miles base:

- The miles base bundles a CUDA forward-compat libcuda `580.82.07`.
- The validation node's host driver is `580.159.03` (nvidia-smi shows "CUDA Version: 13.0").
- CUDA forward-compat requires compat >= host driver. When the older compat (580.82.07) is
  preferred via LD_LIBRARY_PATH, torch.cuda dies with `RuntimeError: ... Error 803: system
  has unsupported display driver / cuda driver combination`, and the SGLang engine fails at
  `get_device()` ("No accelerator available").
- Verified: dropping compat from LD_LIBRARY_PATH restores `torch.cuda.is_available() == True`.
- Fix: `miles.Dockerfile` deletes `/usr/local/cuda*/compat` outright and uses the host
  driver (`/usr/lib64/libcuda.so`), which already supports the image's CUDA 13.0 toolkit.
  slime (NGC) could use compat because NGC's entrypoint enables it only when compat >= host.

### 2. libcuda.so.1 not found in the SGLang subprocess

With compat gone, torch still works (ld.so.cache resolves libcuda), but the SGLang server
subprocess (and Triton / cuda-python loaders, which scan LD_LIBRARY_PATH directly rather
than the cache) fail with `ImportError: libcuda.so.1: cannot open shared object file`. Fix:
append the driver-injection dirs (`/usr/lib64`, `/usr/lib/x86_64-linux-gnu`) to
LD_LIBRARY_PATH and register them with `ldconfig`. Appending (not prepending) means they
are consulted only for libraries nothing else resolves.

### 3. Ray job driver dies on the head (no libcuda)

miles's `MegatronTrainRayActor` imports `mooncake` (P2P weight transfer, libcuda-dependent)
at module load. The Ray job driver runs on the head, which is a non-GPU pod with no libcuda
injected, so the driver dies with the same `ImportError: libcuda.so.1`. Fix: declare a
`gpu_node` custom resource on the worker (`rayStartParams.resources: '{"gpu_node": 1}'`)
and submit with `ray job submit --entrypoint-resources '{"gpu_node": 0.001}'`, which places
the driver on a GPU worker without consuming a GPU logical count (so it does not conflict
with the colocated 8-GPU placement group). Note: `begin_weight_update` / `pull_weights` are
Ray actor methods on miles's rollout engine (`sglang_engine.py`), not HTTP endpoints on an
SGLang fork.

## GPU-generation portability (H200 / B300)

Everything in `miles.Dockerfile`, the recipes, and the manifests is GPU-generation-agnostic:
the base image tag (`MILES_BASE_TAG`) already carries both the sm_90 (Hopper/H200) and
sm_103 (Blackwell/B300) builds internally, and no recipe/env file hard-codes a CUDA or SM
version. The one cluster-specific knob is `GPU_NODE_ROLE` in `env_vars` (substituted into
`kubernetes/raycluster.yaml`'s worker `nodeSelector`), which just needs to match whichever
accelerator NodePool -- H200 or B300 -- you point it at. This test case is hardware-verified
only on H200 (p5en.48xlarge); the analysis above is why B300 (p6-b300.48xlarge) is expected
to work unmodified, not a claim that it has been run.

## Known Issue

- `save_model()` fails with `_pickle.UnpicklingError: pickle data was truncated` in
  Megatron's distributed checkpoint save (`gather_object`), independent of the GRPO loop.
  Blocks the HF<->Megatron round-trip and long checkpointing runs. File upstream on miles.

## Residual patches (of slime's 7, what remains on miles)

- `--sglang-log-level warning` (lowercase; avoids the uvicorn KeyError; recipe-only).
- GPU-less Ray driver Megatron `validate_args` CUDA probe (may surface for 30B MoE; may
  already be fixed in the radixark fork -- UNVERIFIED).

The numpy<2 pin, torch_memory_saver preload `.so` selection, and manual mbridge pin are all
resolved by the miles official image.
