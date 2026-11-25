# Causal protein LM pretraining with OLMo-core: MoE vs FLOP-matched dense

This example pretrains **causal (next-residue) protein language models** with
AI2's [OLMo-core](https://github.com/allenai/OLMo-core), as a controlled pair:
an 8-expert top-2 Mixture-of-Experts and a dense twin whose parameter count
matches the MoE's *active* parameters to 0.039%. Same seed, same data order —
an architecture comparison at equal compute per token.

Unlike the repository's masked-LM protein examples
([`esm2-hyperpod`](../esm2-hyperpod/), [`bionemo`](../../training/bionemo/)),
this example trains autoregressive models, and its contribution is the
**MoE-versus-dense pairing**: the recipe answers "is MoE worth it for my data
budget?" with a controlled experiment rather than a single model.

The recipe keeps no copy of any training code: it runs **OLMo-core's own
"Train an LLM" example** (`src/examples/llm/train.py`, pinned at `v3.0.0`)
with a protein overrides file. All that is maintained here is protein data
preparation, the overrides, and a Slurm launcher.

| | MoE | Dense twin |
|---|---|---|
| Factory | `olmoe_1B_7B` (8 experts, top-2, hidden 2048) | `olmo2_1B` (FFN hidden 4096) |
| Params | 2,819,688,448 total / 1,007,749,120 active | 1,007,355,904 (−0.039% vs MoE active) |
| Trunk | d_model 2048 · 24 layers · 16 heads · QK-norm · RoPE θ 500k · reordered-norm block — **identical**; the arms differ only in the feed-forward layer | |
| Tokenizer | 28 IDs: EOS, PAD, 20 amino acids, 6 ambiguity codes (one token per residue — proteins don't need BPE) | same |

## Prerequisites

- A Slurm cluster with enroot/pyxis (SageMaker HyperPod or AWS ParallelCluster)
  and a shared filesystem, or any multi-GPU node for single-node runs.
- Docker + ECR for the container route; Python ≥ 3.10 for the venv route.

## Steps

### 0. Build the container

```bash
docker build --provenance=false -f 0.Dockerfile -t olmo-protein-core .
```

(`--provenance=false` keeps the pushed manifest enroot-compatible — BuildKit's
default attestation manifest list fails enroot import with "Could not process
JSON input".)

The image pins OLMo-core at `v3.0.0` on a DLC base whose PyTorch (2.10)
satisfies OLMo-core's `torch>=2.10,<2.14`, so the DLC's PyTorch/CUDA/NCCL
stack is used as-is. `grouped_gemm` is installed as a required step — the
full-size model here is the MoE, and OLMo-core's expert matmuls are
substantially slower without it.

For Slurm, convert to squashfs for enroot:

```bash
enroot import -o olmo-protein-core.sqsh dockerd://olmo-protein-core:latest
```

Venv route (single node): clone the same tag — the training script lives in
the repo, not the installed package:

```bash
git clone --branch v3.0.0 https://github.com/allenai/OLMo-core.git
pip install -e ./OLMo-core && pip install grouped_gemm==0.3.0
export OLMO_CORE=$PWD/OLMo-core
```

### 1. Download + tokenize UniRef50

```bash
python 1.prepare_uniref.py --output /fsx/olmo-protein/data/protein
```

Writes **uint8** token shards (`train/part-NN-00000.npy` + a held-out
`val/` shard) — uint8 is the dtype OLMo-core infers for a 28-token
vocabulary, so writer and reader agree by construction. Full UniRef50 is a
46 GB download → 12.1B tokens (~12 GB of shards). Use
`--max-sequences 100000` for a quick pass.

### 2. Train

All model/data settings live in one file, `protein_overrides.sh`, sourced by
both launch paths so they cannot drift apart.

**Slurm** (container):

```bash
cd slurm && mkdir -p logs
DATA_ROOT=/fsx/olmo-protein/data/protein ARCH=moe   sbatch 2.train.sbatch
DATA_ROOT=/fsx/olmo-protein/data/protein ARCH=dense sbatch 2.train.sbatch
```

On HyperPod the script detects `/opt/sagemaker_cluster` and submits with
`srun --auto-resume=1`, so node replacement after a hardware fault resumes
the job automatically. Set `NPROC_PER_NODE=8` (and submit with a matching
`--gres` if your cluster requires one) on 8-GPU instances.

**Single node** (venv route):

```bash
source protein_overrides.sh   # defines COMMON, MOE, DENSE; reads DATA_ROOT
torchrun --nproc-per-node=4 $OLMO_CORE/src/examples/llm/train.py run01-moe \
  --save-folder=./checkpoints/run01-moe "${MOE[@]}" "${COMMON[@]}"
torchrun --nproc-per-node=4 $OLMO_CORE/src/examples/llm/train.py run01-dense \
  --save-folder=./checkpoints/run01-dense "${DENSE[@]}" "${COMMON[@]}"
```

Each arm needs its own `--save-folder`: resuming one arm from the other's
folder fails with `Missing key in checkpoint state_dict`.

**Smoke test on one GPU** (24 GB) — shrink the depth and batch:

```bash
source protein_overrides.sh
torchrun --nproc-per-node=1 $OLMO_CORE/src/examples/llm/train.py smoke-moe \
  --save-folder=/tmp/smoke-moe "${MOE[@]}" "${COMMON[@]}" \
  model.n_layers=2 data_loader.global_batch_size=16384 \
  train_module.rank_microbatch_size=8192 \
  "trainer.max_duration={value: 30, unit: steps}" \
  trainer.callbacks.lm_evaluator.eval_interval=10
```

(`eval_interval=10` so validation loss appears within the 30 steps;
upstream's default interval is 250.)

Any config field can be overridden the same way — dotted keys, YAML values.
Note `trainer.max_duration` needs its unit spelled out as above: the bare
`.value=` form would set epochs under upstream's default config.

### What you'll see

Upstream's script logs `train/CE loss` (and MoE load-balancing/router-Z
losses for the MoE arm) every `metrics_collect_interval` steps, and the
`lm_evaluator` callback reports `uniref-val/CE loss` and `uniref-val/PPL` on
the held-out shard — the number to compare between the two arms. Enable W&B
with `trainer.callbacks.wandb.enabled=true`.

## Operational notes

- **The MoE arm needs sm_80+ GPUs (A100/H100).** The CUTLASS grouped-GEMM
  kernel targets `cutlass::arch::Sm80`; on sm_86 (A10G/g5) it aborts at the
  first expert matmul with "Grouped GEMM execution not possible with HW".
  The dense arm runs anywhere. (Validated: MoE on p5/H100, dense on g5/A10G.)

- **Step-based `max_duration`** (set in the overrides) rather than epochs:
  epoch accounting across resumes historically under-counts; steps don't lie.
  11,584 steps at the overrides' 131,072-token global batch is ~1.52B tokens;
  scale `global_batch_size` or duration to your corpus and budget.
- **Checkpoint cadence bounds your loss** on self-managed EC2, where a GPU
  driver fault can kill training while the instance still passes EC2 status
  checks. On HyperPod, health checks plus `--auto-resume=1` cover this class
  automatically.
- OLMo-core checkpoints are **distributed-friendly by design** (PyTorch
  distributed checkpoint shards plus metadata, resumable across world sizes),
  so the manual unshard step older OLMo recipes needed is gone.
- **Verify the data path by inspecting tokens, not loss.** A mis-read token
  stream can still produce a descending loss curve. After the first run
  starts, decode a batch: token IDs should be 2–27 with single EOS (0)
  separators — a stream alternating real tokens with zeros means a dtype
  mismatch between shards and `dataset.dtype`.
- **MoE router health:** the load-balancing loss is active from step 0
  (`lb_loss_weight=0.01`). In our earlier full-scale campaign on the
  predecessor framework, per-layer expert usage was visibly skewed in the
  first few hundred steps before evening out — brief early imbalance is
  expected; persistent collapse of most tokens onto one or two experts
  past warmup is the signal to investigate.

## Provenance

A predecessor of this recipe (on allenai/OLMo's archived MoE branch) ran a
full-scale version of this comparison on SageMaker HyperPod (g5) and EC2
Capacity Blocks (p5): UniRef50 (24.3B tokens, 2 epochs) and UniRef90 (45.7B
tokens, 1 epoch). Result: the MoE won validation perplexity 10.49 vs 11.01
on UniRef90 with the gap widening monotonically, at ~25% more wall-clock —
below ~30 tokens per parameter the dense twin matched it. That decision
framework is what this recipe lets you reproduce on your own corpus.
