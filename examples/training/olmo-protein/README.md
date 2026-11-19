# Pretrain a protein language model with OLMo-core (MoE and FLOP-matched dense)

This recipe pretrains protein language models with AI2's
[OLMo-core](https://github.com/allenai/OLMo-core) — the actively developed
successor to allenai/OLMo, with **native Mixture-of-Experts support** (no
custom kernel builds). Proteins don't need BPE: the tokenizer is **one token
per amino-acid residue** (28-token vocabulary).

One training script covers two architectures, selected by a flag — an
8-expert top-2 MoE and a dense twin sized so its parameter count matches the
MoE's *active* parameters. Same seed, same data order: a controlled
architecture comparison at equal compute per token.

An earlier version of this recipe (allenai/OLMo, `Muennighoff/MoE` branch) was
validated end-to-end on SageMaker HyperPod (Slurm, g5.12xlarge) and EC2
Capacity Blocks (p5.48xlarge) across UniRef50 (24.3B tokens) and UniRef90
(45.7B tokens); headline result there: the MoE reached val PPL 10.49 vs the
dense twin's 11.01 on UniRef90, at ~25% more wall-clock. This port moves the
recipe to the maintained framework.

**Port validation** (olmo-core 3.0.0, torch 2.13, 1× A10G, real UniRef50
tokens): both smoke arches train with descending loss — MoE CE 4.84 → 3.97
and dense CE 4.66 → 4.32 over 30 steps — with checkpoints written and loaded;
both full-size configs build and parameter-count as designed (MoE 2.820B
total / 1.008B active; dense 1.007B — a 0.07% FLOP-match).

## What you train

| | MoE (`--arch moe`) | Dense twin (`--arch dense`) |
|---|---|---|
| Blocks | 24 × (attn + 8-expert top-2 MoE FFN) | 24 × (attn + MLP) |
| Params | 2.820B total / 1.008B active | 1.007B (= MoE active, 0.07% match) |
| Tokenizer | 28 tokens: 20 AA + 6 ambiguity + EOS/PAD | same |

## Prerequisites

- A Slurm cluster (HyperPod or ParallelCluster) with NVIDIA GPUs and a shared
  filesystem, or any single multi-GPU node for smaller runs.
- Python ≥ 3.10 (OLMo-core requirement). Docker + ECR if using the container.

## Steps

### 0. Build the container (or just `pip install -e` OLMo-core into a venv)

```bash
docker build -f 0.Dockerfile -t olmo-protein-core .
```

### 1. Build the per-residue tokenizer

```bash
python 1.build_tokenizer.py   # writes protein_aa_v1.json (28 tokens)
```

### 2. Download + tokenize UniRef

```bash
python 2.prepare_uniref.py --output $DATA_ROOT   # UniRef50 by default
```

Produces uint16 `.npy` token shards (OLMo-core's native `NumpyFSLDataset`
format) plus a held-out validation shard.

### 3. Train

Single node, straight torchrun:

```bash
export DATA_ROOT=/path/to/data/protein SAVE_ROOT=/path/to/checkpoints
torchrun --nproc-per-node=4 3.train.py myrun --arch moe
torchrun --nproc-per-node=4 3.train.py myrun --arch dense
```

Slurm:

```bash
export DATA_ROOT=/fsx/olmo-protein/data/protein ARCH=moe
sbatch 4.train.sbatch
```

Any config field is overridable from the CLI
(`trainer.max_duration.value=50000 data_loader.global_batch_size=262144 ...`).

**Kick the tires on one GPU first** — `--arch moe-small|dense-small` are tiny
variants of the same shape family that fit a single 24 GB card:

```bash
torchrun --nproc-per-node=1 3.train.py smoke --arch moe-small \
  data_loader.global_batch_size=16384 train_module.rank_microbatch_size=8192 \
  trainer.max_duration.value=30
```

Performance note: install the optional `grouped_gemm` package for fast MoE
expert matmuls — without it olmo-core falls back to a slower loop (fine for
smoke tests, wasteful at scale).

## Operational notes (learned from the original campaign)

- **Step-based max_duration** is the default on purpose — epoch accounting
  across resumes historically under-counts; steps don't lie.
- **Checkpoint cadence bounds your loss** on self-managed EC2, where a GPU
  driver fault can kill training while the instance still passes EC2 status
  checks. OLMo-core's async checkpointer keeps the overhead low; HyperPod's
  health checks + auto-resume cover this class automatically.
- OLMo-core checkpoints are **distributed-friendly by design** (safetensors +
  metadata, resumable across world sizes) — the manual unshard step the old
  OLMo recipe needed is gone.
- Observability: the recipe enables console + GPU-memory metrics; flip
  `WandBCallback(enabled=True)` or add a Comet/custom callback for experiment
  tracking. Metrics include per-layer MoE router load — watch it early: expert
  collapse in the first few hundred steps before the load-balancing loss
  kicks in is normal.
