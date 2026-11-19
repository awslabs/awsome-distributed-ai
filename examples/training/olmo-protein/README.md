# Pretrain a protein language model with OLMo/OLMoE (MoE and dense)

This recipe adapts AI2's [OLMo](https://github.com/allenai/OLMo) engine
(`Muennighoff/MoE` branch — the branch [OLMoE](https://arxiv.org/abs/2409.02060)
was trained with) to **protein sequences**, and pretrains a Mixture-of-Experts
model and a FLOP-matched dense twin from one container. Proteins don't need
BPE: the tokenizer is **one token per amino-acid residue** (28-token
vocabulary), built in step 1.

Validated end-to-end on SageMaker HyperPod (Slurm, g5.12xlarge) and single-node
EC2 Capacity Blocks (p5.48xlarge); runs unchanged on ParallelCluster.

## What you train

| | MoE | Dense twin |
|---|---|---|
| Blocks | 24 × (attn + 8-expert top-2 MoE FFN) | 24 × (attn + MLP), `mlp_ratio=2` |
| Params | 1.61B total / 706M active | 705M (= MoE active → FLOP-matched) |
| Tokenizer | 28 tokens: 20 AA + 6 ambiguity codes + EOS/PAD | same |

The two configs differ only in the FFN block — a controlled MoE-vs-dense
comparison at equal compute per token. On UniRef90 (45.7B tokens) the MoE
reached val PPL 10.49 vs dense 11.01, at ~25% more wall-clock (gradient sync
and optimizer cover all 1.61B params even though only 706M activate per token).

## Prerequisites

- A Slurm cluster (HyperPod or ParallelCluster) with NVIDIA GPUs and a shared
  filesystem (FSx for Lustre recommended), or any single multi-GPU node.
- Docker + ECR for the container (or bake the env into your AMI/enroot image).

## Steps

### 0. Build the container

```bash
docker build -f 0.Dockerfile -t olmo-protein .
# push to your ECR as usual
```

Gotchas encoded in the Dockerfile: megablocks must install with
`--no-build-isolation --no-deps` or it drags in a second torch; CUDA arch list
must cover your GPUs (8.0=A100, 8.6=A10G, 9.0=H100).

### 1. Build the per-residue tokenizer

```bash
python 1.build_tokenizer.py   # writes protein_aa_v1.json (28 tokens)
```

### 2. Download + tokenize UniRef

```bash
python 2.prepare_uniref.py --output $DATA_HOME/data   # UniRef50 by default
```

Produces uint16 `.npy` shards (~23 GB for UniRef50's 12.1B tokens) plus a
held-out validation shard. Point your config's `data.paths` at them.

### 3. Train

```bash
export DATA_HOME=/fsx/olmo-protein
export CONFIG=$DATA_HOME/config/OLMoE-protein.yml   # or the dense twin config
sbatch 3.train.sbatch
```

MoE vs dense = swap `$CONFIG`. Throughput reference: ~8.3k tok/s on one
g5.12xlarge (4×A10G, microbatch 4, fine-grained activation checkpointing);
~240k tok/s on one p5.48xlarge (8×H100, microbatch 8, no act-ckpt).

## Operational notes (learned the expensive way)

- **Resume only from `*-unsharded` checkpoints.** FSDP shard layout is tied to
  world size, and sharded resume OOMs on this branch. The sbatch saves
  unsharded every 1000 steps; `scripts/unshard.py` converts older sharded ones.
  This is also how you move a run between GPU counts (we resumed a 4-GPU g5
  run on an 8-GPU p5 with no loss spike — optimizer state included).
- **Use step-based `max_duration`, never epoch-based.** After a cross-instance
  resume, epoch accounting undercounts and the run can exit rc=0 early.
- **Watch for silent GPU faults on self-managed EC2**: a driver fault kills
  training while the instance still passes EC2 status checks. Checkpoint
  cadence bounds the loss; HyperPod's health checks + auto-resume cover this
  class automatically.
- Checkpoint sizes (MoE): model.pt 6.4 GB, optim.pt 12.9 GB (Adam = 2× model),
  so budget shared-FS/S3 space accordingly.

## Observability

`4.mlflow-mirroring.patch` (applied in the Dockerfile) adds MLflow mirroring
to OLMo's trainer: a no-op-safe `mlflow_log()` beside each `wandb.log` call
(~50 lines). Set `MLFLOW_TRACKING_URI` — a SageMaker managed MLflow ARN works
directly — and you get per-step loss, optimizer stats, and MoE router load per
layer/expert. Without a tracking URI it's inert. W&B works out of the box via
OLMo's native config.
