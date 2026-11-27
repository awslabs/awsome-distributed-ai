# Protein-LM overrides for OLMo-core's src/examples/llm/train.py (v3.0.0).
# Sourced by slurm/2.train.sbatch AND by the single-node commands in README.md,
# so the two launch paths cannot drift apart.
#
# Defines: COMMON, MOE, DENSE (bash arrays), plus DATA_ROOT.
# Requires: DATA_ROOT pointing at the shards from 1.prepare_uniref.py
#           (absolute path — the dataset glob does not expand a ./ prefix).

DATA_ROOT=${DATA_ROOT:-/fsx/olmo-protein/data/protein}
DATA_ROOT=${DATA_ROOT%/}

# Per-residue protein tokenizer: 28 IDs (EOS=0, PAD=1, 20 amino acids,
# 6 ambiguity codes), defined inline — token IDs are written directly by
# 1.prepare_uniref.py; no tokenizer file is needed at training time.
TOK="vocab_size: 28, eos_token_id: 0, pad_token_id: 1, bos_token_id: null, identifier: protein_aa_v1"

COMMON=(
  --sequence-length=2048
  "dataset.paths=[$DATA_ROOT/train/*.npy]" dataset.expand_glob=true dataset.dtype=uint8
  "dataset.tokenizer={$TOK}"
  model.vocab_size=128
  "trainer.callbacks.lm_evaluator.eval_dataset.paths=[$DATA_ROOT/val/part-00-00000.npy]"
  "trainer.callbacks.lm_evaluator.eval_dataset.metadata=[{label: uniref-val}]"
  trainer.callbacks.lm_evaluator.eval_dataset.dtype=uint8
  "trainer.callbacks.lm_evaluator.eval_dataset.tokenizer={$TOK}"
  trainer.callbacks.downstream_evaluator.enabled=false   # scores with the GPT-2 tokenizer
  # training settings (same seed + data order for both arms -> controlled comparison)
  init_seed=6198 data_loader.seed=6198
  data_loader.global_batch_size=131072 train_module.rank_microbatch_size=8192
  train_module.optim.lr=4e-4 train_module.optim.weight_decay=0.1 "train_module.optim.betas=[0.9, 0.95]"
  train_module.optim.fused=true train_module.scheduler.warmup=2000
  "trainer.max_duration={value: 11584, unit: steps}"
  trainer.metrics_collect_interval=10 trainer.callbacks.checkpointer.ephemeral_save_interval=200
)

# 8-expert top-2 MoE: 2,819,688,448 total / 1,007,749,120 active params.
MOE=(--model-factory=olmoe_1B_7B model.n_layers=24
  model.block.feed_forward_moe.num_experts=8 model.block.feed_forward_moe.hidden_size=2048
  model.block.feed_forward_moe.router.top_k=2)

# FLOP-matched dense twin: 1,007,355,904 params (0.039% below MoE active).
# Same trunk as the MoE via upstream's olmo2_1B factory (d_model 2048, 16 heads,
# QK-norm, RoPE theta 500k, reordered-norm block) — the arms differ only in
# the feed-forward layer.
DENSE=(--model-factory=olmo2_1B model.n_layers=24 model.block.feed_forward.hidden_size=4096)
