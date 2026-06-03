# Llama 3 8B pretraining with Megatron-LM

Drop-in companion to `../llama2/`. The data-preprocessing flow is identical
in shape — point `data-preproc-llama2.sbatch` at the Llama 3 tokenizer
(`meta-llama/Meta-Llama-3-8B`) and the corresponding `llama3/` data path,
then run `pretrain-llama3-8b.sbatch`.

The sbatch defaults are tuned for **P6-B300** (8× B300 SXM6 per node, 275 GB
HBM3e each):

- `tensor-model-parallel-size=1`, `pipeline-model-parallel-size=1`,
  `context-parallel-size=2`
- `seq-length=8192`, `micro-batch-size=1`, `global-batch-size=512`
- `--bf16`, `--use-flash-attn`, `--transformer-impl transformer_engine`

Adjust `SBATCH --nodes` and `GLOBAL_BATCH_SIZE` for your scale. Llama 3's
RoPE base (`--rotary-base 500000`) and tokenizer (`HuggingFaceTokenizer`)
differ from Llama 2 — already wired up in this script.
