# Llama2 Training Example on Slurm with MegatronLM

This directory contains instructions and templates for setting up and running Llama2 model training using MegatronLM on a Slurm cluster.

To pretrain Llama2, you must visit <https://huggingface.co/meta-llama/Llama-2-7b-hf> to download the tokenizers files (i.e., `tokenizer.json` and `tokenizer.model`). Registration required. Alternatively, you may train your own tokenizer but this is beyond the scope for this document. Either way, once you have the tokenizer files, you need to upload them to the FSx Lustre that your Slurm cluster mounts.

The remaining steps are similar to the GPT3 example. For more information, please refer to the official Megatron-LM documentation on Llama2 [here](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/llama2.md).


## 1. Preparation

Ensure you have the following prerequisites:

- A functional Slurm cluster.
- Docker, Pyxis, and Enroot installed on the head node and compute nodes.
- An FSx for Lustre filesystem mounted on `/fsx` in all nodes.
- 

Set up the following environment variables in your terminal:

```bash
export DATA_PATH=/fsx # FSx for Lustre shared file-system
```

### 2. Download and prepocess data

```bash
mkdir -p llama2
# Then, place `tokenizer.json` and `tokenizer.model` to this `llama2/` directory.

# Download sample dataset
wget -P llama2 https://huggingface.co/bigscience/misc-test-data/resolve/main/stas/oscar-1GB.jsonl.xz
xz -d llama2/oscar-1GB.jsonl.xz

sbatch 3.data-preproc-llama2.sbatch
```

### 3. Run pretraining job

Edit `pre-train-llama2.sbatch` to choose the model size you want to train. Do this by commenting and uncommenting the related stanzas. Feel free to experiment with the hyperparameters such as parallelism, batches, etc. (for more details, please refer to the [Megatron-LM project](https://github.com/NVIDIA/Megatron-LM/) and the Megatron papers ([Shoeybi20](https://arxiv.org/abs/1909.08053), [Narayanan21](https://arxiv.org/abs/2104.04473)).

```bash
sbatch pre-train-llama2.sbatch
```

Tips: the Llama2 example prints the estimated FLOPS/GPU (enabled via `--log-throughput` in the pretrain `.sbatch` file). You might want to look at [PR-682](https://github.com/NVIDIA/Megatron-LM/pull/682) and decide whether to patch your Megatron-LM to adjust the way FLOPS/GPU is calculated.

