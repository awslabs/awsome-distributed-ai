# Finetuning Llama from Huggingface Weights

This test case showcase how to finetune Llama2 model from HuuggingFace Weights using Megatron DeepSpeed.

## 1. Preparation
Set the following environment variables to run the test cases:

```bash
export APPS_PATH=/fsx/apps
export ENROOT_IMAGE=$APPS_PATH/deepspeed.sqsh
export FSX_PATH=/fsx
export MODEL_PATH=$FSX_PATH/deepspeed
export DATA_PATH=$FSX_PATH/alpaca
```
In this step, we prepares Llama2 dataset and pretrained weights.

This tutorial uses [Stanford Alphaca](https://github.com/tatsu-lab/stanford_alpaca) dataset. Download the dataset with the command below:

```bash
mkdir -p ${DATA_PATH}
wget https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json -O ${DATA_PATH}/alpaca_data.json
```

Llama2 model, which governed by the Meta license and must be downloaded and converted to the standard [Hugging Face](https://huggingface.co/) format prior to running this sample.
You can submit access request from [here](https://ai.meta.com/resources/models-and-libraries/llama-downloads/), we need "Llama 2 & Llama Chat" to be checked. Use the [download.sh](https://github.com/facebookresearch/llama/blob/main/download.sh) in the official repository. You will be asked to input an URL from the email you receive from meta.  

We will assume that you had placed the model and tokenizer as follows on cluster:

```
${MODEL_PATH}/Llama2-meta/
├── 7B/
│   ├── checklist.chk
│   ├── consolidated.00.pth
│   └── params.json
├── tokenizer.model
└── tokenizer_checklist.chk
```

Convert the model weights into HF format:

```bash
sbatch 1.convert-weights-to-hf.sbatch
```

`convert_llama_weights_to_hf.py` transforms the original weights into the Huggingface format as in:

```
${MODEL_PATH}/Llama2-7b-hf
├── config.json
├── generation_config.json
├── pytorch_model-00001-of-00003.bin
├── pytorch_model-00002-of-00003.bin
├── pytorch_model-00003-of-00003.bin
├── pytorch_model.bin.index.json
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer.model
└── tokenizer_config.json
```

Finally, transforms the checkpoint into Megatron DeepSpeed format:

``bash
bash 2.convert-weights-to-mega-ds.sh
```


## 1. Finetuning

Finetuning job can be submitted as follows:

```bash
bash 3.finetune-llama.sh
```