import subprocess
def install_package(package_name):
    subprocess.run(["pip", "install", package_name])

# Example usage
install_package("transformers==4.48.0")

import os
import argparse
from itertools import chain

from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub.hf_api import HfFolder;
from huggingface_hub import snapshot_download


parser = argparse.ArgumentParser()
parser.add_argument('--llama-version', type=int, default=3, help='LLaMA version (default: 3)')
parser.add_argument("--save_path", type=str,default=None, help="path to save the tokenized data")
parser.add_argument("--dataset_name", type=str,default="wikicorpus", help="name of the dataset to use")
parser.add_argument("--dataset_config_name", type=str,default="raw_en", help="dataset config to use")

parser.add_argument(
        "--model_name",
        type=str,
        help="Huggingface Model name",
    )
parser.add_argument(
        "--cache_dir",
        type=str,
        help="Huggingface cache directory"
    )
parser.add_argument(
        "--hf_access_token",
        type=str,
        help="HF access token.",
    )

args = parser.parse_args()
llama_version = args.llama_version

print("*****Args passed by user*********")
print(args)

print("Download tokenizer")
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

HfFolder.save_token(args.hf_access_token)
snapshot_download(repo_id=args.model_name, allow_patterns=["tokenizer*"], ignore_patterns=["*.safetensors","*.safetensors.index.json"],local_dir=args.save_path,local_dir_use_symlinks=False)

block_size = 4096
save_path = f"{args.save_path}/{args.dataset_name}_llama{llama_version}_tokenized_4k"
if llama_version == 3:
    block_size = 8192
    save_path = f"{args.save_path}/{args.dataset_name}_llama{llama_version}_tokenized_8k"

tokenizer_path = args.save_path

save_path = os.path.expanduser(save_path)
tokenizer_path = os.path.expanduser(tokenizer_path)


raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name,trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

column_names = raw_datasets["train"].column_names
text_column_name = "text" if "text" in column_names else column_names[0]

def tokenize_function(examples):
    return tokenizer(examples[text_column_name])
    

tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=column_names,
    load_from_cache_file=True,
    desc="Running tokenizer on dataset",
)

if block_size > tokenizer.model_max_length:
    print("block_size > tokenizer.model_max_length")
block_size = min(block_size, tokenizer.model_max_length)


# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)] for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    load_from_cache_file=True,
    desc=f"Grouping texts in chunks of {block_size}",
)

train_dataset = lm_datasets["train"]
print(len(train_dataset))

train_dataset.save_to_disk(save_path)