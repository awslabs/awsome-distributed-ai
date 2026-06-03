from datasets import load_dataset
import json
import os
import urllib.request

# Get dataset configuration from environment variables
dataset_name = os.environ.get('DATASET_NAME')
dataset_config = os.environ.get('DATASET_CONFIG')

# Validate that required environment variables are set
if not dataset_name:
    raise ValueError("DATASET_NAME environment variable is required but not set")

print(f"Dataset configuration:")
print(f"  - Dataset: {dataset_name}")
print(f"  - Config: {dataset_config if dataset_config and dataset_config.lower() != 'none' else 'None (no config)'}")
print()

# Create data directories and tokenizer subdirectory
data_dir = "data"
tokenizer_dir = os.path.join(data_dir, "tokenizer")
processed_data_dir = os.path.join(data_dir, "processed_data")
os.makedirs(data_dir, exist_ok=True)
os.makedirs(tokenizer_dir, exist_ok=True)
os.makedirs(processed_data_dir, exist_ok=True)

print("Downloading tokenizer files...")

# Download GPT-2 vocab file
vocab_url = "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json"
vocab_file = os.path.join(tokenizer_dir, "gpt2-vocab.json")
print(f"Downloading vocab file to {vocab_file}...")
urllib.request.urlretrieve(vocab_url, vocab_file)
print("Vocab file downloaded successfully")

# Download GPT-2 merges file
merges_url = "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt"
merges_file = os.path.join(tokenizer_dir, "gpt2-merges.txt")
print(f"Downloading merges file to {merges_file}...")
urllib.request.urlretrieve(merges_url, merges_file)
print("Merges file downloaded successfully")

print("Loading and processing dataset...")

# Load the dataset using environment variables
# Handle datasets with and without configurations
try:
    if dataset_config and dataset_config.strip() and dataset_config.lower() != 'none':
        print(f"Loading dataset: {dataset_name} with config: {dataset_config}")
        dataset = load_dataset(dataset_name, dataset_config)
    else:
        print(f"Loading dataset: {dataset_name} (no config)")
        dataset = load_dataset(dataset_name)
except Exception as e:
    print(f"Error loading dataset with config '{dataset_config}': {e}")
    print("Attempting to load dataset without config...")
    try:
        dataset = load_dataset(dataset_name)
        print(f"Successfully loaded dataset: {dataset_name} (no config)")
    except Exception as e2:
        print(f"Error loading dataset without config: {e2}")
        raise e2

# Combine all splits into a single JSONL file
combined_file = os.path.join(data_dir, "train_dataset.jsonl")
print(f"Combining all splits into {combined_file}...")

available_splits = list(dataset.keys())
print(f"Available splits: {available_splits}")

with open(combined_file, "w") as f:
    for split in ["train", "validation", "test"]:
        if split in dataset:
            print(f"Processing {split} split...")
            for item in dataset[split]:
                json.dump({"text": item["text"]}, f)
                f.write("\n")
            print(f"Completed {split} split")
        else:
            print(f"Warning: {split} split not found in dataset")
    
    # Also process any other splits that might exist
    other_splits = [s for s in available_splits if s not in ["train", "validation", "test"]]
    if other_splits:
        print(f"Processing additional splits: {other_splits}")
        for split in other_splits:
            print(f"Processing {split} split...")
            for item in dataset[split]:
                json.dump({"text": item["text"]}, f)
                f.write("\n")
            print(f"Completed {split} split")

print(f"Dataset file saved to {combined_file}")
print(f"Tokenizer files saved to {tokenizer_dir}/ directory")
print("Dataset preparation complete!")
