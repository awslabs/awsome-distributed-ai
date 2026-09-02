#!/usr/bin/env bash
set -xeuo pipefail

# Create data directory
DATA_DIR="${RAY_DATA_HOME}/data/gsm8k"
echo "Creating data directory: ${DATA_DIR}"

# Get the head pod name
HEAD_POD=$(kubectl get pods -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')

if [ -z "$HEAD_POD" ]; then
    echo "Error: Could not find Ray head pod. Is your cluster running?"
    exit 1
fi

echo "Using Ray head pod: ${HEAD_POD}"

# Create Python script to download GSM8K data
cat > /tmp/download_gsm8k.py << 'EOF'
from datasets import load_dataset
import os
import sys
import re

# Get data directory from environment or use default
data_dir = os.environ.get('DATA_DIR', '/fsx/verl/data/gsm8k')

print(f"Creating directory: {data_dir}")
os.makedirs(data_dir, exist_ok=True)

def extract_answer(answer_str):
    """Extract the final numerical answer from GSM8K format (e.g., '#### 18')"""
    match = re.search(r'####\s*(-?[\d,\.]+)', answer_str)
    if match:
        return match.group(1).replace(',', '')
    return None

print("Loading GSM8K dataset from HuggingFace...")
try:
    dataset = load_dataset("openai/gsm8k", "main")
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

print("Adding VERL-required columns (data_source and reward_model)...")

def add_verl_columns(example):
    """Add required columns for VERL reward computation"""
    ground_truth = extract_answer(example['answer'])
    return {
        **example,
        'data_source': 'openai/gsm8k',
        'reward_model': {'ground_truth': ground_truth}
    }

# Process both splits
train_dataset = dataset['train'].map(add_verl_columns)
test_dataset = dataset['test'].map(add_verl_columns)

print("Saving train split to parquet...")
train_path = os.path.join(data_dir, 'train.parquet')
train_dataset.to_parquet(train_path)
print(f"Saved {len(train_dataset)} training samples to {train_path}")

print("Saving test split to parquet...")
test_path = os.path.join(data_dir, 'test.parquet')
test_dataset.to_parquet(test_path)
print(f"Saved {len(test_dataset)} test samples to {test_path}")

print("\nDataset info:")
print(f"Train samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"\nSample train example:")
sample = train_dataset[0]
print(f"  question: {sample['question'][:100]}...")
print(f"  answer: {sample['answer'][:100]}...")
print(f"  data_source: {sample['data_source']}")
print(f"  reward_model: {sample['reward_model']}")

print("\nGSM8K data successfully downloaded and preprocessed for VERL!")
EOF

# Copy script to pod
echo "Copying download script to pod..."
kubectl cp /tmp/download_gsm8k.py ${HEAD_POD}:/tmp/download_gsm8k.py

# Execute the script in the pod
echo "Downloading GSM8K data..."
kubectl exec ${HEAD_POD} -- bash -c "export DATA_DIR=${DATA_DIR}"
kubectl exec ${HEAD_POD} -- python3 /tmp/download_gsm8k.py


# Verify the files exist
echo "Verifying downloaded files..."
kubectl exec ${HEAD_POD} -- ls -lh ${DATA_DIR}/

echo "GSM8K data download complete!"
echo "Data location: ${DATA_DIR}"
echo "  - train.parquet: ~7.5K samples"
echo "  - test.parquet: ~1.3K samples"
