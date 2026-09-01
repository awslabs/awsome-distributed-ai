#!/usr/bin/env bash
set -xeuo pipefail

# Download and preprocess the Geometry3k dataset into parquet files on FSx.
# Output:
#   ${RAY_DATA_HOME:-/fsx/verl}/data/geo3k/train.parquet
#   ${RAY_DATA_HOME:-/fsx/verl}/data/geo3k/test.parquet

DATA_DIR="${RAY_DATA_HOME:-/fsx/verl}/data/geo3k"
echo "Creating data directory: ${DATA_DIR}"

# Get the head pod name
HEAD_POD=$(kubectl get pods -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')

if [ -z "$HEAD_POD" ]; then
    echo "Error: Could not find Ray head pod. Is your cluster running?"
    exit 1
fi

echo "Using Ray head pod: ${HEAD_POD}"

# Create Python script to download and preprocess Geometry3k
cat > /tmp/download_geo3k.py << 'EOF'
"""
Preprocess the Geometry3k dataset to parquet format.
Adapted from https://github.com/volcengine/verl/blob/main/examples/data_preprocess/geo3k.py
"""
import os
import re
import datasets

DATA_DIR = os.environ.get("DATA_DIR", "/fsx/verl/data/geo3k")
os.makedirs(DATA_DIR, exist_ok=True)

data_source = "hiyouga/geometry3k"
instruction_following = (
    r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    r"The reasoning process MUST BE enclosed within <think> </think> tags. "
    r"The final answer MUST BE put in \\boxed{}."
)

print(f"Loading dataset: {data_source}")
dataset = datasets.load_dataset(data_source)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

def make_map_fn(split):
    def process_fn(example, idx):
        problem = example.pop("problem")
        prompt = problem + " " + instruction_following
        answer = example.pop("answer")
        images = example.pop("images")
        data = {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": prompt}],
            "images": images,
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer,
                "question": problem,
            },
        }
        return data
    return process_fn

print("Processing train split...")
train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
print("Processing test split...")
test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)

train_path = os.path.join(DATA_DIR, "train.parquet")
test_path = os.path.join(DATA_DIR, "test.parquet")

print(f"Saving train to {train_path}")
train_dataset.to_parquet(train_path)
print(f"Saving test to {test_path}")
test_dataset.to_parquet(test_path)

print("Done. Summary:")
print(f"Train samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
EOF

# Copy script to pod
echo "Copying download script to pod..."
kubectl cp /tmp/download_geo3k.py ${HEAD_POD}:/tmp/download_geo3k.py

# Execute the script in the pod
echo "Downloading Geometry3k data..."
kubectl exec ${HEAD_POD} -- bash -c "export DATA_DIR=${DATA_DIR}"
kubectl exec ${HEAD_POD} -- python3 /tmp/download_geo3k.py

# Verify the files exist
echo "Verifying downloaded files..."
kubectl exec ${HEAD_POD} -- ls -lh ${DATA_DIR}/

echo "Geometry3k data download complete!"
echo "Data location: ${DATA_DIR}"
echo "  - train.parquet"
echo "  - test.parquet"

