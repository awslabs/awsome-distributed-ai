#!/bin/bash
#SBATCH --job-name=grpo-math-reasoning
#SBATCH --nodes=9
#SBATCH --ntasks-per-node 1

set -ex

## Set libfabric flags to use EFA
export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1

## Set this flag for debugging EFA
#export FI_LOG_LEVEL=warn

## NCCL Environment variables
# export NCCL_DEBUG=INFO

### Increase the send queue depth and can turn NCCL communications into non-blocking.
### https://www.usenix.org/system/files/atc23-choi.pdf
export NCCL_BUFFSIZE=8388608
### Improve performance by increasing buffer size for Send/Recv, Gather, Scatter and Alltoall communications
### https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html
export NCCL_P2P_NET_CHUNKSIZE=524288

### Improve performance for AllReduce by selecting specific protocol and algorithm for specific
### message size and number of ranks.
### More information https://github.com/aws/aws-ofi-nccl/wiki/Algorithm-and-Protocol-Tuner-for-AWS.
export NCCL_TUNER_PLUGIN=/opt/aws-ofi-nccl/install/lib/libnccl-ofi-tuner.so

NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
TRAIN_NODES_NUM=$((SLURM_NNODES - 1))
TRAIN_NODES="${NODELIST[@]:0:$TRAIN_NODES_NUM}"
VLLM_NODE="${NODELIST[$TRAIN_NODES_NUM]}"
head_node_ip=${NODELIST[0]}
GPUS_PER_NODE=8

LAUNCHER="accelerate launch \
    --config_file /grpo-math-reasoning/deepspeed_zero3.yaml \
    --num_processes $((TRAIN_NODES_NUM * GPUS_PER_NODE)) \
    --num_machines ${TRAIN_NODES_NUM} \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    --machine_rank \$SLURM_NODEID "

MODEL="${1:-'Qwen/Qwen2.5-0.5B-Instruct'}"

CMD="/grpo-math-reasoning/train.py --model $MODEL --vllm_server_host $VLLM_NODE"

# Fetch model config and get number of heads for tensor parallel size
CONFIG_URL="https://huggingface.co/$MODEL/raw/main/config.json"
CONFIG_JSON=$(curl -s $CONFIG_URL)
NUM_HEADS=$(echo "$CONFIG_JSON" | python3 -c "import sys, json; config = json.load(sys.stdin); print(config.get('num_key_value_heads', config.get('num_attention_heads', 1)))")
VOCAB_SIZE=$(echo "$CONFIG_JSON" | python3 -c "import sys, json; config = json.load(sys.stdin); print(config.get('vocab_size', 1))")

# Find largest tensor parallel size that divides NUM_HEADS and is <= GPUS_PER_NODE
TENSOR_PARALLEL=1
for ((i=GPUS_PER_NODE; i>=1; i--)); do
    if [ $((NUM_HEADS % i)) -eq 0 ] && [ $((VOCAB_SIZE % i)) -eq 0 ]; then
        TENSOR_PARALLEL=$i
        break
    fi
done

echo "Detected number of heads: $NUM_HEADS"
echo "Number of GPUs per node: $GPUS_PER_NODE"
echo "Using tensor parallel size: $TENSOR_PARALLEL"

srun -l --mpi=pmix --cpu-bind=none --container-image ./trl-base.sqsh \
    --output=grpo-%j.out --error=grpo-%j.err \
    --container-mounts=.:/grpo-math-reasoning,$HF_HOME:$HF_HOME \
    --nodes=$TRAIN_NODES_NUM --ntasks=$TRAIN_NODES_NUM --nodelist="${TRAIN_NODES}" \
    bash -c "$LAUNCHER $CMD" &

srun -l --mpi=pmix --cpu-bind=none --container-image ./trl-base.sqsh \
    --output=vllm-%j.out --error=vllm-%j.out \
    --container-mounts=.:/grpo-math-reasoning,$HF_HOME:$HF_HOME \
    --nodes=1 --ntasks=1 --nodelist="${VLLM_NODE}" \
    trl vllm-serve --model $MODEL --tensor_parallel_size $TENSOR_PARALLEL &

wait