#!/bin/bash
set -euo pipefail

# Submit a one-shot Job inside the cluster to run the Qwen3-235B Megatron recipe
# Requirements:
# - Ray head service reachable at rayml-efa-head-svc.default.svc.cluster.local:10001
# - fsx-claim PVC available for /fsx
# - HF_TOKEN, REGISTRY/IMAGE/TAG set in env_vars

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_VARS_FILE="${SCRIPT_DIR}/env_vars"

if [ ! -f "${ENV_VARS_FILE}" ]; then
  echo "Missing ${ENV_VARS_FILE}. Copy env_vars.example and set your values."
  exit 1
fi

source "${ENV_VARS_FILE}"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN must be set in ${ENV_VARS_FILE}."
  exit 1
fi

IMAGE="${REGISTRY}${IMAGE}:${TAG}"
JOB_NAME="qwen3-235b-submit"
RAY_ADDRESS_DEFAULT="ray://rayml-efa-head-svc.default.svc.cluster.local:10001"
WORKING_DIR="/workspace/verl"
TRAIN_FILE_DEFAULT="/fsx/verl/data/geo3k/train.parquet"
TEST_FILE_DEFAULT="/fsx/verl/data/geo3k/test.parquet"
TRAIN_FILE="${TRAIN_FILE:-$TRAIN_FILE_DEFAULT}"
TEST_FILE="${TEST_FILE:-$TEST_FILE_DEFAULT}"
MODEL_PATH=/fsx/verl/models/Qwen3-VL-235B-A22B-Instruct #"${MODEL_PATH:-Qwen/Qwen3-VL-235B-A22B-Instruct}"
ENGINE="${ENGINE:-vllm}"
GEN_TP="${GEN_TP:-16}"
CP="${CP:-1}"
TP="${TP:-1}"
PP="${PP:-4}"
EP="${EP:-8}"
ETP="${ETP:-1}"

cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  namespace: ${RAY_NAMESPACE:-default}
spec:
  backoffLimit: 0
  template:
    spec:
      dnsPolicy: ClusterFirst
      restartPolicy: Never
      containers:
      - name: submit
        image: ${IMAGE}
        imagePullPolicy: IfNotPresent
        env:
        - name: HF_TOKEN
          value: ${HF_TOKEN}
        - name: RAY_ADDRESS
          value: ${RAY_ADDRESS_DEFAULT}
        - name: RAY_DASHBOARD_PORT
          value: "${RAY_DASHBOARD_PORT:-8265}"
        - name: WORKING_DIR
          value: ${WORKING_DIR}
        - name: RAY_DATA_HOME
          value: ${RAY_DATA_HOME:-/fsx/verl}
        - name: MODEL_PATH
          value: ${MODEL_PATH}
        - name: TRAIN_FILE
          value: ${TRAIN_FILE}
        - name: TEST_FILE
          value: ${TEST_FILE}
        - name: NCCL_DEBUG
          value: "INFO"
        command:
        - /bin/bash
        - -lc
        - |
          set -euo pipefail
          if [ ! -d "${WORKING_DIR}" ]; then
            echo "Working directory ${WORKING_DIR} does not exist in the container. Exiting."
            exit 1
          fi
          cd "${WORKING_DIR}"
          GEN_TP=${GEN_TP:-16}
          CP=${CP:-2}
          TP=${TP:-1}
          PP=${PP:-8}
          EP=${EP:-8}
          ETP=${ETP:-1}
          python3 -m verl.trainer.main_ppo --config-path=config \
            --config-name='ppo_megatron_trainer.yaml' \
            algorithm.adv_estimator=grpo \
            data.train_files="${TRAIN_FILE}" \
            data.val_files="${TEST_FILE}" \
            data.train_batch_size=512 \
            data.max_prompt_length=1024 \
            data.max_response_length=2048 \
            data.filter_overlong_prompts=True \
            data.truncation='error' \
            actor_rollout_ref.model.path="${MODEL_PATH}" \
            actor_rollout_ref.actor.optim.lr=1e-6 \
            actor_rollout_ref.actor.ppo_mini_batch_size=128 \
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
            actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP} \
            actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP} \
            actor_rollout_ref.actor.megatron.context_parallel_size=${CP} \
            actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP} \
            actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP} \
            actor_rollout_ref.actor.use_kl_loss=True \
            actor_rollout_ref.actor.kl_loss_coef=0.01 \
            actor_rollout_ref.actor.kl_loss_type=low_var_kl \
            actor_rollout_ref.actor.entropy_coeff=0 \
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
            actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
            actor_rollout_ref.actor.use_dynamic_bsz=True \
            actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
            actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
            actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
            actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
            actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
            actor_rollout_ref.rollout.name="${ENGINE}" \
            +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
            actor_rollout_ref.rollout.n=5 \
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
            actor_rollout_ref.actor.megatron.use_mbridge=True \
            actor_rollout_ref.actor.megatron.param_offload=True \
            actor_rollout_ref.actor.megatron.optimizer_offload=True \
            actor_rollout_ref.actor.megatron.grad_offload=True \
            actor_rollout_ref.ref.megatron.param_offload=True \
            +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
            +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
            +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
            +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
            +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
            +actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=True \
            +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=flex \
            +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
            +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
            +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
            +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
            +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
            +actor_rollout_ref.actor.megatron.override_transformer_config.account_for_loss_in_pipeline_split=True \
            +actor_rollout_ref.actor.megatron.override_transformer_config.account_for_embedding_in_pipeline_split=True \
            algorithm.use_kl_in_reward=False \
            trainer.critic_warmup=0 \
            trainer.logger='["console","wandb"]' \
            trainer.project_name='verl_grpo_example_geo3k' \
            trainer.experiment_name='qwen3_vl_235b_megatron' \
            trainer.n_gpus_per_node=8 \
            trainer.nnodes=4 \
            trainer.save_freq=20 \
            trainer.test_freq=5 \
            trainer.total_epochs=15
        volumeMounts:
        - name: fsx-storage
          mountPath: /fsx
      volumes:
      - name: fsx-storage
        persistentVolumeClaim:
          claimName: fsx-claim
EOF

echo "Job ${JOB_NAME} applied. Check status with: kubectl get pods -n ${RAY_NAMESPACE:-default} -l job-name=${JOB_NAME}"

