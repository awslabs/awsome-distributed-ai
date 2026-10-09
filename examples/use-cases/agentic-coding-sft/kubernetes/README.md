# Kubernetes manifests

Each file is a `*.yaml-template` rendered with `envsubst` and applied with `kubectl`. The code is
baked into the container image at `/workspace`; datasets, runs and GGUF files live on the shared
FSx for Lustre volume mounted at `/fsx` (under `/fsx/agentic`).

## Prerequisites

- An Amazon SageMaker HyperPod EKS cluster with the **HyperPod training operator** add-on, with at
  least one `ml.g5.2xlarge` GPU node and one CPU node (e.g. `ml.c5.4xlarge`), accessible via
  `kubectl`. See the [architectures directory](../../../../architectures) for cluster setup.
- An Amazon FSx for Lustre persistent volume claim. The templates default to `fsx-pvc`. If your
  cluster uses a different name (e.g. `fsx-claim`), update the generated YAML before applying:

  ```bash
  # GNU sed (Linux):
  sed -i 's/fsx-pvc/fsx-claim/g' <rendered>.yaml
  # BSD sed (macOS): sed -i '' 's/fsx-pvc/fsx-claim/g' <rendered>.yaml
  ```

- The training and comparison images built and pushed to Amazon ECR (see the top-level
  [README](../README.md)).

## Common environment variables

```bash
export AWS_REGION=$(aws ec2 describe-availability-zones --output text --query 'AvailabilityZones[0].[RegionName]')
export ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export REGISTRY=${ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/
export IMAGE_URI=${REGISTRY}agentic-coding-sft:v1
export VLLM_IMAGE_URI=${REGISTRY}agentic-coding-sft-compare:v1
```

## 1. Prepare data (CPU)

```bash
cat prep-agent.yaml-template | envsubst > prep-agent.yaml
kubectl apply -f prep-agent.yaml
```

Writes `train-8k`, `train-16k` and `val` under `/fsx/agentic/data/agent/`.

## 2. Build evaluation tasks (CPU)

```bash
cat make-tasks.yaml-template | envsubst > make-tasks.yaml
kubectl apply -f make-tasks.yaml
```

Writes the 150 task repos under `/fsx/agentic/agent_tasks/v1/`.

## 3. Train (GPU)

Quick run (505 trajectories, 1 epoch, 1 GPU, ~22 minutes):

```bash
export RUN=agentic-workshop AGENT_DATA=train-8k EPOCHS=1 NUM_NODES=1 GRAD_ACC=8
cat train-agent.yaml-template | envsubst > train-agent.yaml
kubectl apply -f train-agent.yaml
```

Full run (3,460 trajectories, 2 epochs, 4 GPUs):

```bash
export RUN=agentic-full AGENT_DATA=train-16k EPOCHS=2 NUM_NODES=4 GRAD_ACC=2
cat train-agent.yaml-template | envsubst > train-agent.yaml
kubectl apply -f train-agent.yaml
```

Follow progress (the training operator names pods `${RUN}-pods-<hash>-<id>`, so select by label):

```bash
kubectl get hyperpodpytorchjob ${RUN}
kubectl logs -f -l job-name=${RUN}
```

Outputs land in `/fsx/agentic/runs/${RUN}/` (`adapter/`, `merged/`, `checkpoint-*`, `summary.json`).

## 4. Evaluate (GPU)

The examples below evaluate the quick run (`agentic-workshop`); substitute `agentic-full` if you
ran the full training instead.

```bash
# base model
export LABEL=base EVAL_MODEL=Qwen/Qwen3-1.7B
cat agent-eval.yaml-template | envsubst > agent-eval-base.yaml
kubectl apply -f agent-eval-base.yaml

# your fine-tune
export LABEL=agentic-workshop EVAL_MODEL=/fsx/agentic/runs/agentic-workshop/merged
cat agent-eval.yaml-template | envsubst > agent-eval-tuned.yaml
kubectl apply -f agent-eval-tuned.yaml
```

Read the resolved rate:

```bash
# Results are written to the FSx volume at /fsx/agentic/agent_eval/<LABEL>.summary.json.
# The eval Job pods are short-lived, so read the file from any running pod that mounts the FSx
# volume (for example the compare pod from step 5), or copy it out with kubectl cp:
kubectl cp compare:/fsx/agentic/agent_eval/agentic-workshop.summary.json ./agentic-workshop.summary.json
cat ./agentic-workshop.summary.json
```

> Note: `kubectl cp` requires a running pod that mounts the FSx volume. If the compare pod from
> step 5 is up, use it; otherwise any pod with the `fsx-pvc` volume mounted works. The base model's
> result is at `base.summary.json`. Expect base ~8-10% and the fine-tune ~30%+ `resolved`.

## 5. Compare interactively (GPU)

```bash
cat compare.yaml-template | envsubst > compare.yaml
kubectl apply -f compare.yaml
kubectl wait --for=condition=Ready pod/compare --timeout=900s
# Interactive menu (showcase / paste a bug / inject a bug / quick scoreboard):
kubectl exec -it compare -- python3 /workspace/compare.py --full "" \
  --yours /fsx/agentic/runs/agentic-workshop/adapter
```

`compare.py` also takes `--mode {menu,showcase,paste,inject,quick}` to pick a mode directly (handy
for a non-interactive run), e.g. append `--mode quick` for the 20-task scoreboard.

> The showcase scenarios are a small, illustrative set sampled at temperature 0.7, so a few tasks
> can swing either way on any given run and are not a measure of overall quality. For real resolved
> rates, use the evaluation in step 4 or `--mode quick`.

> If you interrupt the client (`Ctrl-C`) while a comparison is running, the in-pod vLLM engine can
> stay alive holding the GPU, and a later GPU step may fail with a "Free memory on device ... less
> than desired GPU memory utilization" error. Just delete and recreate the compare pod (or kill the
> stray `VLLM::EngineCore` process in it) to free the GPU.

Delete the pod when done to free the GPU: `kubectl delete pod compare`.

## 6. Export to GGUF (CPU)

```bash
# ghcr.io/ggml-org/llama.cpp:full-b11146 is the official llama.cpp "full" image (includes
# convert_hf_to_gguf.py and llama-quantize). Use :full for the moving-latest tag.
export LLAMACPP_IMAGE=ghcr.io/ggml-org/llama.cpp:full-b11146
export GGUF_NAME=qwen3-1.7b-agentic GGUF_SRC=/fsx/agentic/runs/agentic-workshop/merged
# This job's command uses runtime shell vars ($SRC, $O), so render with SCOPED envsubst
# (list only the template vars) so those shell vars are left intact.
cat gguf-convert.yaml-template | envsubst '${GGUF_SRC} ${GGUF_NAME} ${LLAMACPP_IMAGE}' > gguf-convert.yaml
kubectl apply -f gguf-convert.yaml
```

Produces `/fsx/agentic/gguf/${GGUF_NAME}-{f16,q8_0,q4_k_m}.gguf` for llama.cpp, Ollama or LM Studio.
