# Agentic coding fine-tune on SageMaker HyperPod (Qwen3-1.7B, SFT + LoRA)

This use case turns a small base model that is poor at **agentic coding** into one that reliably
fixes bugs through multi-turn tool use, using LoRA supervised fine-tuning (SFT) on recorded agent
trajectories, orchestrated by the SageMaker HyperPod training operator. The result is small enough
to export to GGUF and run on a laptop.

A coding agent runs inside an IDE or terminal: it explores a repository with tools, edits files,
runs the tests, and stops when the job is done. Small models struggle with *driving tools* over many
turns -- calling the right tool with valid arguments, making edits whose "old text" matches the file
exactly, re-running tests, and knowing when to stop. SFT on agent trajectories teaches these
behaviors.

On a 150-task bug-fix benchmark built from HumanEval (bugs injected by AST mutation, graded by
running the original unit tests in a sandbox):

| Model | Bugs fixed |
| --- | --- |
| Qwen3-1.7B base | 8.7% |
| Fine-tuned here (505 trajectories, ~22 min on 1 GPU) | 33.3% |
| Fine-tuned here (3,460 trajectories, 2 epochs, 4 GPUs) | 53.3% |
| Qwen3-4B base (reference, >2x the size) | 50.7% |

These are single-run numbers; the evaluation samples at temperature 0.7, so expect a few points of
run-to-run variance (base typically ~8-10%, the quick fine-tune ~30%+). The fine-tuned 1.7B model
matches a base model more than twice its size. Quantized to GGUF, the full model scores 54.7% at
Q8_0 and 42.0% at Q4_K_M.

## Repository Structure

```
examples/use-cases/agentic-coding-sft/
├── Dockerfile                 # training + evaluation image (AWS PyTorch DLC)
├── Dockerfile.compare         # vLLM image for the interactive comparison
├── src/
│   ├── requirements.txt
│   ├── prep_agent.py          # SWE-smith `tool` split -> tokenized train-8k / train-16k / val
│   ├── train_agent.py         # LoRA SFT (rank 64), multi-node, Liger fused CE, resume-on-restart
│   ├── make_agent_tasks.py    # build the 150 HumanEval bug-fix task repos
│   ├── agent_env.py           # sandboxed tools (bash, str_replace_editor, submit) + test grading
│   ├── agent_tools.py         # tool schema + system prompt
│   ├── agent_eval.py          # multi-turn agent evaluation loop (vLLM or an OpenAI-compatible server)
│   ├── compare.py             # interactive base vs. fine-tune comparison
│   └── show_agent.py          # transcript viewer
└── kubernetes/
    ├── README.md              # ordered run-through of the manifests
    ├── prep-agent.yaml-template
    ├── make-tasks.yaml-template
    ├── train-agent.yaml-template
    ├── agent-eval.yaml-template
    ├── gguf-convert.yaml-template
    └── compare.yaml-template
```

## Prerequisites

- An Amazon SageMaker HyperPod EKS cluster with the **HyperPod training operator** add-on, accessible
  via `kubectl`. See the [architectures directory](../../../architectures) for cluster setup.
- GPU quota for `ml.g5.2xlarge` (NVIDIA A10G, 24 GB): 1 node for the quick run, up to 4 for the full
  run. Plus a CPU node (e.g. `ml.c5.4xlarge`) for data prep and GGUF conversion.
- An Amazon FSx for Lustre persistent volume claim (templates default to `fsx-pvc`).
- Docker with internet access to build and push images to Amazon ECR.

### GPU topology and EFA

This example targets single-GPU `ml.g5.2xlarge` nodes. Each training pod uses **one GPU**
(`nproc_per_node=1`, `nvidia.com/gpu: 1`), and the full run scales **data-parallel across nodes**
by setting `NUM_NODES` (4 nodes x 1 GPU = 4-way data parallel). It does **not** use multiple GPUs
within a node, and it does **not** configure Elastic Fabric Adapter (EFA), because `ml.g5.2xlarge`
has a single GPU and no EFA; cross-node gradient all-reduce runs over TCP.

If you adapt this to an EFA-capable or multi-GPU-per-node instance (for example `ml.g5.8xlarge` or
`ml.p5.48xlarge`), update `train-agent.yaml-template` to use the node's GPUs and EFA interfaces:

- set `nprocPerNode` / `--nproc_per_node` and `nvidia.com/gpu` to the GPUs per node;
- request EFA with `vpc.amazonaws.com/efa: <interfaces-per-node>` under resources;
- add the EFA/NCCL environment variables (`FI_PROVIDER=efa`, `FI_EFA_FORK_SAFE=1`, and the relevant
  `NCCL_*` settings).

Without these, a larger instance will run but leave most of its GPUs idle and will not use EFA for
inter-node communication.

## 1. Build and push the images

```bash
export AWS_REGION=$(aws ec2 describe-availability-zones --output text --query 'AvailabilityZones[0].[RegionName]')
export ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export REGISTRY=${ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/

# Log in to the AWS DLC public registry and your private ECR
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin 763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${REGISTRY%/}

# Build (use buildx --platform linux/amd64 on Apple Silicon)
docker build --build-arg AWS_REGION=${AWS_REGION} -t ${REGISTRY}agentic-coding-sft:v1 -f Dockerfile .
docker build -t ${REGISTRY}agentic-coding-sft-compare:v1 -f Dockerfile.compare .

# Create repositories if needed, then push
for name in agentic-coding-sft agentic-coding-sft-compare; do
  aws ecr describe-repositories --repository-names $name >/dev/null 2>&1 || aws ecr create-repository --repository-name $name
done
docker push ${REGISTRY}agentic-coding-sft:v1
docker push ${REGISTRY}agentic-coding-sft-compare:v1
```

## 2. Run the pipeline

The ordered manifests -- prepare data, build tasks, train, evaluate, compare, export to GGUF -- are
documented in [kubernetes/README.md](kubernetes/README.md). In short:

1. **Prepare data** (`prep-agent.yaml-template`): downloads the SWE-smith `tool` split, keeps the
   resolved trajectories, converts them to Qwen3 tool-call format, adds an empty think block to each
   assistant turn so training matches inference, tokenizes with the Qwen3 chat template, and masks
   everything except the assistant turns. Produces an 8K-token set (505 trajectories) and a 16K-token
   set (3,460 trajectories).
2. **Build evaluation tasks** (`make-tasks.yaml-template`): injects bugs into HumanEval functions and
   wraps each in a small repo with the original unit tests.
3. **Train** (`train-agent.yaml-template`): LoRA SFT with rank 64. The quick run uses `train-8k`, 1
   epoch, 1 GPU (~22 minutes); the full run uses `train-16k`, 2 epochs, 4 GPUs. Checkpoints every 10
   steps, resumes automatically on restart, and merges the adapter for GGUF conversion.
4. **Evaluate** (`agent-eval.yaml-template`): runs the base model and your fine-tune as agents over
   the 150 tasks and reports the resolved rate.
5. **Compare** (`compare.yaml-template`): an interactive menu -- showcase scenarios, paste your own
   buggy function, inject a bug, or a quick scoreboard -- running base vs. your fine-tune side by side.
6. **Export to GGUF** (`gguf-convert.yaml-template`): converts the merged model to GGUF and quantizes
   to Q8_0 and Q4_K_M for llama.cpp, Ollama or LM Studio.

## How it works

- **Why trajectories.** Single-function coding ability is already present in the base model; what SFT
  on agent trajectories adds is the multi-turn tool-use behavior (valid tool calls, exact edits,
  re-running tests, submitting).
- **Train == inference.** The data is pre-tokenized with the Qwen3 chat template and an empty think
  block on every assistant turn, with the loss masked to the assistant turns only, so the training
  distribution matches how the model is prompted at inference. If you change the model family, re-run
  `prep_agent.py` -- the tokenization is tied to the tokenizer and template.
- **Execution-based grading.** The evaluation does not score text similarity; it runs the model's
  tool calls in a sandbox and runs the repo's unit tests to decide whether the bug is fixed.

## Customizing

| What | Where |
| --- | --- |
| Base model | `--model_id` in `train-agent.yaml-template` (and `prep_agent.py` for tokenization) |
| Data size (8K vs 16K) | `AGENT_DATA` env (`train-8k` or `train-16k`) |
| LoRA rank / alpha | `--lora_r` / `--lora_alpha` in `train-agent.yaml-template` |
| Epochs, nodes | `EPOCHS`, `NUM_NODES` (keep the effective batch at 8 via `GRAD_ACC`) |
| On-device / server evaluation | `agent_eval.py --server_url <endpoint>` against an OpenAI-compatible server |

## Credits

- Agent trajectories: the [SWE-smith](https://github.com/SWE-bench/SWE-smith) `tool` split.
- Bug-fix tasks derived from [HumanEval](https://github.com/openai/human-eval).
- Model: [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B).

## License

This project is licensed under the MIT-0 License. See the [LICENSE](../../../LICENSE) file.
