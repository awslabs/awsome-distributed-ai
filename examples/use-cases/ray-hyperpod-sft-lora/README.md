# Ray on SageMaker HyperPod — SFT + LoRA and Ray Serve (Studio Workflow)

Fine-tune [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) on medical reasoning data using Ray TorchTrainer with LoRA, then serve the model with Ray Serve on the same Ray cluster — all on Amazon SageMaker HyperPod. The Ray cluster is created and managed through the [Ray on HyperPod](https://aws.amazon.com/blogs/machine-learning/introducing-new-ray-capabilities-on-sagemaker-hyperpod/) integration in SageMaker Studio.

Two variants are provided: **FSx** (shared file system) and **S3** (no shared file system needed).

> **Not for clinical use.** The fine-tuned model is a training demo. Its answers can be medically wrong and must not be used for diagnosis or treatment.

## Architecture

```text
SageMaker Studio
├── Create Ray Cluster (Tasks tab → RayCluster → Create)
├── Open Ray Dashboard (IAM-authenticated URL)
├── Open Grafana (pre-built Ray dashboards)
└── Connect Workspace (JupyterLab / Code Editor space)
        │
        ▼
HyperPod EKS Cluster
├── Ray Head Node (no GPU)
├── Ray Workers (GPU instances)
├── KubeRay Operator (manages Ray CRDs)
├── Job Monitoring Agent (hung job detection)
├── HyperPod Observability → Amazon Managed Grafana
└── Shared Storage: FSx for Lustre OR Amazon S3
```

## Files

Run the scripts in order. Each one takes the variant's config file.

| File | Description |
|------|-------------|
| `requirements.txt` | Packages added on top of the SageMaker Distribution 4.5.0 image (both variants). Pinned to the image versions; does not install Ray or PyTorch |
| `0.prepare_dataset.py` | Builds the chat-formatted dataset (900 train / 100 validation records by default) and writes it to FSx or uploads it to S3 |
| `1.train_ray.py` | FSx variant training script — LoRA fine-tuning with Ray TorchTrainer; data, checkpoints and output on `/mnt/fsx` |
| `1.train_ray_s3.py` | S3 variant training script — workers download data from S3, Ray writes checkpoints to S3, rank 0 uploads the model to S3. Heavy ML imports stay off the CPU head node |
| `2.serve.py` | Deploys the merged model as an HTTP endpoint with Ray Serve on the same cluster |
| `args.yaml` | FSx variant config — FSx paths, LoRA and SFT hyperparameters |
| `args-s3.yaml` | S3 variant config — `s3_base_uri` replaces the FSx paths; worker-local `/tmp` directories |
| `kubernetes/raycluster.yaml`, `kubernetes/workspace-pod.yaml` | Only for clusters without the Studio Ray integration: a Ray cluster and a workspace pod to run the steps from (see [Appendix](#appendix-run-without-the-studio-ray-integration)) |

### Key differences

| | FSx variant | S3 variant |
|-|-------------|------------|
| **Shared storage** | FSx for Lustre (`/mnt/fsx/`) | Amazon S3 (`s3_base_uri` in config) |
| **Data access** | Direct file system reads | Workers download from S3 to local `/tmp` |
| **Checkpoints** | Ray writes to FSx (`storage_path`, default `/mnt/fsx/ray-sft-lora/ray_results`) | Ray writes to S3 (`<s3_base_uri>/ray_results`) |
| **Model output** | `output_dir` on FSx | `<s3_base_uri>/output/` |
| **Infrastructure** | FSx PV/PVC on EKS | S3 bucket/prefix + EKS Pod Identity |

## Prerequisites

Four components must be installed on your HyperPod EKS cluster:

| Component | Purpose |
|-----------|---------|
| SageMaker Spaces EKS add-on | JupyterLab / Code Editor spaces that attach to Ray clusters |
| HyperPod Observability EKS add-on | Metrics collection, Grafana dashboards |
| KubeRay operator | Manages RayCluster, RayJob, RayService CRDs |
| HyperPod Ray Endpoint Operator | Authenticated endpoints for the dashboard and remote job submission |

Plus:

- A SageMaker Studio domain with access to the HyperPod cluster.
- **FSx variant**: an FSx for Lustre file system attached to the cluster and mounted at `/mnt/fsx` on the space and the Ray pods.
- **S3 variant**: an S3 bucket and prefix, the EKS Pod Identity Agent add-on, and the Pod Identity setup below.

See the [Ray on HyperPod getting started guide](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-eks-ray.html) for full setup details.

### IAM and EKS Pod Identity (S3 variant)

Ray pods on HyperPod cannot use the node's instance credentials, so every S3 call fails with `Unable to locate credentials` or `ACCESS_DENIED` until the pods get a role through EKS Pod Identity. The Ray head uses the `<ray-cluster-name>` service account, and the workers and the Studio space use `default`, in the Ray namespace. Use a namespace that holds only this Ray cluster and its space, so the association on `default` reaches only those pods.

1. Create an IAM role for EKS Pod Identity, following [Create a Pod Identity role](https://docs.aws.amazon.com/eks/latest/userguide/pod-id-role.html). Its trust policy lets only the Pod Identity service assume it:

    ```json
    {
      "Version": "2012-10-17",
      "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "pods.eks.amazonaws.com"},
        "Action": ["sts:AssumeRole", "sts:TagSession"]
      }]
    }
    ```

2. Give the role read/write access to your prefix only. Replace `<your-bucket>` and `<your-prefix>` (the bucket and path of `s3_base_uri` in `args-s3.yaml`):

    ```json
    {
      "Version": "2012-10-17",
      "Statement": [
        {
          "Sid": "ListThePrefix",
          "Effect": "Allow",
          "Action": "s3:ListBucket",
          "Resource": "arn:aws:s3:::<your-bucket>",
          "Condition": {"StringLike": {"s3:prefix": ["<your-prefix>", "<your-prefix>/*"]}}
        },
        {
          "Sid": "ReadWriteThePrefix",
          "Effect": "Allow",
          "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:AbortMultipartUpload"],
          "Resource": "arn:aws:s3:::<your-bucket>/<your-prefix>/*"
        }
      ]
    }
    ```

3. Associate the role with both service accounts:

    ```bash
    export AWS_REGION=<region>
    export EKS_CLUSTER_NAME=<eks-cluster-name>
    export RAY_NAMESPACE=<ray-namespace>
    export RAY_CLUSTER_NAME=<ray-cluster-name>
    export ROLE_ARN=<role-arn-from-step-1>

    for SA in default "$RAY_CLUSTER_NAME"; do
      aws eks create-pod-identity-association --region "$AWS_REGION" \
          --cluster-name "$EKS_CLUSTER_NAME" --namespace "$RAY_NAMESPACE" \
          --service-account "$SA" --role-arn "$ROLE_ARN"
    done
    ```

Pods pick up a new association only when they start, so create the associations before the Ray cluster and the space, or restart them afterwards.

## Instance Types

Qwen3-0.6B with LoRA trains about 0.5% of its parameters and fits on 1 GPU.

| Role | Instance | Count | Notes |
|------|----------|-------|-------|
| Worker | GPU instance with at least 8 allocatable vCPUs. Tested: FSx variant on `ml.g6e.48xlarge` (L40S), S3 variant on `g6.12xlarge` (L4) | 1 | Each Ray Train worker asks for 1 GPU and `cpus_per_worker` CPUs (default 8). On a smaller instance such as `g5.xlarge` (4 vCPUs), lower `cpus_per_worker`, or Ray never places the worker |
| Head / System | `m5.2xlarge` or similar | 1 | No GPU needed. Needs about 6 vCPUs and 24 GiB free; a busy HyperPod controller group may not have that |

Only the single-worker configuration above has been tested. To use more GPUs, raise `num_workers` (DDP); for models that do not fit on one GPU, also enable the commented-out FSDP settings in the config. Neither multi-worker training nor FSDP has been tested with this example.

## Step 1: Create a Ray cluster from SageMaker Studio

1. Open **SageMaker Studio** and choose **HyperPod** in the left sidebar.
2. Select your HyperPod cluster and open the **Tasks** tab.
3. Choose **RayCluster** in the task type list, then **Create Ray Cluster**:
    - **Cluster name**: for example `ray-medical-finetune`. This is `<ray-cluster-name>` below.
    - **Head instance type**: a CPU instance.
    - **Worker instance type**: a GPU instance (see [Instance Types](#instance-types)); **Worker count**: 1.
    - **Container image**: the default SageMaker Distribution image (Ray pre-installed).
    - Under advanced settings, **enable remote endpoints** for dashboard access and remote job submission.
4. Choose **Create** and wait until the status on the Tasks tab is **Running**.

If your Studio Tasks tab does not offer **RayCluster**, or the cluster lacks the Ray add-ons, see [Appendix: run without the Studio Ray integration](#appendix-run-without-the-studio-ray-integration).

## Step 2: Open the Ray dashboard

On the Tasks tab, choose your Ray cluster, then **Actions** → **Open Ray Dashboard**. Studio opens a short-lived, IAM-authenticated URL.

## Step 3: Attach a Studio space and get the code

1. Create a **HyperPod JupyterLab** or **Code Editor** space, select your Ray cluster in the Ray cluster integration option, and open the space. The space joins the cluster as a zero-compute worker node.
2. In a terminal in the space, clone this repository and set the names Studio shows on the Tasks tab:

    ```bash
    git clone https://github.com/awslabs/awsome-distributed-ai.git
    cd awsome-distributed-ai/examples/use-cases/ray-hyperpod-sft-lora

    export RAY_CLUSTER_NAME=<ray-cluster-name>
    export RAY_NAMESPACE=<ray-namespace>
    export RAY_DASHBOARD=http://${RAY_CLUSTER_NAME}-head-svc.${RAY_NAMESPACE}.svc.cluster.local:8265
    ```

    Pass this URL with `--address`. Do not export it as `RAY_ADDRESS`: the space already uses that variable to join the cluster, and `ray.init()` fails on an `http://` address.

3. **S3 variant**: set `s3_base_uri` in `args-s3.yaml` to `s3://<your-bucket>/<your-prefix>`.
4. **FSx variant**: prepare the FSx folder. The space and the Ray pods run as `sagemaker-user`, and the FSx root is usually owned by `root`, so create one folder for this example and give it to that user (once per file system):

    ```bash
    sudo mkdir -p /mnt/fsx/ray-sft-lora
    sudo chown "$(id -u):$(id -g)" /mnt/fsx/ray-sft-lora
    ```

    Every FSx path in `args.yaml` sits under this folder.

## Step 4: Install the packages in the space

```bash
pip install -r requirements.txt
```

Do not install another Ray version in the space: it must match the cluster's Ray (2.55.1 on the SageMaker Distribution 4.5.0 image), or `ray.init()` fails.

## Step 5: Prepare the dataset

```bash
python 0.prepare_dataset.py --config args-s3.yaml   # S3 variant
python 0.prepare_dataset.py --config args.yaml      # FSx variant
```

The script loads 1,000 samples of [FreedomIntelligence/medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) at a pinned revision, applies the Qwen3 chat template, splits 90/10, and writes `train.json` and `val.json` to the paths in the config.

## Step 6: Submit the training job

Ray uploads this folder (`--working-dir .`) to the cluster and installs `requirements.txt` on the head and the workers when the job starts.

S3 variant, from the space terminal:

```bash
ray job submit --address "$RAY_DASHBOARD" --working-dir . \
    --runtime-env-json '{"pip": "requirements.txt", "excludes": [".git", ".env", "*.pem"], "env_vars": {"USE_TF": "0", "TRANSFORMERS_NO_TF": "1", "NCCL_DEBUG": "WARN"}}' \
    -- python 1.train_ray_s3.py --config args-s3.yaml
```

FSx variant, from the space terminal:

```bash
ray job submit --address "$RAY_DASHBOARD" --working-dir . \
    --runtime-env-json '{"pip": "requirements.txt", "excludes": [".git", ".env", "*.pem"], "env_vars": {"USE_TF": "0", "TRANSFORMERS_NO_TF": "1", "NCCL_DEBUG": "WARN", "HF_HOME": "/mnt/fsx/ray-sft-lora/huggingface"}}' \
    -- python 1.train_ray.py --config args.yaml
```

From a laptop or CI/CD pipeline, the `toolkit-for-ray-on-sagemaker-ai` package resolves the cluster's IAM-authenticated endpoint through the `sagemaker_ray://` address:

```bash
pip install toolkit-for-ray-on-sagemaker-ai==1.0.4
aws eks update-kubeconfig --name <eks-cluster-name> --region <region>

ray job submit --address sagemaker_ray://<ray-cluster-name>/<ray-namespace> \
    --working-dir . \
    --runtime-env-json '{"pip": "requirements.txt", "excludes": [".git", ".env", "*.pem"], "env_vars": {"USE_TF": "0", "TRANSFORMERS_NO_TF": "1"}}' \
    -- python 1.train_ray_s3.py --config args-s3.yaml
```

### Resume after a failure

`FailureConfig(max_failures=3)` restarts the workers when a worker or node fails (HyperPod replaces a faulty node, and Ray reschedules the pods). The training function then loads the last checkpoint Ray reported and continues from that step.

Ray finds an earlier run by `storage_path` and `run_name`. Use a new `run_name` for each experiment. To resume a job that stopped, resubmit it with the same `run_name`.

## Step 7: Monitor the job

- **Ray dashboard**: job status, worker logs and resource use (Step 2).
- **Amazon Managed Grafana**: on the Tasks tab choose **Actions** → **Open Grafana**. The HyperPod Observability add-on provides Ray dashboards (Core, Data, Train, Serve).
- **Hung job detection**: the per-node Job Monitoring Agent detects stalled training and reports to the cluster's CloudWatch log group and the Ray Train Grafana dashboard. For custom log-pattern rules, see the `log_monitoring` module of [toolkit-for-ray-on-sagemaker-ai](https://pypi.org/project/toolkit-for-ray-on-sagemaker-ai/).
- **CLI**: `ray job status --address "$RAY_DASHBOARD" <job-id>` and `ray job logs --address "$RAY_DASHBOARD" <job-id>`.

When the job succeeds, the merged model and the LoRA adapter are in `<s3_base_uri>/output/` (S3) or `output_dir` (FSx).

## Step 8: Serve the model with Ray Serve

Submit the Serve script as a Ray job, so its driver runs on the head node:

```bash
ray job submit --address "$RAY_DASHBOARD" --working-dir . \
    --runtime-env-json '{"excludes": [".git", ".env", "*.pem"], "env_vars": {"USE_TF": "0", "TRANSFORMERS_NO_TF": "1"}}' \
    -- python 2.serve.py --config args-s3.yaml   # FSx variant: --config args.yaml
```

The deployment takes one GPU on a worker, loads the merged model, and listens on port 8000 of the Ray pods. It keeps running after the job ends. Test it from the space:

```bash
curl -s -X POST "http://${RAY_CLUSTER_NAME}-head-svc.${RAY_NAMESPACE}.svc.cluster.local:8000/medical-qa" \
    -H 'Content-Type: application/json' \
    -d '{"prompt": "What are the common symptoms of Type 2 Diabetes?", "max_tokens": 200}'
```

Requests accept a `prompt` of up to 4,000 characters and `max_tokens` from 1 to 512; anything else gets HTTP 400. The head service must expose port 8000; check with `kubectl get svc -n "$RAY_NAMESPACE"`. To serve more traffic, raise `num_replicas` in `2.serve.py`.

Remove the deployment when you are done. The image's `serve` command is the SageMaker inference server, not Ray's CLI, so use Ray's Serve REST API:

```bash
curl -X DELETE "$RAY_DASHBOARD/api/serve/applications/"
```

## Clean up

1. Remove the Serve deployment (Step 8).
2. On Studio's **Tasks** tab, choose your Ray cluster, then **Actions** → **Delete** (without Studio: see the [Appendix](#appendix-run-without-the-studio-ray-integration)).
3. Delete the data and models you no longer need: the `<s3_base_uri>/` prefix (S3), or `/mnt/fsx/ray-sft-lora` (FSx).
4. HyperPod nodes keep running after the Ray cluster is deleted. To stop compute charges, scale the instance groups to 0 or delete the HyperPod cluster.

## Security notes

- **No credentials in code or config.** The S3 variant gets credentials from EKS Pod Identity, scoped to one prefix (see above).
- **Hugging Face tokens.** Qwen3-0.6B and the dataset are not gated, so no token is needed. For a gated model, put `HF_TOKEN` in the Ray pods' environment from a Kubernetes Secret or AWS Secrets Manager. Do not put it in the YAML config or in `runtime_env` `env_vars`: Ray uploads the working directory to the cluster and shows the runtime env in the dashboard.
- **Ray endpoints have no authentication.** Anyone who can reach the dashboard and Jobs API (port 8265), the Ray client port (10001) or Serve (8000) can run code or use the GPUs. Do not expose these ports through a LoadBalancer, NodePort or Ingress. Use Studio's IAM-authenticated endpoints or `kubectl port-forward`, and restrict in-cluster access with Kubernetes NetworkPolicies if your cluster enforces them. See [Ray security](https://docs.ray.io/en/latest/ray-security/index.html).
- **Remote code.** The scripts load models without `trust_remote_code`. Keep it that way unless you have reviewed the model repository's code.

## Known issues

- **`Keras 3 ... not yet supported` / `Failed to import trl.trainer.sft_trainer`**: the SageMaker Distribution image ships TensorFlow 2.19 and Keras 3. The scripts set `USE_TF=0` and `TRANSFORMERS_NO_TF=1`; keep them in `env_vars` when you submit a job.
- **pip dependency-conflict warnings** (for example `fsspec` or `s3fs` versions) when installing `requirements.txt` in the space: `datasets==4.8.5` needs an older `fsspec` than the image. They do not affect this example, which does its S3 I/O through boto3.
- **Job stays pending**: the worker asks for 1 GPU and `cpus_per_worker` CPUs. Lower `cpus_per_worker` on small instances.
- **`flash_attention_2`**: needs the `flash-attn` package, which the image does not include. The configs use `sdpa`.
- **Serve requests are refused**: Ray Serve binds `127.0.0.1` by default; `2.serve.py` binds `0.0.0.0` (`--host`). The head service must also expose port 8000.
- **`PermissionError: '/mnt/fsx/...'`**: the image runs as `sagemaker-user` and the FSx root belongs to `root`. Create the example's folder once with `sudo` (Step 3).
- **`serve shutdown` fails with `/opt/ml/model/inference.py` not found**: in the SageMaker Distribution image, `serve` is the SageMaker inference server. Remove Serve apps with `curl -X DELETE "$RAY_DASHBOARD/api/serve/applications/"`.
- **Python bytecode lines in the worker log** (`LOAD_GLOBAL`, `RETURN_VALUE`, ...) while TRL prepares the dataset: printed by the dataset caching code; training is not affected.

## Appendix: run without the Studio Ray integration

Use this when Studio cannot create the Ray cluster or attach a space (for example, the HyperPod Ray Endpoint Operator is not installed). It replaces Steps 1 to 3; it was tested on a HyperPod EKS cluster with `ml.g6e.48xlarge` workers and an FSx PVC named `fsx-pvc`.

1. Check the names in `kubernetes/raycluster.yaml` and `kubernetes/workspace-pod.yaml`: the FSx PVC (`fsx-pvc`) and the GPU instance type. Then create the Ray cluster and the workspace pod:

    ```bash
    export RAY_NAMESPACE=<namespace-with-your-fsx-pvc>
    kubectl apply -n "$RAY_NAMESPACE" -f kubernetes/raycluster.yaml -f kubernetes/workspace-pod.yaml
    kubectl wait -n "$RAY_NAMESPACE" --for=condition=Ready pod -l ray.io/cluster=ray-medical-finetune --timeout=30m
    kubectl wait -n "$RAY_NAMESPACE" --for=condition=Ready pod/ray-workspace --timeout=30m
    ```

    The first start pulls the SageMaker Distribution images (several GB each), which can take 10 minutes or more.

2. Open a shell in the workspace pod and continue with Step 3, with `RAY_CLUSTER_NAME=ray-medical-finetune`:

    ```bash
    kubectl exec -it -n "$RAY_NAMESPACE" ray-workspace -- bash
    ```

3. When you are done, delete both:

    ```bash
    kubectl delete -n "$RAY_NAMESPACE" -f kubernetes/raycluster.yaml -f kubernetes/workspace-pod.yaml
    ```

Without the Ray Endpoint Operator there is no Studio dashboard link or `sagemaker_ray://` remote submission. Use the `ray job` commands from the workspace pod, or `kubectl port-forward svc/ray-medical-finetune-head-svc 8265:8265` for the dashboard.

## Key Packages

| Package | Purpose |
|---------|---------|
| SageMaker Distribution image 4.5.0 | Default Ray cluster image: Ray 2.55.1, PyTorch 2.8.0, transformers 4.57.6 |
| `trl`, `peft`, `datasets` | SFT training with LoRA (pinned in `requirements.txt`) |
| `toolkit-for-ray-on-sagemaker-ai` | Remote job submission through the `sagemaker_ray://` address |
