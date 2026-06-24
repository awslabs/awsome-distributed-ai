# NVIDIA Dynamo (SGLang) — LLM Inference on SageMaker HyperPod EKS

[NVIDIA Dynamo](https://docs.nvidia.com/dynamo/) is a distributed inference framework
providing KV cache-aware routing, disaggregated prefill/decode, and NIXL-based KV
transfer for LLM serving on Kubernetes. This test case serves two models on a SageMaker
HyperPod EKS cluster with the SGLang backend, each in **aggregated** and **disaggregated**
flavors — four self-contained scenarios driven by the Dynamo operator and its
`DynamoGraphDeployment` (DGD) CRD.

| Scenario | Model | Pattern | GPUs |
|---|---|---|---|
| `gpt-oss-agg`    | GPT-OSS-20B      | Aggregated (1 worker = prefill+decode) | 1 |
| `gpt-oss-disagg` | GPT-OSS-20B      | Disaggregated (prefill + decode, NIXL) | 2 |
| `qwen3.6-agg`    | Qwen3.6-27B-FP8  | Aggregated                             | 1 |
| `qwen3.6-disagg` | Qwen3.6-27B-FP8  | Disaggregated (prefill + decode, NIXL) | 2 |

- **Aggregated**: one worker handles the full request; the frontend routes by KV-cache overlap.
- **Disaggregated**: prefill and decode run on separate workers; the KV cache is transferred
  prefill→decode via NIXL (RDMA where EFA is available, otherwise TCP).

See [BENCHMARKS.md](BENCHMARKS.md) for measured TTFT / ITL / throughput, and
[OBSERVABILITY.md](OBSERVABILITY.md) for wiring Dynamo metrics into the HyperPod Grafana.

## Prerequisites

| Requirement | Detail |
|---|---|
| HyperPod EKS cluster | Two instance groups (see below); `kubectl` configured, `helm` v3+ |
| Control nodes | 2x `ml.c6i.xlarge` — platform components (operator, etcd, NATS) |
| GPU nodes | 2x `ml.g6e.4xlarge` (L40S 48GB), instance group named **`dynamo-workers`** |
| Shared filesystem | FSx for Lustre (model weights, mounted at `/fsx`) |
| Hugging Face token | **Not required** — both models are public. Only needed for gated models. |

> The GPU instance group **must** be named `dynamo-workers` — every scenario YAML uses
> this label for scheduling. For production disaggregation, use EFA-enabled instances
> (e.g. p5) so NIXL uses RDMA instead of TCP.

## 0. Configure environment

```bash
cp env_vars.example env_vars
# edit env_vars: AWS_REGION, AWS_ACCOUNT_ID, EKS_CLUSTER_NAME (defaults are validated values)
source env_vars
aws eks update-kubeconfig --name "$EKS_CLUSTER_NAME" --region "$AWS_REGION"
kubectl get nodes --show-labels | grep instance-group-name
```

## 1. Install the Dynamo platform

Installs the EBS CSI driver (with HyperPod volume permissions), the Dynamo CRDs, and the
platform — **operator** on the control nodes, **etcd + NATS** on the GPU workers (the
control nodes' pod cap can't host them). etcd + NATS provide the cross-node discovery the
operator wires into every scenario.

```bash
./scripts/01-install-platform.sh
# optional: send Dynamo metrics to the HyperPod Grafana dashboard (see OBSERVABILITY.md)
./scripts/01a-setup-observability.sh
```
Verify: `kubectl get pods -n dynamo-system` → operator, etcd, nats all `Running`.

## 2. Download a model

One-time download to FSx Lustre, shared across pods. Public models — no token needed.

```bash
./scripts/02-download-model.sh gpt-oss     # -> /fsx/models/openai-gpt-oss-20b
./scripts/02-download-model.sh qwen3.6     # -> /fsx/models/qwen3.6-27b-fp8
```

## 3. Deploy a scenario

```bash
./scripts/03-deploy-gpt-oss-agg.sh
./scripts/04-deploy-gpt-oss-disagg.sh
./scripts/05-deploy-qwen3.6-agg.sh
./scripts/06-deploy-qwen3.6-disagg.sh
```

Each scenario is a `DynamoGraphDeployment` (CRD) reconciled by the Dynamo operator into
the frontend + worker pods. Workers load the model from FSx (~2 min). Watch:
```bash
kubectl get dynamographdeployment -n dynamo-system
kubectl get pods -n dynamo-system -l nvidia.com/dynamo-graph-deployment-name=<name> -w
```
> DGD names are dot-free: `gpt-oss-agg`, `gpt-oss-disagg`, `qwen36-agg`, `qwen36-disagg`.

> **GPU budget:** each aggregated scenario uses 1 GPU; each disaggregated scenario uses 2.
> With 2 GPU workers you can run two aggregated scenarios at once, or one disaggregated.

## 4. Inference

```bash
./scripts/07-test-inference.sh     # one-shot: models + a chat (auto-detects the deployed scenario)
./scripts/07b-chat.sh              # interactive multi-turn streaming chat (auto-detects)
```
Pass a name (`gpt-oss-agg` | `gpt-oss-disagg` | `qwen36-agg` | `qwen36-disagg`) to override auto-detect.

Or connect directly (OpenAI-compatible) via port-forward to the operator-created frontend
service `<name>-frontend`:
```bash
kubectl port-forward -n dynamo-system svc/<name>-frontend 8000:8000
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"<id>","messages":[{"role":"user","content":"Hello"}],"max_tokens":200}'
```

## 5. Benchmark

AIPerf concurrency sweep, saved to `bench_results/<scenario>/`:
```bash
./scripts/08-benchmark.sh <scenario>
# custom: ./scripts/08-benchmark.sh qwen36-agg 2000 256
# custom: CONCURRENCIES="1,5,10,50" ./scripts/08-benchmark.sh gpt-oss-agg
```

## 6. Cleanup

```bash
./scripts/09-cleanup-inference.sh <scenario>   # or: all
./scripts/10-uninstall-platform.sh             # remove the platform (optional)
```

## Architecture notes

### HyperPod specifics
- Node labels use `sagemaker.amazonaws.com/instance-group-name` (not `node-group-name`).
- EBS CSI needs `sagemaker:AttachClusterNodeVolume` / `DetachClusterNodeVolume` (handled by `01`).
- Model weights live on shared FSx Lustre — no repeated downloads on pod restarts.
- The SGLang runtime image requires the NVIDIA runtime, so frontend pods also run on GPU
  nodes (without requesting a GPU).

### Per-model notes
- **GPT-OSS-20B**: MXFP4 weights; on L40S (Ada) it requires the `triton` attention backend.
- **Qwen3.6-27B-FP8**: hybrid (Gated DeltaNet + full attention) multimodal model. FP8
  weights fit a single L40S. For **disaggregated** serving the decode worker reserves a
  per-slot SSM/mamba state, so `--max-running-requests` is capped (16) and
  `--mem-fraction-static` lowered (0.80) in the disagg YAML to fit 48GB.

### Layout
```
scenarios/
  gpt-oss-agg/      gpt-oss-20b-agg.yaml
  gpt-oss-disagg/   gpt-oss-20b-disagg.yaml
  qwen3.6-agg/      qwen3.6-27b-fp8-agg.yaml
  qwen3.6-disagg/   qwen3.6-27b-fp8-disagg.yaml
  observability/    pod-monitor.yaml
scripts/            01..10 (install, download, deploy, test, benchmark, cleanup)
platform/           values.yaml (Dynamo platform Helm values)
```
Each YAML is a `DynamoGraphDeployment` reconciled by the operator into the frontend +
worker pods and their Services. The serving image is pinned in each YAML (`image:` field).

## Troubleshooting

Hard-won notes from bringing all four scenarios up on L40S — each is already applied in
the YAMLs/scripts here:

| Symptom | Cause | Fix (already applied) |
|---|---|---|
| `/v1/models` empty; frontend 404s on tokenizer | Frontend tried to pull the tokenizer/config from Hugging Face | The Frontend mounts FSx and runs `HF_HUB_OFFLINE=1` so it loads locally |
| Worker crashes: `ValueError: Invalid endpoint format` | A dot in the DGD `metadata.name` breaks the `dyn://<ns>.<comp>.<endpoint>` the operator builds | DGD names are dot-free (`qwen36-agg`, not `qwen3.6-agg`) |
| Disaggregated: 0 completion tokens; decode logs `Connection refused :12345` | NIXL bootstrap server bound to `127.0.0.1` | prefill + decode pass `--host 0.0.0.0` |
| Qwen disaggregated decode OOMs on 48GB | Hybrid SSM/mamba reserves a per-slot state | `--max-running-requests 16` + `--mem-fraction-static 0.80` in the disagg YAML |
| Re-applying a running DGD hangs (new pods `Pending`, old `Running`) | Operator Deployments use RollingUpdate; old pods hold the GPUs | Deploy fresh: `09-cleanup-inference.sh <name>` then re-deploy (or delete the old pods) |
| Operator pod stuck `ImagePullBackOff` on the rbac-proxy sidecar | The chart pins a retired `gcr.io/kubebuilder` image | `01-install-platform.sh` patches it to `quay.io/brancz/kube-rbac-proxy:v0.15.0` after Helm |
