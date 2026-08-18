# Troubleshooting

Symptoms and fixes for OOM, slow training, slow model loading, Megatron-Bridge errors, and
Sandbox Fusion.

For measurement and evaluation pitfalls (which are a different class of problem -- wrong
numbers rather than crashes) see [results.md](results.md).

## OOM Errors

1. Reduce batch size:
   ```bash
   python3 scripts/submit_training.py training.train_batch_size=128
   ```
2. Enable memory offloading:
   ```bash
   python3 scripts/submit_training.py compute.param_offload=true compute.optimizer_offload=true
   ```
3. Reduce LoRA rank:
   ```bash
   python3 scripts/submit_training.py lora.rank=32
   ```

## Slow Training

1. Increase GPU memory utilization:
   ```bash
   python3 scripts/submit_training.py compute.gpu_memory_utilization=0.9
   ```
2. Use dynamic batching: Already enabled by default
3. Check EFA connectivity: `fi_info -p efa`

## Slow Model Loading at Startup

If training startup shows `Fetching N files` progress bars repeated across all workers,
each worker is independently downloading model weights from HuggingFace Hub. This can
take 5-15+ minutes for large models.

1. Pre-download the model to FSx -- see [Download Models to FSx](#5-download-models-to-fsx-recommended)
2. The training scripts auto-detect local models at `${RAY_DATA_HOME}/models/${MODEL_NAME}`
3. Verify the download completed: check for `config.json` in the model directory
   ```bash
   kubectl exec -n <namespace> fsx-utils -- ls /fsx/data/verl/models/<MODEL_NAME>/config.json
   ```

## Megatron-Bridge Errors

Ensure Megatron-Bridge is installed correctly (provides the `megatron.bridge` namespace):
```bash
pip3 install --no-cache-dir --no-deps git+https://github.com/NVIDIA-NeMo/Megatron-Bridge@v0.5.0
```

And environment variables are set:
```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
export VLLM_USE_V1=1
```

## Sandbox Fusion Issues

**`ModuleNotFoundError: No module named 'pyext'`** during training with coding datasets:

This means Sandbox Fusion is not configured. Without a sandbox URL, verl falls back to
local `pyext`-based code execution, which fails on Python 3.12. Ensure `sandbox=enabled`
(the default) when running `submit_training.py`. The Sandbox Fusion service must be
running:

```bash
kubectl get pods -l app=sandbox-fusion
kubectl get svc sandbox-fusion
```

**`curl: (6) Could not resolve host`** when testing the sandbox URL:

DNS resolution failure. The most common cause is a namespace mismatch -- the URL uses one
namespace (e.g., `default`) but the service is deployed in another (e.g., `mvincig`). Check
which namespace your service is in:

```bash
kubectl get svc sandbox-fusion -o jsonpath='{.metadata.namespace}'
```

Ensure `KUBE_NAMESPACE` in `env_vars` matches your actual namespace. The sandbox URL
in `conf/sandbox/enabled.yaml` uses the namespace in the FQDN:

```yaml
# conf/sandbox/enabled.yaml
sandbox:
  url: http://sandbox-fusion.default.svc.cluster.local:8080/run_code
  #                         ^^^^^^^ must match your KUBE_NAMESPACE
```

**Sandbox Fusion pod is `Pending`** (not scheduling):

Check if the pod can schedule on a CPU node. It requires an m5.8xlarge (or whatever
`SANDBOX_INSTANCE_TYPE` is set to) and `privileged: true` security context:

```bash
kubectl describe pod -l app=sandbox-fusion
```

Common causes:
- No m5 node available (all used by Ray head)
- Pod security policy blocking privileged containers
- Node selector doesn't match any available nodes
