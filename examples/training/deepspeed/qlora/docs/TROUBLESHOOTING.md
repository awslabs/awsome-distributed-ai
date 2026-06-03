# Troubleshooting Guide

This guide covers common issues and solutions when running the QLoRA fine-tuning tutorial.

## Table of Contents

1. [HyperPod EKS Cluster Issues](#hyperpod-eks-cluster-issues)
2. [GPU and CUDA Issues](#gpu-and-cuda-issues)
3. [DeepSpeed and Multi-GPU Issues](#deepspeed-and-multi-gpu-issues)
4. [Training Issues](#training-issues)
5. [Docker Issues](#docker-issues)
6. [Model Loading Issues](#model-loading-issues)
7. [Inference Issues](#inference-issues)

---

## HyperPod EKS Cluster Issues

### Cannot Connect to Cluster

**Symptom**: `kubectl` commands fail with connection errors

**Solutions**:

1. Update kubeconfig for HyperPod EKS:
   ```bash
   aws eks update-kubeconfig --region $AWS_REGION --name <cluster-name>
   ```

2. Check context:
   ```bash
   kubectl config current-context
   kubectl config get-contexts
   ```

3. Verify cluster is active:
   ```bash
   aws eks describe-cluster --name <cluster-name> --query 'cluster.status'
   ```

### GPU Nodes Not Available

**Symptom**: GPU nodes show 0 or not ready

**Solutions**:

1. Check node status and labels:
   ```bash
   kubectl get nodes -l node.kubernetes.io/instance-type=ml.g5.12xlarge
   kubectl get nodes -o wide
   ```

2. Check HyperPod node health:
   ```bash
   kubectl get nodes -l sagemaker.amazonaws.com/node-health-status=Schedulable
   ```

3. If no GPU nodes are visible, check the HyperPod cluster instance groups
   in the SageMaker console or via `aws sagemaker describe-cluster`.

---

## GPU and CUDA Issues

### No GPU Detected

**Symptom**: `torch.cuda.is_available()` returns `False`

**Solutions**:

1. Check NVIDIA driver:
   ```bash
   nvidia-smi
   ```

2. Check CUDA version:
   ```bash
   nvcc --version
   ```

3. Reinstall PyTorch with CUDA:
   ```bash
   pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu126
   ```

### CUDA Out of Memory (Single GPU)

**Symptom**: `CUDA out of memory` or `CUDA error: an illegal memory access` during training on a single GPU

This is the most common issue with single-GPU training. The Qwen3-8B QLoRA
setup uses ~20-23GB on a 24GB A10G, leaving minimal headroom for memory spikes
during checkpoint saves.

**Solutions**:

1. **Switch to multi-GPU with DeepSpeed ZeRO-2** (recommended):
   ```yaml
   # In configs/training_config_zero2.yaml
   parallel:
     strategy: "deepspeed_zero2"
   ```
   Then launch with `torchrun --nproc_per_node=4` on a g5.12xlarge.

2. Reduce sequence length:
   ```yaml
   model:
     max_seq_length: 1024  # Reduced from 1536
   ```

3. Reduce batch size:
   ```yaml
   training:
     per_device_train_batch_size: 1
     gradient_accumulation_steps: 8  # Increase to compensate
   ```

4. Enable memory-friendly CUDA allocator:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
   ```

5. Enable gradient checkpointing (enabled by default):
   ```yaml
   parallel:
     gradient_checkpointing: true
   ```

### CUDA Out of Memory (Multi-GPU with ZeRO-2)

**Symptom**: OOM even with 4 GPUs and DeepSpeed ZeRO-2

ZeRO-2 shards optimizer states and gradients, but each GPU still holds the
full model parameters and activations. If you still OOM:

1. Reduce `max_seq_length` (e.g., 1536 -> 1024). Activation memory scales
   quadratically with sequence length in attention.

2. Switch to DeepSpeed ZeRO-3 which also shards model parameters:
   ```yaml
   parallel:
     strategy: "deepspeed_zero3"
   ```

3. Check for memory fragmentation:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
   ```

### NVIDIA Device Plugin Not Working

**Symptom**: Pods stuck in `Pending` with GPU requests

On HyperPod EKS, the NVIDIA device plugin is managed by the HyperPod Helm chart.

**Solutions**:

1. Check device plugin pods:
   ```bash
   kubectl get pods -n kube-system -l app=nvidia-device-plugin-ds
   kubectl logs -n kube-system -l app=nvidia-device-plugin-ds
   ```

2. Verify GPU resources are advertised:
   ```bash
   kubectl describe node <gpu-node-name> | grep -A5 "Allocatable"
   # Should show nvidia.com/gpu: 4 (or similar)
   ```

3. If device plugin pods are missing, verify the HyperPod Helm chart is installed:
   ```bash
   helm list -A | grep -i hyperpod
   ```

---

## DeepSpeed and Multi-GPU Issues

### NCCL Communication Errors

**Symptom**: `NCCL error`, `RuntimeError: NCCL communicator was aborted`, or
training hangs during the first all-reduce operation.

**Solutions**:

1. Check NCCL debug output:
   ```bash
   # Set in kubernetes/qwen3_8b-qlora-zero2.yaml env vars:
   NCCL_DEBUG=INFO
   NCCL_SOCKET_IFNAME=^lo
   ```

2. Verify all GPUs are visible within the pod:
   ```bash
   kubectl exec -it <pod-name> -n ml-training -- nvidia-smi
   ```
   You should see 4 GPUs listed.

3. Increase shared memory. NCCL uses `/dev/shm` for inter-GPU communication:
   ```yaml
   # In qwen3_8b-qlora-zero2.yaml (already configured):
   volumes:
     - name: shm
       emptyDir:
         medium: Memory
         sizeLimit: "64Gi"
   ```

### DeepSpeed Checkpoint Incompatibility

**Symptom**: `ValueError: Can't find a valid checkpoint at /fsx/qwen3-qlora/outputs/checkpoint-XXX`

This happens when trying to resume a DeepSpeed checkpoint from a non-DeepSpeed
run, or vice versa. DeepSpeed checkpoints contain `global_step*` directories;
vanilla HF Trainer checkpoints do not.

**Solutions**:

1. The training script auto-detects this mismatch and skips incompatible
   checkpoints. If you see this warning, training will restart from scratch.

2. To manually check checkpoint format:
   ```bash
   ls /fsx/qwen3-qlora/outputs/checkpoint-*/
   # DeepSpeed: contains global_step*/ directories
   # Vanilla HF: contains pytorch_model.bin or model.safetensors
   ```

3. To force a fresh start, delete old checkpoints:
   ```bash
   kubectl exec -it <pod-name> -n ml-training -- rm -rf /fsx/qwen3-qlora/outputs/checkpoint-*
   ```

### torchrun Errors

**Symptom**: `torchrun` fails to launch or processes crash immediately

**Solutions**:

1. Verify `--nproc_per_node` matches the number of GPUs. The PyTorchJob
   manifests use `nprocPerNode` which the Training Operator passes to
   `torchrun` automatically via the `PET_NPROC_PER_NODE` environment variable.

2. Check that `CUDA_VISIBLE_DEVICES` is NOT set. torchrun manages GPU
   assignment via `LOCAL_RANK`.

3. Verify the `--master_port` is not in use by another process.

---

## Training Issues

### Loss Not Decreasing

**Symptom**: Training loss stays flat or increases

**Solutions**:

1. Lower learning rate:
   ```yaml
   training:
     learning_rate: 1.0e-4  # Reduce from 2e-4
   ```

2. Increase warmup:
   ```yaml
   training:
     warmup_ratio: 0.1  # Increase from 0.03
   ```

3. Check data formatting:
   ```python
   # Inspect a sample
   from datasets import load_dataset
   ds = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en")
   print(ds['train'][0])
   ```

### Training Crashes Mid-Run

**Symptom**: Training stops unexpectedly

**Solutions**:

1. Check for OOM:
   ```bash
   dmesg | grep -i "killed process"
   ```

2. Auto-resume is enabled by default. The training script uses
   `--resume_from_checkpoint=auto` which finds the latest valid checkpoint:
   ```bash
   # Check if checkpoints exist
   kubectl exec -it <pod-name> -n ml-training -- ls /fsx/qwen3-qlora/outputs/
   ```

3. For EKS pods, check shared memory is large enough:
   ```yaml
   # In kubernetes/qwen3_8b-qlora-zero2.yaml (already configured):
   volumes:
     - name: shm
       emptyDir:
         medium: Memory
         sizeLimit: "64Gi"
   ```

### Slow Training

**Symptom**: Training is much slower than expected

**Solutions**:

1. Verify all GPUs are being used:
   ```python
   import torch
   print(f"GPUs available: {torch.cuda.device_count()}")
   for i in range(torch.cuda.device_count()):
       print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
   ```

2. Check for CPU bottleneck:
   ```yaml
   # Increase DataLoader workers
   data:
     num_workers: 8
   ```

3. Disable debugging:
   ```bash
   export CUDA_LAUNCH_BLOCKING=0
   ```

4. Check NCCL performance. Slow all-reduce can bottleneck multi-GPU training:
   ```bash
   export NCCL_DEBUG=INFO  # Check for warnings in logs
   ```

---

## Docker Issues

### Docker Build Fails

**Symptom**: `docker build` fails

**Solutions**:

1. Check Docker is running:
   ```bash
   docker info
   ```

2. Clear Docker cache:
   ```bash
   docker system prune -a
   ```

3. Build with verbose output:
   ```bash
   docker build --progress=plain -t qwen3-qlora-training:test .
   ```

### ECR Push Fails

**Symptom**: Cannot push image to ECR

**Solutions**:

1. Re-authenticate:
   ```bash
   aws ecr get-login-password --region us-east-1 | \
       docker login --username AWS --password-stdin \
       $(aws sts get-caller-identity --query Account --output text).dkr.ecr.us-east-1.amazonaws.com
   ```

2. Check ECR repository exists:
   ```bash
   aws ecr describe-repositories --repository-names qwen3-qlora-training
   ```

3. Check IAM permissions for ECR access

---

## Model Loading Issues

### Transformers Version Error

**Symptom**: `KeyError: 'qwen3'` when loading model

**Solution**: Upgrade transformers:
```bash
pip install 'transformers>=4.51.0'
```

### Model Download Fails

**Symptom**: Network error when downloading from Hugging Face

**Solutions**:

1. Use Hugging Face token:
   ```bash
   huggingface-cli login
   ```

2. Set HF_HOME for caching:
   ```bash
   export HF_HOME=/path/to/cache
   ```

3. Download manually:
   ```bash
   git lfs install
   git clone https://huggingface.co/Qwen/Qwen3-8B
   ```

### bitsandbytes Errors

**Symptom**: `bitsandbytes` import fails

**Solutions**:

1. Reinstall with CUDA:
   ```bash
   pip uninstall bitsandbytes
   pip install 'bitsandbytes>=0.42.0'
   ```

2. Check CUDA compatibility:
   ```python
   import bitsandbytes as bnb
   print(bnb.__version__)
   # Verify 4-bit quantization works
   import torch
   x = torch.randn(64, 64, dtype=torch.float16, device="cuda")
   print("bitsandbytes CUDA OK")
   ```

3. Verify bitsandbytes CUDA detection:
   ```bash
   python -c "import bitsandbytes; print(bitsandbytes.__version__)"
   # bitsandbytes >= 0.42.0 ships pre-compiled CUDA wheels on PyPI
   ```

---

## Inference Issues

### Full-Precision Model OOM

**Symptom**: Loading the merged bf16 model for inference runs out of memory

The merged full-precision model requires ~16GB VRAM. If your GPU has less:

1. Use 4-bit quantized loading (see `src/inference_demo.py` for an example)
2. Use CPU offloading:
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       "Qwen/Qwen3-8B",
       torch_dtype=torch.bfloat16,
       device_map="auto",  # Will offload layers to CPU if needed
   )
   ```

### LoRA Adapter Not Found

**Symptom**: `OSError: outputs/final_model is not a valid LoRA adapter`

**Solutions**:

1. Verify the adapter files exist:
   ```bash
   ls outputs/final_model/
   # Should contain: adapter_config.json, adapter_model.safetensors
   ```

2. If using the EKS-trained model, copy it from the PVC first:
   ```bash
   # Create a temp pod attached to the PVC, then:
   kubectl cp ml-training/<pod-name>:/fsx/qwen3-qlora/outputs/final_model ./outputs/final_model
   ```

---

## SageMaker HyperPod (Slurm) Issues

### CUBLAS_STATUS_INVALID_VALUE

**Symptom**: Training crashes on the first forward pass with:
```
RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling cublasGemmEx
```

**Background**: This error originates from PyTorch's cuBLAS linear algebra
calls, **not** from bitsandbytes — bitsandbytes does not call `cublasGemmEx`
in its 4-bit quantization kernels. There is no systemic incompatibility
between any particular CUDA toolkit version and bitsandbytes; `torch+cu128`
with bitsandbytes 4-bit quantization has been tested successfully on HyperPod
Slurm (torch 2.9.1+cu128, bitsandbytes 0.49.2, 4x A10G, DeepSpeed ZeRO-2).

The most common cause is **conflicting CUDA toolkit versions** on the host —
for example, a system-installed CUDA 12.1 `libcublas.so` being loaded instead
of the one bundled with PyTorch's cu128 wheel.

**Diagnosis**: Check your environment for library conflicts:

```bash
# Check driver version (needs >= 535.x for CUDA 12.x)
nvidia-smi

# Check PyTorch's CUDA runtime version
python -c "import torch; print(torch.version.cuda)"

# Check for conflicting CUDA libraries
ldconfig -p | grep libcublas
# If multiple libcublas.so paths appear pointing to different CUDA versions,
# that is likely the root cause.
```

**Workaround**: If you cannot resolve the library conflict, install a PyTorch
build with an older CUDA toolkit. The CUDA 12.8 driver is forward-compatible
with cu126 binaries:

```bash
pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu126
```

This sidesteps the conflict because the cu126 wheel carries fewer
system-level library dependencies that can clash on CUDA 12.8 hosts. Note
that the Docker container path (used by EKS) is unaffected because the
container bundles its own CUDA toolkit with no host-level conflicts.

### Sbatch script cannot find training code

**Symptom**: `ModuleNotFoundError: No module named 'src'` or file-not-found
errors for config files.

**Root cause**: Slurm copies the batch script to `/var/spool/slurmd/`, so
`realpath "$0"` resolves to the spool directory, not the original location.

**Solution**: Use `SLURM_SUBMIT_DIR` to derive paths:

```bash
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    REPO_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"
fi
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
```

### NCCL abort hides real Python errors

**Symptom**: All ranks crash with `NCCL communicator was aborted` but no
Python traceback is visible.

**Root cause**: When one rank raises an exception after NCCL is initialized,
the other ranks see an NCCL abort. The original exception is lost.

**Solution**: Wrap the suspected code in `try/except` with
`traceback.print_exc()`. Common hidden errors include missing packages
(tensorboard) and deprecated API calls.

### HyperPod auto-resume with Enroot containers

**Symptom**: `srun: error: unrecognized option '--container-image'` when both
`--auto-resume=1` and `--container-image` are used.

**Solution**: Skip auto-resume in container mode. The HyperPod auto-resume
wrapper does not pass through Pyxis plugin flags. See the sbatch scripts for
the conditional logic.

---

## Getting Help

If you're still stuck:

1. Check the logs:
   ```bash
   # EKS pod logs
   kubectl logs qwen3-qlora-training-zero2-master-0 -n ml-training

   # Previous pod attempt logs (if restarted)
   kubectl logs qwen3-qlora-training-zero2-master-0 -n ml-training --previous
   ```

2. Create an issue on GitHub with:
   - Full error message
   - Environment details (GPU, CUDA version, Python version)
   - Steps to reproduce

3. Common commands for debugging:
   ```bash
   # System info
   nvidia-smi
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   python -c "import transformers; print(transformers.__version__)"

   # EKS info
   kubectl get pods -n ml-training
   kubectl describe pod <pod-name> -n ml-training
   kubectl get events -n ml-training --sort-by='.lastTimestamp'
   ```
