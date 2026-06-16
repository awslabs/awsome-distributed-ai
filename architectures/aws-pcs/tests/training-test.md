# Training Tests (Test 7)

Validates end-to-end distributed training using the repository's canonical
FSDP Llama-2 7B test case.

---

## Test 7: FSDP sample training

A short FSDP Llama-2 7B run, using the repo's canonical training case
[`3.test_cases/pytorch/FSDP`](../../../3.test_cases/pytorch/FSDP) and its
[`slurm/llama2_7b-training.sbatch`](../../../3.test_cases/pytorch/FSDP/slurm/llama2_7b-training.sbatch).
Follow that case's README; the only PCS-specific deltas are where things live on the
shared filesystems and the node count:

1. **Build the venv on shared `/fsx`** so every compute node sees it (the canonical
   `slurm/create_venv.sh` creates `./env` in place — run it from a `/fsx` checkout):

   ```bash
   cd /fsx && git clone --depth 1 https://github.com/awslabs/awsome-distributed-ai.git
   cd /fsx/awsome-distributed-ai/3.test_cases/pytorch/FSDP/slurm && bash create_venv.sh
   export HF_TOKEN=hf_xxx          # gated Llama-2 tokenizer
   ```

2. **Keep the HuggingFace cache on `/fsx` (Lustre), not `/home` (NFS).** Concurrent rank
   file-locking on NFS throws `OSError: [Errno 116] Stale file handle`; export
   `HF_HOME=/fsx/.hf-cache` before submitting.

3. **Submit 2 nodes** (the canonical sbatch defaults to 4) on your GPU queue. The venv must
   be on `PATH` for `torchrun` to resolve on every node — point `PATH` at the shared venv via
   `--export` (the canonical sbatch doesn't `activate` it):

   ```bash
   sbatch --nodes=2 --partition=gpu-p6b200 \
     --export=ALL,PATH=/fsx/awsome-distributed-ai/3.test_cases/pytorch/FSDP/slurm/env/bin:$PATH,HF_HOME=/fsx/.hf-cache \
     llama2_7b-training.sbatch
   ```

### Option B — run it in an Enroot/Pyxis container instead of the venv

The same canonical sbatch switches to container mode when `CONTAINER_IMAGE` is set (it adds
`--container-image`/`--container-mounts` and runs `./train.py` inside). Build the image once
on the login node and submit with `CONTAINER_IMAGE` — no venv needed:

```bash
# On the login node (300 GiB root disk + Docker), build + import to /fsx:
cd /fsx/awsome-distributed-ai/3.test_cases/pytorch/FSDP
sudo docker build -t fsdp:pytorch -f Dockerfile .
enroot import -o /fsx/pytorch-fsdp.sqsh dockerd://fsdp:pytorch

# Submit (container mode; mounts $(pwd) into /fsx inside the container):
cd slurm && sbatch --nodes=2 --partition=gpu-p6b300 \
  --export=ALL,CONTAINER_IMAGE=/fsx/pytorch-fsdp.sqsh,HF_HOME=/fsx/.hf-cache,HF_TOKEN=hf_xxx \
  llama2_7b-training.sbatch
```

> If the Dockerfile's `FROM` tag (a `public.ecr.aws/hpc-cloud/nccl-tests` tag) has been
> rotated out of the registry, substitute a current tag from that repo before building.

**Expected** (in `logs/llama2_7b-FSDP_<jobid>.out`), either path:
- NCCL initializes over EFA (`found N nics`) and training logs ~100 steps + a validation
  step, saving checkpoints under `./checkpoints`.
- Throughput per step, e.g. on 2× p6-b300 **~200 TFLOPS/GPU, ~77k tokens/s** (venv) /
  **~193 TFLOPS/GPU** (container); ~60 TFLOPS on 2× p5/H100. (Loss is constant at ln(vocab)
  in this smoke test — a known dataloader/vocab quirk of the test case, not a cluster problem.)

> **Multi-NIC tip:** the canonical sbatch already sets `NCCL_SOCKET_IFNAME=^docker,lo,veth,eth`
> (NCCL auto-selects). Do **not** pin a single interface on P5/P6 — all NICs share one
> subnet and pinning one breaks the cross-node NCCL bootstrap ring.

---
