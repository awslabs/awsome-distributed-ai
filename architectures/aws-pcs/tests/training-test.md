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

   > **HuggingFace rate limits.** The test case streams its dataset from the Hub, so
   > every rank pulls from HF at startup. Large datasets (e.g. `allenai/c4`, 1024
   > shards) can return `429 Too Many Requests` under many concurrent ranks even with
   > an `HF_TOKEN` — the dataset is fetched, not the cluster, so this is an HF-account
   > limit, not a cluster issue. For large multi-node runs, use an HF account with a
   > higher rate limit, an HF mirror, or pre-tokenized data staged on `/fsx` (no
   > streaming).

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

## Test 7b: Megatron-LM GPT-3 (tensor + pipeline parallel)

A second distributed-training path that exercises 3D parallelism (TP/PP/DP) rather than
FSDP, using the repo's canonical Megatron-LM case
[`3.test_cases/megatron/megatron-lm`](../../../3.test_cases/megatron/megatron-lm). Unlike
the FSDP case it does **not** stream from HuggingFace — data is tokenized once to local
`.bin`/`.idx` on `/fsx`, so it has no HF rate-limit exposure. Two PCS-specific deltas:

1. **Build + import the image on the login node** (300 GiB root; the overlay can't live on
   Lustre), writing the `.sqsh` to `/fsx`:

   ```bash
   cd /fsx/awsome-distributed-ai/3.test_cases/megatron/megatron-lm
   sudo docker build -t aws-megatron-lm -f 0.distributed-training.Dockerfile .
   enroot import -o /fsx/aws-megatron-lm.sqsh dockerd://aws-megatron-lm:latest
   ```

2. **Preprocess once, then train**, passing `IMAGE`/`DATA_PATH`/`FSX_MOUNT` as env (the
   sbatch scripts read them). Preprocessing is CPU work but still needs a Pyxis-capable
   node and the login `srun` client:

   ```bash
   cd slurm/gpt3
   IMAGE=/fsx/aws-megatron-lm.sqsh DATA_PATH=/fsx FSX_MOUNT=/fsx:/fsx \
     sbatch -p gpu 1.data-preprocessing.sbatch          # → /fsx/my-gpt2_text_document.{bin,idx}
   # the training sbatch expects data under ${DATA_PATH}/gpt2/ — symlink it there
   mkdir -p /fsx/gpt2 && ln -sf /fsx/my-gpt2_text_document.* /fsx/gpt2/ \
     && ln -sf /fsx/gpt2-vocab.json /fsx/gpt2-merges.txt /fsx/gpt2/
   IMAGE=/fsx/aws-megatron-lm.sqsh DATA_PATH=/fsx FSX_MOUNT=/fsx:/fsx \
     sbatch --nodes=4 -p gpu 2.distributed-training.sbatch
   ```

   For a short smoke run, add `--exit-interval 20` and `--log-throughput` to the megatron
   args in `2.distributed-training.sbatch`. At `NODES≤4` the script picks `TP=4, PP=2,
   GBS=288` automatically.

**Expected:** NCCL over EFA (`NET/OFI Selected provider is efa, fabric is efa-direct (found
8 nics)` on p6-b200); `0` nan/skipped after warmup; `lm loss` begins descending once the
LR warmup kicks in (loss-scale auto-tuning skips the first ~16 steps with `lr=0` by design).

### Verified — p6-b200 ×4 (32 GPU, ap-south-1)

GPT-3 36-layer / hidden 4096 / 32 heads, seq 2048, TP=4 PP=2 (DP=4), GBS=288, fp16 +
activation recompute. Steady **~134 TFLOP/s/GPU** (iters 3–20, ~6.4 s/iter); `lm loss`
**10.91 → 10.46** over iters 17–20 as the cosine LR warmup ramps (grad norm 57→45); 0 nan,
0 skipped post-warmup. This is a general-purpose (un-tuned) container — not a peak MFU
number — but confirms TP+PP+DP 3D parallelism trains correctly over EFA on B200.

---
