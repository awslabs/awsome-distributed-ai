# Lessons Learned

Hard-won knowledge from optimizing Qwen3-8B pre-training on H200 and B300 clusters.

---

## EFA Silent Fallback to TCP

**Symptom:** Multi-node training runs but at single-node throughput. NCCL reports no errors.

**Root cause:** NCCL silently falls back to TCP sockets when the OFI plugin isn't loaded.

**Fix — all three are required:**

```bash
export LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib:$LD_LIBRARY_PATH
export NCCL_TUNER_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-tuner-aws-ofi.so
export FI_PROVIDER=efa
```

**Verification:** Look for `NCCL INFO NET/OFI` in logs (not `NET/Socket`).

The OFI NCCL plugin directory may use either `/opt/amazon/ofi-nccl/lib/` or `/opt/amazon/aws-ofi-nccl/lib/` depending on the EFA installer version. Check which exists.

---

## Enroot Import Workflow

**Never use `mksquashfs` directly.** It produces images that PyXis can't launch.

**Correct workflow:**

```bash
# 1. Build with Docker
sudo docker build -t my-image:latest .

# 2. Import with enroot (requires sudo for Docker socket)
sudo TMPDIR=/fsx/ubuntu/qwen3-8b-pretraining/tmp \
     ENROOT_TEMP_PATH=/fsx/ubuntu/qwen3-8b-pretraining/tmp \
     enroot import --output /fsx/ubuntu/qwen3-8b-pretraining/containers/image.sqsh dockerd://my-image:latest

# 3. Fix permissions
sudo chown $USER:$USER /fsx/ubuntu/qwen3-8b-pretraining/containers/image.sqsh
```

**Three requirements:**
1. `sudo` — Docker socket is `root:docker`, user not in docker group
2. `TMPDIR` on FSx — NeMo containers are 30+ GB, `/tmp` won't fit
3. `docker buildx use default` — image must be in main containerd store

---

## Slurm/PyXis Gotchas

### Slurm NOT in PATH
On some clusters, Slurm binaries live at `/opt/slurm/bin/`. Use full paths if needed:
```bash
/opt/slurm/bin/sbatch script.sh
/opt/slurm/bin/squeue -u ubuntu
/opt/slurm/bin/scontrol show hostname $SLURM_NODELIST
```

### Shell vars don't pass into containers
PyXis containers don't inherit the calling shell's environment. Pass explicitly:
```bash
srun --container-env=VAR1,VAR2,VAR3 ...
```

### Resolve MASTER_ADDR before srun
`scontrol` is not available inside PyXis containers. Compute the head node in the batch script, before the `srun` call:
```bash
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
```

### ntasks-per-node for torchrun
When using `torchrun` (which spawns GPU workers itself), set `--ntasks-per-node=1` in the Slurm script. If using raw `python` with NCCL init, use `--ntasks-per-node=8`.

### Single-node: disable EFA
For intra-node-only jobs, `FI_PROVIDER=efa` causes NCCL failures. Remove it or set `FI_PROVIDER=shm` for single-node debugging.

---

## torch.compile Incompatibility

Set `TORCH_COMPILE_DISABLE=1` in all environments. It fails in every configuration tested (DeepSpeed, HuggingFace multi-node, NeMo 25.07, NeMo 26.02). The performance gain would be minimal since Transformer Engine already provides fused kernels.

---

## Distributed Optimizer Trap at DP=1

`--use-distributed-optimizer` with `DP=1` (single GPU or TP-only parallelism) causes a crash. The sharding logic divides by DP world size and expects DP>=2.

**Rule:** Only enable distributed optimizer when DP>=2. For single-GPU debugging, remove the flag.

---

## Megatron-Bridge API Gotchas (NeMo 26.02)

- **Logging interval** is on `cfg.logger.log_interval`, not `cfg.train.log_interval` (silently ignored)
- **Disable gradient checkpointing** with `cfg.train.recompute_granularity = None` (not `""` or `False`)
- **Checkpoint directory** is `cfg.train.dir` (not `cfg.train.save` or `cfg.train.checkpoint_dir`)
- **Qwen3 bridge recipe** `qwen3_8b_pretrain_config()` provides correct model dimensions — don't manually override

---

## Memory Budget: H200 vs B300

```
H200 (141 GB available):
  Model (BF16):           16 GB
  Gradients (BF16):       16 GB
  Optimizer (sharded/16):  3 GB
  Activations (recompute): ~100 GB  <- with full recompute, MBS=2
  Overhead:                 3 GB
  Total:                 ~138 GB

B300 (288 GB available):
  Model (BF16):           16 GB
  Gradients (BF16):       16 GB
  Optimizer (sharded/16):  3 GB
  Activations (no recomp): ~135 GB  <- NO recompute needed, MBS=4
  Overhead:                 3 GB
  Total:                 ~173 GB (115 GB headroom)
```

B300's extra memory means no recompute overhead -> ~20% fewer FLOPs per step -> directly translates to 1.96x throughput combined with higher peak FLOPS.
