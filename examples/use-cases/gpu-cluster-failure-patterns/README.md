# GPU cluster failure patterns on AWS

This example is the lab code of AIM344, a Builders' Session on diagnosing distributed GPU jobs and verifying recovery. It runs six incidents on a pair of GPU nodes under Slurm with Pyxis and Enroot: a slow job, a checkpoint write that fills its storage, a DataLoader that stops, a wrong result on one GPU, a device-side assertion, and an EFA device reset under a running job.

The lab assumes that you own the cluster for the session: one login node and two GPU nodes of the same type (measured on g7.48xlarge, 8 GPUs and 2 EFA devices per node, AWS PCS), your account can `sudo` on every node, and the cluster can be rebuilt. Diagnosis uses the [GPU Cluster Health Check Suite](../../../validation/gpu-cluster-healthcheck) at the commit in [pins.env](pins.env), `journalctl -k`, `nvidia-smi` and the job logs.

## Files

| File | Purpose |
|---|---|
| `job.sh` | The launcher: `notes`, `run`, `incident [N]`, `rerun`, `resume` |
| `lab.sh` | Commands on the target node: `status`, `collect`, `reset-efa`, `recover` |
| `workload.py` | The training stand-in, the epoch job and their fault modes |
| `common.sh`, `torch-node.sh`, `sweep-rank.sh`, `instance-type.sh` | Slurm and container plumbing for the jobs |
| `check5.sh` | Check 5 of the health suite (two-node NCCL all-reduce) with EFA counter deltas |
| `plugin-state.sh`, `recover-checkpoint.py` | The change and the repair of Incidents 1 and 2 |
| `setup-node.sh`, `prejob-prolog.sh`, `profile.sh` | Node preparation, the pre-job health check and the login profile |
| `efa-counters.py`, `summarize.py`, `inventory.sh` | Measurements and summaries |
| `Dockerfile`, `requirements.txt`, `pins.env` | The two container images and their pins |

## Set up

1. Build both targets of the [Dockerfile](Dockerfile) (`nccl-baseline` and `aim344`), import them with `enroot import`, and copy `aim344.sqsh` and `nccl-baseline.sqsh` to `/opt/aim344` on both GPU nodes.
2. Copy this directory to `/opt/aim344/device-recovery/companion` on the login node and both GPU nodes.
3. On each GPU node, run `sudo bash setup-node.sh ubuntu <health suite archive>`. It unpacks the health suite, installs the Prolog and mounts the 32 MiB checkpoint tmpfs as a systemd mount unit.
4. Write `lab.env` in the lab directory on all three nodes, and source `profile.sh` from `~/.bashrc` on the login node:

    ```bash
    export PARTITION=gpu GPUS_PER_NODE=8 CPUS_PER_RANK=24 MEMORY_PER_NODE=0
    export AIM344_TARGET_NODE=gpu-1 AIM344_PEER_NODE=gpu-2 AIM344_EFA_DEVICE=rdmap83s0
    export AIM344_SLURM_BIN=/opt/aws/pcs/scheduler/slurm-25.11/bin
    export NCCL_ENROOT_IMAGE=/opt/aim344/nccl-baseline.sqsh TORCH_IMAGE=/opt/aim344/aim344.sqsh
    export NCCL_CONTAINER=/opt/aim344/nccl-baseline.sqsh CHECKPOINT_DIR=/run/aim344-checkpoints
    export AIM344_SUITE_ENTRY=/opt/aim344-healthcheck/validation/gpu-cluster-healthcheck/gpu-healthcheck.sh
    export HEALTHCHECK_EXPECTED_EFA_COUNT=2 FI_EFA_USE_HUGE_PAGE=0 NCCL_PROTO=Simple
    ```

The workshop's package builder does these steps for an AWS PCS cluster.

## Run

Take both nodes, then run the healthy jobs and an incident:

```bash
salloc --job-name=aim344-job --exclusive --partition="$PARTITION" --nodes=2 --gres="gpu:$GPUS_PER_NODE" --ntasks-per-node=1 --cpus-per-task="$((GPUS_PER_NODE * CPUS_PER_RANK))" --mem="$MEMORY_PER_NODE"
bash job.sh run
bash job.sh incident 1
bash job.sh resume
```

Job logs are in `$RESULTS_DIR` on the nodes (`/var/tmp/aim344-<user>-<job>`), a copy of each torch job's output is in `~/aim344-results/device-<job>.log`, and the launcher's own steps are in `~/aim344-results/lab.log`.

## How each incident is caused

Each incident is a short recipe that you can lift into your own drills. `job.sh incident N` runs the recipe; the table gives the mechanism.

| N | What breaks | How it is caused | How it is undone |
|---|---|---|---|
| 1 | The all-reduce sweep runs over sockets and is slow | `plugin-state.sh hide` renames `/opt/amazon/ofi-nccl/lib` in the job's container, so NCCL cannot load aws-ofi-nccl | `plugin-state.sh restore` |
| 2 | A checkpoint write fails with `ENOSPC` | `workload.py device --inject --writer-rank=9` fills the 32 MiB checkpoint tmpfs before the first save | `recover-checkpoint.py` removes the filler; the job resumes with `--resume` |
| 3 | The second epoch never delivers a batch | `workload.py dataloader --start-method fork --epochs 2 --dataset-cache=rank-thread`: a thread of the rank holds the dataset's lock when the workers are forked | `--dataset-cache=worker-thread` or `--reinit-in-worker`; `--dataloader-timeout` only detects it |
| 4 | One rank's result differs and nothing fails | `workload.py device --replica-check --corrupt-rank=13 --corrupt-step=37 --corrupt-bit=22` flips one bit on one GPU | Resume from the checkpoint before the step: `--resume --resume-before=37` |
| 5 | A device-side assertion ends one rank (Xid 43 in the kernel log) | `workload.py device --assert-rank=0 --assert-step=201` indexes out of range on the GPU | A new process: `--resume` |
| 6 | Every rank reports an NCCL error at once | A PCIe function-level reset of one EFA device while the job runs (`lab.sh reset-efa`) | Reboot through Slurm (`lab.sh recover`) |

The EFA reset is two writes as root on the node, after draining it so that Slurm starts nothing new there:

```bash
sudo scontrol update nodename=<node> state=drain reason=aim344-device-recovery
srun --jobid=<job> --overlap -w <node> sudo sh -c 'cd /sys/class/infiniband/<efa device>/device && echo flr > reset_method && echo 1 > reset'
```

Recovery is a reboot that returns the node to service when `slurmd` registers again:

```bash
sudo scontrol reboot ASAP nextstate=RESUME reason=aim344-device-recovery <node>
```

After the reboot, run Checks 2 and 6 on the node (`bash lab.sh collect 2`, `bash lab.sh collect 6`) and a short job on the pair (`bash job.sh run`) before you trust it again.

## Tests

`python3 -m unittest discover -s tests` checks the scripts' syntax and starts each incident of `job.sh` against stub `srun` and `scontrol` commands. It needs no GPU. The incidents themselves were measured on g7.48xlarge; a change to `job.sh`, `lab.sh` or `workload.py` needs a run on a GPU pair.
