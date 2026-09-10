# GPU cluster failure patterns on AWS

This is the runnable lab draft for re:Invent session AIM344, a 60-minute Builders' Session at level 300. Diagnose in this order: **storage, host, fabric, software**. Preserve evidence, isolate a suspect node, and replace it rather than attempting repairs during the session.

The Workshop Studio participant guide targets **two g7.48xlarge primary or g7.24xlarge secondary nodes per attendee on AWS PCS**. The earlier rehearsal used two g7e.12xlarge nodes with Slurm version 25.05. The Spain rehearsal used two physical g7.48xlarge nodes with a constrained four-GPU allocation per node. Historical p5.48xlarge reference measurements and Seoul p6-b300.48xlarge EKS observations retain their original scope; each keeps its measured allocation label.

## Validation status

The September PCS rehearsal below used four GPU ranks across two `g7e.12xlarge` nodes in Oregon. Historical Seoul measurements used sixteen GPU ranks across two `p6-b300.48xlarge` nodes; they remain in [VALIDATION.md](VALIDATION.md) with their original scope.

| Round | Observed g7e PCS signature | Remaining boundary |
| --- | --- | --- |
| Round 0: healthy sweep | At 2 GiB, 33.49 GB/s out of place and 33.45 GB/s in place, zero mismatches and positive EFA byte changes. | No A100 production baseline or real arithmetic fault was measured. |
| Round 0: pre-job gate | Synthetic verdict drained the selected node; the job requeued into a held state and completed on the healthy alternate after release. | Real-hardware fault and replacement branches remain untested. |
| Round 1: plugin absent | `NET/Socket`, 13.51 GB/s out of place and 12.98 GB/s in place at 2 GiB, zero mismatches and zero EFA byte growth. Recovery reached 33.45 GB/s out of place and 33.34 GB/s in place. | The missing directory includes network and tuner libraries. Production A100 timing remains unmeasured. |
| Round 2: checkpoint exhaustion | Private 32 MiB tmpfs produced `ENOSPC` and peer watchdog timeouts at approximately 30 s. Recovery resumed the saved checkpoint; the subsequent sweep reached 33.37 GB/s out of place and 33.33 GB/s in place at 2 GiB with zero mismatches. | FSx quotas and an actual FSx outage remain untested. |
| Round 3: late DataLoader workers | Both fork and spawn completed four batches with zero mismatches. The OFI plugin enabled automatic EFA fork safety. | The historical hang did not reproduce on this stack. |

See [the validation record](VALIDATION.md) for exact commands, timings, image digests, raw logs and health-suite limitations. All participant rounds completed in the real PCS allocation. The prior p5.48xlarge reference remains in the 400 GB/s class at 2 GiB; it is not a threshold for g7e or A100. A passing health check and correct workload arithmetic remain separate claims.

## Prerequisites and deployment

Use the repository's [PCS reference architecture](../../../architectures/aws-pcs/README.md) to provision the production environment before the session. Match the deployed Slurm version and GPU-node configuration to the assigned instance type, use a single Availability Zone and EFA-enabled placement, and provide a shared path for this checkout and logs. The Workshop Studio wrapper routes both G7 shapes to their family-specific node template and retains the existing P4 and P5/P6 alternatives. The participant node pair must be exclusive, powered on, and idle before the lab. Connect to the login and compute nodes through Systems Manager.

Provision a private attendee checkpoint path with a bounded quota. Do not exhaust a shared filesystem. The injection writes at most 64 MiB and refuses to continue if that does not exhaust the fixture. A 32 MiB private tmpfs can establish the mechanism during validation; it does not validate Lustre quotas or an FSx outage. Keep checkpoint storage separate from `/dev/shm`, which NCCL may use.

The host prerequisites are a compatible NVIDIA driver, EFA devices, Python version 3, Enroot version 3.5.0, and Pyxis version 0.20.0 compiled against the deployed Slurm release. Record the actual AMI ID, driver version, and Slurm build during the PCS dry run. The prior p5.48xlarge environment used driver version 595.71.05; the Seoul driver is recorded in the measurement evidence.

The base-image pins are in [pins.env](pins.env), and the resolved PyTorch dependencies are pinned in [requirements.txt](requirements.txt):

| Dependency | Pin |
| --- | --- |
| NCCL test image | `public.ecr.aws/hpc-cloud/nccl-tests@sha256:a5390d3f0eb50f3e5854085e2ae0fef90b9eb4f3b7486ed4cab0d47864809d4c` |
| Registry tag resolved to that digest | `cuda13.0.2-efa1.48.0-ofiv1.19.0-ncclv2.30.4-1-testsv2.18.3` |
| Bundled CUDA / NCCL / nccl-tests | CUDA version 13.0.2 / NCCL version 2.30.4 / nccl-tests version 2.18.3 |
| Bundled EFA / aws-ofi-nccl | EFA installer version 1.48.0 / aws-ofi-nccl version 1.19.0 |
| Observed container libfabric on Seoul p6-b300.48xlarge | Version 2.4.0amzn3.0 |
| PyTorch for checkpoint and DataLoader workloads | `torch==2.9.0+cu130`; the wheel reports build NCCL version 2.27.7, while runtime startup logs confirm the base image preloads NCCL version 2.30.4 |
| Lightweight health suite from this repository | Commit `0e2c2c5b47f434380fefb3ff6bb30938a9c3a606` |

Build both Docker targets from the pinned base, then stage their SquashFS images on shared storage. The targets rebuild the pinned NCCL and nccl-tests source versions with native kernels for the Blackwell rehearsal GPU as well as the production GPU architectures. The published base lacks `sm_120` kernels and a usable PTX fallback. Retain both image checksums with the dry-run evidence. The facilitator helper performs the builds, imports and pinned health-archive verification:

```bash
cd examples/use-cases/gpu-cluster-failure-patterns
bash facilitator/prepare-login.sh
sha256sum /fsx/aim344/aim344.sqsh /fsx/aim344/nccl-baseline.sqsh
```

Pre-import both staged images through Pyxis on the assigned compute nodes before attendees arrive, as shown in the facilitator runbook. Pyxis retains named containers across steps; these scripts name their private containers with the Slurm job ID and remove only those containers during cleanup. [Pyxis container options](https://github.com/NVIDIA/pyxis/blob/v0.20.0/README.md) describe that lifecycle.

The PCS launchers run MPI ranks with the participant UID and set `PMIX_MCA_gds=hash` inside the rank after Enroot environment loading. This permits the pinned PMIx version `3.2.5a1` client to connect to the PCS PMIx version `5.0.6` server while retaining native authentication. Only private-container plugin mutation commands remap the participant to root. Do not export `PMIX_` variables around non-MPI steps: the installed Enroot hook would attempt to mount absent PMIx server directories.

```bash
export NCCL_ENROOT_IMAGE=/fsx/aim344/nccl-baseline.sqsh
```

## Round 0: healthy baseline and pre-job gate, 12 minutes

From the shared checkout on the login node, allocate the dedicated pair. Set `PARTITION` to the facilitator-provided partition name. The scripts require exactly 2 allocated nodes. Set `AIM344_SOCKET_IFNAME` only if the interface on the default IPv4 route is not the intended primary interface.

```bash
export PARTITION=aim344
export NCCL_ENROOT_IMAGE=/fsx/aim344/nccl-baseline.sqsh
salloc --partition="$PARTITION" --nodes=2 --ntasks-per-node=1 --exclusive --time=00:45:00
bash 0.baseline.sh
```

The sweep records GPU inventory, topology, instance identity, libfabric enumeration, NCCL debug output, and per-device EFA byte counters. It runs from 8 B through 2 GiB and checks both correctness columns at every size. Record your own in-place and out-of-place bus bandwidth at 2 GiB. Compare later rounds with that same allocation, image, rank count, and command. Bus bandwidth is NCCL's normalized collective metric; it is not a NIC wire-rate reading. [NVIDIA nccl-tests](https://github.com/NVIDIA/nccl-tests) documents the correctness flag and performance columns.

The participant sweep is independent of stock health check identifier `5`, which the facilitator runs separately with the unmodified pinned health suite.

The wrapper selects the primary IP interface and keeps MPI process setup on TCP so MPI's OFI transport does not confound the NCCL experiment. The correctness parser fails closed on a missing summary, a partial sweep, disabled checking, or any mismatches. It does not convert a single bandwidth observation into an automatic health verdict.

### Facilitator preparation for the PCS health gate

**VALIDATED on the g7e PCS rehearsal.** Install the pinned lightweight suite and the lab Prolog on each dedicated compute node. Run these commands from the shared repository checkout on that node through Systems Manager. Preserve existing scheduler configuration and add the Prolog through the PCS-supported configuration path; do not replace another workload's Prolog.

```bash
source examples/use-cases/gpu-cluster-failure-patterns/pins.env
sudo install -d /opt/aim344-healthcheck /opt/aim344
git archive "$HEALTHCHECK_COMMIT" validation/gpu-cluster-healthcheck \
  | sudo tar -x -C /opt/aim344-healthcheck
sudo install -m 0755 examples/use-cases/gpu-cluster-failure-patterns/prejob-prolog.sh /opt/aim344/prejob-prolog.sh
```

PCS exposes `Prolog` at cluster scope, not compute-node-group scope. The facilitator installs a root-owned shared dispatcher that exits for other partitions and invokes `/opt/aim344/prejob-prolog.sh` only for the lab partition. Preserve the existing cluster settings, allow batch requeue, and provide a healthy alternate node. See the Workshop Studio `FACILITATOR_RUNBOOK.md` for the installation and restoration sequence. The Prolog runs health check numbers 0 and 2; keep DCGM diagnostics outside this timed round. Stage and run lightweight check numbers 0, 2, and 3 on both hosts during pre-flight:

```bash
for check in 0 2 3; do
  sudo env PATH="/opt/amazon/efa/bin:$PATH" bash /opt/aim344-healthcheck/validation/gpu-cluster-healthcheck/gpu-healthcheck.sh --check "$check"
done
```

A node already in `DRAIN` cannot receive a job. To demonstrate the transition, mark a dedicated idle node with the synthetic unhealthy verdict using `1.inject-prejob.sh`, then submit a batch job that can initially select that node. The Prolog converts the verdict into the scheduler's drain and requeue behavior. For a deterministic demonstration, record the node weights and temporarily give the healthy alternate a higher weight than the marked node. Restore the original weights after recovery. Do not pin the batch job with `--nodelist`, since that would also pin its requeued attempt.

```bash
# On the dedicated compute node, from the repository root through Systems Manager:
cd examples/use-cases/gpu-cluster-failure-patterns
sudo bash 1.inject-prejob.sh ATTENDEE_USER

# On the login node, after releasing any salloc that occupies these nodes:
sbatch --partition="$PARTITION" --nodes=1 --requeue --job-name=aim344-prejob \
  --output='prejob-%j.out' --wrap='hostname'
squeue --partition="$PARTITION"
sinfo --partition="$PARTITION" --long
```

A failing Prolog drains the node and requeues a batch job. Slurm normally holds that job; `SchedulerParameters=nohold_on_prolog_fail` allows automatic retry. Otherwise the facilitator must release the held job after making the healthy alternate available. Verify the actual setting with `scontrol show config` before the demonstration. See the [Slurm Prolog failure behavior](https://slurm.schedmd.com/prolog_epilog.html).

Read `journalctl -t aim344-prolog` and `journalctl -u slurmd` on the failed node through Systems Manager. This output is in scheduler/system logs, not the batch job's output file. The synthetic verdict is explicitly logged as synthetic. A failed real health check is logged separately.

Recover on the marked compute node with `sudo bash 2.recover-prejob.sh`. After the lightweight checks pass, inspect `scontrol show node NODE` on the login node and resume only the node drained by this exercise using `scontrol update NodeName=NODE State=RESUME`. If a real hardware fault appears, preserve its evidence and replace the node. Restore the recorded scheduling weights after the exercise.

## Round 1: silent fallback and recovery, 15 minutes

Use the same dedicated allocation as the healthy sweep, or establish a fresh baseline after the health-gate exercise. Run the injection, read the log and counter deltas, then confirm recovery:

```bash
bash 3.inject-fallback.sh
bash 4.recover-fallback.sh
```

The injection moves `/opt/amazon/ofi-nccl/lib` to `/opt/amazon/ofi-nccl/lib.aim344-disabled` inside each private writable container. It restores the directory on normal exit or interruption. The explicit recovery script also restores it if necessary, then repeats the healthy sweep. The host installation and shared image remain intact.

On the validated Seoul p6-b300.48xlarge image, moving only `libnccl-net-ofi.so` was ineffective: a separate `libnccl-net.so` copy still loaded, and the sweep reached 830.45 GB/s in-place at 2 GiB with 0 mismatches. Moving the directory also removes the alternate network-plugin and tuner copies from their normal search path. The resulting sweep selected `NET/Socket`, completed correctly, and left all exposed EFA byte counters flat. Confirm all of those observations together; a low bandwidth value alone does not identify the transport.

Read counters on each allocated node. The script reports counter deltas in B and preserves each device name; it does not add overlapping counter families into a purported wire-byte total. Unavailable counters cause an explicit error rather than a false zero. Hostnames can be duplicated by a machine image, so shared snapshot filenames use EC2 instance IDs.

## Round 2: checkpoint writes and collective timeout, 13 minutes

**VALIDATED on PCS g7e and historical Seoul p6-b300 with private tmpfs; UNVALIDATED for FSx quotas.** Use an isolated checkpoint directory with a 32 MiB quota or private tmpfs, separate from NCCL shared memory. A facilitator must prepare it on every allocated host and give the attendee write access. For a private tmpfs mechanism test, run on each dedicated host through Systems Manager:

```bash
sudo install -d -m 0700 /run/aim344-checkpoints
sudo mount -t tmpfs -o size=32M tmpfs /run/aim344-checkpoints
sudo chown ATTENDEE_USER /run/aim344-checkpoints
printf '%s\n' 'AIM344 isolated checkpoint fixture' | sudo tee /run/aim344-checkpoints/.aim344-fixture
```

In the allocation, set `TORCH_IMAGE` to the built SquashFS image and `CHECKPOINT_DIR` to the prepared path. The same path string must resolve on both hosts. A participant-specific Lustre quota path can replace the private tmpfs after the quota is validated.

```bash
export TORCH_IMAGE=/fsx/aim344/aim344.sqsh
export CHECKPOINT_DIR=/run/aim344-checkpoints
bash 5.inject-storage.sh
bash 6.recover-storage.sh
```

The workload completes a real NCCL collective, saves a recoverable checkpoint, then exhausts the bounded checkpoint fixture on the writer rank. Its explicit checkpoint retry policy lasts up to 60 s; the collective deadline is 30 s. On Seoul p6-b300.48xlarge, the writer logged the storage failure before peers reported collective timeouts. The retry policy keeps the writer out of the collective. An immediate storage exception can instead abort a job directly, so a full filesystem by itself does not guarantee a watchdog signature. These durations are configured deadlines, not measured production timings.

The recovery script removes only the lab filler and partial write, keeps the last successful checkpoint, and resumes from its saved step. Diagnose the checkpoint error alongside the NCCL error and repeat `0.baseline.sh` after recovery to check the fabric separately. On Seoul p6-b300.48xlarge, both the write failure and collective timeout were observed, followed by successful recovery. Wait for every failed worker to exit before running the fabric check; the watchdog message can precede process teardown.

## Round 3: late DataLoader workers, 10 minutes, elastic

**VALIDATED non-reproduction on PCS g7e and historical Seoul p6-b300.48xlarge.** These commands are probes; they do not promise a hang. Run them after storage recovery with the PyTorch image staged:

```bash
bash 7.probe-dataloader-fork.sh
bash 8.use-dataloader-spawn.sh
```

The probe completes a collective before creating DataLoader workers, so NCCL initialization and registration actually precede the fork. The control uses `spawn` and disables EFA huge pages. The job deadline bounds an actual hang; no rank is artificially stopped and no synthetic timeout text is printed.

The fork probe completed on Seoul p6-b300.48xlarge with kernel version 7.0.0-1011-aws, Python version 3.10, PyTorch version 2.9.0+cu130, runtime NCCL version 2.30.4, libfabric version 2.4.0amzn3.0, and aws-ofi-nccl version 1.19.0. This is the tested stack for non-reproduction, not the first release that added mitigation. The [plugin's EFA environment guidance](https://github.com/aws/aws-ofi-nccl/blob/master/doc/efa-env-var.md) states that automatic fork-safety setting dates back at least to plugin version 1.2. Record the tested kernel, libfabric, plugin, PyTorch, and process-start method before teaching a version floor. Python's environment mapping may not reflect a C library's `setenv`, so the probe reads the C environment after the first collective.

Both Seoul p6-b300.48xlarge runs completed; teach that observed non-reproduction and the tested stack. It does not establish that arbitrary CUDA work in a forked child is safe. [PyTorch's distributed guidance](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) recommends `spawn` or `forkserver` when combining NCCL with multiworker DataLoaders. If earlier rounds run long, this round becomes a presenter-led probe.

## Cleanup and known edge cases

Run `bash 9.cleanup.sh` before leaving the allocation, then exit the allocation shell. Retain `results/` for comparison. After all jobs stop, the facilitator unmounts and removes only the private checkpoint tmpfs created for this exercise, clears any synthetic health marker, and restores the dedicated partition configuration. Follow the PCS architecture's teardown instructions for infrastructure created for the session.

Known issues that change diagnosis:

- **Separate stock check identifier `5` from the participant sweep.** Prior healthy p5.48xlarge nodes on PCS produced a false severity `RESET` in 2.70 s because the Enroot URI lacked the `#` registry separator and the image's `all_reduce_perf` binary was outside `PATH`. The facilitator records the stock gate's actual outcome and verifies that it reaches NCCL before interpreting its severity. The participant sweep uses the staged native-kernel image and absolute binary path. No unmerged upstream patch is installed.
- **Use `FI_EFA_IFACE`, not `FI_EFA_DEVICE_NAME`.** The latter is silently ignored by libfabric. Prior purported per-device tests repeatedly exercised device index 0, and even a bogus name passed. This lab observes every device's counters. It leaves `FI_EFA_IFACE` unset for the full-node sweep, or preserves the operator-selected `AIM344_EFA_IFACE` for a restricted EFA allocation. Do not ship or depend on the unmerged per-device health-suite fix.
- Prior p5.48xlarge attempts using `NCCL_NET=Socket` with `NCCL_IB_DISABLE=1` aborted with exit code 134 (dimensionless), including an assertion in `verify_active`. The `NCCL_NET_PLUGIN=none` and nonexistent-plugin-path attempts initialized, then failed with exit code 137 (dimensionless). These were not silent fallback signatures.
- MPI can report its own OFI failure before a NCCL watchdog. The checkpoint and DataLoader jobs use `torchrun` without MPI. The bandwidth sweep keeps MPI process setup on TCP while NCCL chooses its own data transport.
- `tc netem` and `iptables` do not sever EFA data traffic, which bypasses the kernel network stack. This lab does not use them as fabric injections.
- On EKS, include only the primary routed interface in MPI TCP setup. The Seoul validation initially failed when MPI also considered a link-local pod-identity interface. Do not set both the MPI TCP include and exclude variables.
- The minimal PyTorch image emits an optional NumPy-import warning. These fixtures use PyTorch tensors directly and completed without NumPy; retain the warning in validation logs.
- Pre-import images on a node with adequate ephemeral storage. The initial Seoul CPU launcher was evicted during image import. That launcher failure is not an NCCL failure.

Use the [failure-signature runbook](RUNBOOK.md) for the 5-minute wrap-up. The framing takes 5 minutes; the round durations above are session budgets, not measured workload runtimes.

## Resource-bounded PCS preparation and Spain outcome

The Spain rehearsal on 2026-09-10 used four GPUs, one EFA, 96 vCPUs and 384 GiB per node. PCS must enforce `ConstrainDevices=yes`, `ConstrainCores=yes`, `ConstrainRAMSpace=yes` and memory-aware scheduling with `SelectTypeParameters=CR_CPU_Memory`. A Slurm memory request alone did not create a finite memory cgroup in the original cluster configuration. `facilitator/allocation-evidence.sh` records GPU UUIDs, CPU affinity and ancestor memory limits; set `AIM_ENFORCE_STANDIN=1` to reject a step outside this budget.

The operator supplies `PARTITION`, `GPUS_PER_NODE`, `CPUS_PER_RANK` and `MEMORY_PER_NODE`, together with the staged `NCCL_ENROOT_IMAGE`, `TORCH_IMAGE` and `CHECKPOINT_DIR`. Spain used `gpu-g7`, four GPUs per node, 24 vCPUs per rank and `384G` of memory per node. Both the pre-job probe and subsequent workload allocation must request those resources:

```bash
sbatch --partition="$PARTITION" --nodes=1 --gres="gpu:$GPUS_PER_NODE" --ntasks-per-node="$GPUS_PER_NODE" --cpus-per-task="$CPUS_PER_RANK" --mem="$MEMORY_PER_NODE" --requeue --job-name=aim344-prejob \
  --output='prejob-%j.out' --wrap='hostname'
salloc --exclusive --partition="$PARTITION" --nodes=2 --gres="gpu:$GPUS_PER_NODE" --ntasks-per-node="$GPUS_PER_NODE" --cpus-per-task="$CPUS_PER_RANK" --mem="$MEMORY_PER_NODE" --time=00:45:00
```

The rehearsed participant sequence ran in a batch allocation with these same resource flags. The interactive shell handoff remains untested. Node-level launch steps use the allocation's CPU count; MPI steps use one rank per allocated GPU. Inspect each step's GPU UUIDs and finite memory limit before accepting its result.

Without Lustre, set `AIM344_NODE_LOCAL=1` and `AIM344_STAGE_DIR=/opt/aim344` for the preparation helpers. Stage the checkout and identical image files at the same path on both nodes, backed by existing NVMe. Complete image preparation as the attendee before locking the dispatcher directory to root ownership. Run `facilitator/prepare-compute.sh` and `facilitator/prepare-prolog.sh` on each node. Pass `--dispatcher /opt/aim344/.prolog/dispatch.sh` to `facilitator/pcs-prolog.py enable`; restoration reads the path and previous cluster settings from its state file. The default helpers retain their Lustre prerequisite when node-local mode is not selected.

On these G7 nodes, `AIM344_EFA_IFACE=rdmap83s0` and `FI_EFA_IFACE=rdmap83s0` selected one EFA. The first sequence with OFI's default SENDRECV protocol measured only 5.20 GB/s at 2 GiB, slower than Socket. Repeating the full sequence with `OFI_NCCL_PROTOCOL=RDMA` measured 20.27 GB/s at 2 GiB, versus 11.03 GB/s on Socket, with zero mismatches. This is a G7-specific measured setting, not a universal OFI default. The other EFA's byte counters remained flat. The private 32 MiB tmpfs reproduced checkpoint exhaustion and recovery; both DataLoader start methods completed. Full image digests, per-node counters, health-suite limitations and module times are in [VALIDATION.md](VALIDATION.md).

## G7 allocation and protocol

The primary allocation is two `g7.48xlarge` nodes; the secondary allocation is two `g7.24xlarge` nodes. Select the instance type in the infrastructure parameter. In the prepared submission shell, read the registered allocation before the participant baseline:

```bash
eval "$(python3 facilitator/allocation-env.py --partition "$PARTITION")"
printf 'GPUs/node=%s; vCPUs/rank=%s; memory request=%s (Slurm all-memory sentinel)\n' "$GPUS_PER_NODE" "$CPUS_PER_RANK" "$MEMORY_PER_NODE"
```

The MPI and Torch launchers source `instance-type.sh`, which reads the DMI product name and falls back to IMDSv2. For the detected `g7.*` family it sets `OFI_NCCL_PROTOCOL=RDMA`. Participants do not set the instance type or protocol. The Spain `g7.24xlarge-stand-in` baseline at a message size of 2 GiB, with four ranks per node and one selected EFA, measured 5.2 GB/s under the default SENDRECV protocol and 20.3 GB/s under RDMA. See [VALIDATION.md](VALIDATION.md) for the observed comparison and its scope.

Leave `AIM344_EFA_IFACE` unset for a whole-node allocation so the provider can use all attached EFA devices. An explicit selector is reserved for a bounded rehearsal allocation. Record each device's counters and the NCCL log; an environment setting alone is not transport evidence.

## Participant host inventory commands

Run these read-only inventory commands on each assigned compute host before the allocated baseline:

```bash
cat /sys/devices/virtual/dmi/id/product_name
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
nvidia-smi topo -m
/opt/amazon/efa/bin/fi_info -p efa
for check in 0 2 3; do
  sudo env PATH="/opt/amazon/efa/bin:$PATH" bash /opt/aim344-healthcheck/validation/gpu-cluster-healthcheck/gpu-healthcheck.sh --check "$check"
done
```
