# GPU cluster failure patterns on AWS

This is the runnable lab draft for re:Invent session AIM344, a 60-minute Builders' Session at level 300. Diagnose in this order: **storage, host, fabric, software**. Preserve evidence, isolate a suspect node, and replace it rather than attempting repairs during the session.

The production target is **2 p5.48xlarge nodes per attendee on AWS PCS with Slurm version 25.11**. Seoul validation uses **2 p6-b300.48xlarge nodes on EKS in ap-northeast-2**. Seoul establishes mechanisms only; its bandwidth and launch behavior are not production p5 baselines or PCS integration evidence.

## Validation status

These are observed results or explicit validation gaps, not predicted transcripts. All new GPU measurements below use the pinned image and 16 GPU ranks across 2 p6-b300.48xlarge nodes in Seoul. Each node contributes 8 GPU ranks.

| Round | Status and observed signature | Production boundary |
| --- | --- | --- |
| Round 0: healthy sweep | **VALIDATED, Seoul mechanism.** At 2 GiB, out-of-place bus bandwidth was 827.38 GB/s and in-place bus bandwidth was 829.38 GB/s. Both columns reported 0 mismatches, the out-of-bounds summary passed, and EFA byte counters increased on both nodes. | The prior p5.48xlarge measurement on PCS was in the **400 GB/s class** at 2 GiB with 0 mismatches. That is a reference class, not a guaranteed threshold. These new Slurm wrappers are **UNVALIDATED** on PCS. |
| Round 0: pre-job gate | **UNVALIDATED, PCS integration.** The supplied scripts implement a synthetic unhealthy verdict followed by the real lightweight health gate. EKS cannot establish Slurm drain, requeue, or log routing. | Requires a dedicated PCS test partition and facilitator access. |
| Round 1: plugin absent | **VALIDATED, Seoul mechanism.** Moving the entire OFI library directory produced `NET/Socket`, 15.53 GB/s in-place at 2 GiB, 0 mismatches, and 0 B change in every exposed EFA byte counter on both nodes. Restoring it recovered to 827.67 GB/s in-place at 2 GiB with 0 mismatches. | **UNVALIDATED** on p5.48xlarge and PCS. The missing directory includes tuner copies as well as network-plugin copies. |
| Round 2: checkpoint exhaustion | **VALIDATED, Seoul mechanism.** A private 32 MiB checkpoint tmpfs filled, the writer logged `No space left on device`, and peers reported the configured 30 s collective timeout. After failed workers exited, a sweep with the checkpoint path still full reached 829.30 GB/s in-place at 2 GiB with 0 mismatches. Checkpoint recovery passed. | FSx quotas, an actual FSx outage, and PCS execution are **UNVALIDATED**. This round used a different p6-b300.48xlarge pair from the initial sweep. |
| Round 3: late DataLoader workers | **VALIDATED, Seoul non-reproduction.** With EFA huge pages enabled and `fork`, the probe completed 4 batches with 0 mismatches. The plugin set the C environment variable `FI_EFA_FORK_SAFE` to the enabled value. The `spawn` control also completed 4 batches with 0 mismatches. | A historical fork hang and PCS execution are **UNVALIDATED**. Do not claim that every forked CUDA workload is safe. |

See [the validation record](VALIDATION.md) for observed excerpts and software versions. The prior p5.48xlarge sweep took 92.88 s. No new p5 performance measurement was made for this draft. No GPU producing incorrect arithmetic was observed; a passing health check and a correct result remain separate claims.

## Prerequisites and deployment

Use the repository's [PCS reference architecture](../../../architectures/aws-pcs/README.md) to provision the production environment before the session. Select Slurm version 25.11, p5.48xlarge compute nodes, a single Availability Zone, EFA-enabled placement, and a shared path for this checkout and logs. The participant node pair must be exclusive, powered on, and idle before the lab. Connect to the login and compute nodes through Systems Manager.

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

Build and export the PyTorch container on a build host with Docker and Enroot installed, then stage the resulting image on shared storage. The baseline uses the original digest directly. Retain the image checksum with the dry-run evidence.

```bash
cd examples/use-cases/gpu-cluster-failure-patterns
docker build -t aim344:torch2.9.0-cu130 .
enroot import -o aim344.sqsh dockerd://aim344:torch2.9.0-cu130
sha256sum aim344.sqsh
```

Pre-import the baseline image on the compute nodes before attendees arrive. Enroot uses `#` after the registry and `:sha256:` for a digest reference. Pyxis retains named containers across steps; these scripts name their private containers with the Slurm job ID and remove only those containers during cleanup. [Pyxis container options](https://github.com/NVIDIA/pyxis/blob/v0.20.0/README.md) describe that lifecycle.

```bash
source pins.env
enroot import -o nccl-baseline.sqsh "$NCCL_ENROOT_IMAGE"
```

## Round 0: healthy baseline and pre-job gate, 12 minutes

From the shared checkout on the login node, allocate the dedicated pair. Set `PARTITION` to the facilitator-provided partition name. The scripts require exactly 2 allocated nodes. Set `AIM344_SOCKET_IFNAME` only if the interface on the default IPv4 route is not the intended primary interface.

```bash
export PARTITION=aim344
salloc --partition="$PARTITION" --nodes=2 --ntasks-per-node=8 --exclusive --time=00:45:00
bash 0.baseline.sh
```

The sweep records GPU inventory, topology, instance identity, libfabric enumeration, NCCL debug output, and per-device EFA byte counters. It runs from 8 B through 2 GiB and checks both correctness columns at every size. Record your own in-place and out-of-place bus bandwidth at 2 GiB. Compare later rounds with that same allocation, image, rank count, and command. Bus bandwidth is NCCL's normalized collective metric; it is not a NIC wire-rate reading. [NVIDIA nccl-tests](https://github.com/NVIDIA/nccl-tests) documents the correctness flag and performance columns.

The corrected standalone invocation replacing stock health check number 5 is:

```bash
source pins.env
srun --nodes=2 --ntasks=16 --ntasks-per-node=8 --mpi=pmix --cpu-bind=none \
  --container-image="$NCCL_ENROOT_IMAGE" \
  /opt/nccl-tests/build/all_reduce_perf -b 8 -e 2G -f 2 -g 1 -c 1
```

For the lab, use `0.baseline.sh`; its wrapper also selects the primary IP interface and keeps MPI process setup on TCP so MPI's OFI transport does not confound the NCCL experiment. The correctness parser fails closed on a missing summary, a partial sweep, disabled checking, or any mismatches. It does not convert a single bandwidth observation into an automatic health verdict.

### Facilitator preparation for the PCS health gate

**UNVALIDATED on PCS.** Install the pinned lightweight suite and the lab Prolog on each dedicated compute node. Run these commands from the shared repository checkout on that node through Systems Manager. Preserve existing scheduler configuration and add the Prolog through the PCS-supported configuration path; do not replace another workload's Prolog.

```bash
source examples/use-cases/gpu-cluster-failure-patterns/pins.env
sudo install -d /opt/aim344-healthcheck /opt/aim344
git archive "$HEALTHCHECK_COMMIT" validation/gpu-cluster-healthcheck \
  | sudo tar -x -C /opt/aim344-healthcheck
sudo install -m 0755 examples/use-cases/gpu-cluster-failure-patterns/prejob-prolog.sh /opt/aim344/prejob-prolog.sh
```

The dedicated lab scheduler configuration must invoke `Prolog=/opt/aim344/prejob-prolog.sh`, allow batch requeue, and provide a healthy alternate node. The Prolog runs health check numbers 0 and 2; keep DCGM diagnostics outside this timed round. Stage and run lightweight check numbers 0, 2, and 3 on both hosts during pre-flight:

```bash
sudo bash /opt/aim344-healthcheck/validation/gpu-cluster-healthcheck/gpu-healthcheck.sh --check 0,2,3
```

A node already in `DRAIN` cannot receive a job. To demonstrate the transition, mark a dedicated idle node with the synthetic unhealthy verdict using `1.inject-prejob.sh`, then submit a batch job that can initially select that node. The Prolog converts the verdict into the scheduler's drain and requeue behavior. For a deterministic demonstration, the facilitator can temporarily restrict the dedicated lab partition to the marked node, then add the healthy alternate after the failed launch. Do not pin the batch job with `--nodelist`, since that would also pin its requeued attempt.

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

A failing Prolog drains the node and requeues a batch job. Slurm normally holds that job; `SchedulerParameters=nohold_on_prolog_fail` allows automatic retry. Otherwise the facilitator must release the held job after making the healthy alternate available. Verify the setting on Slurm version 25.11 during the PCS dry run. See the [Slurm Prolog failure behavior](https://slurm.schedmd.com/prolog_epilog.html).

Read `journalctl -t aim344-prolog` and `journalctl -u slurmd` on the failed node through Systems Manager. This output is in scheduler/system logs, not the batch job's output file. The synthetic verdict is explicitly logged as synthetic. A failed real health check is logged separately.

Recover on the marked compute node with `sudo bash 2.recover-prejob.sh`. After the lightweight checks pass, inspect `scontrol show node NODE` on the login node and resume only the node drained by this exercise using `scontrol update NodeName=NODE State=RESUME`. If a real hardware fault appears, preserve its evidence and replace the node. Restore the dedicated partition's original membership after the exercise.

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

**VALIDATED on Seoul p6-b300.48xlarge with a private tmpfs; UNVALIDATED for FSx quotas and PCS.** Use an isolated checkpoint directory with a 32 MiB quota or private tmpfs, separate from NCCL shared memory. A facilitator must prepare it on every allocated host and give the attendee write access. For a private tmpfs mechanism test, run on each dedicated host through Systems Manager:

```bash
sudo install -d -m 0755 /run/aim344-checkpoints
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

The workload completes a real NCCL collective, saves a recoverable checkpoint, then exhausts the bounded checkpoint fixture on the writer rank. Its explicit checkpoint retry policy lasts up to 60 s; the collective deadline is 30 s. On Seoul p6-b300.48xlarge, the writer logged the storage failure before peers reported collective timeouts. The retry policy keeps the writer out of the collective. An immediate storage exception can instead abort a job directly, so a full filesystem by itself does not guarantee a watchdog signature. These durations are configured deadlines, not measured p5.48xlarge timings.

The recovery script removes only the lab filler and partial write, keeps the last successful checkpoint, and resumes from its saved step. Diagnose the checkpoint error alongside the NCCL error and repeat `0.baseline.sh` after recovery to check the fabric separately. On Seoul p6-b300.48xlarge, both the write failure and collective timeout were observed, followed by successful recovery. Wait for every failed worker to exit before running the fabric check; the watchdog message can precede process teardown.

## Round 3: late DataLoader workers, 10 minutes, elastic

**VALIDATED non-reproduction on Seoul p6-b300.48xlarge; UNVALIDATED on PCS.** These commands are probes; they do not promise a hang. Run them after storage recovery with the PyTorch image staged:

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

- **Stock health check number 5 must not be used.** Prior healthy p5.48xlarge nodes on PCS produced a false severity `RESET` in 2.70 s because the Enroot URI lacked the `#` registry separator and the image's `all_reduce_perf` binary was outside `PATH`. This lab uses the corrected pinned invocation and the absolute binary path.
- **Use `FI_EFA_IFACE`, not `FI_EFA_DEVICE_NAME`.** The latter is silently ignored by libfabric. Prior purported per-device tests repeatedly exercised device index 0, and even a bogus name passed. This lab observes every device's counters and leaves `FI_EFA_IFACE` unset for the full-node sweep. Do not ship or depend on the unmerged per-device health-suite fix.
- Prior p5.48xlarge attempts using `NCCL_NET=Socket` with `NCCL_IB_DISABLE=1` aborted with exit code 134 (dimensionless), including an assertion in `verify_active`. The `NCCL_NET_PLUGIN=none` and nonexistent-plugin-path attempts initialized, then failed with exit code 137 (dimensionless). These were not silent fallback signatures.
- MPI can report its own OFI failure before a NCCL watchdog. The checkpoint and DataLoader jobs use `torchrun` without MPI. The bandwidth sweep keeps MPI process setup on TCP while NCCL chooses its own data transport.
- `tc netem` and `iptables` do not sever EFA data traffic, which bypasses the kernel network stack. This lab does not use them as fabric injections.
- On EKS, include only the primary routed interface in MPI TCP setup. The Seoul validation initially failed when MPI also considered a link-local pod-identity interface. Do not set both the MPI TCP include and exclude variables.
- The minimal PyTorch image emits an optional NumPy-import warning. These fixtures use PyTorch tensors directly and completed without NumPy; retain the warning in validation logs.
- Pre-import images on a node with adequate ephemeral storage. The initial Seoul CPU launcher was evicted during image import. That launcher failure is not an NCCL failure.

Use the [failure-signature runbook](RUNBOOK.md) for the 5-minute wrap-up. The framing takes 5 minutes; the round durations above are session budgets, not measured workload runtimes.
