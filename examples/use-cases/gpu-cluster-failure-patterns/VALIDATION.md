# Validation record

The historical sections dated 2026-09-06 record **two p6-b300.48xlarge nodes on EKS in ap-northeast-2, with eight GPU ranks per node and sixteen GPU ranks per run**. The later sections record the 2026-09-09 PCS rehearsal on two g7e.12xlarge nodes in us-west-2d. The Spain section dated 2026-09-10 records the later constrained G7 rehearsal. Each observation establishes only its stated hardware, software and launch path.

## Runtime and test boundaries

The immutable base image is `public.ecr.aws/hpc-cloud/nccl-tests@sha256:a5390d3f0eb50f3e5854085e2ae0fef90b9eb4f3b7486ed4cab0d47864809d4c`. The measured stack used driver version 595.91.07, kernel version 7.0.0-1011-aws, runtime NCCL version 2.30.4, aws-ofi-nccl version 1.19.0, and libfabric version 2.4.0amzn3.0. The Python fixtures added PyTorch version 2.9.0+cu130 on Python version 3.10. PyTorch's build-version helper reports NCCL version 2.27.7; the startup log confirms that the image's `LD_PRELOAD` loaded runtime NCCL version 2.30.4.

The sweep used the checked-in `sweep-rank.sh` through Open MPI version 4.1.7 with its TCP transport restricted to the primary interface. The Python fixtures used `torch-node.sh` through `torchrun`, without MPI. The checkpoint and DataLoader rounds used a different idle p6-b300.48xlarge pair after the first pair was released.

## Baseline, injection, and recovery

The table contains measurements on the original p6-b300.48xlarge pair. Every sweep checked message sizes from 8 B through 2 GiB and reported 0 mismatches in both columns.

| State | Out-of-place bus bandwidth at 2 GiB | In-place bus bandwidth at 2 GiB | EFA evidence |
| --- | --- | --- | --- |
| Healthy baseline | 827.38 GB/s | 829.38 GB/s | Positive byte deltas on both nodes |
| Only `libnccl-net-ofi.so` moved | 828.49 GB/s | 830.45 GB/s | OFI still selected; alternate network-plugin copy remained |
| Entire OFI library directory moved | 14.94 GB/s | 15.53 GB/s | `NET/Socket`; every exposed EFA byte counter changed by 0 B on both nodes |
| Directory restored | 829.44 GB/s | 827.67 GB/s | OFI selected again; positive byte deltas on both nodes |

The all-reduce command was `/opt/nccl-tests/build/all_reduce_perf -b 8 -e 2G -f 2 -g 1 -c 1`. The following rows are actual output at 2 GiB, in the table's order. NCCL reports size in B, count in elements, time in microseconds, algorithm and bus bandwidth in GB/s, and `#wrong` in mismatches; the root field is a rank index.

```text
# size count type redop root time algbw busbw #wrong time algbw busbw #wrong
  2147483648     536870912     float     sum      -1  4866.61  441.27  827.38       0  4854.84  442.34  829.38       0
  2147483648     536870912     float     sum      -1  4860.09  441.86  828.49       0  4848.59  442.91  830.45       0
  2147483648     536870912     float     sum      -1   269469    7.97   14.94       0   259353    8.28   15.53       0
  2147483648     536870912     float     sum      -1  4854.53  442.37  829.44       0  4864.91  441.42  827.67       0
```

Every one of these completed sweeps reported `# Out of bounds values : 0 OK`. Missing-plugin selection is visible in NCCL debug output; it is not an application error. The full directory removal includes network-plugin and tuner copies, so its bandwidth change is not a controlled measurement of transport alone.

## Storage fault and recovery

On the second p6-b300.48xlarge pair, the healthy checkpoint fixture completed with 0 mismatches. The private checkpoint tmpfs had a configured capacity of 32 MiB. The following are actual writer and peer observations:

```text
checkpoint injection: [Errno 28] No space left on device; filler_bytes=33546240 B
checkpoint write retry: [Errno 28] No space left on device
[rank12]:[E906 03:02:35.091834500 ProcessGroupNCCL.cpp:683] [Rank 12] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=3, OpType=ALLREDUCE, NumelIn=1048576, NumelOut=1048576, Timeout(ms)=30000) ran for 30002 milliseconds before timing out.
```

The checkpoint writer's retry deadline was configured as 60 s and the collective deadline as 30 s. After all failed workers exited, the checkpoint filesystem still had 0 B available, and a clean all-reduce sweep on the same second p6-b300.48xlarge pair reached 829.38 GB/s out-of-place and 829.30 GB/s in-place at 2 GiB with 0 mismatches. Its EFA counters increased. A preceding sweep overlapped worker teardown and is excluded from these results.

Removing only the filler and partial write preserved the last successful checkpoint. The recovery output was:

```text
Resuming from last successful checkpoint: next_step=0 (step index).
Storage workload completed: next_step=1 (step index); 0 mismatches.
```

Both node launchers exited successfully after recovery. This validates the bounded write-exhaustion mechanism and the fixture's retry behavior. FSx quota enforcement and an FSx service outage remain **UNVALIDATED**. The later g7e PCS execution is recorded below.

## DataLoader fork probe

Both configurations completed on the second p6-b300.48xlarge pair. The first used `fork` with EFA huge pages enabled; the second used `spawn` with EFA huge pages disabled. Each created 2 DataLoader workers per GPU rank after a completed collective.

```text
After first collective: C environment FI_EFA_FORK_SAFE=b'1'; DataLoader start_method=fork.
DataLoader workload completed: 4 batches; 0 mismatches.
After first collective: C environment FI_EFA_FORK_SAFE=b'1'; DataLoader start_method=spawn.
DataLoader workload completed: 4 batches; 0 mismatches.
```

The historical hang **did not reproduce** on this recorded stack. These runs do not establish the earliest mitigated version or the safety of arbitrary CUDA operations inside forked children. The minimal Python environment emitted an optional NumPy-import warning, which was retained in the raw logs.

## Historical qualification, 2026-09-06

At this historical Seoul checkpoint, PCS wrappers, pre-job drain/requeue, scheduler logs, Enroot/Pyxis execution and FSx quotas had not been tested. The later g7e PCS section records the completed scheduler and launch checks; FSx quota enforcement and the A100 production target remain **UNVALIDATED**. The historical fork hang and a hardware corruption case were not observed. The inherited p5.48xlarge reference remains the 400 GB/s class at 2 GiB, not the Seoul numbers above.

The local report prepared with this draft retains the exact Kubernetes commands, actual output, discarded attempts, instance IDs, and cleanup evidence. The initial CPU launcher was evicted for insufficient ephemeral storage; the first MPI attempt also failed before CUDA initialization because of interface selection. Neither failure is a GPU fault signature.

## Pinned health-suite matrix on PCS g7e.12xlarge, 2026-09-09

The matrix executes the unmodified suite at commit `0e2c2c5b47f434380fefb3ff6bb30938a9c3a606`. The health tree in the companion checkout matched that pin. No local fix branch was applied. All rows in this section refer to the two PCS `g7e.12xlarge` nodes, one EFA per node, RTX PRO 6000 Blackwell Server Edition GPUs, driver version `595.71.05` and host DCGM version `4.5.3`.

The driver records the exact command, start/end timestamps, elapsed seconds and exit code in each `execution.json`, with stdout/stderr in `command.log`. Example:

```bash
sudo bash /opt/aim344-healthcheck/validation/gpu-cluster-healthcheck/gpu-healthcheck.sh --check 4 --verbose --exclusive --timeout 9300 --results-dir /fsx/aim344/evidence/health-matrix/ip-10-3-132-156/check-4
```

The orchestrator timeout override permits the check's own `9000 s` L4 deadline. Each L4 run has an exclusive one-node Slurm allocation. The pre-event intensive wrapper subsequently uses an exclusive two-node allocation. Raw first-node matrix files are also copied to `/tmp/g7e-e2e/pcs/health-matrix-first/`.

| Check identifier | First g7e node elapsed | Second g7e node elapsed | Exit code, dimensionless | Observed verdict and scope |
|---|---|---|---|---|
| `0`, NVIDIA inventory | `0.503854 s` | `0.336385 s` | `0` on both nodes | PASS; actual GPUs visible |
| `1`, DCGM L2 | `112.720381 s` | `72.876923 s` | `0` on both nodes | PASS; raw software, memory and PCIe tests passed on both GPUs per node |
| `2`, EFA enumeration | `0.716237 s` | `0.676450 s` | `0` on both nodes | PASS with unlisted-profile warnings; expected-count comparison skipped |
| `3`, topology | `0.195121 s` | `0.206140 s` | `0` on both nodes | PASS with unlisted-profile warnings; expected GPU-count/NVLink checks skipped |
| `6`, EFA loopback | `4.293629 s` | `4.295637 s` | `0` on both nodes | PASS for the sole enumerated EFA domain; connectivity only |
| `4`, initial DCGM L4 | `3171.408278 s` | `3170.180226 s` | `0` on both nodes | Raw PASS; inherited exporter restarted, so telemetry isolation requires the separate repeat |

First-node check identifier `4` completed from `14:09:14.966531 UTC` to `15:02:06.374804 UTC`, taking `3171.408278 s`, exit code dimensionless value `0`, verdict `PASS`. Raw `dcgmi diag -r 4 -j` output reports `Pass` for both GPUs in the returned software, memory, diagnostic, nvbandwidth, pulse, memtest, memory-bandwidth, targeted-stress, targeted-power and PCIe tests. The targeted-power test reported approximately `600.0 W` average draw per GPU. These are the tests actually present in the raw output; there is no separately named EUD result to claim. This observed DCGM version supported the returned L4 tests on these Blackwell GPUs without the unmerged `fix/healthcheck-dcgm4-blackwell` branch.

### Unlisted profile and EFA-loopback threshold findings

At the pinned commit, `lib/common.sh:103-110` warns `No profile found for instance type: g7e.12xlarge` and assigns expected GPU and EFA counts of zero, `NVLINK_EXPECTED=false` and the EFA provider. `checks/2-efa-enumeration.sh:30-57` skips expected-count comparison when that count is zero. `checks/3-topology-check.sh:64` skips NVLink validation when `NVLINK_EXPECTED` is false, and its lines `147-154` skip expected GPU-count comparison when the profile count is zero. The observed PASS rows therefore have a coverage gap. `checks/5-nccl-allreduce.sh:65` reads `${EXPECTED_GPU_COUNT:-8}`; the string value `0` is not empty, so it becomes `-g 0`.

The prepared, unapplied candidate is `examples/use-cases/gpu-cluster-failure-patterns/upstream-candidates/g7e-instance-profile.patch` in the AIM344 worktree. It adds `g7e.12xlarge|2|1|false|efa`, supported by actual GPU inventory, EFA enumeration and PCIe topology. `git apply --check` passed. It also repairs the file's missing final newline. The candidate restores expected-count coverage and a nonzero GPU count for stock check identifier `5`; it does not by itself fix that check's container/binary/launcher defects or add a calibrated g7e bandwidth threshold. At the pinned source, `checks/5-nccl-allreduce.sh:30-43` returns a `300 GB/s` default for p4d and zero for unlisted types, including g7e and p4de. Lines `250-259` report a below-threshold warning. Those defaults are not measured acceptance thresholds for this rehearsal. No PR was opened and nothing was pushed.

The addendum's requested `EFA_MIN_BW` default is absent from the pinned implementation. The README describes a `20 Gbit/s` threshold, but `checks/6-efa-loopback.sh:29-70` runs `fi_pingpong`, lines `100-111` enumerate domains, lines `129-152` judge process exit status, and line `187` emits the connectivity PASS. There is no bandwidth parse or threshold comparison. Both one-EFA g7e nodes passed in approximately `4.30 s`; they were not measured against a `20 Gbit/s` criterion. The ignored `FI_EFA_DEVICE_NAME` selector remains a separate multi-device coverage defect, matching the subjects of `fix/healthcheck-slurm-check5-and-efa-iface` and `fix/healthcheck-efa-suite-defects`. Those branches were not used.

Participant module `010` retains check identifiers `0`, `2` and `3`; check identifier `6` stays in the facilitator gate. Its short g7e runtime would fit the session, but the four-EFA production binding path remains unvalidated. The CLI correction from `--check 0,2,3` to a loop was executed: the original command returned `Unknown check: 0,2,3`, while the separate identifiers passed. Source: `gpu-healthcheck.sh:144` and `gpu-healthcheck.sh:310`; evidence `pcs/health-cli-original*` and the matrix outputs.

### L4 telemetry-isolation defect and corrective repeat

After the first raw L4 PASS, host inspection found the inherited `dcgm-exporter` Docker container still running. The pinned `checks/4-dcgm-diag-l4.sh:116-123` kills a matching process and sleeps for `2 s`, but does not stop its container. Docker restarted the exporter immediately: first-node journal output shows container reattachment at `14:09:15.959907 UTC`, during the L4 startup, and the later process had a different PID. The raw PASS is preserved, but this initial L4 attempt is not a telemetry-isolated qualification. The same inherited exporter was present on the second node.

I explicitly stopped only that inherited container on both newly created g7e instances before the separate pre-event intensive sweep. `docker inspect` captured the restart policy, start time, PID, image, and stopped state. Evidence: `pcs/first-exporter-state*`, `pcs/stop-inherited-exporter-first*`, and `pcs/stop-inherited-exporter-second*`. The inherited exporter image was `nvcr.io/nvidia/k8s/dcgm-exporter:3.3.9-3.6.1-ubuntu22.04`. This is a suite preflight defect, not evidence of a Blackwell hardware failure. The runbook explicitly stops containerized telemetry before L4 so a restart policy cannot undo the gate. No upstream patch was applied. The separate pre-event repeat is required to establish isolation.

The raw L4 result does not contain an EUD test. Package inventory found DCGM core, CUDA and proprietary packages at version `4.5.3-1`, but no EUD package and neither documented `/usr/share/nvidia/diagnostic` path. NVIDIA documents EUD as a separately installed component with product and driver matching requirements: [EUD documentation](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-eud.html). Therefore the observed `dcgmi diag -r 4` pass does not establish EUD coverage on this RTX PRO GPU. EUD remains unvalidated; the report names the tests actually returned rather than inferring additional coverage from the numeric run level.

Read-only inspection of the named local branches found `fix/healthcheck-dcgm4-blackwell` at `ab4ecd8e`, whose actual change is a DCGM major-version guard and an L2 false-PASS fix on Blackwell. The observed host version `4.5.3` already exceeds that guard. The branch does not fix the Docker restart-policy defect discovered here; that defect is a separate upstream candidate. The old branch uses the pre-rename health-suite path, so a path-only diff against the current `validation/` tree showed a deletion rather than a meaningful implementation comparison; that discarded inspection is retained as `healthcheck-dcgm4-path-mismatch.diff`. None of the named branches was merged or applied.

### Prolog choice

Keep the lab Prolog. Its synthetic marker is root-owned and scoped to the job user (`prejob-prolog.sh:10-15`); the syslog tag is `aim344-prolog` (line `9`), and each job receives a private results directory under `/var/log/aim344-prolog` (lines `17-19`). The check invocations and their logs are preserved in lines `20-27`. The upstream wrapper has no synthetic marker, uses tag `gpu-healthcheck-prolog` (`slurm/prolog-gpu-healthcheck.sh:25-26`), replaces any supplied `RESULTS_DIR` with a predictable `/tmp` path (line `35`), and gates optional DCGM L2 at lines `37-62`. Layering the teaching marker and private evidence handling around that wrapper would retain more lab-specific code than the existing twenty-eight-line wrapper. Keep the measured L2 diagnostic in the facilitator gate; it took `72.88 s` to `112.72 s` per node. PCS drain, requeue and log routing are recorded separately after the scheduler rehearsal.

### Clean intensive sweep and stock NCCL gate results

The unmodified `slurm/sbatch-intensive.sh` completed in Slurm job identifier `19`, from `2026-09-09 15:13:57.838456 UTC` to `16:06:46.052315 UTC`, taking `3168.213859 s` for the complete two-node wrapper. Both nodes returned exit code dimensionless value `0` for L4 and for EFA loopback. All ten named L4 tests returned `Pass` on each GPU: software, memory, diagnostic, nvbandwidth, pulse, memtest, memory bandwidth, targeted stress, targeted power and PCIe. Host metadata identifies DCGM version `4.5.3` and driver version `595.71.05`. No EUD test appears in these results. Both inherited exporter containers were explicitly stopped before the run and remained stopped in midpoint and post-run inspections. Raw results: `/tmp/g7e-e2e/pcs/pre-event-intensive/job-19-intensive/`; complete timing and output: `pcs/pre-event-intensive-19.execution.json` and `pcs/pre-event-intensive-19.log`.

The wrapper's overall exit code was dimensionless value `1`, because its independent NCCL stage failed before launching the binary. Its aggregate displayed three nodes because it counted the two per-node result directories and the separate NCCL result directory, which repeats the first host's identity. There were two physical nodes. The aggregate `RESET` recommendation came from the NCCL launcher failure and does not establish a hardware fault.

Check identifier `5` was then executed separately in a two-node allocation, Slurm job identifier `20`, with `NCCL_CONTAINER=/fsx/aim344/nccl-baseline.sqsh` and the unmodified pinned script:

| g7e check identifier and mode | Start UTC | Elapsed | Exit code, dimensionless | Observed verdict |
|---|---|---|---|---|
| `5`, plain (`NCCL_ISOLATION_TESTS=0`) | `16:06:46.223953 UTC` | `4.629330 s` | `1` | FAIL / RESET; launcher exit code dimensionless value `2` on both nodes |
| `5`, isolation (`NCCL_ISOLATION_TESTS=1`) | `16:06:50.885544 UTC` | `13.902677 s` | `1` | Both isolation subtests and the full test failed to execute the binary; final FAIL / RESET |

The load-bearing lines were `Running NCCL all_reduce_perf across 2 nodes (0 GPUs/node)` and `error: execve(): all_reduce_perf: No such file or directory`. Plain mode and each isolation output show the same missing executable on both nodes. Source `checks/5-nccl-allreduce.sh:98-108`, lines `143-153`, and lines `197-207` invoke the bare binary; the staged binary is outside `PATH`. This matches the binary-path issue addressed by the unmerged `fix/healthcheck-slurm-check5-and-efa-iface` work. Using the local SquashFS path avoided the separate default registry-URI issue, but neither that script nor its profile was patched. No stock NCCL bandwidth, correctness result, or isolation threshold was measured. The label “NVLink-only” does not establish an NVLink test on g7e, whose physical GPU topology has no NVLink. Evidence: `pcs/check5-plain/`, `pcs/check5-isolation/`, and `pcs/check5-20.log`.

The participant baseline remains a separate executable experiment with the same native-kernel NCCL image and version, an absolute binary path, one MPI rank per detected GPU, `-g 1`, TCP for MPI bootstrap, correctness checking through `2 GiB`, and EFA snapshots. Its results cannot turn the failed stock gate into a PASS. An upstream binary/URI fix and the prepared g7e profile candidate remain necessary before the stock gate can establish its intended scope.

The unchanged upstream quarantine wrapper's healthy control ran in Slurm job identifier `21`, taking `68.132688 s` from `16:07:04.923629 UTC` to `16:08:13.056315 UTC`. Its four lightweight checks passed and it emitted `RETURN_TO_SERVICE`. This establishes the healthy branch only. Its generated action used physical hostname `ip-10-3-132-156`; the actual PCS scheduler name is `g7e-1`, so the generated command must not be copied blindly. Real-hardware failure, replacement and reboot branches remain unvalidated. Evidence: `pcs/quarantine-control-21/` and `pcs/quarantine-healthy-control/`.

### Participant preflight correction and staged compute verification

Slurm job identifier `22` executed `facilitator/prepare-compute.sh` on both nodes in `0.550014 s`, the allocated inventory in `0.683013 s`, the baseline Pyxis pre-import in `4.507691 s`, and the Torch Pyxis pre-import in `16.335933 s`. Each command returned exit code dimensionless value `0`. The real exclusive batch allocation exposed two GPUs and twenty-four available CPU cores per node. Evidence: `pcs/preimport-22/` and `pcs/preimport-22.log`.

The participant module `010` first exposed a preparation-command defect: the direct `/opt/amazon/efa/bin/fi_info -p efa` invocation succeeded, but plain `sudo bash` reset `PATH`, and the suite's check identifier `2` warned `fi_info not found -- libfabric may not be installed` while still reporting PASS. The page and companion README now invoke `sudo env PATH="/opt/amazon/efa/bin:$PATH" bash ...`. The lab Prolog also supplies the installed EFA directory on `PATH`, so the scheduler's minimal environment cannot silently omit that provider probe. This changes lab-owned invocation code only; the pinned upstream suite remains unchanged.

Both corrected participant preflights passed at `16:12:58 UTC`, with no missing-`fi_info` warning. The first and second SSM controller invocation wall times were `7.270626 s` and `2.217044 s`, including transport and polling; the SSM-reported host interval was `1 s` on each node. Both hosts reported RTX PRO 6000 Blackwell Server Edition GPUs, driver version `595.71.05`, `97887 MiB` GPU memory, PCIe `PIX` topology and EFA domain `rdmap49s0-rdm`. Check identifiers `0`, `2` and `3` passed with the already-recorded unlisted-profile limitation. Evidence: `pcs/participant-010-path-fixed-first*` and `pcs/participant-010-path-fixed-second*`; the original attempts are retained separately. The installed twenty-eight-line Prolog's SHA-256 is `c6bb1af22bf9221793d363565b8e67ad7d1171d4c9f5f4ba9b1844c566e2a9db` on both nodes.

### PCS Prolog drain, requeue and alternate-node execution

`facilitator/pcs-prolog.py enable` saved the original empty custom-setting list to `/tmp/aim344-prolog-state.json` and installed the guarded dispatcher plus `JobRequeue=1`. PCS returned ACTIVE after `221.900144 s`, from `16:10:03.826670 UTC` to `16:13:45.726808 UTC`. Effective Slurm configuration showed `Prolog[0]=/fsx/aim344/.prolog/dispatch.sh`, `PrologFlags=Alloc,Contain`, `JobRequeue=1` and `SchedulerParameters=requeue_on_resume_failure`. No automatic no-hold retry setting was added.

The participant round took `258.492687 s` of controller wall time, including inspection and the manual release. The marker was armed for user `ubuntu` on `g7e-1` at `16:15:29 UTC`; its owner UID was dimensionless value `0` and mode was octal `0600`. The alternate node's temporary scheduling weight was dimensionless value `100`, with the first node retaining weight value `1`. The participant submitted the README's one-node requeue-enabled `hostname` batch command without a node-list constraint.

Slurm job identifier `23` initially selected `g7e-1`. At `16:16:13 UTC`, `journalctl -t aim344-prolog` recorded `Synthetic unhealthy verdict: job=23 node=ip-10-3-132-156.` The daemon recorded Prolog exit code dimensionless value `1`, and the scheduler drained the node with reason `Prolog error`. The job became `PENDING`, reason `launch_failed_requeued_held`, with one restart and `ReqNodeList=(null)`. This is observed scheduler behavior, not a simulated transcript.

After confirming `g7e-2` was idle and healthy, the facilitator released only job identifier `23`. It started on `g7e-2` at `16:17:35 UTC`, completed at `16:17:37 UTC`, and printed `ip-10-3-133-16`. Final scheduler state was `COMPLETED`, `ExitCode=0:0`, with one restart. The alternate node's journal recorded the real inventory and EFA checks passing, followed by `Health gate passed: job=23`. Its evidence directory `/var/log/aim344-prolog/job-23.tyhWRn` was root-owned with mode `0700`.

`2.recover-prejob.sh` removed the first node's marker and passed its lightweight checks at `16:18:12 UTC`. Only that node was resumed, and the alternate's scheduling weight was restored to value `1`. The resumed node briefly displayed the normal registration-pending `idle*` state; later scheduling checks establish readiness before further work. Exact command/output records are `pcs/prolog-submit.*`, `pcs/prolog-failed-state.*`, `pcs/prolog-failed-journal-first*`, `pcs/prolog-release.*`, `pcs/prolog-alternate-result.*`, `pcs/prolog-alternate-journal*`, `pcs/prolog-recover-first*`, `pcs/prolog-resume-and-weights.*`, and `pcs/prolog-round.execution.json`.

### Reserved quarantine workflow control

The facilitator's complete reservation sequence was exercised on the recovered first node with no synthetic marker. A root-only Slurm reservation, `aim344-quarantine-20260909-control`, reserved that node's twenty-four CPU cores from `16:21:33 UTC` for a configured duration of four hours. The node was resumed only after the reservation became active. Root job identifier `24` then executed the unchanged quarantine wrapper, passed the four lightweight checks, and emitted `RETURN_TO_SERVICE` at `16:22:46 UTC`. The node was drained again for review. The controller sequence took `76.604700 s`, with exit code dimensionless value `0`. After reviewing the healthy result, only this reservation was deleted and only the recovered node was resumed.

The first attempt failed before any scheduler change because `sudo` could not find `scontrol` on its reset PATH. The runbook now uses the PCS Slurm binary's absolute path for privileged scheduler commands and carries the EFA/Slurm PATH explicitly into privileged batch submissions. The failed attempt is preserved in `pcs/quarantine-reservation-control.*`; the corrected execution and raw diagnostics are in `pcs/quarantine-reservation-control-path-fixed.*` and `pcs/quarantine-reservation-control/`. `pcs/quarantine-reservation-release.*` records cleanup. This validates operator isolation, the healthy diagnostic branch and return-to-service mechanics; it does not establish the real-fault reboot or replacement branches.

### PCS MPI launch compatibility

The first participant baseline in Slurm job identifier `25` failed before NCCL initialization. The pinned image has Open MPI version `4.1.7` with embedded PMIx version `3.2.5a1`; the PCS host has PMIx version `5.0.6` and exposes the `pmix_v5` Slurm plugin. Both Enroot PMI hooks were already installed. The first error was `Framework: gds` / `Component: shmem2`; the following generic “not built with SLURM's PMI support” message did not identify the actual defect. EFA counter changes were `0 B` on both nodes.

A controlled four-rank CPU MPI probe separated data-store selection from user identity. Job identifier `27` used `PMIX_MCA_gds=hash` with container root remapping and still failed MPI initialization. Job identifier `28` used `hash` without root remapping, retained `PMIX_MCA_psec=native`, and completed a real `MPI_Allreduce`: all four ranks reported `size=4` and `sum=6`, both dimensionless values. The controller command took `44.266194 s`, including batch submission and polling. No PMIx security mode or Slurm daemon configuration was changed. The failed jobs left their MPI step daemons stuck in completion; only the identified daemons for job identifiers `25` and `27` were killed on the two rehearsal nodes. The scheduler then released the nodes. Probe job identifier `26` had a C-source quoting error and is not MPI validation evidence.

A follow-up baseline, job identifier `29`, exposed a second invocation issue: exporting a `PMIX_` variable around `--mpi=none` steps makes Enroot version `3.5.0`'s hook select its PMIx branch despite the absent server paths. It failed with `enroot-mount: failed to mount` for `/var/spool/slurmd/pmix.29.3`. The final launchers set `PMIX_MCA_gds=hash` inside only the MPI rank, after container environment loading, and run MPI ranks without root remapping. AIM344's plugin mutation commands retain remapping inside their private writable container. AIM347 also applies MPI TCP bootstrap settings inside its rank wrapper so the image's environment cannot override them. The configured NCCL transport remains the experimental variable.

Evidence: `/tmp/g7e-e2e/pcs/participant-25/`, `pcs/pmix-host-first.log`, `pcs/pmix-host-second.log`, `pcs/pmix-probe-27.log`, `pcs/pmix-probe-28.log`, `pmix-probe-uid-submit.command.json`, and the individual cleanup SSM records. NVIDIA documents the extra PMI hooks and the `hash` compatibility setting in [Pyxis setup](https://github.com/NVIDIA/pyxis/wiki/Setup). The observed native-authentication success supplies the basis for preserving that configuration.

### AIM344 participant sequence on PCS g7e

Slurm job identifier `30` ran the corrected participant commands in order on two `g7e.12xlarge` nodes in `us-west-2d`, four GPU ranks total, twenty-four CPU cores per node, PCS Slurm version `25.05.7`, AMI `ami-0aa5c69145b3bda5e`, and driver version `595.71.05`. The baseline image digest is `sha256:14404e0c24759c7b2a1ec687475d6b67e4a7bbcb98918bfcc8f70f2a1da33287`; the Torch image digest is `sha256:956d1e49609f05738275acaee1010b171a4ee20aaf51bbfb452c36afd54b850d`. NCCL startup logs identify loaded runtime version `2.30.4+cuda13.0`, including the Torch exercises. Torch reports version `2.9.0+cu130` and build-time NCCL version `2.27.7`; that build-time value is not the loaded runtime. The nodes had one EFA device each, PCIe connectivity without NVLink, and no common cluster placement group. Shared FSx was across AZs; the checkpoint fixture was private node-local tmpfs.

The exact batch script is `/tmp/g7e-e2e/pcs/participant-aim344.sbatch`. Module timing records, exact argv and raw output are in `/tmp/g7e-e2e/pcs/participant-30/`; per-sweep snapshots and NCCL logs are in `pcs/aim344-results-30/`. The batch controller returned exit code dimensionless value `0` in `236.512604 s`, from `16:37:41.693157 UTC` to `16:41:38.205753 UTC`. The table measures each command on the batch host and excludes queueing. The health preflight and separate Prolog round are recorded above.

| Module identifier and exact command | Start UTC | Command wall time | Exit code, dimensionless | Completion check and observation on g7e |
| --- | --- | --- | --- | --- |
| `020-baseline`: `bash 0.baseline.sh` | `16:37:45.908760 UTC` | `24.971925 s` | `0` | PASS: at 2 GiB, 33.49 GB/s out of place and 33.45 GB/s in place; zero mismatches; OFI and positive EFA byte deltas |
| `030-fallback`: `bash 3.inject-fallback.sh` | `16:38:10.915335 UTC` | `33.410815 s` | `0` | PASS: NET/Socket; at 2 GiB, 13.51 GB/s out of place and 12.98 GB/s in place; zero mismatches and zero EFA byte growth |
| `030-recovery`: `bash 4.recover-fallback.sh` | `16:38:44.361444 UTC` | `21.380822 s` | `0` | PASS: OFI restored; at 2 GiB, 33.45 GB/s out of place and 33.34 GB/s in place; zero mismatches and positive EFA byte deltas |
| `040-storage-injection`: `bash 5.inject-storage.sh` | `16:39:05.776097 UTC` | `109.045003 s` | `1` | PASS for the fault exercise: ENOSPC after 33546240 B of filler; peer watchdogs at 30053 ms, 30054 ms and 30070 ms; expected failed workload |
| `040-storage-recovery`: `bash 6.recover-storage.sh` | `16:40:54.857811 UTC` | `8.913692 s` | `0` | PASS: resumed saved step index 0, completed next step index 1, zero mismatches |
| `040-fabric-recheck`: `bash 0.baseline.sh` | `16:41:03.804027 UTC` | `13.759041 s` | `0` | PASS: after checkpoint recovery, at 2 GiB, 33.37 GB/s out of place and 33.33 GB/s in place; zero mismatches and positive EFA traffic |
| `050-fork`: `bash 7.probe-dataloader-fork.sh` | `16:41:17.593965 UTC` | `4.182129 s` | `0` | PASS as a non-reproduction: four batches, zero mismatches, C-level FI_EFA_FORK_SAFE enabled |
| `050-spawn`: `bash 8.use-dataloader-spawn.sh` | `16:41:21.808561 UTC` | `5.180565 s` | `0` | PASS: four batches, zero mismatches |
| `050-cleanup`: `bash 9.cleanup.sh` | `16:41:27.021261 UTC` | `2.496284 s` | `0` | PASS: removed only this job's named Enroot containers from node-local storage; shared results retained |

EFA values below are measured changes of `/sys/class/infiniband/rdmap49s0/ports/1/hw_counters/rdma_write_bytes`, with node identity taken from each snapshot. NCCL's `NET/OFI` and `NET/Libfabric` records corroborate the positive changes. The fallback logs select `NET/Socket`; every exposed byte counter remained unchanged on both nodes, rather than being absent. All four sweeps reported `Out of bounds values : 0 OK`.

| g7e phase | First node RDMA write change | Second node RDMA write change |
| --- | --- | --- |
| `020-baseline` | `290483780016 B` | `290484115392 B` |
| `030-fallback` | `0 B` | `0 B` |
| `030-recovery` | `290484453424 B` | `290484050816 B` |
| `040-fabric-recheck` | `290484311600 B` | `290484055696 B` |

The checkpoint writer filled only the `32 MiB` fixture, logged `ENOSPC`, then retried according to the explicit `60 s` policy. The first peer timeout appeared at `16:39:53 UTC`, before the writer's final exception at `16:40:23 UTC`. Watchdog teardown extended the command to `109.045003 s`; the configured `30 s` collective deadline is not the module wall time. Recovery preserved and resumed `last.json`. The fabric recheck in this PCS run happened after recovery, as the participant page specifies; the historical Seoul check while storage remained full is separate evidence.

The late-fork probe and spawn control both completed with zero mismatches. This establishes non-reproduction on the pinned stack, not a universal safety claim for forked CUDA processes. No real GPU arithmetic fault, actual FSx outage, production A100 result, or real-hardware quarantine replacement was demonstrated.

### PCS Prolog restoration after AIM344

With no jobs left in the g7e queue, `facilitator/pcs-prolog.py restore --cluster pcs_jdb4dviivh --region us-west-2 --state-file /tmp/aim344-prolog-state.json` restored the original empty custom-setting list. PCS returned ACTIVE at `2026-09-09 16:45:09 UTC`; the command took `199.414638 s` and returned exit code dimensionless value `0`. Effective Slurm configuration no longer contained `Prolog[0]`. Both g7e nodes were idle before AIM347 began. The root-owned dispatcher and suite remain staged for the next rehearsal, but the dispatcher is no longer configured as the cluster Prolog. Evidence: `/tmp/g7e-e2e/pcs-prolog-restore.*`.

## Spain g7.48xlarge as a g7.24xlarge resource stand-in, 2026-09-10

The physical allocation was two `g7.48xlarge` nodes in `eu-south-2a`, each with eight RTX PRO 4500 Blackwell Server Edition GPUs reporting `32623 MiB/GPU`, compute capability version `12.0` (SM120), `192 vCPUs`, `768 GiB RAM` and two EFA interfaces. Every result in the constrained PCS tables used `--gres=gpu:4 --ntasks-per-node=4 --cpus-per-task=24 --mem=384G` on each node, with `FI_EFA_IFACE=rdmap83s0`. Job logs contain the requested resources, actual GPU UUIDs, `96` available logical processors, CPU affinity indices `96-191` and a cgroup memory ceiling of `412316860416 B` (`384 GiB`). The GPUs were device indices `0-3` on both nodes. The second EFA interface, `rdmap176s0`, had zero byte-counter change. This enforces the resource budget; CPU/GPU NUMA placement, cache state and NVMe bandwidth still differ from a literal `g7.24xlarge` instance.

PCS cluster `pcs_lefqlz8o8e` used queue `gpu-g7`, Slurm version `25.05.9`, AMI identifier `ami-02165285ade1d0209`, Ubuntu version `24.04.4`, kernel version `6.17.0-1020-aws` and driver version `595.91.07`. The compute hosts were `gpu-g7-1` (`10.8.30.139`, instance identifier `i-0bbcee65f305fc7c6`) and `gpu-g7-2` (`10.8.17.100`, instance identifier `i-0af87d1628a5f761b`). PCS was configured with `ConstrainDevices=yes`, `ConstrainCores=yes`, `ConstrainRAMSpace=yes` and `SelectTypeParameters=CR_CPU_Memory`; GRES requests alone had not enforced memory. Existing NVMe was bind-mounted under the lab's `/opt` path and staged identically on both nodes. No FSx or EFS filesystem was created.

### Health, scheduler and image qualification

The unmodified health-suite commit was `0e2c2c5b47f434380fefb3ff6bb30938a9c3a606`. Slurm job identifier `2` ran the required matrix inside the constrained allocation. An earlier attempt, job identifier `1`, exposed an unlimited memory cgroup and was canceled after `51 s`; its diagnostic results are excluded. The corrected matrix allocation lasted `184 s`.

| Health check | First node wall time | Second node wall time | Result |
|---|---|---|---|
| Check identifier `0` | 0.489578 s | 0.485661 s | Exit code 0 on both nodes |
| Check identifier `1` | 167.994481 s | 172.573366 s | Exit code 0 on both nodes |
| Check identifier `2` | 0.673604 s | 0.653570 s | Exit code 0 on both nodes |
| Check identifier `3` | 0.380483 s | 0.377535 s | Exit code 0 on both nodes |
| Check identifier `6` | 4.822204 s | 4.500949 s | Exit code 0 on both nodes |

DCGM version `4.6.1` returned software, memory and PCIe PASS results for all four visible GPUs per node. The native host engine was stopped, then started inside the constrained step, and stopped again before monitoring deployment. The suite still has no `g7.48xlarge` or `g7.24xlarge` profile, so expected-count comparisons in checks with identifiers `2` and `3` were skipped despite their successful CLI status. Check identifier `6` exercised only `rdmap83s0-rdm` and established connectivity. Optional check identifier `4` was not run; its earlier duration was about `53 minutes/node`. Stock check identifier `5` was not rerun. The unapplied candidate `upstream-candidates/g7e-instance-profile.patch` now also contains `g7.24xlarge|4|1|false|efa` and `g7.48xlarge|8|2|false|efa`; `git apply --check` passed against the pinned tree. No upstream patch was applied.

The root-owned dispatcher `/opt/aim344/.prolog/dispatch.sh` was enabled through the PCS cluster setting. Job identifier `3` first failed the synthetic gate on node `gpu-g7-1`, drained that node and requeued with a hold. Releasing the held job allowed it to complete on `gpu-g7-2`. Recovery removed the marker, resumed the first node and restored equal scheduler weights. The arm-through-recovery interval was `168.622 s`. The pre-job host gate inventories all physical devices; its lightweight inventory is separate from the constrained GPU workloads. Enabling the setting took `136.022450 s`; restoring the saved settings took `202.055702 s`. Restoration used the saved dispatcher path and preserved memory accounting and cgroup enforcement. A later redundant start-time adjustment was rejected because the job had already completed; it had no effect.

The Spain ECR repository is `159553542841.dkr.ecr.eu-south-2.amazonaws.com/aim-spain-rehearsal-20260910`. The baseline image digest is `sha256:14404e0c24759c7b2a1ec687475d6b67e4a7bbcb98918bfcc8f70f2a1da33287`; the Torch fixture image digest is `sha256:956d1e49609f05738275acaee1010b171a4ee20aaf51bbfb452c36afd54b850d`. Imported-image SHA-256 values are `1eb22e960e290501ba6f9e206282531db4bed07c5f21d5a698a6801cd87ad3d2` and `d766c475e7e12d29e141df852d451a3831f500b5c6ec17d2972a046e33905cd0`, respectively, identical on both nodes. The executed images contain native SM120 kernels, NCCL version `2.30.4+cuda13.0`, aws-ofi-nccl version `1.19.0`, EFA installer version `1.48.0` and libfabric version `2.4.0`; Torch is version `2.9.0+cu130`. MPI uses `PMIX_MCA_gds=hash` inside the rank wrapper.

### Participant sequence and transport correction

The first complete sequence, job identifier `4`, used OFI's default SENDRECV protocol on this unlisted G7 platform. Its baseline was `5.20 / 5.20 GB/s`, Socket fallback was `11.87 / 11.87 GB/s`, and recovery was `5.11 / 5.07 GB/s`, all at `2 GiB` with zero mismatches. This failed the expected direction of the bandwidth comparison. The raw result is retained. OFI selected EFA, so the slow baseline was not a Socket fallback.

Job identifier `5` repeated the same images, allocation and complete sequence with the G7-specific environment setting `OFI_NCCL_PROTOCOL=RDMA`. Native RDMA capability checks remained enabled. The first and second bandwidth values below are out-of-place and in-place bus bandwidth. Module times include launch, workload and collection. The module sum was `267.529678 s`.

| Module | Wall time | Observed output | Page completion result |
|---|---|---|---|
| `020-baseline` | 31.571793 s | 20.27 / 20.27 GB/s at 2 GiB; 0 mismatches | PASS |
| `030-fallback` | 52.722570 s | 11.03 / 11.02 GB/s at 2 GiB; Socket; 0 EFA bytes; 0 mismatches | PASS |
| `030-recovery` | 24.130863 s | 20.46 / 20.47 GB/s at 2 GiB; OFI restored; 0 mismatches | PASS |
| `040-storage-injection` | 100.527138 s | ENOSPC after 33546240 B of filler; peer collective timeout | PASS: expected fault, exit code 1 |
| `040-storage-recovery` | 17.867311 s | Saved step index 0 resumed to step index 1; 0 mismatches | PASS |
| `040-fabric-recheck` | 24.252407 s | 20.82 / 20.83 GB/s at 2 GiB; 0 mismatches | PASS |
| `050-fork` | 6.575038 s | 4 batches; 0 mismatches; FI_EFA_FORK_SAFE enabled | PASS: non-reproduction recorded |
| `050-spawn` | 7.614720 s | 4 batches; 0 mismatches | PASS |
| `050-cleanup` | 2.267839 s | Private containers and filler removed; last.json retained | PASS |

The corrected baseline increased RDMA-write counters by `197732354992 B` on the first node and `197732094720 B` on the second node. Recovery increased them by `197731752480 B` and `197732051248 B`; the final fabric check increased them by `197731625520 B` and `197732328304 B`. The Socket round changed every exposed EFA byte counter by `0 B`. The unused EFA stayed at zero change throughout both complete sequences. NCCL ring logs reported `GDR 0`; positive EFA traffic does not prove absence of host staging. Set the protocol for the validated G7 environment only; it is not a universal platform default.

The corrected sequence's module records establish completion and exit status for every command, including the expected checkpoint fault. The controller purged job identifier `5` before its final `scontrol show job` state was captured. Its exact final controller state is therefore unavailable; module completion, the node journal and the empty queue are the retained evidence.

The `/run/aim344-checkpoints` fixture is a private `32 MiB` tmpfs. Its ENOSPC/retry/collective-timeout mechanism works without a shared filesystem. FSx quota enforcement or a service outage was not exercised. Both fork and spawn completed, so the historical fork hang did not reproduce. The participant page already accepts this observed outcome.

Raw commands, allocations, UUIDs, health JSON, journals, timing records and complete logs are under `/tmp/spain-rehearsal/phase2/aim344/`; compact derivations are `participant-summary-344.json`, `health-summary.json` and `aim344-efa-rounds.json`. The numbered participant scripts ran in a batch harness, not an interactive participant shell. Literal g7.24xlarge hardware, production A100 execution, optional L4/EUD, stock NCCL health-gate repair, filesystem quotas, participant-role handoff and reboot/replacement-node persistence remain UNVALIDATED.

## Dual-G7 qualification, 2026-09-10

The primary allocation used two exclusive g7.48xlarge nodes, eight GPUs, both EFA devices and all 192 vCPUs per node. Slurm exposed 747110 MiB of schedulable memory per node, requested with its all-memory sentinel. The secondary g7.24xlarge-stand-in repeat used four accessible GPUs, one selected EFA, at most 96 vCPUs and 384 GiB per node on the same physical hosts. Hardware model and DMI labels remain g7.48xlarge in raw stand-in files. Literal g7.24xlarge execution remains unvalidated.

Both repeats unset the participant protocol variable. Every MPI and Torch launcher detected G7 through DMI and printed `OFI_NCCL_PROTOCOL=RDMA`. The full run was Slurm job identifier `16`; the stand-in repeat was job identifier `17`. The source health suite and engine libraries were not patched. The newly added allocation-environment helper read the registered full shape as eight GPUs and 24 vCPUs per rank without an instance-type input.

| Measured allocation | Sweep at 2 GiB | Out-of-place bus bandwidth | In-place bus bandwidth | Correctness |
|---|---|---|---|---|
| g7.48xlarge | baseline-20260910T105059Z | 39.73 GB/s | 39.73 GB/s | 0 / 0 mismatches |
| g7.48xlarge | baseline-20260910T105453Z | 39.83 GB/s | 39.77 GB/s | 0 / 0 mismatches |
| g7.48xlarge | fallback-20260910T105139Z | 16.89 GB/s | 16.73 GB/s | 0 / 0 mismatches |
| g7.48xlarge | recovery-20260910T105225Z | 39.80 GB/s | 39.69 GB/s | 0 / 0 mismatches |
| g7.24xlarge-stand-in | baseline-20260910T105638Z | 20.48 GB/s | 20.57 GB/s | 0 / 0 mismatches |
| g7.24xlarge-stand-in | baseline-20260910T110020Z | 19.97 GB/s | 19.98 GB/s | 0 / 0 mismatches |
| g7.24xlarge-stand-in | fallback-20260910T105711Z | 11.91 GB/s | 11.99 GB/s | 0 / 0 mismatches |
| g7.24xlarge-stand-in | recovery-20260910T105757Z | 20.91 GB/s | 20.96 GB/s | 0 / 0 mismatches |

The earlier g7.24xlarge-stand-in SENDRECV attempt measured 5.20 GB/s, below its Socket control of 11.87 GB/s. Its explicit-RDMA baseline reached 20.27 GB/s. The automatic-gate repeats above preserve that expected transport direction on both allocation shapes.

| Measured allocation | Module | Command wall time | Exit status, dimensionless |
|---|---|---|---|
| g7.24xlarge-stand-in | 020-baseline | 30.305663 s | 0 |
| g7.24xlarge-stand-in | 030-fallback | 51.104230 s | 0 |
| g7.24xlarge-stand-in | 030-recovery | 23.739729 s | 0 |
| g7.24xlarge-stand-in | 040-storage-injection | 100.871467 s | 1 |
| g7.24xlarge-stand-in | 040-storage-recovery | 18.235402 s | 0 |
| g7.24xlarge-stand-in | 040-fabric-recheck | 24.861210 s | 0 |
| g7.24xlarge-stand-in | 050-fork | 6.266013 s | 0 |
| g7.24xlarge-stand-in | 050-spawn | 6.942014 s | 0 |
| g7.24xlarge-stand-in | 050-cleanup | 2.290309 s | 0 |
| g7.48xlarge | 020-baseline | 36.629579 s | 0 |
| g7.48xlarge | 030-fallback | 52.865336 s | 0 |
| g7.48xlarge | 030-recovery | 21.522610 s | 0 |
| g7.48xlarge | 040-storage-injection | 104.124892 s | 1 |
| g7.48xlarge | 040-storage-recovery | 22.482526 s | 0 |
| g7.48xlarge | 040-fabric-recheck | 28.763585 s | 0 |
| g7.48xlarge | 050-fork | 8.911737 s | 0 |
| g7.48xlarge | 050-spawn | 13.533987 s | 0 |
| g7.48xlarge | 050-cleanup | 2.811254 s | 0 |

Full g7.48xlarge module commands totaled 291.645506 s; the g7.24xlarge-stand-in repeat totaled 264.616037 s. The checkpoint injection returned its expected nonzero status; recovery, the collective recheck, fork non-reproduction, spawn control and cleanup completed. These are machine command durations, not human-paced delivery measurements. The previously qualified Spain Prolog drain, held requeue, manual release onto the alternate and restoration procedure was retained rather than rearmed during these workload runs.

| Measured allocation | Node | Health check identifier | Wall time | Exit status, dimensionless |
|---|---|---|---|---|
| g7.48xlarge | ip-10-8-17-100 | 0 | 1.149466 s | 0 |
| g7.48xlarge | ip-10-8-17-100 | 1 | 305.990523 s | 0 |
| g7.48xlarge | ip-10-8-17-100 | 2 | 1.055042 s | 0 |
| g7.48xlarge | ip-10-8-17-100 | 3 | 0.471484 s | 0 |
| g7.48xlarge | ip-10-8-17-100 | 6 | 12.561965 s | 0 |
| g7.48xlarge | ip-10-8-30-139 | 0 | 1.171365 s | 0 |
| g7.48xlarge | ip-10-8-30-139 | 1 | 304.303598 s | 0 |
| g7.48xlarge | ip-10-8-30-139 | 2 | 1.077465 s | 0 |
| g7.48xlarge | ip-10-8-30-139 | 3 | 0.457747 s | 0 |
| g7.48xlarge | ip-10-8-30-139 | 6 | 12.605295 s | 0 |

The first diagnostic attempt began before the retained companion exporter was actually stopped. Its evidence was retained, then the complete matrix was repeated with `aim347-compute-dcgm-1` confirmed stopped. The table reports the clean repeat. Expected-count checks remain skipped by the unmodified upstream G7 profile gap. The unapplied profile candidate remains unapplied.

| Measured allocation | Baseline host | EFA device | RDMA-write received | RDMA-write sent |
|---|---|---|---|---|
| g7.48xlarge | ip-10-8-30-139 | rdmap83s0 | 181914273504 B | 181886889712 B |
| g7.48xlarge | ip-10-8-30-139 | rdmap176s0 | 181904999744 B | 181913760704 B |
| g7.48xlarge | ip-10-8-17-100 | rdmap83s0 | 181900454512 B | 363803035360 B |
| g7.48xlarge | ip-10-8-17-100 | rdmap176s0 | 181900195904 B | 16237888 B |

Both devices carried full-run traffic. Both devices’ counter deltas were zero in the Socket control. The selected stand-in EFA carried traffic while the unused EFA remained flat. NCCL reported `GDR 0`; these results do not prove the absence of host staging. Full raw command, cgroup, inventory, checkpoint, process-start and per-device counter evidence is under `/tmp/g7-dual-instance/pcs/aim344/` and the private Spain evidence prefix `g7-dual/aim344/`.
