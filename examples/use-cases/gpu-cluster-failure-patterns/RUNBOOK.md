# GPU failure-signature runbook

Check **storage, then host, then fabric, then software**. Preserve the earliest error and use the evidence to choose a recovery action. A health check narrows the fault space; correctness requires its own check. Verify the instrument before treating its severity as a hardware diagnosis.

## Signatures and first actions

These observations use two g7e.12xlarge nodes on PCS Slurm version 25.05.7. The participant guide targets two p4d.24xlarge or p4de.24xlarge nodes; those A100 measurements remain unvalidated. Historical Seoul and p5 evidence is retained in [VALIDATION.md](VALIDATION.md).

| Round and symptom | First action | Observed g7e evidence | Recovery and scope |
| --- | --- | --- | --- |
| Round 0: establish healthy operation | Run `0.baseline.sh`; record both bandwidth and mismatch columns at 2 GiB. Inspect the Prolog journal separately for pre-job failures. | 33.49 GB/s out of place and 33.45 GB/s in place, zero mismatches and positive EFA traffic. The synthetic gate drained a node and requeued the job into a held state. | The job completed on the alternate after operator release. A drained node cannot receive an ordinary participant job. |
| Round 1: correct but slower | Compare transport logs and EFA byte deltas with the same allocation's baseline. | Plugin removal selected `NET/Socket`: 13.51 GB/s out of place and 12.98 GB/s in place at 2 GiB, zero mismatches and zero exposed EFA byte changes. | `4.recover-fallback.sh` restored OFI and recovered to 33.45 GB/s out of place and 33.34 GB/s in place. Missing counters are not zero traffic. |
| Round 2: collective timeout during checkpointing | Find the writer's earliest storage error and inspect the private checkpoint path. | `ENOSPC` preceded peer watchdog timeouts around the configured 30 s deadline. The whole failed command took 109.045003 s including startup and teardown. | Remove only the lab filler, resume the last successful checkpoint and recheck the fabric. Recovery and correctness passed. The fixture was private 32 MiB tmpfs, not an FSx outage. |
| Round 3: late DataLoader workers | Compare the process start method and the point at which workers start relative to the first collective. | Both fork and spawn completed four batches with zero mismatches; automatic EFA fork safety was enabled. | Non-reproduction is an admissible result. This does not establish that every forked CUDA workload is safe. |

A nonzero mismatch count rejects correctness even when bandwidth is high. The sweep validates the arithmetic it exercised; it does not certify every training workload or prove that all hardware faults are absent.

## Check the instrument before acting on its verdict

| Instrument behavior | Interpretation and next step |
| --- | --- |
| Stock health check number 5 returns severity `RESET` almost immediately | Prior healthy p5.48xlarge nodes on PCS failed in 2.70 s because of the Enroot URI separator and binary path. Use the separate participant invocation in the README; confirm the test actually reached NCCL before attributing its result to hardware. |
| A test claims to select an EFA device with `FI_EFA_DEVICE_NAME` | That variable is ignored. The pinned suite still uses it; repeated passes cannot establish per-device coverage. Adopt the `FI_EFA_IFACE` correction only after upstream merge and a pin update. |
| MPI reports an OFI error before the expected watchdog | Separate launcher transport from NCCL transport. The lab's bandwidth sweep uses TCP for MPI setup; its checkpoint and DataLoader workloads use `torchrun` without MPI. |
| A pre-job failure requeues into a held state | Inspect the scheduler setting controlling hold on Prolog failure. Automatic retry requires `nohold_on_prolog_fail`; the facilitator must otherwise release the job after making a healthy alternate available. Read `journalctl -t aim344-prolog` and `journalctl -u slurmd`, not only the batch output. |
| `iptables` or `tc netem` changes leave EFA traffic unaffected | EFA bypasses the kernel networking path. Those tools do not establish a severed EFA fabric. |

## Upstream severity and facilitator response

| Severity | Action after checking the instrument | Relationship to the exercise |
| --- | --- | --- |
| `ISOLATE` | Keep the node drained, preserve the raw hardware signal and arrange replacement. | A nonzero correctness mismatch rejects the computation; investigate the allocation before reuse. |
| `RESET` | Hold the node out of participant scheduling, reboot and retest. The upstream parser recommends instance reboot rather than assuming GPU reset is sufficient. | An early container or MPI launch failure is an instrument failure to diagnose, not an established GPU failure. |
| `REBOOT` | Follow the explicit reboot recommendation and rerun the gate before resuming. | Use the same held-out recovery path as `RESET`. |
| `MONITOR` | Keep the warning and review it while required checks and correctness remain passing. | Slow but correct work requires transport, host and storage evidence before replacement is considered. |

The synthetic Prolog marker does not diagnose hardware. The upstream `slurm/sbatch-quarantine-workflow.sh` recommends actions after a lightweight gate and, where needed, deeper diagnostics. The facilitator must arrange scheduler isolation; a drained node cannot receive an ordinary batch job. Preserve the actual Slurm node name because the wrapper's physical `hostname` can differ on PCS. Never resume or replace a node solely from a synthetic exercise verdict.

## Evidence to keep

Record the instance type and IDs, node and GPU-rank counts, image digest, driver and library versions, exact command, exit status, raw logs, both correctness columns, and byte-counter snapshots. Compare message sizes and run configurations consistently. Put units on measurements, and keep g7e, Seoul p6-b300 and historical p5 observations separate from the A100 production target.
