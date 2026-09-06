# GPU failure-signature runbook

Check **storage, then host, then fabric, then software**. Preserve the evidence, isolate a suspect node, and replace it rather than repairing it during the job. A health check narrows the fault space; correctness requires its own check.

## Signatures and first actions

The validation labels distinguish Seoul mechanism evidence on 2 p6-b300.48xlarge nodes from the production target of 2 p5.48xlarge nodes on PCS with Slurm version 25.11.

| Round and symptom | First action | Evidence that distinguishes the cause | Recovery and validation status |
| --- | --- | --- | --- |
| Round 0: establish healthy operation | Run `0.baseline.sh`; record both bus-bandwidth columns and both mismatch counts at 2 GiB. | Seoul p6-b300.48xlarge: 829.38 GB/s in-place at 2 GiB, 0 mismatches, positive EFA byte deltas. Prior p5.48xlarge reference: 400 GB/s class at 2 GiB with 0 mismatches. Separately, inspect scheduler logs for a pre-job rejection; a job output file may contain nothing. | Sweep **VALIDATED on Seoul**. The new PCS wrapper and drain/requeue exercise are **UNVALIDATED**. A node already in `DRAIN` cannot receive a job; the exercise injects the verdict consumed by the Prolog. |
| Round 1: job completes correctly but slows down | Compare with the same-node baseline; inspect NCCL's selected transport and per-device EFA byte deltas. | The complete signature observed on Seoul p6-b300.48xlarge was `NET/Socket`, 15.53 GB/s in-place at 2 GiB, 0 mismatches, and 0 B change in exposed EFA byte counters. Missing counters are not zero traffic. | Restore the OFI library directory using `4.recover-fallback.sh`. Seoul p6-b300.48xlarge recovered to 827.67 GB/s in-place at 2 GiB with 0 mismatches. Mechanism **VALIDATED**; p5/PCS execution **UNVALIDATED**. |
| Round 2: collective timeout during checkpointing | Inspect the actual checkpoint path with `df -h`, `df -i`, and its quota report; read the writer rank's error before interpreting the peer's timeout. | A write failure followed by a writer retry can prevent that rank from joining the next collective. The error reported by a waiting rank does not identify the original failing layer. A direct write exception may terminate the job without any collective timeout. | Remove only the lab filler, resume the last successful checkpoint, and recheck the fabric separately. Write exhaustion, watchdog timeout, and checkpoint recovery **VALIDATED on Seoul p6-b300.48xlarge** using a private 32 MiB tmpfs. A clean sweep while that path remained full reached 829.30 GB/s in-place at 2 GiB with 0 mismatches. FSx and PCS integration are **UNVALIDATED**. |
| Round 3: DataLoader job stalls although the fabric sweep passes | Compare process start methods, worker count, and the point at which workers start relative to the first completed collective. Capture the process tree and tested library versions. | A late fork is a software suspect when a same-node fabric check passes. Automatic EFA fork-safety handling may prevent the historical hang; successful completion is an admissible observation. | Probe with `7.probe-dataloader-fork.sh`; compare with `8.use-dataloader-spawn.sh`. The `fork` probe completed 4 batches with 0 mismatches on Seoul p6-b300.48xlarge, with automatic EFA fork safety enabled. Non-reproduction **VALIDATED** on the recorded stack; historical hang and PCS execution **UNVALIDATED**. |

A nonzero mismatch count rejects correctness even when bandwidth is high. The sweep validates the arithmetic it exercised; it does not certify every training workload or prove that all hardware faults are absent.

## Check the instrument before acting on its verdict

| Instrument behavior | Interpretation and next step |
| --- | --- |
| Stock health check number 5 returns severity `RESET` almost immediately | Prior healthy p5.48xlarge nodes on PCS failed in 2.70 s because of the Enroot URI separator and binary path. Use the pinned corrected invocation in the README; confirm the test actually reached NCCL before attributing its result to hardware. |
| A test claims to select an EFA device with `FI_EFA_DEVICE_NAME` | That variable is ignored. Use `FI_EFA_IFACE`. Repeated passes cannot establish per-device coverage when the selector never took effect. |
| MPI reports an OFI error before the expected watchdog | Separate launcher transport from NCCL transport. The lab's bandwidth sweep uses TCP for MPI setup; its checkpoint and DataLoader workloads use `torchrun` without MPI. |
| A pre-job failure requeues into a held state | Inspect the scheduler setting controlling hold on Prolog failure. Automatic retry requires `nohold_on_prolog_fail`; the facilitator must otherwise release the job after making a healthy alternate available. Read `journalctl -t aim344-prolog` and `journalctl -u slurmd`, not only the batch output. |
| `iptables` or `tc netem` changes leave EFA traffic unaffected | EFA bypasses the kernel networking path. Those tools do not establish a severed EFA fabric. |

## Evidence to keep

Record the instance type and IDs, node and GPU-rank counts, image digest, driver and library versions, exact command, exit status, raw logs, both correctness columns, and byte-counter snapshots. Compare message sizes and run configurations consistently. Put units on measurements, and keep p6-b300.48xlarge observations separate from p5.48xlarge session baselines.
