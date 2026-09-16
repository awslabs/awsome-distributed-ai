# GPU failure-signature runbook

Start with the earliest observed failure. Choose a check that tests the suspected component, preserve the raw result, and verify recovery with the affected workload. The diagnostic owner is [validation/gpu-cluster-healthcheck](../../../validation/gpu-cluster-healthcheck).

## Select the next check

| Observed symptom | Evidence to collect | Recovery and reuse check |
|---|---|---|
| GPU query fails or a GPU disappears | UUID/BDF mapping, suite identifiers `0` and `3`, kernel journal, process state, and the active job log | Keep the target drained. Use the qualified external reboot route when required, restore runtime paths and services, then run the maintenance checks and a fresh collective/storage job. |
| EFA binding or provider domain disappears | PCI BDF, driver symlink, RDMA device, unique provider domains, suite identifiers `2` and `6`, and physical counters | End the old communicator before allowlisted rebind. Reboot if recovery remains incomplete. Requalify the node and take a fresh allocation. |
| Collective times out during checkpointing | Writer error and retry timestamps before the waiting ranks' timeout | Remove only the fixture's filler and partial write with `6.recover-storage.sh`, resume its saved step, and run Check 5 independently. |
| DataLoader stops after communication initialization | Process-start method, worker stacks, queue/lock state, fork-safety flag and software versions | Compare `fork` and `spawn` with the other settings held fixed. Keep successful non-reproduction as the observed outcome. |
| Correct collective runs slowly | Provider log, EFA byte deltas, host resources and repeated baseline from the same configuration | Resolve the observed cause and repeat the same sweep. The optional plugin-fallback exercise has its own baseline. |
| Application output differs from its reference | Failing input, actual and expected output, comparison method, device identities and reproducible command | Preserve the failing result and repeat under controlled placement and software. A passing inventory or collective covers only the work it exercised. |

## Interpret the suite verdict

| Severity or state | Facilitator response |
|---|---|
| `ISOLATE` | Keep the node out of scheduling and preserve the raw failure. A known administrative count mismatch requires recovery and qualification; an unexplained fault needs the platform's repair or replacement investigation. |
| `RESET` or an explicit reboot recommendation | Keep the node drained, use the approved recovery route, and retest before resuming. Confirm that the diagnostic reached the intended operation. |
| `MONITOR` or `WARN` | Preserve and review the warning, its counters and the executed coverage. Do not rewrite it as PASS. |
| Incomplete command | Record the deadline and process state. Use the external maintenance route; userspace timeout does not guarantee termination of a D-state task. |
| Skipped diagnostic coverage | Record which test did not run and why. A Level 4 invocation does not itself establish EUD execution. |

A drained node cannot receive an ordinary participant job. Use the maintenance route for fault diagnostics, then resume only after the required checks pass and the drain reason belongs to this exercise. The real Prolog runs before fresh participant jobs; its synthetic marker remains an optional scheduler demonstration.

## Verify reuse and retain evidence

After a reboot, verify the changed boot identifier, original instance and device identities, staging filesystem, image hashes, private 32 MiB tmpfs marker/ownership, `slurmd`, Prolog and prior telemetry owner. Keep the target drained during maintenance checks. Take a fresh allocation for suite Check 5 and the storage workload before declaring recovery complete.

Record the node and rank counts, exact command, suite commit, image digest, library versions, job identifier, start/end times, exit status, both correctness columns and per-device byte deltas. Keep Check 5's 8 B to 128 MiB sweep separate from the optional 8 B to 2 GiB plugin comparison. [VALIDATION.md](VALIDATION.md) preserves earlier measurements with their original hardware and launch scope.

Keep the dedicated instances and reservation capacity after the exercise. Clear only exercise-owned state and retain the raw evidence.
