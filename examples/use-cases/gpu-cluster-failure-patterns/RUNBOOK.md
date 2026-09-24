# GPU failure-signature runbook

Start with the earliest observed failure. Choose a check that tests the suspected component, preserve the raw result, and verify recovery with the affected workload. The diagnostic owner is [validation/gpu-cluster-healthcheck](../../../validation/gpu-cluster-healthcheck).

For the current participant device session, follow the authoritative [Device recovery participant route](facilitator/DEVICE-RECOVERY.md#participant-session): use supported `./12.device-exercise.sh recover` first, retain isolation when recovery cannot qualify the node, and use `./12.device-exercise.sh replace`. External/manual administration belongs only to historical experiments or separately authorized pre-session work, not an instructor session remedy. The [scoped audit](VALIDATION.md#september-19-2026-scoped-participant-recovery-audit) does not grant full hardware qualification or publication approval.

## A watchdog is a symptom, not a component diagnosis

Read the full coordinator workload log and compare rank identities before selecting a recovery. GPU and EFA reset trials both produced ProcessGroupNCCL watchdogs; the storage fixture produced the same watchdog when the checkpoint writer retried ENOSPC instead of entering the collective. Record the earliest relevant device/kernel or writer error, the affected rank set, sequence and operation, and the result of each diagnostic. Initialization on 16 ranks does not prove that all 16 emitted an error.

Use `bash 12.device-exercise.sh collect kernel` for the recorded fault boot and checks `0`/`3` for GPU or `2`/`6` for EFA. Read both the collection transport status and the inner command result. An unavailable journal or failed query is missing evidence, not a clean device. A post-reboot PASS describes that later boot. The corrected collector retrieves a selected last-512-record kernel window, not a complete journal.

The latest EFA trial recorded reset start/return, watchdogs on all 16 ranks and detected correctness exceptions. Supported recovery refused a retained cumulative retransmission WARN; participant replacement and a fresh collective/storage job plus retrieval passed. The participant GPU attempt had only reset start, an unexpected reboot and eight peer-rank watchdogs; successful later recovery does not resolve its missing reset-return and fault-time kernel evidence. See [the measured scope](VALIDATION.md#september-24-2026-fixed-candidate-participant-efa-recovery).

## Select the next check

| Observed symptom | Evidence to collect | Recovery and reuse check |
|---|---|---|
| GPU query fails or a GPU disappears | UUID/BDF mapping, suite identifiers `0` and `3`, kernel journal, process state, and the active job log | Keep the target isolated. Supported `recover` validates original records and the reboot baseline before recovery; if it cannot qualify the node, retain isolation and use `replace`. Verify reuse with a fresh collective/storage job. |
| EFA binding or provider domain disappears | PCI BDF, driver symlink, RDMA device, unique provider domains, suite identifiers `2` and `6`, and physical counters | Use supported `recover`: idle unbind uses rebind; active FLR uses checked reboot/runtime recovery. If required originals or health qualification are missing, use `replace` and retain isolation. Verify a fresh communicator. |
| Collective times out during checkpointing | Writer error and retry order before the waiting ranks' timeout; retain timestamps when present | Remove only the fixture's filler and partial write with `6.recover-storage.sh`, resume its saved step, and run Check 5 independently. |
| DataLoader stops after communication initialization | Process-start method, worker stacks, queue/lock state, fork-safety flag and software versions | Compare `fork` and `spawn` with the other settings held fixed. Keep successful non-reproduction as the observed outcome. |
| Correct collective runs slowly | Provider log, EFA byte deltas, host resources and repeated baseline from the same configuration | Resolve the observed cause and repeat the same sweep. The optional plugin-fallback exercise has its own baseline. |
| Application output differs from its reference | Failing input, actual and expected output, comparison method, device identities and reproducible command | Preserve the failing result and repeat under controlled placement and software. A passing inventory or collective covers only the work it exercised. |

## Interpret the suite verdict

| Severity or state | Participant response |
|---|---|
| `ISOLATE` | Keep the node out of scheduling and preserve the raw failure. A known administrative count mismatch requires recovery and qualification; an unexplained fault needs the platform's repair or replacement investigation. |
| `RESET` or an explicit reboot recommendation | Keep the node isolated and use supported `recover`/`replace`; a diagnostic recommendation does not authorize a manual reboot or bypass original-state checks. Confirm that the diagnostic reached the intended operation and retest before reuse. |
| `MONITOR` or `WARN` | Preserve and review the warning, its counters and the executed coverage. Do not rewrite it as PASS. |
| Incomplete command | Record the deadline and process state; userspace timeout does not guarantee termination of a D-state task. Follow supported `recover`/`replace` and retain isolation while incomplete; repeat `replace` for the same operation when instructed, not external/manual restoration. |
| Skipped diagnostic coverage | Record which test did not run and why. A Level 4 invocation does not itself establish EUD execution. |

A drained node cannot receive an ordinary participant job. Use the participant route's `collect` commands for fault diagnostics; supported recovery/replacement handles maintenance checks and admission, and must preserve unrelated isolation. Do not manually resume the node. The real Prolog runs before fresh participant jobs; its synthetic marker remains an optional scheduler demonstration.

## Verify reuse and retain evidence

After a supported recovery reboot, retain the controller's checks of the changed boot identifier, original instance and device identities, staging filesystem, image hashes, private 32 MiB tmpfs marker/ownership, `slurmd`, Prolog and prior telemetry owner; these are not manual restoration instructions. Keep isolation through qualification. After recovery or replacement, submit `13.verify-after-recovery.sbatch` from the independent login coordinator for fresh-allocation suite Check 5 and storage verification, as documented in the participant route, before declaring workload reuse verified.

Record the node and rank counts, exact command, suite revision, image digest, library versions, job identifier, start/end times, exit status, both correctness columns and per-device byte deltas. After digest verification and extraction, open `~/aim344-verify-$job/verify-$job/check5/nccl-allreduce-raw.txt` and `~/aim344-verify-$job/verify-$job/check5/nccl-efa-only.txt` with `less`, using the verification job identifier. The terminal log does not contain all raw evidence. Check all 25 main rows and the separate EFA-only isolation row, all 16 GPU rank headers across two MPI processes, and zero out-of-bounds values. Preserve provider/performance warnings and retransmission deltas independently of correctness. Keep Check 5's 8 B to 128 MiB sweep separate from the optional 8 B to 2 GiB plugin comparison. [VALIDATION.md](VALIDATION.md) preserves earlier measurements with their original hardware and launch scope.

When `replace` returns a pending stage with exit code `3`, let the invocation end, then repeat the same command as instructed. Do not overlap control commands, restart the fault or cancel the pending bootstrap. `runtime-ready` is followed by fresh workload verification and participant bundle retrieval, not completion by itself. The measured EFA workload-allocation-to-retrieval interval was 35 minutes 9 seconds, so the unapproved 60-minute proposal uses one selected live branch per table rather than three sequential faults. Device tables compare recorded storage and DataLoader outcomes; storage tables execute storage and DataLoader and compare recorded device outcomes. Both routes then compare correctness coverage and finish this runbook. Keep all full branches available for separately scheduled practice; content-owner agreement and human pacing remain open.

Keep the dedicated instances and reservation capacity after the exercise. Clear only exercise-owned state and retain the raw evidence.
