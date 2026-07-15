---
name: hyperpod-incident-triage
description: Correlation and skip rules for SageMaker HyperPod incident triage. Keeps distinct fault types on the same instance group as separate investigations (the default correlator merges them), and prevents periodic-audit re-investigation of an unchanged cluster. Applies at the Incident Triage stage before any investigation runs.
metadata:
  version: "0.7.0"
  agent_types: ["INCIDENT_TRIAGE"]
---

# HyperPod incident triage

Use these rules when deciding whether a HyperPod incident should be **linked**
to an existing investigation, **skipped**, or investigated on its own. They
refine the default correlation for HyperPod's fault model. When a rule below
does not clearly apply, fall back to your normal triage judgment.

Every HyperPod incident carries a description built by the webhook bridge. Two
kinds arrive:

- **Cluster events** (a real HyperPod fault) — the description contains the
  instance group, a `Description:`, and often a `FailureMessage:`.
- **Periodic audits** (a scheduled health sweep) — the description begins with
  "Periodic audit invocation for HyperPod cluster" and says the audit should
  look for open fault chains.

## Linking rules for HyperPod fault events

Link a new HyperPod fault only to an existing HyperPod investigation that
describes **the same fault on the same component**. Treat the instance group
plus the fault text (`Description:` together with any `FailureMessage:`) as the
identity of the fault.

- **Do link** when both the instance group and the fault text match an open
  investigation — it is the same problem arriving again.
- **Do NOT link** when the instance group is the same but the fault text
  differs. Different fault types on the same instance group have different root
  causes and need separate investigations. For example, a lifecycle-script
  failure on `worker2` is **not** the same incident as a GPU/Xid fault on
  `worker2` — keep them separate even though they share the instance group.
- **Do NOT link** the same fault type across different instance groups (e.g.
  an Xid fault on `worker2` and an Xid fault on `worker3` are two distinct
  hardware faults).
- **Do NOT link** capacity errors for different instance types (e.g.
  `ml.g5.8xlarge` vs `ml.p5.48xlarge`) — the `FailureMessage` differs, so they
  are distinct problems.

When in doubt, prefer a separate investigation over an incorrect link:
over-linking hides distinct problems, which is the failure this skill exists to
prevent.

## Rules for periodic audits

A periodic audit is a scheduled sweep, not a specific fault. Decide based on
whether the cluster's condition has **changed** since the last audit:

- **Skip** the audit if another periodic-audit investigation for the same
  cluster is **currently in progress** — let it finish rather than starting or
  re-activating a parallel one.
- **Link** the audit to the most recent **completed** periodic-audit
  investigation only if the cluster's open problems are **unchanged** — the
  same fault chains, the same CrashLoopBackOff pods, and the same NotReady
  nodes as that prior audit found. There is nothing new to report.
- **Proceed** (investigate) if the cluster shows **any new or changed problem**
  since the last audit — a new fault event, a newly CrashLooping pod, or a
  newly NotReady node. A newly appeared problem must never be absorbed into a
  stale audit; it always gets a fresh investigation.
- **Proceed** if there is no prior audit to compare against.

Never link a periodic audit to an investigation that is still `PENDING_TRIAGE`,
`PENDING_START`, or `IN_PROGRESS` — linking re-activates it and can keep it
running indefinitely. Only skip (concurrent) or link to a completed one.

## Notes

- These are correlation preferences for the triage stage; the full diagnosis
  (timeline, verdict, recommended actions) is produced later by the
  `hyperpod-incident-rca` skill on investigations that proceed.
- Routine `Info`-level activity and scale-in-progress churn (events mentioning
  "lost orchestration-ready status") are already filtered upstream and should
  not, on their own, drive a new investigation.
