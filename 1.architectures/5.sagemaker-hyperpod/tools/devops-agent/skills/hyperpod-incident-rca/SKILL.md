---
name: hyperpod-incident-rca
description: Root-cause analysis for a SageMaker HyperPod incident, after triage has decided to PROCEED. Runs at the INCIDENT_RCA stage. Reads describe-cluster, list-cluster-nodes, list-cluster-events, and HMA CloudWatch streams; reconstructs a timeline; classifies as Suppress / Monitor / Escalate / Resolved against time budgets and recurrence statistics from the HyperPod mental model. Produces a human-readable verdict report with recommended operator actions. The complementary INCIDENT_TRIAGE skill `hyperpod-incident-triage` decides LINKED / SKIPPED / PROCEED before this skill runs.
metadata:
  version: "0.6.1"
  agent_types: ["INCIDENT_RCA"]
---

# HyperPod incident RCA skill

This skill runs at the **INCIDENT_RCA stage**, after DevOps Agent's
triage stage decided to `PROCEED` with a full investigation. By the
time this skill loads, the task already has a primary investigation
slot and the agent has full AWS API access.

**The complementary triage skill** `hyperpod-incident-triage` runs at
the INCIDENT_TRIAGE stage BEFORE this skill, and decides whether an
incoming event should be `LINKED` / `SKIPPED` / `PROCEED` using concise
declarative correlation rules (keep distinct fault types on the same
instance group separate; skip concurrent periodic audits). Separately,
for periodic audits, the **audit Lambda gates volume** — it inspects
Kubernetes state itself and only invokes an investigation when a real
issue is present (plus a daily heartbeat), so a "periodic audit"
investigation reaching this skill already corresponds to something
worth looking at.

When this skill DOES run, it does triage-like classification and
root-cause analysis in one pass because they need the same evidence:
snapshot data
(`describe-cluster`, `describe-cluster-node`) alone cannot distinguish
"HyperPod is auto-recovering" from "HyperPod has given up" — only the
cross-source timeline can.

**Operating policy.** Read-only. Never call a state-changing API.
Present every action that mutates anything (operator-driven replacement,
SSM session, label change) as a **Recommended action (operator runs
this)** block. The customer's operator decides whether to execute it.

**Required reading.** Open
[references/hyperpod-mental-model.md](references/hyperpod-mental-model.md)
in parallel with this skill. The mental-model doc is the ground truth
for what each HyperPod signal means; this skill is the decision
procedure that consumes those meanings.

## Why one skill, not two

A single failed instance can vanish from `list-cluster-nodes` between
retry attempts, and HyperPod may auto-retry from `Failed` status —
neither is a terminal escalation signal on its own. Distinguishing
"still retrying within budget" from "stuck" requires the full
`list-cluster-events` chain, the HMA CloudWatch stream, and the cluster
status all aligned on a wall-clock timeline. Building a separate
"triage" skill that decides without that data would mean re-deciding
incorrectly on every event.

## Relationship to `hyperpod-incident-triage` and the audit Lambda

Two things run before this RCA skill and mean it does NOT need to
re-implement duplicate/stale-audit suppression itself:

- **`hyperpod-incident-triage`** makes the LINK / SKIP / PROCEED
  decision with concise declarative rules (same fault on the same
  component links; different fault types on the same instance group
  stay separate; concurrent periodic audits are skipped).
- **The periodic-audit Lambda gates volume**: for scheduled audits it
  inspects Kubernetes state (CrashLoopBackOff / NotReady) and POSTs the
  webhook only when a real issue exists — so a periodic-audit
  investigation reaching this skill already corresponds to a real
  finding, not an idle poll. HyperPod control-plane faults arrive via
  the event-driven webhook bridge, not the audit.

RCA still does the full Phase 1 gather (including Pod/Node inspection
and Phase 3d threshold checks) because the reasoning stage needs the
full evidence. Earlier RCA "stale-evidence" rules (3 / 3b) were removed
and are not reintroduced — that concern is handled upstream now.

## Workflow

### Trigger modes

The skill is loaded from two trigger sources, and Phase 1 / Phase 3
behavior differs slightly between them:

| Mode | Trigger | Scope |
|---|---|---|
| **Incident mode** | Webhook fire from the bridge Lambda (an EventBridge HyperPod event passed the noise filter). The investigation context carries `clusterName` and usually an `instanceId`. | Focused on the specific incident referenced in the trigger. |
| **Audit mode** | Scheduled `TIME_BASED` trigger (rate(15 minutes)) creating a task against this skill with no per-incident context. | Scan the cluster(s) configured for this Agent Space for any in-flight or recently-resolved fault chains, and re-classify each one. Catches: (1) `Monitor` incidents that have now succeeded (emit `Resolved`), (2) `Monitor` incidents stuck past their re-check budget (escalate), (3) recurring patterns that didn't trigger an EventBridge event recently but persist statistically. |

In both modes the same Phase 1 / Phase 2 / Phase 3 / Phase 4 logic
applies. The differences are noted inline below.

### Phase 1 — Gather (run in parallel)

For the incident referenced in the trigger event (cluster name + optional
instance id), collect these in parallel. **Do not stop on a single
signal — gather all of them before classifying.**

**Audit mode**: the trigger payload won't carry a cluster name or
instance id. Use the cluster(s) reachable from this Agent Space's AWS
account association (typically just one HyperPod cluster — discover
it via `sagemaker list-clusters`). For each cluster, run the full
Phase 1 gather. Then in Phase 3, classify per open fault chain found
in the cluster-events window, not per single trigger.

1. **Cluster state**: `aws sagemaker describe-cluster --cluster-name <name>`
   — current `ClusterStatus`, `NodeRecovery`, `Orchestrator` (Eks vs.
   Slurm), `InstanceGroups[].CurrentCount` / `TargetCount` / `Status`.
2. **Node inventory**: `aws sagemaker list-cluster-nodes --cluster-name <name>`
   plus `describe-cluster-node` for any instance id named in the trigger
   event. **An instance id mentioned in the event that is NOT in
   `list-cluster-nodes` is a signal, not an error** — the node may have
   been removed mid-retry.
3. **Cluster events chain**: `aws sagemaker list-cluster-events --cluster-name <name>`,
   paginated to ≥500 entries or until events reach **7 days** back,
   whichever comes first. This is the **canonical record of replacement
   attempts including failed ones** and survives nodes disappearing
   from the node list. The wider 7-day window also feeds the recurring-
   pattern classification rules in Phase 3 — don't shorten the lookback
   even when the trigger event is recent.
   - Available on EKS clusters and on Slurm clusters with **Continuous
     Provisioning** enabled. If the API returns
     `ValidationException`/equivalent on a Slurm cluster without CP,
     note "event timeline unavailable; confidence degraded" in the
     final report and proceed with the remaining sources.
4. **HMA CloudWatch stream**: log group
   `/aws/sagemaker/Clusters/<NAME>/<CLUSTER_ID>`, streams matching
   `SagemakerHealthMonitoringAgent/*` (filter to the affected instance
   group / instance if known). Filter for `HealthMonitoringAgentDetectionEvent`
   entries and any Xid / DCGM / EFA / OOM messages within the same
   window as Phase 1 step 3.
5. **Lifecycle script stream** (only if a replace attempt is in the
   timeline): same log group, streams matching `LifecycleConfig/*` for
   the affected instance group / instance id. Look for non-zero exit,
   timeout, S3 / IAM errors.
6. **EKS node state** (EKS only): `kubectl get node <name> -o yaml` for
   the affected node (if still present). Surface
   `sagemaker.amazonaws.com/node-health-status`, `fault-types`,
   `fault-reasons`, `fault-details` labels/annotations and any taints.
7. **Slurm node state** (Slurm only, if SSM not required): use what's
   reachable from the control plane — `describe-cluster-node` gives
   most of what we need; deep `scontrol`/`sinfo` requires SSM and is
   out of scope here.
8. **Kubernetes state — MANDATORY in audit mode on EKS clusters.**
   You MUST execute both of the following before entering Phase 2,
   even if the trigger payload has no `k8sChecks` block, even if
   earlier gather steps already suggest a plausible verdict, and
   even if `list-cluster-events` returned a rich history. Live
   Pod/Node state is the source of truth for "what is broken *right
   now*" — the wider event history is context, not a substitute.
   - `kubectl get pods -A -o json` — full cluster Pod state.
   - `kubectl get nodes -o json` — full Node state including
     `status.conditions[]` and their `lastTransitionTime`.

   Skip this step only if: (a) the cluster's orchestrator is Slurm
   (not EKS), OR (b) the trigger is incident-mode (webhook-triggered
   for a specific fault; Pod/Node scan belongs in periodic audit).

   The trigger's `k8sChecks` block supplies **thresholds and
   namespace filters** used by Phase 3d — not a gate on whether
   kubectl runs. Fields:

   ```json
   {
     "enabled": true,
     "crashLoopHoursThreshold": 4,
     "notReadyNodePercentThreshold": 10,
     "notReadyDurationMinutes": 15,
     "ignoreNamespaces": ["kube-public", "kube-node-lease"],
     "systemNamespaces": ["kube-system", "aws-hyperpod", "amazon-cloudwatch"]
   }
   ```

   **Where to find this block at runtime.** The DevOps Agent
   platform preserves the top-level `description` string of the
   incoming task verbatim, but drops nested sub-objects from the
   webhook payload. The audit Lambda therefore inlines the
   `k8sChecks` block into the task description text on a line that
   begins:

   ```
   k8sChecks configuration (parse as JSON, then apply per Phase 1 step 8 + Phase 3d):
   { ... }
   ```

   Locate that line in the task description, extract the JSON
   object on the next line, and use its fields for Phase 3d
   thresholds and namespaces. Do NOT default any field unless the
   whole block is absent — the block's values override the built-in
   defaults regardless of whether they equal the defaults.

   If the block is absent from the description entirely, use
   defaults: `crashLoopHoursThreshold=4`,
   `notReadyNodePercentThreshold=10`, `notReadyDurationMinutes=15`,
   `ignoreNamespaces=["kube-public","kube-node-lease"]`,
   `systemNamespaces=["kube-system","aws-hyperpod","amazon-cloudwatch"]`.

   If the block has `enabled: false`, still run the kubectl commands
   (Phase 1's job is discovery), but Phase 3d will not fire — see
   note there.

### Phase 1 gather sanity gate — verify BEFORE entering Phase 2

Before starting Phase 2 (timeline reconstruction), confirm that
Phase 1 executed the required steps:

- Steps 1, 2, 3 (describe-cluster, list-cluster-nodes,
  list-cluster-events): required for both incident and audit modes.
- Step 8 (kubectl get pods, kubectl get nodes): required in audit
  mode on EKS. Missing here is a **hard error** — do not proceed to
  Phase 2. Instead: run step 8 now, then re-enter this gate.

Do not rationalize skipping step 8 with reasoning like "the
`list-cluster-events` window looks quiet so I don't need to check
pods" or "the LCS storm from earlier is more interesting." Those
are outputs of Phase 3 reasoning, not inputs to Phase 1. Pod state
must be gathered as **evidence** before Phase 3 rules can weigh
"live k8s problem" against "historical event pattern."

### Phase 2 — Reconstruct the timeline

Build a single ordered timeline keyed by UTC timestamp across all
sources, restricted to the affected scope (cluster, instance group, or
instance). Mark each entry with its source. The shape should be:

```
T+0:00   [HMA]            HealthMonitoringAgentDetectionEvent — Xid 79 on GPU 3
T+0:32   [ClusterEvent]   Action:Replace marked on i-aaa (NodeRecovery=Automatic)
T+0:33   [Node]           list-cluster-nodes: i-aaa removed
T+0:34   [ClusterEvent]   Replacement started for instance group worker1
T+18:21  [ClusterEvent]   Replacement failed: EFA health checks did not run successfully
T+18:25  [Node]           list-cluster-nodes: still missing
T+20:10  [ClusterEvent]   Replacement started for instance group worker1   ← second attempt
...
```

This is the artifact the classification phase reasons over. Include it
in the final report regardless of verdict — operators need it to
double-check the agent's call.

### Phase 2b — Build the fault-content signature for each event

Every fault event (Error/Warn cluster event or HMA detection within
the 4h window for signature-set computation, and within the 7d window
for recurrence statistics) gets a **signature string**, formed by
concatenating the event's Description with any FailureMessage the
bridge Lambda enriched it with, prefixed by the InstanceGroup name.

Signature format:

```
<ig>:<full-description-and-failure-message-content>
```

Sources of the content, in order of preference:

1. The bridge Lambda's enriched description (top-level `description`
   field in the incoming trigger payload). This already includes
   `Description: ...` + `FailureMessage: ...` + `InstanceMetadata: ...`
   fields the bridge assembled by calling `DescribeClusterEvent`.
2. For events retrieved via `list-cluster-events` during Phase 1
   pagination (not through the bridge), call
   `describe-cluster-event --event-id <eid>` yourself for each
   Error/Warn EventId. Concatenate `Description` +
   `EventDetails.EventMetadata.Instance.FailureMessage` (and any
   other populated `Instance.*` fields) the same way the bridge does.
3. For HMA CloudWatch stream events: use the full
   `HealthMonitoringAgentDetectionEvent` text.
4. Cluster State Change events: use `ClusterStatus + Description`.

**Why full concatenated content, not hard-coded categories.**
Classifying events into a fixed enum of categories (`gpu-xid`,
`lifecycle-script-failed`, `capacity-insufficient`, etc.) via regex
rules merges distinct fault types — for example, "capacity for
ml.g5.8xlarge", "capacity for ml.p5.48xlarge", "EFA health check
failed", and "generic provisioning failure" all collapse to the same
`instance-creation-failed:generic` key, because the top-level
`Description` field only says `"Failed to provision EC2 Instance in
Cluster ..."`. The actual root cause lives in `FailureMessage` (a
different string per fault type), so include it. Concatenated raw
content is more robust than enum-based categories.

**InstanceGroup extraction**:
- Cluster Event: `detail.EventDetails.InstanceGroupName`
- Node Health / Cluster State Change: `detail.InstanceGroupName`
- Cluster-level fault: empty (`""`) — no IG prefix.

### Phase 2c — Compute recurrence statistics

Over the 7-day `list-cluster-events` window from Phase 1 (and HMA
CloudWatch stream over the same window), compute and record (used by
Phase 3 rules 6–8):

- `replacements_7d_total` — count of `Replace` actions / replacement-
  started cluster events across the whole cluster in the last 7 days.
- `replacements_7d_by_group[<ig>]` — same, partitioned by InstanceGroup.
- `replacements_24h_total` — same metric, 24h window.
- `signature_count_7d[<signature>]` — count of distinct fault events
  with the same signature string in the last 7 days. Uses the
  Phase 2b signature format `<ig>:<content>`.
  **Exclusion**: events whose Description contains
  `"lost orchestration-ready status"` MUST be excluded from
  signature counting. These are scale-in-progress noise (see
  mental-model § "Scale-in-progress emits spurious Warn events"),
  not fault signals. Do NOT count them toward rule 6 thresholds.

Include these counts in the verdict description's "What HyperPod is
doing right now" paragraph when they're ≥2 — even if the verdict
itself doesn't change, the operator should see the count.

### Phase 2d — Compute the signature set

The signature set is used for verdict-title generation. Encoding it in
the title lets the DevOps Agent platform's title-based dedup absorb
back-to-back audits of the same fault set (identical set → identical
title), while a new fault set produces a new title and a fresh
investigation.

- `current_signature_set` — sorted, deduplicated set of signature
  strings (per Phase 2b) for every distinct fault event in the
  4-hour window. Examples:
  - `{"worker4:Description: Failed to provision EC2 Instance in Cluster my-cluster and InstanceGroup worker4. FailureMessage: We currently do not have sufficient capacity to launch new ml.g5.8xlarge instances. Please try again."}`
  - `{"worker2:Description: Instance i-XXXX is unhealthy. HyperPod Health Monitoring Agent (HMA) has detected fault type NvidiaGPUUnhealthy on this node and is unhealthy. Repair action: Replace."}`

  Render in the verdict title as a sorted comma-separated list.
  Full strings; do NOT truncate. If total title length exceeds
  platform limits (rare but possible for very long FailureMessages),
  trim signatures from the middle with `...` markers to preserve
  the beginning (where the discriminating text lives).
- `current_most_recent_event_at` — the latest `EventTime` of any
  Error/Warn cluster event or HMA detection within the window.

### Phase 3 — Classify

**Ordering — MANDATORY.**

1. **First**, run **Phase 3d** (Kubernetes-state checks — CrashLoopBackOff,
   NotReady). Emit any Escalate verdicts from Phase 3d immediately.
2. **Then**, run the fault-chain rules below in this Phase 3 table.

Reason: Phase 3d looks at **live current state**, which is the most
actionable signal for a periodic audit. The main rule table below
looks at the 4h/24h/7d event history, which is context. When both
would fire, the operator needs the live state surfaced regardless
of whether historical patterns are ALSO present. Do not skip Phase
3d because "the event history already provides an interesting
verdict" — that produced the 2026-07-09 miss where a live
CrashLoopBackOff was overshadowed by a recurring-LCS-pattern
verdict from a *already-resolved* 07-08 storm.

Emitting both a Phase 3d verdict and a Phase 3 verdict on the same
audit is expected and correct. Each becomes its own symptom record;
the operator gets both signals via email.

#### Phase 3d — Kubernetes-state checks (audit mode, EKS only)

Run this **BEFORE** the fault-chain classification rules below
(see the "Ordering — MANDATORY" note at the top of Phase 3). Phase
3d uses the `kubectl get pods` / `kubectl get nodes` output from
Phase 1 step 8, which is mandatory in audit-mode-on-EKS regardless
of the k8sChecks payload block. Emit each verdict as an
independent symptom record.

**Threshold + namespace configuration.** Read from the payload's
`data.metadata.k8sChecks` block when present; otherwise use the
defaults documented in Phase 1 step 8. If the block is present with
`enabled: false`, DO NOT emit Phase 3d verdicts (customer has
opted out of the k8s-state Escalate) — but Phase 1's kubectl gather
still ran because it's part of general state discovery, and its
data may still surface in Phase 4 report context.

**Pod check — CrashLoopBackOff duration.** For each Pod in the
`kubectl get pods -A -o json` output:

1. **Namespace classification.** Look up `pod.metadata.namespace`:
   - If it is in `k8sChecks.ignoreNamespaces` → skip this pod entirely.
   - If it is in `k8sChecks.systemNamespaces` → workload class is
     `system-workload`.
   - Otherwise → workload class is `customer-workload`.

   This is a plain set-membership lookup, in this exact order. No
   pattern matching, no wildcards, no precedence rules. The Lambda
   has already validated that `ignoreNamespaces` and
   `systemNamespaces` do not overlap, so a pod's namespace matches
   at most one list.

2. **CrashLoopBackOff detection.** For each container in
   `pod.status.containerStatuses[]`, check
   `state.waiting.reason == "CrashLoopBackOff"`. If so, compute the
   duration since `state.waiting.startedAt` (fall back to
   `lastTransitionTime` on the container's `Ready` condition if
   `startedAt` is not present).
3. **Threshold check.** If duration exceeds
   `k8sChecks.crashLoopHoursThreshold` hours → emit an `Escalate`
   verdict with:
   - Title: `Triage verdict: Escalate — CrashLoopBackOff exceeded threshold :: (<namespace>/<pod>:<container>)`
   - Workload class tag in the description (`system-workload` or
     `customer-workload`) so downstream email routing can decide who
     to page.
   - Include: pod name, namespace, container name, restart count,
     duration in CrashLoopBackOff, and the most recent
     `lastState.terminated.reason` / `.exitCode` if available.

**Node check — NotReady percentage.** From
`kubectl get nodes -o json`:

1. Total node count = length of `items[]`.
2. For each node, examine `status.conditions[]` for
   `type: Ready`. If `status: "False"` or `"Unknown"` AND
   `now() - lastTransitionTime` exceeds
   `k8sChecks.notReadyDurationMinutes` minutes → count as NotReady.
3. If `(NotReady count / total count) * 100` meets or exceeds
   `k8sChecks.notReadyNodePercentThreshold` → emit an `Escalate`
   verdict with:
   - Title: `Triage verdict: Escalate — NotReady nodes exceeded threshold :: (<n>/<total> nodes NotReady)`
   - Include the affected node names, their NotReady durations, and
     any `taints[]` that would explain the state.

**Interaction with the fault-chain classification.** If a HyperPod
fault chain in the main rule table already covers the same
node (e.g. rule 11 says the retry chain is stuck for a specific
instance), the Phase 3d NotReady check may re-flag the same node.
That's fine — the two verdicts have different scopes and different
recommended actions. Emit both; the operator gets richer context.

**Interaction with `Suppress — periodic audit, no open incidents`.**
Rule 1 fires when no HyperPod fault chains are open. Phase 3d can
still fire independently — a CrashLoopBackOff pod is an incident
even when HyperPod's own event stream is quiet. If Phase 3d
produces any verdict, do NOT emit the rule 1 Suppress; Phase 3d has
found something to report.

#### Fault-chain classification rules

Apply the rules below **in order**. Stop at the first match.

**Audit mode**: classify each open fault chain found in the
cluster-events window independently. Emit one verdict per chain
(plus one `Suppress — periodic audit, no open incidents` if no
chains are open AND Phase 3d also emitted nothing). A fault chain
is "open" if it had a fault event within the last 4 hours and has
not yet emitted a successful `Running` transition + 30 min
clean-window.

**Historical-only chains are not open.** If the only fault events
in the 4-hour window belong to a chain that has already completed
(e.g., all replacements succeeded and all affected nodes are back
in `Running` for ≥30 min), the chain is closed. Do not emit an
`Escalate` verdict on the basis of `signature_count_7d` alone
against a closed chain unless the recurrence rule 6/7/8 fires from
statistics that INCLUDE at least one event within the last 4 hours.
A 7-day-old already-recovered pattern is context, not an actionable
incident.

The recurring-pattern rules (6–8) are checked **before** the
single-incident rules (9–13) — a node that's auto-recovering for the
3rd time this week should be flagged as a pattern, not silently
classified as `Monitor — first attempt`.

| # | Signal pattern | Verdict |
|---|---|---|
| 1 | **Audit mode** AND no fault events in the 4-hour window AND no `Monitor` fault chain still open | **Suppress — periodic audit, no open incidents** (skip Phase 4 entirely or emit a minimal record; nothing actionable) |
| 2 | A previously-`Monitor` fault chain now shows the affected instance(s) back in `Running` AND no new HMA detection for that instance in the last 30 min AND cluster status is `InService` | **Resolved — auto-recovery succeeded** (operator gets a closure email; include the original detection time and total elapsed) |
| 3 | Trigger detail-type is `Cluster Event` with `EventLevel=Info` and the timeline shows no node-health activity | **Suppress** |
| 4 | Cluster status is `Failed` or `RollingBack` | **Escalate** (cluster-level) |
| 5 | `NodeRecovery=None` on the cluster AND a node has been marked `Action:*` / `UnschedulablePending*` AND no replacement has started within 5 minutes | **Escalate** — auto-recovery is off; operator must trigger replacement |
| 6 | `signature_count_7d[<signature>] ≥ 3` for ANY signature — same fault content (Description + FailureMessage) has driven ≥3 replacements on the same InstanceGroup in the last 7 days (auto-recovery may be succeeding each time). Because signatures are the full concatenated content, this fires cleanly for repeated same-cause faults (e.g. 3× capacity errors for the same instance type) but does NOT over-merge distinct causes. | **Escalate — recurring fault pattern**: HyperPod is repairing the symptom, but the underlying cause hasn't gone away. Include the signature (first ~200 chars) and the timestamps of all prior occurrences. Recommend operator actions appropriate to the fault content (vendor-exclusion for hardware Xid faults; LCS bug for lifecycle-script failures; capacity request or IG-move for capacity failures; VPC/SG review for EFA health-check failures). Infer the appropriate action class from the FailureMessage content, not from a hard-coded category enum. |
| 7 | `replacements_24h_total ≥ 5` — five or more replacements anywhere in the cluster within 24 hours | **Escalate — fleet-wide instability**: the rate of node churn is abnormal regardless of individual root causes. |
| 8 | `replacements_7d_by_group[<ig>] ≥ 5` — five or more replacements on the same InstanceGroup in 7 days | **Escalate — instance-group instability**: the affected IG (which may be a specific SKU or topology placement) is failing more often than the rest of the cluster. |
| 9 | Exactly one replacement attempt in flight, started within the last 30 minutes, no prior failure in the chain | **Monitor — first attempt** (next re-check in 30 min via scheduled audit) |
| 10 | Multiple replacement attempts in the chain, total elapsed since the first failure ≤ 90 minutes, the most recent attempt is *Running* or *Started* (not yet failed) | **Monitor — elevated** (retry in progress, watch closely) |
| 11 | Multiple replacement attempts, total elapsed > 90 minutes, no successful `Running` transition, AND no new attempt started within the last 30 minutes | **Escalate** — retry chain is stuck |
| 12 | Node was in `Failed` state AND no new replacement attempt has started within the last 30 minutes AND total time in failing chain > 60 minutes | **Escalate** — HyperPod has given up |
| 13 | Instance id from the trigger event is missing from `list-cluster-nodes` AND `list-cluster-events` shows no new attempt for the last 30 minutes AND the most recent attempt failed | **Escalate** — instance vanished, no retry |
| 14 | HMA detection event present but no corresponding `Action:*` / replacement event in the timeline within 10 minutes | **Escalate** — HMA fired without escalating; investigate why (mismatch in node-recovery config, signal didn't classify) |
| 15 | None of the above match | **Monitor — uncategorized** (include the full timeline; flag for review) |

> **Duplicate / stale-audit suppression is handled upstream**, not by
> this skill: the `hyperpod-incident-triage` skill LINKs/SKIPs
> duplicate incident events and concurrent audits, and the
> periodic-audit Lambda only invokes an investigation when a real issue
> is present. The former RCA "stale-evidence" rules (3 / 3b) were
> removed and are not reintroduced.

**Rule 2 (`Resolved`) closes the loop on prior `Monitor` verdicts.**
A previous `Monitor` verdict promised a re-check; this rule provides
that re-check via the scheduled audit. The verdict description
should explicitly say "Incident detected at <T0> is now resolved.
Total auto-recovery time: <duration>. No operator action required."
This is the closure email the operator needs.

**Rule 1 (`Suppress — periodic audit, no open incidents`) is the no-op
case.** When the scheduled audit fires on a healthy cluster, emit a
single minimal record acknowledging the audit ran. The email notifier
filters `Suppress` verdicts so no email is sent.

**Recurring-pattern verdicts (6–8) are `Escalate` even when the
individual incident is auto-recovering correctly.** The reasoning is
in your operational goals: HyperPod's resiliency repairs the symptom,
but a 3× recurrence of a categorized fault on the same IG is an
underlying cause (vendor / capacity pool for `gpu-xid`, code bug for
`lifecycle-script-failed`, pool exhaustion for `capacity-insufficient`,
etc.) that auto-recovery can't fix. The verdict explanation should
explicitly say "this incident is auto-recovering, but the pattern
across the last <N> days warrants human attention" so operators
understand the verdict isn't about *this* incident's resolution.

**Time budgets are not hardcoded constants — they encode the
mental-model doc's "How long things take" section.** A single replace
takes 20–30 min; two attempts plus a slack gap = ~90 min. Don't change
these without updating the mental-model doc first.

### Phase 4 — Report (using DevOps Agent's native schema)

DevOps Agent's investigation output is structured: the terminal tool
is `write_final_investigation_report`, and the agent emits `symptom`
and `finding` records along the way that get serialized into the
final report. **Author the verdict and timeline into that schema
directly — do NOT invent a separate four-section markdown format**
(it would be at the agent's mercy during serialization and may not
survive into the final report).

The agent's schema supports these record types:

| Record type | Used for |
|---|---|
| `symptom` | Observable state the operator would notice (a node went unhealthy, an investigation was triggered, the cluster is in `Failed` state). First-class; survives serialization with title + description verbatim. |
| `finding` with `finding_type: "root_cause"` | An identified hardware/software root cause that cascades to one or more symptoms. |
| `finding` with `finding_type: "cause"` | An intermediate cause linking root_cause to symptoms. |
| `finding` with `finding_type: "hypothesis"` | An unverified explanation. |
| `investigation_gaps[]` | What the agent could not verify and would need the operator to confirm. |

**Emit exactly these records, in this order:**

1. **First `symptom`** — titled
   `Triage verdict: <verdict-name> :: <signature-set>` where
   `<verdict-name>` is one of `Suppress`, `Monitor — first attempt`,
   `Monitor — elevated`, `Monitor — uncategorized`, `Resolved — auto-recovery succeeded`,
   or `Escalate — <reason>`. **The `<signature-set>` suffix is
   required for any verdict that names specific fault evidence**
   (all Escalate / Monitor / Resolved variants) — it's a stable,
   sorted-and-deduplicated representation of the `current_signature_set`
   computed in Phase 2d, formatted as a comma-separated list of the
   `<ig>:<content>` signature strings (per Phase 2b) that drove the
   verdict. Verdict titles use the full concatenated fault-content
   signature, NOT a hard-coded category name. Examples (line-wrapped
   here for readability; the actual titles are single-line):

   - `Triage verdict: Escalate — recurring fault pattern :: (worker4:Description: Failed to provision EC2 Instance ... FailureMessage: We currently do not have sufficient capacity to launch new ml.g5.8xlarge instances. Please try again.)`
   - `Triage verdict: Monitor — first attempt :: (worker2:Description: Instance i-XXXX is unhealthy. HyperPod Health Monitoring Agent (HMA) has detected fault type NvidiaGPUUnhealthy on this node ...)`
   - `Triage verdict: Resolved — auto-recovery succeeded :: (worker2:Description: Instance i-XXXX is unhealthy ...)`
   - `Triage verdict: Escalate — fleet-wide instability :: (worker2:..., worker3:..., worker4:...)`
   - `Triage verdict: Suppress — periodic audit, no open incidents` (no suffix when there's no signature set)

   **Purpose**: the DevOps Agent platform's automatic task-dedup uses
   title equality within a ~30 minute window. Encoding the signature
   set in the title means a NEW signature (a new Xid type on the same
   IG, or the same Xid spreading to a new IG) produces a different
   title and the platform does NOT dedup it — the new investigation
   runs. Conversely, identical signature sets produce identical titles
   and platform dedup quietly absorbs the duplicate audit.

   The description is the prose justification in the shape:

   ```
   Verdict: <name>

   Summary:
   <A single plain-English paragraph (2–4 sentences) that flows naturally and
   weaves together THREE things, in this order: (1) WHAT HAPPENED — the specific
   observed problem: fault type, affected instance group / node / instance type,
   and how many times / how long (never just the generic verdict category);
   (2) THE LIKELY CAUSE — the most probable root cause inferred from the
   FailureMessage / event content and cluster context (not a hardcoded category);
   (3) THE RECOMMENDED ACTION — concrete operator next step(s) appropriate to the
   cause. Write it as prose, NOT as labeled fields or bullets. Lead the paragraph
   with "what happened" so the first sentence works as a standalone headline.
   Example:
   "Repeated capacity errors are failing to provision ml.p5.48xlarge for instance
   group 'faulttest' on cluster my-slurm-cluster — 5 attempts in the last 40 minutes. The
   likely cause is that on-demand capacity for ml.p5.48xlarge is unavailable in
   this Availability Zone and the instance group is not backed by a training plan
   or reserved capacity, so every Continuous-Provisioning retry hits the same
   wall. Recommended: launch in an AZ/Region where you hold reserved capacity for
   this instance type or associate the group with a training plan; otherwise
   reduce the target count to stop the retry loop and request a capacity increase.">

   What HyperPod is doing right now:
   <one paragraph plain English summary referencing the timeline>

   Timeline (UTC):
   <the timeline reconstructed in Phase 2, one event per line, source-tagged>

   Most recent event at:
   <UTC ISO 8601 timestamp of the most recent fault event in the 4-hour window.
   Recommended for all verdicts that contain a non-empty signature set so
   operators reading the report can correlate with their own timelines. Example:
   2026-06-30T01:36:49Z>

   Next re-check:
   <only for Monitor verdicts: UTC timestamp 30 min from now if "first attempt",
   15 min from now if "elevated">

   Recommended actions (operator runs these):
   <only for Escalate verdicts: explicit operator-runnable remediation steps —
   see the per-verdict guidance below and the SSM GPU-UUID step for
   recurring-pattern verdicts. Omit this heading for Suppress / Monitor /
   Resolved verdicts.>
   ```

   **The `Summary:` paragraph is REQUIRED for every non-Suppress verdict.** Write
   it as ONE natural-language paragraph (2–4 sentences) that covers, in order,
   what happened → the likely cause → the recommended action. Do NOT use labeled
   fields, bullets, or JSON — just prose. Lead with "what happened" so the first
   sentence stands alone as a headline (the email notifier uses the paragraph as
   the body summary and its first sentence as the subject). Name the *specific*
   problem (instance type, instance group, counts, AZ) — never restate the
   generic verdict category. Do NOT put `[direct]`/`[proxy]`/`[unverified]`
   confidence tags in this paragraph; those belong on the detailed evidence
   claims later in the description. Suppress verdicts may omit it (no email).

   `related_resources` on this symptom: `["HyperPod cluster <name>"]`
   plus any affected instance IDs.

2. **Additional `symptom` records** — one per observable failure
   condition (e.g. "Instance i-xxxx marked unhealthy with
   NvidiaGPUUnhealthy"). These are the per-resource symptoms the
   agent would naturally produce; keep them as separate records so
   they cross-link with `cascades_to` from the findings.

3. **`finding` records** as appropriate (root cause, intermediate
   cause, hypothesis). Cross-link via `cascades_to: [<symptom-id>]`.

4. **`investigation_gaps[]`** — anything Phase 1 couldn't reach.
   Examples: `list-cluster-events` unavailable on a Slurm cluster
   without Continuous Provisioning; HMA CloudWatch log group not
   yet populated; no SSM access from the agent (always include this
   one with a note that operator can confirm via the suggested SSM
   command).

   **Emit an `investigation_gap` when `DescribeClusterEvent` returned
   no FailureMessage for any Error/Warn event** (i.e. the API returned
   `EventDetails.EventDetails.EventMetadata.Instance` as null or with
   only `NodeLogicalId`). Title: `FailureMessage missing from N event(s)`.
   Description: list the affected `EventId`s + timestamps. This surfaces
   the known HyperPod bug where FailureMessage is populated in the API
   response for capacity errors but omitted for other fault categories —
   operators reviewing the report should escalate to the HyperPod team
   with those EventIds as evidence.

5. **`write_final_investigation_report`** — listing the verdict
   symptom FIRST in `symptoms[]`, then the per-resource symptoms.
   Findings listed under their `finding_type` arrays (`root_cause[]`,
   `cause[]`, `hypothesis[]`). `investigation_gaps[]` populated from
   step 4.

### CRITICAL: the FIRST symptom is the verdict symptom

Downstream automation (email notifier, dashboards, dedup) keys off the
FIRST symptom's title matching `Triage verdict: <name> :: <signature>`.
If you emit a descriptively-titled symptom first (e.g.
`"worker1 lifecycle script execution failures across multiple nodes on my-cluster"`)
the verdict is invisible to the pipeline and no email is sent.

This has been observed to fail in RCA runs. **The first
`symptom` record you emit MUST have `title` beginning with
`Triage verdict:`.** Descriptive titles are for the *second* and later
symptom records, which capture per-resource observations.

### Few-shot examples of the first symptom record

Copy the shape and titles exactly. Substitute your investigation's
concrete data.

**Example A — Escalate for a recurring capacity fault:**

```
title: "Triage verdict: Escalate — recurring fault pattern :: (worker4:Description: Failed to provision EC2 Instance in Cluster prod-01 and InstanceGroup worker4. FailureMessage: We currently do not have sufficient capacity to launch new ml.g5.8xlarge instances. Please try again.)"

description: |
  Verdict: Escalate — recurring fault pattern

  Summary:
  Repeated capacity errors are failing to provision ml.g5.8xlarge for instance group "worker4" — 4 failed replacements in the last 24 hours. The likely cause is that on-demand capacity for ml.g5.8xlarge is unavailable in us-west-2 for this cluster and worker4 is not backed by a training plan or reserved capacity, so each Continuous-Provisioning retry hits the same capacity wall. Recommended: launch worker4 in an AZ/Region where you hold reserved capacity or a training plan for ml.g5.8xlarge, or switch to a SKU with availability (e.g. ml.g6.8xlarge); alternatively lower worker4's target count to stop the retry loop while you request a capacity increase.

  What HyperPod is doing right now:
  HyperPod is auto-retrying, but the same insufficient-capacity error has driven 4 replacements on worker4 in the last 24 hours. Continuous Provisioning is looping without progress; each new attempt hits the same on-demand capacity wall for ml.g5.8xlarge in us-west-2. Operator action is required — the pattern will not self-resolve.

  Timeline (UTC):
  2026-07-08T15:12:03Z  aws.sagemaker Cluster Event Error  Failed to provision EC2 Instance in worker4 (attempt 1)
  2026-07-08T15:31:47Z  aws.sagemaker Cluster Event Error  Failed to provision EC2 Instance in worker4 (attempt 2)
  2026-07-08T15:53:22Z  aws.sagemaker Cluster Event Error  Failed to provision EC2 Instance in worker4 (attempt 3)
  2026-07-08T16:14:59Z  aws.sagemaker Cluster Event Error  Failed to provision EC2 Instance in worker4 (attempt 4)

  Most recent event at:
  2026-07-08T16:14:59Z
```

**Example B — Escalate for coordinated lifecycle-script failures across multiple instances:**

```
title: "Triage verdict: Escalate — coordinated lifecycle-script failure :: (worker1:Description: Lifecycle scripts did not run successfully. Ensure the scripts exist in provided S3 path, are accessible, and run without errors.)"

description: |
  Verdict: Escalate — coordinated lifecycle-script failure

  Summary:
  Every new worker1 instance on cluster my-cluster is failing bootstrap with the same lifecycle-script (LCS) execution error — 6 instances failed across two waves at 2026-07-08T19:21Z and 19:32Z. The likely cause is a bug or missing/inaccessible file in the on_create script in the instance group's S3 path, so each freshly provisioned node hits the identical failure. Recommended: inspect the LCS log stream for one of the affected instances to find the failing command, then fix on_create.sh (or on_create_main.sh) in the cluster's S3 bucket; the retry loop clears on the next attempt once the script succeeds.

  What HyperPod is doing right now:
  HyperPod is retrying, but every new worker1 instance is failing bootstrap with the same LCS execution error. 3 instances (i-0aaaa1111bbbb2222, i-0cccc3333dddd4444, i-0eeee5555ffff6666) failed simultaneously at 2026-07-08T19:21Z, then 3 more at 2026-07-08T19:32Z with the same error. Continuous Provisioning + Automatic NodeRecovery will keep respawning these logical nodes at ~10-minute intervals until an operator fixes the LCS.

  Timeline (UTC):
  2026-07-08T19:20:01Z  aws.sagemaker Cluster Event Info  Instance lifecycle script execution for i-0aaaa1... has Started
  2026-07-08T19:20:05Z  aws.sagemaker Cluster Event Info  Instance lifecycle script execution for i-0cccc3... has Started
  2026-07-08T19:20:07Z  aws.sagemaker Cluster Event Info  Instance lifecycle script execution for i-0eeee5... has Started
  2026-07-08T19:21:02Z  aws.sagemaker Cluster Event Error Lifecycle scripts did not run successfully (i-0aaaa1...)
  2026-07-08T19:21:05Z  aws.sagemaker Cluster Event Error Lifecycle scripts did not run successfully (i-0cccc3...)
  2026-07-08T19:21:08Z  aws.sagemaker Cluster Event Error Lifecycle scripts did not run successfully (i-0eeee5...)
  2026-07-08T19:31:53Z  aws.sagemaker Cluster Event Error Lifecycle scripts did not run successfully (i-07777a...)
  2026-07-08T19:32:04Z  aws.sagemaker Cluster Event Error Lifecycle scripts did not run successfully (i-08888b...)
  2026-07-08T19:32:13Z  aws.sagemaker Cluster Event Error Lifecycle scripts did not run successfully (i-09999c...)

  Most recent event at:
  2026-07-08T19:32:13Z

  Recommended actions (operator runs these):
  Inspect the LCS log stream /aws/sagemaker/Clusters/my-cluster/<cluster-id>/LifecycleConfig/worker1/<instance-id> for any of the affected instances to identify the failing command. Fix on_create.sh (or on_create_main.sh) in s3://<your-lcs-bucket>. Once fixed the retry loop will clear on its next attempt.

related_resources: ["HyperPod cluster my-cluster", "i-0aaaa1111bbbb2222", "i-0cccc3333dddd4444", "i-0eeee5555ffff6666"]
```

**Example C — Monitor (first attempt in flight):**

```
title: "Triage verdict: Monitor — first attempt :: (worker2:Description: Instance i-0abcdef1234567890 is unhealthy. HyperPod Health Monitoring Agent (HMA) has detected fault type NvidiaGPUUnhealthy on this node and is unhealthy. Repair action: Replace.)"

description: |
  Verdict: Monitor — first attempt

  Summary:
  HMA flagged instance i-0abcdef1234567890 in instance group "worker2" as unhealthy (NvidiaGPUUnhealthy) and requested a Replace, and HyperPod has already started the first replacement attempt. The likely cause is a transient or isolated GPU fault on that single node, which HyperPod's automatic recovery is equipped to handle on its own. No operator action is needed right now — the replacement is expected to reach InService within 20-30 min; the next scheduled audit will re-check and emit a Resolved closure if it succeeds.

  What HyperPod is doing right now:
  HMA has flagged i-0abcdef1234567890 as unhealthy (NvidiaGPUUnhealthy) and requested Replace. HyperPod has started the replacement and this is the first attempt in the chain. Expected wall-clock: 20-30 min for the new instance to reach InService.

  Timeline (UTC):
  2026-07-08T18:45:12Z  aws.sagemaker Cluster Node Health Event  Instance i-0abcdef1234567890 unhealthy: NvidiaGPUUnhealthy
  2026-07-08T18:45:15Z  aws.sagemaker Cluster Event Info         Instance deletion is starting as part of instance replacement

  Most recent event at:
  2026-07-08T18:45:15Z

  Next re-check:
  2026-07-08T19:15:15Z
```

**Example D — Suppress (audit found nothing):**

```
title: "Triage verdict: Suppress — periodic audit, no open incidents"

description: |
  Verdict: Suppress — periodic audit, no open incidents

  What HyperPod is doing right now:
  Scheduled audit at 2026-07-08T19:00Z; scanned the last 4 hours of cluster events. No fault events, no open Monitor chains, cluster status InService, all instance groups at target. Nothing to investigate.
```

### Anti-example — do NOT do this

If your first symptom looks like this, downstream automation is broken:

```
title: "worker1 lifecycle script execution failures across multiple nodes on my-cluster"    ← WRONG: missing "Triage verdict:" prefix
description: "HyperPod cluster my-cluster emitted coordinated lifecycle-script execution failures..."
```

The content is fine as a *second* symptom record. But the FIRST symptom
must be the verdict.

**For `Suppress` verdict:** still emit the verdict symptom (so the
operator can audit what was suppressed and why), but skip per-resource
symptoms and findings — there's nothing to root-cause.

**For `Monitor` verdicts:** the verdict symptom's description tells
the human "no action needed; HyperPod is recovering; expected
completion by HH:MM UTC. You will be notified again only if the
situation changes." This is the key UX improvement — silence is bad;
"we're watching and here's why we're not alarming you" is good.

**For `Escalate` verdicts:** include explicit operator-runnable
remediation in the verdict symptom's description, under a
`Recommended actions (operator runs these):` heading. The agent
cannot execute these; the operator must.

**Confidence annotations:** for every material claim in the verdict
description, prefix with one of `[direct]` (observed via API/log
output the agent itself ran), `[proxy]` (inferred from HMA
classification or a correlated source), or `[unverified]` (would need
on-node SSM or AWS-internal data the agent can't reach). This
replaces the separate "Confidence" section.

**`[direct]` requires direct observation, not inference.** A claim
like "all three replacements landed on the same physical host" is
**not** `[direct]` even when three identical fault signatures are
observed — the customer surface (NodeId, InstanceId, ENI, K8s node
name) does not expose physical hardware identity. Such claims are
`[unverified]` at best. Mislabeling inference as `[direct]` is a
serious failure mode because operators trust the annotation to mean
"the agent saw this in the data."

**Hypothesis discipline for recurring-pattern verdicts
(rules 6–8).** When the verdict is one of `Escalate — recurring
fault pattern`, `Escalate — fleet-wide instability`, or
`Escalate — instance-group instability`, the verdict description
MUST enumerate at least **two** competing hypotheses for the root
cause, each labeled `[unverified]` or `[proxy]`, and each paired with
a discriminating operator action. Do not commit to a single root
cause without `[direct]` evidence. The required hypothesis classes
are:

1. **Software / workload** — the workload running on the IG triggers
   the fault on whatever GPU it lands on (NCCL pattern, driver / CUDA
   version, application code path). Discriminator: change the
   workload, or move the IG to a different node and see if the fault
   follows.
2. **Infrastructure path** — an EFA fabric path, leaf switch, or
   shared network resource surfaces as GPU-level errors on workloads
   that hit it. Discriminator: move the IG to a different subnet / AZ.
3. **Statistical hardware** — a bad batch of the same SKU is over-
   represented in the capacity pool. Discriminator: open an AWS
   Support case with the fault signature requesting hardware
   exclusion, or wait + retry later from a different time/pool.

The verdict should NOT include "every replacement is landing on the
same physical hardware" as a stated cause — that's the explanation
**operators are conditioned to expect from on-prem clusters**, but
on HyperPod the EC2 instance is owned by the service account, the
underlying physical host is not exposed on the customer surface, and
EC2 placement is non-deterministic per replacement. Read the
"Recurring fault signature does NOT prove physical-host affinity"
section in [references/hyperpod-mental-model.md](references/hyperpod-mental-model.md)
before authoring this part of the verdict.

**The GPU UUID check is the only way to confirm or refute
physical-host affinity, and it requires SSM (operator-only).** The
verdict's "Recommended actions" section MUST include this check as
an explicit operator step whenever a recurring-pattern verdict is
emitted (rules 6–8). The wording should be:

```
N. Verify or refute "same physical GPU" by capturing GPU UUIDs.
   The agent cannot run this check (requires SSM, which is outside
   the DevOps Agent permission guardrail). For each affected
   instance ID, run:

       aws ssm start-session \
         --target sagemaker-cluster:<cluster-id>_<group>-<instance-id> \
         --document-name AWS-StartNonInteractiveCommand \
         --parameters '{"command":["nvidia-smi -L"]}'

   Compare the UUID strings across the affected instances. If they
   match, the same physical GPU is being recycled — that elevates
   the "statistical hardware" hypothesis to [direct] evidence and
   strengthens the case for an AWS Support exclusion request. If
   they differ, "same physical hardware" is RULED OUT and the
   investigation should pivot to the software/workload and
   infrastructure-path hypotheses instead.
```

Replace `<cluster-id>`, `<group>`, `<instance-id>` with the actual
values from the affected instances in the timeline. List each
instance separately so the operator can run the commands in parallel.

## Inputs the skill expects from the trigger

The webhook payload built by the bridge Lambda passes these fields in
the investigation context:

- `data.metadata.clusterName` — HyperPod cluster name (required)
- `data.metadata.detailType` — `Cluster State Change` /
  `Cluster Node Health Event` / `Cluster Event`
- `data.originalEvent.detail.InstanceId` — affected instance (for node
  health events)
- `data.originalEvent.detail.EventDetails.InstanceGroupName` /
  `InstanceId` (for cluster events)

If the cluster name is missing, abort with "skill cannot run without a
HyperPod cluster name — check the webhook bridge's payload mapping."

## What this skill does NOT do

- **It does not SSH or run SSM on nodes.** The DevOps Agent permission
  guardrail blocks `ssm:StartSession` / `ssm:SendCommand`. When on-node
  evidence is needed (Xid in `dmesg`, DCGM counters, kubelet journal),
  the skill emits a recommendation that includes the exact SSM command
  for the operator to run, never executes it.
- **It does not replace nodes.** `batch-replace-cluster-nodes` is a
  state-changing call; the skill recommends it but never invokes it.
- **It does not modify Slurm or EKS state.** No `scontrol update`, no
  `kubectl cordon`, no label edits.

## Tools used (in-guardrail)

- `sagemaker:DescribeCluster`, `ListClusterNodes`, `DescribeClusterNode`,
  `ListClusterEvents`
- `logs:FilterLogEvents`, `logs:DescribeLogStreams` on
  `/aws/sagemaker/Clusters/*`
- `eks:DescribeCluster` (read-only) and `kubectl get/describe`
  (via the access entry created by `make eks-access`)
- `cloudwatch:GetMetricData` on `ClusterAgent` and
  `SagemakerHealthMonitoringAgent` namespaces — useful for time-series
  confirmation when individual log events are sparse
