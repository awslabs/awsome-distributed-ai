# Implementation notes

Design decisions behind the HyperPod × AWS DevOps Agent solution. Read
[README.md](README.md) first for the value story, architecture, and quick start —
this document is the "how it's built" companion for anyone changing the template,
the Lambdas, or the skills.

Contents:

- [What gets deployed (one stack)](#what-gets-deployed-one-stack)
- [Event → investigation payload mapping](#event--investigation-payload-mapping)
- [Knowledge: HyperPod skills in the Agent Space](#knowledge-hyperpod-skills-in-the-agent-space)
- [The `hyperpod-incident-*` skills — triage + RCA](#the-hyperpod-incident-skills--triage--rca)
- [The periodic audit](#the-periodic-audit)
- [How notifications work](#how-notifications-work)
- [DevOps Agent integration surfaces](#devops-agent-integration-surfaces)

## What gets deployed (one stack)

A single template ([deploy/hyperpod_devops_agent.template.yaml](deploy/hyperpod_devops_agent.template.yaml))
creates everything, using native resource types wherever possible plus two
custom resources for the two gaps CloudFormation can't cover natively:

| Component | Resources |
| --- | --- |
| **Foundation** | `AWS::IAM::Role` (Agent Space monitor role + Webapp role), `AWS::DevOpsAgent::AgentSpace` (with IAM operator app), `AWS::DevOpsAgent::Association` (AWS-monitor, topology discovery). |
| **EKS access** (EKS only) | `AWS::EKS::AccessEntry` granting the Agent Space role read-only `AmazonAIOpsAssistantPolicy` (cluster scope). Skipped for Slurm. |
| **Webhook** | `AWS::SecretsManager::Secret` + `Custom::WebhookProvisioner` — registers/associates the eventChannel and stashes the once-shown URL+HMAC secret. (No native `eventChannel` ServiceType, and the URL/secret aren't exposed via `Fn::GetAtt`, so a custom resource is required.) |
| **Webhook bridge** | Lambda + EventBridge rule (`source: aws.sagemaker`, the 3 HyperPod detail-types) → HMAC-signed POST to the webhook. |
| **Periodic audit** | Lambda + `AWS::Scheduler::Schedule` (15 min) + daily heartbeat schedule. Detects Kubernetes CrashLoop/NotReady (EKS) and fires only on a real issue; heartbeat-only on Slurm. |
| **Email notifier** | Lambda + EventBridge rule (`source: aws.aidevops`, scoped to this Agent Space) + S3 dedup-marker bucket → SES email. |
| **Skills** | `Custom::SkillUploader` uploads the triage + RCA skills (and any staged upstream skills) from the S3 assets bucket. |

Per-cluster resource names are made unique from a slug of the cluster name so
multiple clusters coexist in one account/region.

## Event → investigation payload mapping

| HyperPod detail-type | Investigation `priority` | Title | Description |
| --- | --- | --- | --- |
| Cluster State Change | `HIGH` for `Failed`/`RollingBack`, `LOW` for `Updating`/`Deleting`, else `MEDIUM` | `HyperPod cluster state: {name} -> {status}` | Includes instance-group counts. |
| Node Health Event | `HIGH` if status `Unhealthy`/`Degraded`, else `MEDIUM` | `HyperPod node health: {cluster}/{instance} -> {status}` | Includes `HealthStatusReason`, `RepairAction`, `Recommendation`. |
| Cluster Event | `MEDIUM` | `HyperPod cluster event: {cluster} / {resourceType}` | Includes `Description`, `InstanceGroupName`, `InstanceId`. |

`data.originalEvent` always carries the unmodified EventBridge event so the agent can dig in.

### Noise filter

HyperPod emits many `Info`-level "Cluster Event" entries during routine
operations (a single scale-up can produce 20+: update-started notices, access-entry
updates, per-node provisioning notices, etc.). The Lambda drops events whose
`detail.EventDetails.EventLevel` matches `WEBHOOK_DROP_EVENT_LEVELS`
(default: `Info`). Without this filter a single scale-up would consume many
investigations against the monthly quota.

`DropEventLevels` (default `Info`) and `ClusterFilter` (default: only this
stack's `HyperPodClusterName` — empty = forward all clusters) are CloudFormation
parameters. To change either after deployment, set them in `params.json` and
re-run `make deploy`:

```json
{ "DropEventLevels": "Info,Debug", "ClusterFilter": "cluster-a,cluster-b" }
```

Set `WEBHOOK_LOG_FULL_EVENT=true` to log the full EventBridge envelope per invocation (useful when discovering new event shapes; off by default).

`Cluster State Change` and `Node Health Event` payloads don't carry `EventLevel`, so they're never dropped by this filter — they should always trigger investigations.

## Knowledge: HyperPod skills in the Agent Space

DevOps Agent does not understand HyperPod out of the box — a HyperPod cluster is
"a SageMaker resource" to its topology engine, not a composition of EKS + EC2 +
FSx + lifecycle scripts. Two kinds of skill teach it the mapping:

1. **Our two skills**:
   - **`hyperpod-incident-triage`** (INCIDENT_TRIAGE) — runs at the triage stage
     and decides LINKED / SKIPPED / PROCEED. See
     [skills/hyperpod-incident-triage/SKILL.md](skills/hyperpod-incident-triage/SKILL.md).
     It instructs the triage agent to keep distinct fault types on the same
     instance group separate rather than merging them.
   - **`hyperpod-incident-rca`** (INCIDENT_RCA) — runs after the triage skill
     produces PROCEED. Reads `describe-cluster`, `list-cluster-nodes`,
     `list-cluster-events`, and HMA CloudWatch streams; reconstructs a timeline;
     classifies as Suppress / Monitor / Escalate / Resolved against time budgets
     derived from the [HyperPod mental model](docs/hyperpod-mental-model.md).
     Bundles the mental-model doc as a reference. See
     [skills/hyperpod-incident-rca/SKILL.md](skills/hyperpod-incident-rca/SKILL.md).

2. **Curated upstream skills from [awslabs/agent-plugins](https://github.com/awslabs/agent-plugins)**
   — supporting reference. `make import-upstream-skills` imports a **curated
   subset** of the `hyperpod-*` skills (see "Skill curation" below). Use the
   `SKILLS=...` env var to import a different subset. Subsequent runs `git pull`
   upstream and re-upload.

### Authoring a triage skill that takes effect

An `INCIDENT_TRIAGE` skill is consulted by the triage agent as natural-language
*instructions*, not as code it executes deterministically. Concise declarative
rules are followed far more reliably than a long procedural algorithm. When
authoring one:

- State a **few plain rules** the agent should follow (link/skip/proceed
  *criteria*), not an algorithm to run.
- Describe **identity in words** ("same instance group AND same fault text"),
  not as data structures the agent must build and compare.
- Keep it short. Long, procedural skills read as reference material the agent may
  or may not apply, rather than triage directives.

There is no customer-facing log/API that reports the triage decision or which
skill produced it — the only signals are the EventBridge `Investigation Linked` /
`Skipped` events and `list-executions`. Verify behavior empirically by firing
correlated synthetic events and observing the resulting LINK/SKIP/PROCEED.

### Skill curation: which upstream skills get uploaded

The default upload list excludes upstream skills whose entire procedure depends on
SSM — those are unreachable inside the DevOps Agent permission guardrail (see
"Permission guardrail" below), and loading them gives the agent instructions it
can't execute:

| Upstream skill | Default import? | Why |
| --- | --- | --- |
| `hyperpod-cluster-debugger` | yes | Cluster-level API + kubectl portions work in-guardrail |
| `hyperpod-node-debugger` | yes | API + kubectl node-state portions work in-guardrail |
| `hyperpod-nccl` | yes | API portions only — `kubectl logs`, training-op CRDs |
| `hyperpod-performance-debugger` | yes | API portions only — CloudWatch + EKS topology |
| `hyperpod-slurm-debugger` | **no** | Needs SSM to controller |
| `hyperpod-issue-report` | **no** | Whole skill is on-node collection |
| `hyperpod-version-checker` | **no** | Whole skill is on-node version reads |
| `hyperpod-ssm` | **no** | The SSM driver itself |

Override with `SKILLS='hyperpod-nccl hyperpod-node-debugger' make import-upstream-skills` to import a custom subset. Override `UPSTREAM_REF` to pin to a specific commit/branch/tag.

### Authoring your own skill (custom detection rules)

Drop a directory under `skills/` containing a `SKILL.md` (frontmatter `name:`,
`description:`, and `metadata.agent_types:` as a list) plus optional
`references/` markdown files, then run `make deploy`. `prepare_deployment.py
sync-skills` zips every skill dir under `skills/` (parsing `agent_types` from the
frontmatter — set it to `["INCIDENT_TRIAGE", "INCIDENT_RCA"]` for investigation
skills) and the `SkillUploader` custom resource create/updates the asset by name.
The skill's `description:` field determines when the agent loads it during
investigations. Removing a skill dir + re-deploying deletes the asset.

This is the extension point for your own detection rules — e.g. escalate on a Pod
in CrashLoopBackOff beyond a threshold, or on an instance group whose GPU
resource allocation is chronically low. Encode the rule as plain-English criteria
in a new skill (see the authoring lessons above) and let the agent apply it
during triage or RCA.

### Permission guardrail

The upstream skills were authored for Claude Code / Codex runtimes that can
execute shell directly on HyperPod nodes. DevOps Agent's runtime cannot, and
adding IAM permissions does not change that: DevOps Agent applies a
**permission guardrail** — a session policy set at AssumeRole time — so the
effective permissions are the *intersection* of the IAM role and the guardrail.

The guardrail contains everything in the `AIDevOpsAgentAccessPolicy` managed
policy (the default read-only set) plus a closed allowlist of opt-in permissions
(e.g. `s3:GetObject`/`s3:ListBucket`, `athena:*Query*`, `directconnect:Describe*`,
`glue:GetPartitions`, `kms:Decrypt`). Actions outside that intersection are not
available to the agent regardless of what you grant on the role. See the DevOps
Agent User Guide, "Understanding permission guardrails," for the authoritative
list.

Practical consequences for HyperPod:

- **No on-node access.** `ssm:StartSession` / `ssm:SendCommand` are not in the
  allowlist, so the agent cannot read on-node signals directly (`dmesg`/Xid, DCGM
  ECC counters, EFA fabric counters, kubelet journal, `slurmctld.log`). The RCA
  skill reasons from *proxies* instead: HMA-generated CloudWatch log streams,
  Kubernetes node labels (`sagemaker.amazonaws.com/fault-types`, etc.),
  `list-cluster-events`, and `kubectl describe node`. For HMA-classified faults
  (the common case that drives auto-triggered investigations) the proxy path
  carries the classification the agent needs.
- **`s3:GetObject` is in the allowlist** and takes effect once granted on the
  role — used to read lifecycle-script content (`on_create.sh`) during RCA.
- **`kubectl` works** via the EKS access entry (read-only), independent of the
  IAM guardrail — the on-cluster (not on-node) path.

If a capability isn't in `AIDevOpsAgentAccessPolicy` or the opt-in allowlist,
design a side-channel: write the data to CloudWatch Logs or S3, which the
guardrail can read.

Two skill types appear in the Agent Space's Skills list:
- **USER** — what this repo authors. Edits are deliberate.
- **LEARNED** — what the agent generates about your environment over time (e.g.
  `understanding-agent-space`). Don't edit these.

## The `hyperpod-incident-*` skills — triage + RCA

Triage and RCA are two skills, but the *investigation* logic is a single RCA
skill rather than split further, because distinguishing "still retrying" from
"needs an operator" requires the full timeline at once. A single failed instance
can leave `list-cluster-nodes` between retry attempts, and HyperPod may auto-retry
from `Failed` status — neither is a terminal signal on its own. The RCA skill
correlates `describe-cluster`, `list-cluster-nodes`, `list-cluster-events` (the
canonical record of replacement attempts, including failed ones; available on EKS
and on Slurm with Continuous Provisioning), and HMA CloudWatch streams.

The skill classifies each event into one of these verdicts:

| Verdict | Meaning |
| --- | --- |
| `Suppress` | Routine `Info`-level activity or a periodic audit with no open incidents; no investigation email produced. |
| `Monitor — first attempt` | Recovery in flight, first attempt, within the 30 min budget. Next re-check timestamp included. |
| `Monitor — elevated` | Multiple retry attempts in flight, total elapsed ≤ 90 min. Recovery may still succeed; user is notified so they're not surprised. |
| `Escalate` | Recovery has passed its expected window (no new attempt within 30 min, total elapsed > budget), HyperPod reports `Failed` with no new attempt, an instance left with no retry, or a recurring/fleet-wide pattern crossed threshold. Operator action suggested. |
| `Resolved — auto-recovery succeeded` | A previously-`Monitor` fault chain is back in `Running`/`InService` with no new HMA detection; closes the loop with a "resolved" email. |

`Monitor` verdicts are not silent — the email tells the user "HyperPod is
auto-recovering, expected completion by HH:MM UTC, you'll be notified again only
if the situation changes." The follow-up only fires if the verdict transitions on
a later event.

Time budgets in the skill encode the "How long things take" table in the
[HyperPod mental model](docs/hyperpod-mental-model.md). Update the mental-model
doc first if the budgets need to change.

### First symptom must be the verdict symptom (skill ↔ notifier contract)

The email notifier's subject-line headline and the platform verdict-title dedup
both key off the FIRST symptom record having a title that begins with
`Triage verdict:`. A descriptive first-symptom title breaks both — the notifier
falls back to the raw task title and dedup can't recognize the signature set. The
RCA skill's [CRITICAL: the FIRST symptom is the verdict symptom](skills/hyperpod-incident-rca/SKILL.md)
section pins this down with few-shot examples and an anti-example. The notifier
still degrades gracefully — it picks the first symptom's title or the task title
if no verdict-prefixed symptom exists — but dedup will miss and downstream
automation loses the verdict category.

### Recurrence detection

Webhook-triggered investigations are single-shot: the agent writes a report and
exits. On its own that misses a statistically recurring pattern — each occurrence
may auto-resolve correctly, but a recurring Xid signature across 3+ replacements
on the same instance group in a week is worth surfacing to an operator as one
signal. Phase 2b of the RCA skill computes recurrence statistics over the 7-day
`list-cluster-events` window, and Phase 3 rules fire on threshold crossings:

- `xid_signature_count_7d[(<xid>, <ig>)] ≥ 3` → `Escalate — recurring hardware fault pattern`
- `replacements_24h_total ≥ 5` → `Escalate — fleet-wide instability`
- `replacements_7d_by_group[<ig>] ≥ 5` → `Escalate — instance-group instability`

## The periodic audit

**Division of labor:** HyperPod control-plane conditions (node health, capacity
errors, lifecycle-script failures, cluster state changes) are handled
**event-driven by the webhook bridge**. The periodic audit covers only what the
event stream **cannot**: Kubernetes Pod/Node state, which is not in the HyperPod
event stream.

The audit is an `AWS::Scheduler::Schedule` → Lambda in the same stack. The Lambda
inspects Kubernetes state itself (CrashLoopBackOff / NotReady) and POSTs the
webhook **only when a real issue is found**, plus one daily heartbeat — rather
than firing every 15 min and relying on the skill to suppress. On a healthy
cluster nothing is POSTed, so no investigation runs. On Slurm the audit has no
Kubernetes to poll, so it fires only the heartbeat. Verdicts go through the
email path. The audit is documented in
[README.md](README.md#periodic-audit).

### Kubernetes-state checks

Two rules, both gated behind `K8sChecksEnabled=true` (default):

| Rule | Escalates when | Configurable via |
|---|---|---|
| **CrashLoopBackOff loop** | A container's `restartCount` reaches the floor and its last crash is recent (K8s exposes no loop-start in a snapshot, so `restartCount` is the crash-evidence signal) | `CrashLoopMinRestarts` (default 5) + `CrashLoopRecencyMinutes` (default 15) |
| **NotReady node percentage** | ≥ percent of nodes have been NotReady for ≥ duration | `NotReadyNodePercentThreshold` (default 10) + `NotReadyDurationMinutes` (default 15) |

Namespace handling uses **two plain lists**, no DSL. The Lambda validates at cold
start that they do not overlap; overlapping deployments fail the audit invocation
with a clear error message.

| Parameter | Default | Semantics |
|---|---|---|
| `IgnoreNamespaces` | `kube-public,kube-node-lease` | Pods here are skipped entirely — no verdict, no `kubectl` inspection. |
| `SystemNamespaces` | `kube-system,aws-hyperpod,amazon-cloudwatch` | CrashLoop verdicts on these are tagged `system-workload`. Downstream email routing can page the platform team differently from customer-workload verdicts. |
| everything else | — | Tagged `customer-workload`. |

Two lists rather than a run-time DSL keeps the skill's job to English-language
reasoning over already-structured input — classification is plain set-membership
lookups on already-resolved lists, with the structure pushed into the Lambda.

### Anti-spam: audit titles are stable + issue-descriptive

A naive periodic audit could re-notify the same evidence every cycle. The
solution keeps audit investigation titles issue-descriptive and timestamp-free
(e.g. `HyperPod <cluster>: CrashLoopBackOff (<ns>/<pod>)`), so a recurring issue
produces an identical title and the platform triage stage (plus the
`hyperpod-incident-triage` skill) LINKs or SKIPs the repeat instead of emailing
every cycle. A genuinely new signature (a new Xid type, or the same Xid spreading
to a new instance group) produces a different title and re-notifies. On top of
that, the RCA skill emits `Suppress — periodic audit, evidence is stale` when the
signature set is unchanged, and the email notifier drops `Triage verdict:
Suppress —*` from delivery.

## How notifications work

Three channels stack:

1. **Email (via SES)** — part of the stack. EventBridge rule on
   `source: aws.aidevops`, detail-type prefix `Investigation`, **scoped to this
   stack's Agent Space** (`detail.metadata.agent_space_id`, so multiple clusters
   don't cross-notify) → Lambda → `ses:SendEmail` to the configured recipients. By
   default sends on `Investigation Completed` only (one email per lifecycle);
   `Created` / `Updated` / `Linked` events are ignored to avoid mid-flight spam.
   Override with `EMAIL_DETAIL_TYPES`.
   - **Body composition reads the full journal.** The Lambda calls
     `aidevops:GetBacklogTask` + `aidevops:ListJournalRecords` and renders
     symptoms, findings, and investigation_gaps directly. It does not rely on
     parsing a single verdict-title string, so it degrades gracefully when the RCA
     skill drifts.
   - **Dedup is S3-marker-based, keyed by `execution_id`.** Before every send the
     Lambda does a `HeadObject` against
     `s3://hpda-markers-<slug>-<account>-<region>/emailed/<execution_id>`. If the
     marker exists, the event is dropped. The marker is written *after*
     `ses:SendEmail` returns a MessageId. This defends against the platform
     re-emitting `Investigation Completed` for the same execution. Marker objects
     auto-expire (30 days by default; `MarkerExpirationDays` CFN parameter).
   - **Skip filters (in order):** detail-type allowlist → S3 marker → heartbeat
     all-clear (the "no open issues" daily-liveness payload, never emailed; a
     heartbeat run that *detects a real problem* is built as an issues payload
     and is NOT skipped here) → Suppress-verdict detection → no-actionable-content
     (zero findings AND no verdict symptom). `FORCE_SEND=true` on the stack
     bypasses all of them.
   - SES sender must be verified in `$REGION`. If SES is in sandbox mode, every
     recipient must also be verified.
   - The IAM policy on the Lambda restricts `ses:SendEmail` to the configured
     `EMAIL_SENDER` via the `ses:FromAddress` condition. S3 read/write is scoped to
     the marker bucket only.
2. **DevOps Agent web app** — every investigation is visible at the Agent Space
   console URL in `make stack-outputs`.
3. **Slack / ServiceNow / PagerDuty / Microsoft Teams** — configure once in the
   Agent Space console. The same `aws.aidevops` event stream the email notifier
   listens on is available for any additional fan-out.

## DevOps Agent integration surfaces

DevOps Agent is **alert-driven, not polling**. Investigations only run when
something triggers them: a webhook, a ticket integration, a third-party SaaS
hook, or a manual click. Out of the box it has no concept of "watch HyperPod" —
that's why the webhook bridge and the periodic audit both exist.

| Surface | Direction | Used here? |
| --- | --- | --- |
| **EKS access entry** | Pull | Yes — read-only `kubectl` against the underlying EKS cluster. |
| **Generic webhook** (HMAC) | Into the agent | Yes — webhook bridge for live HyperPod events + periodic-audit Lambda for scheduled audit-mode investigations. |
| **EventBridge `aws.aidevops`** | From the agent | Yes — the email notifier listens on `Investigation` detail-types and sends SES email. |
| **`AWS::Scheduler::Schedule`** → webhook | Into the agent (scheduled) | Yes — the periodic audit synthesizes a webhook event so the scheduled run produces a proper `INVESTIGATION` task with AWS API access and user skills mounted. |
| **Skills** (asset API) | Inside the agent | Yes — `hyperpod-incident-triage` and `hyperpod-incident-rca` run at the triage and RCA stages; a curated subset of upstream `hyperpod-*` skills is imported as supporting reference. |

Operating notes:

- **Agent Space region** — Agent Space is available in a fixed set of regions;
  check `aidevops.<region>.amazonaws.com` resolves before running. One Agent Space
  with an `Aws` monitor association discovers resources across all regions of the
  account; the Agent Space resource itself is single-region.
- **Quotas (per the UG)**: 100 agent spaces / region, 3 concurrent investigations
  / space (adjustable), 10 concurrent on-demand invocations / space.
- **EKS access prerequisite** — the underlying EKS cluster's `authenticationMode`
  must be `API` or `API_AND_CONFIG_MAP`. The setup script verifies this and aborts
  with the corrective `update-cluster-config` command if not.
