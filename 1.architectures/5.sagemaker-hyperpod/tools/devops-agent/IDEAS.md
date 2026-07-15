# Future ideas

Enhancement ideas beyond the current solution. None are built yet; each notes
what it would take and the main constraint to check first.

## Weekly digest for GPU-utilization + FinOps signals

Absolute-utilization thresholds (e.g. "average <80% over 7d") don't work as
incident triggers — real workloads on well-tuned code hit 60–80% MFU,
exploration/notebook work hits 20–40%, so an 80% floor would page ops for every
researcher iterating on a training script. This is a **FinOps / capacity-planning
signal**, not an SRE page.

Better shape: a separate **weekly digest** email (different recipients, different
tone) surfacing:

- Per-instance-group average GPU utilization over the trailing 7d.
- Sudden utilization drop on a previously-hot instance group (e.g. "worker4 was
  70%+ for 6 days, now <10% for 12h and no scale-down") — that delta signal *is*
  actionable and could go straight into the incident channel because it points at
  a stuck job or GPU fault.
- Replacement counts by instance group over 7d (already computed by the RCA
  skill's recurrence rules, currently only surfaced when it crosses the Escalate
  threshold).
- Optional: cost estimate delta week-over-week.

Prerequisites:

- **GPU metrics source.** Container Insights + DCGM exporter aren't on by default
  on HyperPod EKS. Verify what's in place before designing around it.
  Alternatively CloudWatch's built-in EC2 metrics (limited) or Amazon Managed
  Prometheus (see below).
- Delivery: a new EventBridge Scheduler rule (e.g. `cron(0 15 ? * MON *)`) →
  dedicated Lambda → SES email. Skips the DevOps Agent entirely — no investigation
  billed, just a report. Alternatively, use DevOps Agent's native scheduled agents
  if the report should live inside the same product.
- A threshold-based Escalate for the specific delta case can still live in the
  audit-mode skill; the digest handles the ambient reporting.

## Amazon Managed Prometheus (AMP) as a data source

HyperPod has an official AMP integration (the kube-prometheus stack ships DCGM
exporter, node-exporter, and kube-state-metrics into an AMP workspace). It would
be a natural data source for GPU utilization, ECC counters, NVLink stats, Xid
counts, and EFA fabric health.

**Constraint to check first:** the DevOps Agent permission guardrail (see
[IMPLEMENTATION.md](IMPLEMENTATION.md#permission-guardrail)) currently exposes only
the AMP *describe/list* actions, not the time-series *query* actions. So the agent
can enumerate AMP workspaces and see that HyperPod is publishing to one, but can't
query metric data ad hoc during an investigation. Confirm the current guardrail
allowlist before designing an investigation-time query path.

Two directions:

1. **Ad-hoc query during investigation** — only viable if/when the guardrail
   allows AMP query actions. Read-only, scoped by workspace ARN, no on-node blast
   radius.
2. **AMP → CloudWatch/EventBridge bridge** — buildable today. Amazon Managed
   Grafana / AMP alerts can fan out to SNS; route them through EventBridge → the
   webhook bridge → an investigation, the same shape as the SageMaker `Cluster
   Event` stream. This unlocks threshold-based triggers on Prometheus metrics
   (per-Xid-code counts, per-GPU thermal alerts, EFA error rate) that the
   SageMaker EventBridge stream doesn't expose, and complements the recurrence
   detection with sharper trigger signals.

## CloudWatch metrics as a data source

Same purpose as the AMP idea (GPU utilization and hardware-health signals for the
digest and for sharper triggers), but with a simpler permission story: CloudWatch
metric-read actions are in the DevOps Agent default access policy, so the agent
can query CloudWatch time-series data ad hoc during an investigation with no
guardrail changes.

The gating question is therefore not permissions but **what is actually published
to CloudWatch for a HyperPod cluster** — and by default that's thin on GPU/hardware
signal:

- **EC2 vended metrics** (always on, `AWS/EC2`): `CPUUtilization`, network, EBS,
  status checks. No GPU utilization, no ECC/Xid/thermal. Coarse, but a *sudden
  CPUUtilization collapse on a previously-busy instance* is a usable proxy for a
  stalled job.
- **GPU/accelerator metrics require an installed publisher** and are **not on by
  default on HyperPod EKS**: either Container Insights with the
  accelerated-compute add-on (GPU metrics in the `ContainerInsights` namespace) or
  the CloudWatch agent with NVIDIA support (`nvidia_smi_*` — utilization, memory,
  temperature, some ECC). DCGM's richer fabric/NVLink/per-Xid cardinality is more
  the AMP+DCGM story than CloudWatch.

Two directions, mirroring the AMP section:

1. **Ad-hoc query during investigation** — available today. Whether it's *useful*
   depends on whether the cluster has a GPU-metrics publisher installed. If it
   does, the RCA skill could pull per-instance GPU utilization / temperature /
   memory around the incident window to corroborate an HMA verdict. If it doesn't,
   the agent still gets EC2-level CPU/network as a weak proxy.
2. **CloudWatch alarm → SNS → EventBridge → webhook bridge** as a trigger source —
   identical plumbing to AMP's bridge path, sourced from CloudWatch alarms.

**CloudWatch vs. AMP, net:** CloudWatch is queryable today but thin on GPU/hardware
signal unless a publisher is installed, and can get expensive at high metric
cardinality. AMP carries the rich DCGM cardinality HyperPod's official integration
already ships, but its query path depends on the guardrail. They're complementary:
prefer CloudWatch for the buildable-now digest + trigger work; keep AMP in mind for
the deeper per-GPU cardinality CloudWatch can't provide. First concrete step for
either is the same as the digest prerequisite — enumerate what metric namespaces a
real HyperPod EKS cluster actually publishes (`ListMetrics`) before designing
around any of them.

## Agent Instructions as a home for triage rules

AWS DevOps Agent supports **Agent Instructions** (an `AGENTS.md` scoped to a
managed agent) in addition to Skills. The difference that matters here:

| | Skill (used now) | Agent Instructions |
|---|---|---|
| Injection | On demand — the agent matches the skill description to the task and *can skip* it | Always — injected into the system prompt every session |
| Triage scoping | `agent_types: ["INCIDENT_TRIAGE"]` | Scoped to the Incident triage managed agent |
| Format | Markdown or ZIP (+ resource files) | Markdown only, no frontmatter, no resources |
| Count | Many per space | One global + one per managed agent |

**Why it's attractive:** the whole point of the triage rules is *reliable
invocation*. A skill fires only if the agent chooses to apply it; Incident-triage
Agent Instructions are injected every session, which is strictly more reliable for
the "keep different fault types separate / concurrency-skip" rules. The current
triage skill is already concise declarative markdown, so it would drop into an
`AGENTS.md` almost verbatim.

**Why not adopted yet:** there is no programmatic (API/CLI/CloudFormation) path to
manage Agent Instructions — they're edited through the Operator Web App
(Knowledge → Instructions), which would be a manual per-Agent-Space step and break
the one-command-deploy goal. Revisit if a programmatic path becomes available. In
the meantime, an operator who wants maximum triage reliability can paste the triage
rules into Knowledge → Instructions → Incident triage manually.

Agent Instructions are also a natural home for **always-on standing guidance** that
isn't task-specific (e.g. "this Agent Space monitors a HyperPod cluster; treat
`sagemaker:cluster` resources as HyperPod, not generic SageMaker").
