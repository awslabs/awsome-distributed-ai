---
name: hyperpod-devops-agent-solution
description: How this Agent Space monitors SageMaker HyperPod — the design and intent of the HyperPod x DevOps Agent solution (event-driven webhook bridge, Lambda-gated periodic audit, triage/RCA skills, email notifications). Read this to understand WHY a HyperPod investigation was created and what the monitoring pipeline does. For the concrete resource/topology map (ARNs, IDs, log groups), see the understanding-agent-space skill instead.
metadata:
  version: "1.0.0"
  agent_types: ["GENERIC"]
---

# HyperPod x DevOps Agent — how this solution monitors the cluster

This Agent Space is wired to monitor one or more **SageMaker HyperPod**
clusters. This document explains the **design and intent** of that
monitoring so investigations reason correctly and operators (via chat)
can understand how their cluster is watched.

> Scope: this is the *how it works* doc. For the concrete resource map
> (what exists, ARNs, instance IDs, log groups, VPC/subnet/SG IDs), use
> the `understanding-agent-space` skill.

## Two independent detection paths

HyperPod problems reach DevOps Agent through **two** paths, by design:

1. **Event-driven webhook bridge (all HyperPod control-plane faults).**
   A Lambda subscribes to `aws.sagemaker` HyperPod EventBridge events
   (Cluster State Change, Node Health, Cluster Event), drops routine
   `Info`-level noise, and HMAC-POSTs the rest to the DevOps Agent
   generic webhook — which creates an investigation. This is how node
   health faults, capacity errors, lifecycle-script failures, and
   cluster state changes are detected, on **both EKS and Slurm**
   (Slurm requires Continuous Provisioning for the correct event shape).

2. **Periodic-audit Lambda (Kubernetes state only).**
   Some conditions are NOT in the HyperPod event stream — notably
   Kubernetes **CrashLoopBackOff** pods and **NotReady** nodes. A
   scheduled Lambda inspects those via the EKS API and **only invokes an
   investigation when a real issue is present** (thresholds are
   configurable). On a healthy cluster it posts nothing. On **Slurm**
   there is no Kubernetes to inspect, so this path does nothing except
   the heartbeat below.

**Implication for investigations:** if you are investigating a
"HyperPod periodic audit" task, the audit Lambda already detected a
concrete issue (or a control-plane fault arrived via the bridge). Treat
the reported issue as real and **confirm + explain it — don't assume it
was a routine poll**. The Lambda inlines what it found into the task
`description` string (one `- [tag] type on resource: detail` line per
issue) — the DevOps Agent platform preserves the top-level `description`
verbatim but flattens nested payload sub-objects, so the description
text, not a `data.metadata.*` field, is where the skill reads the
findings.

## Daily heartbeat (liveness)

Once per day the audit Lambda fires a **heartbeat** even when the
cluster is healthy, so operators can see the pipeline is alive. The
heartbeat is a **silent liveness signal**: it is visible in the console
and logs but is intentionally **not emailed**. Do not treat a heartbeat
as an incident.

## Triage → RCA → notification

- **Triage** (`hyperpod-incident-triage`, INCIDENT_TRIAGE) decides
  LINK / SKIP / PROCEED. It keeps *different* fault types on the *same*
  instance group as separate investigations (the default correlator
  would merge them) and skips concurrent periodic audits.
- **RCA** (`hyperpod-incident-rca`, INCIDENT_RCA) runs on PROCEED:
  reconstructs the timeline from `list-cluster-events`, cluster/node
  state, and HMA CloudWatch streams; classifies as
  **Suppress / Monitor / Escalate / Resolved**; and writes a verdict
  with a plain-English **Summary** (what happened → likely cause →
  recommended action).
- **Email notifier** sends one SES email per completed investigation
  (scoped to this Agent Space). It suppresses `Suppress` verdicts,
  no-finding investigations, and heartbeats, and leads the subject with
  the Summary's first sentence.

## What this solution does NOT do

- It does **not** poll for HyperPod control-plane faults — those are
  event-driven via the bridge. The audit only covers Kubernetes state.
- The DevOps Agent runtime **cannot open SSM sessions to nodes** (a
  fixed permission guardrail), so on-node signals (dmesg Xid lines, DCGM
  counters, EFA fabric errors, slurmctld logs) are reached only
  indirectly via HMA CloudWatch streams / K8s labels / control-plane
  events — not by shelling into the node.
