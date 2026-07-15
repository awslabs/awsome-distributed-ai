"""Periodic-audit Lambda for the HyperPod x DevOps Agent solution.

Fired on a schedule by EventBridge Scheduler. It inspects Kubernetes Pod/Node
state HERE (CrashLoopBackOff pods, NotReady nodes) and POSTs the DevOps Agent
webhook ONLY when a real issue is found. On a healthy cluster nothing is POSTed,
so no investigation runs and no cost is incurred. A separate daily "heartbeat"
schedule (event input {"trigger":"heartbeat"}) forces one POST per day so
operators can see the pipeline is alive.

Scope note: HyperPod control-plane faults (node health, capacity errors,
lifecycle-script failures, cluster state changes) are handled event-driven by the
webhook bridge — which reads the native EventLevel from the EventBridge event and
forwards the real FailureMessage verbatim. This audit deliberately does NOT
re-scan list-cluster-events; it only covers Kubernetes state, which is not in the
HyperPod event stream. On Slurm (no kubectl) the audit fires only the heartbeat.

Either way the POST is an HMAC-signed "periodic audit" event, identical in shape
to what the webhook bridge sends, so the RCA skill's audit-mode path handles it.
The payload carries data.metadata.detectedIssues so the RCA skill starts from
the finding instead of rediscovering it.

Cluster reads use the EKS API server directly (SigV4 bearer token via boto3 +
stdlib urllib) — no kubernetes client / Lambda layer. The Lambda's execution
role is granted read-only cluster access by its own AWS::EKS::AccessEntry.

Env vars:
  WEBHOOK_SECRET_ARN               Secrets Manager secret {"url","secret"} (shared with bridge)
  CLUSTER_NAME                     HyperPod cluster name (payload metadata)
  EKS_CLUSTER_NAME                 Underlying EKS cluster name (empty for Slurm — kubectl checks skipped)
  K8S_CHECKS_ENABLED               "true"/"false" — enable CrashLoop/NotReady checks
  CRASHLOOP_HOURS_THRESHOLD        Fire if a pod is CrashLoopBackOff longer than this (0 = any)
  NOT_READY_NODE_PERCENT_THRESHOLD Fire if >= this percent of nodes are NotReady (after duration gate)
  NOT_READY_DURATION_MINUTES       A node must be NotReady this long to count
  IGNORE_NAMESPACES                Comma-separated; pods here are skipped entirely
  SYSTEM_NAMESPACES                Comma-separated; CrashLoop here is tagged "system-workload"
"""
import base64
import datetime
import hashlib
import hmac
import json
import os
import ssl
import urllib.request
from urllib.error import HTTPError, URLError

import boto3
from botocore.signers import RequestSigner


_secrets_cache: dict[str, str] = {}
_eks_cache: dict[str, tuple[str, str]] = {}  # cluster -> (endpoint, ca_file_path)


# ----------------------------------------------------------------- config helpers

def _parse_namespace_list(env_name: str) -> list[str]:
    """Parse a comma-separated env var into a de-duplicated, trimmed list."""
    raw = os.environ.get(env_name, "")
    seen: dict[str, None] = {}
    for token in raw.split(","):
        tok = token.strip()
        if tok:
            seen[tok] = None
    return list(seen.keys())


def _k8s_checks_enabled() -> bool:
    return os.environ.get("K8S_CHECKS_ENABLED", "true").strip().lower() in ("true", "1", "yes", "on")


def _namespace_config() -> tuple[list[str], list[str]]:
    """Return (ignore, system) namespace lists; raise on overlap."""
    ignore = _parse_namespace_list("IGNORE_NAMESPACES")
    system = _parse_namespace_list("SYSTEM_NAMESPACES")
    overlap = sorted(set(ignore) & set(system))
    if overlap:
        raise ValueError(
            f"IGNORE_NAMESPACES and SYSTEM_NAMESPACES must not overlap; "
            f"conflicting entries: {overlap}. Fix the CFN parameters and redeploy."
        )
    return ignore, system


def _build_k8s_checks_block() -> dict | None:
    """The structured k8sChecks block echoed into the payload, or None if disabled."""
    if not _k8s_checks_enabled():
        return None
    ignore, system = _namespace_config()
    return {
        "enabled": True,
        "crashLoopHoursThreshold": int(os.environ.get("CRASHLOOP_HOURS_THRESHOLD", "4")),
        "notReadyNodePercentThreshold": int(os.environ.get("NOT_READY_NODE_PERCENT_THRESHOLD", "10")),
        "notReadyDurationMinutes": int(os.environ.get("NOT_READY_DURATION_MINUTES", "15")),
        "ignoreNamespaces": ignore,
        "systemNamespaces": system,
    }


# ----------------------------------------------------------------- kubernetes reads

def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _parse_k8s_time(ts: str) -> datetime.datetime | None:
    """Parse a K8s RFC3339 timestamp (e.g. 2026-07-10T12:00:00Z)."""
    if not ts:
        return None
    try:
        return datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _eks_token(cluster_name: str, region: str) -> str:
    """Generate the EKS bearer token (same scheme as `aws eks get-token`)."""
    session = boto3.session.Session()
    client = session.client("sts", region_name=region)
    signer = RequestSigner(
        client.meta.service_model.service_id,
        region,
        "sts",
        "v4",
        session.get_credentials(),
        session.events,
    )
    signed_url = signer.generate_presigned_url(
        {
            "method": "GET",
            "url": f"https://sts.{region}.amazonaws.com/?Action=GetCallerIdentity&Version=2011-06-15",
            "body": {},
            "headers": {"x-k8s-aws-id": cluster_name},
            "context": {},
        },
        region_name=region,
        expires_in=60,
        operation_name="",
    )
    return "k8s-aws-v1." + base64.urlsafe_b64encode(signed_url.encode()).decode().rstrip("=")


def _eks_endpoint_and_ca(eks_cluster_name: str, region: str) -> tuple[str, str]:
    """Return (api_endpoint, ca_file_path), cached across warm invocations."""
    if eks_cluster_name in _eks_cache:
        return _eks_cache[eks_cluster_name]
    desc = boto3.client("eks", region_name=region).describe_cluster(name=eks_cluster_name)["cluster"]
    endpoint = desc["endpoint"]
    ca_data = desc["certificateAuthority"]["data"]
    ca_path = f"/tmp/eks-ca-{eks_cluster_name}.pem"
    with open(ca_path, "wb") as f:
        f.write(base64.b64decode(ca_data))
    _eks_cache[eks_cluster_name] = (endpoint, ca_path)
    return endpoint, ca_path


def _k8s_get(path: str, eks_cluster_name: str, region: str) -> dict:
    """GET a Kubernetes API path and return the parsed JSON."""
    endpoint, ca_path = _eks_endpoint_and_ca(eks_cluster_name, region)
    token = _eks_token(eks_cluster_name, region)
    ctx = ssl.create_default_context(cafile=ca_path)
    req = urllib.request.Request(
        f"{endpoint}{path}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
        return json.loads(resp.read())


# ----------------------------------------------------------------- detection

def _detect_crashloops(eks_cluster_name: str, region: str, cfg: dict) -> list[dict]:
    """Pods in CrashLoopBackOff longer than the threshold (proxy: pod age)."""
    ignore = set(cfg["ignoreNamespaces"])
    system = set(cfg["systemNamespaces"])
    threshold_h = cfg["crashLoopHoursThreshold"]
    now = _now()
    issues: list[dict] = []
    pods = _k8s_get("/api/v1/pods", eks_cluster_name, region).get("items", [])
    for pod in pods:
        meta = pod.get("metadata", {})
        ns = meta.get("namespace", "")
        if ns in ignore:
            continue
        status = pod.get("status", {})
        start = _parse_k8s_time(status.get("startTime", "")) or _parse_k8s_time(
            meta.get("creationTimestamp", "")
        )
        age_h = (now - start).total_seconds() / 3600 if start else 0.0
        for cs in status.get("containerStatuses", []) or []:
            waiting = (cs.get("state", {}) or {}).get("waiting", {}) or {}
            if waiting.get("reason") == "CrashLoopBackOff" and (threshold_h == 0 or age_h >= threshold_h):
                issues.append({
                    "type": "CrashLoopBackOff",
                    "resource": f"{ns}/{meta.get('name')}:{cs.get('name')}",
                    "detail": f"restartCount={cs.get('restartCount')}, podAgeHours={age_h:.1f}, threshold={threshold_h}h",
                    "tag": "system-workload" if ns in system else "customer-workload",
                })
    return issues


def _detect_notready_nodes(eks_cluster_name: str, region: str, cfg: dict) -> list[dict]:
    """NotReady nodes past the duration gate, if they cross the percent threshold."""
    duration_min = cfg["notReadyDurationMinutes"]
    percent_threshold = cfg["notReadyNodePercentThreshold"]
    now = _now()
    nodes = _k8s_get("/api/v1/nodes", eks_cluster_name, region).get("items", [])
    total = len(nodes)
    if total == 0:
        return []
    not_ready: list[str] = []
    for node in nodes:
        name = node.get("metadata", {}).get("name", "")
        ready = None
        for cond in node.get("status", {}).get("conditions", []) or []:
            if cond.get("type") == "Ready":
                ready = cond
                break
        if ready is None or ready.get("status") == "True":
            continue
        since = _parse_k8s_time(ready.get("lastTransitionTime", ""))
        mins = (now - since).total_seconds() / 60 if since else duration_min + 1
        if mins >= duration_min:
            not_ready.append(name)
    if not not_ready:
        return []
    pct = 100.0 * len(not_ready) / total
    if pct < percent_threshold:
        return []
    return [{
        "type": "NotReadyNodes",
        "resource": ",".join(sorted(not_ready)),
        "detail": f"{len(not_ready)}/{total} nodes NotReady ({pct:.0f}%) >= {percent_threshold}% for >= {duration_min}min",
        "tag": "infrastructure",
    }]


def _detect_issues(cluster_name: str, eks_cluster_name: str, region: str) -> list[dict]:
    """Run all applicable detectors and return the combined issue list.

    NOTE: HyperPod control-plane fault events (node health, capacity errors,
    lifecycle-script failures, cluster state changes) are NOT detected here —
    they are already delivered event-driven by the webhook bridge, which reads
    the native `EventLevel` field from the EventBridge event and forwards the
    real FailureMessage without interpreting it. The periodic audit only covers
    what the event stream cannot: Kubernetes Pod/Node state (CrashLoopBackOff,
    NotReady), which is not in the HyperPod event stream. On Slurm (no kubectl)
    the audit therefore has nothing to poll — it fires only the daily heartbeat.
    """
    issues: list[dict] = []
    # kubectl-based checks require an EKS cluster + K8s checks enabled.
    if eks_cluster_name and _k8s_checks_enabled():
        cfg = _build_k8s_checks_block()
        try:
            issues.extend(_detect_crashloops(eks_cluster_name, region, cfg))
            issues.extend(_detect_notready_nodes(eks_cluster_name, region, cfg))
        except (HTTPError, URLError, ssl.SSLError) as e:
            # Don't silently swallow: a cluster-read failure could hide a real
            # issue. Surface it as an issue so an operator investigates the gap.
            body = ""
            if isinstance(e, HTTPError):
                try:
                    body = e.read().decode()[:500]
                except Exception:  # noqa: BLE001
                    body = "<unreadable>"
            print(f"kubectl read failed: {e!r} body={body}")
            issues.append({
                "type": "AuditReadFailure",
                "resource": eks_cluster_name,
                "detail": f"Could not read cluster state: {e!r}. Investigate audit access.",
                "tag": "infrastructure",
            })
    return issues


# ----------------------------------------------------------------- webhook

def _load_webhook_credentials() -> tuple[str, str]:
    if "url" in _secrets_cache and "secret" in _secrets_cache:
        return _secrets_cache["url"], _secrets_cache["secret"]
    secret_arn = os.environ["WEBHOOK_SECRET_ARN"]
    resp = boto3.client("secretsmanager").get_secret_value(SecretId=secret_arn)
    payload = json.loads(resp["SecretString"])
    _secrets_cache["url"] = payload["url"]
    _secrets_cache["secret"] = payload["secret"]
    return _secrets_cache["url"], _secrets_cache["secret"]


def _build_title(cluster_name: str, detected_issues: list[dict], heartbeat: bool) -> str:
    """A STABLE, issue-descriptive title.

    Intentionally has NO timestamp: the same ongoing issue must produce the same
    title on every audit so the DevOps Agent platform links the repeat
    investigations (one email per issue, not one per 15-min audit). The title
    names what the Lambda detected, e.g.:
        HyperPod my-cluster: CrashLoopBackOff (my-namespace/my-pod)
        HyperPod my-cluster: NotReadyNodes; ClusterFaultEvent (worker-ig)
    """
    if not detected_issues:
        # heartbeat / healthy — stable so consecutive all-clears don't spawn new
        # investigations either.
        return f"HyperPod {cluster_name}: periodic audit — no open issues"
    # Group by issue type, list the affected resources (sorted for determinism).
    by_type: dict[str, list[str]] = {}
    for i in detected_issues:
        by_type.setdefault(i["type"], []).append(i.get("resource", ""))
    parts = []
    for itype in sorted(by_type):
        resources = sorted({r for r in by_type[itype] if r})
        shown = ", ".join(resources[:3]) + ("…" if len(resources) > 3 else "")
        parts.append(f"{itype} ({shown})" if shown else itype)
    return f"HyperPod {cluster_name}: " + "; ".join(parts)


def _build_payload(cluster_name: str, k8s_checks: dict | None,
                   detected_issues: list[dict], heartbeat: bool) -> dict:
    now_iso = _now().strftime("%Y-%m-%dT%H:%M:%S.000Z")
    now_compact = _now().strftime("%Y%m%dT%H%M%SZ")
    incident_id = f"hyperpod-audit-{now_compact}"
    metadata = {
        "region": os.environ.get("AWS_REGION", "unknown"),
        "account": os.environ.get("AWS_ACCOUNT_ID", "unknown"),
        "detailType": "HyperPod Periodic Audit",
        "clusterName": cluster_name,
        "triggerMode": "audit",
        "heartbeat": heartbeat,
        "detectedIssues": detected_issues,
    }
    if k8s_checks is not None:
        metadata["k8sChecks"] = k8s_checks

    # Inline the finding + k8sChecks into the human-readable description. DevOps
    # Agent preserves the top-level `description` verbatim but flattens nested
    # payload sub-objects, so the skill can only rely on what's in this string.
    if heartbeat and not detected_issues:
        description = (
            f"Daily heartbeat audit for HyperPod cluster '{cluster_name}'. "
            f"The audit Lambda inspected cluster state and found NO open issues. "
            f"This is an informational 'all clear' liveness signal, not an incident. "
            f"The hyperpod-incident-rca skill should emit "
            f"'Suppress — periodic audit, no open incidents'."
        )
    elif detected_issues:
        lines = "\n".join(
            f"- [{i['tag']}] {i['type']} on {i['resource']}: {i['detail']}" for i in detected_issues
        )
        description = (
            f"Periodic audit for HyperPod cluster '{cluster_name}' detected "
            f"{len(detected_issues)} issue(s) at the Lambda-side check. The "
            f"hyperpod-incident-rca skill should investigate these in audit mode, "
            f"confirm current state, reconstruct the timeline, and classify each as "
            f"Monitor / Escalate / Resolved:\n{lines}"
        )
    else:
        # Defensive fallback (the handler POSTs only on detected issues or the
        # heartbeat, so this is not normally reached).
        description = (
            f"Periodic audit invocation for HyperPod cluster '{cluster_name}'. "
            f"The hyperpod-incident-rca skill should run in audit mode: discover "
            f"open fault chains in list-cluster-events, classify each, and emit "
            f"Resolved / Monitor / Escalate. If no open chains exist, emit "
            f"'Suppress — periodic audit, no open incidents'."
        )
    if k8s_checks is not None:
        description += (
            "\n\nk8sChecks configuration (parse as JSON, then apply per Phase 1 step 8 + Phase 3d):\n"
            f"{json.dumps(k8s_checks)}"
        )

    priority = "MEDIUM" if detected_issues else "LOW"
    return {
        "eventType": "incident",
        "incidentId": incident_id,
        "action": "created",
        "priority": priority,
        "title": _build_title(cluster_name, detected_issues, heartbeat),
        "description": description,
        "timestamp": now_iso,
        "service": "SageMakerHyperPod",
        "data": {
            "metadata": metadata,
            "originalEvent": {
                "source": "hyperpod-devops-agent-periodic-audit",
                "detail-type": "HyperPod Periodic Audit",
                "time": now_iso,
                "detail": {
                    "ClusterName": cluster_name,
                    "Mode": "audit",
                    "Heartbeat": heartbeat,
                    "DetectedIssues": detected_issues,
                },
            },
        },
    }


def _post(webhook_url: str, secret: str, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    timestamp = _now().strftime("%Y-%m-%dT%H:%M:%S.000Z")
    signature = base64.b64encode(
        hmac.new(
            secret.encode("utf-8"),
            f"{timestamp}:{body.decode('utf-8')}".encode("utf-8"),
            hashlib.sha256,
        ).digest()
    ).decode("utf-8")
    request = urllib.request.Request(
        webhook_url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "x-amzn-event-timestamp": timestamp,
            "x-amzn-event-signature": signature,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as resp:
            print(f"webhook response status={resp.status} body={resp.read(512)!r}")
    except HTTPError as e:
        print(f"webhook HTTP error status={e.code} body={e.read()!r}")
        raise
    except URLError as e:
        print(f"webhook URL error: {e}")
        raise


# ----------------------------------------------------------------- handler

def lambda_handler(event, context):
    cluster_name = os.environ.get("CLUSTER_NAME", "unknown-cluster")
    eks_cluster_name = os.environ.get("EKS_CLUSTER_NAME", "").strip()
    region = os.environ.get("AWS_REGION", "us-west-2")
    trigger = (event or {}).get("trigger", "periodic")
    heartbeat = trigger == "heartbeat"
    k8s_checks = _build_k8s_checks_block()

    # Inspect cluster state and POST only when a real issue is found (plus the
    # daily heartbeat), so a healthy cluster costs zero investigations.
    issues = _detect_issues(cluster_name, eks_cluster_name, region)
    print(
        f"lambda audit: cluster={cluster_name!r} trigger={trigger} "
        f"heartbeat={heartbeat} issues={len(issues)} "
        f"types={sorted({i['type'] for i in issues})}"
    )

    if not issues and not heartbeat:
        print("healthy cluster, no issues — not POSTing webhook (no investigation created)")
        return {"statusCode": 200, "body": json.dumps({"posted": False, "reason": "healthy"})}

    payload = _build_payload(cluster_name, k8s_checks, issues, heartbeat=heartbeat)
    reason = "issues-detected" if issues else "heartbeat"
    print(f"POSTing webhook: reason={reason} incidentId={payload['incidentId']} issues={len(issues)}")
    url, secret = _load_webhook_credentials()
    _post(url, secret, payload)
    return {"statusCode": 200, "body": json.dumps(
        {"posted": True, "reason": reason, "issues": len(issues), "incidentId": payload["incidentId"]}
    )}
