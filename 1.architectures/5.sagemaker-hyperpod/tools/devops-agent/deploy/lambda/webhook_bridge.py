"""Forward HyperPod EventBridge events to the AWS DevOps Agent generic webhook.

The webhook URL and HMAC secret live in AWS Secrets Manager as a single
JSON blob with two keys: {"url": "...", "secret": "..."}. The secret ARN
is passed in via the WEBHOOK_SECRET_ARN env var.

For every HyperPod event this Lambda receives, it builds the DevOps Agent
incident payload, HMAC-signs the request, and POSTs it. A non-2xx response
is logged and re-raised so EventBridge will retry.
"""
import base64
import datetime
import hashlib
import hmac
import json
import os
import urllib.request
from urllib.error import HTTPError, URLError

import boto3


# EventLevel values that should NOT trigger an investigation. "Info" is by far
# the most common (e.g. routine cluster update lifecycle, EKS access entry
# refreshes, instance group scaling). Override with WEBHOOK_DROP_EVENT_LEVELS
# (comma-separated) if needed without redeploying.
_DEFAULT_DROP_LEVELS = {"info"}


_secrets_cache: dict[str, str] = {}
_sagemaker_client = None


def _sagemaker():
    """SageMaker client, lazily created + reused across warm invocations."""
    global _sagemaker_client
    if _sagemaker_client is None:
        _sagemaker_client = boto3.client("sagemaker")
    return _sagemaker_client


def _describe_cluster_event(cluster_name: str, event_id: str) -> dict:
    """Fetch the enriched event detail via DescribeClusterEvent.

    The EventBridge envelope carries EventMetadata.Instance as null even for
    events (e.g. capacity failures) where DescribeClusterEvent returns a
    populated FailureMessage. Bridge calls this API for every non-Info
    event so the enriched fields can be forwarded to the webhook.

    Returns the top-level EventDetails dict or an empty dict on failure.
    Failure modes handled: ValidationException (not Continuous
    NodeProvisioningMode), throttling, transient network errors.
    """
    if not (cluster_name and event_id):
        return {}
    try:
        resp = _sagemaker().describe_cluster_event(
            ClusterName=cluster_name,
            EventId=event_id,
        )
        return resp.get("EventDetails", {}) or {}
    except Exception as e:
        # Log but don't fail the invocation — description-only forwarding is
        # the fallback behaviour. Never silently drop the event.
        print(f"describe_cluster_event failed cluster={cluster_name!r} eventId={event_id!r} err={e!r}")
        return {}


def _load_webhook_credentials() -> tuple[str, str]:
    """Fetch webhook URL + HMAC secret from Secrets Manager, cached per cold start."""
    if "url" in _secrets_cache and "secret" in _secrets_cache:
        return _secrets_cache["url"], _secrets_cache["secret"]

    secret_arn = os.environ["WEBHOOK_SECRET_ARN"]
    client = boto3.client("secretsmanager")
    resp = client.get_secret_value(SecretId=secret_arn)
    payload = json.loads(resp["SecretString"])
    _secrets_cache["url"] = payload["url"]
    _secrets_cache["secret"] = payload["secret"]
    return _secrets_cache["url"], _secrets_cache["secret"]


def _drop_levels() -> set[str]:
    raw = os.environ.get("WEBHOOK_DROP_EVENT_LEVELS")
    if not raw:
        return _DEFAULT_DROP_LEVELS
    return {tok.strip().lower() for tok in raw.split(",") if tok.strip()}


def _event_level(event: dict) -> str | None:
    """Pull EventLevel from the EventBridge envelope.

    SageMaker HyperPod Cluster Event payloads wrap ListClusterEvents data under
    detail.EventDetails, so EventLevel lives at detail.EventDetails.EventLevel.
    The other detail-types (Cluster State Change, Cluster Node Health Event)
    don't carry EventLevel; for those, treat as unknown -> never drop.
    """
    return (
        event.get("detail", {})
        .get("EventDetails", {})
        .get("EventLevel")
    )


def _is_scale_progress_noise(event: dict) -> bool:
    """Detect 'lost orchestration-ready' Warn events emitted during scaling.

    These are progress updates during customer-initiated UpdateCluster
    operations, not incident signals. Filtering here avoids forwarding
    noise to the DevOps Agent webhook. See the mental-model doc section
    "Scale-in-progress emits spurious Warn events with misleading
    FailureMessage" for details. Interim until HyperPod engineering
    removes these from the Warn-level stream.
    """
    detail = event.get("detail", {})
    ev_details = detail.get("EventDetails", {})
    description = ev_details.get("Description", "")
    return "lost orchestration-ready status" in description


def _should_drop(event: dict) -> tuple[bool, str | None]:
    level = _event_level(event)
    if level is None:
        return False, None
    if level.lower() in _drop_levels():
        return True, level
    return False, level


def _cluster_filter() -> set[str]:
    raw = os.environ.get("WEBHOOK_CLUSTER_FILTER", "")
    return {tok.strip() for tok in raw.split(",") if tok.strip()}


def _event_cluster_name(event: dict) -> str | None:
    detail = event.get("detail", {})
    return (
        detail.get("ClusterName")
        or detail.get("EventDetails", {}).get("ClusterName")
    )


def _priority_for(event: dict) -> str:
    """Map HyperPod event severity onto DevOps Agent priority levels."""
    detail = event.get("detail", {})
    detail_type = event.get("detail-type", "")

    if detail_type == "SageMaker HyperPod Cluster Node Health Event":
        health_status = (
            detail.get("HealthSummary", {}).get("HealthStatus", "").lower()
        )
        if "unhealthy" in health_status or "degraded" in health_status:
            return "HIGH"
        return "MEDIUM"

    if detail_type == "SageMaker HyperPod Cluster State Change":
        status = detail.get("ClusterStatus", "").lower()
        if status in {"failed", "rollingback"}:
            return "HIGH"
        if status in {"updating", "deleting"}:
            return "LOW"
        return "MEDIUM"

    return "MEDIUM"


def _shorten(text: str, limit: int = 120) -> str:
    """Trim a fault-summary string for use in the incident title.

    Whitespace-collapse first, then truncate at word boundary if possible,
    then append an ellipsis if trimmed. 120 chars fits the operator-visible
    FailureMessage lengths we've observed (e.g. the capacity error at 91
    chars) while staying under most dashboard title-column widths.
    """
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit - 1)
    if cut <= limit // 2:
        cut = limit - 1
    return text[:cut].rstrip(".,;: ") + "…"


def _title_and_description(event: dict) -> tuple[str, str]:
    detail = event.get("detail", {})
    detail_type = event.get("detail-type", "unknown")
    region = event.get("region", "unknown")
    account = event.get("account", "unknown")
    cluster_name = (
        detail.get("ClusterName")
        or detail.get("EventDetails", {}).get("ClusterName")
        or "unknown-cluster"
    )

    if detail_type == "SageMaker HyperPod Cluster Node Health Event":
        health = detail.get("HealthSummary", {})
        instance_id = detail.get("InstanceId", "unknown-instance")
        instance_group = detail.get("InstanceGroupName") or "cluster"
        # Short fault summary from HMA fields for the title.
        health_reason = health.get("HealthStatusReason", "").strip()
        repair_action = health.get("RepairAction", "").strip()
        summary_parts = []
        if health_reason:
            summary_parts.append(health_reason)
        if repair_action:
            summary_parts.append(f"Repair action: {repair_action}")
        fault_summary = _shorten(" — ".join(summary_parts) or health.get("HealthStatus", "unknown"))
        title = f"HyperPod {instance_group} / {fault_summary}"
        description = (
            f"HyperPod cluster '{cluster_name}' (account {account}, {region}) reported a node "
            f"health event for instance {instance_id}. "
            f"HealthStatus={health.get('HealthStatus', 'n/a')}; "
            f"Reason={health.get('HealthStatusReason', 'n/a')}; "
            f"RepairAction={health.get('RepairAction', 'n/a')}; "
            f"Recommendation={health.get('Recommendation', 'n/a')}. "
            f"Resolve the underlying EKS cluster (for EKS-orchestrated clusters) via "
            f"'aws sagemaker describe-cluster --cluster-name {cluster_name}' and inspect "
            f"node {instance_id} via the SageMaker HyperPod control plane."
        )
        return title, description

    if detail_type == "SageMaker HyperPod Cluster State Change":
        status = detail.get("ClusterStatus", "unknown")
        title = f"HyperPod {cluster_name} / cluster status → {status}"
        groups = detail.get("InstanceGroups", []) or []
        groups_summary = ", ".join(
            f"{g.get('InstanceGroupName')}: {g.get('CurrentCount')}/{g.get('TargetCount')} ({g.get('Status')})"
            for g in groups
        )
        description = (
            f"HyperPod cluster '{cluster_name}' (account {account}, {region}) transitioned to status "
            f"{status}. Instance groups: {groups_summary or 'none'}. Investigate via SageMaker "
            f"DescribeCluster and the underlying EKS resources."
        )
        return title, description

    if detail_type == "SageMaker HyperPod Cluster Event":
        ev_details = detail.get("EventDetails", {})
        resource_type = ev_details.get("ResourceType", "n/a")
        description_text = ev_details.get("Description", "")
        sm_event_id = ev_details.get("EventId", "")
        instance_group = ev_details.get("InstanceGroupName") or "cluster"

        # For non-Info events, enrich the description with fields the
        # EventBridge envelope omits (notably FailureMessage). See
        # README + engineering-team report for why this workaround exists.
        #
        # Walk EVERY subtree under EventMetadata (Cluster, Instance,
        # InstanceGroup, InstanceGroupScaling, InstanceMonitor,
        # InstanceHealth, etc.) — different fault classes populate
        # different subtrees. Cluster-level events put FailureMessage
        # under EventMetadata.Cluster; instance-level under
        # EventMetadata.Instance; a similar pattern likely holds for
        # the other subtrees when populated.
        # Keys are prefixed with the subtree name so
        # `Cluster.FailureMessage` and `Instance.FailureMessage` are
        # distinguishable, and either can drive the title.
        enriched = {}
        if ev_details.get("EventLevel", "").lower() != "info" and sm_event_id:
            api_ev = _describe_cluster_event(cluster_name, sm_event_id)
            api_metadata = (
                api_ev.get("EventDetails", {}).get("EventMetadata", {}) or {}
            )
            for subtree_name, subtree in api_metadata.items():
                if not isinstance(subtree, dict):
                    continue
                for k, v in subtree.items():
                    if v not in (None, "", {}, []):
                        enriched[f"{subtree_name}.{k}"] = v

        # Title picks the most operator-actionable text available:
        # any *.FailureMessage > event Description > resource-type fallback.
        # If multiple subtrees carry a FailureMessage, prefer Instance
        # (most fault-specific), then Cluster, then whatever alphabetical
        # order gives us.
        def _pick_failure_message() -> str:
            preferred = ["Instance.FailureMessage", "Cluster.FailureMessage"]
            for key in preferred:
                v = enriched.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            for key in sorted(enriched):
                if key.endswith(".FailureMessage"):
                    v = enriched[key]
                    if isinstance(v, str) and v.strip():
                        return v.strip()
            return ""

        failure_message = _pick_failure_message()
        if failure_message:
            title_fault = _shorten(failure_message)
        elif description_text:
            title_fault = _shorten(description_text)
        else:
            title_fault = resource_type
        title = f"HyperPod {instance_group} / {title_fault}"

        description = (
            f"HyperPod cluster '{cluster_name}' (account {account}, {region}) emitted a cluster "
            f"event for resource type {resource_type}. "
            f"InstanceGroup={ev_details.get('InstanceGroupName', 'n/a')}; "
            f"InstanceId={ev_details.get('InstanceId', 'n/a')}; "
            f"EventId={sm_event_id or 'n/a'}. "
            f"Description: {description_text or 'n/a'}."
        )
        if failure_message:
            description += f" FailureMessage: {failure_message}."
        # Include any other populated fields (from any subtree) so
        # downstream categorization keys benefit from them too.
        extras = {
            k: v for k, v in enriched.items()
            if not k.endswith(".FailureMessage")
        }
        if extras:
            extras_str = ", ".join(
                f"{k}={json.dumps(v, default=str) if isinstance(v, (dict, list)) else v}"
                for k, v in sorted(extras.items())
            )
            description += f" EventMetadata: {extras_str}."
        return title, description

    return (
        f"HyperPod event: {detail_type} on {cluster_name}",
        f"HyperPod cluster '{cluster_name}' (account {account}, {region}) emitted an event of type "
        f"'{detail_type}'. Raw detail attached.",
    )


def _build_payload(event: dict) -> dict:
    title, description = _title_and_description(event)
    detail = event.get("detail", {})
    event_id = event.get("id") or event.get("time") or datetime.datetime.utcnow().isoformat()

    return {
        "eventType": "incident",
        "incidentId": f"hyperpod-{event_id}",
        "action": "created",
        "priority": _priority_for(event),
        "title": title,
        "description": description,
        "timestamp": event.get("time")
        or datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "service": "SageMakerHyperPod",
        "data": {
            "metadata": {
                "region": event.get("region"),
                "account": event.get("account"),
                "detailType": event.get("detail-type"),
                "clusterName": detail.get("ClusterName")
                or detail.get("EventDetails", {}).get("ClusterName"),
            },
            "originalEvent": event,
        },
    }


def _truthy(env_var: str) -> bool:
    return os.environ.get(env_var, "").strip().lower() in {"1", "true", "yes", "on"}


def _post(webhook_url: str, secret: str, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z")
    signature = base64.b64encode(
        hmac.new(
            secret.encode("utf-8"),
            f"{timestamp}:{body.decode('utf-8')}".encode("utf-8"),
            hashlib.sha256,
        ).digest()
    ).decode("utf-8")

    if _truthy("WEBHOOK_LOG_PAYLOAD"):
        # body is the EXACT bytes being POSTed; log size + content + signed
        # timestamp so you can reproduce the HMAC offline if needed.
        print(f"webhook POST url={webhook_url}")
        print(f"webhook POST timestamp={timestamp} signature={signature}")
        print(f"webhook POST body bytes={len(body)} json={body.decode('utf-8')}")

    if _truthy("WEBHOOK_DRY_RUN"):
        print("webhook DRY RUN: skipping POST (WEBHOOK_DRY_RUN=true)")
        return

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


def lambda_handler(event, context):
    print(f"received event detail-type={event.get('detail-type')!r} id={event.get('id')!r}")
    if _truthy("WEBHOOK_LOG_FULL_EVENT"):
        print(f"full event: {json.dumps(event)}")
    allowlist = _cluster_filter()
    if allowlist:
        cluster = _event_cluster_name(event)
        if cluster not in allowlist:
            print(f"dropping event: cluster={cluster!r} not in WEBHOOK_CLUSTER_FILTER={sorted(allowlist)}")
            return {"statusCode": 200, "body": json.dumps({"dropped": True, "reason": "cluster-not-in-filter"})}

    drop, level = _should_drop(event)
    if drop:
        print(f"dropping event: EventLevel={level!r} is in WEBHOOK_DROP_EVENT_LEVELS")
        return {"statusCode": 200, "body": json.dumps({"dropped": True, "eventLevel": level})}

    if _is_scale_progress_noise(event):
        print("dropping event: scale-in-progress 'lost orchestration-ready' noise")
        return {"statusCode": 200, "body": json.dumps({"dropped": True, "reason": "scale-progress-noise"})}

    webhook_url, secret = _load_webhook_credentials()
    payload = _build_payload(event)
    print(f"payload incidentId={payload['incidentId']} priority={payload['priority']} title={payload['title']!r} eventLevel={level!r}")
    _post(webhook_url, secret, payload)
    return {"statusCode": 200, "body": json.dumps({"incidentId": payload["incidentId"]})}
