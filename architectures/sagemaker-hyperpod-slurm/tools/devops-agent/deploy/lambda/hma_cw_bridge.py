"""Bridge HMA CloudWatch Logs detection events onto EventBridge (Slurm only).

Why this exists
---------------
On EKS-orchestrated HyperPod clusters, the control plane emits a Warn-level
`SageMaker HyperPod Cluster Event` on EventBridge whenever HMA detects a GPU
fault (NvidiaGPUUnhealthy, ECC DBE, etc.). That event flows through the
webhook bridge Lambda into DevOps Agent and produces an investigation email.

On Slurm-orchestrated HyperPod clusters today, the HyperPod control plane
does NOT emit that Warn-level event for HMA-attributed faults. HMA still
writes its detection JSON to the cluster's CloudWatch log group
(SagemakerHealthMonitoringAgent/<group>/<instance>), and HyperPod still
initiates node replacement, but no operator-visible EventBridge event is
produced -- the DevOps Agent pipeline never runs for HMA-origin faults on
Slurm.

This Lambda closes that gap. A CloudWatch Logs subscription filter on the
cluster's log group (with FilterPattern
`{ $.HealthMonitoringAgentDetectionEvent = "HealthEvent" }`) invokes this
Lambda for every HMA detection record. The Lambda extracts fault type +
repair action + instance ID from the HMA record and calls `events:PutEvents`
with a synthetic `SageMaker HyperPod Cluster Event` shaped identically to
what HyperPod emits on EKS -- so the existing webhook bridge picks it up
unchanged, no downstream code changes required.

This is a Slurm-only workaround. Enabled by the CFN parameter
`EnableHmaCloudWatchBridge`, which `deploy.sh` auto-resolves to `true` for
Slurm and `false` for EKS. On EKS the subscription filter and this Lambda
are simply not deployed -- HyperPod's native event still flows.

Environment variables
---------------------
  CLUSTER_NAME     HyperPod cluster name (goes into the synthetic event).
  EVENT_BUS_NAME   EventBridge bus to publish onto. Default: "default".
  LOG_LEVEL        Structured-log level (INFO or DEBUG). Default: INFO.

CloudWatch Logs invocation contract
-----------------------------------
CloudWatch delivers a gzipped, base64-encoded payload:

  {
    "awslogs": {
      "data": "<base64-encoded gzip of the actual JSON>"
    }
  }

The decoded JSON has this shape:

  {
    "messageType": "DATA_MESSAGE",
    "owner": "<account-id>",
    "logGroup": "/aws/sagemaker/Clusters/<name>/<id>",
    "logStream": "SagemakerHealthMonitoringAgent/<group>/<instance>",
    "subscriptionFilters": [...],
    "logEvents": [
      {"id": "...", "timestamp": ..., "message": "<HMA JSON record>"},
      ...
    ]
  }

Each `message` is a JSON string that HMA wrote, e.g.:

  {"level":"info","ts":"2026-07-20T22:53:17Z",
   "msg":"DCGM Policy Violation found ",
   "condition: ":"XID Error","data: ":{"ErrNum":79},
   "HealthMonitoringAgentDetectionEvent":"HealthEvent"}

On EKS the corresponding record (which we don't run against) has richer
context including `"Event detail":{"Action":"replace","Type":"NvidiaGPUUnhealthy",...}`.
We support both shapes below via best-effort field extraction.
"""
import base64
import datetime
import gzip
import json
import os
import re

import boto3


_events_client = None


def _log(level: str, event: str, **kwargs) -> None:
    """Emit one structured-JSON log line for CloudWatch Logs Insights.

    Matches the shape used by webhook_bridge.py / periodic_audit.py /
    email_notifier.py: one JSON object per line with `level`, `event`, and
    arbitrary context kwargs.
    """
    print(json.dumps({"level": level, "event": event, **kwargs}, default=str))


def _events():
    global _events_client
    if _events_client is None:
        _events_client = boto3.client("events")
    return _events_client


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


# Instance-id parsing from the log stream name.
# Stream shape: SagemakerHealthMonitoringAgent/<instance-group>/<instance-id>
_STREAM_RE = re.compile(
    r"^SagemakerHealthMonitoringAgent/(?P<ig>[^/]+)/(?P<iid>i-[0-9a-f]+)$"
)


def _parse_log_stream(log_stream: str) -> tuple[str, str]:
    """Return (instance_group, instance_id) parsed from the log-stream name.

    Falls back to ("", "") if the stream doesn't match the expected shape --
    the synthesized event will carry empty fields rather than fail, and the
    RCA skill can still reason from Description + timestamp.
    """
    if not log_stream:
        return "", ""
    m = _STREAM_RE.match(log_stream)
    if not m:
        return "", ""
    return m.group("ig"), m.group("iid")


def _extract_fault(record: dict) -> tuple[str, str, str]:
    """Return (fault_type, repair_action, message) from an HMA JSON record.

    HMA records come in a few different shapes depending on the emitting
    daemon and orchestrator. We try each known shape in order, falling back
    to best-effort text extraction. Returns empty strings for any field we
    can't confidently populate -- downstream code inlines a raw excerpt.
    """
    # Shape 1: EKS NPD-caught health event (richest)
    #   {"Event detail":{"Action":"replace","Type":"NvidiaGPUUnhealthy",
    #                    "Reason":"NvidiaGpuXidErrorPendingReplacement",
    #                    "Message":"NVRM: Xid (...): 74 ..."}}
    ev = record.get("Event detail") or record.get("EventDetail") or {}
    if isinstance(ev, dict) and ev:
        return (
            str(ev.get("Type") or ev.get("Reason") or "").strip(),
            str(ev.get("Action") or "").strip(),
            str(ev.get("Message") or "").strip(),
        )

    # Shape 2: Slurm DCGM Policy Violation
    #   {"msg":"DCGM Policy Violation found ",
    #    "condition: ":"XID Error","data: ":{"ErrNum":79}}
    msg = str(record.get("msg") or "").strip()
    cond = str(record.get("condition: ") or record.get("condition") or "").strip()
    data = record.get("data: ") or record.get("data") or {}
    if cond:
        # Compose a fault_type from condition + specific error number if present
        fault_type = cond
        if isinstance(data, dict):
            err_num = data.get("ErrNum") or data.get("errNum") or data.get("Xid")
            if err_num is not None:
                fault_type = f"{cond} (code={err_num})"
        # DCGM Policy Violation events on Slurm typically imply replace-grade
        return fault_type, "Replace", msg or f"HMA detected {fault_type}"

    # Shape 3: NPD condition transition
    #   {"msg":"NPD caught positive condition",
    #    "condition details":{"type":"NvidiaGPUUnhealthy","status":"True", ...}}
    cd = record.get("condition details") or {}
    if isinstance(cd, dict) and cd.get("type"):
        return (
            str(cd.get("type") or "").strip(),
            "Replace",  # NPD-caught HMA conditions are treated as replace-grade
            str(cd.get("message") or msg or "").strip(),
        )

    # Shape 4: last-resort — echo the msg as fault_type if that's all we have
    if msg:
        return msg, "", msg

    return "", "", ""


def _build_synthetic_event(
    cluster_name: str,
    instance_group: str,
    instance_id: str,
    fault_type: str,
    repair_action: str,
    original_message: str,
    event_time_iso: str,
    region: str,
    account_id: str,
) -> dict:
    """Shape a synthetic `SageMaker HyperPod Cluster Event` for PutEvents.

    Matches the shape HyperPod emits natively on EKS (see round-2 live
    verification 2026-07-20T22:54:17Z on test-eks), so the existing webhook
    bridge Lambda's Cluster Event handler picks it up verbatim.
    """
    cluster_arn = f"arn:aws:sagemaker:{region}:{account_id}:cluster/{cluster_name}" if account_id and region else ""
    # Description matches EKS control-plane wording so operators see identical
    # subject lines regardless of orchestrator.
    fault_display = fault_type or "unknown fault"
    action_display = repair_action or "unknown"
    description = (
        f"Instance {instance_id} is unhealthy. "
        f"HyperPod Health Monitoring Agent (HMA) has detected fault type "
        f"{fault_display} on this node and is unhealthy. "
        f"Repair action: {action_display}."
    )
    if original_message and original_message not in description:
        description += f" HMA detail: {original_message}"

    detail = {
        "EventDetails": {
            "EventId": "",  # not a real cluster-event ID
            "ClusterArn": cluster_arn,
            "ClusterName": cluster_name,
            "InstanceGroupName": instance_group,
            "InstanceId": instance_id,
            "ResourceType": "Instance",
            "EventTime": event_time_iso,
            "Description": description,
            "EventLevel": "Warn",
            "OperationId": "",
            "EventMetadata": {
                "Instance": {
                    "NodeHealthInfo": {
                        "HealthStatus": "Unhealthy",
                        "HealthStatusReason": (
                            f"HyperPod Health Monitoring Agent (HMA) has detected "
                            f"fault type {fault_display} on this node and is unhealthy."
                        ),
                        "RepairAction": repair_action or "",
                        "Recommendation": (
                            "HyperPod is working on repairing the unhealthy node. "
                            "Please wait for the RepairAction to complete."
                        ),
                    }
                }
            },
            # Provenance marker so operators can distinguish synthesized vs native
            # events in the DevOps Agent journal / CloudWatch trail.
            "SyntheticSource": "hma-cw-bridge",
        }
    }
    return {
        # Custom source. AWS EventBridge reserves the `aws.*` prefix for events
        # emitted by AWS services, so PutEvents from customer code cannot spoof
        # `aws.sagemaker` — a first attempt with that source returned
        # `NotAuthorizedForSourceException`. Naming matches the pattern used by
        # periodic_audit.py's synthesized `originalEvent.source`
        # (`hyperpod-devops-agent-periodic-audit`) for within-codebase
        # consistency. `HyperPodEventRule.EventPattern` in the CFN template
        # explicitly accepts this source alongside `aws.sagemaker`.
        "Source": "hyperpod-devops-agent-hma-cw-bridge",
        "DetailType": "SageMaker HyperPod Cluster Event",
        "Detail": json.dumps(detail),
        "EventBusName": os.environ.get("EVENT_BUS_NAME", "default"),
    }


def _decode_cwlogs_payload(event: dict) -> dict:
    """Decode the gzipped+base64 payload CloudWatch Logs delivers.

    Raises ValueError on malformed input rather than swallowing -- Lambda
    will surface it and CloudWatch will retry.
    """
    if "awslogs" not in event or "data" not in event["awslogs"]:
        raise ValueError("event missing awslogs.data envelope")
    raw = base64.b64decode(event["awslogs"]["data"])
    decompressed = gzip.decompress(raw)
    return json.loads(decompressed)


def _parse_hma_record(message: str) -> dict:
    """Parse the raw log-line message as JSON, or {} on failure.

    HMA writes JSON per line. Non-JSON lines (unlikely given the subscription
    filter's FilterPattern, but not impossible) are silently skipped so a
    single malformed record doesn't fail the whole batch.
    """
    if not message:
        return {}
    try:
        parsed = json.loads(message)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}


def _put_events_with_retry(entries: list[dict], max_attempts: int = 2) -> dict:
    """Best-effort PutEvents with one retry on ThrottlingException.

    Returns the PutEvents response of the last attempt. Non-throttling errors
    bubble up (Lambda logs them; CloudWatch Logs subscription doesn't retry
    async invocations by default -- see IMPLEMENTATION.md for the failure
    semantics).
    """
    last_response = None
    for attempt in range(max_attempts):
        response = _events().put_events(Entries=entries)
        last_response = response
        failed_count = response.get("FailedEntryCount", 0)
        if failed_count == 0:
            return response
        # Throttling shows up as ErrorCode "ThrottlingException" per entry
        transient = any(
            (e.get("ErrorCode") or "") in ("ThrottlingException", "InternalException")
            for e in response.get("Entries", [])
            if "ErrorCode" in e
        )
        if not transient or attempt == max_attempts - 1:
            _log(
                "ERROR",
                "put_events_failed",
                failedCount=failed_count,
                entries=response.get("Entries"),
            )
            return response
        _log("WARN", "put_events_retry", attempt=attempt + 1, failedCount=failed_count)
    return last_response or {}


def lambda_handler(event, context):
    """CloudWatch Logs subscription invocation entry point."""
    cluster_name = os.environ.get("CLUSTER_NAME", "unknown-cluster")
    region = os.environ.get("AWS_REGION", "")
    account_id = ""
    # AWS Lambda populates AWS_LAMBDA_FUNCTION_ARN; parse account from it.
    fn_arn = os.environ.get("AWS_LAMBDA_FUNCTION_ARN", "")
    if fn_arn.startswith("arn:"):
        parts = fn_arn.split(":")
        if len(parts) >= 5:
            account_id = parts[4]

    try:
        payload = _decode_cwlogs_payload(event)
    except (ValueError, gzip.BadGzipFile) as exc:
        _log("ERROR", "payload_decode_failed", error=repr(exc))
        # Re-raise so Lambda records the failure metric; do not silently drop.
        raise

    log_stream = payload.get("logStream", "")
    log_events = payload.get("logEvents", []) or []
    _log(
        "INFO",
        "cw_batch_received",
        logStream=log_stream,
        recordCount=len(log_events),
    )

    instance_group, instance_id = _parse_log_stream(log_stream)

    entries: list[dict] = []
    skipped = 0
    for record_wrapper in log_events:
        message = record_wrapper.get("message", "")
        record = _parse_hma_record(message)
        if not record:
            skipped += 1
            continue
        # Extra defensive gate: even though the CW subscription-filter
        # FilterPattern already selects only records with
        # HealthMonitoringAgentDetectionEvent == "HealthEvent", re-check here
        # so a mis-configured filter can't accidentally forward arbitrary
        # non-HMA log lines onto EventBridge.
        if record.get("HealthMonitoringAgentDetectionEvent") != "HealthEvent":
            skipped += 1
            continue

        fault_type, repair_action, original_message = _extract_fault(record)
        if not fault_type:
            _log(
                "WARN",
                "unparseable_hma_record",
                logStream=log_stream,
                sample=message[:200],
            )
            skipped += 1
            continue

        # Prefer HMA-record timestamp; fall back to CW record timestamp; last
        # resort is now().
        event_time_iso = str(record.get("ts") or "").strip()
        if not event_time_iso:
            ts_ms = record_wrapper.get("timestamp")
            if isinstance(ts_ms, int):
                event_time_iso = datetime.datetime.fromtimestamp(
                    ts_ms / 1000, tz=datetime.timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        if not event_time_iso:
            event_time_iso = _now_iso()

        synthesized = _build_synthetic_event(
            cluster_name=cluster_name,
            instance_group=instance_group,
            instance_id=instance_id,
            fault_type=fault_type,
            repair_action=repair_action,
            original_message=original_message,
            event_time_iso=event_time_iso,
            region=region,
            account_id=account_id,
        )
        entries.append(synthesized)
        _log(
            "INFO",
            "hma_record_processed",
            faultType=fault_type,
            action=repair_action,
            instanceId=instance_id,
            eventTime=event_time_iso,
        )

    if not entries:
        _log("INFO", "no_entries_to_forward", skipped=skipped)
        return {"forwarded": 0, "skipped": skipped}

    # PutEvents accepts up to 10 entries per call; batch defensively.
    forwarded = 0
    for i in range(0, len(entries), 10):
        batch = entries[i : i + 10]
        response = _put_events_with_retry(batch)
        failed_count = response.get("FailedEntryCount", 0)
        forwarded += len(batch) - failed_count

    _log("INFO", "batch_complete", forwarded=forwarded, skipped=skipped)
    return {"forwarded": forwarded, "skipped": skipped}
