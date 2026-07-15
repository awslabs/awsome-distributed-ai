"""Compose HTML email from a completed DevOps Agent investigation.

Design (v2): the notifier receives the aws.aidevops EventBridge event, extracts
the task/execution ids, and pulls the FULL investigation context via
`get_backlog_task` + `list_journal_records`. It composes the email from the
raw journal records — symptoms, findings, investigation_gaps — rather than
depending on a specific verdict-title shape from the RCA skill.

This makes the notifier resilient to skill output drift. As long as the
investigation identifies a root cause, the notifier will surface it.

Filtering (in order):
  1. Only "Investigation Completed" events (single email per lifecycle).
  2. Dedup by execution_id: check S3 for a marker under
     s3://$MARKER_BUCKET/$MARKER_PREFIX/<execution_id>. If present, we've
     already emailed for this execution and drop the event. Otherwise
     write the marker after the email is sent successfully. This is
     robust to DevOps Agent's periodic-audit trigger re-emitting the
     same execution's Investigation Completed event many times.
  3. Suppress-verdict skip: the RCA marked this investigation as Suppress.
  4. No-findings skip: the RCA produced no findings; nothing to report.

Environment variables:
  EMAIL_SENDER              SES-verified From address. Required.
  EMAIL_RECIPIENTS          Comma-separated To addresses. Required.
  EMAIL_DETAIL_TYPES        Detail-types to consider (default: "Investigation Completed").
  CONSOLE_URL_TEMPLATE      URL template. Tokens: %region%, %account%,
                            %agent_space_id%, %task_id%.
  MARKER_BUCKET             S3 bucket name for per-execution dedup markers. Required.
  MARKER_PREFIX             Key prefix inside the bucket (default: "emailed/").
  FORCE_SEND                If "1"/"true", bypass every filter (including the
                            marker check). Useful for debugging.
"""
import datetime
import html
import json
import os
import re

import boto3


_ses_client = None
_devops_client = None
_s3_client = None


def _ses():
    global _ses_client
    if _ses_client is None:
        _ses_client = boto3.client("ses")
    return _ses_client


def _s3():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3")
    return _s3_client


def _devops():
    global _devops_client
    if _devops_client is None:
        _devops_client = boto3.client("devops-agent")
    return _devops_client


def _env_list(name: str, default: str) -> list[str]:
    raw = os.environ.get(name, default)
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _extract_meta(event: dict) -> dict:
    """Pull the identifiers we need from the aws.aidevops event envelope."""
    detail = event.get("detail", {})
    metadata = detail.get("metadata", {}) or {}
    data = detail.get("data", {}) or {}
    return {
        "agent_space_id": metadata.get("agent_space_id") or "",
        "task_id": metadata.get("task_id") or "",
        "execution_id": metadata.get("execution_id") or "",
        "priority": (data.get("priority") or "").upper(),
        "status": data.get("status") or "",
        "created_at": data.get("created_at") or "",
        "updated_at": data.get("updated_at") or "",
    }


def _fetch_task(agent_space_id: str, task_id: str) -> dict:
    if not agent_space_id or not task_id:
        return {}
    try:
        resp = _devops().get_backlog_task(agentSpaceId=agent_space_id, taskId=task_id)
        return resp.get("task", {}) or {}
    except Exception as exc:
        print(f"get_backlog_task failed: {exc!r}")
        return {}


def _fetch_journal(agent_space_id: str, execution_id: str) -> dict:
    """Return parsed records grouped by type, plus the raw records.

    Shape:
        {
            "symptoms":  [ {title, description, ...}, ... ],
            "findings":  [ {title, description, finding_type, ...}, ... ],
            "gaps":      [ str, ... ],
            "raw":       [ raw record dict, ... ],   # keeps createdAt etc.
            "raw_count": int,
        }
    """
    result = {"symptoms": [], "findings": [], "gaps": [], "raw": [], "raw_count": 0}
    if not agent_space_id or not execution_id:
        return result
    try:
        paginator = _devops().get_paginator("list_journal_records")
        for page in paginator.paginate(agentSpaceId=agent_space_id, executionId=execution_id):
            for record in page.get("records", []):
                result["raw_count"] += 1
                result["raw"].append(record)
                rtype = record.get("recordType") or ""
                try:
                    content = json.loads(record.get("content", "{}")) if record.get("content") else {}
                except (ValueError, TypeError):
                    content = {}
                if rtype == "symptom":
                    result["symptoms"].append(content)
                elif rtype == "finding":
                    result["findings"].append(content)
                elif rtype in ("investigation_gap", "gap"):
                    # Content may be a plain string or a dict; capture whichever.
                    if isinstance(content, dict):
                        text = content.get("title") or content.get("description") or json.dumps(content)
                    else:
                        text = str(content)
                    result["gaps"].append(text)
    except Exception as exc:
        print(f"list_journal_records failed: {exc!r}")
    return result


def _has_actionable_content(journal: dict) -> bool:
    """Email-worthy investigations have identifiable findings.

    Empirically, the RCA skill sometimes marks findings as `cause` rather
    than `root_cause` even when the causal chain terminates. Any non-empty
    findings list means the RCA phase produced conclusions worth reporting.
    Suppress-style audits (no fault activity in the window) produce no
    findings at all and are correctly dropped.
    """
    return bool(journal["findings"])


def _marker_key(execution_id: str) -> str:
    prefix = os.environ.get("MARKER_PREFIX", "emailed/")
    if not prefix.endswith("/"):
        prefix = prefix + "/"
    return f"{prefix}{execution_id}"


def _already_emailed(execution_id: str) -> bool:
    """True if we've already emailed for this execution_id.

    Reads a marker object under s3://$MARKER_BUCKET/<prefix>/<execution_id>.
    HeadObject returns 200 for exists, 404 for absent. Any other error we
    treat as 'unknown' → don't skip (fail-open toward sending); the
    put-after-send path will still write the marker so subsequent events
    dedup normally.
    """
    bucket = os.environ.get("MARKER_BUCKET", "")
    if not bucket or not execution_id:
        return False
    key = _marker_key(execution_id)
    try:
        _s3().head_object(Bucket=bucket, Key=key)
        return True
    except Exception as exc:
        # 404 is the normal "object doesn't exist" case. S3 also returns 403
        # for missing objects when the caller lacks ListBucket on the bucket
        # — treat that the same way. Any OTHER error (throttling, network)
        # is logged and we fall through to sending (fail-open).
        code = None
        if hasattr(exc, "response"):
            code = (exc.response or {}).get("Error", {}).get("Code")
        if code in ("404", "403", "NoSuchKey", "NotFound") or "Not Found" in str(exc):
            return False
        print(f"head_object failed for {bucket}/{key}: {exc!r} (treating as not-emailed)")
        return False


def _write_marker(execution_id: str, event: dict, meta: dict) -> None:
    """Record that we emailed for this execution_id.

    The body records enough context to trace back to the source event if
    someone later inspects the bucket for debugging.
    """
    bucket = os.environ.get("MARKER_BUCKET", "")
    if not bucket or not execution_id:
        return
    key = _marker_key(execution_id)
    body = json.dumps({
        "execution_id": execution_id,
        "task_id": meta.get("task_id", ""),
        "agent_space_id": meta.get("agent_space_id", ""),
        "event_id": event.get("id", ""),
        "event_time": event.get("time", ""),
        "priority": meta.get("priority", ""),
    }, sort_keys=True).encode("utf-8")
    try:
        _s3().put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
    except Exception as exc:
        # Log but do not fail — the email has already been sent. Worst-case a
        # subsequent event resends, which is what our old behavior was.
        print(f"put_object failed for {bucket}/{key}: {exc!r} (email sent OK, dedup marker not written)")


def _is_heartbeat(task: dict) -> bool:
    """Detect the daily heartbeat audit (silent liveness — never emailed).

    The periodic-audit Lambda marks heartbeat runs deterministically: the task
    title is 'HyperPod <cluster>: periodic audit — no open issues' and the
    description begins 'Daily heartbeat audit for HyperPod cluster'. Match either
    so this is independent of RCA-skill wording.
    """
    title = (task.get("title") or "")
    desc = (task.get("description") or "")
    return (
        "Daily heartbeat audit for HyperPod cluster" in desc
        or "periodic audit — no open issues" in title
    )


def _is_suppress_verdict(journal: dict) -> bool:
    """Detect a Suppress verdict in the journal.

    The RCA SKILL.md contract is: emit a first symptom titled
        "Triage verdict: <Suppress|Monitor|Escalate|Resolved> — ..."
    We check for that prefix on any symptom title. As a fallback for
    prompt drift, we also scan the first symptom's description for a
    "Verdict: Suppress" line, which the skill's description block also
    contains.
    """
    for s in journal["symptoms"]:
        title = (s.get("title") or "").strip()
        if title.startswith("Triage verdict"):
            body = title.replace("Triage verdict:", "", 1).strip()
            first_word = body.split()[0] if body.split() else ""
            if first_word.lower() == "suppress":
                return True
    if journal["symptoms"]:
        desc = (journal["symptoms"][0].get("description") or "").strip()
        m = re.search(r"^\s*Verdict:\s*Suppress", desc, re.MULTILINE | re.IGNORECASE)
        if m:
            return True
    return False


def _verdict_symptom(journal: dict) -> dict:
    """Return the first symptom whose title starts with 'Triage verdict:', or {}."""
    for s in journal["symptoms"]:
        if (s.get("title") or "").strip().startswith("Triage verdict"):
            return s
    return journal["symptoms"][0] if journal["symptoms"] else {}


# Section headers the RCA skill emits AFTER the Summary paragraph in the verdict
# description. The Summary paragraph runs from 'Summary:' up to the first of these.
_SUMMARY_END_HEADERS = re.compile(
    r"^\s*(What HyperPod is doing right now|Timeline|Most recent event at|"
    r"Next re-check|Recommended actions|Recommendation)\s*:",
    re.IGNORECASE,
)


def _parse_summary_text(description: str) -> str:
    """Extract the RCA skill's free-text 'Summary:' paragraph, or "".

    Contract (RCA SKILL.md Phase 4): the verdict symptom's description contains a
    single natural-language paragraph under a 'Summary:' line that weaves together
    what happened -> likely cause -> recommended action. We capture from just
    after 'Summary:' up to the next known section header (or a blank line that
    precedes one). Deliberately NOT field-structured — nothing to drift. Returns
    "" if no Summary block is present (graceful fallback).
    """
    if not description or "Summary:" not in description:
        return ""
    after = description.split("Summary:", 1)[1]
    lines: list[str] = []
    for raw in after.splitlines():
        if _SUMMARY_END_HEADERS.match(raw):
            break
        line = raw.strip()
        if not line:
            # A blank line ends the paragraph only once we've collected content.
            if lines:
                break
            continue
        lines.append(line)
    return _strip_confidence_tags(" ".join(lines))


def _first_sentence(text: str, limit: int = 160) -> str:
    """First sentence of a paragraph, for use as a headline."""
    if not text:
        return ""
    m = re.search(r"(.+?[.!?])(\s|$)", text)
    sentence = m.group(1).strip() if m else text
    return sentence[:limit]


def _strip_confidence_tags(text: str) -> str:
    """Remove RCA confidence annotations ([direct]/[proxy]/[unverified]) and tidy.

    Those tags are meaningful on detailed evidence claims (kept in the body), but
    they're noise in the one-line email subject/headline.
    """
    cleaned = re.sub(r"\[(?:direct|proxy|unverified)\]\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def _pick_headline(journal: dict, task: dict) -> str:
    """Best short headline for the subject line.

    Priority:
    1. First sentence of the Summary paragraph (specific, human-readable).
    2. The verdict name (post-colon body of the 'Triage verdict:' title).
    3. The first symptom's title.
    4. The task title.
    5. A generic string.

    Confidence tags are already stripped by _parse_summary_text; we also strip
    the fallback branches for safety.
    """
    vs = _verdict_symptom(journal)
    summary = _parse_summary_text(vs.get("description") or "")
    if summary:
        return _first_sentence(summary)

    def _raw() -> str:
        for s in journal["symptoms"]:
            t = (s.get("title") or "").strip()
            if t.startswith("Triage verdict"):
                return t.replace("Triage verdict:", "", 1).strip().split(" :: ", 1)[0]
        if journal["symptoms"]:
            first = (journal["symptoms"][0].get("title") or "").strip()
            if first:
                return first
        if task.get("title"):
            return task["title"]
        return "HyperPod investigation"

    return _strip_confidence_tags(_raw())[:160]


def _console_url(event: dict, meta: dict) -> str:
    template = os.environ.get("CONSOLE_URL_TEMPLATE", "")
    if not template:
        return ""
    values = {
        "%region%": event.get("region", ""),
        "%account%": event.get("account", ""),
        "%agent_space_id%": meta.get("agent_space_id", ""),
        "%task_id%": meta.get("task_id", ""),
    }
    result = template
    for token, value in values.items():
        result = result.replace(token, value or "")
    return result


_PRIORITY_STYLE = {
    "HIGH":   ("HIGH",   "#c62828", "#ffebee"),
    "MEDIUM": ("MEDIUM", "#e65100", "#fff3e0"),
    "LOW":    ("LOW",    "#1565c0", "#e3f2fd"),
}


def _priority_chip_html(priority: str) -> str:
    label, fg, bg = _PRIORITY_STYLE.get(priority, ("?", "#424242", "#eeeeee"))
    return (
        f'<span style="background:{bg};color:{fg};font-weight:600;'
        f'padding:2px 8px;border-radius:3px;font-size:12px;'
        f'letter-spacing:0.5px;font-family:Helvetica,Arial,sans-serif;">'
        f'{html.escape(label)}</span>'
    )


def _format_ts(iso: str) -> str:
    if not iso:
        return "—"
    try:
        dt = datetime.datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return iso


def _format_subject(event: dict, headline: str, priority: str) -> str:
    priority_tag = {"HIGH": "[H]", "MEDIUM": "[M]", "LOW": "[L]"}.get(priority, "[?]")
    return f"{priority_tag} HyperPod: {headline}"[:200]


def _render_section(esc, heading: str, rows_html: str) -> str:
    return (
        f'<h3 style="font-size:13px;color:#616161;text-transform:uppercase;'
        f'letter-spacing:1px;margin:24px 0 8px 0;">{esc(heading)}</h3>'
        f'{rows_html}'
    )


def _render_record_card(esc, title: str, description: str, badge: str = "") -> str:
    badge_html = ""
    if badge:
        badge_html = (
            f'<span style="background:#eeeeee;color:#424242;font-size:11px;'
            f'font-weight:600;padding:2px 8px;border-radius:3px;'
            f'letter-spacing:0.5px;margin-right:8px;">{esc(badge)}</span>'
        )
    return (
        f'<div style="background:#fafafa;border:1px solid #e0e0e0;'
        f'padding:12px 16px;margin:0 0 12px 0;">'
        f'<div style="font-size:13px;font-weight:600;color:#212121;margin-bottom:6px;">'
        f'{badge_html}{esc(title)}</div>'
        f'<pre style="white-space:pre-wrap;word-wrap:break-word;'
        f'font-family:Menlo,Consolas,monospace;font-size:12px;'
        f'line-height:1.5;color:#212121;margin:0;">{esc(description)}</pre>'
        f'</div>'
    )


def _format_body_html(
    event: dict,
    meta: dict,
    task: dict,
    journal: dict,
    headline: str,
    console_url: str,
) -> str:
    esc = html.escape
    priority = meta["priority"]
    reference = task.get("reference") or {}

    facts_rows = "".join(
        f'<tr><td style="color:#616161;padding:4px 12px 4px 0;vertical-align:top;white-space:nowrap;">{esc(k)}</td>'
        f'<td style="padding:4px 0;vertical-align:top;font-family:Menlo,Consolas,monospace;font-size:13px;">{v}</td></tr>'
        for k, v in [
            ("Priority", _priority_chip_html(priority)),
            ("Event", esc(event.get("detail-type", ""))),
            ("Status", esc(meta.get("status", ""))),
            ("Trigger source", esc(reference.get("system") or "—")),
            ("Trigger title", esc(reference.get("title") or "—")),
            ("Investigation ID", esc(meta.get("task_id", "—"))),
            ("Account", esc(event.get("account", "—"))),
            ("Region", esc(event.get("region", "—"))),
            ("Created", esc(_format_ts(meta.get("created_at", "")))),
            ("Updated", esc(_format_ts(meta.get("updated_at", "")))),
        ]
    )

    # Symptoms — first is often the verdict, remaining are per-resource
    symptoms_html = ""
    if journal["symptoms"]:
        rows = []
        for i, s in enumerate(journal["symptoms"][:6]):
            title = (s.get("title") or "").strip() or f"symptom #{i+1}"
            desc = (s.get("description") or "").strip()
            badge = "VERDICT" if title.startswith("Triage verdict") else ("SYMPTOM" if i > 0 else "PRIMARY")
            rows.append(_render_record_card(esc, title, desc[:2000], badge))
        extra = len(journal["symptoms"]) - 6
        if extra > 0:
            rows.append(
                f'<div style="font-size:12px;color:#616161;margin-top:-4px;">'
                f'…and {extra} more symptom{"s" if extra != 1 else ""}. Open the investigation for full detail.'
                f'</div>'
            )
        symptoms_html = _render_section(esc, f"Symptoms ({len(journal['symptoms'])})", "".join(rows))

    # Findings — grouped by finding_type
    findings_html = ""
    if journal["findings"]:
        by_type: dict[str, list[dict]] = {}
        for f in journal["findings"]:
            ft = (f.get("finding_type") or "unknown").lower()
            by_type.setdefault(ft, []).append(f)
        order = ["root_cause", "cause", "hypothesis"]
        rows = []
        for ft in order + [k for k in by_type if k not in order]:
            items = by_type.get(ft, [])
            for f in items:
                title = (f.get("title") or "").strip() or f"{ft} finding"
                desc = (f.get("description") or "").strip()
                rows.append(_render_record_card(esc, title, desc[:2000], ft.replace("_", " ").upper()))
        findings_html = _render_section(esc, f"Findings ({len(journal['findings'])})", "".join(rows))

    # Investigation gaps
    gaps_html = ""
    if journal["gaps"]:
        items = "".join(
            f'<li style="margin-bottom:4px;">{esc(g)}</li>'
            for g in journal["gaps"][:10]
        )
        gaps_html = _render_section(
            esc,
            "Investigation gaps",
            f'<ul style="margin:0;padding-left:20px;font-size:13px;color:#212121;">{items}</ul>'
        )

    link_block = ""
    if console_url:
        link_block = (
            f'<div style="margin-top:24px;">'
            f'<a href="{esc(console_url)}" '
            f'style="display:inline-block;background:#0b5cad;color:#ffffff;'
            f'text-decoration:none;padding:10px 20px;font-weight:600;'
            f'font-size:13px;border-radius:3px;">Open investigation</a></div>'
        )

    banner_html = (
        f'<div style="background:#fff3e0;border-left:4px solid #e65100;'
        f'padding:12px 16px;margin:0 0 16px 0;">'
        f'<div style="color:#616161;font-size:11px;letter-spacing:1px;'
        f'text-transform:uppercase;margin-bottom:4px;">Headline</div>'
        f'<div style="font-size:16px;font-weight:600;color:#111;">{esc(headline)}</div>'
        f'</div>'
    )

    # RCA summary — a single free-text paragraph (what happened -> likely cause
    # -> recommended action) from the verdict symptom's description. Rendered
    # prominently up top.
    summary = _parse_summary_text(_verdict_symptom(journal).get("description") or "")
    summary_html = ""
    if summary:
        summary_html = (
            f'<div style="background:#f8f9fa;border:1px solid #e0e0e0;'
            f'border-radius:3px;padding:14px 16px;margin:0 0 16px 0;">'
            f'<div style="color:#616161;font-size:11px;letter-spacing:1px;'
            f'text-transform:uppercase;margin-bottom:4px;">Summary</div>'
            f'<div style="font-size:14px;color:#111;">{esc(summary)}</div></div>'
        )

    return (
        f'<html><body style="margin:0;padding:24px;background:#f5f5f5;'
        f'font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',Helvetica,Arial,sans-serif;'
        f'color:#212121;font-size:14px;line-height:1.5;">'
        f'<div style="max-width:720px;margin:0 auto;background:#ffffff;'
        f'padding:24px;border:1px solid #e0e0e0;">'
        f'<div style="font-size:11px;color:#9e9e9e;letter-spacing:1px;'
        f'text-transform:uppercase;margin-bottom:4px;">AWS DevOps Agent</div>'
        f'<h1 style="font-size:20px;color:#111;margin:0 0 12px 0;font-weight:600;">'
        f'HyperPod investigation completed</h1>'
        f'{banner_html}'
        f'{summary_html}'
        f'<table style="border-collapse:collapse;font-size:13px;margin:0 0 8px 0;">'
        f'<tbody>{facts_rows}</tbody></table>'
        f'{symptoms_html}'
        f'{findings_html}'
        f'{gaps_html}'
        f'{link_block}'
        f'</div></body></html>'
    )


def _format_body_text(
    event: dict,
    meta: dict,
    task: dict,
    journal: dict,
    headline: str,
    console_url: str,
) -> str:
    """Plain-text fallback."""
    lines = ["HyperPod investigation completed", "=" * 32, "", f"Headline: {headline}", ""]
    summary = _parse_summary_text(_verdict_symptom(journal).get("description") or "")
    if summary:
        lines += ["Summary:", summary, ""]
    lines += [
        f"Priority:         {meta['priority'] or 'n/a'}",
        f"Event:            {event.get('detail-type', '')}",
        f"Status:           {meta.get('status', '')}",
        f"Trigger:          {(task.get('reference') or {}).get('title', '—')}",
        f"Investigation ID: {meta.get('task_id', '—')}",
        f"Account:          {event.get('account', '—')}",
        f"Region:           {event.get('region', '—')}",
        f"Created:          {_format_ts(meta.get('created_at', ''))}",
        f"Updated:          {_format_ts(meta.get('updated_at', ''))}",
    ]
    if console_url:
        lines += ["", f"Open investigation: {console_url}"]

    if journal["symptoms"]:
        lines += ["", "-- Symptoms --"]
        for s in journal["symptoms"][:6]:
            lines += ["", (s.get("title") or "").strip(), (s.get("description") or "").strip()[:1500]]
    if journal["findings"]:
        lines += ["", "-- Findings --"]
        for f in journal["findings"]:
            ft = (f.get("finding_type") or "").upper()
            lines += ["", f"[{ft}] {(f.get('title') or '').strip()}", (f.get("description") or "").strip()[:1500]]
    if journal["gaps"]:
        lines += ["", "-- Investigation gaps --"]
        for g in journal["gaps"][:10]:
            lines.append(f"- {g}")

    return "\n".join(lines)


def lambda_handler(event, context):
    detail_type = event.get("detail-type", "")
    print(f"received event detail-type={detail_type!r} id={event.get('id')!r}")

    allowed = _env_list("EMAIL_DETAIL_TYPES", "Investigation Completed")
    if detail_type not in allowed:
        print(f"skipping: detail-type {detail_type!r} not in allowlist {allowed}")
        return {"statusCode": 200, "body": json.dumps({"skipped": True, "reason": "detail-type"})}

    meta = _extract_meta(event)
    if not meta["agent_space_id"] or not meta["execution_id"]:
        print(f"skipping: missing agent_space_id / execution_id")
        return {"statusCode": 200, "body": json.dumps({"skipped": True, "reason": "missing-ids"})}

    force = _env_bool("FORCE_SEND")

    # Cheap dedup check first — before any API calls to DevOps Agent.
    # If we've already emailed for this execution_id, drop the event.
    if not force and _already_emailed(meta["execution_id"]):
        print(f"skipping: already emailed for execution_id={meta['execution_id']}")
        return {"statusCode": 200, "body": json.dumps({"skipped": True, "reason": "already-emailed"})}

    task = _fetch_task(meta["agent_space_id"], meta["task_id"])
    journal = _fetch_journal(meta["agent_space_id"], meta["execution_id"])
    print(f"journal: symptoms={len(journal['symptoms'])} findings={len(journal['findings'])} gaps={len(journal['gaps'])} raw={journal['raw_count']}")

    # Daily heartbeat is a silent liveness signal — visible in the console/logs
    # but never emailed. The audit Lambda marks heartbeat runs deterministically
    # in the task title/description; match either (skill-independent).
    if not force and _is_heartbeat(task):
        print("skipping: daily heartbeat (silent liveness, no email)")
        return {"statusCode": 200, "body": json.dumps({"skipped": True, "reason": "heartbeat"})}

    if not force and _is_suppress_verdict(journal):
        print("skipping: Suppress verdict")
        return {"statusCode": 200, "body": json.dumps({"skipped": True, "reason": "suppress-verdict"})}
    if not force and not _has_actionable_content(journal):
        print("skipping: no findings in journal (set FORCE_SEND=1 to bypass)")
        return {"statusCode": 200, "body": json.dumps({"skipped": True, "reason": "no-findings"})}

    console_url = _console_url(event, meta)
    headline = _pick_headline(journal, task)
    subject = _format_subject(event, headline, meta["priority"])
    body_html = _format_body_html(event, meta, task, journal, headline, console_url)
    body_text = _format_body_text(event, meta, task, journal, headline, console_url)

    recipients = _env_list("EMAIL_RECIPIENTS", "")
    if not recipients:
        raise RuntimeError("EMAIL_RECIPIENTS is empty")
    sender = os.environ.get("EMAIL_SENDER")
    if not sender:
        raise RuntimeError("EMAIL_SENDER is unset")

    print(f"sending email from={sender!r} to={recipients} subject={subject!r} html_bytes={len(body_html)}")
    resp = _ses().send_email(
        Source=sender,
        Destination={"ToAddresses": recipients},
        Message={
            "Subject": {"Data": subject, "Charset": "UTF-8"},
            "Body": {
                "Text": {"Data": body_text, "Charset": "UTF-8"},
                "Html": {"Data": body_html, "Charset": "UTF-8"},
            },
        },
    )
    print(f"ses MessageId={resp['MessageId']}")

    # Record the send in the dedup bucket so subsequent re-emissions for
    # the same execution_id are skipped. Best-effort: a failure here means
    # a duplicate email might slip through later, not a hard error.
    _write_marker(meta["execution_id"], event, meta)

    return {"statusCode": 200, "body": json.dumps({"messageId": resp["MessageId"]})}
