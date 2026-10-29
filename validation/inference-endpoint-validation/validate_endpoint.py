#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Validate an OpenAI-compatible inference endpoint.

Runs 21 contract checks against a serving endpoint and emits a JSON report
plus a Markdown ``## Test Results`` block suitable for an ADT pull request.

Usage::

    python3 validate_endpoint.py --base-url http://localhost:8000 \\
        --json report.json --markdown report.md
"""

import argparse
import dataclasses
import datetime
import http.client
import json
import os
import re
import socket
import ssl
import sys
import threading
import time
import urllib.parse

# --- Constants ---

TOOL_NAME = "inference-endpoint-validation"
TOOL_VERSION = "1.0.0"
SCHEMA_VERSION = 1
DEFAULT_HEALTH_PATH = "/health"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_CONCURRENCY = 8
DEFAULT_TIMEOUT = 60

PROMPT_SHORT = "Say hello in one sentence."
PROMPT_LONG = "Write a short paragraph about distributed AI inference."
PROMPT_COUNT = "Count from 1 to 5, one number per line."

SEED = 1234
ALLOWED_FINISH = {"stop", "length"}

TEMPLATE_MARKERS = [
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|assistant|>",
    "<|user|>",
    "<|system|>",
    "<s>",
    "</s>",
    "[INST]",
    "[/INST]",
    "<<SYS>>",
    "<</SYS>>",
]

UNKNOWN_MODEL = "iev-nonexistent-model"

GET_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and state, e.g. San Francisco, CA",
                }
            },
            "required": ["location"],
        },
    },
}

CAPITAL_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "capital",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "capital": {"type": "string"},
                "country": {"type": "string"},
            },
            "required": ["capital", "country"],
            "additionalProperties": False,
        },
    },
}

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_WARN = "warn"
STATUS_SKIP = "skip"

SEVERITY_REQUIRED = "required"
SEVERITY_ADVISORY = "advisory"

# --- Helpers ---


def redact_secret(text, secret):
    """Replace *secret* with ``***REDACTED***`` in *text*."""
    if not secret or not text:
        return text or ""
    return text.replace(secret, "***REDACTED***")


def extract_text(message_or_delta):
    """Return ``(content, reasoning)`` from a message or delta dict.

    ``content`` may be a string or ``None``; reasoning is read from
    ``reasoning_content`` or ``reasoning`` (engine compatibility).
    """
    if not message_or_delta:
        return ("", "")
    content = message_or_delta.get("content") or ""
    reasoning = (
        message_or_delta.get("reasoning_content")
        or message_or_delta.get("reasoning")
        or ""
    )
    return (content, reasoning)


def validate_usage(usage_dict):
    """Return ``(ok, detail)`` for a ``usage`` object."""
    if not isinstance(usage_dict, dict):
        return (False, "usage is not a dict")
    prompt = usage_dict.get("prompt_tokens")
    completion = usage_dict.get("completion_tokens")
    total = usage_dict.get("total_tokens")
    if not isinstance(prompt, int):
        return (False, "prompt_tokens missing or not int")
    if not isinstance(completion, int):
        return (False, "completion_tokens missing or not int")
    if not isinstance(total, int):
        return (False, "total_tokens missing or not int")
    if total != prompt + completion:
        return (
            False,
            "total_tokens (%d) != prompt_tokens (%d) + completion_tokens (%d)"
            % (total, prompt, completion),
        )
    return (True, "")


def build_url(base_parts, path):
    """Prepend any path prefix from the parsed base URL to *path*."""
    prefix = base_parts.path
    return prefix + path


# --- HttpClient ---


class HttpClient:
    """Thin HTTP client: one connection per request, bearer auth, JSON and SSE."""

    def __init__(self, base_parts, api_key, timeout):
        self._base_parts = base_parts
        self._api_key = api_key
        self._timeout = timeout

    def _connect(self):
        scheme = self._base_parts.scheme
        host = self._base_parts.hostname
        port = self._base_parts.port
        if scheme == "https":
            ctx = ssl.create_default_context()
            if port:
                conn = http.client.HTTPSConnection(host, port, timeout=self._timeout, context=ctx)
            else:
                conn = http.client.HTTPSConnection(host, timeout=self._timeout, context=ctx)
        else:
            if port:
                conn = http.client.HTTPConnection(host, port, timeout=self._timeout)
            else:
                conn = http.client.HTTPConnection(host, timeout=self._timeout)
        return conn

    def _headers(self, content_type=None, accept=None, include_auth=True):
        h = {}
        if content_type:
            h["Content-Type"] = content_type
        if accept:
            h["Accept"] = accept
        if include_auth and self._api_key:
            h["Authorization"] = "Bearer " + self._api_key
        return h

    def request_json(self, method, path, body=None, include_auth=True):
        """Send a JSON request and return ``(status, parsed_body, raw_body)``."""
        conn = self._connect()
        try:
            url = build_url(self._base_parts, path)
            headers = self._headers(
                content_type="application/json" if body is not None else None,
                accept="application/json",
                include_auth=include_auth,
            )
            encoded = None
            if body is not None:
                if isinstance(body, bytes):
                    encoded = body
                else:
                    encoded = json.dumps(body).encode("utf-8")
            conn.request(method, url, body=encoded, headers=headers)
            resp = conn.getresponse()
            status = resp.status
            raw = resp.read()
            try:
                parsed = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                parsed = None
            return (status, parsed, raw)
        finally:
            conn.close()

    def request_raw(self, method, path, body_bytes, content_type="application/json", include_auth=True):
        """Send raw bytes and return ``(status, raw_body)``."""
        conn = self._connect()
        try:
            url = build_url(self._base_parts, path)
            headers = self._headers(
                content_type=content_type,
                accept="application/json",
                include_auth=include_auth,
            )
            conn.request(method, url, body=body_bytes, headers=headers)
            resp = conn.getresponse()
            status = resp.status
            raw = resp.read()
            return (status, raw)
        finally:
            conn.close()

    def request_sse(self, path, body, include_auth=True):
        """Send a streaming POST and return ``(conn, response)`` for SSE reading.

        Caller is responsible for closing the connection.
        """
        conn = self._connect()
        url = build_url(self._base_parts, path)
        headers = self._headers(
            content_type="application/json",
            accept="text/event-stream",
            include_auth=include_auth,
        )
        encoded = json.dumps(body).encode("utf-8")
        conn.request("POST", url, body=encoded, headers=headers)
        resp = conn.getresponse()
        return (conn, resp)

    @staticmethod
    def iter_sse_events(resp):
        """Yield ``(event_type, data, timestamp)`` from an SSE response.

        Reads the raw byte stream line by line, records arrival time per line,
        and handles SSE framing per the spec.
        """
        event_type = None
        data_lines = []
        first_ts = None

        while True:
            raw_line = resp.fp.readline() if resp.fp else b""
            ts = time.perf_counter()
            if not raw_line:
                # End of stream
                if data_lines:
                    yield (event_type, "\n".join(data_lines), first_ts or ts)
                break
            line = raw_line.rstrip(b"\r\n").decode("utf-8", errors="replace")
            if line == "":
                # Blank line: dispatch event
                if data_lines:
                    yield (event_type, "\n".join(data_lines), first_ts or ts)
                event_type = None
                data_lines = []
                first_ts = None
                continue
            if line.startswith(":"):
                # Comment, ignore
                continue
            if ":" in line:
                field, _, value = line.partition(":")
                if value.startswith(" "):
                    value = value[1:]
            else:
                field = line
                value = ""
            if field == "data":
                if first_ts is None:
                    first_ts = ts
                data_lines.append(value)
            elif field == "event":
                event_type = value
            elif field == "id":
                pass
            elif field == "retry":
                pass

    @staticmethod
    def close_for_cancel(conn):
        """Forcefully close a connection (for stream.cancel)."""
        try:
            if conn.sock:
                conn.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            conn.close()
        except OSError:
            pass


# --- ResponseCache ---


class ResponseCache:
    """Memoizes shared requests so multiple checks can reuse them."""

    def __init__(self, client, model):
        self._client = client
        self._model = model
        self._basic = None
        self._basic_repeat = None
        self._stream = None
        self._template_texts = None

    def _chat_body(self, prompt, **kwargs):
        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "seed": SEED,
        }
        body.update(kwargs)
        return body

    def get_basic(self):
        """Non-streaming chat completion (shared)."""
        if self._basic is None:
            body = self._chat_body(PROMPT_SHORT)
            status, parsed, raw = self._client.request_json(
                "POST", "/v1/chat/completions", body
            )
            self._basic = (status, parsed, raw)
        return self._basic

    def get_basic_repeat(self):
        """Second identical non-streaming request (for determinism)."""
        if self._basic_repeat is None:
            body = self._chat_body(PROMPT_SHORT)
            status, parsed, raw = self._client.request_json(
                "POST", "/v1/chat/completions", body
            )
            self._basic_repeat = (status, parsed, raw)
        return self._basic_repeat

    def get_stream(self):
        """Streaming chat completion without stream_options (shared).

        Returns ``(chunks, events_data)`` where *chunks* is a list of parsed
        JSON objects and *events_data* is a list of ``(data_str, timestamp)``.
        """
        if self._stream is None:
            body = self._chat_body(PROMPT_SHORT, stream=True)
            conn, resp = self._client.request_sse("/v1/chat/completions", body)
            try:
                chunks = []
                events_data = []
                for _etype, data_str, ts in HttpClient.iter_sse_events(resp):
                    events_data.append((data_str, ts))
                    if data_str == "[DONE]":
                        continue
                    try:
                        parsed = json.loads(data_str)
                        chunks.append(parsed)
                    except (json.JSONDecodeError, ValueError):
                        pass
                self._stream = (chunks, events_data)
            finally:
                conn.close()
        return self._stream

    def get_template_texts(self):
        """Collect all text from basic and stream for template-leak check."""
        if self._template_texts is None:
            texts = []
            # From basic
            status, parsed, _ = self.get_basic()
            if status == 200 and parsed:
                for choice in parsed.get("choices", []):
                    c, r = extract_text(choice.get("message"))
                    texts.extend([c, r])
            # From stream
            try:
                chunks, _ = self.get_stream()
                for chunk in chunks:
                    for choice in chunk.get("choices", []):
                        c, r = extract_text(choice.get("delta"))
                        texts.extend([c, r])
            except Exception:
                pass
            self._template_texts = texts
        return self._template_texts


# --- Checks ---


def check_models_list(ctx):
    """models.list: GET /v1/models, verify 200 with data array."""
    try:
        status, parsed, raw = ctx["client"].request_json("GET", "/v1/models")
    except Exception as exc:
        return (STATUS_FAIL, "request error: %s" % exc, {})
    if status != 200:
        return (STATUS_FAIL, "expected 200, got %d" % status, {"status": status})
    if not parsed or not isinstance(parsed.get("data"), list):
        return (STATUS_FAIL, "response missing data array", {})
    models = [m.get("id", "") for m in parsed["data"] if isinstance(m, dict)]
    return (STATUS_PASS, "found %d model(s)" % len(models), {"models": models})


def check_server_version(ctx):
    """server_version: GET /version, 404 means null/skip."""
    try:
        status, parsed, raw = ctx["client"].request_json("GET", "/version")
    except Exception as exc:
        return (STATUS_SKIP, "request error: %s" % exc, {})
    if status == 404:
        return (STATUS_SKIP, "not exposed", {})
    if status != 200:
        return (STATUS_SKIP, "status %d" % status, {})
    version = None
    if parsed and isinstance(parsed, dict):
        version = parsed.get("version", str(parsed))
    elif parsed is not None:
        version = str(parsed)
    return (STATUS_PASS, "version: %s" % version, {"version": version})


def check_health_ready(ctx):
    """health.ready: GET health_path, 200 means pass; 404/405 means skip."""
    health_path = ctx["health_path"]
    try:
        status, parsed, raw = ctx["client"].request_json("GET", health_path)
    except Exception as exc:
        return (STATUS_FAIL, "request error: %s" % exc, {})
    if status == 200:
        return (STATUS_PASS, "health endpoint returned 200", {})
    if status in (404, 405):
        return (STATUS_SKIP, "health endpoint not exposed (status %d)" % status, {})
    return (STATUS_FAIL, "expected 200, got %d" % status, {"status": status})


def check_health_consistent(ctx):
    """health.consistent: health pass + chat must both succeed."""
    health_path = ctx["health_path"]
    try:
        h_status, _, _ = ctx["client"].request_json("GET", health_path)
    except Exception as exc:
        return (STATUS_SKIP, "health request error: %s" % exc, {})
    if h_status in (404, 405):
        return (STATUS_SKIP, "health endpoint not exposed", {})
    try:
        body = {
            "model": ctx["model"],
            "messages": [{"role": "user", "content": PROMPT_SHORT}],
            "temperature": 0,
            "seed": SEED,
        }
        c_status, _, _ = ctx["client"].request_json(
            "POST", "/v1/chat/completions", body
        )
    except Exception as exc:
        if h_status == 200:
            return (STATUS_FAIL, "health 200 but chat failed: %s" % exc, {})
        return (STATUS_FAIL, "both health (%d) and chat failed" % h_status, {})
    if h_status == 200 and c_status == 200:
        return (STATUS_PASS, "health and chat both 200", {})
    if h_status == 200 and c_status != 200:
        return (
            STATUS_FAIL,
            "health 200 but chat returned %d" % c_status,
            {"health_status": h_status, "chat_status": c_status},
        )
    return (STATUS_FAIL, "health returned %d" % h_status, {"health_status": h_status, "chat_status": c_status})


def check_chat_basic(ctx):
    """chat.basic: non-streaming chat completion, verify structure."""
    cache = ctx["cache"]
    status, parsed, raw = cache.get_basic()
    if status != 200:
        return (STATUS_FAIL, "expected 200, got %d" % status, {"status": status})
    if not parsed:
        return (STATUS_FAIL, "invalid JSON body", {})
    choices = parsed.get("choices")
    if not isinstance(choices, list) or len(choices) == 0:
        return (STATUS_FAIL, "no choices in response", {})
    choice = choices[0]
    msg = choice.get("message")
    if not isinstance(msg, dict):
        return (STATUS_FAIL, "choice[0].message not a dict", {})
    content, reasoning = extract_text(msg)
    if not content and not reasoning:
        return (STATUS_FAIL, "empty content and no reasoning", {})
    fr = choice.get("finish_reason")
    if fr not in ALLOWED_FINISH:
        return (STATUS_FAIL, "finish_reason '%s' not in %s" % (fr, ALLOWED_FINISH), {})
    usage = parsed.get("usage")
    ok, detail = validate_usage(usage)
    if not ok:
        return (STATUS_FAIL, detail, {})
    return (STATUS_PASS, "valid response", {"finish_reason": fr})


def check_chat_determinism(ctx):
    """chat.determinism: advisory, compare basic with basic_repeat."""
    cache = ctx["cache"]
    s1, p1, _ = cache.get_basic()
    s2, p2, _ = cache.get_basic_repeat()
    if s1 != 200 or s2 != 200:
        return (STATUS_SKIP, "basic requests failed", {})
    c1, r1 = extract_text(p1.get("choices", [{}])[0].get("message"))
    c2, r2 = extract_text(p2.get("choices", [{}])[0].get("message"))
    if c1 == c2 and r1 == r2:
        return (STATUS_PASS, "responses match", {})
    return (STATUS_WARN, "responses differ at temperature=0 seed=%d" % SEED, {})


def check_chat_stream(ctx):
    """chat.stream: streaming chat completion, verify SSE framing."""
    cache = ctx["cache"]
    try:
        chunks, events_data = cache.get_stream()
    except Exception as exc:
        return (STATUS_FAIL, "stream error: %s" % exc, {})
    if not chunks:
        return (STATUS_FAIL, "no chunks received", {})
    # Check first chunk for role
    first = chunks[0]
    first_choices = first.get("choices", [])
    if first_choices:
        delta = first_choices[0].get("delta", {})
        role = delta.get("role")
        if role and role != "assistant":
            return (STATUS_FAIL, "first chunk role '%s' != 'assistant'" % role, {})
    # Collect text and check finish_reason on last chunk
    found_content = False
    finish_reason = None
    for chunk in chunks:
        for choice in chunk.get("choices", []):
            c, r = extract_text(choice.get("delta"))
            if c or r:
                found_content = True
            fr = choice.get("finish_reason")
            if fr is not None:
                finish_reason = fr
    if not found_content:
        return (STATUS_FAIL, "no delta content in stream", {})
    if finish_reason not in ALLOWED_FINISH:
        return (
            STATUS_FAIL,
            "finish_reason '%s' not in %s" % (finish_reason, ALLOWED_FINISH),
            {},
        )
    # Check [DONE] sentinel
    has_done = any(d == "[DONE]" for d, _ in events_data)
    if not has_done:
        return (STATUS_FAIL, "[DONE] sentinel missing", {})
    return (STATUS_PASS, "valid SSE stream", {"finish_reason": finish_reason})


def check_chat_stream_usage(ctx):
    """chat.stream_usage: advisory, stream with include_usage, verify final usage."""
    client = ctx["client"]
    body = {
        "model": ctx["model"],
        "messages": [{"role": "user", "content": PROMPT_SHORT}],
        "temperature": 0,
        "seed": SEED,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    try:
        conn, resp = client.request_sse("/v1/chat/completions", body)
    except Exception as exc:
        return (STATUS_WARN, "request error: %s" % exc, {})
    try:
        chunks = []
        for _etype, data_str, _ts in HttpClient.iter_sse_events(resp):
            if data_str == "[DONE]":
                continue
            try:
                chunks.append(json.loads(data_str))
            except (json.JSONDecodeError, ValueError):
                pass
    finally:
        conn.close()
    if not chunks:
        return (STATUS_WARN, "no chunks received", {})
    # Look for usage in last chunk(s)
    found_usage = False
    for chunk in reversed(chunks):
        usage = chunk.get("usage")
        if usage:
            ok, detail = validate_usage(usage)
            if not ok:
                return (STATUS_WARN, "stream usage invalid: %s" % detail, {})
            found_usage = True
            break
    if not found_usage:
        return (STATUS_WARN, "no usage in final stream chunk", {})
    return (STATUS_PASS, "stream usage present", {})


def check_chat_stream_parity(ctx):
    """chat.stream_parity: advisory, compare assembled stream with non-stream."""
    cache = ctx["cache"]
    s1, p1, _ = cache.get_basic()
    s2, p2, _ = cache.get_basic_repeat()
    if s1 != 200 or s2 != 200:
        return (STATUS_SKIP, "basic requests failed", {})
    c1, r1 = extract_text(p1.get("choices", [{}])[0].get("message"))
    c2, r2 = extract_text(p2.get("choices", [{}])[0].get("message"))
    if c1 != c2 or r1 != r2:
        return (STATUS_SKIP, "basic != basic_repeat; skipping parity", {})
    # Assemble stream text
    try:
        chunks, _ = cache.get_stream()
    except Exception:
        return (STATUS_SKIP, "stream failed", {})
    stream_content = ""
    stream_reasoning = ""
    for chunk in chunks:
        for choice in chunk.get("choices", []):
            c, r = extract_text(choice.get("delta"))
            stream_content += c
            stream_reasoning += r
    if c1 == stream_content and r1 == stream_reasoning:
        return (STATUS_PASS, "stream and non-stream match", {})
    return (STATUS_WARN, "stream and non-stream text differ", {})


def check_chat_max_tokens(ctx):
    """chat.max_tokens: POST with max_tokens:1, verify short completion."""
    client = ctx["client"]
    body = {
        "model": ctx["model"],
        "messages": [{"role": "user", "content": PROMPT_LONG}],
        "temperature": 0,
        "seed": SEED,
        "max_tokens": 1,
    }
    try:
        status, parsed, raw = client.request_json(
            "POST", "/v1/chat/completions", body
        )
    except Exception as exc:
        return (STATUS_FAIL, "request error: %s" % exc, {})
    if status != 200:
        # Try max_completion_tokens
        body2 = dict(body)
        del body2["max_tokens"]
        body2["max_completion_tokens"] = 1
        try:
            status, parsed, raw = client.request_json(
                "POST", "/v1/chat/completions", body2
            )
        except Exception as exc:
            return (STATUS_FAIL, "request error: %s" % exc, {})
        if status != 200:
            return (STATUS_FAIL, "expected 200, got %d" % status, {})
    if not parsed:
        return (STATUS_FAIL, "invalid JSON body", {})
    choices = parsed.get("choices", [])
    if not choices:
        return (STATUS_FAIL, "no choices", {})
    usage = parsed.get("usage")
    if usage and isinstance(usage, dict):
        ct = usage.get("completion_tokens")
        if isinstance(ct, int) and ct > 1:
            return (
                STATUS_FAIL,
                "completion_tokens %d > 1" % ct,
                {"completion_tokens": ct},
            )
    return (STATUS_PASS, "max_tokens respected", {})


def check_chat_stop(ctx):
    """chat.stop: POST with stop sequence, verify finish_reason='stop'."""
    client = ctx["client"]
    # Use a prompt that should produce "Hello" and stop at "Hello"
    body = {
        "model": ctx["model"],
        "messages": [
            {"role": "user", "content": "Reply with exactly: Hello World"}
        ],
        "temperature": 0,
        "seed": SEED,
        "stop": ["World"],
    }
    try:
        status, parsed, raw = client.request_json(
            "POST", "/v1/chat/completions", body
        )
    except Exception as exc:
        return (STATUS_FAIL, "request error: %s" % exc, {})
    if status != 200:
        return (STATUS_FAIL, "expected 200, got %d" % status, {})
    if not parsed:
        return (STATUS_FAIL, "invalid JSON body", {})
    choices = parsed.get("choices", [])
    if not choices:
        return (STATUS_FAIL, "no choices", {})
    choice = choices[0]
    fr = choice.get("finish_reason")
    if fr != "stop":
        return (STATUS_FAIL, "finish_reason '%s' != 'stop'" % fr, {})
    content, reasoning = extract_text(choice.get("message"))
    text = content + reasoning
    if "World" in text:
        return (STATUS_FAIL, "stop word 'World' found in output", {"text": text})
    return (STATUS_PASS, "stop sequence respected", {"finish_reason": fr})


def check_chat_template_leak(ctx):
    """chat.template_leak: check none of basic/stream contain TEMPLATE_MARKERS."""
    texts = ctx["cache"].get_template_texts()
    combined = " ".join(t for t in texts if t)
    for marker in TEMPLATE_MARKERS:
        if marker in combined:
            return (STATUS_FAIL, "template marker '%s' found in output" % marker, {"marker": marker})
    return (STATUS_PASS, "no template markers in output", {})


def check_completions_basic(ctx):
    """completions.basic: POST /v1/completions, 404/405 means skip."""
    client = ctx["client"]
    body = {
        "model": ctx["model"],
        "prompt": PROMPT_SHORT,
        "temperature": 0,
        "seed": SEED,
        "max_tokens": 64,
    }
    try:
        status, parsed, raw = client.request_json(
            "POST", "/v1/completions", body
        )
    except Exception as exc:
        return (STATUS_FAIL, "request error: %s" % exc, {})
    if status in (404, 405):
        return (STATUS_SKIP, "completions endpoint not exposed", {})
    if status != 200:
        return (STATUS_FAIL, "expected 200, got %d" % status, {"status": status})
    if not parsed:
        return (STATUS_FAIL, "invalid JSON body", {})
    choices = parsed.get("choices", [])
    if not isinstance(choices, list) or len(choices) == 0:
        return (STATUS_FAIL, "no choices in response", {})
    return (STATUS_PASS, "completions endpoint works", {})


def check_tools_call(ctx):
    """tools.call: POST with tools and named tool_choice, verify tool_calls."""
    client = ctx["client"]
    body = {
        "model": ctx["model"],
        "messages": [
            {"role": "user", "content": "What is the weather in San Francisco?"}
        ],
        "temperature": 0,
        "seed": SEED,
        "tools": [GET_WEATHER_TOOL],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
    }
    try:
        status, parsed, raw = client.request_json(
            "POST", "/v1/chat/completions", body
        )
    except Exception as exc:
        return (STATUS_FAIL, "request error: %s" % exc, {})
    if status in (400, 404, 422, 501):
        return (STATUS_SKIP, "tool calling not enabled (status %d)" % status, {})
    if status != 200:
        return (STATUS_FAIL, "expected 200, got %d" % status, {"status": status})
    if not parsed:
        return (STATUS_FAIL, "invalid JSON body", {})
    choices = parsed.get("choices", [])
    if not choices:
        return (STATUS_FAIL, "no choices", {})
    choice = choices[0]
    tool_calls = choice.get("message", {}).get("tool_calls")
    if isinstance(tool_calls, list) and len(tool_calls) > 0:
        return (STATUS_PASS, "tool call returned", {"tool_calls": len(tool_calls)})
    # Check if the model returned the tool call as plain text
    content, _ = extract_text(choice.get("message"))
    if content and ("<tool_call>" in content or '"name"' in content):
        return (
            STATUS_FAIL,
            "tool call in content, not tool_calls; check tool-call parser",
            {},
        )
    return (STATUS_FAIL, "no tool_calls in response", {})


def check_structured_output(ctx):
    """structured_output: POST with response_format json_schema."""
    client = ctx["client"]
    body = {
        "model": ctx["model"],
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "temperature": 0,
        "seed": SEED,
        "response_format": CAPITAL_SCHEMA,
    }
    try:
        status, parsed, raw = client.request_json(
            "POST", "/v1/chat/completions", body
        )
    except Exception as exc:
        return (STATUS_FAIL, "request error: %s" % exc, {})
    if status in (400, 404, 422, 501):
        return (STATUS_SKIP, "structured output not enabled (status %d)" % status, {})
    if status != 200:
        return (STATUS_FAIL, "expected 200, got %d" % status, {"status": status})
    if not parsed:
        return (STATUS_FAIL, "invalid JSON body", {})
    choices = parsed.get("choices", [])
    if not choices:
        return (STATUS_FAIL, "no choices", {})
    content, _ = extract_text(choices[0].get("message"))
    try:
        obj = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return (STATUS_FAIL, "content is not valid JSON", {})
    if "capital" not in obj:
        return (STATUS_FAIL, "'capital' key missing from JSON output", {})
    return (STATUS_PASS, "valid structured output", {})


def check_errors_unknown_model(ctx):
    """errors.unknown_model: POST with UNKNOWN_MODEL, expect 4xx."""
    client = ctx["client"]
    body = {
        "model": UNKNOWN_MODEL,
        "messages": [{"role": "user", "content": PROMPT_SHORT}],
        "temperature": 0,
    }
    try:
        status, parsed, raw = client.request_json(
            "POST", "/v1/chat/completions", body
        )
    except Exception as exc:
        return (STATUS_FAIL, "request error: %s" % exc, {})
    if 400 <= status <= 499:
        return (STATUS_PASS, "got %d for unknown model" % status, {"status": status})
    return (
        STATUS_FAIL,
        "expected 4xx, got %d" % status,
        {"status": status},
    )


def check_errors_malformed_json(ctx):
    """errors.malformed_json: POST raw truncated JSON, expect 4xx."""
    client = ctx["client"]
    raw_body = b'{"model": '
    try:
        status, raw = client.request_raw(
            "POST", "/v1/chat/completions", raw_body
        )
    except Exception as exc:
        return (STATUS_FAIL, "request error: %s" % exc, {})
    if 400 <= status <= 499:
        return (STATUS_PASS, "got %d for malformed JSON" % status, {"status": status})
    return (
        STATUS_FAIL,
        "expected 4xx, got %d" % status,
        {"status": status},
    )


def check_auth_required(ctx):
    """auth.required: if API key set, request without auth, expect 401/403."""
    if not ctx["api_key"]:
        return (STATUS_SKIP, "no API key configured", {})
    client = ctx["client"]
    body = {
        "model": ctx["model"],
        "messages": [{"role": "user", "content": PROMPT_SHORT}],
        "temperature": 0,
    }
    try:
        status, parsed, raw = client.request_json(
            "POST", "/v1/chat/completions", body, include_auth=False
        )
    except Exception as exc:
        return (STATUS_FAIL, "request error: %s" % exc, {})
    if status in (401, 403):
        return (STATUS_PASS, "unauthenticated request rejected (%d)" % status, {})
    return (
        STATUS_FAIL,
        "expected 401/403 without auth, got %d" % status,
        {"status": status},
    )


def check_stream_cancel(ctx):
    """stream.cancel: close socket mid-stream, verify follow-up succeeds."""
    client = ctx["client"]
    body = {
        "model": ctx["model"],
        "messages": [{"role": "user", "content": PROMPT_LONG}],
        "temperature": 0,
        "seed": SEED,
        "stream": True,
    }
    # Start streaming
    try:
        conn, resp = client.request_sse("/v1/chat/completions", body)
    except Exception as exc:
        return (STATUS_FAIL, "stream request error: %s" % exc, {})
    # Read until first non-empty delta text, then cancel
    try:
        for _etype, data_str, _ts in HttpClient.iter_sse_events(resp):
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except (json.JSONDecodeError, ValueError):
                continue
            for choice in chunk.get("choices", []):
                c, r = extract_text(choice.get("delta"))
                if c or r:
                    # Got first content, cancel now
                    HttpClient.close_for_cancel(conn)
                    raise _CancelSentinel()
    except _CancelSentinel:
        pass
    except OSError:
        pass
    # Sleep then follow-up
    time.sleep(0.5)
    try:
        f_status, f_parsed, f_raw = client.request_json(
            "POST",
            "/v1/chat/completions",
            {
                "model": ctx["model"],
                "messages": [{"role": "user", "content": PROMPT_SHORT}],
                "temperature": 0,
                "seed": SEED,
            },
        )
    except Exception as exc:
        return (STATUS_FAIL, "follow-up after cancel failed: %s" % exc, {})
    if f_status == 200:
        return (STATUS_PASS, "endpoint healthy after stream cancel", {})
    return (
        STATUS_FAIL,
        "follow-up returned %d after cancel" % f_status,
        {"status": f_status},
    )


class _CancelSentinel(Exception):
    """Internal signal for stream.cancel flow control."""


def check_concurrency_parallel(ctx):
    """concurrency.parallel: fire N parallel requests, verify all succeed."""
    client = ctx["client"]
    n = ctx["concurrency"]
    barrier = threading.Barrier(n, timeout=ctx["timeout"])
    results = [None] * n

    def _worker(idx):
        try:
            barrier.wait()
            body = {
                "model": ctx["model"],
                "messages": [{"role": "user", "content": PROMPT_COUNT}],
                "temperature": 0,
                "seed": SEED,
            }
            status, parsed, raw = client.request_json(
                "POST", "/v1/chat/completions", body
            )
            results[idx] = (status, parsed)
        except Exception as exc:
            results[idx] = (None, str(exc))

    threads = []
    for i in range(n):
        t = threading.Thread(target=_worker, args=(i,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join(timeout=ctx["timeout"] + 10)
    # Check results
    failed = []
    ids = []
    for i, r in enumerate(results):
        if r is None:
            failed.append("thread %d: no result" % i)
            continue
        status, parsed = r
        if status != 200:
            failed.append("thread %d: status %s" % (i, status))
            continue
        if isinstance(parsed, dict):
            rid = parsed.get("id", "")
            ids.append(rid)
    if failed:
        return (STATUS_FAIL, "; ".join(failed), {})
    # Check unique IDs
    if len(set(ids)) != len(ids):
        return (STATUS_FAIL, "duplicate response IDs", {"ids": ids})
    return (STATUS_PASS, "all %d requests succeeded" % n, {"unique_ids": len(set(ids))})


def check_latency_smoke(ctx):
    """latency.smoke: measure TTFT and total from a single stream, always pass."""
    client = ctx["client"]
    body = {
        "model": ctx["model"],
        "messages": [{"role": "user", "content": PROMPT_SHORT}],
        "temperature": 0,
        "seed": SEED,
        "stream": True,
    }
    t0 = time.perf_counter()
    try:
        conn, resp = client.request_sse("/v1/chat/completions", body)
    except Exception as exc:
        return (STATUS_PASS, "stream failed, no latency: %s" % exc, {})
    try:
        ttft = None
        for _etype, data_str, ts in HttpClient.iter_sse_events(resp):
            if data_str == "[DONE]":
                continue
            try:
                chunk = json.loads(data_str)
            except (json.JSONDecodeError, ValueError):
                continue
            if ttft is None:
                for choice in chunk.get("choices", []):
                    c, r = extract_text(choice.get("delta"))
                    if c or r:
                        ttft = ts - t0
                        break
    finally:
        conn.close()
    total = time.perf_counter() - t0
    ttft_ms = int(ttft * 1000) if ttft is not None else None
    total_ms = int(total * 1000)
    detail = "TTFT=%sms total=%dms" % (
        str(ttft_ms) if ttft_ms is not None else "n/a",
        total_ms,
    )
    return (STATUS_PASS, detail, {"ttft_ms": ttft_ms, "total_ms": total_ms})


CHECKS = [
    ("models.list", SEVERITY_REQUIRED, check_models_list),
    ("server_version", SEVERITY_REQUIRED, check_server_version),
    ("health.ready", SEVERITY_REQUIRED, check_health_ready),
    ("health.consistent", SEVERITY_REQUIRED, check_health_consistent),
    ("chat.basic", SEVERITY_REQUIRED, check_chat_basic),
    ("chat.determinism", SEVERITY_ADVISORY, check_chat_determinism),
    ("chat.stream", SEVERITY_REQUIRED, check_chat_stream),
    ("chat.stream_usage", SEVERITY_ADVISORY, check_chat_stream_usage),
    ("chat.stream_parity", SEVERITY_ADVISORY, check_chat_stream_parity),
    ("chat.max_tokens", SEVERITY_REQUIRED, check_chat_max_tokens),
    ("chat.stop", SEVERITY_REQUIRED, check_chat_stop),
    ("chat.template_leak", SEVERITY_REQUIRED, check_chat_template_leak),
    ("completions.basic", SEVERITY_REQUIRED, check_completions_basic),
    ("tools.call", SEVERITY_REQUIRED, check_tools_call),
    ("structured_output", SEVERITY_REQUIRED, check_structured_output),
    ("errors.unknown_model", SEVERITY_REQUIRED, check_errors_unknown_model),
    ("errors.malformed_json", SEVERITY_REQUIRED, check_errors_malformed_json),
    ("auth.required", SEVERITY_REQUIRED, check_auth_required),
    ("stream.cancel", SEVERITY_REQUIRED, check_stream_cancel),
    ("concurrency.parallel", SEVERITY_REQUIRED, check_concurrency_parallel),
    ("latency.smoke", SEVERITY_REQUIRED, check_latency_smoke),
]

ALL_CHECK_IDS = [c[0] for c in CHECKS]

# --- CheckRunner ---


@dataclasses.dataclass
class CheckResult:
    """Result of one check."""

    id: str
    severity: str
    status: str
    detail: str
    duration_ms: int
    evidence: dict


@dataclasses.dataclass
class RunOptions:
    """Parsed command-line options."""

    base_url: str
    model: str
    api_key: str
    api_key_env: str
    health_path: str
    checks: list
    skip: list
    require: list
    strict: bool
    timeout: int
    concurrency: int
    json_path: str
    markdown_path: str


class CheckRunner:
    """Selects, executes, and scores checks."""

    def __init__(self, opts):
        self._opts = opts
        self._results = []
        self._error = None
        self._started_at = None
        self._duration_ms = 0
        self._base_parts = urllib.parse.urlsplit(opts.base_url)

        # Build selection
        self._selected = set()
        if opts.checks:
            self._selected = set(opts.checks)
        else:
            self._selected = set(ALL_CHECK_IDS)
        if opts.skip:
            self._selected -= set(opts.skip)

        # Build severity map with --require promotion
        self._severity_map = {}
        for cid, sev, _fn in CHECKS:
            self._severity_map[cid] = sev
        if opts.require:
            for cid in opts.require:
                self._severity_map[cid] = SEVERITY_REQUIRED

    def run(self):
        """Execute all selected checks in order."""
        self._started_at = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        run_start = time.perf_counter()

        client = HttpClient(self._base_parts, self._opts.api_key, self._opts.timeout)

        # Discover model
        model = self._opts.model
        if not model:
            try:
                status, parsed, raw = client.request_json("GET", "/v1/models")
                if status == 200 and parsed and isinstance(parsed.get("data"), list):
                    data = parsed["data"]
                    if data and isinstance(data[0], dict):
                        model = data[0].get("id", "")
            except (ConnectionError, socket.gaierror, ssl.SSLError,
                    socket.timeout, TimeoutError, http.client.HTTPException,
                    OSError) as exc:
                self._error = "unreachable: %s" % exc
                self._set_all_skip()
                self._duration_ms = int((time.perf_counter() - run_start) * 1000)
                return
            except Exception as exc:
                self._error = "unreachable: %s" % exc
                self._set_all_skip()
                self._duration_ms = int((time.perf_counter() - run_start) * 1000)
                return
        if not model:
            model = "default"

        cache = ResponseCache(client, model)

        ctx = {
            "client": client,
            "model": model,
            "cache": cache,
            "api_key": self._opts.api_key,
            "health_path": self._opts.health_path,
            "timeout": self._opts.timeout,
            "concurrency": self._opts.concurrency,
        }

        for cid, _sev, fn in CHECKS:
            severity = self._severity_map.get(cid, _sev)
            if cid not in self._selected:
                self._results.append(
                    CheckResult(
                        id=cid,
                        severity=severity,
                        status=STATUS_SKIP,
                        detail="not selected",
                        duration_ms=0,
                        evidence={},
                    )
                )
                continue
            t0 = time.perf_counter()
            try:
                status, detail, evidence = fn(ctx)
            except Exception as exc:
                status = STATUS_FAIL
                detail = "exception: %s" % exc
                evidence = {}
            dur = int((time.perf_counter() - t0) * 1000)
            # Map advisory status: advisory checks can only pass, warn, or skip
            if severity == SEVERITY_ADVISORY and status == STATUS_FAIL:
                status = STATUS_WARN
            self._results.append(
                CheckResult(
                    id=cid,
                    severity=severity,
                    status=status,
                    detail=detail,
                    duration_ms=dur,
                    evidence=evidence if isinstance(evidence, dict) else {},
                )
            )

        self._duration_ms = int((time.perf_counter() - run_start) * 1000)

    def _set_all_skip(self):
        """Mark all checks as skip (unreachable endpoint)."""
        for cid, _sev, _fn in CHECKS:
            severity = self._severity_map.get(cid, _sev)
            self._results.append(
                CheckResult(
                    id=cid,
                    severity=severity,
                    status=STATUS_SKIP,
                    detail=self._error or "unreachable",
                    duration_ms=0,
                    evidence={},
                )
            )

    @property
    def exit_code(self):
        """Compute exit code: 0=pass, 1=failure, 2=unreachable/usage."""
        if self._error:
            return 2
        has_fail = False
        has_warn = False
        for r in self._results:
            if r.status == STATUS_FAIL:
                if self._severity_map.get(r.id, r.severity) == SEVERITY_REQUIRED:
                    has_fail = True
            if r.status == STATUS_WARN:
                has_warn = True
        if has_fail:
            return 1
        if self._opts.strict and has_warn:
            return 1
        return 0

    @property
    def results(self):
        return list(self._results)

    @property
    def summary(self):
        counts = {STATUS_PASS: 0, STATUS_FAIL: 0, STATUS_WARN: 0, STATUS_SKIP: 0}
        for r in self._results:
            counts[r.status] = counts.get(r.status, 0) + 1
        return {
            "pass": counts[STATUS_PASS],
            "fail": counts[STATUS_FAIL],
            "warn": counts[STATUS_WARN],
            "skip": counts[STATUS_SKIP],
            "total": len(self._results),
            "duration_ms": self._duration_ms,
        }

    @property
    def started_at(self):
        return self._started_at

    @property
    def model(self):
        for r in self._results:
            if r.id == "models.list" and r.evidence.get("models"):
                return r.evidence["models"][0] if r.evidence["models"] else "unknown"
        return self._opts.model or "unknown"

    @property
    def error(self):
        return self._error


# --- Reporters ---


def report_json(runner, opts):
    """Write the JSON report."""
    secret = opts.api_key
    data = {
        "tool_name": TOOL_NAME,
        "tool_version": TOOL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "started_at": runner.started_at,
        "base_url": redact_secret(opts.base_url, secret),
        "model": runner.model,
        "checks": [],
        "summary": runner.summary,
        "exit_code": runner.exit_code,
    }
    if runner.error:
        data["error"] = redact_secret(runner.error, secret)
    for r in runner.results:
        data["checks"].append(
            {
                "id": r.id,
                "severity": r.severity,
                "status": r.status,
                "detail": redact_secret(r.detail, secret),
                "duration_ms": r.duration_ms,
            }
        )
    raw = json.dumps(data, indent=2)
    raw = redact_secret(raw, secret)
    if opts.json_path:
        with open(opts.json_path, "w", encoding="utf-8") as f:
            f.write(raw)
            f.write("\n")
    return raw


def report_markdown(runner, opts):
    """Write the Markdown report."""
    secret = opts.api_key
    lines = []
    lines.append("## Test Results")
    lines.append("")
    lines.append("| Check | Severity | Status | Detail |")
    lines.append("|-------|----------|--------|--------|")
    for r in runner.results:
        detail = redact_secret(r.detail, secret).replace("|", "\\|")
        lines.append(
            "| `%s` | %s | %s | %s |" % (r.id, r.severity, r.status, detail)
        )
    lines.append("")
    s = runner.summary
    lines.append(
        "Summary: %d pass, %d fail, %d warn, %d skip"
        % (s["pass"], s["fail"], s["warn"], s["skip"])
    )
    lines.append("")
    md = "\n".join(lines)
    md = redact_secret(md, secret)
    if opts.markdown_path:
        with open(opts.markdown_path, "w", encoding="utf-8") as f:
            f.write(md)
    return md


def report_stdout(runner, opts):
    """Print results to stdout."""
    secret = opts.api_key
    is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    def _color(status):
        if not is_tty:
            return status
        colors = {
            STATUS_PASS: "\033[32m%s\033[0m",
            STATUS_FAIL: "\033[31m%s\033[0m",
            STATUS_WARN: "\033[33m%s\033[0m",
            STATUS_SKIP: "\033[90m%s\033[0m",
        }
        return colors.get(status, "%s") % status

    if runner.error:
        print(redact_secret("Error: %s" % runner.error, secret), file=sys.stderr)

    # Header
    print("%-25s %-10s %-6s %s" % ("CHECK", "SEVERITY", "STATUS", "DETAIL"))
    print("-" * 78)
    for r in runner.results:
        detail = redact_secret(r.detail, secret)
        print(
            "%-25s %-10s %-6s %s" % (r.id, r.severity, _color(r.status), detail)
        )
    print("-" * 78)
    s = runner.summary
    print(
        "Summary: %d pass, %d fail, %d warn, %d skip (%.1fs)"
        % (s["pass"], s["fail"], s["warn"], s["skip"], s["duration_ms"] / 1000)
    )


# --- Main ---


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        prog="validate_endpoint",
        description="Validate an OpenAI-compatible inference endpoint.",
    )
    parser.add_argument(
        "--base-url", required=True, help="Base URL of the endpoint (e.g. http://localhost:8000)"
    )
    parser.add_argument(
        "--model", default=None, help="Model name. Auto-detected from /v1/models if omitted."
    )
    parser.add_argument(
        "--api-key-env",
        default=DEFAULT_API_KEY_ENV,
        help="Environment variable holding the API key (default: %s)" % DEFAULT_API_KEY_ENV,
    )
    parser.add_argument(
        "--health-path",
        default=DEFAULT_HEALTH_PATH,
        help="Health endpoint path (default: %s)" % DEFAULT_HEALTH_PATH,
    )
    parser.add_argument(
        "--checks",
        default=None,
        help="Comma-separated list of check IDs to run (default: all)",
    )
    parser.add_argument(
        "--skip",
        default=None,
        help="Comma-separated list of check IDs to skip",
    )
    parser.add_argument(
        "--require",
        default=None,
        help="Comma-separated list of advisory check IDs to promote to required",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 on any warning",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Request timeout in seconds (default: %d)" % DEFAULT_TIMEOUT,
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Number of parallel requests (default: %d)" % DEFAULT_CONCURRENCY,
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help="Write JSON report to this file",
    )
    parser.add_argument(
        "--markdown",
        dest="markdown_path",
        default=None,
        help="Write Markdown report to this file",
    )
    args = parser.parse_args()

    # Parse and validate base-url
    base_url = args.base_url.rstrip("/")
    parts = urllib.parse.urlsplit(base_url)
    if parts.scheme not in ("http", "https"):
        print("Error: --base-url must use http or https scheme", file=sys.stderr)
        sys.exit(2)
    if not parts.hostname:
        print("Error: --base-url must include a hostname", file=sys.stderr)
        sys.exit(2)

    # Read API key from environment
    api_key = os.environ.get(args.api_key_env, "")

    # Parse comma-separated lists
    checks = [c.strip() for c in args.checks.split(",") if c.strip()] if args.checks else []
    skip = [c.strip() for c in args.skip.split(",") if c.strip()] if args.skip else []
    require = [c.strip() for c in args.require.split(",") if c.strip()] if args.require else []

    # Validate check IDs
    all_ids = set(ALL_CHECK_IDS)
    for cid in checks + skip + require:
        if cid not in all_ids:
            print("Error: unknown check ID '%s'" % cid, file=sys.stderr)
            sys.exit(2)

    opts = RunOptions(
        base_url=base_url,
        model=args.model or "",
        api_key=api_key,
        api_key_env=args.api_key_env,
        health_path=args.health_path,
        checks=checks,
        skip=skip,
        require=require,
        strict=args.strict,
        timeout=args.timeout,
        concurrency=args.concurrency,
        json_path=args.json_path,
        markdown_path=args.markdown_path,
    )

    runner = CheckRunner(opts)
    try:
        runner.run()
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        sys.exit(2)

    report_stdout(runner, opts)
    if opts.json_path:
        report_json(runner, opts)
    if opts.markdown_path:
        report_markdown(runner, opts)

    sys.exit(runner.exit_code)


if __name__ == "__main__":
    main()
