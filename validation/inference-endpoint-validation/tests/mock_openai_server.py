# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Deterministic mock OpenAI-compatible server for offline acceptance testing.

Supports conforming mode, minimal mode (--minimal), API-key enforcement
(--require-api-key), and single injectable defects (--defect <name>).

Usage::

    python3 tests/mock_openai_server.py --port 0
    python3 tests/mock_openai_server.py --port 8000 --defect stop_ignored
    python3 tests/mock_openai_server.py --port 0 --minimal
    python3 tests/mock_openai_server.py --port 0 --require-api-key test-secret
"""

import argparse
import http.server
import json
import socket
import sys
import threading
import time

# --- Constants ---

_MODEL_ID = "mock-model-v1"
_VERSION = "0.1.0-mock"

# Deterministic response texts (short prompt)
_SHORT_CONTENT = "Hello! How can I help you today?"
_LONG_CONTENT = (
    "Distributed AI inference allows models to be served across multiple nodes, "
    "reducing latency and increasing throughput for real-world applications."
)
_COUNT_CONTENT = "1\n2\n3\n4\n5"

# Structured output
_CAPITAL_JSON = '{"capital": "Paris", "country": "France"}'

# Tool call
_TOOL_CALL = {
    "id": "call_mock_001",
    "type": "function",
    "function": {
        "name": "get_weather",
        "arguments": '{"location": "San Francisco, CA"}',
    },
}

# Usage defaults
_PROMPT_TOKENS = 12
_COMPLETION_TOKENS = 8
_TOTAL_TOKENS = _PROMPT_TOKENS + _COMPLETION_TOKENS


def _usage(prompt=_PROMPT_TOKENS, completion=_COMPLETION_TOKENS):
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _make_chat_response(content, finish_reason="stop", model=_MODEL_ID,
                         tool_calls=None, usage=None, resp_id="chatcmpl-mock001"):
    msg = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    choice = {
        "index": 0,
        "message": msg,
        "finish_reason": finish_reason,
    }
    return {
        "id": resp_id,
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [choice],
        "usage": usage or _usage(),
    }


def _sse_frame(data_str):
    return "data: %s\n\n" % data_str


def _make_stream_chunks(content, model=_MODEL_ID, finish_reason="stop",
                         include_usage=False, resp_id="chatcmpl-mock001"):
    """Return a list of SSE data strings for a streaming response."""
    chunks = []
    # First chunk: role
    chunks.append(json.dumps({
        "id": resp_id,
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
    }))
    # Content chunks: one per word
    words = content.split(" ") if content else [""]
    for i, word in enumerate(words):
        text = word if i == 0 else " " + word
        chunks.append(json.dumps({
            "id": resp_id,
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
        }))
    # Final chunk: finish_reason
    final = {
        "id": resp_id,
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }
    if include_usage:
        final["usage"] = _usage()
    chunks.append(json.dumps(final))
    return chunks


class _RequestCounter:
    """Thread-safe request counter for unique IDs."""

    def __init__(self):
        self._lock = threading.Lock()
        self._count = 0

    def next_id(self):
        with self._lock:
            self._count += 1
            return "chatcmpl-mock%05d" % self._count


_counter = _RequestCounter()


class MockHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for the mock OpenAI-compatible server."""

    # Suppress default logging
    def log_message(self, format, *args):
        pass

    def _send_json(self, status, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, chunks):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for chunk_str in chunks:
            frame = _sse_frame(chunk_str)
            try:
                self.wfile.write(frame.encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return
        # Send [DONE]
        try:
            self.wfile.write(_sse_frame("[DONE]").encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _check_auth(self):
        required_key = self.server.require_api_key
        if not required_key:
            return True
        auth = self.headers.get("Authorization", "")
        if auth == "Bearer " + required_key:
            return True
        self._send_json(401, {"error": {"message": "Unauthorized", "type": "auth_error", "code": 401}})
        return False

    def _get_defect(self):
        return self.server.defect

    def _is_minimal(self):
        return self.server.minimal

    def do_GET(self):
        if not self._check_auth():
            return

        path = self.path.split("?")[0]

        if path == "/v1/models":
            self._send_json(200, {
                "object": "list",
                "data": [{"id": _MODEL_ID, "object": "model", "created": 1700000000, "owned_by": "mock"}],
            })
            return

        if path == "/version":
            if self._is_minimal():
                self._send_json(404, {"error": "not found"})
                return
            self._send_json(200, {"version": _VERSION})
            return

        if path == "/health" or path == self.server.health_path:
            if self._is_minimal():
                self._send_json(404, {"error": "not found"})
                return
            defect = self._get_defect()
            if defect == "health_inconsistent":
                # Health says OK but chat will fail
                self._send_json(200, {"status": "ok"})
                return
            self._send_json(200, {"status": "ok"})
            return

        self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if not self._check_auth():
            return

        path = self.path.split("?")[0]
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length) if content_length > 0 else b""

        # Try to parse JSON body
        body = None
        try:
            body = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            if self._get_defect() == "malformed_json_accepts":
                # Defect: accept malformed JSON with a 200 response
                resp_id = _counter.next_id()
                self._send_json(200, _make_chat_response(
                    content=_SHORT_CONTENT, finish_reason="stop",
                    usage=_usage(), resp_id=resp_id,
                ))
                return
            self._send_json(400, {"error": {"message": "malformed JSON", "type": "invalid_request_error"}})
            return

        if path == "/v1/chat/completions":
            self._handle_chat(body)
            return

        if path == "/v1/completions":
            if self._is_minimal():
                self._send_json(404, {"error": "not found"})
                return
            self._handle_completions(body)
            return

        self._send_json(404, {"error": "not found"})

    def _handle_chat(self, body):
        defect = self._get_defect()
        model = body.get("model", _MODEL_ID)
        stream = body.get("stream", False)
        include_usage = False
        stream_opts = body.get("stream_options")
        if isinstance(stream_opts, dict):
            include_usage = stream_opts.get("include_usage", False)

        # Unknown model check
        if model == "iev-nonexistent-model":
            if defect == "unknown_model_accepts":
                # Defect: accept unknown model instead of rejecting
                pass
            else:
                self._send_json(404, {"error": {"message": "model not found", "type": "invalid_request_error"}})
                return

        # Health inconsistent: chat returns 503
        if defect == "health_inconsistent":
            self._send_json(503, {"error": {"message": "service unavailable"}})
            return

        # Tool call request
        tools = body.get("tools")
        tool_choice = body.get("tool_choice")
        if tools and tool_choice:
            if self._is_minimal():
                self._send_json(400, {"error": {"message": "tools not supported"}})
                return
            resp_id = _counter.next_id()
            self._send_json(200, _make_chat_response(
                content=None, finish_reason="tool_calls",
                tool_calls=[_TOOL_CALL], usage=_usage(), resp_id=resp_id,
            ))
            return

        # Structured output
        response_format = body.get("response_format")
        if response_format and isinstance(response_format, dict):
            rf_type = response_format.get("type")
            if rf_type == "json_schema":
                if self._is_minimal():
                    self._send_json(400, {"error": {"message": "structured output not supported"}})
                    return
                resp_id = _counter.next_id()
                self._send_json(200, _make_chat_response(
                    content=_CAPITAL_JSON, finish_reason="stop",
                    usage=_usage(), resp_id=resp_id,
                ))
                return

        # Determine content based on the prompt
        messages = body.get("messages", [])
        user_msg = ""
        for m in messages:
            if m.get("role") == "user":
                user_msg = m.get("content", "")

        content = _SHORT_CONTENT
        if "paragraph" in user_msg.lower() or "distributed" in user_msg.lower():
            content = _LONG_CONTENT
        elif "count" in user_msg.lower() or "1 to 5" in user_msg.lower():
            content = _COUNT_CONTENT
        elif "hello world" in user_msg.lower() or "exactly" in user_msg.lower():
            content = "Hello World"

        # Apply defects for non-streaming first
        finish_reason = "stop"
        max_tokens = body.get("max_tokens") or body.get("max_completion_tokens")
        stop = body.get("stop")

        # Handle stop sequence
        if stop and isinstance(stop, list):
            if defect == "stop_ignored":
                # Defect: ignore stop sequence, return full content
                finish_reason = "length"
            else:
                for seq in stop:
                    if seq in content:
                        idx = content.index(seq)
                        content = content[:idx]
                        finish_reason = "stop"
                        break

        # Handle max_tokens
        if max_tokens is not None and isinstance(max_tokens, int):
            if defect == "max_tokens_ignored":
                # Defect: ignore max_tokens limit
                pass
            else:
                # Simulate respecting max_tokens by limiting output
                if max_tokens == 1:
                    words = content.split()
                    content = words[0] if words else ""
                    finish_reason = "length"

        # Template leak defect
        if defect == "template_leak":
            content = "<|im_start|>assistant\n" + content

        if defect == "concurrency_duplicate_ids":
            resp_id = "chatcmpl-dup-fixed"
        else:
            resp_id = _counter.next_id()

        if stream:
            self._handle_stream_chat(content, model, finish_reason,
                                      include_usage, defect, resp_id)
            return

        # Determinism defect: alternate content for requests with the same prompt
        # so that the second and third requests for a prompt differ.
        if defect == "determinism_mismatch":
            det_map = getattr(self.server, '_determinism_counts', {})
            key = user_msg
            count = det_map.get(key, 0)
            det_map[key] = count + 1
            self.server._determinism_counts = det_map
            # Vary on even counts (0-indexed), so count=1 is varied, count=2 is normal, etc.
            if count % 2 == 1:
                content = content + " (varied)"

        usage = _usage()
        if max_tokens == 1 and defect != "max_tokens_ignored":
            usage = _usage(prompt=_PROMPT_TOKENS, completion=1)

        if defect == "usage_invalid" and max_tokens is None:
            usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 999}

        self._send_json(200, _make_chat_response(
            content=content, finish_reason=finish_reason,
            model=model, usage=usage, resp_id=resp_id,
        ))

    def _handle_stream_chat(self, content, model, finish_reason,
                             include_usage, defect, resp_id):
        if defect == "malformed_stream":
            # Send invalid SSE framing
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                # Send a valid first chunk with role
                first = json.dumps({
                    "id": resp_id,
                    "object": "chat.completion.chunk",
                    "created": 1700000000,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
                })
                self.wfile.write(("data: %s\n\n" % first).encode("utf-8"))
                self.wfile.flush()
                # Send a chunk with content
                c = json.dumps({
                    "id": resp_id,
                    "object": "chat.completion.chunk",
                    "created": 1700000000,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}],
                })
                self.wfile.write(("data: %s\n\n" % c).encode("utf-8"))
                self.wfile.flush()
                # Then corrupt: missing [DONE] and incomplete
                self.wfile.write(b"data: {broken json\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            return

        if defect == "no_stream_usage" and include_usage:
            # Defect: omit usage even though stream_options.include_usage was requested
            include_usage = False

        if defect == "stream_parity_mismatch":
            content = content + " [stream-extra]"

        if defect == "cancel_kills_server":
            # Send chunks slowly so client can cancel mid-stream
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            # First chunk with role
            first = json.dumps({
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
            })
            try:
                self.wfile.write(("data: %s\n\n" % first).encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return
            # Content chunk
            c1 = json.dumps({
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
            })
            try:
                self.wfile.write(("data: %s\n\n" % c1).encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return
            # Simulate dying after cancel: stop the server from responding
            # by hanging here to simulate being "killed"
            # The follow-up request will also fail because server shuts down
            self.server._cancel_kill_triggered = True
            # Sleep long enough for the follow-up to timeout
            time.sleep(5)
            return

        chunks = _make_stream_chunks(content, model, finish_reason,
                                      include_usage, resp_id)

        # Determinism defect for stream: alter content
        if defect == "determinism_mismatch":
            count = getattr(self.server, '_stream_determinism_count', 0)
            self.server._stream_determinism_count = count + 1
            if count > 0:
                chunks = _make_stream_chunks(
                    content + " (stream-varied)", model, finish_reason,
                    include_usage, resp_id)

        self._send_sse(chunks)

    def _handle_completions(self, body):
        resp_id = _counter.next_id()
        self._send_json(200, {
            "id": resp_id,
            "object": "text_completion",
            "created": 1700000000,
            "model": body.get("model", _MODEL_ID),
            "choices": [{"index": 0, "text": _SHORT_CONTENT, "finish_reason": "stop"}],
            "usage": _usage(),
        })

    def _handle_concurrency(self, body):
        defect = self._get_defect()
        if defect == "concurrency_duplicate_ids":
            # Always return the same ID
            return "chatcmpl-dup-fixed"
        return _counter.next_id()


class MockServer(http.server.HTTPServer):
    """Extended server with configurable defect and mode."""

    allow_reuse_address = True

    def __init__(self, port, defect=None, minimal=False, require_api_key=None,
                 health_path="/health"):
        self.defect = defect
        self.minimal = minimal
        self.require_api_key = require_api_key
        self.health_path = health_path
        self._determinism_count = 0
        self._stream_determinism_count = 0
        self._cancel_kill_triggered = False
        super().__init__(("127.0.0.1", port), MockHandler)


def start_server(port=0, defect=None, minimal=False, require_api_key=None,
                 health_path="/health"):
    """Start the mock server and return (server, port, thread)."""
    server = MockServer(port, defect=defect, minimal=minimal,
                        require_api_key=require_api_key, health_path=health_path)
    actual_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, actual_port, thread


def main():
    parser = argparse.ArgumentParser(
        description="Deterministic mock OpenAI-compatible server.",
    )
    parser.add_argument("--port", type=int, default=0, help="Port (0 for random)")
    parser.add_argument("--defect", default=None, help="Injectable defect name")
    parser.add_argument("--minimal", action="store_true", help="Minimal mode")
    parser.add_argument("--require-api-key", default=None, help="Required API key")
    parser.add_argument("--health-path", default="/health", help="Health endpoint path")
    args = parser.parse_args()

    server, port, thread = start_server(
        port=args.port,
        defect=args.defect,
        minimal=args.minimal,
        require_api_key=args.require_api_key,
        health_path=args.health_path,
    )
    # Print port for the runner to capture
    print(port, flush=True)
    try:
        thread.join()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
