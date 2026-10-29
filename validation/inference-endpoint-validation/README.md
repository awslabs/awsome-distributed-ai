# Inference Endpoint Validation

## Overview

A single-file, standard-library Python validator that checks an OpenAI-compatible inference endpoint against 21 contract checks and emits a JSON report plus a Markdown `## Test Results` block suitable for an ADT pull request.

Supported engines: vLLM, SGLang, NVIDIA Dynamo frontend, TensorRT-LLM `trtllm-serve`, NVIDIA NIM, and routers in front of them.

The tool validates the serving API that a user actually calls. It does not perform load testing, throughput sweeps, roofline analysis, model accuracy evaluation, GPU or EFA health checks, or engine deployment and tuning. Those are covered by other tools (AIPerf, `vllm bench`, `gpu-cluster-healthcheck`, lm-evaluation-harness, and others).

## Prerequisites

| Requirement | Detail |
|-------------|--------|
| Python | 3.10 or later |
| Dependencies | None. Standard library only; no `pip install` needed. |
| Endpoint | A running OpenAI-compatible inference endpoint reachable over HTTP or HTTPS |

## Quick Start

```bash
# Clone the repository
git clone https://github.com/awslabs/awsome-distributed-ai.git
cd awsome-distributed-ai/validation/inference-endpoint-validation

# Run against a local endpoint
python3 validate_endpoint.py --base-url http://localhost:8000

# Run and save reports
python3 validate_endpoint.py --base-url http://localhost:8000 \
    --json report.json --markdown report.md

# Run with an API key (reads from OPENAI_API_KEY by default)
export OPENAI_API_KEY="your-key"
python3 validate_endpoint.py --base-url http://localhost:8000

# Run against NIM with its health path
python3 validate_endpoint.py --base-url http://localhost:8000 \
    --health-path /v1/health/ready
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--base-url` | (required) | Base URL of the endpoint (e.g. `http://localhost:8000`) |
| `--model` | auto-detected | Model name. Auto-detected from `/v1/models` if omitted. |
| `--api-key-env` | `OPENAI_API_KEY` | Environment variable holding the API key |
| `--health-path` | `/health` | Health endpoint path |
| `--checks` | all | Comma-separated list of check IDs to run |
| `--skip` | none | Comma-separated list of check IDs to skip |
| `--require` | none | Comma-separated list of advisory check IDs to promote to required |
| `--strict` | off | Exit 1 on any warning |
| `--timeout` | `60` | Request timeout in seconds |
| `--concurrency` | `8` | Number of parallel requests for `concurrency.parallel` |
| `--json` | none | Write JSON report to this file |
| `--markdown` | none | Write Markdown report to this file |

## Check Catalog

All 21 checks run in the order shown. Severity determines how a failure affects the exit code.

| ID | Severity | What It Verifies |
|----|----------|------------------|
| `models.list` | required | `GET /v1/models` returns 200 with a `data` array |
| `server_version` | required | `GET /version` returns version info; 404 means skip |
| `health.ready` | required | `GET <health-path>` returns 200; 404/405 means skip |
| `health.consistent` | required | Health 200 and chat 200 agree; catches health-up but chat-down |
| `chat.basic` | required | Non-streaming chat completion returns valid structure, choices, usage |
| `chat.determinism` | advisory | Two identical requests at temperature=0 produce the same output |
| `chat.stream` | required | Streaming chat returns valid SSE framing with `[DONE]` sentinel |
| `chat.stream_usage` | advisory | Streaming with `include_usage` includes usage in the final chunk |
| `chat.stream_parity` | advisory | Assembled stream text matches non-streaming response |
| `chat.max_tokens` | required | `max_tokens: 1` is respected (falls back to `max_completion_tokens`) |
| `chat.stop` | required | Stop sequence halts generation and `finish_reason` is `stop` |
| `chat.template_leak` | required | No chat-template markers appear in output |
| `completions.basic` | required | `POST /v1/completions` works; 404/405 means skip |
| `tools.call` | required | Tool calling with named `tool_choice` returns `tool_calls` |
| `structured_output` | required | `response_format` with `json_schema` returns valid JSON matching the schema |
| `errors.unknown_model` | required | Unknown model name returns 4xx |
| `errors.malformed_json` | required | Truncated JSON body returns 4xx |
| `auth.required` | required | If an API key is set, unauthenticated requests get 401 or 403 |
| `stream.cancel` | required | Endpoint stays healthy after a client disconnects mid-stream |
| `concurrency.parallel` | required | N parallel requests all succeed with unique response IDs |
| `latency.smoke` | required | Measures TTFT and total latency from one stream; always passes |

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | All checks passed (warnings allowed unless `--strict`) |
| `1` | At least one required check failed, or `--strict` and any warning |
| `2` | Usage error (bad flag, unknown check ID) or endpoint unreachable |

## Reports

### JSON

Written with `--json <path>`. Top-level fields:

```json
{
  "tool_name": "inference-endpoint-validation",
  "tool_version": "1.0.0",
  "schema_version": 1,
  "started_at": "2026-01-15T12:00:00Z",
  "base_url": "http://localhost:8000",
  "model": "Qwen/Qwen2.5-3B-Instruct",
  "checks": [
    {
      "id": "models.list",
      "severity": "required",
      "status": "pass",
      "detail": "found 1 model(s)",
      "duration_ms": 12
    }
  ],
  "summary": {
    "pass": 18,
    "fail": 0,
    "warn": 2,
    "skip": 1,
    "total": 21,
    "duration_ms": 4500
  },
  "exit_code": 0
}
```

API keys are redacted from all report fields.

### Markdown

Written with `--markdown <path>`. Produces a `## Test Results` table suitable for pasting into an ADT pull request.

## Engine Notes

### NVIDIA NIM

NIM exposes its health endpoint at `/v1/health/ready` instead of the default `/health`. Pass `--health-path /v1/health/ready` to enable the `health.ready` and `health.consistent` checks.

### Tool calling

Tool calling support varies by engine and model. vLLM requires `--enable-auto-tool-choice` and a model-specific `--tool-call-parser` (for example `--tool-call-parser hermes`). If tool calling is not enabled, `tools.call` returns skip (status 400/404/422/501).

The `--enable-auto-tool-choice` flag only applies to engines that use `auto` as the tool choice strategy. For named `tool_choice` (which this check uses), most engines work without extra flags if the model supports tools.

### Structured output

Structured output via `response_format` with `json_schema` is supported by vLLM (with `--enable-auto-tool-choice`), SGLang, and NIM. Engines that do not support it return 400/422/501 and the check is skipped.

### Reasoning parsers

Some engines expose reasoning content via `reasoning_content` or `reasoning` fields in the message or delta. The validator extracts both and treats either as valid content for checks like `chat.basic` and `chat.stream`.

## Kubernetes Job Usage

The file `kubernetes/validate-endpoint-job.yaml` provides a Kubernetes Job that runs the validator inside a cluster. It uses:

- A ConfigMap to mount `validate_endpoint.py` into the container
- A Secret to provide the API key
- The `python:3.12.16-slim` image (pinned)

### Setup

Create the ConfigMap from the script:

```bash
kubectl create configmap inference-endpoint-validation \
    --from-file=validate_endpoint.py=validate_endpoint.py
```

Create the Secret with your API key (skip if the endpoint does not require authentication):

```bash
kubectl create secret generic inference-endpoint-api-key \
    --from-literal=api-key='your-api-key-here'
```

Edit the Job to set the endpoint URL and any other flags, then apply:

```bash
kubectl apply -f kubernetes/validate-endpoint-job.yaml
```

Check results:

```bash
kubectl logs job/inference-endpoint-validation
```

## Running the Tests

The offline test suite uses a mock OpenAI-compatible server and runs without a live endpoint:

```bash
cd validation/inference-endpoint-validation

# Run the acceptance suite
python3 tests/acceptance.py --require-all

# Or use unittest discover
python3 -m unittest discover -s tests -v
```

## Known Issues

- Advisory checks (`chat.determinism`, `chat.stream_usage`, `chat.stream_parity`) may produce different results across engines. Some engines do not guarantee deterministic output at temperature=0 with a seed, and `stream_options` support varies.
- `completions.basic` is skipped on engines that only expose the chat completions API.
- `tools.call` and `structured_output` require engine-side configuration on some engines. A skip result does not indicate a bug.
- `auth.required` is skipped when no API key is configured via the `--api-key-env` environment variable.
- `latency.smoke` always passes. Its TTFT and total values are informational and should not be used for benchmarking. Use AIPerf or `vllm bench` for performance measurement.
