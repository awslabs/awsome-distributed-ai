#!/usr/bin/env bash
# ================================================================
# Inference Gateway — End-to-End Validation Script
# ================================================================
# Tests the full gateway routing chain:
#   ALB → Body-Based Router (BBR) → llm-d Endpoint Picker → vLLM
#
# The script discovers the gateway's ALB from the InferenceGatewayConfig
# (IGWC) object, NOT from the per-model Ingresses that the operator
# creates for each InferenceEndpointConfig. The per-model ALBs bypass
# the gateway entirely — requests to them never touch the BBR or EPP.
#
# Steps 1-5: non-streaming validation (model routing, unknown-model 404)
# Steps 6-9: streaming validation (SSE framing, model match, [DONE],
#             arrival-time spread to detect proxy buffering)
#
# Prerequisites:
#   - kubectl configured for the HyperPod EKS cluster
#   - Both InferenceEndpointConfig resources deployed and Ready
#   - Must run from inside the VPC (both ALBs are internal)
#     e.g., kubectl apply -f k8s/03-test-pod.yaml
#
# Usage:
#   IGWC_NAME=qwen-gateway NAMESPACE=default ./scripts/0.test-gateway.sh
# ================================================================
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────
NAMESPACE="${NAMESPACE:-default}"
IGWC_NAME="${IGWC_NAME:-qwen-gateway}"
MODEL_7B="Qwen/Qwen2.5-7B-Instruct"
MODEL_3B="Qwen/Qwen2.5-3B-Instruct"
UNKNOWN_MODEL="does-not-exist/fake-model"
MAX_TOKENS=32
STREAM_MAX_TOKENS=128       # Enough tokens for a clear time spread
TIMEOUT=30
STREAM_TIMEOUT=60           # Streaming responses take longer
CHECKS_PASSED=0
CHECKS_TOTAL=12             # 4 non-streaming + 8 streaming

# Minimum time spread (seconds) between first and last content chunks.
# A well-behaved streaming proxy delivers chunks as they are generated;
# a buffering proxy holds them back and releases them in a burst.
# 0.5 s is conservative — real runs on HyperPod with 128 tokens typically
# show 2-4 s of spread.  Adjust after the first real run if needed.
STREAM_MIN_SPREAD_S="0.5"

# ── Helpers ────────────────────────────────────────────────────
pass() { echo "  ✅ PASS: $1"; CHECKS_PASSED=$((CHECKS_PASSED + 1)); }
fail() { echo "  ❌ FAIL: $1"; }
info() { echo "  ℹ️  $1"; }

# ── Step 1: Discover gateway endpoint ──────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Inference Gateway — End-to-End Validation"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Step 1: Discovering gateway endpoint..."

GATEWAY_ENDPOINT=$(kubectl get ingress "${IGWC_NAME}-alb" \
  -n hyperpod-inference-system \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' \
  2>/dev/null || true)

if [[ -z "${GATEWAY_ENDPOINT}" ]]; then
  echo "  ❌ FAIL: Ingress hyperpod-inference-system/${IGWC_NAME}-alb has no hostname."
  echo "  Check: kubectl get ingress ${IGWC_NAME}-alb -n hyperpod-inference-system"
  exit 1
fi
echo "  Gateway endpoint: ${GATEWAY_ENDPOINT}"

# Detect protocol from IGWC TLS status
if [[ -n "$(kubectl get igwc "${IGWC_NAME}" -n "${NAMESPACE}" \
  -o jsonpath='{.status.tls.acmArn}' 2>/dev/null)" ]]; then
  PROTO="https"
else
  PROTO="http"
fi
ENDPOINT_PATH="/v1/chat/completions"
CURL_OPTS="-sk"

echo "  Protocol: ${PROTO}"
echo "  URL: ${PROTO}://${GATEWAY_ENDPOINT}${ENDPOINT_PATH}"
echo ""

# ── Step 2: Verify InferenceGatewayConfig ──────────────────────
echo "Step 2: Checking InferenceGatewayConfig..."

IGWC_STATUS=$(kubectl get igwc "${IGWC_NAME}" -n "${NAMESPACE}" \
  -o jsonpath='{.status}' 2>/dev/null || true)

if [[ -z "${IGWC_STATUS}" ]]; then
  echo "  ❌ FAIL: InferenceGatewayConfig '${IGWC_NAME}' not found in namespace '${NAMESPACE}'."
  exit 1
fi

# Check that both schedulers are present and Available
SCHEDULER_COUNT=$(kubectl get igwc "${IGWC_NAME}" -n "${NAMESPACE}" \
  -o jsonpath='{.status.schedulers}' 2>/dev/null \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d))" 2>/dev/null || echo "0")

AVAILABLE_COUNT=$(kubectl get igwc "${IGWC_NAME}" -n "${NAMESPACE}" \
  -o jsonpath='{.status.schedulers}' 2>/dev/null \
  | python3 -c "
import sys,json
scheds = json.load(sys.stdin)
print(sum(1 for s in scheds if s.get('rolloutState') == 'Available'))
" 2>/dev/null || echo "0")

if [[ "${SCHEDULER_COUNT}" -ge 2 && "${AVAILABLE_COUNT}" -ge 2 ]]; then
  pass "IGWC '${IGWC_NAME}' has ${SCHEDULER_COUNT} schedulers (${AVAILABLE_COUNT} Available)"
else
  fail "IGWC '${IGWC_NAME}' has ${SCHEDULER_COUNT} schedulers (${AVAILABLE_COUNT} Available), expected ≥2 Available"
fi
echo ""

# ── Step 3: Test 7B model ──────────────────────────────────────
echo "Step 3: Testing ${MODEL_7B}..."

RESPONSE_7B=$(curl ${CURL_OPTS} --max-time "${TIMEOUT}" \
  "${PROTO}://${GATEWAY_ENDPOINT}${ENDPOINT_PATH}" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL_7B}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in one sentence.\"}],
    \"max_tokens\": ${MAX_TOKENS}
  }" 2>/dev/null || echo "{}")

RETURNED_MODEL_7B=$(echo "${RESPONSE_7B}" | jq -r '.model // empty' 2>/dev/null || true)
ID_7B=$(echo "${RESPONSE_7B}" | jq -r '.id // empty' 2>/dev/null || true)

if [[ "${RETURNED_MODEL_7B}" == "${MODEL_7B}" ]]; then
  pass "${MODEL_7B} — model matched, id: ${ID_7B}"
else
  fail "${MODEL_7B} — expected model '${MODEL_7B}', got '${RETURNED_MODEL_7B}'"
  info "Response: ${RESPONSE_7B}"
fi
echo ""

# ── Step 4: Test 3B model ──────────────────────────────────────
echo "Step 4: Testing ${MODEL_3B}..."

RESPONSE_3B=$(curl ${CURL_OPTS} --max-time "${TIMEOUT}" \
  "${PROTO}://${GATEWAY_ENDPOINT}${ENDPOINT_PATH}" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL_3B}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in one sentence.\"}],
    \"max_tokens\": ${MAX_TOKENS}
  }" 2>/dev/null || echo "{}")

RETURNED_MODEL_3B=$(echo "${RESPONSE_3B}" | jq -r '.model // empty' 2>/dev/null || true)
ID_3B=$(echo "${RESPONSE_3B}" | jq -r '.id // empty' 2>/dev/null || true)

if [[ "${RETURNED_MODEL_3B}" == "${MODEL_3B}" ]]; then
  pass "${MODEL_3B} — model matched, id: ${ID_3B}"
else
  fail "${MODEL_3B} — expected model '${MODEL_3B}', got '${RETURNED_MODEL_3B}'"
  info "Response: ${RESPONSE_3B}"
fi
echo ""

# ── Step 5: Test unknown model (expects 404) ───────────────────
echo "Step 5: Testing unknown model rejection..."

HTTP_CODE=$(curl ${CURL_OPTS} --max-time "${TIMEOUT}" \
  -o /dev/null -w '%{http_code}' \
  "${PROTO}://${GATEWAY_ENDPOINT}${ENDPOINT_PATH}" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${UNKNOWN_MODEL}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}],
    \"max_tokens\": ${MAX_TOKENS}
  }" 2>/dev/null || true)

if [[ "${HTTP_CODE}" == "404" ]]; then
  pass "Unknown model '${UNKNOWN_MODEL}' rejected with HTTP 404"
else
  fail "Unknown model '${UNKNOWN_MODEL}' returned HTTP ${HTTP_CODE}, expected 404"
fi
echo ""

# ════════════════════════════════════════════════════════════════
#  Streaming validation (Steps 6-9)
# ════════════════════════════════════════════════════════════════
# A proxy on the gateway path (ALB, Envoy, reverse-proxy sidecar)
# can break streaming without affecting non-streaming requests.
# These steps verify SSE framing, model routing, the [DONE]
# sentinel, and that chunks arrive incrementally — not in a burst
# that would indicate proxy buffering.
# ════════════════════════════════════════════════════════════════

# ── High-resolution timestamp ──────────────────────────────────
# BusyBox date (Alpine) does not support %N.  Detect once and pick
# the best available clock: GNU date (nanoseconds), perl
# (Time::HiRes, microseconds), python3, or fallback to seconds.
if date +%s.%N 2>/dev/null | grep -q '\..*[0-9]'; then
  _ts() { date +%s.%N; }
elif command -v perl >/dev/null 2>&1 && \
     perl -MTime::HiRes=time -e 'printf "%.6f\n",time' >/dev/null 2>&1; then
  _ts() { perl -MTime::HiRes=time -e 'printf "%.6f\n",time'; }
elif command -v python3 >/dev/null 2>&1; then
  _ts() { python3 -c 'import time; print(f"{time.time():.6f}")'; }
else
  _ts() { date +%s; }
fi

# ── stream_test() ──────────────────────────────────────────────
# Sends a streaming chat request and validates the response.
#
# Arguments:
#   $1  step number (for display)
#   $2  model name
#
# Checks (each counted separately):
#   1. HTTP 200
#   2. model field in chunks matches the requested model
#   3. Stream ends with data: [DONE]
#   4. First content chunk arrives well before the last
# ────────────────────────────────────────────────────────────────
stream_test() {
  local STEP="$1"
  local MODEL="$2"
  local TMPDIR
  TMPDIR=$(mktemp -d)
  local RAW="${TMPDIR}/raw"          # timestamped SSE lines
  local HDR="${TMPDIR}/headers"      # response headers

  echo "Step ${STEP}: Streaming test for ${MODEL}..."

  # Stream the response.  --no-buffer ensures curl writes each chunk
  # as it arrives.  We prefix every line with a high-resolution
  # timestamp so we can measure arrival-time spread later.
  #
  # The -D flag captures the HTTP status line and headers separately.
  # We use --max-time to bound the overall request.
  curl ${CURL_OPTS} --no-buffer \
    --max-time "${STREAM_TIMEOUT}" \
    -D "${HDR}" \
    "${PROTO}://${GATEWAY_ENDPOINT}${ENDPOINT_PATH}" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${MODEL}\",
      \"messages\": [{\"role\": \"user\", \"content\": \"Write a short paragraph about cloud computing.\"}],
      \"max_tokens\": ${STREAM_MAX_TOKENS},
      \"stream\": true
    }" 2>/dev/null \
    | while IFS= read -r line; do
        echo "$(_ts) ${line}"
      done > "${RAW}" || true

  # ── Check 1: HTTP 200 ───────────────────────────────────────
  local HTTP_STATUS
  HTTP_STATUS=$(head -1 "${HDR}" 2>/dev/null \
    | sed -n 's/.*\([0-9][0-9][0-9]\).*/\1/p' || true)

  if [[ "${HTTP_STATUS}" == "200" ]]; then
    pass "${MODEL} streaming — HTTP ${HTTP_STATUS}"
  else
    fail "${MODEL} streaming — expected HTTP 200, got '${HTTP_STATUS}'"
    info "Headers: $(cat "${HDR}" 2>/dev/null)"
    rm -rf "${TMPDIR}"
    return
  fi

  # ── Parse SSE data lines ────────────────────────────────────
  # Each timestamped line looks like:
  #   1719500000.123456 data: {"id":"...","model":"...","choices":[...]}
  # or:
  #   1719500000.234567 data: [DONE]
  #
  # We extract only "data: " lines (ignoring blank keep-alive lines
  # and any "event:" lines).

  # ── Check 2: model field matches ────────────────────────────
  # Extract the model from the first data chunk that is valid JSON.
  local CHUNK_MODEL
  CHUNK_MODEL=$(grep ' data: {' "${RAW}" \
    | head -1 \
    | sed 's/^[0-9.]* data: //' \
    | jq -r '.model // empty' 2>/dev/null || true)

  if [[ "${CHUNK_MODEL}" == "${MODEL}" ]]; then
    pass "${MODEL} streaming — model field matches"
  else
    fail "${MODEL} streaming — expected model '${MODEL}', got '${CHUNK_MODEL}'"
  fi

  # ── Check 3: stream ends with data: [DONE] ─────────────────
  local LAST_DATA
  LAST_DATA=$(grep ' data: ' "${RAW}" | tail -1 | sed 's/^[0-9.]* //' || true)

  if [[ "${LAST_DATA}" == "data: [DONE]" ]]; then
    pass "${MODEL} streaming — ends with data: [DONE]"
  else
    fail "${MODEL} streaming — last data line: '${LAST_DATA}', expected 'data: [DONE]'"
  fi

  # ── Check 4: arrival-time spread ────────────────────────────
  # Extract timestamps of lines containing content deltas.
  # A content delta has choices[0].delta.content set (non-null).
  local CONTENT_TIMES
  CONTENT_TIMES=$(grep ' data: {' "${RAW}" \
    | while IFS= read -r ts_line; do
        local ts="${ts_line%% *}"
        local json="${ts_line#* data: }"
        local has_content
        has_content=$(echo "${json}" \
          | jq -r '.choices[0].delta.content // empty' 2>/dev/null || true)
        if [[ -n "${has_content}" ]]; then
          echo "${ts}"
        fi
      done || true)

  local FIRST_TS LAST_TS
  FIRST_TS=$(echo "${CONTENT_TIMES}" | head -1 || true)
  LAST_TS=$(echo "${CONTENT_TIMES}" | tail -1 || true)

  if [[ -n "${FIRST_TS}" && -n "${LAST_TS}" ]]; then
    # Compute spread using awk for floating-point subtraction
    local SPREAD
    SPREAD=$(awk "BEGIN { printf \"%.3f\", ${LAST_TS} - ${FIRST_TS} }")

    if awk "BEGIN { exit !(${SPREAD} >= ${STREAM_MIN_SPREAD_S}) }"; then
      pass "${MODEL} streaming — time spread ${SPREAD} s ≥ ${STREAM_MIN_SPREAD_S} s (first→last content chunk)"
    else
      fail "${MODEL} streaming — time spread ${SPREAD} s < ${STREAM_MIN_SPREAD_S} s (chunks may be buffered)"
    fi
    info "Content chunks: $(echo "${CONTENT_TIMES}" | wc -l | tr -d ' '), first: ${FIRST_TS}, last: ${LAST_TS}, spread: ${SPREAD} s"
  else
    fail "${MODEL} streaming — no content chunks received"
  fi

  rm -rf "${TMPDIR}"
  echo ""
}

# ── Step 6-7: Stream 7B model ─────────────────────────────────
stream_test 6 "${MODEL_7B}"

# ── Step 8-9: Stream 3B model ─────────────────────────────────
stream_test 8 "${MODEL_3B}"

# ── Summary ────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════"
if [[ "${CHECKS_PASSED}" -eq "${CHECKS_TOTAL}" ]]; then
  echo "  ✅ All ${CHECKS_TOTAL} checks passed!"
else
  echo "  ⚠️  ${CHECKS_PASSED}/${CHECKS_TOTAL} checks passed"
fi
echo ""
echo "  Gateway : ${PROTO}://${GATEWAY_ENDPOINT}"
echo "  IGWC    : ${IGWC_NAME}"
echo "  Path    : ${ENDPOINT_PATH}"
echo "  Tests   : ${CHECKS_PASSED}/${CHECKS_TOTAL}"
echo "            non-streaming: Steps 2-5 (4 checks)"
echo "            streaming:     Steps 6-9 (8 checks)"
echo "═══════════════════════════════════════════════════════════"
echo ""

[[ "${CHECKS_PASSED}" -eq "${CHECKS_TOTAL}" ]] || exit 1
