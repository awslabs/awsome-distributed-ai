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
TIMEOUT=30
CHECKS_PASSED=0
CHECKS_TOTAL=4

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
echo "═══════════════════════════════════════════════════════════"
echo ""

[[ "${CHECKS_PASSED}" -eq "${CHECKS_TOTAL}" ]] || exit 1