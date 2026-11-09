#!/usr/bin/env bash
#
# Render the NVCF control-plane environment and secrets files, and create the
# registry pull secrets, for the multi-cluster EKS topology.
#
# The NGC API key is read from AWS Secrets Manager and written only into
# git-ignored files under the release bundle. It is never echoed.
#
# Usage:
#   BUNDLE_DIR=/path/to/nvcf-1.0.4 ./render-config.sh
#
# Required env (with defaults):
#   BUNDLE_DIR            extracted NVCF release bundle root
#   HELMFILE_ENV          environment name                  (default: eks-multi)
#   CONTROL_PLANE_CONTEXT kubectl context, control plane     (default: nvcf-cp)
#   NGC_SECRET_ID         Secrets Manager secret id          (default: nvcf/ngc-api-key)
#   NGC_SECRET_REGION     region holding that secret         (default: us-west-2)
#   GATEWAY_NAMESPACE     namespace of nvcf-gateway          (default: envoy-gateway)
#   DOMAIN                private DNS zone (see setup-dns.sh) (default: nvcf.internal)
#   COMPUTE_CONTEXT       kubectl context, GPU cluster       (default: nvcf-compute)
set -euo pipefail

HELMFILE_ENV="${HELMFILE_ENV:-eks-multi}"
CONTROL_PLANE_CONTEXT="${CONTROL_PLANE_CONTEXT:-nvcf-cp}"
COMPUTE_CONTEXT="${COMPUTE_CONTEXT:-nvcf-compute}"
NGC_SECRET_ID="${NGC_SECRET_ID:-nvcf/ngc-api-key}"
NGC_SECRET_REGION="${NGC_SECRET_REGION:-us-west-2}"
GATEWAY_NAMESPACE="${GATEWAY_NAMESPACE:-envoy-gateway}"
DOMAIN="${DOMAIN:-nvcf.internal}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/../infra/controlplane-env.yaml.template"
COMPUTE_TEMPLATE="${SCRIPT_DIR}/../infra/compute-env.yaml.template"

: "${BUNDLE_DIR:?set BUNDLE_DIR to the extracted NVCF release bundle root}"
STACK="${BUNDLE_DIR}/deploy/stacks/self-managed"
COMPUTE_STACK="${BUNDLE_DIR}/deploy/stacks/nvcf-compute-plane"
[ -d "$STACK" ] || { echo "not a bundle: ${STACK} missing" >&2; exit 1; }
[ -f "$TEMPLATE" ] || { echo "missing template: ${TEMPLATE}" >&2; exit 1; }
[ -f "$COMPUTE_TEMPLATE" ] || { echo "missing template: ${COMPUTE_TEMPLATE}" >&2; exit 1; }

# --- Gateway address -------------------------------------------------------
# Produced by Step 1 (infra/gateway.yaml). Every hostname and Host header the
# control plane advertises is derived from it, so it must exist first.
GATEWAY_ADDR="$(kubectl --context "$CONTROL_PLANE_CONTEXT" get gateway nvcf-gateway \
  -n "$GATEWAY_NAMESPACE" -o jsonpath='{.status.addresses[0].value}' 2>/dev/null || true)"
[ -n "$GATEWAY_ADDR" ] || { echo "nvcf-gateway has no address yet; wait for the NLB" >&2; exit 1; }
echo "GATEWAY_ADDR=${GATEWAY_ADDR}"
echo "DOMAIN=${DOMAIN}"

# --- NGC key ---------------------------------------------------------------
# The secret stores a JSON object: {"NGC_API_KEY": "..."}.
NGC_API_KEY="$(aws secretsmanager get-secret-value \
  --secret-id "$NGC_SECRET_ID" --region "$NGC_SECRET_REGION" \
  --query SecretString --output text | jq -r '.NGC_API_KEY')"
[ -n "$NGC_API_KEY" ] && [ "$NGC_API_KEY" != "null" ] \
  || { echo "could not read NGC_API_KEY from ${NGC_SECRET_ID}" >&2; exit 1; }
echo "NGC key loaded (${#NGC_API_KEY} chars)"

# --- environments/<env>.yaml ----------------------------------------------
# python3 -I: isolated mode, so nothing is imported from the bundle directory.
ENV_FILE="${STACK}/environments/${HELMFILE_ENV}.yaml"
DOMAIN="$DOMAIN" NGC_API_KEY="$NGC_API_KEY" \
python3 -I - "$TEMPLATE" "$ENV_FILE" <<'PY'
import os, sys
src, dst = sys.argv[1], sys.argv[2]
with open(src, encoding="utf-8") as fh:
    text = fh.read()
text = text.replace("__DOMAIN__", os.environ["DOMAIN"])
text = text.replace("__NGC_API_KEY__", os.environ["NGC_API_KEY"])
for leftover in ("__DOMAIN__", "__NGC_API_KEY__"):
    assert leftover not in text, f"unrendered placeholder {leftover}"
with open(dst, "w", encoding="utf-8") as fh:
    fh.write(text)
print(f"wrote {dst}")
PY
chmod 600 "$ENV_FILE"

# --- secrets/<env>-secrets.yaml -------------------------------------------
# Each REPLACE_WITH_BASE64_DOCKER_CREDENTIAL is a RAW docker auth string:
# base64("$oauthtoken:<key>") — NOT a base64 dockerconfigjson. The wrong
# encoding makes `function create` fail with
# 400 "must be base64 encoded username:password format".
SECRETS_FILE="${STACK}/secrets/${HELMFILE_ENV}-secrets.yaml"
cp "${STACK}/secrets/secrets.yaml.template" "$SECRETS_FILE"
DOCKER_CRED_B64="$(printf '$oauthtoken:%s' "$NGC_API_KEY" | openssl base64 -A)"
DOCKER_CRED_B64="$DOCKER_CRED_B64" python3 -I - "$SECRETS_FILE" <<'PY'
import os, sys
path = sys.argv[1]
with open(path, encoding="utf-8") as fh:
    text = fh.read()
n = text.count("REPLACE_WITH_BASE64_DOCKER_CREDENTIAL")
text = text.replace("REPLACE_WITH_BASE64_DOCKER_CREDENTIAL", os.environ["DOCKER_CRED_B64"])
with open(path, "w", encoding="utf-8") as fh:
    fh.write(text)
print(f"substituted {n} docker credential placeholder(s) in {path}")
PY
chmod 600 "$SECRETS_FILE"

# --- registry pull secrets ------------------------------------------------
# The control-plane charts reference `nvcr-pull-secret` by name in every
# namespace they deploy into, so it must exist in all of them.
for ns in cassandra-system nats-system nvcf api-keys ess sis vault-system cert-manager; do
  kubectl --context "$CONTROL_PLANE_CONTEXT" create namespace "$ns" \
    --dry-run=client -o yaml | kubectl --context "$CONTROL_PLANE_CONTEXT" apply -f - >/dev/null
  kubectl --context "$CONTROL_PLANE_CONTEXT" create secret docker-registry nvcr-pull-secret \
    --docker-server=nvcr.io --docker-username='$oauthtoken' --docker-password="$NGC_API_KEY" \
    -n "$ns" --dry-run=client -o yaml | kubectl --context "$CONTROL_PLANE_CONTEXT" apply -f - >/dev/null
  echo "  nvcr-pull-secret ready in ${ns}"
done

# --- compute-plane environment + pull secret ------------------------------
COMPUTE_ENV_FILE="${COMPUTE_STACK}/environments/${HELMFILE_ENV}.yaml"
DOMAIN="$DOMAIN" NGC_API_KEY="$NGC_API_KEY" \
python3 -I - "$COMPUTE_TEMPLATE" "$COMPUTE_ENV_FILE" <<'PY'
import os, sys
src, dst = sys.argv[1], sys.argv[2]
with open(src, encoding="utf-8") as fh:
    text = fh.read()
text = text.replace("__DOMAIN__", os.environ["DOMAIN"])
text = text.replace("__NGC_API_KEY__", os.environ["NGC_API_KEY"])
for leftover in ("__DOMAIN__", "__NGC_API_KEY__"):
    assert leftover not in text, f"unrendered placeholder {leftover}"
with open(dst, "w", encoding="utf-8") as fh:
    fh.write(text)
print(f"wrote {dst}")
PY
chmod 600 "$COMPUTE_ENV_FILE"

# The operator propagates this secret to the namespaces it manages, so it only
# needs to exist in nvca-operator on the compute cluster.
kubectl --context "$COMPUTE_CONTEXT" create namespace nvca-operator \
  --dry-run=client -o yaml | kubectl --context "$COMPUTE_CONTEXT" apply -f - >/dev/null
kubectl --context "$COMPUTE_CONTEXT" create secret docker-registry nvcr-pull-secret \
  --docker-server=nvcr.io --docker-username='$oauthtoken' --docker-password="$NGC_API_KEY" \
  -n nvca-operator --dry-run=client -o yaml \
  | kubectl --context "$COMPUTE_CONTEXT" apply -f - >/dev/null
echo "  nvcr-pull-secret ready in nvca-operator (compute cluster)"

# A compute-scoped kubeconfig for `make register-cluster`.
kubectl --context "$COMPUTE_CONTEXT" config view --raw --minify --flatten \
  > "${BUNDLE_DIR}/compute-kubeconfig.yaml"
chmod 600 "${BUNDLE_DIR}/compute-kubeconfig.yaml"
echo "  wrote ${BUNDLE_DIR}/compute-kubeconfig.yaml"

# --- nvcf-cli config ------------------------------------------------------
# base_* point at 127.0.0.1 because the Gateway NLB is internal and this host is
# outside the VPC; run `kubectl port-forward -n envoy-gateway-system
# svc/envoy-envoy-gateway-nvcf-gateway-<hash> 8080:80 10081:10081` first.
# The *_host values stay as the real Gateway hostnames: they are sent as HTTP
# Host headers so the Gateway's HTTPRoutes still match through the tunnel.
CLI_CONFIG="${BUNDLE_DIR}/nvcf-cli.yaml"
cat > "$CLI_CONFIG" <<EOF
api_keys_service_id: "nvidia-cloud-functions-ncp-service-id-aketm"
api_keys_issuer_service: "nvcf-api"
api_keys_owner_id: "svc@nvcf-api.local"
client_id: "nvcf-default"

base_http_url: "http://127.0.0.1:8080"
invoke_url: "http://127.0.0.1:8080"
base_grpc_url: "127.0.0.1:10081"
api_keys_service_url: "http://127.0.0.1:8080"
icms_url: "http://127.0.0.1:8080"
api_host: "api.${DOMAIN}"
api_keys_host: "api-keys.${DOMAIN}"
invoke_host: "invocation.${DOMAIN}"
icms_host: "sis.${DOMAIN}"
EOF
chmod 600 "$CLI_CONFIG"
echo "  wrote ${CLI_CONFIG}"

echo "done. next: make -C ${STACK} install HELMFILE_ENV=${HELMFILE_ENV}"
