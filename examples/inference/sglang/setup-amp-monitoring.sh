#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# One-shot setup for the AMP (Amazon Managed Prometheus) monitoring path:
#   1. Create (or reuse) an AMP workspace.
#   2. Enable the cluster OIDC provider and create the AMP ingest IAM role,
#      bound to the amp-iamproxy-ingest-service-account ServiceAccount (eksctl).
#   3. Render prometheus-agent-amp.yaml with the real workspace id, role ARN and
#      region, then kubectl apply it.
#
# Idempotent — safe to re-run. Re-running reuses an existing workspace with the
# same alias and overrides the existing ServiceAccount / Deployment.
#
# Usage:
#   ./setup-amp-monitoring.sh <CLUSTER_NAME> [REGION] [AMP_ALIAS]
#
# Example:
#   ./setup-amp-monitoring.sh eks-hypd-0512-b2ad us-west-2 sglang-kimi
#
# Prereqs: awscli, eksctl, kubectl, envsubst on PATH; AWS creds with AMP + IAM
# permissions; kubectl context pointing at the target cluster.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <CLUSTER_NAME> [REGION] [AMP_ALIAS]" >&2
    exit 1
fi

CLUSTER_NAME="$1"
REGION="${2:-$(aws configure get region)}"
AMP_ALIAS="${3:-sglang-monitoring}"
SA_NAME="amp-iamproxy-ingest-service-account"
SA_NAMESPACE="default"

if [[ -z "${REGION}" ]]; then
    echo "ERROR: no region given and none in 'aws configure get region'." >&2
    exit 1
fi

echo "==> Cluster:   ${CLUSTER_NAME}"
echo "    Region:    ${REGION}"
echo "    AMP alias: ${AMP_ALIAS}"
echo

# ---------------------------------------------------------------------------
# 1. AMP workspace — reuse the first workspace matching the alias, else create.
# ---------------------------------------------------------------------------
echo "==> [1/3] AMP workspace"
WORKSPACE_ID="$(aws amp list-workspaces --region "${REGION}" --alias "${AMP_ALIAS}" \
    --query 'workspaces[0].workspaceId' --output text 2>/dev/null || true)"

if [[ -z "${WORKSPACE_ID}" || "${WORKSPACE_ID}" == "None" ]]; then
    echo "    creating workspace alias=${AMP_ALIAS}"
    WORKSPACE_ID="$(aws amp create-workspace --alias "${AMP_ALIAS}" --region "${REGION}" \
        --query 'workspaceId' --output text)"
    aws amp wait workspace-active --workspace-id "${WORKSPACE_ID}" --region "${REGION}"
else
    echo "    reusing existing workspace"
fi
echo "    workspaceId: ${WORKSPACE_ID}"

# ---------------------------------------------------------------------------
# 2. OIDC provider + ingest IAM role bound to the ServiceAccount (eksctl).
# ---------------------------------------------------------------------------
echo "==> [2/3] OIDC provider + ingest IAM role"
eksctl utils associate-iam-oidc-provider \
    --cluster "${CLUSTER_NAME}" --region "${REGION}" --approve

eksctl create iamserviceaccount \
    --name "${SA_NAME}" \
    --namespace "${SA_NAMESPACE}" \
    --cluster "${CLUSTER_NAME}" --region "${REGION}" \
    --attach-policy-arn arn:aws:iam::aws:policy/AmazonPrometheusRemoteWriteAccess \
    --approve --override-existing-serviceaccounts

ROLE_ARN="$(kubectl get sa "${SA_NAME}" -n "${SA_NAMESPACE}" \
    -o jsonpath='{.metadata.annotations.eks\.amazonaws\.com/role-arn}')"
if [[ -z "${ROLE_ARN}" ]]; then
    echo "ERROR: ServiceAccount ${SA_NAME} has no role-arn annotation after eksctl." >&2
    exit 1
fi
echo "    role-arn: ${ROLE_ARN}"

# ---------------------------------------------------------------------------
# 3. Render prometheus-agent-amp.yaml and apply.
#    The manifest's ServiceAccount block re-asserts the placeholder annotation,
#    so we substitute the real role ARN there too (matches what eksctl wrote).
# ---------------------------------------------------------------------------
echo "==> [3/3] Rendering and applying prometheus-agent-amp.yaml"
sed \
    -e "s|<YOUR_AMP_INGEST_ROLE_ARN>|${ROLE_ARN}|g" \
    -e "s|<YOUR_AMP_WORKSPACE_ID>|${WORKSPACE_ID}|g" \
    -e "s|<region>|${REGION}|g" \
    "${SCRIPT_DIR}/prometheus-agent-amp.yaml" \
    | kubectl apply -f -

# The substitutions land in the ConfigMap, not the Deployment's pod spec, so
# `apply` reports the Deployment "unchanged" and won't restart the pod — it
# would keep running the old (placeholder) config. Force a fresh pod so it
# remounts and reloads the updated ConfigMap. Idempotent.
kubectl rollout restart deployment/prometheus-agent
kubectl rollout status deployment/prometheus-agent --timeout=120s

echo
echo "==> Done. AMP remote-write target:"
echo "    https://aps-workspaces.${REGION}.amazonaws.com/workspaces/${WORKSPACE_ID}/api/v1/remote_write"
echo
echo "    Watch the agent come up (should leave CrashLoopBackOff):"
echo "      kubectl rollout status deployment/prometheus-agent"
echo "      kubectl logs -f -l app=prometheus-agent"
echo
echo "    Next: deploy DCGM exporter and wire up Amazon Managed Grafana"
echo "    (see README.md -> 'GPU metrics' and 'Amazon Managed Grafana')."
