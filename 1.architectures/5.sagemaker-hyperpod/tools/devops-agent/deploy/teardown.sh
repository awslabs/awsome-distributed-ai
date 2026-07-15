#!/usr/bin/env bash
# Tear down the unified stack.
#
# The two custom resources clean themselves up on stack delete (they run
# before the AgentSpace via DependsOn, in reverse order):
#   - WebhookProvisioner disassociates + deregisters the eventChannel so the
#     AgentSpace has no leftover associations blocking its deletion.
#   - SkillUploader deletes the uploaded skill assets.
# After the stack is gone, the S3 assets bucket (owned by the Makefile, not the
# stack) is emptied and removed.
#
# Env overrides: REGION, STACK_NAME, PARAMS_FILE, KEEP_ASSETS_BUCKET=yes.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${PARAMS_FILE:=${HERE}/params.json}"

if [[ -z "${REGION:-}" ]]; then
    if [[ -f "${PARAMS_FILE}" ]]; then
        REGION="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('Region',''))" "${PARAMS_FILE}" 2>/dev/null || echo "")"
    fi
fi
if [[ -z "${REGION:-}" ]]; then
    REGION="$(aws configure get region 2>/dev/null || echo "")"
fi
if [[ -z "${REGION}" ]]; then
    echo "Error: no region. Set REGION=..." >&2
    exit 1
fi

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"

# Derive the same per-cluster slug deploy.sh uses (from params.json), so the
# stack name + assets bucket resolve to the same values that were created.
HYPERPOD_CLUSTER_NAME=""
if [[ -f "${PARAMS_FILE}" ]]; then
    HYPERPOD_CLUSTER_NAME="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('HyperPodClusterName',''))" "${PARAMS_FILE}" 2>/dev/null || echo "")"
fi
NAME_PREFIX="$(python3 - "${HYPERPOD_CLUSTER_NAME}" <<'PY'
import re, sys
s = (sys.argv[1] if len(sys.argv) > 1 else "").lower()
s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
s = re.sub(r"-{2,}", "-", s)[:20].strip("-")
print(s or "cluster")
PY
)"

: "${STACK_NAME:=hyperpod-devops-agent-${NAME_PREFIX}}"

# The marker bucket is stack-managed, but CloudFormation refuses to delete a
# non-empty S3 bucket. Empty it first so the stack delete succeeds. Name is
# deterministic and matches the template's MarkerBucket.
MARKER_BUCKET="hpda-markers-${NAME_PREFIX}-${ACCOUNT_ID}-${REGION}"
if aws s3api head-bucket --bucket "${MARKER_BUCKET}" --region "${REGION}" 2>/dev/null; then
    echo "==> Emptying marker bucket s3://${MARKER_BUCKET} (so the stack can delete it)"
    aws s3 rm "s3://${MARKER_BUCKET}" --recursive --region "${REGION}" >/dev/null 2>&1 || true
fi

echo "==> Deleting stack ${STACK_NAME} in ${REGION}"
if aws cloudformation describe-stacks --region "${REGION}" --stack-name "${STACK_NAME}" >/dev/null 2>&1; then
    aws cloudformation delete-stack --region "${REGION}" --stack-name "${STACK_NAME}"
    echo "    waiting for stack delete to complete..."
    aws cloudformation wait stack-delete-complete --region "${REGION}" --stack-name "${STACK_NAME}" || true
    echo "    stack deleted"
else
    echo "    stack not found"
fi

if [[ "${KEEP_ASSETS_BUCKET:-no}" != "yes" ]]; then
    ASSETS_BUCKET="hpda-assets-${NAME_PREFIX}-${ACCOUNT_ID}-${REGION}"
    echo
    echo "==> Removing assets bucket s3://${ASSETS_BUCKET}"
    if aws s3api head-bucket --bucket "${ASSETS_BUCKET}" --region "${REGION}" 2>/dev/null; then
        aws s3 rm "s3://${ASSETS_BUCKET}" --recursive --region "${REGION}" >/dev/null 2>&1 || true
        aws s3api delete-bucket --bucket "${ASSETS_BUCKET}" --region "${REGION}" >/dev/null 2>&1 \
            && echo "    removed bucket" || echo "    could not remove bucket (may be non-empty or already gone)"
    else
        echo "    bucket not found"
    fi
fi

echo
echo "Teardown complete."
