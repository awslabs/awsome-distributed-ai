#!/usr/bin/env bash
# One-command deploy of the unified HyperPod x DevOps Agent stack.
#
#   1. Resolve region + account.
#   2. Read params.json (flat key->value map; see params.example.json).
#   3. Auto-discover the underlying EKS cluster name from the HyperPod cluster
#      (empty for Slurm -> EKS access is skipped by the template).
#   4. Ensure the S3 assets bucket exists, then sync the skills into it and
#      capture the manifest + content-hash version.
#   5. Embed the Lambda sources into hyperpod_devops_agent.yaml.
#   6. aws cloudformation deploy with the assembled parameters.
#
# Env overrides: REGION, STACK_NAME, PARAMS_FILE, SKIP_SKILLS=yes,
#   TEMPLATE_SRC (source template), TEMPLATE_OUT (generated deployable template).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# STACK_NAME defaults per-cluster (below, once the cluster slug is known) so
# multiple clusters get distinct stacks. Override by exporting STACK_NAME.
: "${PARAMS_FILE:=${HERE}/params.json}"
: "${TEMPLATE_SRC:=${HERE}/hyperpod_devops_agent.template.yaml}"
: "${TEMPLATE_OUT:=${HERE}/hyperpod_devops_agent.yaml}"

if [[ ! -f "${PARAMS_FILE}" ]]; then
    echo "Error: ${PARAMS_FILE} not found. Copy deploy/params.example.json to deploy/params.json and edit it." >&2
    exit 1
fi

# Region: env override, else params.json, else AWS CLI default.
if [[ -z "${REGION:-}" ]]; then
    REGION="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('Region',''))" "${PARAMS_FILE}")"
fi
if [[ -z "${REGION}" ]]; then
    REGION="$(aws configure get region 2>/dev/null || echo "")"
fi
if [[ -z "${REGION}" ]]; then
    echo "Error: no region. Set REGION=..., add \"Region\" to params.json, or 'aws configure set region <region>'." >&2
    exit 1
fi

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"

# ------------------------------------------------------------ read params.json
# Build the USER_PARAMS array of "Key=Value" entries for every real
# CloudFormation parameter (skip __-prefixed keys and the Region helper).
# Read line-by-line (not mapfile) for bash 3.2 compatibility (macOS default).
# Render the params first into a variable (NOT a process substitution): a
# validation failure inside `$(...)` propagates via set -e, whereas a failure
# inside `< <(...)` does not — the loop would just read fewer lines and the
# deploy would proceed with a silently-dropped param. The Python is passed via
# `-c '...'` rather than a heredoc because bash 3.2 (the macOS default this
# script supports) can't parse a heredoc inside command substitution.
PARAMS_RENDERED="$(python3 -c '
import json, sys
data = json.load(open(sys.argv[1]))
reserved = {"Region"}
for k, v in data.items():
    if k.startswith("__") or k in reserved:
        continue
    # Values must be JSON strings. A natural-JSON boolean/number would render via
    # its Python repr (true -> "True"), failing the template AllowedValues with a
    # confusing error; a value with a newline would split into a bogus extra
    # --parameter-overrides entry. Reject up front -- see params.example.json.
    if not isinstance(v, str):
        sys.exit("params.json: %s must be a JSON string (got %s) -- see params.example.json" % (k, type(v).__name__))
    print("%s=%s" % (k, v))
' "${PARAMS_FILE}")"
USER_PARAMS=()
while IFS= read -r line; do
    [[ -n "${line}" ]] && USER_PARAMS+=("${line}")
done <<< "${PARAMS_RENDERED}"

HYPERPOD_CLUSTER_NAME="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('HyperPodClusterName',''))" "${PARAMS_FILE}")"
if [[ -z "${HYPERPOD_CLUSTER_NAME}" ]]; then
    echo "Error: HyperPodClusterName is required in ${PARAMS_FILE}." >&2
    exit 1
fi

# Bucket-safe, <=20-char slug of the cluster name. Makes per-cluster resource
# names (buckets, IAM roles via NamePrefix) unique so multiple clusters can
# coexist in one account/region. Lowercase, non-alnum -> '-', collapse repeats,
# trim leading/trailing '-', cap at 20 chars.
NAME_PREFIX="$(python3 - "${HYPERPOD_CLUSTER_NAME}" <<'PY'
import re, sys
s = sys.argv[1].lower()
s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
s = re.sub(r"-{2,}", "-", s)[:20].strip("-")
print(s or "cluster")
PY
)"

# Per-cluster default stack name (override with STACK_NAME env).
: "${STACK_NAME:=hyperpod-devops-agent-${NAME_PREFIX}}"

echo "==> Configuration"
echo "    Region:           ${REGION}"
echo "    Account:          ${ACCOUNT_ID}"
echo "    HyperPod cluster: ${HYPERPOD_CLUSTER_NAME}"
echo "    Name prefix:      ${NAME_PREFIX}"
echo "    Stack name:       ${STACK_NAME}"

# ------------------------------------------------- auto-discover EKS cluster name
# One describe-cluster call, WITHOUT suppressing stderr, so a nonexistent cluster
# (typo) or an API failure (permissions, wrong region) is a hard error — not
# silently treated as "Slurm". A typo'd name must not deploy a billing-relevant
# stack that monitors nothing.
if ! CLUSTER_JSON="$(aws sagemaker describe-cluster \
        --cluster-name "${HYPERPOD_CLUSTER_NAME}" \
        --region "${REGION}" --output json)"; then
    echo "Error: could not describe HyperPod cluster '${HYPERPOD_CLUSTER_NAME}' in ${REGION}." >&2
    echo "  Verify the cluster name and region (and that your credentials can call" >&2
    echo "  sagemaker:DescribeCluster). See the error above." >&2
    exit 1
fi

# Branch on the presence of the EKS orchestrator ARN in the SUCCESSFUL response.
EKS_ARN="$(printf '%s' "${CLUSTER_JSON}" | python3 -c "import json,sys; print((json.load(sys.stdin).get('Orchestrator') or {}).get('Eks', {}).get('ClusterArn') or '')")"
if [[ -n "${EKS_ARN}" ]]; then
    EKS_CLUSTER_NAME="${EKS_ARN##*/}"
    echo "    EKS cluster:      ${EKS_CLUSTER_NAME} (auto-discovered)"
    # Pre-flight: EKS access entries need API/API_AND_CONFIG_MAP auth mode.
    AUTH_MODE="$(aws eks describe-cluster --name "${EKS_CLUSTER_NAME}" --region "${REGION}" \
        --query 'cluster.accessConfig.authenticationMode' --output text 2>/dev/null || echo "")"
    if [[ "${AUTH_MODE}" != "API" && "${AUTH_MODE}" != "API_AND_CONFIG_MAP" ]]; then
        echo "Error: EKS cluster '${EKS_CLUSTER_NAME}' has authenticationMode=${AUTH_MODE}." >&2
        echo "  DevOps Agent EKS access requires API or API_AND_CONFIG_MAP. Update with:" >&2
        echo "  aws eks update-cluster-config --name ${EKS_CLUSTER_NAME} --region ${REGION} \\" >&2
        echo "      --access-config authenticationMode=API_AND_CONFIG_MAP" >&2
        exit 1
    fi
else
    EKS_CLUSTER_NAME=""
    echo "    EKS cluster:      <none — Slurm-orchestrated; EKS access skipped>"
    # Continuous Provisioning is a prerequisite for Slurm clusters: without it,
    # SageMaker does not support list-cluster-events (the canonical record of
    # replacement attempts the RCA skill reconstructs its timeline from), and the
    # HyperPod EventBridge event shape differs from what the webhook bridge and
    # skills expect. Warn loudly; don't hard-fail (the operator may accept the
    # reduced coverage, and EKS-orchestrated clusters are always Continuous).
    PROV_MODE="$(printf '%s' "${CLUSTER_JSON}" | python3 -c "import json,sys; print(json.load(sys.stdin).get('NodeProvisioningMode') or '')")"
    echo "    Provisioning:     ${PROV_MODE:-<none>}"
    if [[ "${PROV_MODE}" != "Continuous" ]]; then
        echo "    WARNING: cluster '${HYPERPOD_CLUSTER_NAME}' is NOT in Continuous Provisioning mode." >&2
        echo "             Slurm clusters require Continuous Provisioning for full coverage:" >&2
        echo "               - list-cluster-events is unsupported (RCA timeline reconstruction degrades)" >&2
        echo "               - the HyperPod EventBridge event format differs (bridge/skills expect Continuous)" >&2
        echo "             Deploying anyway; investigations will have reduced fidelity on this cluster." >&2
    fi
fi

# --------------------------------------------------------- boto3 preflight
# Building the skill-uploader Lambda bundles a current boto3 (the Lambda runtime's
# built-in boto3 predates the DevOps Agent Asset API). Verify it's present up front
# so we fail cleanly BEFORE creating any AWS resources, rather than mid-deploy.
# Skipped when SKIP_SKILLS=yes (build-skill-uploader doesn't run then).
if [[ "${SKIP_SKILLS:-no}" != "yes" ]]; then
    if ! python3 "${HERE}/prepare_deployment.py" check-boto3; then
        exit 1
    fi
fi

# ------------------------------------------------------ ensure assets bucket
# The bucket is always needed: it holds the staged skill zips AND (because the
# embedded template exceeds CloudFormation's 51,200-byte inline limit) the
# template itself, which `aws cloudformation deploy --s3-bucket` uploads.
ASSETS_BUCKET="hpda-assets-${NAME_PREFIX}-${ACCOUNT_ID}-${REGION}"
echo
echo "==> Ensuring assets bucket s3://${ASSETS_BUCKET}"
if ! aws s3api head-bucket --bucket "${ASSETS_BUCKET}" --region "${REGION}" 2>/dev/null; then
    if [[ "${REGION}" == "us-east-1" ]]; then
        aws s3api create-bucket --bucket "${ASSETS_BUCKET}" --region "${REGION}" >/dev/null
    else
        aws s3api create-bucket --bucket "${ASSETS_BUCKET}" --region "${REGION}" \
            --create-bucket-configuration "LocationConstraint=${REGION}" >/dev/null
    fi
    echo "    created bucket"
else
    echo "    bucket already exists"
fi

# ------------------------------------------------------------------ sync skills
SKILLS_BUCKET=""
SKILLS_VERSION="none"
SKILLS_MANIFEST="[]"
UPLOADER_BUCKET=""
UPLOADER_KEY=""
if [[ "${SKIP_SKILLS:-no}" != "yes" ]]; then
    echo
    echo "==> Syncing skills to S3"
    SYNC_JSON="$(python3 "${HERE}/prepare_deployment.py" sync-skills --bucket "${ASSETS_BUCKET}" --region "${REGION}")"
    SKILLS_BUCKET="${ASSETS_BUCKET}"
    SKILLS_VERSION="$(echo "${SYNC_JSON}" | python3 -c "import json,sys; print(json.load(sys.stdin)['version'])")"
    SKILLS_MANIFEST="$(echo "${SYNC_JSON}" | python3 -c "import json,sys; print(json.dumps(json.load(sys.stdin)['manifest']))")"
    echo "    skills version: ${SKILLS_VERSION}"

    echo
    echo "==> Packaging skill-uploader Lambda (bundles the installed boto3 for the Asset API)"
    UPLOADER_JSON="$(python3 "${HERE}/prepare_deployment.py" build-skill-uploader --bucket "${ASSETS_BUCKET}" --region "${REGION}")"
    UPLOADER_BUCKET="$(echo "${UPLOADER_JSON}" | python3 -c "import json,sys; print(json.load(sys.stdin)['bucket'])")"
    UPLOADER_KEY="$(echo "${UPLOADER_JSON}" | python3 -c "import json,sys; print(json.load(sys.stdin)['key'])")"
else
    echo
    echo "==> SKIP_SKILLS=yes — skill upload disabled for this deploy"
fi

# ------------------------------------------------------------------ embed + deploy
echo
echo "==> Embedding Lambda code into template"
python3 "${HERE}/prepare_deployment.py" embed --in "${TEMPLATE_SRC}" --out "${TEMPLATE_OUT}"

# Assemble parameter overrides: user params first, then the auto-derived ones
# (later values win in aws cloudformation deploy, so auto-derived override any
# stale user entries for these managed keys).
PARAMS=("${USER_PARAMS[@]}")
PARAMS+=("NamePrefix=${NAME_PREFIX}")
PARAMS+=("EksClusterName=${EKS_CLUSTER_NAME}")
PARAMS+=("AssetsBucket=${SKILLS_BUCKET}")
PARAMS+=("SkillsVersion=${SKILLS_VERSION}")
PARAMS+=("SkillsManifest=${SKILLS_MANIFEST}")
PARAMS+=("SkillUploaderS3Bucket=${UPLOADER_BUCKET}")
PARAMS+=("SkillUploaderS3Key=${UPLOADER_KEY}")

echo
echo "==> Deploying stack ${STACK_NAME}"
# The embedded template exceeds CloudFormation's 51,200-byte inline limit, so
# --s3-bucket uploads it to S3 first (staged under the assets bucket).
aws cloudformation deploy \
    --region "${REGION}" \
    --stack-name "${STACK_NAME}" \
    --template-file "${TEMPLATE_OUT}" \
    --s3-bucket "${ASSETS_BUCKET}" \
    --s3-prefix cfn-templates \
    --capabilities CAPABILITY_NAMED_IAM \
    --parameter-overrides "${PARAMS[@]}"

echo
echo "==> Stack outputs"
aws cloudformation describe-stacks \
    --region "${REGION}" \
    --stack-name "${STACK_NAME}" \
    --query 'Stacks[0].Outputs' \
    --output table \
    --no-cli-pager

echo
echo "Done."
