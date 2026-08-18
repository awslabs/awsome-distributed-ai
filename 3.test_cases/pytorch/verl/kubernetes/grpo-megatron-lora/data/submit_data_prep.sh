#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# =============================================================================
# Submit Data Preparation as a Ray Job
# Convenience wrapper that sources env_vars, sets up RAY_ADDRESS,
# and submits data/prepare_data.py as a Ray job.
#
# Usage:
#   ./data/submit_data_prep.sh --datasets eurus apps taco codecontests \
#       --output-dir /fsx/data/verl/data
#
#   ./data/submit_data_prep.sh --datasets eurus apps taco codecontests \
#       --output-dir /fsx/data/verl/data \
#       --mix --mix-output /fsx/data/verl/data/mixed \
#       --mix-ratios eurus:1.0 apps:0.5 taco:0.3 codecontests:1.0
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load environment variables
if [ -f "${REPO_DIR}/env_vars" ]; then
    source "${REPO_DIR}/env_vars"
fi

# Ray dashboard address (exposed via ALB ingress with OIDC auth)
# Default assumes a local port-forward (see README). For an ALB-hosted dashboard,
# set RAY_ADDRESS (or RAY_DASHBOARD_HOSTNAME) in env_vars; RAY_ADDRESS wins below.
RAY_DASHBOARD_HOSTNAME="${RAY_DASHBOARD_HOSTNAME:-localhost:8265}"
# Accept a bare host (assume https, ALB case) or a full URL (port-forward case).
case "${RAY_DASHBOARD_HOSTNAME}" in
    *://*) _ray_default="${RAY_DASHBOARD_HOSTNAME}" ;;
    localhost:*|127.0.0.1:*) _ray_default="http://${RAY_DASHBOARD_HOSTNAME}" ;;
    *) _ray_default="https://${RAY_DASHBOARD_HOSTNAME}" ;;
esac
RAY_ADDRESS="${RAY_ADDRESS:-${_ray_default}}"

# Validate connectivity
echo "=============================================="
echo "Submit Data Preparation Job"
echo "=============================================="
echo "Ray address: ${RAY_ADDRESS}"
echo "Working dir: ${REPO_DIR}"
echo "Arguments:   $*"
echo "=============================================="

# Check Ray dashboard is reachable
if ! curl -s --connect-timeout 5 "${RAY_ADDRESS}" > /dev/null 2>&1; then
    echo ""
    echo "WARNING: Cannot reach Ray dashboard at ${RAY_ADDRESS}"
    echo ""
    echo "Set RAY_ADDRESS or RAY_DASHBOARD_HOSTNAME to the correct address."
    echo ""
    read -r -p "Continue anyway? [y/N] " response
    if [[ ! "${response}" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Build runtime env JSON with environment variables and pip deps for the Ray job.
# Ray 2.53+ removed --env-var; use --runtime-env-json instead.
# Data prep requires 'datasets' and 'numpy' which aren't in the base Ray image.
HF_HOME_VAL="${HF_HOME:-/fsx/data/verl/cache/huggingface}"
RUNTIME_ENV_JSON='{"env_vars": {"HF_HOME": "'"${HF_HOME_VAL}"'"'
if [ -n "${HF_TOKEN:-}" ]; then
    RUNTIME_ENV_JSON="${RUNTIME_ENV_JSON}"', "HF_TOKEN": "'"${HF_TOKEN}"'"'
fi
# Upper bounds matter here: this step produces the parquet every run trains and
# evaluates on, so an unbounded datasets/pyarrow major bump can silently change the
# on-disk schema between two runs being compared, and "pip_check": false hides it.
# numpy is pinned exactly to the version the reported eval runs recorded.
RUNTIME_ENV_JSON="${RUNTIME_ENV_JSON}"'}, "pip": {"packages": ["datasets>=3.0,<5", "numpy==2.4.6", "pyarrow>=14,<25", "huggingface_hub>=0.26,<1"], "pip_check": false}}'

# Build Ray CLI auth headers if available (OIDC-protected dashboard)
HEADERS_ARGS=()
if [ -n "${RAY_HEADERS:-}" ]; then
    HEADERS_ARGS=(--headers "${RAY_HEADERS}")
fi

# Submit the Ray job
echo ""
echo "Submitting Ray job..."
ray job submit \
    --address "${RAY_ADDRESS}" \
    ${HEADERS_ARGS[@]+"${HEADERS_ARGS[@]}"} \
    --working-dir "${REPO_DIR}" \
    --no-wait \
    --runtime-env-json "${RUNTIME_ENV_JSON}" \
    -- python data/prepare_data.py "$@"

echo ""
echo "=============================================="
echo "Job submitted! Monitor at: ${RAY_ADDRESS}/#/jobs"
echo "=============================================="
