#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# ============================================================
# Run a miles GRPO recipe FROM the Ray head pod, so the operator's machine needs only
# kubectl (plus AWS auth) -- no local `ray` CLI, no `kubectl port-forward`, no local `envsubst`.
#
# Why this exists: the shipped recipes end in `ray job submit --address http://127.0.0.1:8265`,
# which normally runs on your laptop and therefore needs the ray CLI installed locally, a
# port-forward to the dashboard, and a ray version that matches the cluster (the wheel for the
# cluster's exact ray version may not even exist for your local Python). Running the recipe
# INSIDE the head pod removes all three: ray is already there, 127.0.0.1:8265 is the head's own
# dashboard, and the version always matches. This is a convenience wrapper around the SAME
# recipe -- the local flow in the README Quick Start still works unchanged.
#
# What it does NOT do: it does not create infrastructure, build images, prepare data, or deploy
# manifests. Do those first per the README (steps 0-5): the RayCluster must already be deployed
# with its head Running on a CUDA-capable (GPU) node, and env_vars must be filled in.
#
# Usage:
#   ./run-on-cluster.sh [--recipe run_grpo_qwen3_4b.sh] [--env ./env_vars] [--namespace default] [--dry-run]
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="${NAMESPACE:-default}"
RECIPE="run_grpo_qwen3_4b.sh"
ENV_FILE="${SCRIPT_DIR}/env_vars"
DRY_RUN=0
REMOTE_DIR="/tmp/miles-run"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --recipe)         RECIPE="$2"; shift 2;;
    --env)            ENV_FILE="$2"; shift 2;;
    -n|--namespace)   NAMESPACE="$2"; shift 2;;
    --dry-run)        DRY_RUN=1; shift;;
    -h|--help)        sed -n '2,22p' "${BASH_SOURCE[0]}"; exit 0;;
    *) echo "[ERROR] unknown argument: $1" >&2; exit 1;;
  esac
done

# ---- preconditions (fail-fast, with the fix in the message) ----
command -v kubectl >/dev/null 2>&1 || { echo "[ERROR] kubectl not found on PATH." >&2; exit 1; }
[[ -f "${ENV_FILE}" ]] || { echo "[ERROR] env file not found: ${ENV_FILE}. Copy env_vars.colocated.example to env_vars and fill it in (README step 1)." >&2; exit 1; }
[[ -f "${SCRIPT_DIR}/recipe/${RECIPE}" ]] || { echo "[ERROR] recipe not found: recipe/${RECIPE}" >&2; exit 1; }

# ---- discover: the Ray head pod ----
HEAD="$(kubectl -n "${NAMESPACE}" get pods -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
[[ -n "${HEAD}" ]] || { echo "[ERROR] no Ray head pod (label ray.io/node-type=head) in namespace '${NAMESPACE}'. Deploy the RayCluster first (README step 5)." >&2; exit 1; }
PHASE="$(kubectl -n "${NAMESPACE}" get pod "${HEAD}" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
[[ "${PHASE}" == "Running" ]] || { echo "[ERROR] head pod '${HEAD}' is '${PHASE:-unknown}', not Running. Wait for it (it pulls the ~18 GB image on first start)." >&2; exit 1; }

echo "[INFO] namespace: ${NAMESPACE}"
echo "[INFO] head pod:  ${HEAD}"
echo "[INFO] recipe:    recipe/${RECIPE}"
echo "[INFO] env file:  ${ENV_FILE}"
echo "[INFO] remote:    ${REMOTE_DIR} (recipe/, scripts/, env_vars)"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DRY-RUN] would ship recipe/ scripts/ and '${ENV_FILE}' to ${HEAD}:${REMOTE_DIR}, then run:"
  echo "  kubectl -n ${NAMESPACE} exec ${HEAD} -- bash -lc 'cd ${REMOTE_DIR} && ENV_FILE=${REMOTE_DIR}/env_vars bash recipe/${RECIPE}'"
  exit 0
fi

# ---- ship: tar-pipe the test-case files into the head pod ----
# (kubectl cp is tar under the hood; piping tar ourselves lets us exclude .git and send exactly
# these paths.) The env file is copied to ${REMOTE_DIR}/env_vars regardless of its local name.
kubectl -n "${NAMESPACE}" exec "${HEAD}" -- bash -lc "rm -rf ${REMOTE_DIR} && mkdir -p ${REMOTE_DIR}"
tar cf - -C "${SCRIPT_DIR}" recipe scripts | kubectl -n "${NAMESPACE}" exec -i "${HEAD}" -- tar xf - -C "${REMOTE_DIR}"
tar cf - -C "$(cd "$(dirname "${ENV_FILE}")" && pwd)" "$(basename "${ENV_FILE}")" | kubectl -n "${NAMESPACE}" exec -i "${HEAD}" -- tar xf - -C "${REMOTE_DIR}"
if [[ "$(basename "${ENV_FILE}")" != "env_vars" ]]; then
  kubectl -n "${NAMESPACE}" exec "${HEAD}" -- bash -lc "mv -f ${REMOTE_DIR}/$(basename "${ENV_FILE}") ${REMOTE_DIR}/env_vars"
fi

# ---- run: exec the recipe inside the head pod ----
echo "[INFO] launching recipe inside ${HEAD}; ray job submit targets the head's own dashboard (127.0.0.1:8265)."
kubectl -n "${NAMESPACE}" exec "${HEAD}" -- bash -lc "cd ${REMOTE_DIR} && ENV_FILE=${REMOTE_DIR}/env_vars bash recipe/${RECIPE}"
