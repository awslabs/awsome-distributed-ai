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
# with its head and workers Running, and env_vars must be filled in.
#
# Usage:
#   ./run-on-cluster.sh [--recipe run_grpo_qwen3_4b.sh] [--env ./env_vars]
#                       [--namespace <ns>] [--cluster miles-ray] [--dry-run]
#
# NAMESPACE: flag, then the caller's export, then the env file; the last two disagreeing is
# reported. RAY_CLUSTER_NAME: flag, then the env file, then an inherited export.
# --cluster must match metadata.name in kubernetes/raycluster.yaml.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE="run_grpo_qwen3_4b.sh"
ENV_FILE="${SCRIPT_DIR}/env_vars"
DRY_RUN=0
REMOTE_DIR="/tmp/miles-run"
# ray.io/cluster=<metadata.name> is the only label that tells this cluster's head from another
# Ray cluster's in the same namespace.
RAY_CLUSTER="${RAY_CLUSTER_NAME:-}"
RAY_CLUSTER_FROM_FLAG=0
NAMESPACE_FROM_FLAG=0
NAMESPACE="${NAMESPACE:-}"
CALLER_NAMESPACE="${NAMESPACE}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --recipe)         RECIPE="$2"; shift 2;;
    --env)            ENV_FILE="$2"; shift 2;;
    -n|--namespace)   NAMESPACE="$2"; NAMESPACE_FROM_FLAG=1; shift 2;;
    --cluster)        RAY_CLUSTER="$2"; RAY_CLUSTER_FROM_FLAG=1; shift 2;;
    --dry-run)        DRY_RUN=1; shift;;
    -h|--help)        awk '/^# ={10,}$/{b++; next} b==1' "${BASH_SOURCE[0]}"; exit 0;;
    *) echo "[ERROR] unknown argument: $1" >&2; exit 1;;
  esac
done

# ---- preconditions (fail-fast, with the fix in the message) ----
command -v kubectl >/dev/null 2>&1 || { echo "[ERROR] kubectl not found on PATH." >&2; exit 1; }
[[ -f "${ENV_FILE}" ]] || { echo "[ERROR] env file not found: ${ENV_FILE}. Copy env_vars.colocated.example (dense 4B) or env_vars.moe.example (30B MoE) to env_vars and fill it in (README step 1)." >&2; exit 1; }
[[ -f "${SCRIPT_DIR}/recipe/${RECIPE}" ]] || { echo "[ERROR] recipe not found: recipe/${RECIPE}" >&2; exit 1; }

# Sourced once in a subshell with the exit status checked: a swallowed failure would mean
# submitting to whatever cluster sits in the default namespace.
# shellcheck source=/dev/null
ENV_VALUES="$( ( set +u; source "${ENV_FILE}" >/dev/null 2>&1 || exit 1
                 printf '%s\n%s\n%s\n' "${NAMESPACE:-}" "${WORKER_REPLICAS:-}" "${RAY_CLUSTER_NAME:-}" ) )" || {
  echo "[ERROR] sourcing '${ENV_FILE}' failed. Run 'bash -n ${ENV_FILE}' and 'source ${ENV_FILE}'" >&2
  echo "[ERROR] to see the error; this script cannot tell which cluster to target without it." >&2
  exit 1
}
ENV_NAMESPACE="$(printf '%s' "${ENV_VALUES}" | sed -n '1p')"
ENV_WORKER_REPLICAS="$(printf '%s' "${ENV_VALUES}" | sed -n '2p')"
ENV_RAY_CLUSTER="$(printf '%s' "${ENV_VALUES}" | sed -n '3p')"

# The caller's shell wins over the file so an export still works; a disagreement is reported.
if [[ "${NAMESPACE_FROM_FLAG}" == "0" ]]; then
  if [[ -n "${CALLER_NAMESPACE}" ]]; then
    if [[ -n "${ENV_NAMESPACE}" && "${ENV_NAMESPACE}" != "${CALLER_NAMESPACE}" ]]; then
      echo "[WARN] NAMESPACE is '${CALLER_NAMESPACE}' in your shell but '${ENV_NAMESPACE}' in" >&2
      echo "[WARN] '${ENV_FILE}'. Using '${CALLER_NAMESPACE}'; pass --namespace to be explicit." >&2
    fi
  elif [[ -n "${ENV_NAMESPACE}" ]]; then
    NAMESPACE="${ENV_NAMESPACE}"
  fi
fi
NAMESPACE="${NAMESPACE:-default}"
if [[ "${RAY_CLUSTER_FROM_FLAG}" == "0" && -n "${ENV_RAY_CLUSTER}" ]]; then
  RAY_CLUSTER="${ENV_RAY_CLUSTER}"
fi
if [[ -z "${RAY_CLUSTER}" ]]; then
  echo "[ERROR] RAY_CLUSTER_NAME is not set in '${ENV_FILE}'. It names the RayCluster this" >&2
  echo "[ERROR] script submits to and is what renders metadata.name in the manifest, so it" >&2
  echo "[ERROR] cannot be guessed. Re-copy env_vars from env_vars.colocated.example or" >&2
  echo "[ERROR] env_vars.moe.example, or pass --cluster <name>." >&2
  exit 1
fi
# An unusable value must stop the run, not silently disable the worker check below.
if ! [[ "${ENV_WORKER_REPLICAS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] WORKER_REPLICAS from '${ENV_FILE}' is '${ENV_WORKER_REPLICAS}', not a positive" >&2
  echo "[ERROR] integer. It is derived at the end of the example env files; an env_vars copied" >&2
  echo "[ERROR] before that block existed will not have it. Without it this script cannot tell" >&2
  echo "[ERROR] whether the cluster has the workers the recipe is about to ask for." >&2
  exit 1
fi

# .items[0] on ray.io/node-type=head alone picks an arbitrary head when the namespace holds more
# than one Ray cluster, and every precondition below then validates that unrelated pod.
SELECTOR="ray.io/cluster=${RAY_CLUSTER},ray.io/node-type=head"
# `while read` rather than mapfile: mapfile is bash 4+ and macOS ships 3.2.
HEADS=()
while IFS= read -r _line; do
  [[ -n "${_line}" ]] && HEADS+=("${_line}")
done < <(kubectl -n "${NAMESPACE}" get pods -l "${SELECTOR}" \
           -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null || true)
if [[ "${#HEADS[@]}" -eq 0 ]]; then
  echo "[ERROR] no Ray head pod matching '${SELECTOR}' in namespace '${NAMESPACE}'." >&2
  echo "[ERROR] Deploy the RayCluster first (README step 4). If your cluster has a different" >&2
  echo "[ERROR] metadata.name, pass --cluster <name>." >&2
  exit 1
fi
if [[ "${#HEADS[@]}" -gt 1 ]]; then
  echo "[ERROR] ${#HEADS[@]} head pods match '${SELECTOR}' in namespace '${NAMESPACE}':" >&2
  printf '[ERROR]   %s\n' ${HEADS[@]+"${HEADS[@]}"} >&2
  echo "[ERROR] Refusing to guess which one to submit to." >&2
  exit 1
fi
HEAD="${HEADS[0]}"
# Ready, not Running: a container reaches Running before Ray's GCS and dashboard are up.
HEAD_STATE="$(kubectl -n "${NAMESPACE}" get pod "${HEAD}" \
                -o jsonpath='{.status.phase}{" "}{range .status.conditions[?(@.type=="Ready")]}{.status}{end}' \
                2>/dev/null || true)"
if [[ "${HEAD_STATE}" != "Running True" ]]; then
  echo "[ERROR] head pod '${HEAD}' is '${HEAD_STATE:-unknown}', not Running and Ready." >&2
  echo "[ERROR] Wait for it; the first start pulls the ~18 GB image." >&2
  exit 1
fi

# ---- verify: the workers the recipe will ask for actually exist and are Ready ----
# With the workers still Pending the recipe submits, prints a banner, and then waits on a
# placement group that cannot form -- which reads as a slow start.
WORKERS=()
while IFS= read -r _line; do
  [[ -n "${_line}" ]] && WORKERS+=("${_line}")
done < <(kubectl -n "${NAMESPACE}" get pods \
           -l "ray.io/cluster=${RAY_CLUSTER},ray.io/node-type=worker" \
           -o jsonpath='{range .items[*]}{.metadata.name}{" "}{.status.phase}{" "}{range .status.conditions[?(@.type=="Ready")]}{.status}{end}{"\n"}{end}' \
           2>/dev/null || true)
unset _line
READY_WORKERS=0
# bash before 4.4 treats "${arr[@]}" on an empty array as unbound under `set -u`, so "no workers
# at all" -- the case this check exists for -- would die before printing the diagnosis.
for _w in ${WORKERS[@]+"${WORKERS[@]}"}; do
  [[ "${_w}" == *" Running True" ]] && READY_WORKERS=$((READY_WORKERS + 1))
done
unset _w
if [[ "${READY_WORKERS}" -lt "${ENV_WORKER_REPLICAS}" ]]; then
  echo "[ERROR] ${READY_WORKERS} of ${ENV_WORKER_REPLICAS} GPU workers are Running and Ready" >&2
  echo "[ERROR] in namespace '${NAMESPACE}' for cluster '${RAY_CLUSTER}':" >&2
  if [[ "${#WORKERS[@]}" -gt 0 ]]; then
    printf '[ERROR]   %s\n' ${WORKERS[@]+"${WORKERS[@]}"} >&2
  else
    echo "[ERROR]   <no worker pods at all>" >&2
  fi
  echo "[ERROR] The recipe would submit and then wait forever on placement. A worker stuck" >&2
  echo "[ERROR] Pending is usually FailedScheduling: check the GPU node label, the EFA count," >&2
  echo "[ERROR] and whether the pool has ${ENV_WORKER_REPLICAS} free nodes." >&2
  exit 1
fi

echo "[INFO] namespace: ${NAMESPACE}"
echo "[INFO] cluster:   ${RAY_CLUSTER}"
echo "[INFO] head pod:  ${HEAD}"
echo "[INFO] workers:   ${READY_WORKERS} Ready of ${ENV_WORKER_REPLICAS}"
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
