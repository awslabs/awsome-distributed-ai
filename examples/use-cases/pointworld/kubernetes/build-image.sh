#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# build-image.sh -- build pointworld.Dockerfile on-cluster with Kaniko and push
# to Amazon ECR. Produces a linux/amd64 image regardless of your workstation
# architecture (Apple Silicon / arm64 cannot build this image natively).
#
# Usage (from the test case root):
#   source env_vars
#   ./kubernetes/build-image.sh
#
# Requires: kubectl, aws CLI, and env_vars sourced (IMAGE_URI, REGISTRY,
# AWS_REGION, NAMESPACE). Creates an ECR repo if missing, stages the Dockerfile
# as a ConfigMap, creates a short-lived ECR credential secret, runs the Kaniko
# Pod, tails its logs, and cleans up the helper resources.

set -euo pipefail

for v in IMAGE_URI REGISTRY AWS_REGION NAMESPACE; do
  if [ -z "${!v:-}" ]; then
    echo "ERROR: \$$v is not set. Run 'source env_vars' first." >&2
    exit 1
  fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE="${SCRIPT_DIR}/../pointworld.Dockerfile"
MANIFEST="${SCRIPT_DIR}/build-image.yaml"
POD=pointworld-build
CM=pointworld-dockerfile
SECRET=pointworld-ecr-cred
REPO_NAME="${IMAGE_URI#*/}"; REPO_NAME="${REPO_NAME%%:*}"   # strip registry + tag

cleanup() {
  echo "== cleanup helper resources =="
  kubectl delete configmap "$CM" -n "$NAMESPACE" --ignore-not-found >/dev/null 2>&1 || true
  kubectl delete secret "$SECRET" -n "$NAMESPACE" --ignore-not-found >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "== ensure ECR repo exists: ${REPO_NAME} =="
aws ecr describe-repositories --repository-names "$REPO_NAME" --region "$AWS_REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$REPO_NAME" --region "$AWS_REGION" >/dev/null

echo "== stage Dockerfile as ConfigMap: ${CM} =="
kubectl delete configmap "$CM" -n "$NAMESPACE" --ignore-not-found >/dev/null 2>&1 || true
kubectl create configmap "$CM" -n "$NAMESPACE" --from-file=Dockerfile="$DOCKERFILE"

echo "== create ECR credential secret: ${SECRET} =="
TOKEN=$(aws ecr get-login-password --region "$AWS_REGION")
AUTH=$(printf 'AWS:%s' "$TOKEN" | base64 | tr -d '\n')
DOCKERCFG=$(printf '{"auths":{"%s":{"auth":"%s"}}}' "$REGISTRY" "$AUTH")
kubectl delete secret "$SECRET" -n "$NAMESPACE" --ignore-not-found >/dev/null 2>&1 || true
kubectl create secret generic "$SECRET" -n "$NAMESPACE" \
  --from-literal=.dockerconfigjson="$DOCKERCFG"

echo "== launch Kaniko build (-> ${IMAGE_URI}) =="
kubectl delete pod "$POD" -n "$NAMESPACE" --ignore-not-found >/dev/null 2>&1 || true
envsubst '$IMAGE_URI $NAMESPACE' < "$MANIFEST" | kubectl apply -f -

echo "== wait for build pod to start =="
kubectl wait --for=condition=Initialized "pod/${POD}" -n "$NAMESPACE" --timeout=300s || true
echo "== streaming Kaniko logs (the torch/CUDA install + push takes ~10-15 min) =="
# Stream logs for visibility. If the stream drops (e.g. a client timeout), the
# poll loop below still waits for the real pod phase before deciding success.
kubectl logs -f "pod/${POD}" -c kaniko -n "$NAMESPACE" 2>/dev/null || true

echo "== waiting for build pod to reach a terminal phase =="
PHASE=Unknown
for _ in $(seq 1 120); do          # up to ~20 min
  PHASE=$(kubectl get pod "$POD" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || echo Unknown)
  case "$PHASE" in
    Succeeded|Failed) break ;;
  esac
  sleep 10
done

echo "== build pod phase: ${PHASE} =="
if [ "$PHASE" = "Succeeded" ]; then
  echo "Build succeeded. Image pushed to ${IMAGE_URI}"
  kubectl delete pod "$POD" -n "$NAMESPACE" --ignore-not-found >/dev/null 2>&1 || true
else
  echo "Build did NOT succeed (phase=${PHASE}). Leaving pod ${POD} for inspection:" >&2
  echo "  kubectl describe pod ${POD} -n ${NAMESPACE}" >&2
  echo "  kubectl logs ${POD} -c kaniko -n ${NAMESPACE}" >&2
  exit 1
fi
