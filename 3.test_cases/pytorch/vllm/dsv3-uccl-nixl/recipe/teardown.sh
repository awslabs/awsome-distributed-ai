#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Delete all Deployments this sample created. Keeps the namespace and HF secret.

set -euo pipefail

if [[ -z "${NAMESPACE:-}" ]]; then
    echo "ERROR: source setup/env_vars first." >&2
    exit 1
fi

echo "==> Deleting dsv3-disagg Deployments in namespace ${NAMESPACE}"
kubectl delete deployment --selector=app=dsv3-disagg --namespace "${NAMESPACE}" --ignore-not-found
echo "==> Done."
