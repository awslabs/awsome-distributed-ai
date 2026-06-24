#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Setup Dynamo observability: metrics → AMP → Grafana
# Creates a stable metrics service and patches HyperPod ObservabilityConfig
set -e

NAMESPACE=dynamo-system

echo "📊 Setting up Dynamo observability..."

# --- Step 1: Create stable metrics service ---
echo ""
echo "📡 Step 1: Creating metrics service..."
kubectl get svc dynamo-metrics -n "$NAMESPACE" >/dev/null 2>&1 && echo "  Service already exists, skipping" || {
  cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Service
metadata:
  name: dynamo-metrics
  namespace: ${NAMESPACE}
spec:
  selector:
    nvidia.com/dynamo-component-type: frontend
  ports:
    - name: metrics
      port: 8000
      targetPort: 8000
EOF
  echo "  ✅ Service created"
}

# --- Step 2: Patch ObservabilityConfig ---
echo ""
echo "🔧 Step 2: Patching ObservabilityConfig..."
OBS_CONFIG=$(kubectl get observabilityconfig -n hyperpod-observability -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
if [ -z "$OBS_CONFIG" ]; then
  echo "  ⚠️  No ObservabilityConfig found. Skipping (metrics won't reach AMP/Grafana)."
else
  EXISTING=$(kubectl get observabilityconfig "$OBS_CONFIG" -n hyperpod-observability -o jsonpath='{.spec.customServiceScrapeTargets}' 2>/dev/null)
  if echo "$EXISTING" | grep -q "dynamo-metrics"; then
    echo "  Scrape target already configured, skipping"
  else
    kubectl patch observabilityconfig "$OBS_CONFIG" -n hyperpod-observability --type merge -p '{
      "spec": {
        "customServiceScrapeTargets": [
          {
            "target": "dynamo-metrics.dynamo-system.svc.cluster.local:8000",
            "metricsPath": "/metrics",
            "scrapeInterval": 30
          }
        ]
      }
    }'
    echo "  ✅ ObservabilityConfig patched"
  fi
fi

# --- Step 3: Verify ---
echo ""
echo "⏳ Step 3: Verifying..."
sleep 5
EP=$(kubectl get endpoints dynamo-metrics -n "$NAMESPACE" -o jsonpath='{.subsets[0].addresses[0].ip}' 2>/dev/null)
if [ -n "$EP" ]; then
  echo "  ✅ Metrics service resolving to ${EP}"
else
  echo "  ⚠️  No endpoints yet (deploy inference first, then metrics will flow)"
fi

echo ""
echo "✅ Observability setup complete!"
echo "   Metrics: dynamo_frontend_* → AMP → Grafana"
echo "   See OBSERVABILITY.md for Grafana access and queries."
