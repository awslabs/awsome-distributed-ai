#!/bin/bash
# Script to add Ray metrics scraping to HyperPod ObservabilityConfig
# This uses customServiceScrapeTargets which won't get overwritten by the operator

set -e

# Load environment variables
source setup/env_vars

echo "Adding Ray metrics scraping to ObservabilityConfig..."
echo "Ray Namespace: ${RAY_NAMESPACE}"
export RAY_CLUSTER_NAME=$(kubectl get raycluster -o jsonpath='{.items[0].metadata.name}')
echo "Ray Cluster: $RAY_CLUSTER_NAME"

# Create the patch JSON
# We're adding customServiceScrapeTargets to scrape Ray head metrics
cat > /tmp/ray-metrics-patch.json <<EOF
{
  "spec": {
    "customServiceScrapeTargets": [
      {
        "target": "${RAY_CLUSTER_NAME}-head-svc.${RAY_NAMESPACE}.svc.cluster.local:8080",
        "metricsPath": "/metrics",
        "scrapeInterval": 30
      }
    ]
  }
}
EOF

echo ""
echo "Patch content:"
cat /tmp/ray-metrics-patch.json
echo ""

# Apply the patch
kubectl patch observabilityconfig hyperpod-observability \
  -n hyperpod-observability \
  --type merge \
  --patch-file /tmp/ray-metrics-patch.json

echo ""
echo "✓ Patch applied successfully!"
echo ""
echo "The operator will reconcile this change within ~10 minutes."
echo "To speed it up, restart the OTEL collector:"
echo ""
echo "  kubectl rollout restart deployment hyperpod-observability-central-collector -n hyperpod-observability"
echo ""

# Cleanup
rm /tmp/ray-metrics-patch.json
