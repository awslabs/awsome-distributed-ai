#!/bin/bash
set -euo pipefail

SUPPORTED_TYPES=(
    "ml.p4d.24xlarge" "ml.p4de.24xlarge"
    "ml.p5.48xlarge" "ml.p5e.48xlarge" "ml.p5en.48xlarge" "ml.p6e-gb200.36xlarge"
    "ml.trn1.2xlarge" "ml.trn1.32xlarge" "ml.trn1n.32xlarge" "ml.trn2.48xlarge"
    "ml.trn2u.48xlarge" "ml.p6-b200.48xlarge" "ml.p6-b300.48xlarge"
)

is_supported_instance() {
    local instance_type="$1"
    for supported in "${SUPPORTED_TYPES[@]}"; do
        if [ "$instance_type" = "$supported" ]; then
            return 0
        fi
    done
    return 1
}

# Fetch all node data in a single API call, filter to supported nodes,
# and dynamically extract all topology.k8s.aws/network-node-layer-* labels
fetch_supported_nodes_json() {
    kubectl get nodes -o json | jq -c '[
        .items[] |
        select(
            .metadata.labels["node.kubernetes.io/instance-type"] as $it |
            ['"$(printf '"%s",' "${SUPPORTED_TYPES[@]}" | sed 's/,$//')"'] | index($it)
        ) |
        {
            name: .metadata.name,
            instance_type: .metadata.labels["node.kubernetes.io/instance-type"],
            layers: [
                .metadata.labels | to_entries[] |
                select(.key | startswith("topology.k8s.aws/network-node-layer-")) |
                { num: (.key | ltrimstr("topology.k8s.aws/network-node-layer-") | tonumber), val: .value }
            ] | sort_by(.num)
        }
    ]'
}

echo "Fetching cluster node data..."
NODES_JSON=$(fetch_supported_nodes_json)

total_nodes=$(kubectl get nodes --no-headers | wc -l | tr -d ' ')
supported_count=$(echo "$NODES_JSON" | jq 'length')
skipped=$((total_nodes - supported_count))

if [ "$skipped" -gt 0 ]; then
    echo "Skipping $skipped node(s) with unsupported instance types."
fi

if [ "$supported_count" -eq 0 ]; then
    echo "No supported instance types found in the cluster."
    exit 1
fi
echo "Found $supported_count supported node(s)."

# Detect max layer count from the data
max_layer=$(echo "$NODES_JSON" | jq '[.[].layers[].num] | max')
echo "Detected $max_layer topology layer(s)."

echo "Building topology..."

mermaid="flowchart TD"
mermaid+=$'\n    A["Cluster Topology"]'

for layer_num in $(seq 1 "$max_layer"); do
    unique_values=($(echo "$NODES_JSON" | jq -r --argjson n "$layer_num" '[.[].layers[] | select(.num == $n) | .val] | unique | .[]'))

    for val in "${unique_values[@]}"; do
        val_id=$(echo "$val" | sed 's/[^a-zA-Z0-9]/_/g')

        if [ "$layer_num" -eq 1 ]; then
            mermaid+=$'\n    A --> L1_'"${val_id}[\"Layer 1: ${val}\"]"
        else
            parent_layer=$((layer_num - 1))
            parent=""
            parent=$(echo "$NODES_JSON" | jq -r --argjson n "$layer_num" --argjson pn "$parent_layer" --arg v "$val" \
                '[.[] | select(.layers[] | select(.num == $n and .val == $v)) | .layers[] | select(.num == $pn) | .val][0] // empty')
            parent_id=$(echo "$parent" | sed 's/[^a-zA-Z0-9]/_/g')
            mermaid+=$'\n    L'"${parent_layer}_${parent_id} --> L${layer_num}_${val_id}[\"Layer ${layer_num}: ${val}\"]"
        fi
    done
done

# Link nodes to their deepest layer
echo "$NODES_JSON" | jq -r --argjson n "$max_layer" '.[] | "\(.name) \(.layers[] | select(.num == $n) | .val)"' | while read -r node deepest_parent; do
    node_id=$(echo "$node" | sed 's/[^a-zA-Z0-9]/_/g')
    parent_id=$(echo "$deepest_parent" | sed 's/[^a-zA-Z0-9]/_/g')
    echo "    L${max_layer}_${parent_id} --> N_${node_id}[\"${node}\"]"
done | {
    while IFS= read -r line; do
        mermaid+=$'\n'"$line"
    done

    echo "$mermaid"

    OUTPUT_FILE="topology.html"
    cat > "$OUTPUT_FILE" <<EOF
<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Cluster Topology</title></head>
<body>
  <pre class="mermaid">
${mermaid}
  </pre>
  <script src="https://cdn.jsdelivr.net/npm/mermaid@11.14.0/dist/mermaid.min.js"></script>
  <script>mermaid.initialize({startOnLoad:true});</script>
</body></html>
EOF

    echo "Topology saved to $OUTPUT_FILE — open in a browser to view"
}
