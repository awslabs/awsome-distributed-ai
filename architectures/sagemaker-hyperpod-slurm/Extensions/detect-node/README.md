# Node Detection for SageMaker HyperPod Extensions

Utility script that detects the node type and writes node-specific information
to `/opt/ml/config/nodeinfo.json`. Designed to run once at the start of
`OnInitComplete`, before other extensions that need to know the node type.

## Why This Exists

HyperPod extension scripts often need to know whether they're running on a
controller, compute, or login node. For example, Slurm accounting setup should
only run on the controller, while GPU metric exporters should only run on
compute nodes.

### Why Not Detect via Slurm Services?

Detecting node type by checking Slurm daemons (e.g., `systemctl is-active
slurmctld`) is unreliable because `OnInitComplete` and lifecycle scripts can
run before or in parallel with Slurm service startup. Using the platform
config files is the recommended approach — it is deterministic, instant, and
has no dependency on Slurm or any other service being ready.

### The Reliable Method: Platform Config Files

HyperPod writes two JSON config files to `/opt/ml/config/` on every node
**before** any lifecycle scripts run:

- `provisioning_parameters.json` |-- contains `controller_group` and
  `login_group` field names that identify which instance groups serve
  which roles.
- `resource_config.json` |-- contains all instance groups with their instances,
  including each instance's `CustomerIpAddress`.

The detection algorithm:
1. Get this node's IP address (via UDP socket |-- no network call needed)
2. Find which instance group contains this IP in `resource_config.json`
3. Compare that group's name to `provisioning_parameters.json`:
   - Matches `controller_group`  **controller**
   - Matches `login_group`  **login**
   - Otherwise  **compute**

This is deterministic, instant, and has zero dependency on Slurm or any
other service. It works during initial cluster creation, scale-up, and
node replacement.

## Output

The script writes `/opt/ml/config/nodeinfo.json`:

```json
{
  "node_type": "controller",
  "instance_group_name": "controller",
  "instance_name": "controller-1",
  "instance_id": "i-042a465749062463f",
  "instance_type": "ml.t3.medium",
  "ip_address": "10.1.197.201",
  "cluster_name": "my-cluster",
  "cluster_arn": "arn:aws:sagemaker:us-east-1:123456789012:cluster/abc123"
}
```

## Usage

### In run_extensions.sh

Run `detect_node.sh` first, before other extensions:

```bash
# Detect node type (writes /opt/ml/config/nodeinfo.json)
run_feature "detect-node" "$SCRIPT_DIR/detect-node/detect_node.sh"

# Other extensions can now read nodeinfo.json
run_feature "add-users" "$SCRIPT_DIR/add-users/add_users.sh"
run_feature "observability" "$SCRIPT_DIR/observability/setup_observability.sh"
```

### Reading nodeinfo.json from bash

```bash
NODE_TYPE=$(python3 -c "import json; print(json.load(open('/opt/ml/config/nodeinfo.json'))['node_type'])")

if [[ "$NODE_TYPE" == "controller" ]]; then
    echo "Running on controller"
fi
```

### Reading nodeinfo.json from Python

```python
import json

with open('/opt/ml/config/nodeinfo.json') as f:
    nodeinfo = json.load(f)

if nodeinfo['node_type'] == 'controller':
    print("Running on controller")
```

## Files

| File | Description |
|------|-------------|
| `detect_node.sh` | Node detection script (writes nodeinfo.json) |
| `README.md` | This documentation |
