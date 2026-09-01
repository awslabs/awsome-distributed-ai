# Cluster Topology Visualization

Visualizes the inter-node network topology of a SageMaker HyperPod EKS cluster using Kubernetes `topology.k8s.aws/network-node-layer-*` labels. Outputs a [Mermaid](https://mermaid.js.org/) flowchart to stdout and generates an HTML file for local viewing.

## Prerequisites

- `kubectl` configured for your HyperPod EKS cluster
- `jq` installed
- Cluster must contain at least one node with a supported instance type

## Supported Instance Types

| Family | Instance Types |
|--------|---------------|
| P4 | `ml.p4d.24xlarge`, `ml.p4de.24xlarge` |
| P5 | `ml.p5.48xlarge`, `ml.p5e.48xlarge`, `ml.p5en.48xlarge` |
| P6 | `ml.p6e-gb200.36xlarge`, `ml.p6-b200.48xlarge`, `ml.p6-b300.48xlarge` |
| Trn1 | `ml.trn1.2xlarge`, `ml.trn1.32xlarge`, `ml.trn1n.32xlarge` |
| Trn2 | `ml.trn2.48xlarge`, `ml.trn2u.48xlarge` |

Nodes with unsupported instance types (e.g., `t3.medium`) are automatically skipped.

> **Note:** The number of topology layers is detected dynamically. Most instance types have 3 layers, but `p6-b200` and `p6-b300` instances have 4. The script handles both automatically.

## Usage

```bash
bash visualize_topology.sh
```

## Output

1. Mermaid flowchart printed to stdout
2. `topology.html` generated in the current directory — open in a browser to view

### Example Mermaid Output

```mermaid
flowchart TD
    A["Cluster Topology"]
    A --> L1_nn_a29d806520fbf708b["Layer 1: nn-a29d806520fbf708b"]
    L1_nn_a29d806520fbf708b --> L2_nn_57b8dbe21569e0155["Layer 2: nn-57b8dbe21569e0155"]
    L2_nn_57b8dbe21569e0155 --> L3_nn_4ff6cc69685dce646["Layer 3: nn-4ff6cc69685dce646"]
    L3_nn_4ff6cc69685dce646 --> N_ip_10_192_7_11_ec2_internal["ip-10-192-7-11.ec2.internal"]
```
