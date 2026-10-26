# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Group Kubernetes nodes and pods by EC2 network placement.

Amazon EKS labels each node of a supported instance type with the network
nodes above it, topology.k8s.aws/network-node-layer-1 (top) through
topology.k8s.aws/network-node-layer-3 (closest to the instance). Two nodes
with the same layer-3 label are closest; nodes that differ at layer 3 but
share layer 2 are one level farther apart. Raw network node IDs are compared
in memory and printed relabelled (L3-a, L3-b, ...) unless --raw is given.

  groups  each node (or probe pod's node) with relabelled layers, then the
          layer-3 groups
  pairs   every pair of probe pods with the number of layers they share,
          in a shuffled, seeded order
  orders  a sorted and an interleaved node order of probe pods, and how
          many ring hops cross layer 3

Usage:
  kubectl get nodes -o json > nodes.json
  kubectl get pods -l app=placement-probe -o json > pods.json
  python3 topology.py groups --nodes nodes.json [--pods pods.json]
                             [--instance-type p5en.48xlarge] [--raw]
  python3 topology.py pairs --nodes nodes.json --pods pods.json [--seed 42]
  python3 topology.py orders --nodes nodes.json --pods pods.json
                             [--only pod-a,pod-b,...]
"""

import argparse
import itertools
import json
import random
import sys

KEY = "topology.k8s.aws/network-node-layer-%d"
TYPE_KEY = "node.kubernetes.io/instance-type"
LAYERS = (1, 2, 3)


def load_json(path):
    """Parse a JSON file."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_nodes(path, instance_type=None):
    """Node name -> [layer-1, layer-2, layer-3 label], optionally filtered."""
    nodes = {}
    for n in load_json(path)["items"]:
        labels = n["metadata"].get("labels", {})
        if instance_type and labels.get(TYPE_KEY) != instance_type:
            continue
        nodes[n["metadata"]["name"]] = [labels.get(KEY % i) for i in LAYERS]
    return nodes


def load_pods(path):
    """Running pod name -> node name."""
    pods = {}
    for p in load_json(path)["items"]:
        node = p["spec"].get("nodeName")
        if p["status"].get("phase") == "Running" and node:
            pods[p["metadata"]["name"]] = node
    return pods


def relabeller(nodes, raw=False):
    """Return a function mapping layer labels to names (L3-a, L3-b, ...).

    Names follow the sorted order of the raw IDs of every labelled node, so
    the same nodes.json gives the same names in every command. With
    raw=True the function returns the raw IDs.
    """
    maps = []
    for i in range(len(LAYERS)):
        ids = sorted({n[i] for n in nodes.values() if n[i]})
        maps.append({v: f"L{i + 1}-{chr(97 + k)}" for k, v in enumerate(ids)})

    def relabel(layers):
        out = []
        for i, v in enumerate(layers):
            if not v:
                out.append("-")
            else:
                out.append(v if raw else maps[i][v])
        return out

    return relabel


def shared_layers(a, b):
    """Layers two nodes share from the top (3 = same layer-3 node)."""
    d = 0
    for i in range(len(LAYERS)):
        if a[i] and a[i] == b[i]:
            d += 1
        else:
            break
    return d


def crossings(order, l3):
    """Ring hops (cyclic, as NCCL's ring) that change layer-3 node."""
    n = len(order)
    if n < 2:
        return 0
    return sum(l3[order[i]] != l3[order[(i + 1) % n]] for i in range(n))


def interleave(pods, l3):
    """Order pods so as few (cyclic) neighbours as possible share layer 3."""
    groups = {}
    for p in sorted(pods):
        groups.setdefault(l3[p], []).append(p)
    order = []
    while any(groups.values()):
        prev = l3[order[-1]] if order else None
        first = l3[order[0]] if order else None
        last_pick = sum(len(v) for v in groups.values()) == 1
        cands = sorted(
            (g for g, v in groups.items() if v),
            key=lambda g: (-len(groups[g]), str(g)),
        )
        pick = next(
            (g for g in cands if g != prev and not (last_pick and g == first)),
            None,
        )
        pick = pick or next((g for g in cands if g != prev), None)
        order.append(groups[pick or cands[0]].pop(0))
    return order


def cmd_groups(nodes, pods, relabel):
    """Print nodes with relabelled layers, then a layer-3 group summary."""
    if pods:
        rows = [
            (relabel(nodes[n]), p, n) for p, n in pods.items() if n in nodes
        ]
    else:
        rows = [(relabel(v), "", n) for n, v in nodes.items()]
    for r, p, n in sorted(rows):
        print(" ".join(r + ([p] if p else []) + [n]))
    if not rows:
        sys.exit("no labelled nodes found")
    unlabelled = sum(1 for r, _, _ in rows if r[2] == "-")
    groups = {}
    for r, _, n in rows:
        groups.setdefault(tuple(r), []).append(n)
    sizes = ", ".join(f"{k[2]}: {len(v)}" for k, v in sorted(groups.items()))
    print(f"\n{len(rows)} nodes in {len(groups)} layer-3 groups ({sizes})")
    if unlabelled:
        print(
            f"{unlabelled} nodes have no network-node labels "
            "(instance type without instance topology support?)"
        )


def cmd_pairs(nodes, pods, seed):
    """Print every pod pair with its shared layers, in a seeded order."""
    pairs = [
        (x, y, shared_layers(nodes[pods[x]], nodes[pods[y]]))
        for x, y in itertools.combinations(sorted(pods), 2)
    ]
    # Shuffle the measurement order so near and far pairs interleave in
    # time. Not security-relevant.
    random.Random(seed).shuffle(pairs)  # noqa: S311  # nosec B311
    for x, y, d in pairs:
        print(x, y, d)


def cmd_orders(nodes, pods, relabel, only):
    """Print the sorted and interleaved orders and their crossings."""
    use = only.split(",") if only else sorted(pods)
    unknown = [p for p in use if p not in pods]
    if unknown:
        sys.exit(f"not running probe pods: {' '.join(unknown)}")
    l3 = {p: nodes[pods[p]][2] for p in use}
    srt = sorted(use, key=lambda p: (nodes[pods[p]], pods[p]))
    ilv = interleave(use, l3)
    for name, order in (("sorted", srt), ("interleaved", ilv)):
        hops = crossings(order, l3)
        print(f"{name} {hops} {len(order)} {' '.join(order)}")
    names = [f"{p}={relabel(nodes[pods[p]])[2]}" for p in srt]
    print("layer3", " ".join(names))


def parse_args():
    """Command-line options."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("cmd", choices=["groups", "pairs", "orders"])
    ap.add_argument("--nodes", required=True, help="kubectl get nodes -o json")
    ap.add_argument("--pods", help="kubectl get pods -o json")
    ap.add_argument("--instance-type")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--only",
        help="comma-separated pod names to use, in any order (orders)",
    )
    ap.add_argument(
        "--raw",
        action="store_true",
        help="print raw network node IDs instead of short names",
    )
    return ap.parse_args()


def main():
    """Parse arguments and run one command."""
    a = parse_args()
    nodes = load_nodes(a.nodes, a.instance_type)
    pods = load_pods(a.pods) if a.pods else None
    relabel = relabeller(nodes, a.raw)
    if a.cmd == "groups":
        cmd_groups(nodes, pods, relabel)
        return
    if not pods:
        sys.exit("--pods is required for " + a.cmd)
    missing = [p for p, n in pods.items() if n not in nodes or not nodes[n][2]]
    if missing:
        sys.exit(
            "pods on nodes without network-node labels: " + " ".join(missing)
        )
    if a.cmd == "pairs":
        cmd_pairs(nodes, pods, a.seed)
    else:
        cmd_orders(nodes, pods, relabel, a.only)


if __name__ == "__main__":
    main()
