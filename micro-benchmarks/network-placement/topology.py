# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Group Kubernetes nodes and pods by EC2 network placement, from the topology.k8s.aws/network-node-layer-1..3 labels.

Amazon EKS labels each node of a supported instance type with the network nodes above it, from layer 1 (top) to
layer 3 (closest to the instance). Two nodes with the same layer-3 label are closest; nodes that differ at layer 3
but share layer 2 are one level farther apart. Raw network node IDs are compared in memory and printed relabelled
(L3-a, L3-b, ...) unless --raw is given.

  groups  print each node (or each probe pod's node) with its relabelled layers, then the layer-3 groups
  pairs   print every pair of probe pods with the number of layers they share, in a shuffled, seeded order
  orders  print a sorted and an interleaved rank order of probe pods, and how many ring hops cross layer 3

Usage:
  kubectl get nodes -o json > nodes.json; kubectl get pods -l app=placement-probe -o json > pods.json
  python3 topology.py groups --nodes nodes.json [--pods pods.json] [--instance-type p5en.48xlarge] [--raw]
  python3 topology.py pairs  --nodes nodes.json --pods pods.json [--seed 42]
  python3 topology.py orders --nodes nodes.json --pods pods.json [--only pod-a,pod-b,...]
"""
import argparse
import itertools
import json
import random
import sys

KEY = "topology.k8s.aws/network-node-layer-%d"
LAYERS = (1, 2, 3)


def load_nodes(path, instance_type=None):
    nodes = {}
    for n in json.load(open(path))["items"]:
        labels = n["metadata"].get("labels", {})
        if instance_type and labels.get("node.kubernetes.io/instance-type") != instance_type:
            continue
        nodes[n["metadata"]["name"]] = [labels.get(KEY % i) for i in LAYERS]
    return nodes


def load_pods(path):
    """Running pods -> node name."""
    return {p["metadata"]["name"]: p["spec"]["nodeName"] for p in json.load(open(path))["items"]
            if p["status"].get("phase") == "Running" and p["spec"].get("nodeName")}


class Relabel:
    """Short names per layer (L3-a, L3-b, ...), assigned in sorted order of the raw IDs of every labelled node, so the
    same nodes.json gives the same names in every command."""

    def __init__(self, nodes, raw=False):
        self.raw = raw
        self.maps = [{v: f"L{i + 1}-{chr(97 + k)}" for k, v in enumerate(sorted({n[i] for n in nodes.values() if n[i]}))}
                     for i in range(len(LAYERS))]

    def __call__(self, layers):
        return [("-" if not v else v if self.raw else self.maps[i][v]) for i, v in enumerate(layers)]


def shared_layers(a, b):
    """Number of network-node layers two nodes share, counted from the top (3 = same layer-3 node)."""
    d = 0
    for x, y in zip(a, b):
        if x and x == y:
            d += 1
        else:
            break
    return d


def crossings(order, l3):
    """Ring hops between consecutive instances (cyclic, as NCCL's ring) that change layer-3 node."""
    n = len(order)
    return sum(l3[order[i]] != l3[order[(i + 1) % n]] for i in range(n)) if n > 1 else 0


def interleave(pods, l3):
    """Order pods so that as few neighbours as possible (cyclically) share a layer-3 node."""
    groups = {}
    for p in sorted(pods):
        groups.setdefault(l3[p], []).append(p)
    order = []
    while any(groups.values()):
        prev = l3[order[-1]] if order else None
        first = l3[order[0]] if order else None
        last_pick = sum(len(v) for v in groups.values()) == 1
        cands = sorted((g for g, v in groups.items() if v), key=lambda g: (-len(groups[g]), str(g)))
        pick = next((g for g in cands if g != prev and not (last_pick and g == first)), None)
        pick = pick or next((g for g in cands if g != prev), None) or cands[0]
        order.append(groups[pick].pop(0))
    return order


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["groups", "pairs", "orders"])
    ap.add_argument("--nodes", required=True, help="kubectl get nodes -o json")
    ap.add_argument("--pods", help="kubectl get pods -l app=placement-probe -o json")
    ap.add_argument("--instance-type")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", help="comma-separated pod names to use, in any order (orders)")
    ap.add_argument("--raw", action="store_true", help="print raw network node IDs instead of short names")
    a = ap.parse_args()
    nodes = load_nodes(a.nodes, a.instance_type)
    pods = load_pods(a.pods) if a.pods else None
    rel = Relabel(nodes, a.raw)

    if a.cmd == "groups":
        rows = ([(rel(nodes[n]), p, n) for p, n in pods.items() if n in nodes] if pods
                else [(rel(v), "", n) for n, v in nodes.items()])
        for r, p, n in sorted(rows):
            print(" ".join(r + ([p] if p else []) + [n]))
        if not rows:
            sys.exit("no labelled nodes found")
        unlabelled = sum(1 for r, _, _ in rows if r[2] == "-")
        groups = {}
        for r, p, n in rows:
            groups.setdefault(tuple(r), []).append(n)
        sizes = ", ".join(f"{k[2]}: {len(v)}" for k, v in sorted(groups.items()))
        print(f"\n{len(rows)} nodes in {len(groups)} layer-3 groups ({sizes})")
        if unlabelled:
            print(f"{unlabelled} nodes have no network-node labels (instance type without instance topology support?)")
        return

    if not pods:
        sys.exit("--pods is required for " + a.cmd)
    missing = [p for p, n in pods.items() if n not in nodes or not nodes[n][2]]
    if missing:
        sys.exit(f"pods on nodes without network-node labels: {' '.join(missing)}")

    if a.cmd == "pairs":
        pairs = [(x, y, shared_layers(nodes[pods[x]], nodes[pods[y]])) for x, y in itertools.combinations(sorted(pods), 2)]
        random.Random(a.seed).shuffle(pairs)
        for x, y, d in pairs:
            print(x, y, d)
        return

    use = a.only.split(",") if a.only else sorted(pods)
    unknown = [p for p in use if p not in pods]
    if unknown:
        sys.exit(f"not running probe pods: {' '.join(unknown)}")
    l3 = {p: nodes[pods[p]][2] for p in use}
    srt = sorted(use, key=lambda p: (nodes[pods[p]], pods[p]))
    ilv = interleave(use, l3)
    for name, order in (("sorted", srt), ("interleaved", ilv)):
        print(f"{name} {crossings(order, l3)} {len(order)} {' '.join(order)}")
    print("layer3", " ".join(f"{p}={rel(nodes[pods[p]])[2]}" for p in srt))


if __name__ == "__main__":
    main()
