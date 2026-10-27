# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Group efa-latency-matrix.yaml's results by network distance.

Each pair of instances is labelled with the number of network-node layers
its two nodes share, counted from topology.k8s.aws/network-node-layer-1
down. On a three-layer type such as p5en, 3 means the same bottom-layer
network node; on p6-b200 and p6-b300 that is 4. Round 1 of every pair is
dropped as warm-up.

Usage:
  kubectl logs job/efa-latency-launcher > latency.log
  kubectl get nodes -o json > nodes.json
  python3 analyze_efa_latency.py latency.log nodes.json
"""

import json
import statistics as st
import sys
from collections import defaultdict

LAYER = "topology.k8s.aws/network-node-layer-{}"


def layers(node):
    """Return the network nodes above a node, from layer 1 down."""
    labels = node["metadata"].get("labels", {})
    out = []
    for i in range(1, 5):
        value = labels.get(LAYER.format(i))
        if value is None:
            break
        out.append(value)
    return tuple(out)


def shared(first, second):
    """Return how many leading layers two network-node tuples share."""
    count = 0
    while count < min(len(first), len(second)):
        if first[count] != second[count]:
            break
        count += 1
    return count


def read_log(path):
    """Return host->node, (a, b, size)->usecs, (host, size)->usecs, NAs."""
    host_node, pairs, floors = {}, defaultdict(list), defaultdict(list)
    missing = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.strip().split(",")
            if parts[0] == "NODE" and len(parts) == 3:
                host_node[parts[1]] = parts[2]
            elif parts[0] in {"PAIR", "FLOOR"} and len(parts) == 6:
                kind, rnd, first, second, size, usec = parts
                if usec == "NA":
                    missing += 1
                elif kind == "FLOOR":
                    floors[(first, int(size))].append(float(usec))
                elif rnd != "1":
                    pairs[(first, second, int(size))].append(float(usec))
    return host_node, pairs, floors, missing


def band_summary(size, bands, floor):
    """Print the floor and each shared-layer band for one message size."""
    print(f"\n{size} B, usec/xfer (mean of rounds per pair):")
    if floor:
        print(
            f"  floor (within an instance): {len(floor)} instances, mean "
            f"{st.mean(floor):.2f} ({min(floor):.2f} to {max(floor):.2f})"
        )
    depths = sorted(bands, reverse=True)
    for depth in depths:
        v = bands[depth]
        print(
            f"  share {depth} layers: {len(v)} pairs, mean "
            f"{st.mean(v):.2f} ({min(v):.2f} to {max(v):.2f})"
        )
    for i in range(1, len(depths)):
        near, far = depths[i - 1], depths[i]
        gap = st.mean(bands[far]) - st.mean(bands[near])
        apart = max(bands[near]) < min(bands[far])
        print(
            f"  {near} -> {far} shared layers adds {gap:+.2f} us "
            f"({gap / st.mean(bands[near]):+.0%}); every {near}-layer "
            f"pair faster than every {far}-layer pair: {apart}"
        )


def pair_rows(log, nodes):
    """Return (size, shared layers, node a, node b, usecs) per pair."""
    host_node, pairs = log[0], log[1]
    rows = []
    for (first, second, size), usecs in pairs.items():
        node_a, node_b = host_node[first], host_node[second]
        depth = shared(nodes[node_a], nodes[node_b])
        rows.append((size, depth, node_a, node_b, usecs))
    return sorted(rows, key=lambda r: (r[0], -r[1], st.mean(r[4])))


def main():
    """Print per-pair means, then a summary per message size."""
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    log = read_log(sys.argv[1])
    with open(sys.argv[2], encoding="utf-8") as fh:
        nodes = {
            n["metadata"]["name"]: layers(n) for n in json.load(fh)["items"]
        }
    rows = pair_rows(log, nodes)
    print(
        "size_bytes,shared_layers,node_a,node_b,rounds,mean_us,min_us,max_us"
    )
    for size, depth, node_a, node_b, v in rows:
        print(
            f"{size},{depth},{node_a},{node_b},{len(v)},{st.mean(v):.2f},"
            f"{min(v):.2f},{max(v):.2f}"
        )
    for size in sorted({r[0] for r in rows}):
        bands = defaultdict(list)
        for row in rows:
            if row[0] == size:
                bands[row[1]].append(st.mean(row[4]))
        floor = [st.mean(v) for (_h, sz), v in log[2].items() if sz == size]
        band_summary(size, bands, floor)
    if log[3]:
        print(f"\n{log[3]} measurements returned no result (NA)")


if __name__ == "__main__":
    main()
