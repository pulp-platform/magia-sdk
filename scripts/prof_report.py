#!/usr/bin/env python3

import argparse
import csv
import re
from collections import defaultdict

# one line per (node, tile); the layer's name comes from its header line
NODE_LINE = re.compile(r"\[PROF\] hid=(\d+) layer_idx=(\d+) node_idx=(\d+) node=(\S+) "
                       r"op=(\S+) node_cycles=(\d+) alloc=(\d+) in=(\d+) cmp=(\d+) "
                       r"out=(\d+) snc=(\d+) prep=(\d+)")
LAYER_HEADER = re.compile(r"\[PROF\] ### Profiling for Layer (\d+) \((.+)\) ###")

# node_cycles is the whole node, the rest are its parts; what is left is unaccounted
FIELDS = ("node_cycles", "alloc", "in", "cmp", "out", "snc", "prep")
PARTS = ("alloc", "in", "cmp", "out", "snc", "prep")


def parse_args():
    parser = argparse.ArgumentParser(description="Report the per-node cycle profiling of a MAGIA run")

    parser.add_argument("log", help="run log containing the [PROF] lines")
    parser.add_argument("--agg", choices=("crit", "mean"), default="crit",
                        help="how to aggregate the tiles: critical tile (default) or mean")
    parser.add_argument("--top", type=int, default=15, help="how many nodes to rank (default: 15)")
    parser.add_argument("--csv", help="also write the per-node per-tile table to this file")

    args = parser.parse_args()
    return args


def parse_log(path):
    samples = defaultdict(dict)
    layer_names, node_names, node_layer, node_op = {}, {}, {}, {}

    with open(path) as f:
        for line in f:
            header = LAYER_HEADER.search(line)
            if header:
                layer_names[int(header.group(1))] = header.group(2)
                continue

            data = NODE_LINE.search(line)
            if data:
                tile, layer, node = (int(x) for x in data.groups()[:3])
                values = (int(x) for x in data.groups()[5:12])
                samples[node][tile] = dict(zip(FIELDS, values))
                node_names[node] = data.group(4)
                node_op[node] = data.group(5)
                node_layer[node] = layer

    if not samples:
        raise SystemExit(f"no [PROF] data lines in {path}: "
                         "was the run built with enable_node_profiling=1?")

    return samples, layer_names, node_names, node_layer, node_op


def aggregate_tiles(per_tile, field, how):
    if how == "mean":
        return sum(tile[field] for tile in per_tile.values()) // len(per_tile)

    critical = max(per_tile, key=lambda tile: per_tile[tile]["node_cycles"])
    return per_tile[critical][field]


def collapse_nodes(samples, how):
    return {node: {field: aggregate_tiles(per_tile, field, how) for field in FIELDS}
            for node, per_tile in samples.items()}


def unaccounted(entry):
    return entry["node_cycles"] - sum(entry[field] for field in PARTS)


def group_nodes(nodes, key_of):
    totals = defaultdict(lambda: defaultdict(int))
    count = defaultdict(int)
    order = []

    for node in sorted(nodes):
        key = key_of(node)
        if key not in totals:
            order.append(key)
        count[key] += 1
        for field in FIELDS:
            totals[key][field] += nodes[node][field]

    return totals, count, order


def print_group_table(title, label, totals, count, order, grand_total):
    print(f"\n{title}")
    print(f"{label:36s} {'nodes':>5s} {'cycles':>10s} {'%':>5s} {'compute':>9s} "
          f"{'dma(in+out)':>11s} {'prep(host)':>10s} {'sync':>8s} {'alloc':>7s} {'other':>8s}")

    def row(name, n, e):
        print(f"{name[:36]:36s} {n:5d} {e['node_cycles']:10d} "
              f"{100 * e['node_cycles'] / grand_total:5.1f} "
              f"{e['cmp']:9d} {e['in'] + e['out']:11d} {e['prep']:10d} {e['snc']:8d} "
              f"{e['alloc']:7d} {unaccounted(e):8d}")

    for key in order:
        row(key, count[key], totals[key])

    total = {field: sum(totals[key][field] for key in order) for field in FIELDS}
    row("TOTAL", sum(count.values()), total)
    print(f"{'  share of total':36s} {'':5s} {'':10s} {'':5s} "
          f"{100 * total['cmp'] / grand_total:8.1f}% "
          f"{100 * (total['in'] + total['out']) / grand_total:10.1f}% "
          f"{100 * total['prep'] / grand_total:9.1f}% "
          f"{100 * total['snc'] / grand_total:7.1f}% "
          f"{100 * total['alloc'] / grand_total:6.1f}% "
          f"{100 * unaccounted(total) / grand_total:7.1f}%")


def print_top_nodes(nodes, node_names, node_op, top):
    print(f"\ntop {top} nodes by cycles")
    print(f"{'idx':>4s} {'op':12s} {'cycles':>10s} {'cmp':>8s} {'in':>7s} {'out':>7s} "
          f"{'prep':>9s} {'snc':>7s}  node")

    ranked = sorted(nodes, key=lambda node: nodes[node]["node_cycles"], reverse=True)
    for node in ranked[:top]:
        e = nodes[node]
        print(f"{node:4d} {node_op.get(node, ''):12s} {e['node_cycles']:10d} {e['cmp']:8d} "
              f"{e['in']:7d} {e['out']:7d} {e['prep']:9d} {e['snc']:7d}  "
              f"{node_names.get(node, '')[:46]}")


def write_node_csv(path, samples, node_names, node_op, node_layer, layer_names):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node_idx", "node", "op", "layer_idx", "layer", "hid"] + list(FIELDS))

        for node in sorted(samples):
            layer = node_layer.get(node, 0)
            for tile in sorted(samples[node]):
                writer.writerow([node, node_names.get(node, ""), node_op.get(node, ""), layer,
                                 layer_names.get(layer, ""), tile]
                                + [samples[node][tile][field] for field in FIELDS])

    print(f"\nper-node per-tile CSV -> {path}")


def main():
    args = parse_args()

    samples, layer_names, node_names, node_layer, node_op = parse_log(args.log)
    nodes = collapse_nodes(samples, args.agg)

    tiles = sorted({tile for per_tile in samples.values() for tile in per_tile})
    grand_total = sum(entry["node_cycles"] for entry in nodes.values())
    print(f"nodes={len(nodes)} layers={len(layer_names) or '?'} tiles={tiles} "
          f"aggregation={args.agg}")

    def layer_of(node):
        idx = node_layer.get(node, 0)
        return f"[{idx}] {layer_names.get(idx, '?')}"

    print_group_table("cycles per layer", "layer", *group_nodes(nodes, layer_of), grand_total)
    print_top_nodes(nodes, node_names, node_op, args.top)
    print_group_table("cycles per operator", "operator", *group_nodes(nodes, lambda n: node_op.get(n, "?")), grand_total)

    if args.csv:
        write_node_csv(args.csv, samples, node_names, node_op, node_layer, layer_names)


if __name__ == "__main__":
    main()
