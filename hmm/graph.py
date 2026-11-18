"""Part A, step 1: turn a trip's road network into a weighted directed graph.

build_graph(d) gives the decoder everything it needs about the roads:
  arc_from, arc_to : start / end node of every arc (indexed by arc_id)
  arc_len          : length of every arc in metres (indexed by arc_id)
  graph            : n_nodes x n_nodes sparse matrix, graph[u, v] = metres from u to v
The graph is what scipy's dijkstra runs on to find shortest paths between arcs.

Usage (quick check): python hmm/graph.py "<Student folder>" [trip_id ...]
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# Reuse the shared helpers in src/ so lengths are in the same metres as everyone else.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from load_data import load_trip
from candidates import arc_geometry

MIN_LEN_M = 1e-3  # scipy treats a 0 in a sparse matrix as "no edge", so keep lengths > 0


def build_graph(d):
    """Arc arrays + a CSR node graph for one trip (d comes from load_trip)."""
    arc_from = d["arcs"]["from_node"].to_numpy()
    arc_to = d["arcs"]["to_node"].to_numpy()

    # Arc length = straight-line distance between its two end nodes, in metres.
    ax, ay, bx, by = arc_geometry(d)
    arc_len = np.maximum(np.hypot(bx - ax, by - ay), MIN_LEN_M)

    # Edges for the graph: drop self-loops (u -> u never helps a shortest path).
    keep = arc_from != arc_to
    edges = pd.DataFrame({"u": arc_from[keep], "v": arc_to[keep], "w": arc_len[keep]})
    # Same (u, v) pair listed twice: keep the shorter one.
    # (csr_matrix would otherwise ADD the two lengths together, which is wrong.)
    edges = edges.groupby(["u", "v"], as_index=False)["w"].min()

    n_nodes = len(d["nodes"])
    graph = csr_matrix((edges["w"].to_numpy(), (edges["u"].to_numpy(), edges["v"].to_numpy())),
                       shape=(n_nodes, n_nodes))

    return {
        "arc_from": arc_from,
        "arc_to": arc_to,
        "arc_len": arc_len,
        "graph": graph,
        "n_self_loops": int((~keep).sum()),
        "n_duplicates": int(keep.sum() - len(edges)),
    }


def describe(trip_folder):
    """Print size, arc-length stats and strong connectivity for one trip."""
    d = load_trip(trip_folder)
    t0 = time.perf_counter()
    g = build_graph(d)
    secs = time.perf_counter() - t0

    G = g["graph"]
    L = g["arc_len"]
    n_scc, comp = connected_components(G, directed=True, connection="strong")
    sizes = np.bincount(comp)
    big = sizes.argmax()

    # How much of the reference route sits inside the largest strongly connected piece?
    route = d["route"]["arc_id"].to_numpy()
    route_in_big = np.mean((comp[g["arc_from"][route]] == big) & (comp[g["arc_to"][route]] == big))

    print(f"\n=== trip {d['trip_id']} ===  (build_graph took {secs:.2f} s)")
    print(f"nodes / arcs              : {len(d['nodes']):,} / {len(d['arcs']):,}")
    print(f"graph shape / edges (nnz) : {G.shape} / {G.nnz:,}")
    print(f"self-loops dropped        : {g['n_self_loops']}")
    print(f"duplicate pairs merged    : {g['n_duplicates']}")
    print("arc length (m)            : "
          f"min {L.min():.3f}  median {np.median(L):.1f}  mean {L.mean():.1f}  "
          f"p95 {np.percentile(L, 95):.1f}  max {L.max():.1f}")
    print(f"arcs shorter than 1 m     : {(L < 1).sum()}")
    print(f"strongly connected comps  : {n_scc:,}")
    print(f"largest SCC               : {sizes[big]:,} nodes "
          f"({100 * sizes[big] / len(comp):.1f}% of all nodes)")
    print(f"single-node SCCs          : {(sizes == 1).sum():,}")
    print(f"route arcs in largest SCC : {100 * route_in_big:.1f}%")


if __name__ == "__main__":
    root = sys.argv[1]  # path to "Student folder"
    trips = sys.argv[2:] or ["00000046", "00000013"]
    for trip_id in trips:
        describe(os.path.join(root, "train", trip_id))
