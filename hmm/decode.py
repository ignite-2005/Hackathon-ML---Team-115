"""Part A, step 8: hand-off to Person D (API). One call decodes one trip.

    from decode import decode_trip
    out = decode_trip(trip_folder)                       # Gaussian emissions (no classifier)
    out = decode_trip(trip_folder, model, make_features) # classifier emissions (later)

Reads ONLY the .track, .nodes and .arcs files: the reference .route is never opened, so
labels cannot leak into a prediction, and it works on evaluator trips that have no .route.
All settings come from hmm/hmm_config.json, so the API runs exactly the tested setup.

Returns a JSON-ready dict:
  trip_id      : folder name
  point_arcs   : arc_id per GPS point, in track order (-1 = unmatched: no road within radius_m)
  route        : arc_ids the car drove, in order (repeats collapsed, loops <= max_loop_m
                 cut, see route.py; jumps NOT filled yet)
  route_lonlat : [[lon_from, lat_from], [lon_to, lat_to]] for each route arc (for drawing)
  stats        : points, unmatched, chains, chain-break reasons, seconds
  config       : the settings used

Usage (quick check): python hmm/decode.py "<trip folder>" [out.json]
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)
from candidates import generate_candidates
from graph import build_graph
from states import build_states
from emissions import gaussian_log_emissions, classifier_log_emissions
from edge_cases import decode_points
from route import build_route

CONFIG_PATH = os.path.join(HERE, "hmm_config.json")


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return json.load(f)


def load_trip_no_labels(folder):
    """Like src/load_data.load_trip, but without the .route file (inference only)."""
    trip_id = os.path.basename(os.path.normpath(folder))
    read = lambda ext, cols: pd.read_csv(os.path.join(folder, f"{trip_id}.{ext}"),
                                         sep=r"\s+", header=None, names=cols)
    nodes = read("nodes", ["lon", "lat"])
    arcs = read("arcs", ["from_node", "to_node"])
    nodes.index.name, arcs.index.name = "node_id", "arc_id"
    return {"trip_id": trip_id, "track": read("track", ["lon", "lat", "t"]),
            "nodes": nodes, "arcs": arcs}


def decode_trip(trip_folder, model=None, make_features=None, config=None):
    """Arc per GPS point + driven route for one trip folder.

    model         : Person B's classifier (anything with predict_proba), or None for the
                    Gaussian baseline.
    make_features : function(d, cands) -> feature table, one row per candidate row in the
                    same order. Needed when model is given (Person B provides it).
    """
    t0 = time.perf_counter()
    cfg = config or load_config()
    d = load_trip_no_labels(trip_folder)
    cands = generate_candidates(d, cfg["radius_m"], cfg["max_k"])
    g = build_graph(d)

    if model is None:
        emission = "gaussian"
        log_e = None                         # decode_points builds Gaussian emissions
    else:
        if make_features is None:
            raise ValueError("pass make_features(d, cands) together with the model")
        emission = "classifier"
        proba = model.predict_proba(make_features(d, cands))[:, 1]
        log_e = classifier_log_emissions(build_states(d, cands), proba)

    res = decode_points(d, cfg["sigma_m"], cfg["beta_m"], w=cfg["w"],
                        gap_s=cfg["gap_threshold_s"], skip_k=cfg["skip_k"],
                        fill_s=cfg["fill_s"], cands=cands, log_e=log_e, g=g,
                        trans_kw={"neg_tol": cfg["neg_tol_m"],
                                  "max_speed": cfg["max_speed_ms"], "slack": cfg["slack_m"]})

    route = build_route(res["arcs"], g["arc_len"], cfg["max_loop_m"])
    lon, lat = d["nodes"]["lon"].to_numpy(), d["nodes"]["lat"].to_numpy()
    f, t = g["arc_from"][route], g["arc_to"][route]
    stats = {k: v for k, v in res["stats"].items() if k != "secs"}
    stats["secs"] = round(time.perf_counter() - t0, 2)
    return {
        "trip_id": d["trip_id"],
        "point_arcs": [int(a) for a in res["arcs"]],
        "route": [int(a) for a in route],
        "route_lonlat": [[[float(lon[a]), float(lat[a])], [float(lon[b]), float(lat[b])]]
                         for a, b in zip(f, t)],
        "stats": {k: (int(v) if isinstance(v, (np.integer, int)) else v)
                  for k, v in stats.items()},
        "config": {**cfg, "emission": emission},
    }


if __name__ == "__main__":
    out = decode_trip(sys.argv[1])
    print(f"trip {out['trip_id']}: {len(out['point_arcs']):,} points -> "
          f"{len(out['route'])} route arcs; stats {out['stats']}")
    print(f"first 10 point arcs: {out['point_arcs'][:10]}")
    print(f"first 10 route arcs: {out['route'][:10]}")
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w") as f:
            json.dump(out, f)
        print(f"saved {sys.argv[2]}")
