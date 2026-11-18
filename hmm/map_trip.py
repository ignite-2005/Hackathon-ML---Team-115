"""Folium map of one trip + a breakdown of where Viterbi disagrees with the labels.

Layers (each can be switched on/off): nearby roads, GPS points, reference route,
nearest-arc route, Viterbi route, and the GPS points Viterbi got wrong.
Labels / .route are used only to draw and score, never to decode.

Usage: python hmm/map_trip.py "<Student folder>" student_test/00000077
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import folium

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)
from load_data import load_trip
from candidates import align_route
from transitions import PARAMS_PATH, true_transitions
from viterbi import decode
from metrics import route_from_points


def lines_layer(name, arcs, d, color, weight, opacity=0.8):
    """One GeoJSON layer with a line per arc (much lighter than one PolyLine per arc)."""
    lon, lat = d["nodes"]["lon"].values, d["nodes"]["lat"].values
    f, t = d["arcs"]["from_node"].values, d["arcs"]["to_node"].values
    coords = [[[lon[f[a]], lat[f[a]]], [lon[t[a]], lat[t[a]]]] for a in arcs]
    layer = folium.FeatureGroup(name=name)
    folium.GeoJson({"type": "Feature", "properties": {},
                    "geometry": {"type": "MultiLineString", "coordinates": coords}},
                   style_function=lambda _: {"color": color, "weight": weight,
                                             "opacity": opacity}).add_to(layer)
    return layer


def error_breakdown(d, res, truth):
    """Classify every point where Viterbi's arc differs from the label."""
    arcs = d["arcs"]
    f, t = arcs["from_node"].values, arcs["to_node"].values
    ref = set(d["route"]["arc_id"])
    hmm = res["viterbi"]
    wrong = np.flatnonzero(hmm != truth)
    kinds = []
    for i in wrong:
        p, q = hmm[i], truth[i]
        if p < 0:
            kinds.append("unmatched")
        elif f[p] == t[q] and t[p] == f[q]:
            kinds.append("reverse direction of true arc")
        elif {f[p], t[p]} & {f[q], t[q]}:
            kinds.append("neighbouring arc (shares a node)" + (", in .route" if p in ref else ""))
        elif p in ref:
            kinds.append("other arc that IS in .route")
        else:
            kinds.append("other arc, not in .route")
    kinds = pd.Series(kinds, dtype=object)

    # runs of consecutive wrong points
    runs, start = [], None
    is_wrong = hmm != truth
    for i, w in enumerate(np.r_[is_wrong, False]):
        if w and start is None:
            start = i
        elif not w and start is not None:
            runs.append((start, i - start))
            start = None
    return wrong, kinds, sorted(runs, key=lambda r: -r[1])


if __name__ == "__main__":
    root, tp = sys.argv[1], sys.argv[2]
    with open(PARAMS_PATH) as fh:
        p = json.load(fh)
    d = load_trip(os.path.join(root, *tp.split("/")))
    res = decode(d, p["sigma_m"], p["beta_m"])
    labels = align_route(d)
    truth = labels["true_arc"].to_numpy()
    tr = d["track"]

    # ---- diagnosis ----
    wrong, kinds, runs = error_breakdown(d, res, truth)
    print(f"trip {d['trip_id']}: {len(tr):,} points, Viterbi wrong at {len(wrong):,} "
          f"({100 * len(wrong) / len(tr):.1f}%)")
    print("\nwhat Viterbi chose instead of the labelled arc:")
    print(kinds.value_counts().to_string())
    print(f"\nwrong points form {len(runs)} runs; longest (start obs_idx, length): {runs[:6]}")
    print(f"label quality: median GPS-to-true-arc {labels['true_dist_m'].median():.1f} m, "
          f"p95 {labels['true_dist_m'].quantile(0.95):.1f} m")
    r = d["route"]["arc_id"].values
    a_f, a_t = d["arcs"]["from_node"].values, d["arcs"]["to_node"].values
    print(f".route: {len(r)} arcs, breaks (to[a] != from[b]): {(a_t[r[:-1]] != a_f[r[1:]]).sum()}, "
          f"arcs used more than once: {int((pd.Series(r).value_counts() > 1).sum())}")
    tt = true_transitions(d, labels)
    bad = tt[~np.isfinite(tt["gap"])]
    print(f"impossible TRUE transitions: {len(bad)}  {bad['reason'].value_counts().to_dict()}")
    if len(bad):
        print(bad[["obs_idx", "dt", "d_gc", "arc_a", "arc_b", "oa", "ob", "len_a"]]
              .round(1).head(8).to_string(index=False))
    vr = set(route_from_points(res["viterbi"]).tolist())
    extra = vr - set(r.tolist())
    print(f"Viterbi route arcs not in .route: {len(extra)}; of these, reverse of a .route arc: "
          f"{sum(1 for a in extra if any((a_f[a] == a_t[b] and a_t[a] == a_f[b]) for b in r))}")

    # ---- map ----
    m = folium.Map(location=[tr["lat"].mean(), tr["lon"].mean()], zoom_start=14,
                   tiles="OpenStreetMap")
    pad = 0.003
    lon, lat = d["nodes"]["lon"].values, d["nodes"]["lat"].values
    near = np.flatnonzero(
        (lon[a_f] > tr["lon"].min() - pad) & (lon[a_f] < tr["lon"].max() + pad) &
        (lat[a_f] > tr["lat"].min() - pad) & (lat[a_f] < tr["lat"].max() + pad))
    lines_layer("Road network (nearby)", near, d, "#999999", 1, 0.6).add_to(m)
    lines_layer("Reference route (.route)", r, d, "green", 9, 0.5).add_to(m)
    lines_layer("Nearest-arc route", route_from_points(res["nearest"]), d, "orange", 4).add_to(m)
    lines_layer("Viterbi route", route_from_points(res["viterbi"]), d, "purple", 3).add_to(m)

    gps = folium.FeatureGroup(name="GPS points")
    for i, (la, lo, t) in enumerate(zip(tr["lat"], tr["lon"], tr["t"])):
        folium.CircleMarker([la, lo], radius=2, color="blue", fill=True, fill_opacity=0.8,
                            tooltip=f"obs {i}, t={t:.0f}s").add_to(gps)
    gps.add_to(m)
    bad_pts = folium.FeatureGroup(name="Viterbi wrong points", show=False)
    for i, k in zip(wrong, kinds):
        folium.CircleMarker([tr["lat"].iloc[i], tr["lon"].iloc[i]], radius=4, color="red",
                            fill=True, tooltip=f"obs {i}: {k}").add_to(bad_pts)
    bad_pts.add_to(m)

    folium.Marker([tr["lat"].iloc[0], tr["lon"].iloc[0]], tooltip="Start",
                  icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([tr["lat"].iloc[-1], tr["lon"].iloc[-1]], tooltip="End",
                  icon=folium.Icon(color="red")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[tr["lat"].min(), tr["lon"].min()], [tr["lat"].max(), tr["lon"].max()]])
    out = os.path.join(REPO, "outputs", "hmm", f"map_{d['trip_id']}.html")
    m.save(out)
    print(f"\nsaved {os.path.relpath(out, REPO)}")
