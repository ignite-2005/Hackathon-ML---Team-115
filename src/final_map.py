"""Final Folium map for one trip: GPS, road network, reference route, predicted route.

Prediction CSV format (agree this with Person C):
    obs_idx, pred_arc_id                       (one row per GPS point)
    optional extra column: clf_arc_id          (classifier-only prediction, to compare)

Usage:
    python final_map.py "<Student folder>" <trip_id> <predictions.csv> <out.html>
"""
import os
import sys
import numpy as np
import pandas as pd
import folium

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_data import load_trip


def arc_lines(arcs, nodes, arc_ids):
    """Consecutive duplicate arcs collapsed; one [lat, lon] pair per arc."""
    lon, lat = nodes["lon"].values, nodes["lat"].values
    a = arcs["from_node"].values
    b = arcs["to_node"].values
    ids = pd.Series(arc_ids).dropna().astype(int)
    ids = ids[ids.ne(ids.shift())].values  # drop repeats in a row
    return ids, [[[lat[a[i]], lon[a[i]]], [lat[b[i]], lon[b[i]]]] for i in ids]


def add_arc_layer(m, name, lines, color, weight, dash=None, show=True):
    g = folium.FeatureGroup(name=name, show=show)
    for seg in lines:
        folium.PolyLine(seg, color=color, weight=weight, opacity=0.85,
                        dash_array=dash).add_to(g)
    g.add_to(m)


def build_map(d, pred, out_path):
    tr, nodes, arcs, route = d["track"], d["nodes"], d["arcs"], d["route"]
    lon, lat = nodes["lon"].values, nodes["lat"].values
    m = folium.Map(location=[tr["lat"].mean(), tr["lon"].mean()], zoom_start=14,
                   tiles="OpenStreetMap")

    # Road network near the trip, as one light layer
    pad = 0.003
    keep = ((lon > tr["lon"].min() - pad) & (lon < tr["lon"].max() + pad) &
            (lat > tr["lat"].min() - pad) & (lat < tr["lat"].max() + pad))
    roads = [[[lon[a], lat[a]], [lon[b], lat[b]]]
             for a, b in zip(arcs["from_node"], arcs["to_node"]) if keep[a] and keep[b]]
    g = folium.FeatureGroup(name="Road network (nearby)")
    folium.GeoJson({"type": "Feature", "properties": {},
                    "geometry": {"type": "MultiLineString", "coordinates": roads}},
                   style_function=lambda _: {"color": "#9e9e9e", "weight": 1}).add_to(g)
    g.add_to(m)

    # Reference route (green, wide, underneath)
    _, ref_lines = arc_lines(arcs, nodes, route["arc_id"])
    add_arc_layer(m, "Reference route", ref_lines, "green", 8)

    # Classifier-only prediction (orange dashed), if provided
    if "clf_arc_id" in pred.columns:
        _, clf_lines = arc_lines(arcs, nodes, pred.sort_values("obs_idx")["clf_arc_id"])
        add_arc_layer(m, "Classifier-only prediction", clf_lines, "orange", 3,
                      dash="6 6", show=False)

    # Final predicted route (red, on top of the reference)
    _, pred_lines = arc_lines(arcs, nodes, pred.sort_values("obs_idx")["pred_arc_id"])
    add_arc_layer(m, "Predicted route (HMM)", pred_lines, "red", 3)

    # GPS points (small blue dots)
    gps = folium.FeatureGroup(name="GPS observations")
    step = max(1, len(tr) // 3000)  # thin very long trips so the page stays fast
    for _, p in tr.iloc[::step].iterrows():
        folium.CircleMarker([p["lat"], p["lon"]], radius=2, color="blue", fill=True,
                            fill_opacity=0.7, weight=0,
                            tooltip=f"t={p['t']:.0f}s").add_to(gps)
    gps.add_to(m)

    folium.Marker([tr["lat"].iloc[0], tr["lon"].iloc[0]], tooltip="Start",
                  icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([tr["lat"].iloc[-1], tr["lon"].iloc[-1]], tooltip="End",
                  icon=folium.Icon(color="red")).add_to(m)

    extra = ('<span style="color:orange">&#9476;</span> Classifier-only prediction<br>'
             if "clf_arc_id" in pred.columns else "")
    legend = f"""<div style="position:fixed;bottom:30px;left:30px;z-index:9999;
      background:white;padding:8px 12px;border:1px solid #888;font-size:13px">
      <b>Trip {d['trip_id']}</b><br>
      <span style="color:blue">&#9679;</span> GPS observations<br>
      <span style="color:green">&#9644;</span> Reference route<br>
      <span style="color:red">&#9644;</span> Predicted route (HMM)<br>
      {extra}<span style="color:#9e9e9e">&#9644;</span> Road network</div>"""
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[tr["lat"].min(), tr["lon"].min()], [tr["lat"].max(), tr["lon"].max()]])
    m.save(out_path)


if __name__ == "__main__":
    root, trip_id, pred_csv, out = sys.argv[1:5]
    part = "train" if os.path.isdir(os.path.join(root, "train", trip_id)) else "student_test"
    d = load_trip(os.path.join(root, part, trip_id))
    pred = pd.read_csv(pred_csv)
    if "trip_id" in pred.columns:
        pred = pred[pred["trip_id"].astype(str).str.zfill(8) == trip_id]
    if len(pred) == 0:
        sys.exit(f"No predictions for trip {trip_id} in {pred_csv}")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    build_map(d, pred, out)
    print("saved", out)
