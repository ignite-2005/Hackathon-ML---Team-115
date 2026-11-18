"""Step 2: EDA. Per-trip stats, network/route checks, plots and a Folium map.

Usage: python eda.py "<Student folder>" <out_dir> [trip_id_for_map]
"""
import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folium

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_data import load_trip
from geo import to_metres, trip_origin


def trip_stats(d, partition):
    tr, nodes, arcs, route = d["track"], d["nodes"], d["arcs"], d["route"]
    origin = trip_origin(d)

    # GPS movement in metres
    x, y = to_metres(tr["lon"], tr["lat"], origin)
    step_m = np.hypot(np.diff(x), np.diff(y))
    dt = np.diff(tr["t"].values)
    speed = np.where(dt > 0, step_m / np.where(dt > 0, dt, 1), np.nan)

    # Road network
    g = nx.DiGraph()
    g.add_nodes_from(range(len(nodes)))
    g.add_edges_from(zip(arcs["from_node"], arcs["to_node"]))
    comps = list(nx.weakly_connected_components(g))
    edge_set = set(zip(arcs["from_node"], arcs["to_node"]))
    pct_two_way = np.mean([(b, a) in edge_set for a, b in edge_set]) * 100

    # Reference route: does each arc end where the next one starts?
    r = route["arc_id"].values
    ends = arcs["to_node"].values[r[:-1]]
    starts = arcs["from_node"].values[r[1:]]
    breaks = int((ends != starts).sum())
    repeats = int((r[:-1] == r[1:]).sum())

    return {
        "partition": partition,
        "trip_id": d["trip_id"],
        "proj_origin": origin,
        "n_gps": len(tr),
        "gps_length_km": round(step_m.sum() / 1000, 2),
        "median_speed_ms": round(np.nanmedian(speed), 2),
        "n_speed_over_50ms": int((speed > 50).sum()),
        "n_gaps_over_10s": int((dt > 10).sum()),
        "lon_min": tr["lon"].min(), "lon_max": tr["lon"].max(),
        "lat_min": tr["lat"].min(), "lat_max": tr["lat"].max(),
        "n_nodes": len(nodes),
        "n_arcs": len(arcs),
        "n_isolated_nodes": nx.number_of_isolates(g),
        "n_components": len(comps),
        "largest_comp_pct": round(max(len(c) for c in comps) / len(nodes) * 100, 1),
        "pct_two_way_arcs": round(pct_two_way, 1),
        "n_route_arcs": len(r),
        "route_breaks": breaks,
        "route_repeats": repeats,
    }, dt, speed


def plot_hist(values, title, xlabel, path, bins=50, log=True):
    plt.figure(figsize=(7, 4))
    plt.hist(values, bins=bins, log=log)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count (log)" if log else "count")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_trip(d, path):
    """GPS points vs reference route in metres for one trip."""
    tr, nodes, arcs, route = d["track"], d["nodes"], d["arcs"], d["route"]
    origin = trip_origin(d)
    nx_, ny_ = to_metres(nodes["lon"], nodes["lat"], origin)
    gx, gy = to_metres(tr["lon"], tr["lat"], origin)
    plt.figure(figsize=(7, 7))
    # Only the road arcs near the trip, so the plot stays readable
    pad = 300
    keep = ((nx_ > gx.min() - pad) & (nx_ < gx.max() + pad) &
            (ny_ > gy.min() - pad) & (ny_ < gy.max() + pad))
    for a, b in zip(arcs["from_node"], arcs["to_node"]):
        if keep[a] and keep[b]:
            plt.plot([nx_[a], nx_[b]], [ny_[a], ny_[b]], color="lightgrey", lw=0.6)
    for k, arc in enumerate(route["arc_id"]):
        a, b = arcs.loc[arc, ["from_node", "to_node"]]
        plt.plot([nx_[a], nx_[b]], [ny_[a], ny_[b]], color="green", lw=2.5,
                 label="reference route" if k == 0 else None)
    plt.scatter(gx, gy, s=3, color="blue", label="GPS points")
    plt.gca().set_aspect("equal")
    plt.title(f"Trip {d['trip_id']} (local metres)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def folium_map(d, path):
    """Interactive map: GPS points, nearby roads, reference route."""
    tr, nodes, arcs, route = d["track"], d["nodes"], d["arcs"], d["route"]
    m = folium.Map(location=[tr["lat"].mean(), tr["lon"].mean()], zoom_start=14,
                   tiles="OpenStreetMap")

    # Roads near the trip, drawn as one GeoJSON layer (much lighter than one line per arc)
    pad = 0.003
    lon, lat = nodes["lon"].values, nodes["lat"].values
    keep = ((lon > tr["lon"].min() - pad) & (lon < tr["lon"].max() + pad) &
            (lat > tr["lat"].min() - pad) & (lat < tr["lat"].max() + pad))
    lines = [[[lon[a], lat[a]], [lon[b], lat[b]]]
             for a, b in zip(arcs["from_node"], arcs["to_node"]) if keep[a] and keep[b]]
    roads = folium.FeatureGroup(name="Road network (nearby)")
    folium.GeoJson({"type": "Feature", "properties": {},
                    "geometry": {"type": "MultiLineString", "coordinates": lines}},
                   style_function=lambda _: {"color": "#999999", "weight": 1}).add_to(roads)
    roads.add_to(m)

    # Reference route
    ref = folium.FeatureGroup(name="Reference route")
    for arc in route["arc_id"]:
        a, b = arcs.loc[arc, ["from_node", "to_node"]]
        folium.PolyLine([[lat[a], lon[a]], [lat[b], lon[b]]], color="green",
                        weight=5, opacity=0.8).add_to(ref)
    ref.add_to(m)

    # GPS points
    gps = folium.FeatureGroup(name="GPS observations")
    for _, p in tr.iterrows():
        folium.CircleMarker([p["lat"], p["lon"]], radius=2, color="blue",
                            fill=True, fill_opacity=0.8,
                            tooltip=f"t={p['t']:.0f}s").add_to(gps)
    gps.add_to(m)

    # Start / end markers
    folium.Marker([tr["lat"].iloc[0], tr["lon"].iloc[0]], tooltip="Start",
                  icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([tr["lat"].iloc[-1], tr["lon"].iloc[-1]], tooltip="End",
                  icon=folium.Icon(color="red")).add_to(m)

    legend = """<div style="position:fixed;bottom:30px;left:30px;z-index:9999;
      background:white;padding:8px 12px;border:1px solid #888;font-size:13px">
      <b>Trip {tid}</b><br>
      <span style="color:blue">&#9679;</span> GPS observations<br>
      <span style="color:green">&#9644;</span> Reference route<br>
      <span style="color:#999">&#9644;</span> Road network</div>""".format(tid=d["trip_id"])
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[tr["lat"].min(), tr["lon"].min()], [tr["lat"].max(), tr["lon"].max()]])
    m.save(path)


if __name__ == "__main__":
    root, out = sys.argv[1], sys.argv[2]
    map_trip = sys.argv[3] if len(sys.argv) > 3 else None
    os.makedirs(out, exist_ok=True)

    rows, all_dt, all_speed = [], [], []
    for part in ["train", "student_test"]:
        pdir = os.path.join(root, part)
        for tid in sorted(os.listdir(pdir)):
            d = load_trip(os.path.join(pdir, tid))
            s, dt, sp = trip_stats(d, part)
            rows.append(s)
            if part == "train":  # distributions use the training partition only
                all_dt.append(dt)
                all_speed.append(sp)
    stats = pd.DataFrame(rows)
    stats.to_csv(os.path.join(out, "trip_stats.csv"), index=False)

    dt = np.concatenate(all_dt)
    sp = np.concatenate(all_speed)
    plot_hist(np.clip(dt, 0, 60), "Time between GPS points (train, clipped at 60 s)",
              "seconds", os.path.join(out, "gap_hist.png"), bins=60)
    plot_hist(np.clip(sp[~np.isnan(sp)], 0, 80), "Speed between GPS points (train, clipped at 80 m/s)",
              "m/s", os.path.join(out, "speed_hist.png"), bins=80)
    tr_stats = stats[stats.partition == "train"]
    plot_hist(tr_stats["n_gps"], "GPS points per trip (train)", "points",
              os.path.join(out, "gps_per_trip.png"), bins=30, log=False)
    plot_hist(tr_stats["gps_length_km"], "Trip length (train)", "km",
              os.path.join(out, "trip_length.png"), bins=30, log=False)

    # Where the trips are
    plt.figure(figsize=(6, 6))
    for _, r in stats.iterrows():
        c = "tab:blue" if r.partition == "train" else "tab:orange"
        plt.plot([r.lon_min, r.lon_max, r.lon_max, r.lon_min, r.lon_min],
                 [r.lat_min, r.lat_min, r.lat_max, r.lat_max, r.lat_min], color=c, lw=0.8)
    plt.title("Bounding box of each trip (blue=train, orange=student_test)")
    plt.xlabel("longitude"); plt.ylabel("latitude")
    plt.tight_layout(); plt.savefig(os.path.join(out, "trip_extents.png"), dpi=120); plt.close()

    if map_trip:
        part = "train" if os.path.isdir(os.path.join(root, "train", map_trip)) else "student_test"
        d = load_trip(os.path.join(root, part, map_trip))
        plot_trip(d, os.path.join(out, f"trip_{map_trip}.png"))
        folium_map(d, os.path.join(out, f"map_{map_trip}.html"))

    pd.set_option("display.width", 200)
    cols = ["gps_length_km", "median_speed_ms", "n_speed_over_50ms", "n_gaps_over_10s",
            "n_components", "largest_comp_pct", "n_isolated_nodes", "pct_two_way_arcs",
            "route_breaks", "route_repeats"]
    print(stats.groupby("partition")[cols].median().T)
    print("\nTrips with route breaks:", (stats.route_breaks > 0).sum(), "of", len(stats))
    print("Trips with GPS jumps > 50 m/s:", (stats.n_speed_over_50ms > 0).sum())
