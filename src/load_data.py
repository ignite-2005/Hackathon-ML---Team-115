"""Step 1: load PathSnap trips and build a dataset inventory."""
import os
import sys
import pandas as pd


def load_trip(folder):
    """Read the four files of one trip into DataFrames with named columns."""
    trip_id = os.path.basename(os.path.normpath(folder))
    path = lambda ext: os.path.join(folder, f"{trip_id}.{ext}")

    track = pd.read_csv(path("track"), sep=r"\s+", header=None,
                        names=["lon", "lat", "t"])
    nodes = pd.read_csv(path("nodes"), sep=r"\s+", header=None,
                        names=["lon", "lat"])
    arcs = pd.read_csv(path("arcs"), sep=r"\s+", header=None,
                       names=["from_node", "to_node"])
    route = pd.read_csv(path("route"), sep=r"\s+", header=None,
                        names=["arc_id"])

    # The row number is the ID (0-based), so keep it as a column.
    nodes.index.name = "node_id"
    arcs.index.name = "arc_id"
    return {"trip_id": trip_id, "track": track, "nodes": nodes,
            "arcs": arcs, "route": route}


def file_size_kb(folder, trip_id, ext):
    p = os.path.join(folder, f"{trip_id}.{ext}")
    return round(os.path.getsize(p) / 1024, 1) if os.path.exists(p) else None


def inventory(partition_dir, partition_name):
    """One row per trip: counts, sizes and basic sanity checks."""
    rows = []
    for trip_id in sorted(os.listdir(partition_dir)):
        folder = os.path.join(partition_dir, trip_id)
        if not os.path.isdir(folder):
            continue
        d = load_trip(folder)
        tr, ar, ro = d["track"], d["arcs"], d["route"]
        gaps = tr["t"].diff().dropna()
        rows.append({
            "partition": partition_name,
            "trip_id": trip_id,
            "n_gps": len(tr),
            "n_nodes": len(d["nodes"]),
            "n_arcs": len(ar),
            "n_route_arcs": len(ro),
            "track_kb": file_size_kb(folder, trip_id, "track"),
            "arcs_kb": file_size_kb(folder, trip_id, "arcs"),
            "missing_gps": int(tr.isna().any(axis=1).sum()),
            "timestamps_sorted": bool(tr["t"].is_monotonic_increasing),
            "duration_s": tr["t"].iloc[-1] - tr["t"].iloc[0],
            "pct_1s_steps": round((gaps == 1).mean() * 100, 1),
            "max_gap_s": gaps.max(),
            "route_ids_valid": bool(ro["arc_id"].between(0, len(ar) - 1).all()),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    root = sys.argv[1]  # path to "Student folder"
    out = sys.argv[2] if len(sys.argv) > 2 else "inventory.csv"
    inv = pd.concat([inventory(os.path.join(root, "train"), "train"),
                     inventory(os.path.join(root, "student_test"), "student_test")])
    inv.to_csv(out, index=False)
    print(inv.groupby("partition").agg(
        trips=("trip_id", "count"),
        gps_total=("n_gps", "sum"),
        gps_median=("n_gps", "median"),
        arcs_median=("n_arcs", "median"),
        missing_gps=("missing_gps", "sum"),
        unsorted_trips=("timestamps_sorted", lambda s: (~s).sum()),
        bad_route_ids=("route_ids_valid", lambda s: (~s).sum()),
    ))
    print("\nTrips with the largest time gaps:")
    print(inv.sort_values("max_gap_s", ascending=False)
             [["partition", "trip_id", "n_gps", "max_gap_s", "pct_1s_steps"]].head(8)
             .to_string(index=False))
