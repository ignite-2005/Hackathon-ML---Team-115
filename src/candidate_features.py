"""Build model features from GPS and directed arc geometry."""

import numpy as np
import pandas as pd

from src.candidates import arc_geometry, generate_candidates
from src.geo import to_metres, trip_origin

KEY = ["trip_id", "obs_idx"]
FEATURE_COLUMNS = [
    "dist_m", "heading_diff_deg", "speed_mps", "arc_len_m", "offset_fraction",
    "candidate_rank", "candidate_count", "distance_above_nearest_m",
]


def _track_motion(trip, max_gap_s):
    track = trip["track"]
    x, y = to_metres(track["lon"], track["lat"], trip_origin(trip))
    dt = np.diff(track["t"].to_numpy(dtype=float))
    dx, dy = np.diff(x), np.diff(y)
    valid = np.isfinite(dt) & np.isfinite(dx) & np.isfinite(dy)
    valid &= (dt > 0) & (dt <= max_gap_s)
    dt, dx, dy = (np.where(valid, values, 0.0) for values in (dt, dx, dy))

    def adjacent_sum(values):
        return np.r_[0.0, values] + np.r_[values, 0.0]

    elapsed = adjacent_sum(dt)
    speed = np.divide(
        adjacent_sum(np.hypot(dx, dy)), elapsed,
        out=np.full(len(track), np.nan), where=elapsed > 0,
    )
    vx, vy = adjacent_sum(dx), adjacent_sum(dy)
    heading = np.arctan2(vy, vx)
    heading[np.hypot(vx, vy) < 1e-6] = np.nan
    return speed, heading


def build_features(trip, candidates=None, *, radius_m=50.0, max_k=16, max_heading_gap_s=5.0):
    """Add inference-safe features to Person A's candidate rows for one trip."""
    if candidates is None:
        candidates = generate_candidates(trip, radius_m=radius_m, max_k=max_k)
    rows = candidates.sort_values(["obs_idx", "dist_m", "arc_id"]).reset_index(drop=True)
    if rows.empty:
        for name in FEATURE_COLUMNS:
            if name not in rows:
                rows[name] = pd.Series(dtype=float)
        return rows

    speed, heading = _track_motion(trip, max_heading_gap_s)
    obs = rows["obs_idx"].to_numpy(dtype=np.intp)
    arc = rows["arc_id"].to_numpy(dtype=np.intp)
    ax, ay, bx, by = arc_geometry(trip)
    arc_heading = np.arctan2(by[arc] - ay[arc], bx[arc] - ax[arc])
    angle = np.degrees(heading[obs] - arc_heading)
    lengths = rows["arc_len_m"].to_numpy(dtype=float)

    rows["speed_mps"] = speed[obs]
    rows["heading_diff_deg"] = np.where(
        lengths > 0, np.abs((angle + 180.0) % 360.0 - 180.0), np.nan
    )
    rows["offset_fraction"] = np.divide(
        rows["offset_m"].to_numpy(dtype=float), lengths,
        out=np.full(len(rows), np.nan), where=lengths > 0,
    )
    grouped = rows.groupby("obs_idx", sort=False)
    rows["candidate_rank"] = grouped.cumcount() + 1
    rows["candidate_count"] = grouped["arc_id"].transform("size")
    rows["distance_above_nearest_m"] = rows["dist_m"] - grouped["dist_m"].transform("min")
    return rows
