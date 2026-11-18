"""Shared geometry helpers. Every team member imports from here so metres match.

The trips are spread across the world, so one fixed projection doesn't work.
Each trip is projected to the UTM zone that contains it (chosen from the median
GPS position). Within one trip, ALWAYS use the same epsg for GPS and road nodes.
"""
from functools import lru_cache
import numpy as np
from pyproj import Transformer


def utm_epsg(lon, lat):
    """EPSG code of the UTM zone containing (lon, lat). 326xx north, 327xx south."""
    zone = int((lon + 180) // 6) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


def trip_epsg(d):
    """UTM EPSG code for one trip, from the median GPS position."""
    return utm_epsg(float(d["track"]["lon"].median()), float(d["track"]["lat"].median()))


@lru_cache(maxsize=None)
def _transformer(epsg):
    return Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)


def to_metres(lon, lat, epsg):
    """lon/lat arrays (degrees) -> x/y arrays (metres) in the given UTM zone."""
    x, y = _transformer(epsg).transform(np.asarray(lon, dtype=float),
                                        np.asarray(lat, dtype=float))
    return np.asarray(x), np.asarray(y)


def project_onto_segments(px, py, ax, ay, bx, by):
    """Project points P onto segments A->B (all arrays, same length).

    Returns dist_m (point to closest point on segment), offset_m (distance from A
    along the segment to that closest point) and seg_len_m.
    """
    dx, dy = bx - ax, by - ay
    seg_len = np.hypot(dx, dy)
    len2 = np.where(seg_len > 0, seg_len ** 2, 1.0)
    # fraction along the segment, clamped to [0, 1] so we stay on the segment
    frac = np.clip(((px - ax) * dx + (py - ay) * dy) / len2, 0.0, 1.0)
    qx, qy = ax + frac * dx, ay + frac * dy
    dist = np.hypot(px - qx, py - qy)
    return dist, frac * seg_len, seg_len
