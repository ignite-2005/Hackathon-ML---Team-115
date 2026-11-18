"""Shared geometry helpers. Every team member imports from here so metres match."""
import numpy as np
from pyproj import Transformer

# WGS84 lon/lat -> British National Grid (EPSG:27700), metres. The data is in England.
_TO_BNG = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)


def to_metres(lon, lat):
    """lon/lat arrays (degrees) -> x/y arrays (metres, EPSG:27700)."""
    x, y = _TO_BNG.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
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
