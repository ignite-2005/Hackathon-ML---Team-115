"""Shared geometry helpers. Every team member imports from here so metres match.

The trips are spread across the world (Russia, USA, Japan, Brazil, ...), so there is
no single national grid that works for all of them. Each trip gets its own local
azimuthal-equidistant projection centred on the middle of its road network, which
keeps distances accurate (well under 1% error) within the trip's area.
Always use the same origin for the track and the nodes of one trip: trip_origin(d).
"""
from functools import lru_cache
import numpy as np
from pyproj import Transformer


def trip_origin(d):
    """Projection centre for a trip: mean lon/lat of its road-network nodes.

    Uses only the nodes, so it is available for evaluator trips at inference time.
    """
    return (round(float(d["nodes"]["lon"].mean()), 6),
            round(float(d["nodes"]["lat"].mean()), 6))


@lru_cache(maxsize=256)
def _transformer(lon0, lat0):
    return Transformer.from_crs(
        "EPSG:4326",
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m",
        always_xy=True)


def to_metres(lon, lat, origin):
    """lon/lat arrays (degrees) -> x/y arrays (metres) in the trip's local projection.

    origin = trip_origin(d) for the trip these coordinates belong to.
    """
    x, y = _transformer(*origin).transform(np.asarray(lon, dtype=float),
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
