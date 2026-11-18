"""Metrics for a predicted arc per GPS point (labels / .route used for scoring only).

route_from_points : per-point arcs -> route (drop unmatched -1, collapse repeats)
continuity        : share of consecutive route arcs that really connect (to[a] == from[b])
route_prf         : precision / recall / F1 of the route's arcs vs the reference .route arcs
                    (as sets of arc_ids: an arc counts once however often it is driven)
The route here is NOT gap-filled with shortest paths yet, so continuity shows the raw
quality of the decoder's output.
"""
import numpy as np


def route_from_points(point_arcs):
    """[5, 5, 5, -1, 5, 7, 7] -> [5, 7]."""
    a = np.asarray(point_arcs)
    a = a[a >= 0]
    if len(a) == 0:
        return a
    keep = np.r_[True, a[1:] != a[:-1]]
    return a[keep]


def continuity(route, arc_from, arc_to):
    """(connected pairs, total pairs) for consecutive arcs of the route."""
    if len(route) < 2:
        return 0, 0
    ok = arc_to[route[:-1]] == arc_from[route[1:]]
    return int(ok.sum()), len(ok)


def route_prf(route, ref_route):
    """(true positives, predicted arcs, reference arcs) as unique arc sets."""
    pred, ref = set(np.asarray(route).tolist()), set(np.asarray(ref_route).tolist())
    return len(pred & ref), len(pred), len(ref)


def stopped_mask(x, y, window=10, net_m=20.0):
    """True where the car is (almost) stopped: the GPS moves less than net_m metres net
    between point t-window and t+window. Used to report Top-1 for stopped / moving points
    separately (during stops the GPS gives no direction, so labels are ambiguous too)."""
    n = len(x)
    lo = np.clip(np.arange(n) - window, 0, n - 1)
    hi = np.clip(np.arange(n) + window, 0, n - 1)
    return np.hypot(x[hi] - x[lo], y[hi] - y[lo]) < net_m


def prf(tp, n_pred, n_ref):
    """precision, recall, F1 from counts (0 when undefined)."""
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_ref if n_ref else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)
