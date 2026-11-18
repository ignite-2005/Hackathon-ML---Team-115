"""Route builder: per-point arcs -> the route the car drove.

1. Collapse: drop unmatched points (-1) and repeated consecutive arcs  [5,5,-1,5,7] -> [5,7]
2. Remove short loops: if the route comes back to an arc it already used, and the arcs
   driven in between add up to at most max_loop_m metres, the detour is cut out
   (the first visit is kept). Example: a -> reverse(a) -> a  becomes  a.

Why step 2: while the car is stopped the GPS jitters and Viterbi flips between the two
directions of a road, so collapsing alone gives a -> rev(a) -> a -> rev(a) ... On student_test
trip 77 that made the route 629 arcs long for a 268-arc reference.
Why 50 m: real routes do loop back (10 of 90 reference routes revisit an arc), but the
shortest real loop in any reference route is 76 m (trip 86), the next is 202 m, while the
median predicted loop is 23 m. So 50 m cuts jitter loops and never a real one seen in the data.

Not done here yet: filling jumps between non-adjacent arcs with shortest-path arcs.
"""
import numpy as np
from metrics import route_from_points

MAX_LOOP_M = 50.0


def remove_short_loops(route, arc_len, max_loop_m=MAX_LOOP_M):
    """Cut every detour of <= max_loop_m metres that returns to an arc already in the route."""
    out = []          # the cleaned route so far
    last = {}         # arc -> its last position in `out`
    for a in route:
        i = last.get(a)
        if i is not None and arc_len[out[i + 1:]].sum() <= max_loop_m:
            removed = out[i + 1:]
            del out[i + 1:]
            for b in set(removed):          # forget positions that were cut
                earlier = [k for k in range(len(out)) if out[k] == b]
                if earlier:
                    last[b] = earlier[-1]
                else:
                    del last[b]
            continue                        # back on arc a: nothing new to add
        out.append(a)
        last[a] = len(out) - 1
    return np.array(out, dtype=int)


def build_route(point_arcs, arc_len, max_loop_m=MAX_LOOP_M):
    """Per-point arcs -> collapsed route with short loops removed."""
    return remove_short_loops(route_from_points(point_arcs).tolist(), arc_len, max_loop_m)


if __name__ == "__main__":
    # tiny self-checks (arc lengths 10 m, except arc 9 = 100 m)
    L = np.full(10, 10.0)
    L[9] = 100.0
    assert build_route([1, 1, 2, 1, 2, 3], L).tolist() == [1, 2, 3]       # 1 2 1 2 -> 1 2
    assert build_route([1, 2, 3, 1, 4], L).tolist() == [1, 4]             # 20 m loop cut
    assert build_route([1, 9, 1, 4], L).tolist() == [1, 9, 1, 4]          # 100 m loop kept
    assert build_route([1, -1, 1, 2], L).tolist() == [1, 2]
    print("route.py self-checks passed")
