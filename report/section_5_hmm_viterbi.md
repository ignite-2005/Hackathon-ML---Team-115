# 5. HMM and Viterbi route reconstruction

The classifier (Sections 3 and 4) scores every candidate arc of a GPS point on its own. Choosing the best arc point by point ignores the road network, so the result can jump between parallel roads or flip between the two directions of a road. A Hidden Markov Model adds that missing context. It picks the **sequence** of arcs that both fits the GPS points and can actually be driven. Code: `hmm/` (graph, states, emissions, transitions, Viterbi, edge cases, route builder, `decode.py`).

## 5.1 Observations and hidden states

- **Observations:** the GPS points of a trip in time order, `t = 0 ... n-1` (timestamps checked to be sorted; no trip has repeated timestamps).
- **Hidden state at point t:** the road arc the car is really on. Its possible values are point t's **candidate arcs**: arcs within 50 m, at most the 16 nearest (Section 3). The number of states differs per point (student_test: 1 to 16).
- `build_states` keeps, for every point, the candidate arc ids, their distance and offset along the arc, and their row numbers in the candidate table. The row numbers link the classifier's probabilities to the right states.
- **Road graph** (`build_graph`): a sparse node-to-node matrix with `graph[from, to] = arc length` in metres, built with the shared per-trip projection. Self-loops are dropped (24 over all 90 trips). Duplicate (from, to) pairs keep the shortest arc (1,891 merged). Lengths are clamped to at least 1e-3 m, because a zero in a sparse matrix means "no edge" (106 arcs).

## 5.2 Emission scores

Two versions were built and compared:

| Emission | Formula | Notes |
| --- | --- | --- |
| Gaussian (baseline) | `log_e = -0.5 (dist_m / sigma)^2` | sigma = 1.4826 x median GPS-to-true-arc distance on train = **7.35 m** (a robust standard deviation; 183,260 points, trips 33 and 54 left out because their labels are unreliable) |
| Classifier (final) | `log_e = log(p + 1e-9)` | p = tuned Random forest probability for that (point, arc). The 1e-9 avoids log(0). Probabilities are matched to states by (obs_idx, arc_id). |

Distance alone cannot tell the direction of travel. Both directions of a two-way road are equally close. On student_test, the nearest arc was right at only **54.3%** of points. The classifier adds heading and speed and reaches 85.6% (Section 4).

## 5.3 Transition scores

Following Newson and Krumm (2009), a move from (arc a, offset oa) at point t-1 to (arc b, offset ob) at point t is scored by comparing the distance driven along the roads with the straight-line distance between the two GPS points:

- Same arc: `d_route = ob - oa`. Up to 30 m backwards is accepted as GPS noise (see below). More than that is impossible.
- Otherwise: `d_route = (len_a - oa) + SP(end of a, start of b) + ob`. When b starts where a ends, SP = 0.
- `d_gc` = straight-line distance between the two GPS points, and `log_t = -|d_route - d_gc| / beta`.
- SP is the shortest road distance (scipy `dijkstra`), limited to `dt x 50 m/s + 100 m` and cached per start node. No path within that limit means the move is impossible (`log_t = -inf`). The largest network (trip 13: 155k nodes, 307k arcs) decodes in about 2 s.

**Starting value of beta.** The median `|d_route - d_gc|` over the true arc pairs of 180,204 train steps, divided by ln 2, gives **beta = 1.70 m**. This is small because points are 1 s apart, so road and straight-line distances nearly always agree.

**Backwards tolerance.** With the brief's 10 m limit, 1,759 true moves (0.98%) came out impossible. 13 were paths beyond the cutoff (160 to 250 m in 1 s), caused by label noise near junctions. 1,746 were "moved backwards more than 10 m on the same arc". For each of these we compared the GPS heading over points t-5 to t+5 with the arc's direction:

| Cause | Cases |
| --- | ---: |
| GPS jitter (heading matches the arc, one fix jumps back) | 1,041 |
| Wrong direction in the reference labels (reverse arc exists, heading opposite) | 256 |
| Heading opposite, no reverse arc (one-way arc) | 72 |
| Unclear (car almost still, or moving sideways at junctions) | 377 |

Most cases were GPS noise, so the tolerance was raised to **30 m**. This left only 21 impossible true moves (0.01%). Going backwards is still penalised by `|d_route - d_gc|`. The 328 wrong-direction labels are a labelling issue, reported to Person A. They slightly understate our accuracy.

## 5.4 Viterbi decoding

In log space, for each point t and candidate j:

`delta_t(j) = max_i [ delta_{t-1}(i) + log_t(i, j) ] + w * log_e_t(j)`,  with `back_t(j) = argmax_i`.

The best final candidate is followed back through the back pointers to recover the path. `delta` is shifted so its maximum is 0 at every step. This does not change the result but keeps the numbers small on long trips.

**Check:** Viterbi must return the highest-scoring path. On train trip 46 its path scored -4,890.6, against -5,219.4 for the true arc sequence. Choosing the nearest arc at every point gave a score of -inf on all trips we checked, because it contains impossible moves.

## 5.5 Gaps, missing candidates and disconnected segments

**Chain breaks.** Viterbi cannot continue across certain points. There it ends the current chain, backtracks it and starts a new one:

| Reason | train (80 trips) | student_test |
| --- | ---: | ---: |
| Time gap over the threshold (steps with dt > 60 s) | 69 | 1 |
| Point with no candidate (no arc within 50 m) | 1,711 points (+16,519 in trip 54, whose GPS is off the map) | 0 |
| Every move impossible (Gaussian decoder) | rare (e.g. 4 in trip 2) | 1 (trip 56) |

**Points with no candidates** are left **unmatched (-1)** and counted. We also tried filling them from the nearest matched point in time (up to 5 or 30 s). On trip 2 this filled 108 and 202 points, but **none were correct**. These points are a median of 222 m from the labelled arc, and the matched points around them were only 1% correct. The GPS and the reference route simply disagree there. Filling was therefore left off.

**Stationary points.** We tried the optional idea of skipping points within k x sigma of the last kept point and giving skipped points the nearest neighbouring decoded arc. It did **not** help (student_test, Gaussian emissions, route measured before the loop fix below):

| | Top-1 | Route F1 | Points decoded |
| --- | ---: | ---: | ---: |
| No skipping | **89.6%** | **91.9%** | 100% |
| Skip within 2 sigma | 88.5% | 88.8% | 57% |
| Skip within 3 sigma | 87.6% | 85.0% | 42% |

**Building the route (disconnected segments).** The route is made from the per-point arcs by:
1. Collapsing repeats.
2. Cutting **short loops**. If the route returns to an arc it already used and the arcs in between total 50 m or less, the detour is removed.

Without step 2, a stopped car with jittery GPS made the route flip `a -> reverse(a) -> a`. Trip 77's route had 629 arcs for a 268-arc reference. Real routes do loop (10 of 90 reference routes revisit an arc), but the shortest real loop is 76 m, so 50 m only removes jitter. Over student_test this cut route length from 3,829 to 3,352 arcs (reference: 3,477) and U-turn pairs from 514 to 89. Gaussian route F1 rose from 91.9% to 92.7%. Consecutive route arcs that are not directly connected (Viterbi moved between them through a shortest path) are kept as they are. This is why route continuity stays around 90% (see 5.7).

## 5.6 Did the HMM improve the route?

**Decoder tuning.** beta, w and the gap threshold were tuned on **train only**. We used **out-of-fold** probabilities: each trip was scored by a model trained on the other folds, with Section 3's folds. Trips 33 and 54 were left out, and the setting with the best point Top-1 was chosen. Multiplying every score by the same factor does not change Viterbi's best path, so **only the product w x beta matters**. The grid shows this: (beta 5, w 1), (beta 10, w 0.5) and (beta 20, w 0.25) all give 92.09% train Top-1. The best classifier setting was beta = 1, w = 4, gap 120 s (train Top-1 92.58%). The gap threshold hardly matters (15 to 120 s: 92.31 to 92.58%).

**student_test, all 10 trips (19,160 points), tuned Random forest:**

| Method | Top-1 | Route precision | Route recall | Route F1 | Continuity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nearest arc | 54.26% | 56.27 | 63.25 | 59.56 | 47.25 |
| Classifier only (argmax p) | 85.57% | 83.69 | 87.09 | 85.35 | 76.39 |
| Gaussian + Viterbi (sigma 7.35, beta 1.70, w 1) | 89.59% | 94.69 | 90.78 | 92.70 | 89.95 |
| Classifier + Viterbi (beta 1.70, w 1, gap 60 s) | 92.86% | **96.12** | **90.01** | **92.96** | **90.10** |
| **Classifier + Viterbi (tuned: beta 1, w 4, gap 120 s), deployed** | **93.16%** | 95.81 | 89.38 | 92.49 | 90.09 |

Route metrics here pool arc counts over all trips. The API (Section 7) reports the same Top-1 (93.16%) but averages route metrics per trip weighted by GPS points (route F1 93.71%, continuity 92.13%).

**Per trip, classifier only vs classifier + HMM (deployed):**

| Trip | 00 | 07 | 09 | 40 | 52 | 56 | 65 | 76 | 77 | 91 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Top-1, classifier only | 93.8 | 91.9 | 86.5 | 92.0 | 95.4 | 88.7 | 84.8 | 85.8 | 69.2 | 85.4 |
| Top-1, + HMM | 96.7 | 99.3 | 91.7 | 95.4 | 100.0 | 95.5 | 91.7 | 96.3 | 84.0 | 92.0 |
| Route F1, classifier only | 86.4 | 97.2 | 84.5 | 88.8 | 85.7 | 89.4 | 81.8 | 77.6 | 78.7 | 81.0 |
| Route F1, + HMM | 94.8 | 98.4 | 89.4 | 97.4 | 93.3 | 93.0 | 93.7 | 89.6 | 93.9 | 85.6 |

**Yes.** The HMM improved Top-1 by 7.6 points (85.57% to 93.16%) and route F1 by 7.1 points (85.35% to 92.49%). Continuity rose from 76.4% to 90.1%. **There was no trip where the HMM did not help.** It beat the classifier alone on all 10 trips, on both Top-1 and route F1.

The HMM also helps without the classifier (Gaussian + Viterbi: 89.59%). Combining both is best. Decoder tuning raised Top-1 slightly (+0.30) but lowered route F1 slightly (-0.47), because it was tuned for Top-1. Using the untuned forest instead gives almost the same result (93.39% Top-1), so adding the HMM mattered much more than tuning the classifier.

**Weakest trip: 77** (84.0%). Its car is stopped for 36.3% of the points (net GPS movement under 20 m over +-10 points). Every other test trip is stopped for 13% or less. While stopped, the GPS still jumps about 7 m per second in random directions. For example, from points 2312 to 2587 it "travels" 2,090 m inside a 31 x 61 m box. There is no direction information, so both directions of the road fit equally well. With Gaussian emissions, Top-1 on trip 77 was 50.4% on stopped points against 89.8% on moving points. 524 of its 959 errors were the reverse of the true arc. The labels are equally ambiguous there, so some of these "errors" cannot be resolved from GPS at all.

## 5.7 Limitations

- **Jumps not filled.** Gaps between non-adjacent route arcs are not yet filled with the shortest-path arcs between them. This would raise continuity and recall.
- **Stops.** Long stops remain the main source of error (trip 77). Skipping stationary points did not fix it.
- **Small tuning gains.** The decoder settings were picked by train Top-1. On student_test the tuned and untuned decoders are within 0.3 points, so the gain from tuning is small.
- **Label noise.** About 0.2% of train transitions have reference labels in the wrong direction (5.3), which slightly understates measured accuracy.

**Reproduce:** `python hmm/transitions.py DATA` (sigma and beta), `python hmm/viterbi.py DATA`, `python hmm/edge_cases.py DATA`, `python hmm/classifier_hmm.py probas|tune|compare DATA --params tuned`, `python api/pipeline.py DATA/student_test`. Results are in `outputs/hmm/`.
