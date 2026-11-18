# PathSnap: Dataset, Preprocessing and Feature Engineering

This document summarises the dataset work done so far: loading and inventory, EDA, preprocessing
(projection, candidate generation, label alignment, quality filtering, radius choice, road graph)
and the candidate feature engineering.

Source files: `src/load_data.py`, `src/eda.py`, `src/geo.py`, `src/candidates.py`,
`src/radius_sweep.py`, `hmm/graph.py` and `src/candidate_features.py` (all on `main`).

---

## 1. Dataset

The raw data is in `Student folder/` (git-ignored under `data/`), split into two partitions:

| Partition | Trips | GPS points (total) | GPS points per trip (median / min / max) | Road arcs per trip (median / max) | Route arcs per trip (median) |
| --- | ---: | ---: | --- | --- | ---: |
| `train` | 80 | 208,417 | 1,529 / 361 / 19,313 | 19,527 / 307,392 | 280 |
| `student_test` | 10 | 19,160 | 1,420 / 262 / 3,958 | 13,145 / 56,027 | 250 |

Each trip is a folder `<trip_id>/` containing four whitespace-separated files without headers:

| File | Columns | Meaning |
| --- | --- | --- |
| `<id>.track` | `lon lat t` | GPS observations with timestamps in seconds |
| `<id>.nodes` | `lon lat` | Road network nodes. The row number is `node_id` (0-based) |
| `<id>.arcs` | `from_node to_node` | Directed road arcs. The row number is `arc_id` (0-based) |
| `<id>.route` | `arc_id` | Ordered reference route: the ground truth used for training and evaluation |

The trips come from many countries (Germany, Russia, USA, Japan, Brazil and others).
`outputs/eda/trip_extents.png` shows the bounding box of every trip.

## 2. Loading and inventory (`src/load_data.py`)

- `load_trip(folder)` reads the four files into pandas DataFrames with named columns. It keeps
  the row index as `node_id` / `arc_id`.
- `inventory()` writes one row per trip to `outputs/inventory.csv` with these sanity checks:
  - **Missing GPS values:** 0 in both partitions.
  - **Timestamps sorted:** every trip is monotonic.
  - **Route arc IDs valid:** every `route` ID falls in `[0, n_arcs-1]`.
  - **Sampling:** most steps are 1 s (median share of 1 s steps: 99.1% train, 99.95% test).
  - **Large time gaps** exist in a few train trips: `00000090` (one gap of 153,064 s, about 42 h),
    `00000078` (52,810 s), `00000069` (2,658 s), `00000021` (2,161 s). The test partition's
    largest gap is 210 s.

## 3. Exploratory data analysis (`src/eda.py`)

Per-trip statistics go to `outputs/eda/trip_stats.csv`. Distributions use **train only**, so the
test data does not influence any decision.

| Statistic (median per trip) | train | student_test |
| --- | ---: | ---: |
| GPS track length (km) | 24.0 | 21.0 |
| Median speed (m/s) | 12.7 | 15.0 |
| Weakly connected components in the road network | 20 | 11 |
| Share of nodes in the largest component | 98.1% | 97.1% |
| Two-way arcs (an `a→b` arc paired with `b→a`) | 95.1% | 93.3% |
| Isolated nodes | 0 | 0 |

Findings:
- **GPS jumps:** 39 of 80 train trips and 3 of 10 test trips have at least one step faster than
  50 m/s, which is unrealistic and points to GPS noise or outliers.
- **Time gaps over 10 s:** 40 train trips and 2 test trips have them.
- **Route breaks:** in 10 train trips and 1 test trip, at least one route arc does not end where
  the next one starts. The reference route is therefore not always a connected path. Repeated
  consecutive arcs: none.
- **Road network:** most of the network sits in one large component. Nearly all arcs are two-way,
  so a single road usually gives **two candidate arcs** (one per direction). This is why
  heading-based features matter.

Plots in `outputs/eda/`: `gap_hist.png`, `speed_hist.png`, `gps_per_trip.png`, `trip_length.png`,
`trip_extents.png`, plus a static plot and an interactive Folium map for trips `00000033`,
`00000048` and `00000054`.

## 4. Preprocessing

### 4.1 Projection to metres (`src/geo.py`)
The trips span the globe, so no single national grid fits all of them. Each trip gets its own
**local azimuthal-equidistant (AEQD) projection**:
- `trip_origin(d)` returns the mean lon/lat of the trip's **road nodes** (not GPS), so it is also
  available at inference time.
- `to_metres(lon, lat, origin)` converts WGS84 to local x/y metres with pyproj. Distance error is
  well under 1% within a trip's area.
- GPS points and nodes of a trip always use the same origin, so all team code produces matching
  metres.
- `project_onto_segments()` projects points onto arc segments and returns the point-to-segment
  distance, the offset along the arc and the arc length. The projection fraction is clamped to
  `[0, 1]`.

### 4.2 Candidate generation (`src/candidates.py → generate_candidates`)
For each GPS point, we find the road arcs it could lie on:
1. Project the GPS points and arc endpoints to metres.
2. Build a **Shapely STRtree** spatial index over the arc lines, then query all arcs within
   `radius_m` of each point (`dwithin`).
3. Compute the exact distance and offset for each pair, sort by distance, and keep the
   **`max_k = 16` nearest** arcs per point.

Output columns: `trip_id, obs_idx, timestamp, x, y, arc_id, dist_m, offset_m, arc_len_m`.
This step uses only the GPS track and the road network, so it is safe to run at inference.

### 4.3 Label creation by route alignment (`align_route`, `add_labels`)
The reference route is an ordered list of arcs and is much shorter than the GPS track, so each GPS
point has to be assigned a route arc:
- A **monotonic dynamic-programming alignment** (similar to DTW) picks a route position for every
  point. Positions never move backwards in time, and the total point-to-arc distance is minimised.
- Each point gets `route_pos`, `true_arc` and `true_dist_m`.
- `add_labels()` sets `label = 1` for the candidate whose `arc_id == true_arc` and 0 otherwise.
- These labels use the reference route and are for **training and evaluation only**. They must
  never be used in the prediction pipeline.

### 4.4 Label quality filter
If a trip's **median GPS-to-reference-arc distance is above 50 m** (`MAX_MEDIAN_TRUE_DIST_M`),
the GPS and the reference route disagree and the trip's labels are unreliable.
- 2 train trips fail: `00000033` (median 122.4 m, candidate miss rate 66.9%) and `00000054`
  (median 140.1 m, miss rate 90.5%). They are excluded from training and from the radius sweep.
- All 10 test trips pass (median distance 4.2–6.8 m).
- For the trips that pass, the median true distance is about 4.6 m.

### 4.5 Choosing the candidate radius (`src/radius_sweep.py`)
This sweep uses **train only**: 78 good-quality trips, 183,260 GPS points, `max_k = 16`.
The miss rate is the share of points whose true arc is not among their candidates.

| Radius (m) | Miss rate (%) | Candidates per point |
| ---: | ---: | ---: |
| 10 | 19.37 | 2.2 |
| 20 | 4.58 | 3.9 |
| 30 | 2.50 | 5.4 |
| 40 | 1.73 | 6.7 |
| **50** | **1.49** | **7.8** |
| 75 | 1.01 | 9.9 |
| 100 | 0.78 | 11.2 |

**Chosen: 50 m.** Beyond 50 m the miss rate improves only slowly while the candidate count keeps
growing. At 50 m, the per-trip candidate summaries (`outputs/candidates/candidate_summary_*.csv`)
show about 9 candidates per point and a miss rate of **1.49%** on good train trips and **0.17%** on
student_test. The candidate and label tables are saved as parquet files
(`candidates_<part>.parquet`, `point_labels_<part>.parquet`), which are git-ignored.

### 4.6 Road graph for the HMM decoder (`hmm/graph.py → build_graph`)
- Arc length is the straight-line distance in metres between the arc's end nodes. The minimum is
  1 mm, because scipy treats a 0 entry as "no edge".
- **Self-loops** (`u → u`) are dropped.
- **Duplicate `(u, v)` pairs** keep the shorter length. Otherwise `csr_matrix` would add the
  two lengths together.
- The output is a sparse `n_nodes × n_nodes` CSR matrix of metres, ready for scipy Dijkstra
  shortest paths between candidate arcs. The script also reports strongly connected components
  and the share of route arcs inside the largest one.

## 5. Feature engineering (`src/candidate_features.py`)

`build_features(trip, candidates)` adds features to every (GPS point, candidate arc) row. All of
them use only the GPS track and the road network, so they are safe at inference.

| Feature | Definition | Why it helps |
| --- | --- | --- |
| `dist_m` | Distance from the GPS point to the arc segment | The main geometric signal |
| `heading_diff_deg` | Absolute angle (0–180°) between the GPS heading and the arc direction (`from→to`) | Separates the two directions of a two-way road and crossing roads |
| `speed_mps` | GPS speed from adjacent steps | Heading is reliable at speed and noisy when slow |
| `arc_len_m` | Arc length | Short arcs near junctions behave differently |
| `offset_fraction` | `offset_m / arc_len_m`: where along the arc the point projects (0 = start, 1 = end) | Shows whether the point projects into the arc's interior or onto one of its endpoints |
| `candidate_rank` | Rank of this arc by distance among the point's candidates (1 = nearest) | Relative closeness |
| `candidate_count` | Number of candidates for the point | Dense areas, such as junctions and parallel roads, are harder |
| `distance_above_nearest_m` | `dist_m` minus the distance of the point's nearest candidate | Relative margin, independent of the GPS error scale |

**Heading and speed:** each point uses the displacement summed over its previous and next steps.
A step counts only if `0 < dt ≤ 5 s` (`max_heading_gap_s`). Heading is set to NaN when the point is
stationary (movement under 1e-6 m). Speed is NaN when no valid step exists. Missing values are
filled with the **median, fitted inside each CV fold** (the `SimpleImputer` in the model pipeline).

**Training set:** about 9 candidate rows per GPS point, with a binary `label`. Cross-validation
uses `GroupKFold` grouped by trip, so no trip appears in both fit and validation data. The 2
low-quality trips are excluded.

## 6. Leakage safeguards

- The projection origin comes from road nodes only, so it is available for unseen trips.
- Candidates and features use only the track and the network. The reference route is used only
  for labels and evaluation.
- The radius, the quality threshold and the EDA distributions were chosen on `train` only.
- CV folds are grouped by trip, and preprocessing such as imputation and scaling is fitted
  inside each fold.

## 7. Known data issues and open points

- GPS jumps over 50 m/s in about half the train trips. They are not removed yet. The HMM
  transition model or a speed filter could handle them.
- Very long time gaps (`00000090`, `00000078`). Consider splitting the track at large gaps.
- Route breaks in 11 trips mean the reference route is not always connected.
- Two-way roads give near-identical distances for both directions, so the heading feature and
  the HMM transitions have to resolve them.
- Trip `00000002` has a 12% candidate miss rate at 50 m despite passing the quality filter.
  It is worth inspecting.
