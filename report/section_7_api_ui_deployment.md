# 7. API, UI and Deployment

## 7.1 Two-laptop architecture

| Laptop | Component | Code | Responsibilities |
| --- | --- | --- | --- |
| Laptop 1 | FastAPI backend | `api/main.py`, `api/pipeline.py` | Loads the trained classifier and HMM config once at start-up. Holds the trip files, including the reference routes. Runs the full pipeline for the trip ID it receives, computes accuracy on the server and logs every request to its console |
| Laptop 2 | Streamlit UI | `ui/app.py` | Lists the trips the API offers, sends the selected trip ID to the API, and shows the predicted route on a Folium map with the scores the API returned |

```
Laptop 2 (UI)                                Laptop 1 (API)
streamlit ui/app.py   -- GET  /trips ----->  list trip folders in PATHSNAP_TRIPS_DIR
                      -- POST /predict --->  .track/.nodes/.arcs -> candidates (50 m, k<=16)
                         {"trip_id": ...}      -> 8 features per candidate -> Random forest
                                               -> log-probabilities as HMM emissions
                                               -> Viterbi (road-graph transitions) -> route
                                             .route (server only) -> Top-1, route P/R/F1,
                      <-- JSON ------------     continuity; logged to the console
draw GPS + predicted route, show scores
```

Both laptops join the same Wi-Fi. The API listens on `0.0.0.0:8000`, and the UI is given
`http://<Laptop-1-IP>:8000`.

## 7.2 Inference pipeline behind the API

`api/pipeline.py → Pipeline.predict(folder)` chains the team's components without changing them:

1. `hmm/decode.load_trip_no_labels` reads **only** `.track`, `.nodes` and `.arcs`. The `.route`
   file is never opened during prediction.
2. `src/candidates.generate_candidates` finds the arcs within 50 m of each GPS point and keeps
   the nearest 16 (Section 2).
3. `src/candidate_features.build_features` computes the 8 features. Before scoring, the pipeline
   checks that the feature rows are in exactly the same order as the candidate rows, because the
   decoder pairs probabilities with candidates by position.
4. The **Random forest** from `results/person_b_final/best_model.joblib` (selected in Section 3)
   gives `P(candidate is the true arc)`. These probabilities become the HMM emissions
   (`log(p + 1e-9)`).
5. `hmm/decode.decode_trip` runs Viterbi with transitions from road-network shortest-path
   distances (settings in `hmm/hmm_config.json`) and builds the driven route (Section 5).
6. The classifier-only prediction (the highest-probability candidate per point, no HMM) is
   computed from the same probabilities, so the API can report both side by side.

At start-up the API checks that the candidate radius and `max_k` in the model bundle match
`hmm_config.json`.

## 7.3 Endpoints

| Method and path | Request | Response |
| --- | --- | --- |
| `GET /health` | none | `status`, model name, trips folder, number of trips |
| `GET /trips` | none | `{"trips": ["00000000", ...], "count": 10}` |
| `POST /predict` | `{"trip_id": "00000077"}` | `point_arcs` (predicted arc per GPS point, -1 = none), `route` (driven arc IDs), `route_lonlat` (route geometry), `gps` (input points), `scores` (see below), `decoder_stats`, `timing_s` |

`scores` holds `hmm` and `classifier_only` blocks. Each block has `top1_accuracy`,
`route_continuity`, `route_precision`, `route_recall`, `route_f1` and `unmatched_points`.
`scores` is `null` if the server has no `.route` file for the trip.

**Ground truth stays on Laptop 1.** The model never sees `.route`. The response carries only the
metrics computed from it. The reference route and the per-point labels are never sent to the UI,
and `tests/test_api.py` checks this. Trip IDs are validated against the folder listing, so a
request such as `"../train"` is rejected with a 404.

**Console logging.** Every request is logged on Laptop 1: method, path, client IP, status and
time. Each `/predict` also logs the pipeline stages, the number of points and route arcs, and the
server-side Top-1, route F1, precision, recall and continuity. Example from
`results/api/api_console_log_student_test.txt`:

```
16:18:55 | INFO | === Trip 00000077: running pipeline (candidates -> features -> Random forest -> Viterbi)
16:18:56 | INFO | Trip 00000077: 3911 GPS points -> 310 route arcs, 0 unmatched, 1.22 s
16:18:56 | INFO | Trip 00000077: Top-1 accuracy  HMM 83.74% | classifier-only 69.29%
16:18:56 | INFO | Trip 00000077: route F1 95.22% | precision 93.84% | recall 96.64% | continuity 93.85%
16:18:57 | INFO | POST /predict from 127.0.0.1 -> 200 (1519 ms)
```
A request for an unknown or path-like ID is logged as `WARNING | Unknown trip id '../train'`
and answered with a 404.

## 7.4 User interface

The UI asks the API for the trip list, so it automatically shows whichever ten trips are in the
API's trips folder on evaluation day. The evaluator picks a trip and presses **Run trip**, or
presses **Run all 10** to test every trip. For each trip the UI shows:

- Server-side Top-1 accuracy (with the gain over classifier-only), route F1, precision, recall
  and continuity.
- A Folium map with the GPS observations (blue points), the predicted route (red line), start and
  end markers, a legend and layer controls.
- Run details: points, route arcs, unmatched points, Viterbi chains and timings. Also a
  classifier-only vs classifier + HMM comparison table.
- A summary table of every trip run so far, with pooled Top-1.

![UI home: connected to the API, ten trips listed](figures/ui_home.png)

![UI after running trip 00000000: scores, predicted route on the Folium map, classifier vs HMM](figures/ui_trip_result.png)

![UI after Run all 10: per-trip results pooled over 19,160 points](figures/ui_all_trips.png)

*These screenshots were taken with the API and UI on one laptop (`localhost`). For the final
version of this section, retake them on the two-laptop setup, together with a screenshot of the
API console on Laptop 1 showing the incoming requests (see the checklist in `README.md`).*

## 7.5 Results through the API (student_test, 10 trips)

All ten trips were run through the API pipeline (`python api/pipeline.py "<student_test>" --out
results/api/student_test_pipeline_results.csv`). These are the same numbers the UI showed.

| Trip | GPS points | Top-1 classifier only | Top-1 classifier + HMM | Route F1 (HMM) | Route continuity (HMM) | Time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 00000000 | 2,503 | 93.33% | 97.00% | 94.77% | 95.63% | 0.53 |
| 00000007 | 1,354 | 93.13% | 100.00% | 99.18% | 100.00% | 0.25 |
| 00000009 | 1,049 | 85.89% | 90.56% | 89.00% | 85.05% | 0.46 |
| 00000040 | 2,995 | 91.89% | 94.52% | 97.39% | 96.12% | 0.65 |
| 00000052 | 262 | 95.04% | 100.00% | 93.26% | 89.02% | 0.17 |
| 00000056 | 3,958 | 88.61% | 95.40% | 93.76% | 89.08% | 1.19 |
| 00000065 | 768 | 85.03% | 92.32% | 95.37% | 92.59% | 0.32 |
| 00000076 | 1,486 | 84.93% | 95.63% | 89.86% | 87.32% | 1.02 |
| 00000077 | 3,911 | 69.29% | 83.74% | 95.22% | 93.85% | 1.05 |
| 00000091 | 874 | 85.13% | 93.02% | 86.67% | 83.42% | 0.34 |
| **Pooled (by points)** | **19,160** | **85.47%** | **93.00%** | **94.31%** | **92.31%** | **5.98** |

The HMM raises Top-1 on every trip (+7.5 points pooled) and lifts the point-weighted route
continuity from 74.3% to 92.3%. The classifier-only Top-1 (85.47%) agrees with Person B's
independent measurement (85.46%, Section 3). A full trip takes 0.2–1.2 s on the server.
Including the JSON transfer, the UI receives the answer in about 2 s.

## 7.6 Deployment

**Laptop 1 (API):**
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
.\deploy\run_api.ps1 -TripsDir "C:\path\to\evaluation_trips"
```
The script prints the laptop's LAN addresses and the firewall rule needed to allow port 8000.

**Laptop 2 (UI):**
```powershell
pip install -r requirements.txt
.\deploy\run_ui.ps1 -ApiUrl "http://<Laptop-1-IP>:8000"
```
The browser opens at `http://localhost:8501`.

**Versions.** `scikit-learn` is pinned to 1.8.0, the version the model was saved with. Unpickling
it with a different version raises an `InconsistentVersionWarning` and can change results.

**Tests:** `python -m unittest tests.test_api` uses a synthetic trip and checks the endpoints, the
server-side scoring, a trip without `.route`, rejection of unknown or path-like IDs, and that no
ground truth leaks into the response.
