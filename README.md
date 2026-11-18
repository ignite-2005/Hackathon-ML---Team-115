# PathSnap - Team 115

**Distributed map-matching and route reconstruction** (UE24CS352A Machine Learning, PES University).

Noisy GPS traces are matched to a road network. For every GPS point we generate the nearby road
arcs (candidates), score each candidate with a trained classifier, and decode the most likely
sequence of arcs with an HMM/Viterbi decoder that respects road connectivity. The system runs
on two laptops: a FastAPI backend (model, decoder, server-side evaluation) and an evaluator UI.

```
GPS track + road network
   -> candidates (arcs within 50 m, nearest 16)            src/candidates.py
   -> 8 features per (point, arc)                          src/candidate_features.py
   -> Random forest P(arc is correct)                      src/train_models.py
   -> HMM emissions + road-graph transitions -> Viterbi    hmm/
   -> predicted arc per point + driven route               api/pipeline.py
```

Results on the 10 `student_test` trips through the API: **Top-1 accuracy 93.00%**
(classifier only: 85.47%). Route F1 is 94.3% and route continuity is 92.3% (both weighted by GPS
points). See `results/api/student_test_pipeline_results.csv`.

## Repository layout

| Folder | Contents | Owner |
| --- | --- | --- |
| `src/` | Data loading and inventory (`load_data.py`), EDA plots and Folium maps (`eda.py`), per-trip metric projection (`geo.py`), candidate generation and labels (`candidates.py`), radius sweep (`radius_sweep.py`), features (`candidate_features.py`), model training and comparison (`train_models.py`, `neural_network_curve.py`) | A, B |
| `hmm/` | Road graph, HMM states, emissions, transitions, Viterbi, edge cases, route building, metrics, decoder entry point (`decode.py`), map of a decoded trip (`map_trip.py`), config (`hmm_config.json`) | C |
| `tuning/` | Bayesian optimisation (Optuna) | C |
| `api/` | FastAPI backend (`main.py`) and the inference pipeline (`pipeline.py`) | D |
| `ui/` | Streamlit evaluator UI (`app.py`) | D |
| `deploy/` | `run_api.ps1` (Laptop 1) and `run_ui.ps1` (Laptop 2) | D |
| `tests/` | Unit tests (`test_person_b.py`, `test_api.py`) | B, D |
| `outputs/` | EDA, candidate, sweep and HMM outputs (CSV, PNG, Folium HTML) | A, C |
| `results/` | Final model `person_b_final/best_model.joblib`, model comparison results, API results and console log (`results/api/`) | B, D |
| `report/` | Report sections (Markdown), figures, `build_report.py` and the combined PDF | all |

The dataset is **not** in the repository. Put it anywhere and pass its path to the scripts.
`data/` is git-ignored if you want to keep it inside the repo.

```
Student folder/
  train/<trip_id>/<trip_id>.track|.nodes|.arcs|.route        (80 trips)
  student_test/<trip_id>/...                                 (10 trips)
```
`.track` = `lon lat t`, `.nodes` = `lon lat` (row = node id), `.arcs` = `from_node to_node`
(row = arc id), `.route` = reference arc ids in driving order.

## Installation

Python 3.12 on Windows (tested). The same steps work on macOS/Linux with `source .venv/bin/activate`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`scikit-learn` is pinned to **1.8.0**, the version `best_model.joblib` was saved with. Keep it
the same on Laptop 1, otherwise unpickling warns and predictions may change.

## Reproducing the pipeline

Run from the repository root. `DATA` = path to the `Student folder` that contains `train` and `student_test`.

| Step | Command | Output |
| --- | --- | --- |
| Inventory | `python src/load_data.py DATA outputs/inventory.csv` | per-trip counts and checks |
| EDA and Folium map | `python src/eda.py DATA outputs/eda 00000033` | plots, `trip_stats.csv`, `map_*.html` |
| Candidates and labels | `python src/candidates.py DATA outputs/candidates` | parquet files and miss-rate summaries |
| Radius sweep (train) | `python src/radius_sweep.py DATA outputs/sweep` | `radius_sweep_train.csv/png` |
| Model comparison | `python -m src.train_models DATA --output results/person_b_final` | comparison table, plots, `best_model.joblib` |
| HMM / Viterbi | `python hmm/viterbi.py DATA` | `outputs/hmm/viterbi_student_test.csv` |
| Full pipeline offline | `python api/pipeline.py DATA/student_test --out results/api/student_test_pipeline_results.csv` | per-trip Top-1 and route metrics |
| Tests | `python -m unittest tests.test_api tests.test_person_b` | |
| Report PDF | `python report/build_report.py` | `report/PathSnap_Team115_Report.pdf` |

## Running the system (two laptops)

Both laptops must be on the **same Wi-Fi**, with the repo and requirements installed on each.

**Laptop 1: API** (holds the model, the decoder and the trip files, including the reference routes)
```powershell
.\deploy\run_api.ps1 -TripsDir "C:\path\to\evaluation_trips"
```
This is the same as running `$env:PATHSNAP_TRIPS_DIR="C:\path\to\evaluation_trips"` and then
`uvicorn api.main:app --host 0.0.0.0 --port 8000`.
The script prints this laptop's IP addresses. If Laptop 2 cannot connect, allow the port once
from an admin PowerShell:
`New-NetFirewallRule -DisplayName PathSnapAPI -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow`

**Laptop 2: UI**
```powershell
.\deploy\run_ui.ps1 -ApiUrl "http://<Laptop-1-IP>:8000"
```
This is the same as `streamlit run ui/app.py`, with the API URL entered in the sidebar.
Then open `http://localhost:8501`, pick a trip and press **Run trip**, or press **Run all 10**.

**API endpoints**

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Model loaded, trips folder, number of trips |
| `GET /trips` | Trip IDs for the UI to list |
| `POST /predict` `{"trip_id": "00000077"}` | Runs the full pipeline. Returns the predicted arc per GPS point, the route and its geometry, and the server-side scores (Top-1, route precision/recall/F1, continuity; classifier only vs classifier + HMM) |

Interactive API docs: `http://<Laptop-1-IP>:8000/docs`.

**Ground truth never leaves Laptop 1.** Prediction reads only `.track/.nodes/.arcs`. The `.route`
file is used only to compute the scores on the server. The response carries those numbers, never
the reference route. Every request and its scores are logged to the Laptop 1 console.

### Evaluation-day checklist
1. Laptop 1: copy the ten evaluation trips into one folder and start `run_api.ps1 -TripsDir <folder>`.
2. Laptop 2: `run_ui.ps1 -ApiUrl http://<ip>:8000`. The sidebar should show "Connected".
3. Press **Run all 10** once to check every trip works.
4. Screenshot for report section 7: the UI on Laptop 2, and the API console on Laptop 1 showing the requests.

## Team

| Person | Role | Report sections |
| --- | --- | --- |
| A (Shivani) | Data loading, EDA, candidate generation, Folium maps | 2.2, 6 |
| B (Surjo) | Feature engineering, model training and comparison | 3 |
| C | Bayesian optimisation, HMM / Viterbi | 4, 5 |
| D (Shree Krishna) | API, UI, deployment, repo, report assembly | 7 |
