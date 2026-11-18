"""PathSnap FastAPI backend (Laptop 1).

Run from the repository root:
    set PATHSNAP_TRIPS_DIR=C:\\path\\to\\evaluation_trips      (PowerShell: $env:PATHSNAP_TRIPS_DIR=...)
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints
  GET  /health           model + config loaded, trips folder, number of trips
  GET  /trips            the trip IDs the UI should list
  POST /predict          {"trip_id": "00000077"} -> predicted route + server-side scores

Ground truth stays on this laptop: the model only sees .track/.nodes/.arcs, and the
response carries metrics computed from .route, never the reference route itself.
Every request is logged to this console (the evaluator is shown these logs).
"""
import logging
import os
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.pipeline import REPO, Pipeline, list_trips

TRIPS_DIR = os.path.abspath(os.environ.get(
    "PATHSNAP_TRIPS_DIR", os.path.join(REPO, "data", "student_test")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("pathsnap")

app = FastAPI(title="PathSnap API", version="1.0",
              description="Map-matching: classifier emissions + HMM/Viterbi decoding")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

log.info("Loading model and HMM config ...")
PIPE = Pipeline()
log.info("Model: %s | features: %s", PIPE.model_name, PIPE.feature_options)
log.info("Trips folder: %s (%d trips)", TRIPS_DIR, len(list_trips(TRIPS_DIR)))


class PredictRequest(BaseModel):
    trip_id: str


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    log.info("%s %s from %s -> %d (%.0f ms)", request.method, request.url.path,
             request.client.host if request.client else "?", response.status_code,
             1000 * (time.perf_counter() - t0))
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model": PIPE.model_name, "trips_dir": TRIPS_DIR,
            "n_trips": len(list_trips(TRIPS_DIR)), "emission": "classifier"}


@app.get("/trips")
def trips():
    ids = list_trips(TRIPS_DIR)
    return {"trips": ids, "count": len(ids)}


@app.post("/predict")
def predict(req: PredictRequest):
    trip_id = req.trip_id.strip()
    # only IDs that exist in the trips folder (also blocks paths like "../x")
    if trip_id not in list_trips(TRIPS_DIR):
        log.warning("Unknown trip id %r", trip_id)
        raise HTTPException(404, f"trip {trip_id!r} not found")
    folder = os.path.join(TRIPS_DIR, trip_id)
    log.info("=== Trip %s: running pipeline (candidates -> features -> %s -> Viterbi)",
             trip_id, PIPE.model_name)

    t0 = time.perf_counter()
    try:
        pred = PIPE.predict(folder)
    except Exception as e:  # report the failure to the UI instead of a bare 500
        log.exception("Trip %s: prediction failed", trip_id)
        raise HTTPException(500, f"prediction failed: {e}")
    t_pred = time.perf_counter() - t0
    log.info("Trip %s: %d GPS points -> %d route arcs, %d unmatched, %.2f s",
             trip_id, len(pred["point_arcs"]), len(pred["route"]),
             pred["stats"].get("unmatched", 0), t_pred)

    t1 = time.perf_counter()
    scores = PIPE.evaluate(folder, pred)
    t_eval = time.perf_counter() - t1
    if scores is None:
        log.info("Trip %s: no reference route on server, accuracy not computed", trip_id)
    else:
        h, c = scores["hmm"], scores["classifier_only"]
        log.info("Trip %s: Top-1 accuracy  HMM %.2f%% | classifier-only %.2f%%",
                 trip_id, 100 * h["top1_accuracy"], 100 * c["top1_accuracy"])
        log.info("Trip %s: route F1 %.2f%% | precision %.2f%% | recall %.2f%% | continuity %.2f%%",
                 trip_id, 100 * h["route_f1"], 100 * h["route_precision"],
                 100 * h["route_recall"], 100 * (h["route_continuity"] or 0))

    return {
        "trip_id": trip_id,
        "model": pred["model_name"],
        "n_points": len(pred["point_arcs"]),
        "point_arcs": pred["point_arcs"],          # predicted arc per GPS point (-1 = none)
        "route": pred["route"],                    # predicted driven arcs, in order
        "route_lonlat": pred["route_lonlat"],      # geometry of the predicted route
        "gps": PIPE.gps(folder),                   # [[lat, lon], ...] input observations
        "scores": scores,                          # computed here, from server-side labels
        "decoder_stats": pred["stats"],
        "timing_s": {"prediction": round(t_pred, 3), "evaluation": round(t_eval, 3)},
    }
