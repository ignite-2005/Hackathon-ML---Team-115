"""PathSnap evaluator UI (Laptop 2).

Run:  streamlit run ui/app.py
Set the API address in the sidebar (or PATHSNAP_API_URL), e.g. http://192.168.1.20:8000

The UI only talks to the API: it lists the trips the API offers, sends the chosen trip ID,
and draws the returned predicted route over the GPS points. It never reads trip files and
never receives the reference route, only the accuracy numbers the API computes.
"""
import os

import folium
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="PathSnap - Team 115", layout="wide")

DEFAULT_API = os.environ.get("PATHSNAP_API_URL", "http://localhost:8000")
TIMEOUT_S = 300

if "results" not in st.session_state:
    st.session_state.results = {}      # trip_id -> API response


@st.cache_data(ttl=30, show_spinner=False)   # Streamlit reruns on every click; don't spam the API
def api_get(base, path):
    r = requests.get(f"{base}{path}", timeout=10)
    r.raise_for_status()
    return r.json()


def api_predict(base, trip_id):
    r = requests.post(f"{base}/predict", json={"trip_id": trip_id}, timeout=TIMEOUT_S)
    if r.status_code != 200:
        raise RuntimeError(f"API {r.status_code}: {r.text}")
    return r.json()


def pct(v):
    return "N/A" if v is None else f"{100 * v:.2f}%"


def build_map(res):
    gps = res["gps"]
    lats = [p[0] for p in gps]
    lons = [p[1] for p in gps]
    m = folium.Map(location=[sum(lats) / len(lats), sum(lons) / len(lons)],
                   zoom_start=14, tiles="OpenStreetMap")

    pts = folium.FeatureGroup(name="GPS observations")
    step = max(1, len(gps) // 2500)          # keep the page light for long trips
    for lat, lon in gps[::step]:
        folium.CircleMarker([lat, lon], radius=2, color="#1f6feb", fill=True,
                            fill_opacity=0.8, weight=1).add_to(pts)
    pts.add_to(m)

    route = folium.FeatureGroup(name="Predicted route (classifier + HMM)")
    for (lon_a, lat_a), (lon_b, lat_b) in res["route_lonlat"]:
        folium.PolyLine([[lat_a, lon_a], [lat_b, lon_b]], color="#d62728", weight=5,
                        opacity=0.85).add_to(route)
    route.add_to(m)

    folium.Marker(gps[0], tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(gps[-1], tooltip="End", icon=folium.Icon(color="black")).add_to(m)
    legend = f"""<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;
      padding:8px 12px;border:1px solid #888;font-size:13px;color:#222">
      <b>Trip {res['trip_id']}</b><br>
      <span style="color:#1f6feb">&#9679;</span> GPS observations<br>
      <span style="color:#d62728">&#9644;</span> Predicted route<br>
      <span style="color:green">&#9679;</span> Start &nbsp;
      <span style="color:black">&#9679;</span> End</div>"""
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    return m


# ---------------- sidebar: connection ----------------
st.sidebar.title("PathSnap")
api = st.sidebar.text_input("API URL (Laptop 1)", DEFAULT_API).rstrip("/")
try:
    health = api_get(api, "/health")
    st.sidebar.success(f"Connected - model: {health['model']}")
    trip_ids = api_get(api, "/trips")["trips"]
except Exception as e:
    st.sidebar.error(f"Cannot reach API at {api}\n\n{e}")
    st.title("PathSnap: map-matching evaluator")
    st.info("Start the API on Laptop 1 and enter its address in the sidebar.")
    st.stop()

st.sidebar.caption(f"{len(trip_ids)} evaluation trips available")
if st.sidebar.button("Clear results"):
    st.session_state.results = {}

# ---------------- main ----------------
st.title("PathSnap: map-matching evaluator")
st.caption("Team 115 · noisy GPS → road arcs with a classifier + HMM/Viterbi decoder. "
           "Accuracy is computed on the API server; ground truth never leaves it.")

if not trip_ids:
    st.warning("The API has no trips in its trips folder.")
    st.stop()

# Optional link ?trip=<id> opens the UI with that trip selected and runs it once
# (still a normal API call, handy for sharing a result or taking screenshots).
linked = st.query_params.get("trip")
auto_run = linked in trip_ids and not st.session_state.get("auto_ran")

col_sel, col_run, col_all = st.columns([3, 1, 1])
trip = col_sel.selectbox("Evaluation trip", trip_ids,
                         index=trip_ids.index(linked) if linked in trip_ids else 0)
run_one = col_run.button("Run trip", type="primary", width="stretch")
run_all = col_all.button(f"Run all {len(trip_ids)}", width="stretch")

if run_one or auto_run:
    st.session_state.auto_ran = True
    with st.spinner(f"API is decoding trip {trip} ..."):
        try:
            st.session_state.results[trip] = api_predict(api, trip)
        except Exception as e:
            st.error(str(e))

if run_all:
    bar = st.progress(0.0, text="Running all trips through the API ...")
    for i, t in enumerate(trip_ids, 1):
        try:
            st.session_state.results[t] = api_predict(api, t)
        except Exception as e:
            st.error(f"Trip {t}: {e}")
        bar.progress(i / len(trip_ids), text=f"Trip {t} done ({i}/{len(trip_ids)})")
    bar.empty()

res = st.session_state.results.get(trip)
if res:
    s = res["scores"]
    st.subheader(f"Trip {trip}")
    if s:
        h, c = s["hmm"], s["classifier_only"]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Top-1 accuracy (HMM)", pct(h["top1_accuracy"]),
                  f"{100 * (h['top1_accuracy'] - c['top1_accuracy']):+.2f} pts vs classifier only")
        m2.metric("Route F1", pct(h["route_f1"]))
        m3.metric("Route precision", pct(h["route_precision"]))
        m4.metric("Route recall", pct(h["route_recall"]))
        m5.metric("Route continuity", pct(h["route_continuity"]))
    else:
        st.info("The server has no reference route for this trip, so no accuracy is shown.")

    left, right = st.columns([3, 1])
    with left:
        st.iframe(build_map(res).get_root().render(), height=600)
    with right:
        st.markdown("**Run details**")
        details = {
            "Model": res["model"],
            "GPS points": f"{res['n_points']:,}",
            "Predicted route arcs": f"{len(res['route']):,}",
            "Unmatched points": str(res["decoder_stats"].get("unmatched", 0)),
            "Viterbi chains": str(res["decoder_stats"].get("chains", 1)),
            "Prediction time": f"{res['timing_s']['prediction']:.2f} s",
            "Evaluation time": f"{res['timing_s']['evaluation']:.2f} s",
        }
        st.table(pd.DataFrame({"value": details}))

    if s:
        st.markdown("**Classifier only vs classifier + HMM/Viterbi** (computed on the API server)")
        names = {"top1_accuracy": "Top-1 accuracy", "route_continuity": "Route continuity",
                 "route_precision": "Route precision", "route_recall": "Route recall",
                 "route_f1": "Route F1"}
        st.table(pd.DataFrame({
            "Classifier only": {names[k]: pct(c[k]) for k in names},
            "Classifier + HMM": {names[k]: pct(h[k]) for k in names},
            "Change (pts)": {names[k]: f"{100 * ((h[k] or 0) - (c[k] or 0)):+.2f}" for k in names},
        }).T)
else:
    st.info("Pick a trip and press **Run trip**.")

# ---------------- summary over every trip run so far ----------------
done = st.session_state.results
if len(done) > 1:
    st.subheader("All trips run so far")
    rows = []
    for t, r in sorted(done.items()):
        sc = r["scores"] or {}
        h, c = sc.get("hmm", {}), sc.get("classifier_only", {})
        rows.append({"trip": t, "points": r["n_points"],
                     "Top-1 HMM": h.get("top1_accuracy"), "Top-1 classifier": c.get("top1_accuracy"),
                     "route F1": h.get("route_f1"), "continuity": h.get("route_continuity"),
                     "time (s)": r["timing_s"]["prediction"]})
    df = pd.DataFrame(rows)
    pts = df["points"].sum()
    st.dataframe(df.style.format({k: "{:.2%}" for k in
                                  ["Top-1 HMM", "Top-1 classifier", "route F1", "continuity"]} | {"time (s)": "{:.2f}", "points": "{:,}"},
                                 na_rep="N/A"), width="stretch", hide_index=True)
    if df["Top-1 HMM"].notna().all():
        st.caption(f"Pooled over {pts:,} points: Top-1 HMM "
                   f"{(df['Top-1 HMM'] * df['points']).sum() / pts:.2%}, classifier only "
                   f"{(df['Top-1 classifier'] * df['points']).sum() / pts:.2%}")
