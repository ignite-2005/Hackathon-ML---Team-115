"""API checks (Person D): endpoints, server-side scoring, no ground-truth leak.

Run from the repository root:  python -m unittest tests.test_api
Uses a tiny synthetic trip, so the real dataset is not needed.
"""
import importlib
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


def write_trip(root, trip_id, with_route=True):
    """Straight two-way road along the equator, 4 GPS points driving east."""
    folder = Path(root) / trip_id
    folder.mkdir(parents=True)
    files = {
        "track": "".join(f"{0.0001 + 0.0002 * i} 0.00001 {i}\n" for i in range(4)),
        "nodes": "0 0\n0.001 0\n",
        "arcs": "0 1\n1 0\n",
    }
    if with_route:
        files["route"] = "0\n"
    for ext, text in files.items():
        (folder / f"{trip_id}.{ext}").write_text(text, encoding="utf-8")


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = TemporaryDirectory()
        write_trip(cls.tmp.name, "00000001")
        write_trip(cls.tmp.name, "00000002", with_route=False)
        os.environ["PATHSNAP_TRIPS_DIR"] = cls.tmp.name
        import api.main
        main = importlib.reload(api.main)        # pick up the temporary trips folder
        from fastapi.testclient import TestClient
        cls.client = TestClient(main.app)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_health_and_trips(self):
        h = self.client.get("/health").json()
        self.assertEqual(h["status"], "ok")
        self.assertEqual(self.client.get("/trips").json()["trips"], ["00000001", "00000002"])

    def test_predict_scores_on_server(self):
        r = self.client.post("/predict", json={"trip_id": "00000001"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(len(body["point_arcs"]), 4)
        self.assertEqual(len(body["gps"]), 4)
        self.assertEqual(body["route"], [0])                    # drives arc 0 (eastbound)
        self.assertEqual(body["scores"]["hmm"]["top1_accuracy"], 1.0)
        # the reference route / per-point labels must never be sent to the UI
        text = r.text
        for leak in ("true_arc", "reference_route", "true_dist"):
            self.assertNotIn(leak, text)

    def test_trip_without_route_has_no_scores(self):
        body = self.client.post("/predict", json={"trip_id": "00000002"}).json()
        self.assertIsNone(body["scores"])
        self.assertEqual(len(body["point_arcs"]), 4)

    def test_unknown_or_path_trip_rejected(self):
        for bad in ("99999999", "../00000001", ".."):
            self.assertEqual(self.client.post("/predict", json={"trip_id": bad}).status_code, 404)


if __name__ == "__main__":
    unittest.main()
