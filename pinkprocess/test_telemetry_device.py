"""Device protocol, Q8 arithmetic on INT8 lanes, and weight recovery."""

import json
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from uuid import uuid4

from .fixed_point import FEATURE_BOUNDS
from .telemetry_device import BASE, VERSIONS, TelemetryDevice, SCORES


PROFILES = Path(__file__).resolve().parents[1] / "profiles"


class DotProduct:
    def __init__(self):
        self.mmio = self
        self.rows = {}

    def write(self, offset, row):
        assert offset % 64 == 0 and len(row) == 64
        self.rows[offset//64] = row

    def read_row(self, index):
        return self.rows[index]

    def submit(self, activation):
        assert len(activation) == 64
        self.scores = {i:sum(x*y for x,y in zip(memoryview(activation).cast("b"),
                                                 memoryview(row).cast("b"))) for i,row in self.rows.items()}

    def read(self, register):
        return self.scores[(register-SCORES)//4] & 0xffffffff


class TelemetryDeviceTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "active.json"
        self.core = DotProduct()
        self.device = TelemetryDevice(self.core, PROFILES, self.path)

    def call(self, op, **kwargs):
        return self.device.handle({"op": op, **kwargs})

    def test_fixed_point_vectors_and_rollback(self):
        saved = json.loads((PROFILES / "service-b-fixed-point-vectors.json").read_text())
        rows = saved["vectors"]
        inputs = [row["features"] for row in rows]
        before = self.call("query_batch", features=inputs)
        self.call("apply", version=VERSIONS[1])
        after = self.call("query_batch", features=inputs)
        self.assertNotEqual(before["weights_sha256"], after["weights_sha256"])
        self.assertEqual(after["scores"], [row["expected_score"] for row in rows])
        self.assertEqual(after["alerts"], [row["expected_alert"] for row in rows])
        reconnected = TelemetryDevice(DotProduct(), PROFILES, self.path)
        self.assertEqual(reconnected.handle({"op":"status"})["version"], VERSIONS[1])
        self.call("rollback")
        self.assertEqual(before, self.call("query_batch", features=inputs))

    def test_all_feature_corners(self):
        for version in VERSIONS:
            self.call("apply", version=version)
            cases = list(product(*FEATURE_BOUNDS))
            result = self.call("query_batch", features=cases)
            profile = self.device.models[version].profile
            self.assertEqual(result["scores"], [profile.score(case) for case in cases])

    def test_apply_failure_and_validation(self):
        before = self.call("status")
        with patch.object(self.device, "_save", side_effect=OSError("full")):
            with self.assertRaises(OSError):
                self.call("apply", version=VERSIONS[1])
        self.assertEqual(self.core.read_row(0), self.device.models[BASE].row)
        for op, kwargs in [("apply", {"version":"unknown"}),
                           ("query_batch", {"features":[[True, 0, 0]]}),
                           ("query_batch", {"features":[[257, 0, 0]]}),
                           ("query_batch", {"features":[[0, 0, 0]]*513})]:
            with self.assertRaises(ValueError):
                self.call(op, **kwargs)
        self.assertEqual(self.call("status"), before)

    def test_backlog_rows_survive_onset_weight_changes(self):
        before = self.call("backlog", features=[50,-2,-2,1])
        self.call("apply", version=VERSIONS[1])
        self.assertEqual(before, self.call("backlog", features=[50,-2,-2,1]))
        with self.assertRaises(ValueError):
            self.call("backlog", features=[128]*6)

    def test_window(self):
        readings = [{"time_ms":100*i, "busy":5, "waiting":0} for i in range(50)]
        result = self.call("query", capacity=10, readings=readings)
        reference = self.call("query_batch", features=[[128, 0, 0]])
        self.assertEqual(result["score"], reference["scores"][0])
        self.assertEqual(result["time_ms"], 4900)
        readings[-1]["waiting"] = 1
        with self.assertRaises(ValueError):
            self.call("query", capacity=10, readings=readings)

    def test_stream_scores_at_ingest_and_queries_reuse_the_result(self):
        self.device.stream.clock = lambda: 10
        self.assertTrue(self.call("status")["stream_supported"])
        session = str(uuid4())
        request = dict(session=session, source="Test", capacity=10)
        rows = [{"time_ms": 100 * i, "busy": i % 10, "waiting": 0} for i in range(50)]
        result = self.device.ingest(dict(request, readings=rows))
        self.assertEqual((result["state"], result["fpga_submissions"]), ("ready", 1))
        forecast = self.call("stream_forecast")["forecast"]
        self.assertEqual(forecast["time_ms"], 4900)
        self.assertEqual(forecast["score"], self.call("query", capacity=10, readings=rows)["score"])

        occupied = dict(time_ms=5000, busy=10, waiting=2)
        self.device.ingest(dict(request, readings=[occupied]))
        forecast = self.call("stream_forecast")["forecast"]
        self.assertEqual(forecast["targets"], ["persists", "grows"])
        observations = self.call("stream_observations", window_ms=250, session=session)["observations"]
        self.assertEqual(observations["sample_count"], 3)
        self.assertEqual(observations["peak_busy"], occupied)
        self.assertEqual(observations["peak_waiting"], occupied)
        self.assertEqual(self.call("stream_status")["fpga_submissions"], 2)

        self.call("apply", version=VERSIONS[1])
        self.assertIsNone(self.call("stream_forecast")["forecast"])
        self.device.ingest(dict(request, readings=[dict(time_ms=5100, busy=5, waiting=0)]))
        self.assertEqual(self.call("stream_forecast")["forecast"]["version"], VERSIONS[1])
        self.assertEqual(self.call("stream_status")["fpga_submissions"], 3)

    def test_stream_retries_do_not_refresh_stale_data(self):
        self.device.stream.clock = lambda: now
        now = 10
        request = dict(session=str(uuid4()), source="Test", capacity=10)
        rows = [{"time_ms": 100 * i, "busy": 5, "waiting": 0} for i in range(50)]
        self.device.ingest(dict(request, readings=rows))
        now = 12
        result = self.device.ingest(dict(request, readings=rows[-1:]))
        self.assertEqual((result["state"], result["accepted"]), ("stale", 0))
        self.assertIsNone(self.call("stream_forecast")["forecast"])
        self.assertEqual(self.call("stream_observations")["observations"]["sample_count"], 50)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.device.ingest(dict(request, readings=[dict(rows[-1], busy=6)]))
        result = self.device.ingest(dict(request, readings=[dict(time_ms=5000, busy=5, waiting=0)]))
        self.assertEqual((result["state"], result["sample_count"]), ("warming", 1))
        result = self.device.ingest(dict(request, readings=[dict(time_ms=5300, busy=5, waiting=0)]))
        self.assertEqual((result["state"], result["sample_count"]), ("warming", 1))
        self.assertEqual(self.call("stream_status")["fpga_submissions"], 1)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            self.device.ingest(dict(request, readings=[dict(time_ms=5200, busy=5, waiting=0)]))

    def test_stream_evidence_pairs_cached_model_with_observed_statistics(self):
        self.device.stream.clock = lambda: now
        now = 10
        request = dict(session=str(uuid4()), source="Live pool", capacity=10)
        rows = [dict(time_ms=100 * i, busy=6 if i < 25 else 10, waiting=2 if i >= 40 else 0)
                for i in range(50)]
        self.assertTrue(self.call("status")["stream_evidence_supported"])
        self.device.ingest(dict(request, readings=rows))
        evidence = self.call("stream_evidence", window_ms=10_000, session=request["session"])
        self.assertEqual((evidence["source"], evidence["origin"], evidence["fresh"]), ("Live pool", "live", True))
        self.assertEqual(evidence["latest_time_ms"], 4900)
        self.assertEqual(evidence["fpga_submissions"], 1)
        self.assertEqual(evidence["forecast"], self.call("stream_forecast")["forecast"])
        self.assertEqual(evidence["forecast"]["model"], "queue_outlook")
        observations = evidence["observations"]
        self.assertEqual((observations["covered_ms"], observations["complete"]), (5000, False))
        self.assertEqual(observations["busy"], dict(minimum=6, maximum=10, mean=8, change=4,
                                                     full_samples=25, mean_utilization=0.8))
        self.assertEqual(observations["waiting"], dict(minimum=0, maximum=2, mean=0.4, change=2,
                                                        positive_samples=10, onsets=1, clearances=0))
        intervals = observations["intervals"]
        self.assertEqual(sum(interval["sample_count"] for interval in intervals), 50)
        self.assertEqual(intervals[0]["start_ms"], 0)
        self.assertEqual(intervals[-1]["end_ms"], 4900)
        self.assertIn("retained", " ".join(evidence["limitations"]))

        now = 12
        stale = self.call("stream_evidence")
        self.assertEqual((stale["state"], stale["age_ms"], stale["forecast"]), ("stale", 2000, None))
        self.assertTrue(stale["observations"]["complete"])
        self.assertIn("stale", " ".join(stale["limitations"]))

    def test_stream_evidence_limits_partial_history_to_the_matching_session(self):
        self.device.stream.clock = lambda: 10
        self.device.ingest(dict(session=str(uuid4()), source="Demo", origin="demo", capacity=10,
                                readings=[dict(time_ms=900, busy=5, waiting=0)]))
        evidence = self.call("stream_evidence")
        self.assertEqual((evidence["state"], evidence["forecast"]), ("warming", None))
        self.assertEqual(evidence["observations"]["sample_count"], 1)
        self.assertIn("demo", " ".join(evidence["limitations"]))
        missing = self.call("stream_evidence", session=str(uuid4()))
        self.assertEqual((missing["state"], missing["observations"], missing["forecast"]), ("idle", None, None))
        for window_ms in (True, 99, 60_001):
            with self.subTest(window_ms=window_ms), self.assertRaisesRegex(ValueError, "lookback"):
                self.call("stream_evidence", window_ms=window_ms)

    def test_stream_hardware_failure_blocks_retry_and_queries_until_recovery(self):
        request = dict(session=str(uuid4()), source="Test", capacity=10)
        rows = [{"time_ms": 100 * i, "busy": 5, "waiting": 0} for i in range(50)]
        with patch.object(self.core, "submit", side_effect=OSError("FPGA unavailable")):
            with self.assertRaisesRegex(OSError, "FPGA unavailable"):
                self.device.ingest(dict(request, readings=rows))
        with self.assertRaisesRegex(RuntimeError, "requires recovery"):
            self.device.ingest(dict(request, readings=rows))
        with self.assertRaisesRegex(RuntimeError, "requires recovery"):
            self.call("stream_forecast")

    def test_stream_live_source_takes_priority_and_stop_is_scoped(self):
        self.device.stream.clock = lambda: 10
        reading = [dict(time_ms=0, busy=5, waiting=0)]
        demo = dict(session=str(uuid4()), source="Demo", capacity=10, origin="demo", readings=reading)
        live = dict(session=str(uuid4()), source="Live", capacity=10, readings=reading)
        self.device.ingest(demo)
        self.device.ingest(live)
        self.device.ingest(dict(session=demo["session"], stop=True))
        self.assertEqual(self.call("stream_status")["source"], "Live")
        with self.assertRaisesRegex(ValueError, "has ended"):
            self.device.ingest(demo)
        with self.assertRaisesRegex(ValueError, "active"):
            self.device.ingest(dict(demo, session=str(uuid4())))

        self.device.ingest(dict(session=live["session"], stop=True))
        self.assertEqual(self.call("stream_status")["state"], "idle")
        self.assertEqual(self.device.ingest(dict(demo, session=str(uuid4())))["origin"], "demo")


if __name__ == "__main__":
    unittest.main()
