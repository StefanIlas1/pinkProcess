import json
from threading import Thread
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

from .kit_app import Kit, make_server
from .kit_dataset import DEMO_CAPACITY, demo_csv, prepare_dataset


class MockBoard:
    def __init__(self):
        self.version = "base"
        self.batches = []
        self.change_midway = False
        self.evidence_supported = False
        self.stream = {}
        self.stream_by_action = {}
        self.stream_requests = []

    def close(self):
        pass

    def discover(self):
        return [{"host": "pynq.local", "name": "Test PYNQ-Z2"}]

    def connect(self, host):
        return self.request("status")

    def request(self, action, **params):
        if action.startswith("stream_"):
            self.stream_requests.append((action, params))
            return dict(self.stream_by_action.get(action, self.stream))
        if action == "apply":
            self.version = params["version"]
        elif action == "rollback":
            self.version = "base"
        result = {}
        if action == "status" and self.evidence_supported:
            result["stream_evidence_supported"] = True
        if action == "query_batch":
            batch = params["features"]
            self.batches.append(len(batch))
            if self.change_midway and len(self.batches) > 1:
                self.version = "other"
            scores = [sum(vector) for vector in batch]
            result.update(scores=scores, alerts=[s >= 200 for s in scores])
        return dict(ok=True, backend="PYNQ-Z2", model="test", version=self.version,
                    versions=["base", "other"], weights_sha256=self.version,
                    threshold=200, score_scale=65536, **result)


class KitTest(unittest.TestCase):
    def setUp(self):
        self.transport = MockBoard()
        self.kit = Kit(self.transport)

    def test_language_model_version_commands_are_forwarded(self):
        self.kit.action({"action": "connect", "host": "pynq.local"})
        active = {"name": "original", "checkpoint_sha256": "0" * 64}
        with patch.object(self.transport, "request", return_value={"ok": True, "active": active}) as send:
            for action, fields in (
                ("lm_versions", {}),
                ("lm_save", {"name": "Astra", "target_row": 12, "response": {"row": [1, 2]}}),
                ("lm_select", {"name": "Astra"}),
                ("lm_restore", {}),
            ):
                with self.subTest(action=action):
                    result = self.kit.action({"action": action, **fields})
                    send.assert_called_with(action, **fields)
                    self.assertEqual(result["host"], "pynq.local")
                    self.assertEqual(result["active"], active)
            with self.assertRaisesRegex(ValueError, "Invalid language-model version request"):
                self.kit.action({"action": "lm_select", "name": "Astra", "weights": ...})
            self.assertEqual(send.call_count, 4)

    def test_saved_capture_pages_are_forwarded(self):
        self.kit.action({"action": "connect", "host": "pynq.local"})
        with patch.object(self.transport, "request", return_value={"ok": True, "captures": [], "before": None}) as send:
            result = self.kit.action({"action": "lm_captures"})
            self.assertEqual(result, {"ok": True, "captures": [], "before": None, "host": "pynq.local"})
            send.assert_called_once_with("lm_captures")
            self.kit.action({"action": "lm_captures", "before": 4})
            send.assert_called_with("lm_captures", before=4)
            with self.assertRaisesRegex(ValueError, "Invalid saved-capture listing request"):
                self.kit.action({"action": "lm_captures", "question": "private"})

    def test_select_then_query_latest(self):
        self.assertIsNone(self.kit.action({"action": "status"})["host"])
        data = dict(action="query", question="latest", csv=demo_csv(), capacity=DEMO_CAPACITY)
        with self.assertRaisesRegex(ValueError, "Connect a board"):
            self.kit.action(data)
        address = self.kit.action({"action": "discover"})["boards"][0]["host"]
        self.kit.action({"action": "connect", "host": address})
        result = self.kit.action(data)
        expected = prepare_dataset(data["csv"], data["capacity"])
        self.assertEqual(self.transport.batches, [1])
        self.assertEqual(result["rows"][0]["time_ms"], expected["times_ms"][-1])
        self.assertEqual(result["rows"][0]["score"], sum(expected["features"][-1]))
        self.assertEqual(self.kit.action({"action": "apply", "version": "other"})["status"]["version"], "other")
        self.assertEqual(self.kit.action({"action": "rollback"})["status"]["version"], "base")

    def test_batch_identity(self):
        self.kit.action({"action": "connect", "host": "pynq.local"})
        # 49 warmup readings plus 513 eligible windows exercise the batch boundary.
        csv_text = "time_ms,busy,waiting\n" + "".join(f"{i * 100},9,0\n" for i in range(562))
        data = dict(action="query", question="warnings", csv=csv_text, capacity=DEMO_CAPACITY)
        result = self.kit.action(data)
        self.assertEqual(self.transport.batches, [512, 1])
        self.assertEqual(result["windows"], 513)
        self.assertEqual(result["rows"][-1]["time_ms"], 56100)
        self.transport.batches.clear()
        self.transport.change_midway = True
        with self.assertRaisesRegex(RuntimeError, "model changed"):
            self.kit.action(data)

    def test_live_questions_request_only_board_owned_state(self):
        self.kit.action({"action": "connect", "host": "pynq.local"})
        first = {"time_ms": 2000, "busy": 9, "waiting": 1}
        latest = {"time_ms": 11900, "busy": 10, "waiting": 3}
        peak = {"time_ms": 10400, "busy": 10, "waiting": 4}
        self.transport.stream = dict(
            state="ready", fresh=True, capacity=10, latest=latest,
            observations=dict(sample_count=100, start_ms=2000, end_ms=11900,
                              first=first, latest=latest, peak_busy=latest, peak_waiting=peak),
            forecast=dict(model="queue_outlook", scores=[1, -1]),
        )
        cases = (
            ("How many requests are waiting now?", "Waiting requests: 3.",
             ("stream_status", {})),
            ("What was the peak queue in the last 30 seconds?", "peaked at 4 at 10.4 s",
             ("stream_observations", {"window_ms": 30000})),
            ("Is the queue growing over the last 10 seconds?", "rose from 1 to 3",
             ("stream_observations", {"window_ms": 10000})),
            ("What does the FPGA predict for the next ten seconds?", "remain nonempty",
             ("stream_forecast", {})),
        )
        for question, answer, action in cases:
            with self.subTest(question=question):
                result = self.kit.action({"action": "ask_live", "question": question})
                self.assertIn(answer, result["answer"])
                self.assertEqual(self.transport.stream_requests[-1], action)
        self.assertEqual(len(self.transport.stream_requests), 4)
        self.assertEqual(self.transport.batches, [])

    def test_open_ended_question_uses_atomic_board_evidence(self):
        self.transport.evidence_supported = True
        self.kit.action({"action": "connect", "host": "pynq.local"})
        question = "Why are requests backing up, and what should I check next?"
        latest = dict(time_ms=11900, busy=10, waiting=3)
        self.transport.stream = dict(state="ready", fresh=True, session="test-session",
                                     source="Test pool", origin="live", latest=latest, capacity=10,
                                     observations=dict(waiting=dict(mean=1.5), peak_waiting=latest),
                                     forecast=dict(model="queue_outlook", version="test-v1",
                                                   time_ms=11900, horizon_ms=10000, scores=[1, -1]))
        answer = "Three requests are waiting and all 10 connections are busy. Check connection hold times."
        with patch("pinkprocess.kit_app.analyze_live_evidence",
                   return_value=dict(answer=answer, model="local-test", errors=[])) as analyze:
            reply = self.kit.action(dict(action="ask_live", question=question))
        self.assertEqual(self.transport.stream_requests, [("stream_evidence", {"window_ms": 60000})])
        sent_question, bundle = analyze.call_args.args
        self.assertEqual(sent_question, question)
        self.assertEqual(bundle["latest"], latest)
        self.assertEqual(bundle["forecast"], dict(version="test-v1", time_ms=11900, horizon_ms=10000))
        self.assertIn("remain nonempty", bundle["forecast_interpretation"])
        self.assertIn("100.0%", bundle["observed_now"])
        self.assertIn("3 requests", bundle["observed_peak"])
        self.assertEqual(bundle["observations"]["waiting"]["mean"], 1.5)
        self.assertEqual(reply["snapshot"]["forecast"]["scores"], [1, -1])
        self.assertEqual((reply["answer"], reply["analyst_model"], reply["analyst_location"]),
                         (answer, "local-test", "Mac"))
        self.assertEqual(self.transport.batches, [])

    def test_legacy_evidence_preserves_observation_and_forecast_timestamps(self):
        self.kit.action({"action": "connect", "host": "pynq.local"})
        latest = dict(time_ms=11900, busy=10, waiting=3)
        forecast_reading = dict(time_ms=12000, busy=10, waiting=4)
        self.transport.stream_by_action = {
            "stream_observations": dict(state="ready", fresh=True, session="old-session", latest=latest),
            "stream_forecast": dict(state="ready", fresh=True, session="old-session", latest=forecast_reading,
                                    forecast=dict(model="queue_outlook", version="old-v1", time_ms=12000,
                                                  horizon_ms=10000, scores=[1, 1])),
        }
        question = "How do the forecast and observations compare?"
        with patch("pinkprocess.kit_app.analyze_live_evidence",
                   return_value=dict(answer="Based on the captured readings.", model="local-test", errors=[])) as analyze:
            self.kit.action(dict(action="ask_live", question=question))
            sent_question, bundle = analyze.call_args.args
            self.assertEqual(sent_question, question)
            self.assertEqual(bundle["latest"], latest)
            self.assertEqual(bundle["forecast_reading"], forecast_reading)
            self.assertEqual(bundle["forecast"]["time_ms"], 12000)
            self.assertIn("current 4 waiting", bundle["forecast_interpretation"])
            self.assertEqual(self.transport.stream_requests,
                             [("stream_observations", {"window_ms": 60000}),
                              ("stream_forecast", {"session": "old-session"})])
            analyze.reset_mock()
            self.transport.stream_by_action["stream_forecast"]["session"] = "another-session"
            with self.assertRaisesRegex(ValueError, "source changed"):
                self.kit.action(dict(action="ask_live", question=question))
            analyze.assert_not_called()

    def test_open_ended_question_handles_unavailable_data_and_rejects_extra_input(self):
        self.transport.evidence_supported = True
        self.kit.action({"action": "connect", "host": "pynq.local"})
        ask = {"action": "ask_live", "question": "What looks wrong with the pool?"}
        with patch("pinkprocess.kit_app.analyze_live_evidence") as analyze:
            self.transport.stream = dict(state="idle", fresh=False, latest=None)
            self.assertIn("no telemetry", self.kit.action(ask)["answer"])
            self.transport.stream = dict(state="stale", fresh=False, latest={"busy": 9, "waiting": 0})
            self.assertIn("telemetry is stale", self.kit.action(ask)["answer"])
            analyze.assert_not_called()
        with self.assertRaisesRegex(ValueError, "only question text"):
            self.kit.action({**ask, "csv": "time_ms,busy,waiting"})
        for question in (None, "  ", "a" * 501):
            with self.subTest(question=question), self.assertRaisesRegex(ValueError, "500 characters"):
                self.kit.action({**ask, "question": question})
        self.assertEqual(len(self.transport.stream_requests), 2)

    def test_unrelated_question_is_deferred_to_analyst(self):
        self.transport.evidence_supported = True
        self.kit.action({"action": "connect", "host": "pynq.local"})
        self.transport.stream = dict(state="warming", fresh=True, latest={"busy": 9, "waiting": 0})
        answer = "The connection-pool data cannot forecast the weather."
        with patch("pinkprocess.kit_app.analyze_live_evidence",
                   return_value=dict(answer=answer, model="local-test", errors=[])) as analyze:
            self.assertEqual(self.kit.action(dict(action="ask_live", question="Forecast the weather"))["answer"],
                             answer)
        self.assertEqual(analyze.call_args.args[0], "Forecast the weather")

    def test_http_origin_and_request(self):
        with make_server() as server:
            server.kit = self.kit
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.addCleanup(thread.join)
            origin = f"http://127.0.0.1:{server.server_port}"
            headers = {"Content-Type": "application/json"}
            data = json.dumps({"action": "discover"}).encode()
            try:
                with urlopen(origin + "/api/setup") as response:
                    self.assertEqual(json.load(response)["csv"], demo_csv())
                with urlopen(origin + "/api/live-setup") as response:
                    self.assertEqual(set(json.load(response)), {"examples"})
                with self.assertRaises(HTTPError) as failure:
                    urlopen(Request(origin + "/api/action", data, headers))
                self.assertEqual(failure.exception.code, 403)
                failure.exception.close()
                headers["Origin"] = origin
                with urlopen(Request(origin + "/api/action", data, headers)) as response:
                    self.assertEqual(json.load(response)["boards"], self.transport.discover())
            finally:
                server.shutdown()


if __name__ == "__main__":
    unittest.main()
