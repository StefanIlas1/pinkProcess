"""Exercise producer and reader over loopback HTTPS with a simulated FPGA."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from http.client import HTTPSConnection
import json
from pathlib import Path
import shutil
import ssl
import subprocess
from tempfile import TemporaryDirectory
from threading import Lock, Thread
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch
from uuid import uuid4

from pinkprocess.backlog import VERSION as BACKLOG_VERSION, features as backlog_features
from pinkprocess.dataset import Example
from pinkprocess.fixed_point import encode_features
from pinkprocess.kit_app import Kit, make_server
from pinkprocess.kit_server import Handler, KitServer
from pinkprocess.kit_transport import BoardTransport
from pinkprocess.simulator import Reading
from pinkprocess.telemetry_device import BASE, VERSIONS, TelemetryDevice
from pinkprocess.test_telemetry_device import DotProduct, PROFILES


class CountingClassifier(DotProduct):
    def __init__(self):
        super().__init__()
        self.submissions = []

    def submit(self, activation):
        super().submit(activation)
        self.submissions.append(activation)


class QuietHandler(Handler):
    def log_message(self, *_):
        pass


@contextmanager
def serving(server):
    with server:
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        try:
            yield server
        finally:
            server.shutdown()
            thread.join()


@unittest.skipUnless(shutil.which("openssl"), "OpenSSL is required for the loopback TLS fixture")
class LiveIntegrationTest(unittest.TestCase):
    host = "127.0.0.1"
    token = "a" * 64
    capacity = 16

    @classmethod
    def setUpClass(cls):
        temporary = TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        key, certificate = directory / "board.key", directory / "board.crt"
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                        "-keyout", str(key), "-out", str(certificate), "-days", "1",
                        "-subj", "/CN=localhost"], check=True, capture_output=True)
        cls.tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        cls.tls.load_cert_chain(certificate, key)
        cls.client_tls = ssl.create_default_context(cadata=certificate.read_text())
        cls.client_tls.check_hostname = False
        cls.pairing_dir = directory
        (directory / "test-pairing.json").write_text(json.dumps(
            {"host": cls.host, "token": cls.token, "certificate": certificate.read_text()}))

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.state_file = Path(temporary.name) / "active.json"
        self.classifier = CountingClassifier()
        self.session = str(uuid4())
        self.device = TelemetryDevice(self.classifier, PROFILES, self.state_file)
        self.now = 100
        self.device.stream.clock = lambda: self.now
        self.server = KitServer((self.host, 0), QuietHandler)
        self.server.tls = self.tls
        self.server.authorization = ("Bearer " + self.token).encode()
        self.server.model = self.device
        self.server.model_lock = Lock()
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(serving(self.server))

    def exchange(self, path, payload, *, token=None):
        token = self.token if token is None else token
        connection = HTTPSConnection(self.host, self.server.server_port,
                                     timeout=2, context=self.client_tls)
        try:
            connection.request("POST", path, json.dumps(payload), {
                "Content-Type": "application/json", "Authorization": "Bearer " + token})
            response = connection.getresponse()
            return response.status, json.load(response)
        finally:
            connection.close()

    def feed(self, readings, *, token=None, **kwargs):
        payload = {"session": self.session, "source": "loopback-pool", "capacity": self.capacity,
                   "readings": readings}
        payload.update(kwargs)
        return self.exchange("/v1/telemetry", payload, token=token)

    def query(self, op, **kwargs):
        code, result = self.exchange("/v1/query", {"op": op, **kwargs})
        self.assertEqual(code, 200, result)
        return result

    def readings(self, start, stop, *, queue=False):
        return [{"time_ms": 100 * i, "busy": 5 + i % 8, "waiting": i - start + 1 if queue else 0}
                for i in range(start, stop)]

    def accepted(self, readings, **kwargs):
        code, result = self.feed(readings, **kwargs)
        self.assertEqual(code, 200, result)
        return result

    def test_ingestion_scores_both_queue_states_without_reader(self):
        onset = self.readings(0, 50)
        self.assertEqual(self.accepted(onset[:49])["accepted"], 49)
        self.assertEqual(len(self.classifier.submissions), 0)
        self.assertEqual(self.accepted(onset[49:])["accepted"], 1)
        self.assertEqual(len(self.classifier.submissions), 1)

        answer = self.query("stream_forecast")
        self.assertEqual(answer["state"], "ready")
        self.assertEqual(answer["sample_count"], 50)
        self.assertEqual(answer["session"], self.session)
        self.assertEqual(answer["latest"], onset[-1])
        self.assertTrue(answer["fresh"])
        forecast = answer["forecast"]
        window = tuple(Reading(**row) for row in onset)
        vector = encode_features(Example(window[-1].time_ms, window, None), self.capacity)
        model = self.device.models[BASE]
        self.assertEqual(forecast["version"], BASE)
        self.assertEqual(forecast["weights_sha256"], model.digest)
        self.assertEqual(forecast["time_ms"], 4900)
        self.assertEqual(forecast["score"], model.profile.score(vector))
        self.assertEqual(forecast["alert"], forecast["score"] >= model.profile.threshold)
        self.assertEqual(self.query("stream_forecast")["forecast"], forecast)
        self.assertEqual(len(self.classifier.submissions), 1)

        occupied = self.readings(50, 51, queue=True)
        self.accepted(occupied)
        self.assertEqual(len(self.classifier.submissions), 2)
        answer = self.query("stream_forecast")
        forecast = answer["forecast"]
        self.assertEqual(answer["latest"], occupied[-1])
        self.assertEqual(forecast["time_ms"], 5000)
        self.assertEqual(forecast["version"], BACKLOG_VERSION)
        self.assertEqual(forecast["weights_sha256"], self.device.backlog_hash)
        vector = backlog_features(window[1:] + (Reading(**occupied[-1]),), self.capacity)
        expected = [sum(a * b for a, b in zip(vector, row)) for row in self.device.backlog_rows]
        self.assertEqual(forecast["scores"], expected)
        self.assertEqual(len(self.classifier.submissions), 2)

    def test_invalid_batches_and_unauthenticated_reads_preserve_completed_window(self):
        self.accepted(self.readings(0, 50))
        before = self.query("stream_forecast")["forecast"]
        correct = self.readings(50, 52)
        malformed = [correct[0], {**correct[1], "time_ms": correct[0]["time_ms"]}]
        out_of_range = [correct[0], {**correct[1], "busy": self.capacity + 1}]
        for status, body in (
            self.feed(malformed),
            self.feed(out_of_range),
            self.feed(correct, capacity=self.capacity + 1),
        ):
            self.assertEqual(status, 422, body)
        self.assertEqual(self.feed(correct, token="wrong")[0], 403)
        self.assertEqual(self.exchange("/v1/query", {"op": "stream_forecast"}, token="wrong")[0], 403)
        unchanged = self.query("stream_forecast")
        self.assertEqual(unchanged["sample_count"], 50)
        self.assertEqual(unchanged["forecast"], before)
        self.assertEqual(len(self.classifier.submissions), 1)
        self.assertEqual(self.accepted(correct)["accepted"], 2)
        self.assertEqual(self.query("stream_forecast")["forecast"]["time_ms"], 5100)

    def test_retry_and_stale_takeover_preserve_stream_ownership(self):
        initial = self.readings(0, 50)
        next_session = str(uuid4())
        self.accepted(initial)
        self.now += 1
        self.assertEqual(self.accepted(initial[47:])["accepted"], 0)
        self.assertEqual(self.query("stream_status")["age_ms"], 1000)
        self.assertEqual(self.feed(self.readings(50, 51), session=next_session)[0], 422)
        conflicting = [{**initial[-1], "waiting": 1}, *self.readings(50, 51)]
        self.assertEqual(self.feed(conflicting)[0], 422)
        self.assertEqual(len(self.classifier.submissions), 1)
        overlap = initial[47:] + self.readings(50, 51)
        self.assertEqual(self.accepted(overlap)["accepted"], 1)
        self.assertEqual(self.query("stream_forecast")["forecast"]["time_ms"], 5000)

        self.now += 2
        stale = self.query("stream_forecast")
        self.assertEqual(stale["state"], "stale")
        self.assertIsNone(stale["forecast"])
        self.assertEqual(self.accepted(overlap)["accepted"], 0)
        self.assertEqual(self.query("stream_status")["state"], "stale")
        self.assertEqual(len(self.classifier.submissions), 2)

        takeover = self.accepted(self.readings(0, 49), session=next_session, capacity=self.capacity + 1)
        self.assertEqual(takeover["state"], "warming")
        self.assertEqual(takeover["sample_count"], 49)
        self.assertEqual(self.query("stream_status", session=self.session)["state"], "idle")
        self.assertEqual(self.feed(self.readings(51, 52))[0], 422)
        self.assertEqual(self.accepted(self.readings(49, 50), session=next_session,
                                       capacity=self.capacity + 1)["state"], "ready")
        self.assertEqual(len(self.classifier.submissions), 3)
        status, stopped = self.exchange("/v1/telemetry", {"session": next_session, "stop": True})
        self.assertEqual(status, 200)
        self.assertEqual(stopped["state"], "idle")

    def test_cadence_gap_requires_new_continuous_window(self):
        self.accepted(self.readings(0, 50))
        self.assertEqual(len(self.classifier.submissions), 1)
        self.accepted(self.readings(51, 100))
        warming = self.query("stream_forecast")
        self.assertEqual(warming["state"], "warming")
        self.assertIsNone(warming["forecast"])
        self.assertEqual(warming["sample_count"], 49)
        self.assertEqual(len(self.classifier.submissions), 1)
        self.accepted(self.readings(100, 101))
        self.assertEqual(self.query("stream_forecast")["forecast"]["time_ms"], 10_000)
        self.assertEqual(len(self.classifier.submissions), 2)

    def test_classifier_failure_requires_restart_and_warmup(self):
        self.query("apply", version=VERSIONS[1])
        self.accepted(self.readings(0, 50))
        with patch.object(self.classifier, "submit", side_effect=OSError("simulated MMIO failure")):
            self.assertEqual(self.feed(self.readings(50, 51))[0], 503)
        self.assertEqual(self.exchange("/v1/query", {"op": "stream_forecast"})[0], 503)
        self.assertEqual(self.feed(self.readings(50, 51))[0], 503)

        restarted = TelemetryDevice(self.classifier, PROFILES, self.state_file)
        restarted.stream.clock = lambda: self.now
        with self.server.model_lock:
            self.server.model = restarted
        self.assertEqual(self.query("stream_status")["state"], "idle")
        self.assertEqual(self.query("status")["version"], VERSIONS[1])
        self.accepted(self.readings(51, 100))
        self.assertEqual(self.query("stream_forecast")["state"], "warming")
        self.accepted(self.readings(100, 101))
        self.assertEqual(self.query("stream_forecast")["forecast"]["version"], VERSIONS[1])
        self.assertEqual(len(self.classifier.submissions), 2)

    def test_profile_switch_recomputes_on_ingress_only(self):
        readings = self.readings(0, 60)
        self.accepted(readings[:50])
        forecast = self.query("stream_forecast")["forecast"]
        self.query("apply", version=VERSIONS[1])
        self.assertEqual(len(self.classifier.submissions), 1)
        self.assertIsNone(self.query("stream_forecast")["forecast"])

        def read_many():
            return [self.query("stream_forecast") for _ in range(12)]

        with ThreadPoolExecutor(max_workers=2) as pool:
            concurrent_reader = pool.submit(read_many)
            for index in range(50, len(readings)):
                self.accepted(readings[index:index + 1])
            snapshots = concurrent_reader.result(timeout=5)

        for snapshot in snapshots:
            candidate = snapshot["forecast"]
            if candidate is None:
                self.assertEqual(snapshot["state"], "warming")
                continue
            self.assertEqual(candidate["version"], VERSIONS[1])
            model = self.device.models[VERSIONS[1]]
            self.assertEqual(candidate["weights_sha256"], model.digest)
            index = candidate["time_ms"] // 100
            self.assertEqual(snapshot["latest"], readings[index])
            window = tuple(Reading(**row) for row in readings[index - 49:index + 1])
            vector = encode_features(Example(window[-1].time_ms, window, None), self.capacity)
            self.assertEqual(candidate["score"], model.profile.score(vector))
        self.assertEqual(len(self.classifier.submissions), 11)
        current = self.query("stream_forecast")["forecast"]
        self.assertEqual(current["version"], VERSIONS[1])
        self.assertNotEqual(current["weights_sha256"], forecast["weights_sha256"])
        self.assertEqual(current["time_ms"], 5900)

    def test_native_backend_queries_board_feed_without_uploading_readings(self):
        self.accepted(self.readings(0, 50, queue=True))
        self.assertEqual(len(self.classifier.submissions), 1)
        mac = make_server()
        mac.kit = Kit(BoardTransport(self.pairing_dir))
        with serving(mac), patch("pinkprocess.kit_transport.PORT", self.server.server_port):
            origin = f"http://{self.host}:{mac.server_port}"

            def action(**payload):
                request = Request(origin + "/api/action", json.dumps(payload).encode(),
                                  {"Content-Type": "application/json", "Origin": origin})
                with urlopen(request, timeout=2) as response:
                    return json.load(response)

            connection = action(action="connect", host=self.host)
            self.assertTrue(connection["status"]["stream_evidence_supported"])
            self.assertEqual(action(action="stream_status")["snapshot"]["session"], self.session)
            for question, text in (
                ("How many requests are waiting now?", "Waiting requests: 50."),
                ("What was the peak queue in the last 30 seconds?", "peaked at 50"),
                ("Is the queue growing over the last 10 seconds?", "rose from 1 to 50"),
                ("What does the FPGA predict for the next ten seconds?", "The FPGA predicts"),
            ):
                with self.subTest(question=question):
                    reply = action(action="ask_live", question=question)
                    self.assertIn(text, reply["answer"])
                    self.assertEqual(reply["snapshot"]["session"], self.session)
            question = "What seems wrong with this pool and what should I investigate next?"
            answer = "The observed queue grew. Check how long requests hold connections."
            with patch("pinkprocess.kit_app.analyze_live_evidence", return_value={
                "answer": answer, "model": "local-test", "errors": [],
            }) as analyst:
                reply = action(action="ask_live", question=question)
            sent_question, model_evidence = analyst.call_args.args
            self.assertEqual(sent_question, question)
            self.assertEqual(model_evidence["latest"]["waiting"], 50)
            self.assertEqual(model_evidence["forecast"], dict(version=BACKLOG_VERSION, time_ms=4900,
                                                              horizon_ms=10000))
            self.assertIn("50 requests", model_evidence["observed_peak"])
            self.assertIn("The FPGA predicts", model_evidence["forecast_interpretation"])
            self.assertEqual(reply["answer"], answer)
            self.assertEqual(reply["analyst_model"], "local-test")
            evidence = reply["snapshot"]
            self.assertEqual((evidence["session"], evidence["source"]), (self.session, "loopback-pool"))
            self.assertEqual(evidence["observations"]["waiting"]["change"], 49)
            self.assertEqual(evidence["observations"]["end_ms"], evidence["forecast"]["time_ms"])
            self.assertEqual(evidence["forecast"]["version"], BACKLOG_VERSION)
            self.assertEqual(len(evidence["forecast"]["scores"]), 2)
            self.assertEqual(self.exchange("/v1/query", {"op": "stream_evidence"}, token="wrong")[0], 403)
            with self.assertRaises(HTTPError) as rejected:
                action(action="ask_live", question="How many are waiting now?", readings=[])
            self.assertEqual(rejected.exception.code, 400)
            rejected.exception.close()
        self.assertEqual(len(self.classifier.submissions), 1)


if __name__ == "__main__":
    unittest.main()
