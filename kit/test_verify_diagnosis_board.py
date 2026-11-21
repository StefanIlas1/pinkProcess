"""Exercise the paired diagnosis verifier and protected ingress without a physical kit."""

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from uuid import uuid4

from kit.verify_diagnosis_board import REFERENCE, verify
from lm_reference.flan_prompt import telemetry_prompt
from pinkprocess.kit_evidence import analyst_evidence
from pinkprocess.telemetry_device import TelemetryDevice
from pinkprocess.telemetry_stream import BOARD_DEMO_SOURCE, FIXTURE_SOURCE_PREFIX
from pinkprocess.test_telemetry_device import DotProduct, PROFILES


class PairedBoard:
    def __init__(self, directory, reference, *, starts_with_demo=True):
        self.reference = reference
        self.device = TelemetryDevice(DotProduct(), PROFILES, directory / "state.json")
        self.time = 0
        self.device.stream.clock = lambda: self.time
        self.active = dict(name="original", checkpoint_sha256=reference["weights_sha256"],
                           classifier_sha256="1" * 64, target_row=None)
        self.boot = str(uuid4())
        self.stops = []
        self.publishes = []
        self.starts_with_demo = starts_with_demo
        self.legacy = False
        self.capture_error = False
        self.corrupt_capture = False
        self.insert_real_at_ingress = False
        self.bad_answer = False
        self.no_projections = False
        self.job = None
        if starts_with_demo:
            self.start_demo()

    def start_demo(self):
        self.device.ingest(dict(session=str(uuid4()), source=BOARD_DEMO_SOURCE, origin="demo", capacity=4,
                                readings=[dict(time_ms=i * 100, busy=1, waiting=0) for i in range(50)]))

    def start_real(self):
        self.real_session = str(uuid4())
        self.device.ingest(dict(session=self.real_session, source="Production database", origin="live",
                                capacity=4, readings=[dict(time_ms=0, busy=2, waiting=0)]))

    def send(self, **params):
        if params.get("stop"):
            self.stops.append(params["session"])
        else:
            self.publishes.append(params)
            if self.insert_real_at_ingress:
                self.start_real()
                self.time += 10
        return self.device.ingest(params)

    def request(self, op, **params):
        if op == "lm_versions":
            return dict(boot_id=self.boot, active=self.active, versions=[self.active])
        if op == "lm_start":
            if self.capture_error:
                raise RuntimeError("Injected capture error")
            snapshot = self.device.handle(dict(op="stream_evidence", window_ms=60_000))
            case = next(item for item in self.reference["cases"] if snapshot["source"].endswith(item["case"]))
            capture = dict(question=params["question"], captured_at_ms=1234, snapshot=snapshot,
                           prompt=telemetry_prompt(params["question"], analyst_evidence(snapshot)))
            digest = sha256(json.dumps(capture, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if self.corrupt_capture:
                digest = "0" * 64
            self.job = dict(job_id=str(uuid4()), capture_id=digest, active=self.active, captured_at_ms=1234,
                            case=case)
            return {**self.job, "snapshot": snapshot}
        if op == "lm_result":
            # Generation must be allowed to finish off the captured evidence after releasing the fixture.
            assert self.stops[-1] == self.publishes[-1]["session"]
            return dict(**self.job, state="complete",
                        answer="Changed answer." if self.bad_answer else self.job["case"]["answer"],
                        fabric_projections=0 if self.no_projections else 256)
        if op == "stream_status":
            if self.starts_with_demo and self.device.stream.active is None:
                self.start_demo()
            status = self.device.handle(dict(op=op))
            if self.legacy:
                status.pop("fixture_guard_supported")
            return status
        raise AssertionError(op)

    def status(self):
        return {**self.device.handle(dict(op="status")), "lm_supported": True}


class PairedDiagnosisTest(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.directory = Path(tmp.name)
        self.reference = json.loads(REFERENCE.read_bytes())
        self.board = PairedBoard(self.directory, self.reference)
        self.report = {"passed": False}

    def run_verify(self):
        verify(self.board, self.board.send, self.board.status(), self.reference, self.report,
               clock=lambda: self.board.time, wait=self.advance)

    def advance(self, seconds):
        self.board.time += seconds

    def test_pass_matches_three_frozen_answers_and_resumes_demo(self):
        self.run_verify()
        self.assertTrue(self.report["passed"])
        self.assertEqual(len(self.report["cases"]), 3)
        self.assertEqual(self.board.stops, [payload["session"] for payload in self.board.publishes])
        for case in self.report["cases"]:
            self.assertTrue(case["matches_reference"])
            self.assertGreater(case["result"]["fabric_projections"], 0)
            self.assertEqual(case["producer_after_release"]["source"], BOARD_DEMO_SOURCE)

    def test_idle_board_remains_idle(self):
        self.board = PairedBoard(self.directory, self.reference, starts_with_demo=False)
        self.run_verify()
        self.assertTrue(self.report["passed"])
        self.assertEqual(self.board.request("stream_status")["state"], "idle")

    def test_active_real_producer_and_unguarded_board_are_untouched(self):
        self.board.legacy = True
        with self.assertRaisesRegex(RuntimeError, "guarded synthetic telemetry"):
            self.run_verify()
        self.assertFalse(self.board.publishes)
        self.board.legacy = False
        self.board.start_real()
        self.board.time += 10
        with self.assertRaisesRegex(RuntimeError, "non-demo producer"):
            self.run_verify()
        self.assertEqual(self.board.request("stream_status")["session"], self.board.real_session)
        self.assertFalse(self.board.publishes)
        self.assertFalse(self.board.stops)

    def test_real_producer_racing_ingress_is_safe_even_after_it_stales(self):
        self.board.insert_real_at_ingress = True
        with self.assertRaisesRegex(ValueError, "no longer owns"):
            self.run_verify()
        self.assertEqual(self.board.request("stream_status")["session"], self.board.real_session)
        self.assertEqual(self.board.stops, [self.board.publishes[0]["session"]])

    def test_capture_error_releases_only_own_fixture_and_demo_resumes(self):
        self.board.capture_error = True
        with self.assertRaisesRegex(RuntimeError, "Injected capture error"):
            self.run_verify()
        self.assertEqual(self.board.request("stream_status")["source"], BOARD_DEMO_SOURCE)
        self.assertEqual(self.board.stops, [self.board.publishes[0]["session"]])

    def test_bad_capture_digest_fails_and_releases_the_fixture(self):
        self.board.corrupt_capture = True
        with self.assertRaisesRegex(RuntimeError, "capture ID failed SHA-256"):
            self.run_verify()
        self.assertEqual(self.board.stops, [self.board.publishes[0]["session"]])

    def test_wrong_answer_and_missing_fabric_proof_fail(self):
        self.board.bad_answer = True
        with self.assertRaisesRegex(RuntimeError, "differs from the frozen"):
            self.run_verify()
        self.assertEqual([case["matches_reference"] for case in self.report["cases"]], [False] * 3)
        self.board.bad_answer = False
        self.board.no_projections = True
        with self.assertRaisesRegex(RuntimeError, "PL projection count"):
            self.run_verify()
        self.assertEqual(self.board.request("stream_status")["source"], BOARD_DEMO_SOURCE)

    def test_synthetic_fixture_without_guard_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "guarded demo source"):
            self.board.device.ingest(dict(session=str(uuid4()), source=FIXTURE_SOURCE_PREFIX + " test",
                                          origin="demo", capacity=4, readings=[dict(time_ms=0, busy=0, waiting=0)]))
        self.assertEqual(self.board.request("stream_status")["source"], BOARD_DEMO_SOURCE)


if __name__ == "__main__":
    unittest.main()
