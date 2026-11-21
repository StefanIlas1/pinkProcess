"""Exercise the physical acceptance state machine against a protocol double."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from uuid import uuid4

from kit.verify_language_board import Verification


class Board:
    def __init__(self):
        self.base = dict(name="original", checkpoint_sha256="a", classifier_sha256="b", target_row=None)
        self.active = self.base
        self.versions = [self.base]
        self.boot = str(uuid4())
        self.job = None
        self.iteration = 0
        self.readings = 0

    def request(self, op, **params):
        if op == "lm_start":
            self.iteration = 0
            self.job = dict(state="running", job_id=str(uuid4()), capture_id="evidence", active=self.active)
            return {**self.job, "snapshot": {"latest": {"waiting": 4}}}
        if op == "lm_result":
            self.iteration += 1
            if self.iteration <= 2:
                return self.job
            prefix = "Now " if self.active != self.base else ""
            return {**self.job, "state": "complete", "answer": prefix + "4 requests are waiting.", "fabric_projections": 314}
        if op == "stream_status":
            self.readings += 1
            return dict(session="live", fresh=True, latest_time_ms=self.readings, fpga_submissions=self.readings)
        if op == "lm_row":
            return dict(token="Now", row=[3], target_row=15, activation=[2], base_sha256="a",
                        max_score=5, row_scale=1, calibration_job_id=self.job["job_id"])
        if op == "lm_save":
            self.versions = [self.base, dict(name=params["name"], checkpoint_sha256="c",
                                             classifier_sha256="d", target_row=params["target_row"])]
        if op == "lm_select":
            self.active = self.versions[1]
        if op == "lm_restore":
            self.active = self.base
        return dict(boot_id=self.boot, versions=self.versions, active=self.active)


class AcceptanceTest(unittest.TestCase):
    def test_two_phase_power_cycle_and_exact_replay(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.json"
            board = Board()
            run = Verification(board, path, interval=0, author=lambda instruction, context: {"row": [4]})
            run.before({"backend": "PYNQ-Z2"}, "How did the queue behave?", "Astra")
            self.assertEqual(run.data["phase"], "awaiting_power_cycle")
            resumed = Verification(board, path, interval=0)
            with self.assertRaisesRegex(RuntimeError, "unchanged"):
                resumed.after()
            board.boot = str(uuid4())
            resumed.after(witnessed=True)
            self.assertEqual(resumed.data["phase"], "complete")
            self.assertEqual(resumed.data["final"]["active"], board.base)
            self.assertTrue(resumed.data["operator_witnessed_cold_power_cycle"])
