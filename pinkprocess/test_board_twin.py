"""The Mac board twin runs the board language service on the INT8 checkpoints when they are installed."""

from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep
import unittest

from .board_twin import SoftwareFabric, checkpoint_dirs, make_board
from .board_sim import SimulatedBoard
from .kit_data import sample_csv
from .telemetry_device import SCORES

FOUND = checkpoint_dirs()


def finish(board, job):
    for _ in range(300):
        if job.get("state") != "running":
            return job
        sleep(0.1)
        job = board.request("lm_result", job_id=job["job_id"])
    return job


class SoftwareFabricTest(unittest.TestCase):
    def test_scores_are_signed_dot_products(self):
        fabric = SoftwareFabric()
        fabric.write(64, bytes([1, 255] + [0] * 62))
        fabric.submit(bytes([3, 2] + [0] * 62))
        self.assertEqual(fabric.read(SCORES + 4), 1)


class MissingCheckpointTest(unittest.TestCase):
    def test_falls_back_to_the_plain_simulator(self):
        with TemporaryDirectory() as empty:
            self.assertEqual(checkpoint_dirs([Path(empty)]), {})
        self.assertIsInstance(make_board(simulated_only=True), SimulatedBoard)


@unittest.skipUnless("original" in FOUND, "INT8 checkpoints are not installed")
class BoardTwinTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.board = make_board()

    def test_answers_use_the_uploaded_readings(self):
        upload = self.board.request("lm_dataset_upload", csv=sample_csv("pool-recovery"), capacity=12, name="pool-recovery.csv")
        self.assertEqual(upload["snapshot"]["observations"]["peak_waiting"]["waiting"], 87)
        for name in [version["name"] for version in upload["models"]]:
            self.board.request("lm_dataset_select", dataset_id=upload["dataset_id"], name=name)
            job = finish(self.board, self.board.request("lm_dataset_ask", dataset_id=upload["dataset_id"],
                                                        question="What's wrong with my system?", stage="final"))
            self.assertEqual(job["state"], "complete", job)
            self.assertTrue(job["answer"].strip())
            if name != "Telemetry v1":
                self.assertIn("87", job["answer"])

    def test_lists_every_installed_checkpoint(self):
        names = [version["name"] for version in self.board.request("lm_versions")["versions"]]
        self.assertEqual(names[0], "original")
        self.assertEqual(len(names), len(FOUND))


if __name__ == "__main__":
    unittest.main()
