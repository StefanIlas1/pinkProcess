import os
import unittest
from unittest.mock import patch

from .board_sim import HOST, SimulatedBoard
from .kit_app import Kit, make_server
from .kit_dataset import pool_sample_csv
from .kit_transport import BoardTransport


class SimulatedBoardTest(unittest.TestCase):
    def setUp(self):
        self.board = SimulatedBoard()
        self.kit = Kit(self.board)
        self.kit.action({"action": "connect", "host": HOST})

    def test_it_reports_a_connected_board_with_telemetry(self):
        status = self.kit.action({"action": "status"})
        self.assertEqual(status["host"], HOST)
        self.assertTrue(status["status"]["lm_supported"])
        snapshot = self.kit.action({"action": "stream_status"})["snapshot"]
        self.assertEqual(snapshot["state"], "ready")
        self.assertGreaterEqual(snapshot["latest"]["busy"], 0)

    def test_a_question_streams_then_completes(self):
        with patch("pinkprocess.board_sim.ANSWER_SECONDS", 0):
            job = self.kit.action({"action": "ask_begin", "question": "What looks wrong?"})
            self.assertEqual(job["state"], "running")
            done = self.kit.action({"action": "answer_progress", "job_id": job["job_id"]})
        self.assertEqual(done["state"], "complete")
        self.assertIn("requests waiting", done["answer"])

    def test_dataset_upload_and_followup(self):
        dataset = self.kit.action({"action": "dataset_upload", "csv": pool_sample_csv(), "capacity": 12, "name": "pool.csv"})
        self.assertEqual(dataset["selected_model"], "original")
        with patch("pinkprocess.board_sim.ANSWER_SECONDS", 0):
            job = self.kit.action({"action": "dataset_ask", "dataset_id": dataset["dataset_id"], "question": "Peak?", "stage": "final"})
            done = self.kit.action({"action": "answer_progress", "job_id": job["job_id"]})
        self.assertEqual(done["state"], "complete")

    def test_unsupported_actions_are_clean_errors(self):
        with self.assertRaises(ValueError):
            self.kit.action({"action": "dataset_synthesize", "dataset_id": "x", "name": "n", "target_row": "Now", "instruction": "i"})

    def test_the_flag_is_off_by_default_and_connects_when_set(self):
        with patch.dict(os.environ, {"PYNQ_SIM_BOARD": "1"}):
            with make_server(port=0) as server:
                self.assertEqual(server.kit.host, HOST)
                self.assertIsInstance(server.kit.transport, SimulatedBoard)
        environment = {key: value for key, value in os.environ.items() if key != "PYNQ_SIM_BOARD"}
        with patch.dict(os.environ, environment, clear=True):
            with make_server(port=0) as server:
                self.assertIsNone(server.kit.host)
                self.assertIsInstance(server.kit.transport, BoardTransport)


if __name__ == "__main__":
    unittest.main()
