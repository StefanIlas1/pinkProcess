import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from .live_analyst import MODEL, analyze


class LiveAnalystTest(unittest.TestCase):
    @patch("pinkprocess.live_analyst.HTTPConnection")
    def test_generated_answer_uses_question_and_board_evidence(self, connect):
        connection = connect.return_value
        response = connection.getresponse.return_value
        response.status = 200
        response.read.return_value = json.dumps({
            "model": MODEL, "done": True, "done_reason": "stop",
            "message": {"content": " The observed queue peaked at four. "},
        }).encode()
        evidence = {"observations": {"waiting": {"maximum": 4}}}
        question = "How high did the queue get?"

        result = analyze(question, evidence)

        self.assertEqual(result["answer"], "The observed queue peaked at four.")
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["model"], MODEL)
        self.assertEqual(result["provenance"]["execution"], "Mac")
        self.assertEqual(connect.call_args.args[:2], ("127.0.0.1", 11434))
        method, path, body, _ = connection.request.call_args.args
        self.assertEqual((method, path), ("POST", "/api/chat"))
        request = json.loads(body)
        self.assertEqual(request["model"], MODEL)
        self.assertEqual(json.loads(request["messages"][1]["content"]),
                         {"question": question, "evidence": evidence})
        connection.close.assert_called_once_with()

    @patch("pinkprocess.live_analyst.HTTPConnection")
    def test_missing_model_and_service_never_supply_an_answer(self, connect):
        connection = connect.return_value
        connection.getresponse.return_value.status = 404
        result = analyze("How's the pool?", {"state": "ready"})
        self.assertIsNone(result["answer"])
        self.assertIn("ollama pull", result["errors"][0])

        connection.request.side_effect = ConnectionRefusedError
        result = analyze("How's the pool?", {"state": "ready"})
        self.assertIsNone(result["answer"])
        self.assertIn("ollama serve", result["errors"][0])

    @patch("pinkprocess.live_analyst.HTTPConnection")
    def test_incomplete_or_invalid_answers_are_discarded(self, connect):
        response = connect.return_value.getresponse.return_value
        response.status = 200
        for payload in (
            {"model": MODEL, "done": False, "message": {"content": "Partial"}},
            {"model": MODEL, "done": True, "done_reason": "length",
             "message": {"content": "Truncated"}},
            {"model": "other", "done": True, "message": {"content": "Unexpected model"}},
            {"model": MODEL, "done": True, "message": {"content": " "}},
        ):
            with self.subTest(payload=payload):
                response.read.return_value = json.dumps(payload).encode()
                result = analyze("How's the pool?", {"state": "ready"})
                self.assertIsNone(result["answer"])
                self.assertTrue(result["errors"])

    @patch("pinkprocess.live_analyst._ollama_process", None)
    @patch("pinkprocess.live_analyst.subprocess.Popen")
    @patch("pinkprocess.live_analyst.HTTPConnection")
    def test_installed_ollama_can_start_on_demand(self, connect, launch):
        connection = connect.return_value
        connection.connect.side_effect = [ConnectionRefusedError, ConnectionRefusedError, None]
        connection.getresponse.return_value.status = 404
        binary = Mock(spec=Path)
        binary.is_file.return_value = True
        with patch("pinkprocess.live_analyst.OLLAMA_BINARIES", (binary,)):
            result = analyze("How's the pool?", {"state": "ready"})

        launch.assert_called_once()
        self.assertEqual(launch.call_args.args[0], [str(binary), "serve"])
        self.assertTrue(launch.call_args.kwargs["start_new_session"])
        self.assertEqual(launch.call_args.kwargs["env"]["OLLAMA_HOST"], "127.0.0.1:11434")
        self.assertIn("ollama pull", result["errors"][0])
