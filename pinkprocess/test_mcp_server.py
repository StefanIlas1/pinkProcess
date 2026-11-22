import io
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from .kit_accounts import Accounts
from .kit_data import SAMPLES, DatasetStore
from .mcp_server import Connector, interpreter_ok, mcp_config, pick_python, serve

ROOT = Path(__file__).resolve().parent.parent


def rpc(identifier, method, params=None):
    message = {"jsonrpc": "2.0", "method": method}
    if identifier is not None:
        message["id"] = identifier
    if params is not None:
        message["params"] = params
    return json.dumps(message)


def run_lines(connector, lines):
    sink = io.StringIO()
    serve(connector, io.StringIO("\n".join(lines) + "\n"), sink)
    return [json.loads(line) for line in sink.getvalue().splitlines()]


def tool(identifier, tool_name, **arguments):
    return rpc(identifier, "tools/call", {"name": tool_name, "arguments": arguments})


def payload(reply):
    return json.loads(reply["result"]["content"][0]["text"])


class McpTest(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.history = Path(self.directory.name) / "history"
        self.data = Path(self.directory.name) / "datasets"
        accounts = Accounts(self.history, environ={})
        accounts.action({"action": "local_login", "name": "Ana"})
        self.chat = accounts.action({"action": "history_save", "question": "Peak?", "answer": "19", "dataset": "pool.csv"})["chat_id"]
        DatasetStore(self.data).save("mine.csv", "t,v\n1,2\n2,3\n3,9\n")
        self.connector = Connector(self.history, self.data, environ={})

    def test_handshake_and_tool_list(self):
        replies = run_lines(self.connector, [
            rpc(1, "initialize", {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}),
            rpc(None, "notifications/initialized"), rpc(2, "ping"), rpc(3, "tools/list"), rpc(4, "nope")])
        self.assertEqual([reply["id"] for reply in replies], [1, 2, 3, 4])
        self.assertEqual(replies[0]["result"]["protocolVersion"], "2025-03-26")
        self.assertEqual(replies[0]["result"]["serverInfo"]["name"], "pynq-process")
        self.assertEqual(replies[1]["result"], {})
        self.assertEqual({item["name"] for item in replies[2]["result"]["tools"]},
                         {"list_sessions", "get_session", "list_datasets", "profile_dataset", "ask_data"})
        self.assertEqual(replies[3]["error"]["code"], -32601)

    def test_sessions_datasets_and_questions(self):
        replies = run_lines(self.connector, [
            tool(1, "list_sessions"), tool(2, "list_sessions", user="Ana"), tool(3, "get_session", id=self.chat),
            tool(4, "list_datasets"), tool(5, "profile_dataset", name="mine.csv"),
            tool(6, "ask_data", dataset="API latency", question="What was the peak?", style="short"),
            tool(7, "ask_data", dataset="mine.csv", question="average?", style="summary")])
        self.assertEqual(payload(replies[0])["sessions"][0]["id"], self.chat)
        self.assertEqual(payload(replies[1])["sessions"][0]["questions"], 1)
        self.assertEqual(payload(replies[2])["messages"][1]["text"], "19")
        listing = payload(replies[3])
        self.assertEqual(listing["imported"], ["mine.csv"])
        self.assertEqual(len(listing["samples"]), len(SAMPLES))
        self.assertEqual(payload(replies[4])["rows"], 3)
        self.assertIn("peaked", payload(replies[5])["answer"])
        self.assertEqual(payload(replies[5])["where"], "this Mac")
        self.assertIn("averages", payload(replies[6])["answer"])

    def test_errors_are_reported_to_the_caller(self):
        replies = run_lines(self.connector, [
            "not json", "[]", rpc(1, "tools/call", {"name": "missing"}), rpc(2, "tools/call", {"name": "ask_data", "arguments": []}),
            tool(3, "get_session", id="nope"), tool(4, "profile_dataset", name="../../etc/passwd"),
            tool(5, "list_sessions", user="Nobody"), tool(6, "ask_data", dataset="mine.csv", question="x", style="loud"),
            rpc(7, "tools/list", [])])
        self.assertEqual(replies[0]["error"]["code"], -32700)
        self.assertEqual(replies[1]["error"]["code"], -32600)
        self.assertEqual(replies[2]["error"]["code"], -32602)
        self.assertEqual(replies[3]["error"]["code"], -32602)
        for reply in replies[4:8]:
            self.assertTrue(reply["result"]["isError"])
        self.assertEqual(replies[8]["error"]["code"], -32602)

    def test_user_comes_from_the_environment_or_the_newest_profile(self):
        Accounts(self.history, environ={}).action({"action": "local_login", "name": "Ben"})
        ben = Accounts(self.history, environ={})
        ben.action({"action": "local_login", "name": "Ben"})
        ben.action({"action": "session_save", "dataset": "ben.csv"})
        self.assertEqual(payload(run_lines(self.connector, [tool(1, "list_sessions")])[0])["sessions"][0]["dataset"], "ben.csv")
        chosen = Connector(self.history, self.data, environ={"PYNQ_USER": "Ana"})
        self.assertEqual(payload(run_lines(chosen, [tool(1, "list_sessions")])[0])["sessions"][0]["dataset"], "pool.csv")

    def test_empty_history_is_a_clear_error(self):
        with TemporaryDirectory() as empty:
            reply = run_lines(Connector(Path(empty), Path(empty), environ={}), [tool(1, "list_sessions")])[0]
        self.assertTrue(reply["result"]["isError"])
        self.assertIn("No saved sessions", reply["result"]["content"][0]["text"])

    def test_runs_as_a_module_over_stdio(self):
        environment = {**os.environ, "PYTHONPATH": str(ROOT), "PYNQ_HISTORY_DIR": str(self.history), "PYNQ_DATA_DIR": str(self.data)}
        lines = [rpc(1, "initialize", {"protocolVersion": "2024-11-05"}), rpc(None, "notifications/initialized"), tool(2, "list_datasets")]
        done = subprocess.run([sys.executable, "-m", "pinkprocess.mcp_server"], input="\n".join(lines) + "\n", capture_output=True,
                              text=True, timeout=30, env=environment, cwd=self.directory.name)
        replies = [json.loads(line) for line in done.stdout.splitlines()]
        self.assertEqual([reply["id"] for reply in replies], [1, 2])
        self.assertEqual(payload(replies[1])["imported"], ["mine.csv"])

    def test_initialize_runs_under_the_interpreter_the_config_picks(self):
        python = mcp_config({})["command"]
        if not interpreter_ok(python):
            self.skipTest("No Python 3.10 or newer interpreter is available.")
        environment = {**os.environ, "PYTHONPATH": str(ROOT), "PYNQ_HISTORY_DIR": str(self.history), "PYNQ_DATA_DIR": str(self.data)}
        done = subprocess.run([python, "-m", "pinkprocess.mcp_server"], input=rpc(1, "initialize", {"protocolVersion": "2024-11-05"}) + "\n",
                              capture_output=True, text=True, timeout=30, env=environment, cwd=self.directory.name)
        self.assertEqual(json.loads(done.stdout.splitlines()[0])["id"], 1)

    def test_old_interpreters_can_import_the_server(self):
        old = "/usr/bin/python3"
        if not Path(old).is_file() or interpreter_ok(old):
            self.skipTest("No older system interpreter to check.")
        done = subprocess.run([old, "-c", "import pinkprocess.mcp_server"], capture_output=True, text=True, timeout=30,
                              env={**os.environ, "PYTHONPATH": str(ROOT)})
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_picked_python_is_modern(self):
        self.assertTrue(Path(pick_python()).is_absolute())

    def test_config_points_at_this_checkout(self):
        config = mcp_config({})
        entry = json.loads(config["config_json"])["mcpServers"]["pynq-process"]
        self.assertEqual(entry["args"], ["-m", "pinkprocess.mcp_server"])
        self.assertTrue(Path(entry["command"]).is_absolute())
        self.assertEqual(entry["env"]["PYTHONPATH"], str(ROOT))
        with patch.object(sys, "frozen", True, create=True), patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PYNQ_REPO_DIR", None)
            with patch("pinkprocess.mcp_server.Path.cwd", return_value=Path(self.directory.name)):
                frozen = mcp_config({})
            self.assertIn("note", frozen)
            self.assertNotEqual(Path(frozen["command"]).name, "backend")
            with patch.dict(os.environ, {"PYNQ_REPO_DIR": str(ROOT)}):
                self.assertEqual(mcp_config({})["env"]["PYTHONPATH"], str(ROOT))


if __name__ == "__main__":
    unittest.main()
