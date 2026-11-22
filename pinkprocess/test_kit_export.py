import csv
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from .kit_accounts import Accounts
from .kit_export import export_action, export_session, safe_filename

SESSION = {
    "id": "abc", "title": "pool/../telemetry?.csv", "dataset": "pool.csv", "model": "queue-onset-logistic",
    "notes": "Ask ops about retries.", "updated_ms": 1700000000000,
    "timings": {"data_ms": 4.2, "profile_ms": 1.5, "fabric_ms": 30},
    "messages": [
        {"role": "user", "text": "Peak?", "time_ms": 1700000000000},
        {"role": "answer", "text": "waiting peaked at 19", "readings": "300 readings of waiting", "style": "short"},
        {"role": "user", "text": "=cmd|' /C calc'!A0"},
        {"role": "answer", "text": "line one\nline, two", "table": {"columns": ["Time", "Event"], "rows": [["1 s", "a|b"]]}},
    ],
}


class ExportTest(unittest.TestCase):
    def test_markdown_has_every_section(self):
        result = export_session(SESSION, "md")
        text = result["content"]
        self.assertEqual(result["mime"], "text/markdown")
        for expected in ("pool.csv", "queue-onset-logistic", "Hot path", "FPGA forecast | 30 ms", "### 1. Peak?",
                         "Answer style: short", "Readings behind this answer: 300 readings of waiting",
                         "Answer style: table", "a\\|b", "## Notes", "Ask ops about retries."):
            self.assertIn(expected, text)

    def test_csv_has_one_row_per_answer_and_blocks_formulas(self):
        rows = list(csv.reader(io.StringIO(export_session(SESSION, "csv")["content"])))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1][1:4], ["Peak?", "short", "waiting peaked at 19"])
        self.assertTrue(rows[2][1].startswith("'="))
        self.assertEqual(rows[2][3], "line one\nline, two")

    def test_json_is_the_raw_session(self):
        self.assertEqual(json.loads(export_session(SESSION, "json")["content"]), SESSION)

    def test_filename_is_sanitized(self):
        name = export_session(SESSION, "md")["filename"]
        self.assertRegex(name, r"^[a-z0-9-]+\.md$")
        self.assertEqual(safe_filename("../../etc/passwd", "csv"), "etc-passwd.csv")
        self.assertEqual(safe_filename("", "json"), "session.json")

    def test_bad_format_and_session_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Markdown"):
            export_session(SESSION, "pdf")
        with self.assertRaises(ValueError):
            export_session("nope", "md")

    def test_sessions_without_history_still_export(self):
        result = export_session({"messages": "broken", "notes": 5}, "md")
        self.assertIn("No questions were asked", result["content"])

    def test_action_reads_saved_history_or_the_posted_session(self):
        with TemporaryDirectory() as directory:
            accounts = Accounts(Path(directory), environ={})
            with self.assertRaises(ValueError):
                export_action({"action": "export", "session_id": "x", "format": "md"}, accounts)
            accounts.action({"action": "local_login", "name": "Ana"})
            saved = accounts.action({"action": "history_save", "question": "Why?", "answer": "Busy.", "dataset": "d.csv", "style": "summary"})
            result = export_action({"action": "export", "session_id": saved["chat_id"], "format": "csv"}, accounts)
            self.assertIn("Why?,summary,Busy.", result["content"])
            posted = export_action({"action": "export", "session_id": None, "session": SESSION, "format": "md"}, accounts)
            self.assertIn("pool.csv", posted["content"])
            for bad in ({"session_id": "missing"}, {"session_id": 4}, {"session_id": None}, {"session_id": None, "session": []},
                        {"session_id": None, "session": SESSION, "extra": 1}):
                with self.assertRaises(ValueError):
                    export_action({"action": "export", "format": "md", **bad}, accounts)


if __name__ == "__main__":
    unittest.main()
