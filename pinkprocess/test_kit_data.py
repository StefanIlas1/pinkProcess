import unittest

from .kit_app import Kit
from .kit_data import SAMPLES, STYLES, answer_question, parse_table, profile_table, sample_csv


class NoBoard:
    def close(self):
        pass


class DataTest(unittest.TestCase):
    def profile(self, sample_id):
        return profile_table(parse_table(sample_csv(sample_id), sample_id + ".csv"), sample_id)

    def test_pool_data_recommends_the_queue_model(self):
        profile = self.profile("pool")
        self.assertTrue(profile["queue_ready"])
        self.assertEqual(profile["time_column"], "time_ms")
        self.assertEqual(profile["recommendation"]["model_id"], "queue-onset-logistic")
        self.assertIn("Queue-onset INT8", profile["recommendation"]["reason"])

    def test_other_signals_are_detected(self):
        self.assertIn("latency", self.profile("latency")["signals"])
        self.assertIn("temperature", self.profile("temperature")["signals"])
        self.assertIn("sales", self.profile("sales")["signals"])
        self.assertEqual(self.profile("sales")["recommendation"]["model_id"], "timeseries-anomaly-int8")
        self.assertEqual(self.profile("temperature")["time_column"], "timestamp")

    def test_json_lines_are_read(self):
        table = parse_table('{"time": 1, "value": 2}\n{"time": 2, "value": 9}\n{"time": 3, "value": 3}\n', "a.jsonl")
        self.assertEqual(table["format"], "JSON lines")
        self.assertEqual(table["types"]["value"], "number")
        self.assertEqual(table["time"], "time")

    def test_bad_files_are_rejected(self):
        for text in ("", "a,b\n", "a,a\n1,2\n", "a,b\n1\n", "{broken"):
            with self.assertRaises(ValueError):
                parse_table(text, "x.csv")

    def test_every_answer_style(self):
        table = parse_table(sample_csv("latency"), "latency.csv")
        short = answer_question(table, "what is the peak?", "short")
        self.assertIn("peaked at 644.4", short["answer"])
        self.assertGreater(len(answer_question(table, "summary", "summary")["answer"].splitlines()), 2)
        transcript = answer_question(table, "what happened", "transcript")
        self.assertIn("PEAK", transcript["answer"])
        self.assertTrue(transcript["events"])
        grid = answer_question(table, "events", "table")["table"]
        self.assertEqual(len(grid["columns"]), 3)
        with self.assertRaises(ValueError):
            answer_question(table, "x", "poem")
        self.assertEqual(STYLES, ("short", "summary", "transcript", "table"))

    def test_kit_actions_work_without_a_board(self):
        kit = Kit(NoBoard())
        with self.assertRaises(ValueError):
            kit.action({"action": "data_ask", "question": "hi", "style": "short"})
        loaded = kit.action({"action": "load_sample", "id": SAMPLES[1]["id"]})
        self.assertEqual(loaded["profile"]["rows"], 240)
        self.assertIn("data_ms", loaded["timings"])
        answer = kit.action({"action": "data_ask", "question": "is it growing?", "style": "summary"})
        self.assertEqual(answer["where"], "this Mac")
        self.assertIn("answer_ms", answer["timings"])
        imported = kit.action({"action": "import_data", "name": "x.csv", "text": "t,v\n1,2\n2,3\n"})
        self.assertEqual(imported["profile"]["name"], "x.csv")
        with self.assertRaises(ValueError):
            kit.action({"action": "load_sample", "id": "nope"})
        with self.assertRaises(ValueError):
            kit.action({"action": "data_ask", "question": "x", "style": "short", "extra": 1})


if __name__ == "__main__":
    unittest.main()
