import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from .kit_app import Kit
from .kit_data import SAMPLES, STYLES, DatasetStore, answer_question, parse_table, profile_table, queue_ready, safe_dataset_name, sample_csv


class NoBoard:
    def close(self):
        pass


class DataTest(unittest.TestCase):
    def profile(self, sample_id):
        return profile_table(parse_table(sample_csv(sample_id), sample_id + ".csv"), sample_id)

    def test_pool_data_recommends_the_queue_model(self):
        profile = self.profile("pool-recovery")
        self.assertTrue(profile["queue_ready"])
        self.assertEqual(profile["time_column"], "time_ms")
        self.assertEqual(profile["recommendation"]["model_id"], "queue-onset-logistic")
        self.assertIn("Queue-onset INT8", profile["recommendation"]["reason"])

    def test_other_signals_are_detected(self):
        self.assertIn("latency", self.profile("api-latency")["signals"])
        self.assertIn("temperature", self.profile("server-thermal")["signals"])
        self.assertIn("sales", self.profile("sales-seasonal")["signals"])
        self.assertEqual(self.profile("sales-seasonal")["recommendation"]["model_id"], "flan-t5-small-int8")
        self.assertEqual(self.profile("server-thermal")["time_column"], "timestamp")

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
        table = parse_table(sample_csv("api-latency"), "api-latency.csv")
        short = answer_question(table, "what is the peak?", "short")
        self.assertIn("peaked at 586.3", short["answer"])
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
        loaded = kit.action({"action": "load_sample", "id": "api-latency"})
        self.assertEqual(loaded["profile"]["rows"], 720)
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

    def test_csv_edge_cases(self):
        text = "\ufefftime,note,value\r\n1,\"a, b\",5\r\n\r\n  ,  ,  \r\n2,\"say \"\"hi\"\"\",N/A\r\n3,x,7,\r\n"
        table = parse_table(text, "edge.csv")
        self.assertEqual(table["columns"], ["time", "note", "value"])
        self.assertEqual(table["rows"], 3)
        self.assertEqual(table["data"]["value"], [5.0, None, 7.0])
        self.assertEqual(table["types"]["value"], "number")
        self.assertEqual(table["data"]["note"][1], 'say "hi"')
        semicolons = parse_table("time;load\n1;0,5\n2;7\n".replace("0,5", "5"), "s.csv")
        self.assertEqual(semicolons["columns"], ["time", "load"])
        self.assertEqual(parse_table("a\tb\n1\t2\n", "t.tsv")["columns"], ["a", "b"])

    def test_mixed_and_odd_values_do_not_crash(self):
        table = parse_table("t,v,d\n1,x,0001-01-01\n2,3,9999-12-31T23:59:59Z\n", "m.csv")
        self.assertEqual(table["types"]["v"], "text")
        with self.assertRaisesRegex(ValueError, "numeric"):
            answer_question(table, "latest", "short")
        self.assertEqual(answer_question(parse_table("t,v\n0001-01-01,1\n9999-12-31,5\n", "d.csv"), "latest", "short")["style"], "short")
        records = "\n".join(json.dumps(row) for row in ({"t": 1, "v": 2.5}, {"t": 2, "v": None}, {"t": 3, "v": float("nan")}))
        self.assertEqual(parse_table(records, "x.jsonl")["data"]["v"], [2.5, None, None])

    def test_clear_messages_for_bad_files(self):
        cases = {"t,v\n1,2\n3\n": "Line 3 has 1 columns", "t,v\n\"1,2\n": "stray quote", "{\"a\": 1}\nnot json\n": "Line 2",
                 "[" * 100000: "nested too deeply|Could not read", "t,t\n1,2\n": "unique name", "   \n": "empty",
                 "t,v\n1," + "x" * 6000 + "\n": "too long", ",".join(f"c{i}" for i in range(41)) + "\n" + ",".join("1" * 41) + "\n": "at most 40",
                 "[1, 2]": "object", "t\n": "header row"}
        for text, message in cases.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    parse_table(text, "bad")
        with self.assertRaisesRegex(ValueError, "at most"):
            parse_table("t,v\n" + "1,2\n" * 50001, "big.csv")
        with self.assertRaises(ValueError):
            parse_table(None, "bad")

    def test_import_limits(self):
        kit = Kit(NoBoard())
        with self.assertRaisesRegex(ValueError, "20 MB"):
            kit.action({"action": "import_data", "name": "big.csv", "text": "x" * (20 * 1024 * 1024 + 1)})
        for bad in ({"name": "", "text": "t,v\n1,2\n"}, {"name": "a.csv", "text": 5}, {"name": 5, "text": "x"}):
            with self.assertRaises(ValueError):
                kit.action({"action": "import_data", **bad})
        with self.assertRaisesRegex(ValueError, "Choose an action"):
            kit.action({"action": ["x"]})

    def test_queue_data_with_gaps_is_not_sent_to_the_board(self):
        rows = "time_ms,busy,waiting\n" + "".join(f"{i * 100},{i % 5},{'' if i == 7 else 1}\n" for i in range(80))
        table = parse_table(rows, "pool.csv")
        self.assertFalse(queue_ready(table))
        self.assertTrue(queue_ready(parse_table(sample_csv("pool-recovery"), "pool-recovery.csv")))

    def test_import_store_keeps_recent_files_and_blocks_traversal(self):
        with TemporaryDirectory() as directory:
            store = DatasetStore(Path(directory) / "data", keep=2)
            for name in ("a.csv", "b.csv", "c.csv"):
                store.save(name, "t,v\n1,2\n")
            self.assertEqual(len(store.names()), 2)
            self.assertEqual(store.save("../../escape.csv", "t,v\n1,2\n"), "escape.csv")
            self.assertEqual(safe_dataset_name("../.."), "dataset.csv")
            with self.assertRaises(ValueError):
                store.read("../../etc/passwd")
            self.assertEqual(list(Path(directory).glob("*.csv")), [])
            kit = Kit(NoBoard(), store=store)
            kit.action({"action": "import_data", "name": "kept.csv", "text": "t,v\n1,2\n2,3\n"})
            kit.action({"action": "load_sample", "id": "pool-recovery"})
            self.assertIn("kept.csv", store.names())
            self.assertNotIn("pool-recovery.csv", store.names())


if __name__ == "__main__":
    unittest.main()
