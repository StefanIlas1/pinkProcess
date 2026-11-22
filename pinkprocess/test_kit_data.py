import json
from pathlib import Path
from random import Random
from time import perf_counter
from tempfile import TemporaryDirectory
import unittest

from .kit_app import Kit
from .kit_data import SAMPLES, STYLES, DatasetStore, answer_question, parse_table, profile_table, queue_ready, safe_dataset_name, sample_csv


class NoBoard:
    def close(self):
        pass


WRONG = "What's wrong with my system?"


def top_row(sample_id):
    table = parse_table(sample_csv(sample_id), sample_id)
    return answer_question(table, WRONG, "table")["table"]["rows"]


class AnalystTest(unittest.TestCase):
    def test_every_style_keeps_the_same_keys(self):
        table = parse_table(sample_csv("api-latency"), "api-latency.csv")
        for style in STYLES:
            result = answer_question(table, WRONG, style)
            self.assertEqual(set(result), {"style", "focus", "readings", "table", "answer", "events"})
            self.assertEqual(result["style"], style)
            self.assertTrue(result["events"])
            self.assertEqual(result["table"] is not None, style == "table")
        lines = answer_question(table, WRONG, "summary")["answer"].splitlines()
        self.assertTrue(3 <= len(lines) <= 5)

    def test_checkout_slowdown_is_traced_to_the_endpoint_region_and_pool(self):
        row = top_row("checkout-api-10k")[0]
        start, end, _, signal, _, _, moved, top = row
        self.assertEqual(signal, "latency")
        self.assertTrue("14:10" <= start <= "14:14", start)
        self.assertTrue("14:29" <= end <= "14:34", end)
        self.assertEqual(top.split(" (")[0], "/checkout in eu-west")
        self.assertIn("DB pool wait", moved)
        self.assertIn("error rate went from 0.", moved)
        summary = answer_question(parse_table(sample_csv("checkout-api-10k"), "x"), WRONG, "summary")["answer"]
        lines = summary.splitlines()
        self.assertTrue(lines[0].startswith("Latency went from 141 ms to"))
        self.assertIn("/checkout in eu-west", lines[0])
        self.assertTrue(lines[1].startswith("Likely cause: DB pool wait, which went from 6"))
        self.assertNotIn("x normal", summary)

    def test_cold_chain_failure_is_truck_t07_and_the_doors_are_second(self):
        rows = top_row("cold-chain-10k")
        start, end, _, signal, peak, _, moved, top = rows[0]
        self.assertEqual(signal, "temperature")
        self.assertTrue("Nov 19 13:50" <= start <= "Nov 19 14:15", start)
        self.assertTrue("Nov 19 19:00" <= end <= "Nov 19 19:30", end)
        self.assertIn("T07", top)
        self.assertIn("compressor", moved)
        self.assertGreater(float(peak.split()[0]), 8)
        self.assertIn("T03", rows[1][7])
        self.assertIn("door open", rows[1][6])
        self.assertLess(float(rows[1][4].split()[0]), 7)
        summary = answer_question(parse_table(sample_csv("cold-chain-10k"), "x"), WRONG, "summary")["answer"].splitlines()
        self.assertTrue(summary[1].startswith("Compressor went from 6.4"))
        self.assertIn("at Nov 19 14:00, then temperature rose", summary[1])
        self.assertTrue(any(line.startswith("Smaller: truck T03") and "normal handling" in line for line in summary))

    def test_card_outage_is_found_in_one_region_and_cash_is_left_alone(self):
        rows = top_row("store-sales-10k")
        start, end, _, signal, _, _, _, top = rows[0]
        self.assertIn(signal, ("orders", "revenue"))
        self.assertTrue("Oct 30 11:30" <= start <= "Oct 30 12:30", start)
        self.assertTrue("Oct 30 16:30" <= end <= "Oct 30 17:30", end)
        self.assertEqual(top.split(" (")[0], "card in north")
        self.assertNotIn("cash", " ".join(row[7] for row in rows))
        self.assertIn("orders went from", rows[0][6])
        self.assertIn("per hour", rows[0][4])
        summary = answer_question(parse_table(sample_csv("store-sales-10k"), "x"), WRONG, "summary")["answer"]
        self.assertIn("stayed normal for cash in north", summary)
        self.assertIn("on card in north", summary)
        self.assertRegex(summary, r"Orders went from 1\d per hour to [0-2] per hour")

    def test_errors_move_with_the_api_latency_bursts(self):
        rows = top_row("api-latency")
        self.assertEqual(rows[0][3], "p95 latency")
        self.assertTrue("12:58" <= rows[0][0] <= "13:04", rows[0][0])
        self.assertIn("errors", rows[0][6])
        self.assertEqual(len(rows), 2)

    def test_the_drifting_sensor_is_named_with_the_gap(self):
        table = parse_table(sample_csv("iot-drift"), "iot-drift.csv")
        result = answer_question(table, WRONG, "short")
        self.assertIn("sensor_b_c is drifting away from sensor_a_c", result["answer"])
        row = answer_question(table, WRONG, "table")["table"]["rows"][0]
        self.assertTrue("11:10" <= row[0] <= "11:30", row[0])
        self.assertIn("4.", row[4])

    def test_epoch_and_relative_times_read_as_clock_and_offsets(self):
        logs = answer_question(parse_table(sample_csv("log-events"), "log-events.jsonl"), WRONG, "table")["table"]["rows"][0]
        self.assertTrue("t+11:30" <= logs[0] <= "t+11:50", logs[0])
        self.assertIn("error rate", logs[6])
        epoch = "\n".join(["ts,value"] + [f"{1_700_000_000_000 + index * 60_000},{10 if not 40 <= index < 60 else 90}" for index in range(120)])
        rows = answer_question(parse_table(epoch, "epoch.csv"), WRONG, "table")["table"]["rows"]
        self.assertRegex(rows[0][0], r"^\d\d:\d\d$")
        self.assertEqual(rows[0][2], "20 min")

    def test_a_flat_series_with_one_spike_is_not_hidden_by_the_spike(self):
        text = "t,v\n" + "".join(f"{index},{50 + (index % 3)}\n" for index in range(300)) + "".join(f"{300 + index},400\n" for index in range(6)) + "".join(f"{306 + index},{50 + index % 3}\n" for index in range(100))
        rows = answer_question(parse_table(text, "spike.csv"), WRONG, "table")["table"]["rows"]
        self.assertEqual(len(rows), 1)
        self.assertIn("400", rows[0][4])

    def test_quiet_data_reads_as_quiet(self):
        text = "t,v\n" + "".join(f"{index},{50 + (index % 3)}\n" for index in range(200))
        result = answer_question(parse_table(text, "quiet.csv"), WRONG, "short")
        self.assertTrue(result["answer"].startswith("Nothing unusual"))

    def test_fifty_thousand_rows_answer_fast_and_repeat_calls_reuse_the_work(self):
        rng = Random(5)
        lines = ["timestamp,endpoint,region,status,latency_ms,cpu_pct"]
        for index in range(50_000):
            slow = 30_000 <= index < 31_500 and index % 3 == 0
            lines.append(f"2026-01-01 {index // 3600 % 24:02d}:{index // 60 % 60:02d}:{index % 60:02d},/e{rng.randint(1, 8)},r{rng.randint(1, 4)},"
                         f"{rng.choice([200, 200, 200, 500])},{rng.uniform(20, 90) + (900 if slow else 0):.1f},{rng.uniform(10, 60):.1f}")
        table = parse_table("\n".join(lines), "big.csv")
        started = perf_counter()
        first = answer_question(table, WRONG, "summary")
        first_seconds = perf_counter() - started
        started = perf_counter()
        for style in STYLES:
            answer_question(table, WRONG, style)
        self.assertLess(first_seconds, 1.0)
        self.assertLess(perf_counter() - started, 0.2)
        self.assertEqual(first["focus"], "latency_ms")


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
        self.assertEqual(grid["columns"], ["Start", "End", "Duration", "Signal", "Peak", "Normal", "Moved with it", "Top category"])
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
