from pathlib import Path
import json
import unittest

from .kit_data import DATASETS_DIR, MAX_ROWS, SAMPLES, parse_table, profile_table, sample_csv


class DatasetFolderTest(unittest.TestCase):
    def test_manifest_matches_the_files(self):
        listed = json.loads((DATASETS_DIR / "manifest.json").read_text())
        self.assertEqual({entry["file"] for entry in listed}, {sample["file"] for sample in SAMPLES})
        on_disk = {path.name for path in DATASETS_DIR.iterdir() if path.suffix in (".csv", ".jsonl")}
        self.assertEqual(on_disk, {entry["file"] for entry in listed})
        readme = (DATASETS_DIR / "README.md").read_text()
        for entry in listed:
            self.assertIn(entry["file"], readme)

    def test_every_dataset_profiles_and_gets_a_recommendation(self):
        profiled = 0
        for sample in SAMPLES:
            if sample["id"] == "malformed":
                continue
            with self.subTest(sample=sample["id"]):
                table = parse_table(sample_csv(sample["id"]), sample["file"])
                profile = profile_table(table, sample["file"])
                self.assertTrue(profile["recommendation"]["model_id"])
                self.assertTrue(profile["recommendation"]["reason"])
                self.assertGreater(profile["rows"], 0)
                profiled += 1
        self.assertGreaterEqual(profiled, 10)

    def test_the_malformed_file_is_rejected_with_a_message(self):
        with self.assertRaisesRegex(ValueError, "columns"):
            parse_table(sample_csv("malformed"), "malformed.csv")

    def test_the_large_file_sits_at_the_row_limit(self):
        table = parse_table(sample_csv("large-50k"), "large-50k.csv")
        self.assertEqual(table["rows"], MAX_ROWS)

    def test_pool_files_are_queue_ready_and_the_jsonl_is_read(self):
        for sample in SAMPLES:
            if sample["id"].startswith("pool-"):
                self.assertTrue(profile_table(parse_table(sample_csv(sample["id"]), sample["file"]), sample["file"])["queue_ready"], sample["id"])
        events = parse_table(sample_csv("log-events"), "log-events.jsonl")
        self.assertEqual(events["format"], "JSON lines")

    def test_files_stay_small_except_the_large_one(self):
        for path in DATASETS_DIR.iterdir():
            if path.suffix in (".csv", ".jsonl") and path.name != "large-50k.csv":
                self.assertLess(path.stat().st_size, 200_000, path.name)

    def test_generator_output_is_deterministic(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("generate", Path(DATASETS_DIR / "generate.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.pool_series(14, module.recovery), module.pool_series(14, module.recovery))


if __name__ == "__main__":
    unittest.main()
