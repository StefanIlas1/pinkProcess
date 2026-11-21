import unittest

from .dataset import build_examples
from .fixed_point import encode_features
from .kit_dataset import DEMO_CAPACITY, demo_csv, prepare_dataset
from .simulator import simulate


class DatasetTest(unittest.TestCase):
    def test_backlog_uses_current_queue_even_when_every_row_is_waiting(self):
        rows = "time_ms,busy,waiting\n" + "".join(f"{100*i},4,8\n" for i in range(50))
        result = prepare_dataset(rows, 4, backlog=True)
        self.assertEqual(result["waiting"], 8)
        self.assertEqual(result["features"], [8,0,0,1])
        self.assertEqual(prepare_dataset(rows+"5000,4,0\n", 4, backlog=True)["waiting"], 0)

    def test_export_matches_training_features_and_times(self):
        parsed = prepare_dataset(demo_csv(), DEMO_CAPACITY)
        actual = dict(zip(parsed["times_ms"], parsed["features"]))
        run = simulate(200, duration_ms=20_000, capacity=DEMO_CAPACITY)
        examples = build_examples(run)
        self.assertTrue(examples)
        for example in examples:
            self.assertEqual(actual[example.time_ms], list(encode_features(example, run.capacity)))
        self.assertEqual(len(actual) + parsed["skipped_waiting"], len(run.readings) - 49)

    def test_reject_gap(self):
        with self.assertRaisesRegex(ValueError, "100 ms"):
            prepare_dataset("time_ms,busy,waiting\n0,0,0\n200,0,0", 12)
