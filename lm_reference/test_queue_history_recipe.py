"""Pin the exact pre-quantization training stream and keep evaluation examples out."""

import hashlib
import json
from pathlib import Path
import random
import unittest

from kit.reference_forecast import make_forecaster
from lm_reference.train_queue_history import example


ROOT = Path(__file__).resolve().parents[1]


class QueueHistoryRecipeTests(unittest.TestCase):
    def test_exact_training_examples_and_held_out_prompt_separation(self):
        # Recorded from the qualified run's development generator, before porting it.
        expected = [
            "233c1ed003412be8684a8013b252300096aa8b4e6291b90eacdfb196bba69d11",
            "3b10af71d97cb929b1f00d443b1e863cdfc608e47aa3fe7c96e4a72c07c16262",
            "2206fbb1cfc0bc79b3977024f2ffc8a9b9b37c76f2161d130402f284e743c10c",
            "0f280b1de3e87f6bb92599afb105b3898547edc37fa84c52cc6c25ebb7f202a7",
        ]
        review = json.loads((ROOT / "kit/evidence/grounded-history-independent-review.json").read_text())
        evaluation_prompts = {case["grounded_360_prompt_sha256"] for case in review["paired"]}
        training_prompts = set()
        rng = random.Random(4227)
        tasks = [0] * 16 + list(range(1, 11))
        forecast = make_forecaster(ROOT / "profiles")
        actual = []
        for _ in expected:
            digest = hashlib.sha256()
            for _ in range(90 * 8):
                pair = example(rng, rng.choice(tasks), forecast=forecast, detailed=True,
                               diagnostic=True, balanced_history=True)
                digest.update(json.dumps(pair, ensure_ascii=False, separators=(",", ":")).encode() + b"\n")
                training_prompts.add(hashlib.sha256(pair[0].encode()).hexdigest())
            actual.append(digest.hexdigest())
        self.assertEqual(actual, expected)
        self.assertTrue(training_prompts.isdisjoint(evaluation_prompts))
