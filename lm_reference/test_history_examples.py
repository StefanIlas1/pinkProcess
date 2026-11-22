"""Regression coverage for the opt-in queue-history training distribution."""
import hashlib
import json
import random
import unittest

from lm_reference.adapt_flan import example


class HistoryExamples(unittest.TestCase):
    def test_historical_training_recipe(self):
        pairs = [example(random.Random(i), 0, detailed=True, diagnostic=True) for i in range(8)]
        self.assertEqual(hashlib.sha256(json.dumps(pairs).encode()).hexdigest(),
                         "64dda00cdfad132f377033f88733a29da69a1c23db7044ba8db344bf164ab4f4")

    def test_balanced_history_examples_distinguish_states_and_net_direction(self):
        rng = random.Random(83126)
        counts = dict(queue_free=0, recovered=0, active=0)
        directions = set()
        for _ in range(1200):
            prompt, answer = example(rng, 0, detailed=True, diagnostic=True, balanced_history=True)
            if answer.startswith("No backlog was observed"):
                counts["queue_free"] += 1
                self.assertIn("Peak queue: 0 requests", prompt)
            elif answer.startswith("The queue has cleared"):
                counts["recovered"] += 1
                self.assertNotIn("Peak queue: 0 requests", prompt)
                self.assertIn("Current queue: 0 waiting requests", prompt)
            else:
                counts["active"] += 1
                self.assertNotIn("Current queue: 0 waiting requests", prompt)
                for direction in ("rose", "fell"):
                    if "Waiting " + direction in answer:
                        directions.add(direction)
        self.assertTrue(all(300 < count < 500 for count in counts.values()))
        self.assertEqual(directions, {"rose", "fell"})
