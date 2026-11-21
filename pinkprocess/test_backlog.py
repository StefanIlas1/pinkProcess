import unittest

from .backlog import features
from .simulator import Reading


class QueueOutlookTest(unittest.TestCase):
    def vector(self, before, current):
        return features([Reading(t*100, 4, before if t < 49 else current)
                         for t in range(50)], 4)

    def test_flat_queue_persists_without_growth(self):
        self.assertEqual(self.vector(20, 20), [20, 0, 0, 1])

    def test_negative_slope_and_large_backlog(self):
        self.assertEqual(self.vector(25, 10), [10, -15, -15, 1])
        self.assertEqual(self.vector(10000, 10001), [127, 0, 1, 1])
