import unittest

from kit.evaluate_delivery import rate_preparer, score


class DeliveryProtocolTest(unittest.TestCase):
    def test_arrivals_are_causal_and_half_open(self):
        telemetry = "time_ms,busy,waiting\n" + "".join(
            f"{t},0,0\n" for t in range(0, 5100, 100))
        data = rate_preparer([4_000_000, 4_000_001, 5_000_000, 5_000_001])(telemetry, 4)
        # At 5s only two arrivals fall in (4s,5s]. Q8(2 * .16 / 4) = 20.
        self.assertEqual(data["features"][-1], [0, 0, 20])

    def test_credit_uses_display_time_and_collapses_positive_streaks(self):
        metrics = score([6200, 10050], [(5000, 250, True), (6000, 50, True),
                                       (7000, 50, False), (10000, 80, True),
                                       (11000, 50, False), (12000, 50, True)])
        self.assertEqual(metrics, dict(episodes=2, warned=1, missed=1, early=0,
                                       false_alerts=1, alerts_arriving_after_onset=1))

    def test_only_one_credit_per_episode(self):
        metrics = score([9500], [(5000, 50, True), (6000, 50, False), (7000, 50, True)])
        self.assertEqual((metrics["early"], metrics["warned"]), (1, 1))
