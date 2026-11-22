import unittest

from lm_reference.flan_prompt import measured_history, telemetry_prompt


class MeasuredHistoryTests(unittest.TestCase):
    def evidence(self, first, peak, current):
        return {"capacity": 4, "latest": {"waiting": current, "busy": 2, "time_ms": 5000},
                "observations": {"first": {"waiting": first},
                                 "peak_waiting": {"waiting": peak, "time_ms": 2000}}}

    def test_history_wording_is_bound_to_measured_endpoints(self):
        for first, peak, current, expected in [
            (0, 0, 0, "No backlog was observed in this window."),
            (0, 7, 0, "The queue cleared by the final sample."),
            (0, 9, 7, "Active queue. Waiting rose from 0 to 7."),
            (8, 9, 2, "Active queue. Waiting fell from 8 to 2."),
            (2, 9, 2, "Active queue. First and final waiting counts were both 2."),
        ]:
            with self.subTest(first=first, peak=peak, current=current):
                data = self.evidence(first, peak, current)
                self.assertEqual(measured_history(data), expected)
                original = telemetry_prompt("Why?", data)
                self.assertNotIn(expected, original)
                self.assertEqual(telemetry_prompt("Why?", data, history=True),
                                 original.replace("\nQuestion:", " Measured queue history: " + expected + "\nQuestion:"))

    def test_incomplete_or_inconsistent_history_has_no_cue(self):
        for evidence in ({}, self.evidence(4, 2, 0), self.evidence(-1, 3, 2)):
            self.assertEqual(measured_history(evidence), "")
