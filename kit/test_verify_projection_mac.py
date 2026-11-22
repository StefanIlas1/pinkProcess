import unittest
from types import SimpleNamespace

try:
    import numpy
except ImportError:
    numpy = None


@unittest.skipIf(numpy is None, "The exact FLAN reference requires NumPy.")
class MacProjectionVerificationTest(unittest.TestCase):
    def test_does_not_remove_another_producer(self):
        from kit.verify_projection_mac import UNSEEN, fixture_capture

        mac = SimpleNamespace(ready_stream=lambda: {"source": "Production pool", "origin": "live"})
        calls = []
        with self.assertRaisesRegex(RuntimeError, "SQLite demo"):
            with fixture_capture(mac, lambda **payload: calls.append(payload), UNSEEN[0]):
                self.fail("Published a synthetic fixture over a different producer.")
        self.assertEqual(calls, [])

    def test_releases_the_guarded_fixture_when_capture_fails(self):
        from kit.verify_projection_mac import UNSEEN, fixture_capture
        from pinkprocess.telemetry_stream import BOARD_DEMO_SOURCE

        mac = SimpleNamespace(ready_stream=lambda: {
            "source": BOARD_DEMO_SOURCE, "origin": "demo", "session": "observed-demo",
            "fixture_guard_supported": True, "fpga_submissions": 10,
        })
        calls = []

        def send(**payload):
            calls.append(payload)
            if not payload.get("stop"):
                return {"accepted": len(payload["readings"]), "state": "ready",
                        "session": payload["session"], "fpga_submissions": 11}

        with self.assertRaisesRegex(ValueError, "capture failed"):
            with fixture_capture(mac, send, UNSEEN[0]):
                raise ValueError("capture failed")
        published, stopped = calls
        self.assertEqual(published["fixture_after_session"], "observed-demo")
        self.assertEqual(published["readings"][-1]["waiting"], 0)
        self.assertEqual(stopped, {"session": published["session"], "stop": True})

    def test_preserves_lexical_miss_and_credits_explicit_uncertainty(self):
        from kit.verify_projection_mac import review_rubric

        answer = "4 requests are waiting. The cause is unknown. Measure arrivals and connection hold times."
        lexical, reviewed, note = review_rubric("ambiguous", answer, {})
        self.assertFalse(lexical["recognizes_cause_uncertainty"])
        self.assertTrue(all(reviewed.values()))
        self.assertIn("The cause is unknown", note)


if __name__ == "__main__":
    unittest.main()
