from threading import Event, Thread, Timer
from time import monotonic, sleep
import unittest

from .board_demo import run_demo
from .kit_server import InferenceLock
from .simulator import SAMPLE_MS
from .telemetry_stream import TelemetryStream


class BoardSamplingTest(unittest.TestCase):
    def test_measurements_keep_their_cadence_during_a_slow_publish(self):
        stopped = Event()
        stream = TelemetryStream(lambda *_: None)
        batches = []

        def publish(payload):
            if len(batches) == 1:
                sleep(0.27)
            batches.append(payload["readings"])
            if stream.ingest(payload)["sample_count"] >= 7:
                stopped.set()

        deadline = Timer(3, stopped.set)
        deadline.start()
        try:
            run_demo(publish, stopped)
        finally:
            deadline.cancel()
        times = [sample["time_ms"] for readings in batches for sample in readings]
        self.assertGreaterEqual(len(times), 7)
        self.assertGreaterEqual(max(map(len, batches)), 2)
        self.assertEqual(times, list(range(times[0], times[-1] + SAMPLE_MS, SAMPLE_MS)))

    def test_waiting_telemetry_runs_before_a_reacquiring_inference_thread(self):
        lock = InferenceLock()
        order = []

        def telemetry():
            with lock:
                order.append("telemetry")

        with lock:
            thread = Thread(target=telemetry)
            thread.start()
            deadline = monotonic() + 1
            while monotonic() < deadline:
                with lock.condition:
                    if lock.next_ticket == 2:
                        break
                sleep(0.001)
            else:
                self.fail("Telemetry did not queue for the FPGA lock.")

        with lock:
            order.append("inference")
        thread.join(1)
        self.assertEqual(order, ["telemetry", "inference"])


if __name__ == "__main__":
    unittest.main()
