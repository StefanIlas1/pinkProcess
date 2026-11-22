from contextlib import contextmanager
from threading import Event, Thread
from types import SimpleNamespace
import unittest

import numpy as np

from kit import benchmark_projection_overlay as benchmark
from lm_reference import projection_overlay as projection


class Buffer(np.ndarray):
    def flush(self):
        self.events.append(("flush", self.device_address))

    def invalidate(self):
        self.events.append(("invalidate", self.device_address))


class FakeOverlay:
    def __init__(self, *, stalled=False):
        self.projection = SimpleNamespace(mmio=self)
        self.stalled = stalled
        self.registers = {projection.CORE_ID: projection.PROJECTION_ID, projection.CONTROL: 0}
        self.buffers = {}
        self.events = []

    @contextmanager
    def allocate(self, *, shape, dtype):
        address = (len(self.buffers) + 1) * 65536
        buffer = np.zeros(shape, dtype).view(Buffer)
        buffer.device_address = address
        buffer.events = self.events
        self.buffers[address] = buffer
        try:
            yield buffer
        finally:
            del self.buffers[address]
            self.events.append(("free", address))

    def read(self, register):
        return self.registers[register]

    def write(self, register, value):
        self.registers[register] = value
        if register != projection.CONTROL:
            return
        self.events.append(("launch",))
        if self.stalled:
            return  # Control retains bit zero (busy) until reset.
        a, w, y = (self.buffers[self.registers[r]] for r in (
            projection.A_ADDRESS, projection.W_ADDRESS, projection.Y_ADDRESS))
        y[:] = benchmark.simulate_projection(a, w, self.registers[projection.MODE])
        self.registers[projection.CONTROL] = projection.DONE

    def download(self):
        self.events.append(("reset",))
        self.registers[projection.CONTROL] = 0


class ProjectionOverlayTest(unittest.TestCase):
    def setUp(self):
        random = np.random.default_rng(9)
        self.w = random.integers(-128, 128, (65, 19), dtype=np.int8)
        self.a = random.integers(-128, 128, (9, 19), dtype=np.int8)

    def test_encoder_and_decoder_partial_tiles(self):
        for activations, mode, positions, outputs in ((self.a, "encode", 2, 9), (self.a[:1], "decode", 1, 2)):
            with self.subTest(mode=mode):
                overlay = FakeOverlay()

                def run(a, w, decode):
                    return benchmark.run_projection(overlay, overlay.allocate, a, w, decode, 1)

                report = benchmark.benchmark(self.w, activations, run, 1)
                self.assertEqual((report["mode"], report["position_tiles"], report["output_tiles"]),
                                 (mode, positions, outputs))
                self.assertEqual(overlay.registers[projection.DEPTH], 19)
                self.assertEqual(overlay.registers[projection.POSITION_TILES], positions)
                self.assertEqual(overlay.registers[projection.OUTPUT_TILES], outputs)
                self.assertEqual([event[0] for event in overlay.events],
                                 ["flush", "flush", "flush", "launch", "invalidate", "free", "free", "free"])

    def test_timeout_reprograms_before_freeing_buffers(self):
        overlay = FakeOverlay(stalled=True)
        a, w = projection.pack_projection(self.w, self.a)
        with self.assertRaises(TimeoutError):
            benchmark.run_projection(overlay, overlay.allocate, a, w, False, 0)
        self.assertEqual([event[0] for event in overlay.events],
                         ["flush", "flush", "flush", "launch", "reset", "free", "free", "free"])

    def test_live_timeout_retains_all_buffers_until_idle_and_disables_dispatch(self):
        overlay = FakeOverlay(stalled=True)
        query = projection.QueryProjection(overlay, overlay.allocate, timeout=0)
        name = "encoder.block.0.layer.0.SelfAttention.q"
        with self.assertRaises(TimeoutError):
            query.project(name, self.w, self.a)
        self.assertEqual(len(overlay.buffers), 3)
        self.assertIsNone(query.project(name, self.w, self.a))
        self.assertEqual(overlay.events.count(("launch",)), 1)

        closed = Event()

        def shutdown():
            query.close()
            closed.set()

        worker = Thread(target=shutdown, daemon=True)
        worker.start()
        self.assertFalse(closed.wait(0.02))
        overlay.registers[projection.CONTROL] = projection.DONE
        self.assertTrue(closed.wait(1))
        self.assertEqual(len(overlay.buffers), 0)
        self.assertNotIn(("reset",), overlay.events)

    def test_optional_core_detection(self):
        self.assertIsNone(projection.projection_mmio(SimpleNamespace(), required=False))
        overlay = FakeOverlay()
        self.assertIs(projection.projection_mmio(overlay, required=False), overlay)
        overlay.registers[projection.CORE_ID] = 0
        self.assertIsNone(projection.projection_mmio(overlay, required=False))


if __name__ == "__main__":
    unittest.main()
