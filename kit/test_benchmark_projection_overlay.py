from contextlib import contextmanager, redirect_stderr
import json
from io import StringIO
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import monotonic, sleep
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
        self.allocations = []

    @contextmanager
    def allocate(self, *, shape, dtype):
        address = (len(self.buffers) + 1) * 16 * 1024 * 1024
        buffer = np.zeros(shape, dtype).view(Buffer)
        buffer.device_address = address
        buffer.events = self.events
        self.buffers[address] = buffer
        self.allocations.append(buffer.nbytes)
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

    def test_full_signed_vocabulary_and_segment_boundaries(self):
        overlay = FakeOverlay()
        runner = projection.OverlayProjection(overlay, overlay.allocate, scope="all")
        weights = np.full((32128, 512), -128, dtype=np.int8)
        weights[4095] = 127
        weights[4096] = -127
        weights[-1] = 0
        activations = np.full((1, 512), -128, dtype=np.int8)
        expected = activations.astype(np.int32) @ weights.astype(np.int32).T
        np.testing.assert_array_equal(runner.project("lm_head", weights, activations), expected)
        self.assertEqual(overlay.events.count(("launch",)), 8)
        self.assertEqual(max(overlay.allocations), projection.MAX_SEGMENT_BUFFER_BYTES)
        self.assertFalse(overlay.buffers)

    def test_service_scope_selection(self):
        from pinkprocess.kit_server import parse_args

        base = ["--identity", "identity", "--bitstream", "image.bit", "--profiles", "profiles",
                "--state-file", "state", "--language-model", "model", "--manifest-sha256", "pin"]
        self.assertEqual(parse_args(base + ["--projection-overlay"]).projection_scope, "query")
        self.assertEqual(parse_args(base + ["--projection-overlay", "--projection-scope", "all"]).projection_scope,
                         "all")
        with redirect_stderr(StringIO()) as stderr, self.assertRaises(SystemExit):
            parse_args(base + ["--projection-scope", "all"])
        self.assertIn("requires --projection-overlay", stderr.getvalue())

    def replay(self, snapshot, question, expected, *, edit=None):
        from lm_reference.flan import Flan
        from lm_reference.flan_prompt import telemetry_prompt
        from lm_reference.model_assets import file_digest
        from lm_reference.weight_versions import WeightVersionStore
        from pinkprocess.board_language import BoardLanguage
        from pinkprocess.kit_evidence import analyst_evidence

        model_path = os.getenv("PINKPROCESS_FLAN_MODEL")
        if not model_path:
            self.skipTest("Set PINKPROCESS_FLAN_MODEL to the actual diagnostic checkpoint for replay.")
        root = Path(__file__).resolve().parents[1]
        manifest = file_digest(root / "profiles/flan-telemetry-diagnostic/model.json")

        class LegacyFabric:
            def matmat_array(self, *args):
                raise AssertionError("All-projection replay used legacy arithmetic.")

            def verify_row_bytes(self, payload):
                self.readback = payload

        with TemporaryDirectory() as temporary:
            overlay = FakeOverlay()
            runner = projection.OverlayProjection(overlay, overlay.allocate, scope="all")
            runtime = Flan(model_path, LegacyFabric(), query_projection=runner)
            versions = WeightVersionStore.from_flan(Path(temporary) / "versions.sqlite", model_path,
                                                    manifest_sha256=manifest)
            language = BoardLanguage(runtime, versions, temporary, lambda: snapshot)

            def run(result, capture_id=None):
                started = language.handle(dict(op="lm_start", question=question, capture_id=capture_id))
                deadline = monotonic() + 120
                while True:
                    actual = language.handle(dict(op="lm_result", job_id=started["job_id"]))
                    if actual["state"] != "running":
                        break
                    if monotonic() >= deadline:
                        self.fail("Software-backed generation exceeded the replay deadline.")
                    sleep(0.05)
                self.assertEqual(actual["state"], "complete", actual.get("error"))
                self.assertEqual(actual["answer"], result["answer"])
                self.assertEqual(actual["active"], result["active"])
                self.assertEqual(actual["fabric_projections"], result["fabric_projections"])
                self.assertEqual(actual["accelerated_projections"], result["fabric_projections"])
                self.assertFalse(overlay.buffers)
                return actual["capture_id"]

            capture_id = run(expected)
            if edit is not None:
                response, edited = edit
                name, row = edited["active"]["name"], edited["active"]["target_row"]
                language.handle(dict(op="lm_save", name=name, target_row=row, response=response))
                selected = language.handle(dict(op="lm_select", name=name))
                self.assertEqual(selected["active"], edited["active"])
                self.assertEqual(runtime.fabric.readback, bytes(n & 255 for n in response["row"]))
                # The exact edit is shared across platforms. Later tokens can diverge
                # because attention and normalization also use host floating point.
                reference = Flan(model_path, None, reference=True)
                reference.install_weight_version(versions.active())
                edited_answer = reference.generate(telemetry_prompt(question, analyst_evidence(snapshot)))
                self.assertEqual(edited_answer.split()[0], edited["answer"].split()[0])
                self.assertNotEqual(edited_answer, expected["answer"])
                run({**edited, "answer": edited_answer}, capture_id)
                restored = language.handle(dict(op="lm_restore"))
                self.assertEqual(restored["active"], expected["active"])
                width = versions.layout.width
                self.assertEqual(runtime.fabric.readback, versions.classifier[row*width:(row+1)*width].tobytes())
                run(expected, capture_id)
            runner.close()

    def test_real_checkpoint_original_replay(self):
        path = Path(__file__).resolve().parent / "evidence/three-diagnosis-traces-physical.json"
        report = json.loads(path.read_bytes())
        case = next(case for case in report["cases"] if case["case"] == "ambiguous-cause")
        self.replay(case["started"]["snapshot"], report["question"], case["result"])

    def test_real_checkpoint_saved_edit_and_rollback(self):
        path = os.getenv("PINKPROCESS_FLAN_EDIT_REPORT")
        if not path:
            self.skipTest("Set PINKPROCESS_FLAN_EDIT_REPORT to the saved physical edit report for replay.")
        report = json.loads(Path(path).read_bytes())
        baseline, edited = report["baseline"], report["edited"]
        self.replay(baseline["started"]["snapshot"], report["question"], baseline["result"],
                    edit=(report["astra"]["response"], edited["result"]))


if __name__ == "__main__":
    unittest.main()
