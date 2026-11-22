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
from unittest.mock import patch

import numpy as np

from kit import benchmark_projection_overlay as benchmark
from lm_reference import projection_argmax as argmax
from lm_reference import projection_overlay as projection


class Buffer(np.ndarray):
    def __array_finalize__(self, source):
        if hasattr(source, "device_address"):
            self.device_address = source.device_address + self.ctypes.data - source.ctypes.data
            self.events = source.events

    def flush(self):
        self.events.append(("flush", self.device_address))

    def invalidate(self):
        self.events.append(("invalidate", self.device_address))


class FakeOverlay:
    def __init__(self, *, stalled=False, queue=False):
        self.projection = SimpleNamespace(mmio=self)
        self.stalled = stalled
        self.registers = {projection.CORE_ID: projection.PROJECTION_ID, projection.CONTROL: 0}
        if queue:
            self.registers[projection.QUEUE_ID] = projection.PROJECTION_QUEUE_ID
        self.queued = []
        self.buffers = {}
        self.events = []
        self.allocations = []

    @contextmanager
    def allocate(self, *, shape, dtype):
        step = 64 * 1024 * 1024
        address = next(address for address in range(step, 64 * step, step) if address not in self.buffers)
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
        if register == projection.QUEUE_CONTROL:
            assert projection.QUEUE_ID in self.registers
            return len(self.queued)
        return self.registers[register]

    def write(self, register, value):
        self.registers[register] = value
        if register == projection.QUEUE_CONTROL:
            assert projection.QUEUE_ID in self.registers
            if value == projection.QUEUE_CLEAR:
                self.queued.clear()
            elif value == projection.QUEUE_PUSH:
                self.queued.append(self.registers.copy())
            elif value == projection.QUEUE_RUN:
                self.events.append(("queue_run", len(self.queued)))
                self.registers[projection.CONTROL] = projection.BUSY
                if self.stalled:
                    return
                for command in self.queued:
                    self.execute(command)
                self.queued.clear()
                self.registers[projection.CONTROL] = projection.DONE
            return
        if register != projection.CONTROL:
            return
        self.events.append(("launch",))
        if self.stalled:
            return  # Control retains bit zero (busy) until reset.
        self.execute(self.registers)
        self.registers[projection.CONTROL] = projection.DONE

    def execute(self, command):
        def region(address):
            for base, buffer in self.buffers.items():
                if base <= address < base + buffer.nbytes:
                    return buffer.reshape(-1)[(address - base) // buffer.itemsize:]
            raise AssertionError(f"Unallocated DDR address {address:#x}")

        a, w, y = (region(command[r]) for r in (
            projection.A_ADDRESS, projection.W_ADDRESS, projection.Y_ADDRESS))
        positions, outputs, depth, decode = (command[r] for r in (
            projection.POSITION_TILES, projection.OUTPUT_TILES, projection.DEPTH, projection.MODE))
        lanes = 64 if decode else 8
        a = a.reshape(-1)[:positions * depth * 8].reshape(positions, depth, 8)
        w = w.reshape(-1)[:outputs * depth * lanes].reshape(outputs, depth, lanes)
        y.reshape(-1)[:positions * outputs * 64] = benchmark.simulate_projection(a, w, decode).ravel()

    def download(self):
        self.events.append(("reset",))
        self.registers[projection.CONTROL] = 0


class ProjectionOverlayTest(unittest.TestCase):
    def setUp(self):
        random = np.random.default_rng(9)
        self.w = random.integers(-128, 128, (65, 19), dtype=np.int8)
        self.a = random.integers(-128, 128, (9, 19), dtype=np.int8)

    def test_segmented_head_uses_device_scores_and_preserves_first_tie(self):
        class ArgmaxMMIO:
            registers = {argmax.CONTROL: 0, argmax.CORE_ID: argmax.ARGMAX_ID}

            def read(self, address):
                return self.registers[address]

            def write(self, address, value):
                self.registers[address] = value
                if address != argmax.CONTROL:
                    return
                rows = self.registers[argmax.ROWS]
                scales = overlay.buffers[self.registers[argmax.SCALE_ADDRESS]][:rows]
                dots = overlay.buffers[self.registers[argmax.SCORE_ADDRESS]][:rows]
                activation = np.uint32(self.registers[argmax.ACTIVATION_SCALE]).view(np.float32)
                logits = dots.astype(np.float32) * activation * scales
                winner = int(np.argmax(logits))
                self.registers.update({argmax.CONTROL: argmax.DONE, argmax.WINNER: winner,
                                       argmax.WINNER_SCORE: int(logits[winner].view(np.uint32))})

        overlay = FakeOverlay()
        reducer = ArgmaxMMIO()
        overlay.projection_argmax = SimpleNamespace(mmio=reducer)
        self.assertIs(argmax.require_argmax_mmio(overlay), reducer)
        runner = projection.OverlayProjection(overlay, overlay.allocate, scope="all", argmax_mmio=reducer,
                                              resident_head_mib=1)
        self.addCleanup(runner.close)
        weights = np.tile(np.array([8, -1], dtype=np.int8), (129, 1))
        runner.bind_model({"lm_head.weight": weights})
        activations = np.array([[1, 0]], dtype=np.int8)
        scales = np.ones(129, dtype=np.float32)
        scales[-1] = scales[64] = 4
        with patch.object(projection, "MAX_SEGMENT_BUFFER_BYTES", 256):
            self.assertEqual(runner.head_argmax(weights, activations, np.float32(0.5), scales), 64)
            self.assertEqual(runner.profile()["argmax_commands"], 3)
            self.assertFalse(any(event[0] == "invalidate" for event in overlay.events))
            scales[-1] = 5
            self.assertEqual(runner.head_argmax(weights, activations, np.float32(0.5), scales), 128)
            self.assertEqual(runner.profile()["resident_weight_hits"], 3)

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
        self.assertEqual(len(overlay.allocations), 3)
        self.assertEqual(len(overlay.buffers), 3)
        runner.close()
        self.assertFalse(overlay.buffers)

    def test_resident_full_head_uploads_once_and_tracks_active_weights(self):
        overlay = FakeOverlay(queue=True)
        runner = projection.OverlayProjection(overlay, overlay.allocate, scope="all", queue=True,
                                              resident_head_mib=16)
        self.addCleanup(runner.close)
        weights = np.full((32128, 512), -128, dtype=np.int8)
        activations = np.full((1, 512), -128, dtype=np.int8)
        runner.bind_model({"lm_head.weight": weights})
        first = runner.project("lm_head", weights, activations)
        self.assertEqual(runner.profile()["resident_weight_uploads"], 8)
        self.assertEqual(runner.profile()["resident_weight_bytes"], weights.nbytes)
        self.assertEqual(runner.profile()["weight_bytes"], weights.nbytes)
        addresses = {buffer.device_address for buffer, _ in runner.resident_head.values()}
        runner.start_generation()
        np.testing.assert_array_equal(runner.project("lm_head", weights, activations), first)
        self.assertEqual(runner.profile()["resident_weight_hits"], 8)
        self.assertEqual(runner.profile().get("weight_bytes", 0), 0)
        self.assertIn(overlay.registers[projection.W_ADDRESS], addresses)
        # A queued group can continue using its ordinary scratch alongside residents.
        a = self.a[:1]
        for result in runner.project_many([("q", self.w), ("k", self.w)], a):
            np.testing.assert_array_equal(result, a.astype(np.int32) @ self.w.astype(np.int32).T)
        self.assertEqual(runner.profile()["queue_runs"], 1)
        weights[4096] = 127
        runner.invalidate_weights("lm_head")
        self.assertTrue(addresses.isdisjoint(overlay.buffers))
        edited = runner.project("lm_head", weights, activations)
        self.assertEqual(edited[0, 4096], -128 * 127 * 512)
        self.assertEqual(np.count_nonzero(edited != first), 1)

    def test_resident_allocation_failure_keeps_scratch_and_does_not_retry_each_token(self):
        overlay = FakeOverlay()
        attempts = []

        def allocate(*, shape, dtype):
            if len(shape) == 3:
                attempts.append(shape)
                if len(attempts) > 1:
                    raise RuntimeError("Simulated CMA exhaustion")
            return overlay.allocate(shape=shape, dtype=dtype)

        runner = projection.OverlayProjection(overlay, allocate, scope="all", resident_head_mib=2)
        self.addCleanup(runner.close)
        runner.bind_model({"lm_head.weight": self.w})
        expected = self.a[:1].astype(np.int32) @ self.w.astype(np.int32).T
        with patch.object(projection, "MAX_SEGMENT_BUFFER_BYTES", 19 * 64):
            for _ in range(3):
                np.testing.assert_array_equal(runner.project("lm_head", self.w, self.a[:1]), expected)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(runner.profile()["resident_allocation_failures"], 1)
        self.assertEqual(runner.profile()["resident_weight_hits"], 2)

    def test_resident_timeout_keeps_buffers_through_weight_invalidation(self):
        overlay = FakeOverlay()
        runner = projection.OverlayProjection(overlay, overlay.allocate, scope="all", timeout=0,
                                              resident_head_mib=2)
        runner.bind_model({"lm_head.weight": self.w})
        runner.project("lm_head", self.w, self.a[:1])
        addresses = set(overlay.buffers)
        overlay.stalled = True
        try:
            with self.assertRaises(TimeoutError):
                runner.project("lm_head", self.w, self.a[:1])
            runner.invalidate_weights("lm_head")
            runner.bind_model({"lm_head.weight": self.w.copy()})
            self.assertEqual(set(overlay.buffers), addresses)
            self.assertTrue(runner.resident_head)
        finally:
            overlay.registers[projection.CONTROL] = projection.DONE
            runner.close()
        self.assertFalse(overlay.buffers)

    def test_resident_is_evicted_once_if_ordinary_workspace_needs_cma(self):
        overlay = FakeOverlay()
        failures = []

        def allocate(*, shape, dtype):
            if len(shape) == 1 and dtype == np.int32 and shape[0] > 128 and runner.resident_head:
                failures.append(shape)
                raise OSError("Simulated CMA exhaustion")
            return overlay.allocate(shape=shape, dtype=dtype)

        runner = projection.OverlayProjection(overlay, allocate, scope="all", resident_head_mib=1)
        self.addCleanup(runner.close)
        runner.bind_model({"lm_head.weight": self.w})
        head = runner.project("lm_head", self.w, self.a[:1])
        self.assertTrue(runner.resident_head)
        expected = self.a.astype(np.int32) @ self.w.astype(np.int32).T
        np.testing.assert_array_equal(runner.project("encoder", self.w, self.a), expected)
        np.testing.assert_array_equal(runner.project("lm_head", self.w, self.a[:1]), head)
        self.assertEqual(len(failures), 1)
        self.assertEqual(runner.profile()["resident_workspace_fallbacks"], 1)
        self.assertFalse(runner.resident_head)

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
        self.assertTrue(parse_args(base + ["--projection-overlay", "--projection-scope", "all",
                                          "--projection-queue"]).projection_queue)
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parse_args(base + ["--projection-overlay", "--projection-queue"])
        resident = ["--projection-resident-head-mib", "16"]
        self.assertEqual(parse_args(base + ["--projection-overlay", "--projection-scope", "all"]
                                    + resident).projection_resident_head_mib, 16)
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parse_args(base + ["--projection-overlay"] + resident)

    def test_shared_input_legacy_and_queue(self):
        matrices = [("q", self.w), ("k", self.w[:17]), ("v", self.w[::-1])]
        for queued in (False, True):
            for activations in (self.a, self.a[:1]):
                with self.subTest(queued=queued, batch=len(activations)):
                    overlay = FakeOverlay(queue=queued)
                    runner = projection.OverlayProjection(overlay, overlay.allocate, scope="all", queue=queued)
                    results = runner.project_many(matrices, activations)
                    for y, (_, weights) in zip(results, matrices):
                        np.testing.assert_array_equal(y, activations.astype(np.int32) @ weights.astype(np.int32).T)
                    self.assertEqual(runner.profile()["commands"], 3)
                    self.assertEqual(runner.profile()["shared_activation_uploads_saved"], 2)
                    if queued:
                        self.assertEqual(overlay.events.count(("queue_run", 3)), 1)
                        self.assertEqual(runner.profile()["queue_runs"], 1)
                    else:
                        self.assertEqual(overlay.events.count(("launch",)), 3)
                    runner.close()

    def test_fused_decoder_rows_reuse_and_invalidate_combined_weight_layout(self):
        overlay = FakeOverlay(queue=True)
        runner = projection.OverlayProjection(overlay, overlay.allocate, scope="all", queue=True)
        self.addCleanup(runner.close)
        rng = np.random.default_rng(38)
        a = rng.integers(-127, 128, size=(1, 48), dtype=np.int8)
        matrices = [(name, rng.integers(-127, 128, size=(rows, 48), dtype=np.int8))
                    for name, rows in (("q", 64), ("k", 128), ("v", 64))]
        runner.bind_model({name + ".weight": matrix for name, matrix in matrices})
        for iteration in range(3):
            if iteration == 2:
                matrices[1][1][0] *= -1
                runner.invalidate_weights("k")
            for (_, w), result in zip(matrices, runner.project_many(matrices, a)):
                np.testing.assert_array_equal(result, a.astype(np.int32) @ w.astype(np.int32).T)
        phases = runner.profile()
        self.assertEqual(phases["commands"], 3)
        self.assertEqual(phases["fused_decoder_commands_saved"], 6)
        self.assertEqual(phases["weight_cache_hits"], 1)
        self.assertEqual(phases["weight_cache_misses"], 2)

    def test_queue_timeout_retains_all_regions(self):
        overlay = FakeOverlay(queue=True, stalled=True)
        runner = projection.OverlayProjection(overlay, overlay.allocate, scope="all", queue=True, timeout=0)
        with self.assertRaises(TimeoutError):
            runner.project_many([("q", self.w), ("k", self.w)], self.a)
        self.assertEqual(len(overlay.buffers), 3)
        self.assertEqual(overlay.events.count(("queue_run", 2)), 1)
        overlay.registers[projection.CONTROL] = projection.DONE
        runner.close()
        self.assertFalse(overlay.buffers)

    def replay(self, snapshot, question, expected, *, edit=None, queue=False):
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
            overlay = FakeOverlay(queue=queue)
            runner = projection.OverlayProjection(overlay, overlay.allocate, scope="all", queue=queue)
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
                self.assertEqual(len(overlay.buffers), 3)
                phases = actual["projection_phases"]
                self.assertGreaterEqual(phases["commands"] + phases.get("fused_decoder_commands_saved", 0),
                                        actual["accelerated_projections"])
                if queue:
                    self.assertGreater(phases["fused_decoder_commands_saved"], 0)
                    self.assertGreater(actual["projection_phases"]["queue_runs"], 0)
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
            self.assertFalse(overlay.buffers)

    def test_real_checkpoint_original_replay(self):
        path = Path(__file__).resolve().parent / "evidence/three-diagnosis-traces-physical.json"
        report = json.loads(path.read_bytes())
        case = next(case for case in report["cases"] if case["case"] == "ambiguous-cause")
        self.replay(case["started"]["snapshot"], report["question"], case["result"], queue=True)

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
