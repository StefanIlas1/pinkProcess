"""Numerical regressions for the board-only FLAN projection interface."""
import hashlib
import unittest
from types import SimpleNamespace

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "The embedded language runtime requires NumPy.")
class FlanProjectionTests(unittest.TestCase):
    def setUp(self):
        from lm_reference.flan import Flan

        self.model = Flan.__new__(Flan)
        self.weights = np.array([[3, -11, 4, -13], [6, 12, -1, 8]], dtype="int8")
        self.model.t = {"lm_head.weight": self.weights.copy(),
                        "lm_head.weight.scale": np.array([0.3, 0.25], dtype="float32")}
        self.model._original = None
        self.model.query_projection = None
        self.model.first_output_activation = None
        self.model.first_output_max = None
        self.model.fabric_projections = 0
        self.model.accelerated_projections = 0

    def test_signed_activations_and_patched_matrix_reach_the_fabric(self):
        model = self.model

        class Fabric:
            def matmat(self, w, q, rows):
                self.last = np.array(q)
                return q.astype("int32") @ np.frombuffer(w, dtype="int8").reshape(rows, -1).T

        model.fabric = Fabric()
        inputs = np.array([[-1., 0.25, -0.5, 1.]], dtype="float32")
        base = model.linear("lm_head", inputs)
        self.assertAlmostEqual(model.first_output_max, float(base.max() * 127), places=3)
        self.assertEqual(model.first_output_activation, [-127, 32, -64, 127])
        model.select_head_row(0, bytes([1, 1, 1, 1]))
        patched = model.linear("lm_head", inputs)
        self.assertNotEqual(patched[0, 0], base[0, 0])
        self.assertEqual(patched[0, 1], base[0, 1])
        model.restore_head()
        self.assertTrue(np.array_equal(base, model.linear("lm_head", inputs)))
        self.assertEqual(model.fabric_projections, 3)

    def test_bad_version_digest_restores_original_projection(self):
        version = SimpleNamespace(patch=SimpleNamespace(row=1, payload=bytes(4)), classifier_sha256="0"*64)
        with self.assertRaises(ValueError):
            self.model.install_weight_version(version)
        digest = hashlib.sha256(memoryview(self.weights)).hexdigest()
        self.assertEqual(self.model.install_weight_version(SimpleNamespace(patch=None, classifier_sha256=digest)), digest)

    def test_named_accelerator_preserves_scaling_and_legacy_fallback(self):
        from kit.test_benchmark_projection_overlay import FakeOverlay
        from lm_reference.projection_overlay import QueryProjection

        class Legacy:
            calls = 0

            def matmat_array(self, weights, activations, rows):
                self.calls += 1
                matrix = np.frombuffer(weights, dtype=np.int8).reshape(rows, -1)
                return activations.astype(np.int32) @ matrix.astype(np.int32).T

        model = self.model
        model.fabric = Legacy()
        overlay = FakeOverlay()
        names = ("encoder.block.0.layer.0.SelfAttention.q", "decoder.block.0.layer.0.SelfAttention.q",
                 "encoder.block.0.layer.0.SelfAttention.k", "encoder.block.1.layer.0.SelfAttention.q")
        inputs = np.array([[-1., 0.25, -0.5, 1.], [0.32, -0.8, 0.5, 0.4]], dtype=np.float32)
        for name in names:
            model.t[name + ".weight"] = model.t["lm_head.weight"]
            model.t[name + ".weight.scale"] = model.t["lm_head.weight.scale"]
        batches = (inputs, inputs[:1], inputs, inputs)
        expected = [model.linear(name, batch) for name, batch in zip(names, batches)]
        model.query_projection = QueryProjection(overlay, overlay.allocate)
        for name, batch, reference in zip(names, batches, expected):
            np.testing.assert_array_equal(model.linear(name, batch), reference)
        self.assertEqual(overlay.events.count(("launch",)), 2)
        self.assertEqual(model.fabric.calls, 6)
        self.assertEqual(model.fabric_projections, 8)
        self.assertEqual(model.accelerated_projections, 2)
