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
        self.model.first_output_activation = None
        self.model.first_output_max = None
        self.model.fabric_projections = 0

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
