import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from threading import Thread
from time import monotonic, sleep
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from lm_reference.projection_native import NativeProjection, _load_library
from lm_reference.projection_overlay import pack_projection, unpack_projection


class ProjectionNativeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which(os.environ.get("CXX", "c++"))
        if compiler is None:
            raise unittest.SkipTest("A C++ compiler is required to test the optional native helper.")
        staging = TemporaryDirectory()
        cls.addClassCleanup(staging.cleanup)
        root = Path(__file__).resolve().parents[1]
        library = Path(staging.name) / "projection_native.so"
        subprocess.run([compiler, "-std=c++11", "-O3", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                        str(root / "fpga/projection_native.cpp"), "-o", str(library)], check=True)
        cls.library = _load_library(library)
        cls.addClassCleanup(_load_library.cache_clear)

    def setUp(self):
        self.registers = np.zeros(9, dtype=np.uint32)
        self.native = NativeProjection(self.registers, self.library)

    def test_load_recognizes_mapping_and_optional_library(self):
        with patch("lm_reference.projection_native._load_library", return_value=self.library):
            self.assertIsNotNone(NativeProjection.load(SimpleNamespace(array=self.registers)))
            self.assertIsNone(NativeProjection.load(SimpleNamespace(array=self.registers[::2])))
        with patch("lm_reference.projection_native._load_library", return_value=None):
            self.assertIsNone(NativeProjection.load(SimpleNamespace(array=self.registers)))

    def test_pack_and_unpack_match_numpy_with_signed_padding(self):
        random = np.random.default_rng(912)
        for batch, rows, depth in ((1, 65, 37), (9, 15, 33), (16, 16, 1), (1, 32128, 512)):
            with self.subTest(batch=batch, rows=rows, depth=depth):
                source_activations = random.integers(-128, 128, (batch, depth * 2), dtype=np.int8)[:, ::-2]
                source_weights = random.integers(-128, 128, (rows, depth), dtype=np.int8)[::-1]
                a, w = pack_projection(source_weights, source_activations)
                a_native, w_native = np.full_like(a, 101), np.full_like(w, 101)
                self.native.pack_activations(a_native, source_activations)
                self.native.pack_weights(w_native, source_weights, batch == 1)
                np.testing.assert_array_equal(a, a_native)
                np.testing.assert_array_equal(w, w_native)
                scores = random.integers(-(1 << 31), 1 << 31, (len(a), len(w), 64), dtype=np.int32)
                out = np.zeros((batch, rows * 2), dtype=np.int32)[:, ::-2]
                self.native.unpack_scores(out, scores)
                np.testing.assert_array_equal(out, unpack_projection(scores, batch, rows))

    def test_launch_programs_arguments_and_returns_status(self):
        def finish_after_launch():
            deadline = monotonic() + 1
            while self.registers[0] != 1 and monotonic() < deadline:
                sleep(0.0001)
            if self.registers[0] == 1:
                self.registers[0] = 4

        thread = Thread(target=finish_after_launch)
        thread.start()
        try:
            status = self.native.dispatch([64, 128, 256], True, 512, 1, 7, 1)
        finally:
            thread.join()
        self.assertEqual(status, 4)
        np.testing.assert_array_equal(self.registers[1:8], [1, 512, 1, 7, 64, 128, 256])

    def test_busy_and_timeout(self):
        self.registers[0] = 1
        with self.assertRaisesRegex(RuntimeError, "already busy"):
            self.native.dispatch([64, 128, 256], False, 2, 1, 1, 0.01)
        self.assertEqual(self.registers[2], 0)
        self.registers[0] = 0
        with self.assertRaises(TimeoutError):
            self.native.dispatch([64, 128, 256], False, 2, 1, 1, 0.01)


if __name__ == "__main__":
    unittest.main()
