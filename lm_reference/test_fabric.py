import ctypes
from pathlib import Path
import random
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Lock
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fpga.check_classifier import COMMAND, INPUT, SCORES, STATUS, evaluate, simulate, writes
from lm_reference.fabric import ROWS, WIDTH, TiledMatvec
from lm_reference.live_demo import PynqClassifier

try:
    import numpy as np
except ImportError:
    np = None


def signed(values):
    return [value if value < 128 else value - 256 for value in values]


def reference(weights, activation, rows):
    columns = len(activation)
    return [sum(w * x for w, x in zip(signed(weights[row * columns:(row + 1) * columns]), activation))
            for row in range(rows)]


class OverlayModel:
    def __init__(self, lock=None, *, record=False):
        self.weights = bytearray(ROWS * WIDTH)
        self.lock = lock
        self.weight_revision = 0
        self.operations = [] if record else None
        self.calls = 0

    def write(self, address, values):
        assert self.lock is None or self.lock.locked()
        assert address % WIDTH == len(values) % WIDTH == 0
        assert address + len(values) <= len(self.weights)
        self.weights[address:address + len(values)] = values
        self.weight_revision += 1
        if self.operations is not None:
            self.operations.extend(writes(address, values))

    def score(self, activation):
        assert self.lock is None or self.lock.locked()
        assert len(activation) == WIDTH
        self.calls += 1
        scores = reference(self.weights, signed(activation), ROWS)
        if self.operations is not None:
            self.operations.extend(evaluate(activation, scores))
        return scores


class TiledMatvecTest(unittest.TestCase):
    def test_tiles_and_signed_wide_accumulators(self):
        lock = Lock()
        overlay = OverlayModel(lock)
        reserved = bytes(range(192))
        with lock:
            overlay.write(0, reserved)
        fabric = TiledMatvec(overlay, lock=lock)
        rows, columns = 513, 131
        generator = random.Random(42)
        weights = generator.randbytes(rows * columns)
        activation = [-(1 << 31), (1 << 31) - 1, -129, 128, 127, -128] * 21 + [0, 1, 0, 0, -1]
        expected = reference(weights, activation, rows)

        self.assertGreater(max(abs(score) for score in expected), 1 << 31)
        self.assertEqual(fabric.matvec(weights, activation, rows), expected)
        self.assertEqual(overlay.weight_revision, 7)
        self.assertEqual(overlay.weights[:len(reserved)], reserved)

    def test_batch_reuses_uploads_and_external_write_invalidates_resident_tile(self):
        overlay = OverlayModel()
        fabric = TiledMatvec(overlay)
        weights = bytes([255, 2, 3, 254])
        activations = [bytes([1, 254]), [128, -129], [0, 0]]
        expected = [[-5, 7], [-386, 642], [0, 0]]

        self.assertEqual(fabric.matmat(weights, activations, 2), expected)
        self.assertEqual(overlay.weight_revision, 1)
        self.assertEqual(overlay.calls, 3)
        overlay.write(3 * WIDTH, bytes(2 * WIDTH))
        self.assertEqual(fabric.matmat(weights, activations, 2), expected)
        self.assertEqual(overlay.weight_revision, 3)

    def test_disabling_reuse_and_rejecting_invalid_inputs(self):
        overlay = OverlayModel()
        fabric = TiledMatvec(overlay, reuse=False)
        for _ in range(2):
            self.assertEqual(fabric.matvec(bytes([128]), [2], 1), [-256])
        self.assertEqual(overlay.weight_revision, 2)
        with self.assertRaises(ValueError):
            fabric.matvec(b"\x01", [1 << 31], 1)
        with self.assertRaises(ValueError):
            fabric.matmat(b"\x01", [[1], [1, 2]], 1)

    @unittest.skipIf(np is None, "NumPy required for the array fast path")
    def test_array_path_matches_reference_and_pass_count(self):
        class ScoreIntoOverlay(OverlayModel):
            def score_rows_into(self, activation, first, output):
                output[:] = self.score(activation)[first:first + len(output)]

        rows, columns, batch = 513, 65, 4
        rng = np.random.default_rng(42)
        weights = rng.integers(-128, 128, (rows, columns), dtype=np.int8).tobytes()
        for dtype in (np.int8, np.int32):
            limits = np.iinfo(dtype)
            vectors = rng.integers(limits.min, limits.max, (batch, columns), dtype=dtype)
            vectors[0] = 0
            vectors[1, 0] = limits.min
            vectors[2, 0] = limits.max
            with self.subTest(dtype=dtype):
                lock = Lock()
                reference_overlay = OverlayModel(lock)
                array_overlay = ScoreIntoOverlay(lock)
                expected = TiledMatvec(reference_overlay, lock=lock).matmat(weights, vectors, rows)
                actual = TiledMatvec(array_overlay, lock=lock).matmat_array(weights, vectors, rows)
                self.assertEqual(actual.dtype, np.int64)
                self.assertEqual(actual.tolist(), expected)
                self.assertEqual(array_overlay.calls, reference_overlay.calls)
                self.assertEqual(array_overlay.weight_revision, 4)

    @unittest.skipUnless(shutil.which("iverilog") and shutil.which("vvp"), "Icarus Verilog required")
    def test_signed_planes_and_column_tiles_against_rtl(self):
        overlay = OverlayModel(record=True)
        fabric = TiledMatvec(overlay)
        # The RTL scores every row, including unused rows, so initialize the full BRAM.
        overlay.write(0, bytes(range(256)) * (ROWS * WIDTH // 256))
        rows = 2
        weights = bytes([128, 127]) * 65
        activation = [128, -129] * 32 + [65537]
        self.assertEqual(fabric.matvec(weights, activation, rows), reference(weights, activation, rows))
        simulate(overlay.operations)


@unittest.skipIf(np is None, "NumPy required to emulate the PYNQ memory map")
class PynqTransportTest(unittest.TestCase):
    def fake_pynq(self):
        array = np.zeros(STATUS // 4 + 2, dtype="<u4")
        scalar_reads = []
        scalar_writes = []

        def read(address):
            scalar_reads.append(address)
            return 2 if address == STATUS else int(array[address // 4])

        def write(address, value):
            scalar_writes.append((address, value))
            array[address // 4] = value

        mmio = SimpleNamespace(array=array, read=read, write=write)
        overlay = SimpleNamespace(classifier=SimpleNamespace(mmio=mmio))
        with patch.dict(sys.modules, {"pynq": SimpleNamespace(Overlay=lambda _: overlay)}):
            classifier = PynqClassifier("fake.bit")
        return classifier, array, scalar_reads, scalar_writes

    def test_weight_bulk_write_and_score_slice_leave_control_access_scalar(self):
        with patch("lm_reference.live_demo.platform.system", return_value="Darwin"):
            classifier, array, scalar_reads, scalar_writes = self.fake_pynq()

        tile = bytes(range(64))
        classifier.write(3 * WIDTH, tile)
        self.assertEqual(array[3 * WIDTH // 4:4 * WIDTH // 4].tobytes(), tile)
        self.assertFalse(scalar_writes)
        self.assertEqual(classifier.weight_revision, 1)
        classifier.write(4, b"\x11\x22\x33")
        self.assertEqual(scalar_writes, [(4, 0x00332211)])
        self.assertEqual(classifier.weight_revision, 2)

        # Score reads bypass the scalar API and preserve the signed word encoding.
        array[SCORES // 4 + 3:SCORES // 4 + 5] = [0xFFFFFFFF, 12345]
        scalar_reads.clear()
        scalar_writes.clear()
        self.assertEqual(classifier.score_rows(bytes(WIDTH), 3, 2), [-1, 12345])
        self.assertEqual(scalar_reads, [STATUS, STATUS])
        self.assertEqual(scalar_writes, [(address, 0) for address in range(INPUT, SCORES, 4)] + [(COMMAND, 1)])
        self.assertEqual(classifier.weight_revision, 2)
        self.assertEqual(len(classifier.score(bytes(WIDTH))), ROWS)

    @unittest.skipUnless(shutil.which("cc"), "C compiler required for the native MMIO test")
    def test_arm_word_helper_and_missing_library_fallback(self):
        with TemporaryDirectory() as parent:
            library = Path(parent) / "mmio_words.so"
            source = Path(__file__).resolve().parents[1] / "fpga" / "mmio_words.c"
            subprocess.run(["cc", "-O2", "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
            native = ctypes.CDLL(str(library))
            for helper in (native, None):
                with self.subTest(native=helper is not None):
                    with (patch("lm_reference.live_demo.platform.system", return_value="Linux"),
                          patch("lm_reference.live_demo.platform.machine", return_value="armv7l"),
                          patch("lm_reference.live_demo.ctypes.CDLL", return_value=helper,
                                side_effect=OSError if helper is None else None)):
                        classifier, array, scalar_reads, scalar_writes = self.fake_pynq()
                    classifier.write(4, b"\x11\x22\x33\x44")
                    self.assertEqual(array[1], 0x44332211)
                    self.assertEqual(scalar_writes, [] if helper else [(4, 0x44332211)])
                    array[SCORES // 4 + 3:SCORES // 4 + 5] = [0x80000000, 12345]
                    scalar_reads.clear()
                    result = np.zeros(2, dtype=np.int32)
                    classifier.score_rows_into(bytes(WIDTH), 3, result)
                    self.assertEqual(result.tolist(), [-2147483648, 12345])
                    expected_reads = [STATUS, STATUS]
                    if helper is None:
                        expected_reads.extend([SCORES + 12, SCORES + 16])
                    self.assertEqual(scalar_reads, expected_reads)


if __name__ == "__main__":
    unittest.main()
