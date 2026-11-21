import hashlib
import io
import json
import shlex
import unittest
from unittest.mock import patch

from lm_reference.patch_operator import BoardStateError, MemoryClassifier, PatchSession, RowPatch, operate
from lm_reference.reference import MODEL_HASH, QUANTIZED_MODEL_HASH, ROWS, WIDTH


class RecordingClassifier(MemoryClassifier):
    def __init__(self):
        super().__init__()
        self.writes = []
        self.interrupt_write = False

    def write(self, address, values):
        self.writes.append((address, bytes(values)))
        if self.interrupt_write:
            self.interrupt_write = False
            self.memory[address:address + 4] = values[:4]
            raise OSError("injected interruption after one AXI word")
        super().write(address, values)


class PatchOperatorTest(unittest.TestCase):
    def setUp(self):
        self.row = 326
        self.base = bytes([3]) * (ROWS * WIDTH)
        self.payload = bytes([7]) * WIDTH
        self.classifier = RecordingClassifier()
        self.session = PatchSession(self.classifier, self.base)
        self.classifier.writes.clear()
        self.document = {
            "format": "pinkprocess.int8-output-row.v1",
            "target_row": self.row,
            "model_sha256": MODEL_HASH,
            "base_classifier_sha256": hashlib.sha256(self.base).hexdigest(),
            "original_row_sha256": hashlib.sha256(bytes([3]) * WIDTH).hexdigest(),
            "payload_bytes": WIDTH,
            "payload_sha256": hashlib.sha256(self.payload).hexdigest(),
            "payload_hex": self.payload.hex(),
        }
        self.valid = RowPatch.parse(self.document)

    def test_invalid_patches_never_write(self):
        invalid = [
            ("model_sha256", "0" * 64, "different base model"),
            ("base_classifier_sha256", "0" * 64, "different base classifier"),
            ("original_row_sha256", "0" * 64, "original row SHA-256"),
            ("payload_hex", self.payload[:-1].hex(), "exactly 64 bytes"),
            ("payload_sha256", "0" * 64, "payload SHA-256 mismatch"),
            ("target_row", ROWS, "Target row"),
        ]
        for field, value, message in invalid:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                self.session.apply(RowPatch.parse({**self.document, field: value}))
        self.assertEqual(self.classifier.writes, [])
        self.assertEqual(self.classifier.read_matrix(), self.base)

    def test_apply_retries_and_rollback(self):
        applied = self.session.apply(self.valid)
        self.assertEqual(applied, self.session.apply(self.valid))
        self.assertEqual(len(self.classifier.writes), 1)
        self.assertEqual(self.classifier.read_row(self.row), self.payload)
        self.assertEqual(self.session.verify()["observed_classifier_sha256"],
                         applied["expected_classifier_sha256"])
        restored = self.session.rollback()
        self.session.rollback()
        self.assertEqual(len(self.classifier.writes), 2)
        self.assertIsNone(restored["active_row"])
        self.assertEqual(self.classifier.read_matrix(), self.base)

    def test_partial_write_is_restored(self):
        self.classifier.interrupt_write = True
        with self.assertRaisesRegex(RuntimeError, "original row was restored"):
            self.session.apply(self.valid)
        self.assertIsNone(self.session.active)
        self.assertEqual(self.classifier.read_matrix(), self.base)

    def test_legacy_patch_cannot_change_the_integer_checkpoint(self):
        session = PatchSession(self.classifier, self.base, quantized_model_hash=QUANTIZED_MODEL_HASH)
        self.classifier.writes.clear()
        with self.assertRaisesRegex(ValueError, "different inference checkpoint"):
            session.apply(self.valid)
        self.assertEqual(self.classifier.writes, [])
        fixed = RowPatch.parse({**self.document, "quantized_model_sha256": QUANTIZED_MODEL_HASH})
        session.apply(fixed)
        self.assertEqual(self.classifier.read_row(self.row), self.payload)

    def test_shutdown_restores_active_patch(self):
        self.session.apply(self.valid)
        with patch("sys.stdin", io.StringIO()):
            operate(self.session, None)
        self.assertIsNone(self.session.active)
        self.assertEqual(self.classifier.read_matrix(), self.base)

    def test_generate_accepts_a_quoted_prompt_and_keeps_the_default(self):
        prompt = "Lily's rabbit found a key."
        commands = io.StringIO(shlex.join(["generate", prompt]) + "\ngenerate\n")
        output = io.StringIO()
        with patch("sys.stdin", commands), patch("sys.stdout", output):
            operate(self.session, lambda text=None: {"prompt": text})
        generated = [json.loads(line)["result"]["prompt"] for line in output.getvalue().splitlines()]
        self.assertEqual(generated, [prompt, None])

    def test_failed_restoration_keeps_target_for_recovery(self):
        self.session.apply(self.valid)
        self.classifier.interrupt_write = True
        with self.assertRaisesRegex(BoardStateError, "Could not verify restoration"):
            self.session.rollback()
        self.assertEqual(self.session.active, self.valid)
        self.session.rollback()
        self.assertEqual(self.classifier.read_matrix(), self.base)


if __name__ == "__main__":
    unittest.main()
