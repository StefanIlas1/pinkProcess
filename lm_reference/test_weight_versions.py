from contextlib import closing
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from lm_reference.patch_operator import RowPatch
from lm_reference.weight_versions import ORIGINAL, WeightLayout, WeightVersionStore


class WeightVersionTest(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.database = Path(directory.name) / "versions.sqlite3"
        self.base = b"prefix" + bytes(range(12)) + b"suffix"
        self.layout = WeightLayout(3, 4, len(b"prefix"), sha256(self.base).hexdigest(), "b" * 64)
        self.store = self.reopen()

    def reopen(self):
        return WeightVersionStore(self.database, self.base, layout=self.layout)

    def test_named_versions_survive_restart_and_restore(self):
        first = self.store.save_astra("Astra pass 1", 1, {"row": [-127, -1, 0, 127]})
        second = self.store.save_astra("Astra pass 2", 2, {"row": [3, 1, 4, 1]})
        self.assertEqual(self.store.active().name, ORIGINAL)
        self.assertEqual([row["name"] for row in self.store.list()], [ORIGINAL, first.name, second.name])
        self.assertEqual(self.store.select(first.name), first)

        restarted = self.reopen()
        active = restarted.active()
        expected = self.base[:10] + bytes([129, 255, 0, 127]) + self.base[14:]
        self.assertEqual(active.checkpoint, expected)
        self.assertEqual(active.checkpoint_sha256, sha256(expected).hexdigest())
        self.assertEqual(active.classifier_sha256, sha256(expected[6:18]).hexdigest())
        self.assertEqual(restarted.select(ORIGINAL).checkpoint, self.base)
        original = self.reopen()
        self.assertEqual(original.active().name, ORIGINAL)
        self.assertEqual(original.get(first.name), first)
        self.assertEqual(original.get(second.name), second)

    def test_wrong_base_and_wrong_patch_do_not_save(self):
        with self.assertRaisesRegex(ValueError, "Base inference checkpoint SHA-256 mismatch"):
            WeightVersionStore(self.database, self.base[::-1], layout=self.layout)
        version = self.store.save_astra("First", 0, {"row": [1, 2, 3, 4]})
        patch = RowPatch(0, "a" * 64, version.patch.classifier_hash,
                         version.patch.original_row_hash, version.patch.payload,
                         version.patch.quantized_model_hash)
        with self.assertRaisesRegex(ValueError, "different checkpoint"):
            self.store.save("Wrong base", patch)
        self.assertEqual(self.store.save(version.name, version.patch), version)
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.store.save_astra(version.name, 0, {"row": [4, 3, 2, 1]})
        with self.assertRaisesRegex(ValueError, "reserved"):
            self.store.save_astra("Original", 0, {"row": [4, 3, 2, 1]})

    def test_flan_adapter_uses_pinned_manifest_and_its_weights_digest(self):
        model_dir = self.database.parent
        weights = model_dir / "weights.bin"
        weights.write_bytes(self.base)
        manifest = json.dumps({
            "model": "google/flan-t5-small",
            "weights_sha256": self.layout.checkpoint_sha256,
            "tensors": {"lm_head.weight": {"dtype": "|i1", "shape": [3, 4], "offset": 6}},
        }).encode()
        (model_dir / "model.json").write_bytes(manifest)
        database = model_dir / "flan.sqlite3"
        pin = sha256(manifest).hexdigest()
        store = WeightVersionStore.from_flan(database, model_dir, manifest_sha256=pin)
        self.assertIsInstance(store.base, memoryview)
        self.assertTrue(store.base.readonly)
        self.assertIsInstance(store.active().checkpoint, bytes)
        self.assertEqual(store.active().checkpoint, self.base)
        with self.assertRaisesRegex(ValueError, "manifest SHA-256 mismatch"):
            WeightVersionStore.from_flan(database, model_dir, manifest_sha256="0" * 64)
        weights.write_bytes(self.base[::-1])
        with self.assertRaisesRegex(ValueError, "checkpoint SHA-256 mismatch"):
            WeightVersionStore.from_flan(database, model_dir, manifest_sha256=pin)

    def test_corrupt_version_cannot_be_selected(self):
        version = self.store.save_astra("Corrupt", 0, {"row": [1, 2, 3, 4]})
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute("UPDATE versions SET payload = ?", (b"oops",))
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.store.select(version.name)
        self.assertEqual(self.reopen().active().name, ORIGINAL)

    def test_interrupted_select_preserves_active_version(self):
        version = self.store.save_astra("Interrupted", 0, {"row": [1, 2, 3, 4]})
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute("""CREATE TRIGGER interrupt_selection BEFORE UPDATE ON model_state
                                  BEGIN SELECT RAISE(ABORT, 'interrupted'); END""")
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.store.select(version.name)
        store = self.reopen()
        self.assertEqual(store.active().name, ORIGINAL)
        self.assertEqual(store.get(version.name), version)


if __name__ == "__main__":
    unittest.main()
