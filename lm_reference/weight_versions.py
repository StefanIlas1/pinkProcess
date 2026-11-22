"""Persist named INT8 output-row edits and verify their exact inference weights."""

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
import mmap
from pathlib import Path
import re
import sqlite3

from lm_reference.patch_operator import RowPatch
from lm_reference.quantize import CONFIG, EXP_MAX, EXP_STEPS
from lm_reference.reference import MODEL_HASH, QUANTIZED_MODEL_HASH, ROWS, WIDTH

ORIGINAL = "original"


@dataclass(frozen=True)
class WeightLayout:
    """Board-configured identity and location of the row-major INT8 output matrix."""

    rows: int
    width: int
    classifier_offset: int
    checkpoint_sha256: str
    model_sha256: str


# The PPQ16V1 classifier precedes the rotary and exponential INT32 tables.
_TINYSTORIES_CLASSIFIER_OFFSET = 425708 - ROWS * WIDTH - 4 * (
    CONFIG[-1] * (CONFIG[0] // CONFIG[3]) + EXP_MAX * EXP_STEPS + 1
)
TINYSTORIES = WeightLayout(
    ROWS, WIDTH, _TINYSTORIES_CLASSIFIER_OFFSET, QUANTIZED_MODEL_HASH, MODEL_HASH
)


def _digest(*parts):
    digest = sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


def _name(name):
    if (not isinstance(name, str) or not 1 <= len(name) <= 80 or name != name.strip()
            or not name.isprintable()):
        raise ValueError("Version names must contain 1 to 80 printable characters without surrounding spaces.")
    return name


@dataclass(frozen=True)
class WeightVersion:
    name: str
    checkpoint_sha256: str
    classifier_sha256: str
    patch: RowPatch | None
    _base: bytes | memoryview = field(repr=False, compare=False)
    _layout: WeightLayout = field(repr=False, compare=False)

    base_name: str = ORIGINAL

    @property
    def checkpoint(self):
        if self.patch is None:
            return bytes(self._base)
        start = self._layout.classifier_offset + self.patch.row * self._layout.width
        source = memoryview(self._base)
        return b"".join((source[:start], self.patch.payload, source[start + self._layout.width:]))

    def metadata(self):
        return {
            "name": self.name,
            "checkpoint_sha256": self.checkpoint_sha256,
            "classifier_sha256": self.classifier_sha256,
            "target_row": self.patch.row if self.patch is not None else None,
            **({"base_name": self.base_name} if self.base_name != ORIGINAL else {}),
        }


class WeightVersionStore:
    @classmethod
    def from_flan(cls, database, model_dir, *, manifest_sha256):
        """Open a FLAN-T5 store using a board-configured manifest SHA-256."""
        model_dir = Path(model_dir)
        manifest_bytes = (model_dir / "model.json").read_bytes()
        if _digest(manifest_bytes) != manifest_sha256:
            raise ValueError("FLAN-T5 manifest SHA-256 mismatch.")
        try:
            manifest = json.loads(manifest_bytes)
            classifier = manifest["tensors"]["lm_head.weight"]
            if manifest["model"] != "google/flan-t5-small" or classifier["dtype"] != "|i1":
                raise ValueError("Expected a FLAN-T5-small model with an INT8 output classifier.")
            rows, width = classifier["shape"]
            offset = classifier["offset"]
            weights_sha256 = manifest["weights_sha256"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid FLAN-T5 weights manifest.") from exc
        layout = WeightLayout(rows, width, offset, weights_sha256, manifest_sha256)
        # The installer publishes read-only, content-addressed model files. Both
        # this verifier and the runtime can share the kernel's clean file pages.
        with (model_dir / "weights.bin").open("rb") as source:
            if not source.seek(0, 2):
                raise ValueError("Base inference checkpoint is empty.")
            mapped = memoryview(mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ))
        return cls(database, mapped, layout=layout)

    def __init__(self, database, base_checkpoint, *, layout=TINYSTORIES):
        self.checkpoints = {}
        self.checkpoint_stores = {}
        self.database = Path(database)
        self.base = (base_checkpoint if isinstance(base_checkpoint, memoryview) and base_checkpoint.readonly
                     and isinstance(base_checkpoint.obj, mmap.mmap) else bytes(base_checkpoint))
        self.layout = layout
        offset = layout.classifier_offset
        if any(type(value) is not int for value in (layout.rows, layout.width, offset)):
            raise ValueError("Output classifier geometry must use integer values.")
        end = offset + layout.rows * layout.width
        if layout.rows < 1 or layout.width < 1 or offset < 0 or end > len(self.base):
            raise ValueError("Output classifier geometry exceeds checkpoint bounds.")
        if _digest(self.base) != layout.checkpoint_sha256:
            raise ValueError("Base inference checkpoint SHA-256 mismatch.")
        if not re.fullmatch(r"[0-9a-f]{64}", layout.model_sha256):
            raise ValueError("Invalid base model SHA-256.")
        self.classifier = memoryview(self.base)[offset:end]
        self.original = WeightVersion(
            ORIGINAL, layout.checkpoint_sha256, _digest(self.classifier), None, self.base, layout,
        )
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.identity = (layout.checkpoint_sha256, layout.model_sha256,
                         self.original.classifier_sha256, layout.rows, layout.width, offset)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS versions (
                    name TEXT PRIMARY KEY,
                    target_row INTEGER NOT NULL,
                    payload BLOB NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    checkpoint_sha256 TEXT NOT NULL,
                    classifier_sha256 TEXT NOT NULL
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS model_state (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    base_checkpoint_sha256 TEXT NOT NULL,
                    model_sha256 TEXT NOT NULL,
                    base_classifier_sha256 TEXT NOT NULL,
                    classifier_rows INTEGER NOT NULL,
                    classifier_width INTEGER NOT NULL,
                    classifier_offset INTEGER NOT NULL,
                    selected TEXT NOT NULL
                )
            """)
            connection.execute(
                "INSERT OR IGNORE INTO model_state VALUES (1, ?, ?, ?, ?, ?, ?, ?)",
                (*self.identity, ORIGINAL),
            )
            connection.execute("""CREATE TABLE IF NOT EXISTS full_checkpoints (
                name TEXT PRIMARY KEY, checkpoint_sha256 TEXT NOT NULL, manifest_sha256 TEXT NOT NULL
            )""")
            connection.execute("""CREATE TABLE IF NOT EXISTS version_roots (
                name TEXT PRIMARY KEY REFERENCES versions(name), base_name TEXT NOT NULL REFERENCES full_checkpoints(name)
            )""")
            self._selected(connection)

    def register_checkpoint(self, name, source):
        """Expose a verified model while pinning this public name across restarts."""
        name = _name(name)
        if name.casefold() == ORIGINAL:
            raise ValueError("Expected a named, complete checkpoint.")
        identity = (source.layout.checkpoint_sha256, source.layout.model_sha256)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._selected(connection)
            if connection.execute("SELECT 1 FROM versions WHERE name=?", (name,)).fetchone():
                raise ValueError(f"A saved row edit already uses the name {name!r}.")
            connection.execute("INSERT OR IGNORE INTO full_checkpoints VALUES(?,?,?)", (name, *identity))
            if connection.execute("SELECT checkpoint_sha256,manifest_sha256 FROM full_checkpoints WHERE name=?",
                                  (name,)).fetchone() != identity:
                raise ValueError(f"A different checkpoint is pinned to {name!r}.")
        self.checkpoints[name] = replace(source.original, name=name, base_name=name)
        self.checkpoint_stores[name] = source

    def checkpoint_store(self, base_name):
        if base_name == ORIGINAL:
            return self
        if base_name not in self.checkpoint_stores:
            raise ValueError(f"Checkpoint {base_name!r} is unavailable.")
        return self.checkpoint_stores[base_name]

    def base_for_version(self, name):
        if name in self.checkpoints:
            return name
        if name == ORIGINAL:
            return ORIGINAL
        with self._connect() as connection:
            row = connection.execute("""SELECT r.base_name FROM versions v
                    LEFT JOIN version_roots r ON r.name=v.name WHERE v.name=?""", (name,)).fetchone()
        if row is None:
            raise ValueError(f"Unknown weight version {name!r}.")
        return row[0] or ORIGINAL

    @contextmanager
    def _connect(self):
        try:
            connection = sqlite3.connect(self.database)
            try:
                connection.execute("PRAGMA synchronous = FULL")
                with connection:
                    yield connection
            finally:
                connection.close()
        except sqlite3.Error as exc:
            raise RuntimeError(f"Weight version database error: {exc}") from exc

    def _selected(self, connection):
        state = connection.execute(
            """SELECT base_checkpoint_sha256, model_sha256, base_classifier_sha256,
                      classifier_rows, classifier_width, classifier_offset, selected
                 FROM model_state WHERE singleton = 1"""
        ).fetchone()
        if state is None or state[:-1] != self.identity:
            raise ValueError("Weight version store belongs to a different base checkpoint or classifier.")
        return state[-1]

    def _patch(self, row, payload):
        layout = self.layout
        if type(row) is not int or not 0 <= row < layout.rows:
            raise ValueError(f"Target row must be an integer from 0 to {layout.rows - 1}.")
        if not isinstance(payload, bytes) or len(payload) != layout.width or 128 in payload:
            raise ValueError(f"Replacement row must contain exactly {layout.width} symmetric INT8 weights.")
        start = row * layout.width
        return RowPatch(
            row, layout.model_sha256, self.original.classifier_sha256,
            _digest(self.classifier[start:start + layout.width]), payload, layout.checkpoint_sha256,
        )

    def _version(self, name, patch):
        layout = self.layout
        start = patch.row * layout.width
        end = start + layout.width
        offset = layout.classifier_offset
        checkpoint = memoryview(self.base)
        return WeightVersion(
            name,
            _digest(checkpoint[:offset + start], patch.payload, checkpoint[offset + end:]),
            _digest(self.classifier[:start], patch.payload, self.classifier[end:]),
            patch, self.base, layout,
        )

    def _get(self, connection, name):
        if name == ORIGINAL:
            return self.original
        if name in self.checkpoints:
            return self.checkpoints[name]
        record = connection.execute(
            """SELECT v.target_row,v.payload,v.payload_sha256,v.checkpoint_sha256,v.classifier_sha256,r.base_name
                 FROM versions v LEFT JOIN version_roots r ON r.name=v.name WHERE v.name = ?""", (name,),
        ).fetchone()
        if record is None:
            raise ValueError(f"Unknown weight version {name!r}.")
        row, payload, payload_sha256, checkpoint_sha256, classifier_sha256, root = record
        store = self.checkpoint_store(root or ORIGINAL)
        patch = store._patch(row, payload)
        version = replace(store._version(name, patch), base_name=root or ORIGINAL)
        if (payload_sha256, checkpoint_sha256, classifier_sha256) != (
            _digest(payload), version.checkpoint_sha256, version.classifier_sha256
        ):
            raise ValueError(f"Weight version {name!r} failed SHA-256 verification.")
        return version

    def get(self, name):
        with self._connect() as connection:
            connection.execute("BEGIN")
            self._selected(connection)
            return self._get(connection, _name(name))

    def active(self):
        with self._connect() as connection:
            connection.execute("BEGIN")
            return self._get(connection, self._selected(connection))

    def list(self):
        with self._connect() as connection:
            connection.execute("BEGIN")
            self._selected(connection)
            records = connection.execute(
                """SELECT v.name,v.checkpoint_sha256,v.classifier_sha256,v.target_row,r.base_name
                   FROM versions v LEFT JOIN version_roots r ON r.name=v.name ORDER BY v.name"""
            ).fetchall()
        keys = ("name", "checkpoint_sha256", "classifier_sha256", "target_row")
        return [self.original.metadata(), *(version.metadata() for version in self.checkpoints.values()),
                *({**dict(zip(keys, row)), **({"base_name": row[4]} if row[4] else {})}
                  for row in records if row[4] is None or row[4] in self.checkpoints)]

    def save(self, name, patch, *, base_name=ORIGINAL):
        """Save an immutable version. Repeating the same save is safe."""
        name = _name(name)
        if name.casefold() == ORIGINAL:
            raise ValueError("The original weight version is reserved.")
        store = self.checkpoint_store(base_name)
        if patch != store._patch(patch.row, patch.payload):
            raise ValueError("Patch targets a different checkpoint, classifier or original row.")
        version = replace(store._version(name, patch), base_name=base_name)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._selected(connection)
            if connection.execute("SELECT 1 FROM full_checkpoints WHERE name=?", (name,)).fetchone():
                raise ValueError(f"The name {name!r} belongs to a full checkpoint.")
            existing = connection.execute("SELECT 1 FROM versions WHERE name = ?", (name,)).fetchone()
            if existing:
                if self._get(connection, name) != version:
                    raise ValueError(f"Weight version {name!r} already exists.")
            else:
                connection.execute(
                    "INSERT INTO versions VALUES (?, ?, ?, ?, ?, ?)",
                    (name, patch.row, patch.payload, _digest(patch.payload),
                     version.checkpoint_sha256, version.classifier_sha256),
                )
                if base_name != ORIGINAL:
                    connection.execute("INSERT INTO version_roots VALUES(?,?)", (name, base_name))
        return version

    def save_astra(self, name, target_row, response, *, base_name=ORIGINAL):
        store = self.checkpoint_store(base_name)
        weights = response.get("row") if isinstance(response, dict) else None
        if (not isinstance(weights, list) or len(weights) != store.layout.width
                or any(type(value) is not int or not -127 <= value <= 127 for value in weights)):
            raise ValueError(
                f"Astra response must contain {store.layout.width} signed INT8 values under the row key."
            )
        if "target_row" in response and (
            type(response["target_row"]) is not int or response["target_row"] != target_row
        ):
            raise ValueError("Astra response targets a different output row.")
        if response.get("base_sha256", store.layout.checkpoint_sha256) != store.layout.checkpoint_sha256:
            raise ValueError("Astra response targets a different base checkpoint.")
        return self.save(name, store._patch(target_row, bytes(value & 255 for value in weights)), base_name=base_name)

    def select(self, name):
        """Record the verified version to reload on the next start."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._selected(connection)
            version = self._get(connection, _name(name))
            connection.execute("UPDATE model_state SET selected = ? WHERE singleton = 1", (version.name,))
        return version
