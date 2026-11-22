"""A board twin on the Mac: the real board language service, INT8 checkpoints and queue scoring, run in software."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from threading import Thread

from .board_language import HISTORY_MODEL, BoardLanguage
from .board_sim import SimulatedBoard
from .kit_dataset import WINDOW_SIZE, dataset_snapshot
from .telemetry_device import SCORES, TelemetryDevice

ROOT = Path(__file__).resolve().parent.parent
PROFILES = ROOT / "profiles"
SEARCH_DIRS = (Path.home() / "Library" / "Application Support" / "PYNQ Process" / "checkpoints", ROOT / "build" / "flan")
CHECKPOINTS = (
    ("original", "diagnostic-install", "14d7912c66a78edf407bbb272f7aa04b953af920827a9a035275194ec4f2f2f2"),
    (HISTORY_MODEL, "queue-history-v2", "d8d97a3327aae96859b3ea5c165ccc860f06d55fe38a79d03c50da9203b22aeb"),
    ("Telemetry v1", "resident", "8ef4344f91f034573a6c39d2d6607a1177ba7deb2404e3b09523ef01388f3807"),
)
WARM_QUESTION = "What's wrong with my system?"


class SoftwareFabric:
    def __init__(self):
        self.mmio = self
        self.rows = {}
        self.scores = {}

    def write(self, offset, row):
        self.rows[offset // 64] = bytes(row)

    def read_row(self, index):
        return self.rows[index]

    def submit(self, activation):
        lanes = memoryview(bytes(activation)).cast("b")
        self.scores = {index: sum(x * y for x, y in zip(lanes, memoryview(row).cast("b"))) for index, row in self.rows.items()}

    def read(self, register):
        return self.scores[(register - SCORES) // 4] & 0xffffffff


def checkpoint_dirs(search=None):
    search = [Path(os.environ["PYNQ_CHECKPOINTS"])] if os.environ.get("PYNQ_CHECKPOINTS") else list(search or SEARCH_DIRS)
    found = {}
    for name, folder, manifest in CHECKPOINTS:
        for base in search:
            if (base / folder / "model.json").exists() and (base / folder / "weights.bin").exists():
                found[name] = (base / folder, manifest)
                break
    return found


class TwinLanguage(BoardLanguage):
    def __init__(self, runtimes, versions, state_dir, snapshot, dataset_score):
        self.runtimes = runtimes
        super().__init__(runtimes["original"], versions, state_dir, snapshot, dataset_score=dataset_score,
                         history_runtime=runtimes.get(HISTORY_MODEL))

    def _install(self, version, previous=None):
        self.runtime = self.runtimes.get(version.base_name, self.base_runtime)
        self.runtime.install_weight_version(version)


class BoardTwin(SimulatedBoard):
    def __init__(self, found, state_dir=None, seed=11):
        super().__init__(seed)
        from lm_reference.flan import Flan
        from lm_reference.weight_versions import WeightVersionStore

        self.state_dir = Path(state_dir or tempfile.mkdtemp(prefix="pynq-twin-"))
        self.device = TelemetryDevice(SoftwareFabric(), PROFILES, self.state_dir / "active.json")
        path, manifest = found["original"]
        versions = WeightVersionStore.from_flan(self.state_dir / "lm-versions.sqlite", path, manifest_sha256=manifest)
        runtimes = {"original": Flan(path, None, reference=True)}
        for name, (path, manifest) in found.items():
            if name == "original":
                continue
            versions.register_checkpoint(name, WeightVersionStore.from_flan(
                self.state_dir / f"lm-{len(runtimes)}.sqlite", path, manifest_sha256=manifest))
            runtimes[name] = Flan(path, None, reference=True)
        self.language = TwinLanguage(runtimes, versions, self.state_dir, self.live_snapshot, self.score)
        self.warm = Thread(target=self.preload, daemon=True)
        self.warm.start()

    def preload(self):
        try:
            self.language.runtime.generate("Answer the question using the measurements. Current queue: 0 waiting requests.\n"
                                           f"Question: {WARM_QUESTION}\nAnswer:", max_tokens=8)
        except Exception:
            pass

    def score(self, window, capacity):
        return self.device._score_stream_window(window, capacity, stream=False)

    def live_snapshot(self):
        position = max(self.index(), WINDOW_SIZE)
        window = self.run.readings[position - WINDOW_SIZE:position + 1]
        text = "time_ms,busy,waiting\n" + "".join(f"{r.time_ms},{r.busy},{r.waiting}\n" for r in window)
        snapshot = dataset_snapshot("live", "Simulated connection pool", text, self.run.capacity, self.score)
        snapshot.update(origin="demo", fresh=True, session="sim-session", age_ms=200, limitations=[
            "Measurements alone do not identify a cause or request-level latency."])
        return snapshot

    def status(self):
        return {**super().status(), "backend": "PYNQ-Z2", "lm_supported": True}

    def request(self, action, **params):
        if action.startswith("lm_"):
            try:
                return self.language.handle(dict(op=action, **params))
            except (ValueError, KeyError) as exc:
                return dict(ok=False, error=str(exc))
        if action in ("stream_observations", "stream_forecast", "stream_evidence", "peak", "trend"):
            return self.live_snapshot()
        return super().request(action, **params)


def make_board(simulated_only=False):
    if not simulated_only:
        found = checkpoint_dirs()
        if "original" in found:
            try:
                return BoardTwin(found)
            except (ImportError, OSError, ValueError, RuntimeError):
                pass
    return SimulatedBoard()
