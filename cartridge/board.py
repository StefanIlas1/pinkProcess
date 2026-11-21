"""Run the kernel on a PYNQ-Z2 overlay. Not yet exercised on hardware."""

import json
import time
from pathlib import Path

from .runtime import REFUSE, ROOT, Device

BITSTREAM = ROOT / "hw" / "build" / "cartridge_kernel.bit"
OFFSETS = ROOT / "hw" / "build" / "offsets.json"


def available() -> bool:
    try:
        import pynq  # noqa: F401
    except ImportError:
        return False
    return BITSTREAM.exists() and OFFSETS.exists()


class BoardDevice(Device):
    """Same decisions as Device, with scores computed in the programmable logic."""

    backend = "PYNQ-Z2 overlay"

    def __init__(self, directory: Path = ROOT / "cartridges"):
        from pynq import Overlay

        super().__init__(directory)
        self.offsets = json.loads(OFFSETS.read_text(encoding="utf-8"))
        self.kernel = Overlay(str(BITSTREAM)).cartridge_kernel_0
        self.load(self.active)

    def load(self, name: str) -> None:
        started = time.perf_counter()
        cartridge = self.library[name]
        packed = bytes(value & 0xFF for row in cartridge.weights for value in row)
        self.kernel.mmio.write(self.offsets["weights"], packed)
        for slot, value in enumerate(cartridge.bias):
            self.kernel.mmio.write(self.offsets["bias"] + 4 * slot, value & 0xFFFFFFFF)
        self.kernel.mmio.write(self.offsets["refuse_slot"], cartridge.labels.index(REFUSE))
        self.active = name
        self.switch_microseconds = (time.perf_counter() - started) * 1e6

    def run(self, features: list[tuple[int, int]]) -> tuple[str, int, list[int]]:
        cartridge = self.library[self.active]
        indices = [index for index, _ in features] + [0] * (64 - len(features))
        counts = [count for _, count in features] + [0] * (64 - len(features))
        self.kernel.mmio.write(self.offsets["feature_index"], b"".join(i.to_bytes(2, "little") for i in indices))
        self.kernel.mmio.write(self.offsets["feature_count"], bytes(counts))
        self.kernel.mmio.write(self.offsets["active"], len(features))
        self.kernel.mmio.write(0x00, 1)
        while not self.kernel.mmio.read(0x00) & 0x2:
            pass
        totals = []
        for slot in range(len(cartridge.bias)):
            raw = self.kernel.mmio.read(self.offsets["scores"] + 4 * slot)
            totals.append(raw - (1 << 32) if raw & 0x80000000 else raw)
        best = self.kernel.mmio.read(self.offsets["best_slot"]) & 0xFF
        if totals != cartridge.scores(features):
            raise RuntimeError("Overlay scores differ from the integer reference.")
        return cartridge.labels[best], totals[best] - totals[cartridge.labels.index(REFUSE)], totals
