"""Integer inference over cartridges: the arithmetic the FPGA kernel performs."""

import json
import time
from array import array
from dataclasses import dataclass
from pathlib import Path

from .features import encode
from .resources import WEIGHTS
REFUSE = "refuse"


@dataclass(frozen=True)
class Cartridge:
    name: str
    document: dict
    labels: tuple[str, ...]
    bias: tuple[int, ...]
    weights: tuple[tuple[int, ...], ...]

    @classmethod
    def load(cls, path: Path) -> "Cartridge":
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("schema_version") != 1:
            raise ValueError("Unsupported cartridge contract.")
        return cls(document["name"], document, tuple(document["labels"]), tuple(document["bias"]),
                   tuple(tuple(row) for row in document["weights"]))

    def scores(self, features: list[tuple[int, int]]) -> list[int]:
        totals = list(self.bias)
        for index, count in features:
            row = self.weights[index]
            for slot in range(len(totals)):
                totals[slot] += count * row[slot]
        return totals

    def decide(self, features: list[tuple[int, int]]) -> tuple[str, int, list[int]]:
        """Best card, its integer margin over the refuse class, and every score."""
        totals = self.scores(features)
        refuse_slot = self.labels.index(REFUSE)
        best = max(range(refuse_slot), key=totals.__getitem__)
        return self.labels[best], totals[best] - totals[refuse_slot], totals


class Device:
    """One kernel, one loaded cartridge at a time. Loading a cartridge is the domain switch."""

    backend = "software replay"

    def __init__(self, directory: Path = WEIGHTS):
        self.library = {path.stem: Cartridge.load(path) for path in sorted(directory.glob("*.json"))}
        self.images = {name: array("b", [value for row in item.weights for value in row])
                       for name, item in self.library.items()}
        self.active = next(iter(self.library))
        self.memory = array("b", self.images[self.active])
        self.switch_microseconds = 0.0

    def load(self, name: str) -> None:
        """Overwrite the kernel's weight memory with another cartridge's image."""
        started = time.perf_counter()
        self.memory[:] = self.images[name]
        self.active = name
        self.switch_microseconds = (time.perf_counter() - started) * 1e6

    def run(self, features: list[tuple[int, int]]) -> tuple[str, int, list[int]]:
        cartridge = self.library[self.active]
        slots = len(cartridge.bias)
        totals = list(cartridge.bias)
        for index, count in features:
            row = self.memory[index * slots:(index + 1) * slots]
            for slot in range(slots):
                totals[slot] += count * row[slot]
        refuse_slot = cartridge.labels.index(REFUSE)
        best = max(range(refuse_slot), key=totals.__getitem__)
        return cartridge.labels[best], totals[best] - totals[refuse_slot], totals

    def ask(self, question: str, cartridge: str = "auto") -> dict:
        features = encode(question)
        started = time.perf_counter()
        previous = self.active
        if cartridge == "auto":
            candidates = []
            for name in self.library:
                self.load(name)
                candidates.append((self.run(features), name))
            (card, margin, totals), chosen = max(candidates, key=lambda item: item[0][1])
        else:
            chosen = cartridge
        self.load(chosen)
        card, margin, totals = self.run(features)
        elapsed = (time.perf_counter() - started) * 1e6
        active = self.library[chosen]
        answered = margin > 0 and bool(features)
        return {
            "question": question,
            "cartridge": chosen,
            "switched": cartridge == "auto" and chosen != previous,
            "answered": answered,
            "card": {"id": card, **active.document["cards"][card]} if answered else None,
            "margin": margin,
            "scores": dict(zip(active.labels, totals)),
            "active_features": len(features),
            "microseconds": round(elapsed, 1),
            "backend": self.backend,
        }

    def describe(self) -> dict:
        return {
            "backend": self.backend,
            "active": self.active,
            "cartridges": [
                {key: item.document[key] for key in ("name", "title", "tagline", "notice", "weight_bytes", "examples")}
                | {"cards": len(item.document["cards"])}
                for item in self.library.values()
            ],
        }
