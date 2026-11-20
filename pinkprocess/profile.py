"""Versioned, portable parameters for a connection-pool logistic model."""

import json
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path

from .dataset import FORECAST_MS, OBSERVATION_MS, Example
from .simulator import SAMPLE_MS

TREND_MS = 2_000
FULL_WINDOW_MS = 1_000
FEATURES = (
    "current_busy_fraction",
    "busy_fraction_change_per_second_over_2s",
    "fraction_of_last_1s_readings_at_capacity",
)
CONTRACT = {
    "schema_version": 1,
    "model": "pool_wait_logistic",
    "sample_ms": SAMPLE_MS,
    "observation_ms": OBSERVATION_MS,
    "forecast_ms": FORECAST_MS,
    "features": list(FEATURES),
}


def feature_ratios(example: Example, capacity: int) -> tuple[tuple[int, int], ...]:
    """Return exact numerator/denominator pairs in profile feature order."""
    current = example.window[-1].busy
    earlier = example.window[-1 - TREND_MS // SAMPLE_MS].busy
    recent = example.window[-(FULL_WINDOW_MS // SAMPLE_MS):]
    full_count = sum(reading.busy == capacity for reading in recent)
    return (current, capacity), ((current - earlier) * 1000, capacity * TREND_MS), (full_count, len(recent))


def profile_features(example: Example, capacity: int) -> tuple[float, float, float]:
    utilization, trend, full = feature_ratios(example, capacity)
    return utilization[0] / utilization[1], trend[0] / trend[1], full[0] / full[1]


@dataclass(frozen=True)
class Profile:
    weights: tuple[float, float, float]
    intercept: float
    threshold: float

    def score(self, values: tuple[float, float, float]) -> float:
        return self.intercept + sum(weight * value for weight, value in zip(self.weights, values))

    def save(self, path: Path) -> None:
        document = {**CONTRACT, **asdict(self)}
        path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Profile":
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or any(
            document.get(key) != expected for key, expected in CONTRACT.items()
        ):
            raise ValueError("Unsupported connection-pool profile contract.")
        return cls.from_coefficients(document, threshold=document.get("threshold"))

    @classmethod
    def from_coefficients(cls, document: dict, *, threshold: float = 0.0) -> "Profile":
        if not isinstance(document, dict):
            raise ValueError("Coefficients must be a JSON object.")
        weights = document.get("weights")
        if not isinstance(weights, list) or len(weights) != len(FEATURES):
            raise ValueError("The profile requires one weight per feature.")
        intercept = document.get("intercept")
        if any(type(value) not in (int, float) or not isfinite(value)
               for value in (*weights, intercept, threshold)):
            raise ValueError("Profile weights, intercept, and threshold must be finite numbers.")
        return cls(tuple(weights), intercept, threshold)
