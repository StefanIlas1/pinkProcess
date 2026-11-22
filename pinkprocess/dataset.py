"""Build observable histories and event-derived targets."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

from .simulator import Reading, Run, SAMPLE_MS

OBSERVATION_MS = 5_000
FORECAST_MS = 10_000
TRAIN_SEEDS = tuple(range(100, 108))
TEST_SEEDS = tuple(range(200, 204))


@dataclass(frozen=True)
class Example:
    time_ms: int
    window: tuple[Reading, ...]
    first_wait_ms: int | None

    @property
    def label(self) -> bool:
        return self.first_wait_ms is not None


def build_examples(run: Run) -> list[Example]:
    """Use readings from (t - 5s, t] and arrivals from (t, t + 10s]."""
    window_size = OBSERVATION_MS // SAMPLE_MS
    examples = []
    for index in range(window_size, len(run.readings)):
        reading = run.readings[index]
        now = reading.time_ms
        if now + FORECAST_MS > run.duration_ms:
            break
        if reading.waiting:
            continue
        next_wait = bisect_right(run.wait_arrivals_ms, now)
        first_wait_ms = None
        if next_wait < len(run.wait_arrivals_ms):
            candidate = run.wait_arrivals_ms[next_wait]
            if candidate <= now + FORECAST_MS:
                first_wait_ms = candidate
        examples.append(Example(now, run.readings[index - window_size + 1:index + 1],
                                first_wait_ms))
    return examples
