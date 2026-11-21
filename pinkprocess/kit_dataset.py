"""Convert sampled pool readings to the training profile's fixed-point inputs."""

import csv
import io
from collections import deque

from .backlog import features as backlog_features
from .dataset import OBSERVATION_MS, Example
from .fixed_point import encode_features
from .simulator import SAMPLE_MS, Reading, simulate

DEMO_CAPACITY = 12
WINDOW_SIZE = OBSERVATION_MS // SAMPLE_MS
MAX_ROWS = WINDOW_SIZE + 2048 - 1


def demo_csv() -> str:
    run = simulate(200, duration_ms=20_000, capacity=DEMO_CAPACITY)
    return "time_ms,busy,waiting\n" + "".join(
        f"{row.time_ms},{row.busy},{row.waiting}\n" for row in run.readings
    )


def prepare_dataset(csv_text: str, capacity: int, *, backlog=False) -> dict:
    if type(capacity) is not int or capacity < 1:
        raise ValueError("Capacity must be a positive integer.")
    if not isinstance(csv_text, str):
        raise ValueError("Provide a CSV file with pool readings.")
    rows = csv.reader(io.StringIO(csv_text.removeprefix("\ufeff")), strict=True)
    header = next(rows, None)
    if header != ["time_ms", "busy", "waiting"]:
        raise ValueError("Use the CSV header time_ms,busy,waiting.")
    window = deque(maxlen=WINDOW_SIZE)
    features, times = [], []
    sample_count = skipped = 0
    start = end = None
    try:
        for line, row in enumerate(rows, 2):
            if not row:
                continue
            sample_count += 1
            if sample_count > MAX_ROWS:
                raise ValueError(f"Select at most {MAX_ROWS} consecutive readings.")
            try:
                time_ms, busy, waiting = map(int, row)
            except ValueError as exc:
                raise ValueError(f"Row {line} requires three integers.") from exc
            if time_ms < 0 or not 0 <= busy <= capacity or waiting < 0:
                raise ValueError(f"Check the timestamp, busy count and waiting count on row {line}.")
            if end is not None and time_ms != end + SAMPLE_MS:
                raise ValueError(f"Row {line} must be exactly {SAMPLE_MS} ms after the previous reading.")
            if start is None:
                start = time_ms
            end = time_ms
            window.append(Reading(time_ms, busy, waiting))
            if not backlog and len(window) == WINDOW_SIZE:
                if waiting:
                    skipped += 1
                else:
                    times.append(time_ms)
                    features.append(list(encode_features(Example(time_ms, tuple(window), None), capacity)))
    except csv.Error as exc:
        raise ValueError(f"Invalid CSV near line {rows.line_num}.") from exc
    if sample_count < WINDOW_SIZE:
        raise ValueError(f"Provide at least {WINDOW_SIZE} readings at 100 ms intervals.")
    if backlog:
        return dict(end_ms=end, sample_count=sample_count, waiting=window[-1].waiting,
                    features=backlog_features(list(window), capacity))
    if not times:
        raise ValueError("A queue was already present at every eligible timestamp.")
    return dict(features=features, times_ms=times, sample_count=sample_count,
                skipped_waiting=skipped, start_ms=start, end_ms=end)
