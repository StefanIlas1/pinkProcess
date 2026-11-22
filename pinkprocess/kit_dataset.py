"""Convert sampled pool readings to the training profile's fixed-point inputs."""

import csv
import io
from collections import deque
from dataclasses import asdict

from .backlog import features as backlog_features
from .dataset import OBSERVATION_MS, Example
from .fixed_point import encode_features
from .simulator import SAMPLE_MS, Reading, simulate
from .telemetry_stream import StreamSession, TelemetryStream

POOL_SAMPLE_CAPACITY = 12
WINDOW_SIZE = OBSERVATION_MS // SAMPLE_MS
MAX_ROWS = WINDOW_SIZE + 2048 - 1
MAX_DATASET_REQUEST = 128 * 1024


def pool_sample_csv() -> str:
    run = simulate(200, duration_ms=20_000, capacity=POOL_SAMPLE_CAPACITY)
    return "time_ms,busy,waiting\n" + "".join(
        f"{row.time_ms},{row.busy},{row.waiting}\n" for row in run.readings
    )


def read_dataset(csv_text: str, capacity: int) -> list[Reading]:
    if type(capacity) is not int or not 0 < capacity < 2**31:
        raise ValueError("Capacity must be a positive 32-bit integer.")
    if not isinstance(csv_text, str):
        raise ValueError("Provide a CSV file with pool readings.")
    rows = csv.reader(io.StringIO(csv_text.removeprefix("\ufeff")), strict=True)
    readings = []
    try:
        if next(rows, None) != ["time_ms", "busy", "waiting"]:
            raise ValueError("Use the CSV header time_ms,busy,waiting.")
        for line, row in enumerate(rows, 2):
            if not row:
                continue
            if len(readings) >= MAX_ROWS:
                raise ValueError(f"Select at most {MAX_ROWS} consecutive readings.")
            try:
                time_ms, busy, waiting = map(int, row)
            except ValueError as exc:
                raise ValueError(f"Row {line} requires three integers.") from exc
            if not 0 <= time_ms < 2**63 or not 0 <= busy <= capacity or not 0 <= waiting < 2**31:
                raise ValueError(f"Check the timestamp, busy count and waiting count on row {line}.")
            if readings and time_ms != readings[-1].time_ms + SAMPLE_MS:
                raise ValueError(f"Row {line} must be exactly {SAMPLE_MS} ms after the previous reading.")
            readings.append(Reading(time_ms, busy, waiting))
    except csv.Error as exc:
        raise ValueError(f"Invalid CSV near line {rows.line_num}.") from exc
    if len(readings) < WINDOW_SIZE:
        raise ValueError(f"Provide at least {WINDOW_SIZE} readings at 100 ms intervals.")
    return readings


def prepare_dataset(csv_text: str, capacity: int, *, backlog=False) -> dict:
    readings = read_dataset(csv_text, capacity)
    sample_count = len(readings)
    start, end = readings[0].time_ms, readings[-1].time_ms
    features, times = [], []
    skipped = 0
    window = deque(readings[-WINDOW_SIZE:], maxlen=WINDOW_SIZE)
    if not backlog:
        window.clear()
        for reading in readings:
            window.append(reading)
            if len(window) < WINDOW_SIZE:
                continue
            if reading.waiting:
                skipped += 1
            else:
                times.append(reading.time_ms)
                features.append(list(encode_features(Example(reading.time_ms, tuple(window), None), capacity)))
    if backlog:
        return dict(end_ms=end, sample_count=sample_count, waiting=window[-1].waiting,
                    features=backlog_features(list(window), capacity))
    if not times:
        raise ValueError("A queue was already present at every eligible timestamp.")
    return dict(features=features, times_ms=times, sample_count=sample_count,
                skipped_waiting=skipped, start_ms=start, end_ms=end)


def dataset_snapshot(dataset_id, name, csv_text, capacity, score):
    rows = read_dataset(csv_text, capacity)
    stream = TelemetryStream(score)
    stream.active = StreamSession(dataset_id, name, 'dataset', capacity, history=deque(rows))
    forecast = score(rows[-WINDOW_SIZE:], capacity)
    return dict(ok=True, backend='PYNQ-Z2', state='ready', fresh=False, source=name, origin='dataset',
                capacity=capacity, latest=asdict(rows[-1]), sample_count=len(rows),
                observations=stream.observations(len(rows) * SAMPLE_MS, detailed=True),
                forecast=forecast, forecast_reading=asdict(rows[-1]),
                limitations=['The readings came from an uploaded CSV. The forecast applies to its final reading.',
                             'Measurements alone do not identify a cause or request-level latency.'])
