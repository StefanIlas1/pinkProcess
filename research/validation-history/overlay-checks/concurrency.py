"""Isolated physical test of real stream ingest during projection on the candidate overlay."""
import argparse
import hashlib
import json
from pathlib import Path
from threading import Event, Thread
from time import monotonic, sleep
import uuid

import numpy as np
from pynq import allocate

from fpga.check_classifier import board_lock
from kit.benchmark_projection import load_capture
from lm_reference.live_demo import PynqClassifier
from lm_reference.projection_overlay import BUSY, CONTROL, projection_mmio, QueryProjection
from pinkprocess.kit_server import InferenceLock
from pinkprocess.telemetry_device import TelemetryDevice
from pinkprocess.telemetry_stream import TelemetryStream

QUERY = "encoder.block.0.layer.0.SelfAttention.q"


class TrackedClassifier(PynqClassifier):
    def __init__(self, bitstream):
        super().__init__(bitstream)
        self.projection = projection_mmio(self.overlay)
        self.scoring = []

    def submit(self, activation):
        start = monotonic()
        projection_busy = self.projection.read(CONTROL) & BUSY
        super().submit(activation)
        self.scoring.append((start, monotonic(), projection_busy))


def row(index):
    return dict(time_ms=100 * index, busy=4 if index % 8 >= 4 else index % 4,
                waiting=1 + index % 3 if index % 8 >= 6 else 0)


def telemetry(device, lock, trace, *, wake=None, done=None):
    with lock:
        device.stream = TelemetryStream(device._score_stream_window)
        data = dict(session=str(uuid.uuid4()), source="Projection concurrency test", origin="demo", capacity=4)
        device.ingest({**data, "readings": [row(index) for index in range(49)]})
        start_count = device.stream.fpga_submissions
    if wake is not None:
        wake.wait()
    start = monotonic()
    try:
        for index in range(20):
            if wake is not None:
                sleep(max(0, start + 0.1 * index - monotonic()))
            with lock:
                snapshot = device.ingest({**data, "readings": [row(49 + index)]})
                forecast = device.handle({"op": "stream_forecast"})["forecast"]
            if snapshot["accepted"] != 1 or not forecast:
                raise RuntimeError("Telemetry did not accept and score the next reading.")
            trace.append(forecast)
        if device.stream.fpga_submissions - start_count != len(trace):
            raise RuntimeError("Unexpected hardware submission count for the telemetry stream.")
    finally:
        if done is not None:
            done.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bitstream", type=Path)
    parser.add_argument("capture", type=Path)
    parser.add_argument("profiles", type=Path)
    parser.add_argument("state", type=Path)
    args = parser.parse_args()
    weights, activations = load_capture(args.capture)
    expected = activations.astype(np.int32) @ weights.astype(np.int32).T

    with board_lock():
        classifier = TrackedClassifier(args.bitstream)
        device = TelemetryDevice(classifier, args.profiles, args.state)
        lock = InferenceLock()
        sequential = []
        telemetry(device, lock, sequential)
        classifier.scoring.clear()
        query = QueryProjection(classifier.projection, allocate)
        actual = query.project(QUERY, weights, activations)
        np.testing.assert_array_equal(actual, expected)
        concurrent = []
        errors = []
        wake, done = Event(), Event()

        def producer():
            try:
                telemetry(device, lock, concurrent, wake=wake, done=done)
            except Exception as exc:
                errors.append(exc)
                done.set()

        worker = Thread(target=producer)
        worker.start()
        windows = []
        trial_started = monotonic()
        wake.set()
        try:
            while not done.is_set():
                start = monotonic()
                result = query.project(QUERY, weights, activations)
                windows.append((start, monotonic()))
                np.testing.assert_array_equal(result, actual)
        finally:
            worker.join()
            query.close()
        if errors:
            raise errors[0]
        if concurrent != sequential:
            raise RuntimeError("Concurrent telemetry differed from sequential FPGA telemetry.")
        duration = monotonic() - trial_started
        overlap = sum(any(ts < end and te > start for start, end in windows)
                      for ts, te, _ in classifier.scoring)
        hardware_overlap = sum(bool(busy) for _, _, busy in classifier.scoring)
        if not windows or not overlap:
            raise RuntimeError("No telemetry hardware submissions overlapped a projection call.")
        print(json.dumps({
            "backend": "physical_pynq_z2",
            "scope": "isolated candidate overlay with genuine TelemetryDevice stream ingest and reserved-row classifier",
            "bitstream_sha256": hashlib.sha256(args.bitstream.read_bytes()).hexdigest(),
            "elapsed_seconds": duration,
            "encoder_projection_calls": len(windows),
            "telemetry_fpga_submissions": len(classifier.scoring),
            "telemetry_fpga_submissions_overlapping_projection_calls": overlap,
            "telemetry_submissions_observing_projection_busy_at_start": hardware_overlap,
            "same_forecasts_as_sequential": concurrent == sequential,
            "projection_coincides_with_integer_reference": True,
        }, indent=2))


if __name__ == "__main__":
    main()
