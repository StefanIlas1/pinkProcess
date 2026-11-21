"""Serve the kit's resident quantized forecasting models over a board-local pipe."""

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from fpga.check_classifier import SCORES, board_lock
from lm_reference.live_demo import PynqClassifier
from .backlog import INPUTS, TARGETS, VERSION as BACKLOG_VERSION
from .backlog import features as backlog_features
from .dataset import Example, FORECAST_MS, OBSERVATION_MS
from .fixed_point import FEATURE_BOUNDS, SCORE_SCALE, FixedProfile, encode_features
from .profile import Profile
from .simulator import Reading, SAMPLE_MS
from .telemetry_stream import QUERY_OPS, TelemetryStream


BASE = "pool-logistic-v1"
VERSIONS = (BASE, "service-b-astra-v1")
WIDTH = 64
FEATURE_LANES = 3
WEIGHT_LANES = 4
BIAS_LANES = WIDTH - 3 * FEATURE_LANES * WEIGHT_LANES - 1


def split_int8(value, count):
    """Split an integer into a fixed number of signed INT8 summands."""
    if abs(value) > 127 * count:
        raise ValueError("Value exceeds the representable FPGA coefficient range.")
    sign = -1 if value < 0 else 1
    whole, remainder = divmod(abs(value), 127)
    result = [sign * 127] * whole
    if remainder:
        result.append(sign * remainder)
    return result + [0] * (count - len(result))


def unsigned_bytes(values):
    return bytes(value & 255 for value in values)


@dataclass(frozen=True)
class Model:
    profile: FixedProfile
    row: bytes

    @classmethod
    def load(cls, path):
        profile = FixedProfile.quantize(Profile.load(path))
        # Each Q8 weight times a Q8 feature becomes the sum of all products
        # between their INT8 summands. The FPGA also adds the Q16 intercept.
        row = []
        for weight in profile.weights:
            for part in split_int8(weight, WEIGHT_LANES):
                row.extend([part] * FEATURE_LANES)
        whole, remainder = divmod(abs(profile.intercept), 127)
        sign = -1 if profile.intercept < 0 else 1
        row.extend(split_int8(sign * whole, BIAS_LANES))
        row.append(sign * remainder)
        return cls(profile, unsigned_bytes(row))

    @property
    def digest(self):
        return hashlib.sha256(self.row).hexdigest()


def activation(features):
    if not isinstance(features, (list, tuple)) or len(features) != len(FEATURE_BOUNDS):
        raise ValueError("Each window requires three integer Q8 feature values.")
    result = []
    for value, (low, high) in zip(features, FEATURE_BOUNDS):
        if type(value) is not int or not low <= value <= high:
            raise ValueError("Quantized feature is outside the forecasting model's range.")
        result.extend(split_int8(value, FEATURE_LANES) * WEIGHT_LANES)
    return unsigned_bytes(result + [127] * BIAS_LANES + [1])


def parse_window(request):
    capacity, rows = request.get("capacity"), request.get("readings")
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("Capacity must be a positive integer.")
    if not isinstance(rows, list) or len(rows) != OBSERVATION_MS // SAMPLE_MS:
        raise ValueError("Supply exactly 50 telemetry readings.")
    readings = []
    for row in rows:
        if not isinstance(row, dict) or any(type(row.get(k)) is not int
                                            for k in ("time_ms", "busy", "waiting")):
            raise ValueError("Each reading requires integer time_ms, busy and waiting.")
        reading = Reading(row["time_ms"], row["busy"], row["waiting"])
        if reading.time_ms < 0 or not 0 <= reading.busy <= capacity or reading.waiting < 0:
            raise ValueError("Reading exceeds the valid time, utilization or queue range.")
        if readings and reading.time_ms - readings[-1].time_ms != SAMPLE_MS:
            raise ValueError("The model requires continuous 100 ms samples.")
        readings.append(reading)
    if readings[-1].waiting:
        raise ValueError("Requests are already waiting at this timestamp.")
    return Example(readings[-1].time_ms, tuple(readings), None), capacity


class TelemetryDevice:
    def __init__(self, classifier, profiles, state_file):
        self.classifier = classifier
        self.models = {version: Model.load(profiles / (version + ".json")) for version in VERSIONS}
        self.state_file = state_file
        self.version = BASE
        if state_file.exists():
            saved = json.loads(state_file.read_text())
            self.version = saved["version"]
            model = self.models[self.version]
            if saved != self._record(self.version, model):
                raise ValueError("Saved version does not match the installed telemetry profile.")
        self.ready = False
        document = json.loads((profiles / (BACKLOG_VERSION + ".json")).read_text())
        rows = document["weights"]
        if (document["version"] != BACKLOG_VERSION or document["inputs"] != list(INPUTS)
                or document["targets"] != list(TARGETS) or len(rows) != len(TARGETS)
                or any(len(row) != len(INPUTS) or any(type(w) is not int or abs(w)>127 for w in row)
                       for row in rows)):
            raise ValueError("Invalid queue-outlook weight profile.")
        self.backlog_rows = rows
        packed = [unsigned_bytes(row).ljust(WIDTH, b"\0") for row in rows]
        self.backlog_hash = hashlib.sha256(b"".join(packed)).hexdigest()
        for i, row in enumerate(packed, 1):
            self.classifier.write(i * WIDTH, row)
            if self.classifier.read_row(i) != row:
                raise RuntimeError("Queue-outlook weight readback failed.")
        self._install(self.models[self.version])
        self.ready = True
        self.stream = TelemetryStream(self._score_stream_window)

    @staticmethod
    def _record(version, model):
        return {"version": version, "weights_sha256": model.digest, "threshold": model.profile.threshold}

    def _install(self, model):
        self.classifier.write(0, model.row)
        if self.classifier.read_row(0) != model.row:
            raise RuntimeError("FPGA weight verification failed.")

    def _save(self, record):
        file = NamedTemporaryFile(mode="w", dir=self.state_file.parent, delete=False)
        temporary = Path(file.name)
        try:
            with file:
                json.dump(record, file)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self.state_file)
        finally:
            temporary.unlink(missing_ok=True)

    def apply(self, version):
        if version not in VERSIONS:
            raise ValueError("Select an installed model version.")
        previous_version = self.version
        old = self.models[self.version]
        new = self.models[version]
        try:
            self._install(new)
            self._save(self._record(version, new))
        except Exception:
            self.ready = False
            self._install(old)
            self.ready = True
            raise
        self.version = version
        if version != previous_version:
            self.stream.invalidate_forecast()

    def ingest(self, request):
        if not self.ready:
            raise RuntimeError("The FPGA requires recovery. Restart the board service.")
        try:
            return self.stream.ingest(request)
        except (RuntimeError, OSError):
            self.ready = False
            self.stream.invalidate_forecast()
            raise

    def _score_stream_window(self, window, capacity):
        if window[-1].waiting:
            return {**self.handle({"op": "backlog", "features": backlog_features(window, capacity)},
                                  stream=True),
                    "time_ms": window[-1].time_ms, "targets": list(TARGETS)}
        return self.handle({"op": "query", "capacity": capacity,
                            "readings": [asdict(reading) for reading in window]}, stream=True)

    def _submit(self, encoded, stream):
        self.classifier.submit(encoded)
        if stream:
            self.stream.record_submission()

    def handle(self, request, *, stream=False):
        if not self.ready:
            raise RuntimeError("The FPGA requires recovery. Restart the board service.")
        if not isinstance(request, dict):
            raise ValueError("Expected a JSON object.")
        op = request.get("op")
        result = {}
        if op in QUERY_OPS:
            return self.stream.query(request)
        if op == "status":
            result["stream_supported"] = True
            result["stream_evidence_supported"] = True
        if op == "backlog":
            vector = request.get("features")
            if (not isinstance(vector, list) or len(vector) != len(INPUTS) or vector[-1] != 1
                    or any(type(v) is not int or abs(v)>127 for v in vector)):
                raise ValueError("Invalid queue-outlook feature vector.")
            self._submit(unsigned_bytes(vector).ljust(WIDTH, b"\0"), stream)
            scores = []
            for i, row in enumerate(self.backlog_rows, 1):
                word = self.classifier.mmio.read(SCORES + 4*i)
                score = word - (1 << 32) if word & (1 << 31) else word
                if score != sum(a*b for a,b in zip(vector,row)):
                    self.ready = False
                    raise RuntimeError("Queue-outlook score verification failed.")
                scores.append(score)
            return dict(ok=True, backend="PYNQ-Z2", model="queue_outlook", version=BACKLOG_VERSION,
                        weights_sha256=self.backlog_hash, scores=scores, horizon_ms=FORECAST_MS)
        if op in ("apply", "rollback"):
            self.apply(request.get("version") if op == "apply" else BASE)
        elif op in ("query_batch", "query"):
            if op == "query":
                example, capacity = parse_window(request)
                vectors = [encode_features(example, capacity)]
            else:
                vectors = request.get("features")
            if not isinstance(vectors, list) or not 1 <= len(vectors) <= 512:
                raise ValueError("Each board query accepts 1 to 512 windows.")
            inputs = [activation(features) for features in vectors]
            profile = self.models[self.version].profile
            scores = []
            for features, encoded in zip(vectors, inputs):
                self._submit(encoded, stream)
                word = self.classifier.mmio.read(SCORES)
                score = word - (1 << 32) if word & (1 << 31) else word
                if score != profile.score(features):
                    self.ready = False
                    raise RuntimeError("Live FPGA score disagrees with the integer reference. Restart the board service.")
                scores.append(score)
            alerts = [score >= profile.threshold for score in scores]
            if op == "query":
                result.update(time_ms=example.time_ms, score=scores[0], alert=alerts[0])
            else:
                result.update(scores=scores, alerts=alerts)
        elif op != "status":
            raise ValueError("Unsupported board operation.")
        model = self.models[self.version]
        return {"ok": True, "backend": "PYNQ-Z2", "model": "pool_wait_logistic",
                **self._record(self.version, model), "versions": list(VERSIONS), "backlog_version": BACKLOG_VERSION,
                "score_scale": SCORE_SCALE, "horizon_ms": FORECAST_MS, **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bitstream", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--state-file", type=Path)
    args = parser.parse_args()
    with board_lock():
        device = TelemetryDevice(PynqClassifier(args.bitstream), args.profiles,
                                 args.state_file or args.profiles / ".telemetry-active.json")
        for line in sys.stdin:
            try:
                response = device.handle(json.loads(line))
            except (ValueError, RuntimeError, OSError) as exc:
                response = {"ok": False, "error": str(exc)}
            print(json.dumps(response, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
