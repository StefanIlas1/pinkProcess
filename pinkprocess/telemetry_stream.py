"""Bounded, board-resident history and forecasts for one active telemetry source."""

from collections import deque
from dataclasses import asdict, dataclass, field
from itertools import islice
from time import monotonic
from uuid import UUID

from .dataset import OBSERVATION_MS
from .simulator import Reading, SAMPLE_MS


FRESH_SECONDS = 2
MAX_LOOKBACK_MS = 60_000
WINDOW_SIZE = OBSERVATION_MS // SAMPLE_MS
HISTORY_SIZE = MAX_LOOKBACK_MS // SAMPLE_MS
MAX_BATCH = 512
QUERY_OPS = ("stream_status", "stream_forecast", "stream_observations", "stream_evidence")
BOARD_DEMO_SOURCE = "Board SQLite demo (generated traffic)"
FIXTURE_SOURCE_PREFIX = "Synthetic acceptance fixture"


def session_id(value):
    try:
        return str(UUID(value))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("A telemetry session UUID is required.") from exc


def parse_batch(payload, capacity):
    if type(capacity) is not int or not 1 <= capacity < 2**31:
        raise ValueError("Capacity must be a positive 32-bit integer.")
    if not isinstance(payload, list) or not 1 <= len(payload) <= MAX_BATCH:
        raise ValueError(f"Send 1 to {MAX_BATCH} readings per telemetry batch.")
    result = []
    for record in payload:
        if not isinstance(record, dict):
            raise ValueError("Expected a telemetry reading object.")
        time_ms, busy, waiting = (record.get(key) for key in ("time_ms", "busy", "waiting"))
        if (type(time_ms) is not int or not 0 <= time_ms < 2**63
                or type(busy) is not int or not 0 <= busy <= capacity
                or type(waiting) is not int or not 0 <= waiting < 2**31):
            raise ValueError("Invalid telemetry timestamp, utilization or queue depth.")
        if result and time_ms <= result[-1].time_ms:
            raise ValueError("Telemetry timestamps must increase within a batch.")
        result.append(Reading(time_ms, busy, waiting))
    return result


def metric_summary(values):
    return dict(minimum=min(values), maximum=max(values), mean=sum(values) / len(values),
                change=values[-1] - values[0])


def summarize_rows(rows, capacity):
    busy = [row.busy for row in rows]
    waiting = [row.waiting for row in rows]
    return dict(start_ms=rows[0].time_ms, end_ms=rows[-1].time_ms, sample_count=len(rows),
                busy=dict(**metric_summary(busy), full_samples=busy.count(capacity),
                          mean_utilization=sum(busy) / (len(busy) * capacity)),
                waiting=dict(**metric_summary(waiting), positive_samples=len(waiting) - waiting.count(0),
                             onsets=sum(before == 0 and after > 0 for before, after in zip(waiting, waiting[1:])),
                             clearances=sum(before > 0 and after == 0 for before, after in zip(waiting, waiting[1:]))))


@dataclass
class StreamSession:
    id: str
    source: str
    origin: str
    capacity: int
    history: deque = field(default_factory=lambda: deque(maxlen=HISTORY_SIZE))
    arrived: float | None = None
    forecast: dict | None = None

    def fresh(self, now):
        return self.arrived is not None and now - self.arrived < FRESH_SECONDS


class TelemetryStream:
    def __init__(self, score, clock=monotonic):
        self.score = score
        self.clock = clock
        self.active = None
        self.retired = deque(maxlen=64)
        self.fpga_submissions = 0

    def invalidate_forecast(self):
        if self.active is not None:
            self.active.forecast = None

    def record_submission(self):
        self.fpga_submissions += 1

    def ingest(self, request):
        if not isinstance(request, dict):
            raise ValueError("Expected a JSON object.")
        session = session_id(request.get("session"))
        active = self.active
        now = self.clock()
        if request.get("stop") is True:
            if active is not None and active.id == session:
                self.retired.append(session)
                self.active = None
            return self.snapshot(now, session, accepted=0)

        source = request.get("source")
        origin = request.get("origin", "live")
        capacity = request.get("capacity")
        if not isinstance(source, str) or not source.strip() or len(source) > 128:
            raise ValueError("Supply a telemetry source label of 1 to 128 characters.")
        if origin not in ("live", "demo"):
            raise ValueError("Telemetry origin must be live or demo.")
        guarded = "fixture_after_session" in request
        if guarded or source.startswith(FIXTURE_SOURCE_PREFIX):
            if not guarded or origin != "demo" or not source.startswith(FIXTURE_SOURCE_PREFIX + " "):
                raise ValueError("Synthetic acceptance fixtures need a guarded demo source.")
            expected = request["fixture_after_session"]
            if expected is not None:
                expected = session_id(expected)
            current = active.id if active else None
            if current != expected or (active is not None and (
                active.id == session or active.origin != "demo" or active.source != BOARD_DEMO_SOURCE
            )):
                raise ValueError("The expected SQLite demo or idle stream no longer owns the board.")
        rows = parse_batch(request.get("readings"), capacity)

        if active is None or active.id != session:
            if session in self.retired:
                raise ValueError("Telemetry session has ended. Start a new session.")
            if active is not None:
                takeover = guarded or (active.origin == "demo" and origin == "live")
                if active.fresh(now) and not takeover:
                    raise ValueError("Another telemetry session is active.")
                self.retired.append(active.id)
            active = self.active = StreamSession(session, source, origin, capacity)
        elif (source, origin, capacity) != (active.source, active.origin, active.capacity):
            raise ValueError("Telemetry source, origin and capacity are fixed for each session.")

        if active.history:
            latest_time = active.history[-1].time_ms
            seen = {row.time_ms: row for row in active.history}
            for row in rows:
                if row.time_ms > latest_time:
                    break
                if row.time_ms not in seen:
                    raise ValueError("Previous telemetry reading is unavailable for retry verification.")
                if row != seen[row.time_ms]:
                    raise ValueError("Telemetry conflicts with a previously accepted reading.")
            rows = [row for row in rows if row.time_ms > latest_time]

        for row in rows:
            if active.history and (not active.fresh(now)
                                   or row.time_ms - active.history[-1].time_ms != SAMPLE_MS):
                active.history.clear()
            active.history.append(row)
            active.forecast = None
            active.arrived = now
            if len(active.history) >= WINDOW_SIZE:
                window = tuple(islice(reversed(active.history), WINDOW_SIZE))[::-1]
                active.forecast = self.score(window, capacity)
        return self.snapshot(self.clock(), session, accepted=len(rows))

    def snapshot(self, now, session=None, **fields):
        active = self.active
        if active is None or (session is not None and session != active.id):
            return dict(ok=True, backend="PYNQ-Z2", state="idle", session=None, source=None,
                        origin=None, capacity=None, sample_count=0, latest=None,
                        latest_time_ms=None, fresh=False, age_ms=None,
                        fpga_submissions=self.fpga_submissions, **fields)

        latest = active.history[-1]
        fresh = active.fresh(now)
        state = "stale" if not fresh else "ready" if active.forecast else "warming"
        return dict(ok=True, backend="PYNQ-Z2", state=state, session=active.id, source=active.source,
                    origin=active.origin, capacity=active.capacity, sample_count=len(active.history),
                    latest=asdict(latest), latest_time_ms=latest.time_ms, fresh=fresh,
                    age_ms=int((now - active.arrived) * 1000),
                    fpga_submissions=self.fpga_submissions, **fields)

    def query(self, request):
        session = session_id(request["session"]) if "session" in request else None
        result = self.snapshot(self.clock(), session)
        op = request["op"]
        if op == "stream_status":
            result["fixture_guard_supported"] = True
        if op in ("stream_forecast", "stream_evidence"):
            result["forecast"] = self.active.forecast if result["state"] == "ready" else None
        if op in ("stream_observations", "stream_evidence"):
            window_ms = request.get("window_ms", OBSERVATION_MS)
            if type(window_ms) is not int or not SAMPLE_MS <= window_ms <= MAX_LOOKBACK_MS:
                raise ValueError(f"Observation lookback must be {SAMPLE_MS} to {MAX_LOOKBACK_MS} ms.")
            evidence = op == "stream_evidence"
            result["observations"] = self.observations(window_ms, detailed=evidence) if result["session"] else None
            if evidence:
                result.update(sample_interval_ms=SAMPLE_MS, forecast_input_window_ms=OBSERVATION_MS,
                              limitations=self.evidence_limitations(result))
        return result

    @staticmethod
    def evidence_limitations(result):
        state = result["state"]
        if state == "idle":
            return ["No telemetry session matches this request."]
        limitations = ["Sample timestamps come from the telemetry source. The board measures receipt age.",
                       "Samples describe utilization and queue depth for one source. They do not identify causes or request-level latency."]
        if result["origin"] == "demo":
            limitations.append("The source is the board's demo telemetry.")
        if state == "stale":
            limitations.append("These observations are stale. No current forecast is available.")
        elif state == "warming":
            limitations.append("No current forecast is cached. Scoring requires a continuous input window.")
        else:
            limitations.append("The forecast is a model estimate for its stated horizon. Scores are not calibrated probabilities.")
        if not result["observations"]["complete"]:
            limitations.append("The requested lookback extends beyond the continuous samples retained for this session.")
        return limitations

    def observations(self, window_ms, *, detailed=False):
        history = self.active.history
        cutoff = history[-1].time_ms - window_ms
        rows = [row for row in history if row.time_ms > cutoff]
        result = dict(window_ms=window_ms, sample_count=len(rows), start_ms=rows[0].time_ms,
                      end_ms=rows[-1].time_ms, first=asdict(rows[0]), latest=asdict(rows[-1]),
                      peak_busy=asdict(max(rows, key=lambda row: row.busy)),
                      peak_waiting=asdict(max(rows, key=lambda row: row.waiting)))
        if detailed:
            capacity = self.active.capacity
            covered_ms = min(window_ms, len(rows) * SAMPLE_MS)
            # Group adjacent samples from the latest observation in intervals of at least one second.
            requested_samples = (window_ms + SAMPLE_MS - 1) // SAMPLE_MS
            interval_size = max(1000 // SAMPLE_MS, (requested_samples + 5) // 6)
            intervals = [summarize_rows(rows[max(0, end - interval_size):end], capacity)
                         for end in range(len(rows), 0, -interval_size)]
            result.update(summarize_rows(rows, capacity), covered_ms=covered_ms,
                          complete=covered_ms == window_ms, intervals=intervals[::-1])
        return result
