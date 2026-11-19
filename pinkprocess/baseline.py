"""Fit a two-feature warning rule and score warning onsets by queue episode."""

from bisect import bisect_right
from dataclasses import dataclass

from .dataset import Example, FORECAST_MS, OBSERVATION_MS
from .simulator import Run, SAMPLE_MS

TREND_MS = 2_000


def features(example: Example, capacity: int) -> tuple[float, float]:
    current_busy = example.window[-1].busy
    earlier_busy = example.window[-1 - TREND_MS // SAMPLE_MS].busy
    utilization = current_busy / capacity
    change_per_second = (current_busy - earlier_busy) / capacity / (TREND_MS / 1000)
    return utilization, change_per_second


@dataclass(frozen=True)
class Rule:
    trend_weight_seconds: float
    threshold: float

    def score(self, utilization: float, change_per_second: float) -> float:
        return utilization + self.trend_weight_seconds * change_per_second


@dataclass
class Results:
    episodes: int
    lead_times_ms: list[int]
    false_alerts: int
    repeated_alerts: int

    @property
    def warned(self) -> int:
        return len(self.lead_times_ms)

    @property
    def missed(self) -> int:
        return self.episodes - self.warned

    @property
    def event_f1(self) -> float:
        total = 2 * self.warned + self.missed + self.false_alerts
        return 2 * self.warned / total if total else 0.0


def score_run(rule: Rule, run: Run, rows: list[tuple[int, float, float]]) -> Results:
    starts = [episode.start_ms for episode in run.episodes
              if episode.start_ms > OBSERVATION_MS]
    first_alerts = {}
    false_alerts = repeated_alerts = 0
    previous_positive_ms = None
    for time_ms, utilization, change in rows:
        if rule.score(utilization, change) < rule.threshold:
            previous_positive_ms = None
            continue
        continuation = previous_positive_ms == time_ms - SAMPLE_MS
        previous_positive_ms = time_ms
        if continuation:
            continue
        next_index = bisect_right(starts, time_ms)
        if next_index == len(starts) or starts[next_index] > time_ms + FORECAST_MS:
            false_alerts += 1
        else:
            start_ms = starts[next_index]
            if start_ms in first_alerts:
                repeated_alerts += 1
            else:
                first_alerts[start_ms] = time_ms

    return Results(len(starts), [start - alert for start, alert in first_alerts.items()],
                   false_alerts, repeated_alerts)


def evaluate(rule: Rule, runs: list[tuple[Run, list[tuple[int, float, float]]]]) -> Results:
    combined = Results(0, [], 0, 0)
    for run, rows in runs:
        result = score_run(rule, run, rows)
        combined.episodes += result.episodes
        combined.lead_times_ms.extend(result.lead_times_ms)
        combined.false_alerts += result.false_alerts
        combined.repeated_alerts += result.repeated_alerts
    return combined


def fit_rule(training_runs: list[tuple[Run, list[tuple[int, float, float]]]]) -> Rule:
    """Choose the rule with the best pooled training event F1."""
    candidates = [Rule(weight, threshold / 100)
                  for weight in (0.5, 1.0, 1.5, 2.0)
                  for threshold in range(65, 111, 5)]

    def ranking(rule: Rule) -> tuple[float, int]:
        result = evaluate(rule, training_runs)
        return result.event_f1, -result.false_alerts

    return max(candidates, key=ranking)
