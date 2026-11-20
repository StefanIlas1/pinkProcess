"""Run the reproducible training and held-out evaluation demo."""

from statistics import median

from .baseline import TREND_MS, Rule, evaluate, feature_rows, features, fit_rule
from .dataset import FORECAST_MS, TEST_SEEDS, TRAIN_SEEDS, Example, build_examples
from .simulator import SAMPLE_MS, Run, simulate


def forecast_readings(run: Run, time_ms: int):
    index = time_ms // SAMPLE_MS
    return run.readings[index + 1:index + 1 + FORECAST_MS // SAMPLE_MS]


def comparison(rule: Rule, test_runs: list[tuple[Run, list[Example]]]):
    """Find equal-utilization examples with opposite predictions and outcomes."""
    recoveries = {}
    exhaustions = []
    for run, examples in test_runs:
        for example in examples:
            utilization, trend = features(example, run.capacity)
            predicted = rule.score(utilization, trend) >= rule.threshold
            busy = example.window[-1].busy
            key = (busy, run.capacity)
            if predicted and example.label:
                exhaustions.append((key, run, example))
            if not predicted and not example.label:
                end = forecast_readings(run, example.time_ms)[-1]
                if end.busy < busy:
                    recoveries.setdefault(key, {}).setdefault(run.seed, (run, example))

    for key, run, example in exhaustions:
        for seed, recovery in recoveries.get(key, {}).items():
            if seed != run.seed:
                return recovery, (run, example)
    return None


def show_example(rule: Rule, run: Run, example: Example, show_window: bool = False):
    utilization, trend = features(example, run.capacity)
    score = rule.score(utilization, trend)
    time_ms = example.time_ms
    earlier = example.window[-1 - TREND_MS // SAMPLE_MS]
    future = forecast_readings(run, time_ms)
    print(f"  Seed {run.seed}, t={time_ms / 1000:.3f}s")
    print(f"    At {earlier.time_ms / 1000:.3f}s: {earlier.busy}/{run.capacity} busy. "
          f"Now: {example.window[-1].busy}/{run.capacity} busy, 0 waiting.")
    print(f"    Score = {utilization:.6f} + {rule.trend_weight_seconds:g} * "
          f"({trend:+.6f}/s) = {score:.6f}. "
          f"Predict wait: {score >= rule.threshold}. Actual label: {example.label}.")
    if show_window:
        print("    Busy counts in the 50 input readings (10 readings per line):")
        for index in range(0, len(example.window), 10):
            block = example.window[index:index + 10]
            counts = " ".join(f"{reading.busy:2}" for reading in block)
            print(f"      {block[0].time_ms / 1000:6.1f}s: {counts}")
        print("    Waiting count is recorded at each reading.")
        queued = [f"{reading.time_ms / 1000:.1f}s={reading.waiting}"
                  for reading in example.window if reading.waiting]
        print(f"    Nonzero waiting counts: {', '.join(queued) if queued else 'none'}.")
    if example.first_wait_ms is None:
        print(f"    No arrival waited through {future[-1].time_ms / 1000:.3f}s. "
              f"Pool ended at {future[-1].busy}/{run.capacity} busy.")
    else:
        episode = next(episode for episode in run.episodes
                       if episode.start_ms == example.first_wait_ms)
        delay = (episode.start_ms - time_ms) / 1000
        print(f"    At {episode.start_ms / 1000:.3f}s a new request found all "
              f"{run.capacity} connections busy and queued ({delay:.3f}s later).")
        if episode.end_ms is not None:
            print(f"    The queue emptied at {episode.end_ms / 1000:.3f}s.")
        else:
            print("    The queue was still nonempty at run end.")


def main():
    training = [(run, feature_rows(run, build_examples(run)))
                for run in map(simulate, TRAIN_SEEDS)]
    rule = fit_rule(training)
    held_out = [(run, build_examples(run)) for run in map(simulate, TEST_SEEDS)]
    results = evaluate(rule, [(run, feature_rows(run, examples))
                              for run, examples in held_out])
    hours = sum(run.duration_ms for run, _ in held_out) / 3_600_000
    print(f"Training seeds {TRAIN_SEEDS}; held-out seeds {TEST_SEEDS}")
    print(f"Held-out examples: {sum(len(examples) for _, examples in held_out):,}")
    print(f"Score = utilization + {rule.trend_weight_seconds:g}s * change per second. "
          f"Predict wait at score >= {rule.threshold:.2f}.")
    print(f"Warned about {results.warned}/{results.episodes} queue episodes; "
          f"missed {results.missed}.")
    print(f"False alerts: {results.false_alerts} in {hours:.3f} simulated hours "
          f"({results.false_alerts / hours:.1f}/hour).")
    print(f"Repeat alert onsets matched to already-warned episodes: {results.repeated_alerts}.")
    if results.lead_times_ms:
        leads = results.lead_times_ms
        print(f"First-warning lead time: median {median(leads) / 1000:.3f}s, "
              f"range {min(leads) / 1000:.3f}–{max(leads) / 1000:.3f}s.")
    else:
        print("First-warning lead time unavailable because no episode was warned.")

    pair = comparison(rule, held_out)
    if pair:
        recovery, exhaustion = pair
        print("\nRecovery:")
        show_example(rule, *recovery)
        print("\nExhaustion:")
        show_example(rule, *exhaustion, show_window=True)
    else:
        print("\nNo equal-utilization pair with opposite predictions and outcomes.")


if __name__ == "__main__":
    main()
