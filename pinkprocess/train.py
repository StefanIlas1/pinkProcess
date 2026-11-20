"""Fit, calibrate, and evaluate a portable logistic profile."""

from argparse import ArgumentParser
from collections import Counter
from dataclasses import replace
from math import exp, inf, isfinite, nextafter
from pathlib import Path
from statistics import median

from .baseline import Results, combine_results, evaluate, feature_rows, fit_rule, score_alerts
from .dataset import TEST_SEEDS, TRAIN_SEEDS, Example, build_examples
from .profile import Profile, profile_features
from .simulator import Run, simulate


def label_counts(training: list[tuple[Run, list[Example]]]):
    return Counter((profile_features(example, run.capacity), example.label)
                   for run, examples in training for example in examples)


def fit(training: list[tuple[Run, list[Example]]]) -> Profile:
    """Minimize average logistic loss with batch gradient descent."""
    counts = label_counts(training)
    sample_count = sum(counts.values())
    if not sample_count:
        raise ValueError("Fitting requires at least one eligible training example.")
    step = 1.0 / sample_count
    weights = [0.0, 0.0, 0.0]
    intercept = 0.0
    for _ in range(800):
        weight_gradients = [0.0, 0.0, 0.0]
        intercept_gradient = 0.0
        for (values, label), count in counts.items():
            score = intercept + sum(weight * value for weight, value in zip(weights, values))
            probability = 1 / (1 + exp(-score))
            error = count * (probability - label)
            intercept_gradient += error
            for index, value in enumerate(values):
                weight_gradients[index] += error * value
        intercept -= step * intercept_gradient
        for index, gradient in enumerate(weight_gradients):
            weights[index] -= step * gradient
    return Profile(tuple(weights), intercept, threshold=0.0)


def scored_runs(
    profile: Profile, runs: list[tuple[Run, list[Example]]]
) -> list[tuple[Run, list[tuple[int, float]]]]:
    return [(run, [(example.time_ms, profile.score(profile_features(example, run.capacity)))
                   for example in examples]) for run, examples in runs]


def evaluate_scores(runs: list[tuple[Run, list[tuple[int, float]]]], threshold: float) -> Results:
    return combine_results(score_alerts(run, scores, threshold) for run, scores in runs)


def calibrate(runs: list[tuple[Run, list[tuple[int, float]]]], false_alert_budget: int) -> float:
    """Warn about as many validation episodes as possible within the alert budget."""
    thresholds = {score for _, scores in runs for _, score in scores}
    if not thresholds:
        raise ValueError("Calibration requires at least one eligible validation example.")
    thresholds.add(nextafter(max(thresholds), inf))
    if not all(map(isfinite, thresholds)):
        raise ValueError("Calibration scores must have a finite threshold range.")
    best_threshold = max(thresholds)
    best_rank = (0, 0)
    for threshold in sorted(thresholds, reverse=True):
        result = evaluate_scores(runs, threshold)
        rank = (result.warned, -result.false_alerts)
        if result.false_alerts <= false_alert_budget and rank > best_rank:
            best_rank = rank
            best_threshold = threshold
    return best_threshold


def show_results(name: str, results: Results, hours: float) -> None:
    leads = results.lead_times_ms
    lead = f"{median(leads) / 1_000:.3f}s" if leads else "unavailable"
    print(f"{name}: warned {results.warned}/{results.episodes}, missed {results.missed}, "
          f"false alerts {results.false_alerts} ({results.false_alerts / hours:.1f}/hour), "
          f"median first-warning lead {lead}")


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Where to write the fitted JSON profile")
    output = parser.parse_args().output
    fit_seeds, validation_seeds = TRAIN_SEEDS[:-2], TRAIN_SEEDS[-2:]
    training = [(run, build_examples(run)) for run in map(simulate, fit_seeds)]
    validation = [(run, build_examples(run)) for run in map(simulate, validation_seeds)]
    baseline = fit_rule([(run, feature_rows(run, examples)) for run, examples in training])
    baseline_validation = evaluate(
        baseline, [(run, feature_rows(run, examples)) for run, examples in validation]
    )

    profile = fit(training)
    validation_scores = scored_runs(profile, validation)
    profile = replace(profile, threshold=calibrate(validation_scores, baseline_validation.false_alerts))
    logistic_validation = evaluate_scores(validation_scores, profile.threshold)
    output.parent.mkdir(parents=True, exist_ok=True)
    profile.save(output)
    loaded = Profile.load(output)

    test = [(run, build_examples(run)) for run in map(simulate, TEST_SEEDS)]
    test_scores = scored_runs(loaded, test)
    if loaded != profile or test_scores != scored_runs(profile, test):
        raise RuntimeError("Reloaded profile changed the scores or alert threshold.")

    print(f"Fit seeds {fit_seeds}, validation seeds {validation_seeds}, held-out seeds {TEST_SEEDS}")
    print(f"Validation false alerts: logistic {logistic_validation.false_alerts}, "
          f"baseline budget {baseline_validation.false_alerts}")
    hours = sum(run.duration_ms for run, _ in test) / 3_600_000
    show_results("Baseline", evaluate(baseline, [(run, feature_rows(run, examples))
                                               for run, examples in test]), hours)
    show_results("Logistic", evaluate_scores(test_scores, loaded.threshold), hours)
    example_count = sum(len(rows) for _, rows in test_scores)
    print(f"Saved {output} and verified all {example_count:,} held-out scores after reload.")


if __name__ == "__main__":
    main()
