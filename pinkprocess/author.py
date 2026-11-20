"""Prepare a second-service weight-authoring packet and evaluate returned coefficients."""

import csv
import json
from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path

from .baseline import evaluate, feature_rows, fit_rule
from .dataset import build_examples
from .profile import FEATURES, Profile
from .simulator import simulate
from .train import calibrate, evaluate_scores, fit, label_counts, scored_runs, show_results

MEAN_HOLD_MS = 1_100
FIT_SEEDS = tuple(range(300, 306))
VALIDATION_SEEDS = (306, 307)
TEST_SEEDS = tuple(range(400, 404))


def service_runs(seeds: tuple[int, ...]):
    runs = (simulate(seed, mean_hold_ms=MEAN_HOLD_MS) for seed in seeds)
    return [(run, build_examples(run)) for run in runs]


def write_packet(path: Path, training) -> None:
    counts = label_counts(training)
    with path.open("w", encoding="utf-8", newline="") as packet:
        packet.write(
            "Propose coefficients for pinkProcess's three-feature logistic regression.\n"
            "The CSV contains labeled training examples from a synthetic service.\n"
            "A label of 1 means an arriving request queued for a connection in the next 10s.\n"
            "Each row represents count identical examples with that feature vector and label.\n"
            "Estimate coefficients from these rows, weighting each row by count.\n"
            "Use the feature column order below and the raw, unscaled feature values.\n"
            "Return only a JSON object with a weights array of three finite numbers "
            "and a finite intercept. The score is intercept plus the weighted sum.\n"
            "The alert threshold will be calibrated locally.\n\n"
        )
        writer = csv.writer(packet, lineterminator="\n")
        writer.writerow((*FEATURES, "label", "count"))
        writer.writerows((*values, int(label), count)
                         for (values, label), count in sorted(counts.items()))


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--astra", type=Path, help="JSON containing Astra's weights and intercept")
    args = parser.parse_args()
    packet = args.output_dir / "astra-training.txt"
    conventional_path = args.output_dir / "conventional.json"
    astra_path = args.output_dir / "astra-profile.json"
    profiles = [("Conventional", conventional_path, None)]
    if args.astra is not None:
        if args.astra.resolve() in {path.resolve() for path in (packet, conventional_path, astra_path)}:
            parser.error("--astra must be a separate input file from the generated outputs.")
        try:
            proposal = Profile.from_coefficients(json.loads(args.astra.read_text(encoding="utf-8")))
        except (OSError, ValueError, OverflowError) as exc:
            parser.error(f"Could not load Astra's coefficients: {exc}")
        profiles.append(("Astra input", astra_path, proposal))

    training = service_runs(FIT_SEEDS)
    validation = service_runs(VALIDATION_SEEDS)
    test = service_runs(TEST_SEEDS)
    baseline = fit_rule([(run, feature_rows(run, examples)) for run, examples in training])
    validation_budget = evaluate(
        baseline, [(run, feature_rows(run, examples)) for run, examples in validation]
    ).false_alerts
    hours = sum(run.duration_ms for run, _ in test) / 3_600_000
    print(f"Second service: average hold {MEAN_HOLD_MS} ms, validation budget {validation_budget} alerts")
    show_results("Utilization baseline",
                 evaluate(baseline, [(run, feature_rows(run, examples)) for run, examples in test]),
                 hours)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_packet(packet, training)
    print(f"Training packet: {packet}")
    for name, path, coefficients in profiles:
        profile = coefficients if coefficients is not None else fit(training)
        try:
            threshold = calibrate(scored_runs(profile, validation), validation_budget)
        except ValueError as exc:
            parser.error(f"Could not calibrate {name}: {exc}")
        profile = replace(profile, threshold=threshold)
        profile.save(path)
        loaded = Profile.load(path)
        test_scores = scored_runs(loaded, test)
        if loaded != profile or test_scores != scored_runs(profile, test):
            raise RuntimeError(f"Reload changed parameters or scores for {path}.")
        show_results(name, evaluate_scores(test_scores, loaded.threshold), hours)
        print(f"Saved {path} and verified all {sum(len(rows) for _, rows in test_scores):,} scores.")


if __name__ == "__main__":
    main()
