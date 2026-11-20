"""Replay a saved profile with integer arithmetic and export FPGA test vectors."""

import json
from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from .author import TEST_SEEDS, service_runs
from .dataset import Example
from .profile import FEATURES, TREND_MS, Profile, feature_ratios, profile_features
from .train import evaluate_scores, show_results

FEATURE_SCALE = 256
SCORE_SCALE = FEATURE_SCALE ** 2
FEATURE_BOUNDS = (
    (0, FEATURE_SCALE),
    (-FEATURE_SCALE * 1000 // TREND_MS, FEATURE_SCALE * 1000 // TREND_MS),
    (0, FEATURE_SCALE),
)


def encode_ratio(numerator: int, denominator: int) -> int:
    """Scale an exact fraction and round to nearest, with ties to even."""
    lower, remainder = divmod(numerator * FEATURE_SCALE, denominator)
    halfway = 2 * remainder
    return lower + int(halfway > denominator or (halfway == denominator and lower % 2 == 1))


def encode_features(example: Example, capacity: int) -> tuple[int, int, int]:
    utilization, trend, full = feature_ratios(example, capacity)
    return encode_ratio(*utilization), encode_ratio(*trend), encode_ratio(*full)


def signed_bits(low: int, high: int) -> int:
    """Width of a two's-complement integer that covers the given interval."""
    return 1 + max((~low).bit_length() if low < 0 else 0, max(0, high).bit_length())


@dataclass(frozen=True)
class FixedProfile:
    weights: tuple[int, int, int]
    intercept: int
    threshold: int

    @classmethod
    def quantize(cls, profile: Profile) -> "FixedProfile":
        w1, w2, w3 = profile.weights
        return cls(
            (round(w1 * FEATURE_SCALE), round(w2 * FEATURE_SCALE), round(w3 * FEATURE_SCALE)),
            round(profile.intercept * SCORE_SCALE),
            round(profile.threshold * SCORE_SCALE),
        )

    def score(self, values: tuple[int, int, int]) -> int:
        return self.intercept + sum(weight * value for weight, value in zip(self.weights, values))

    def numeric_limits(self) -> dict:
        products = [sorted((weight * low, weight * high))
                    for weight, (low, high) in zip(self.weights, FEATURE_BOUNDS)]
        lower = self.intercept + sum(low for low, _ in products)
        upper = self.intercept + sum(high for _, high in products)
        # Every term's interval contains zero, so these bounds cover partial sums too.
        return {
            "feature_bounds": FEATURE_BOUNDS,
            "feature_signed_bits": [signed_bits(low, high) for low, high in FEATURE_BOUNDS],
            "weight_signed_bits": [signed_bits(weight, weight) for weight in self.weights],
            "product_signed_bits": [signed_bits(low, high) for low, high in products],
            "accumulator_bounds": [lower, upper],
            "accumulator_signed_bits": signed_bits(lower, upper),
            "comparison_signed_bits": signed_bits(min(lower, self.threshold), max(upper, self.threshold)),
        }


def replay(profile: Profile, fixed: FixedProfile):
    runs = service_runs(TEST_SEEDS)
    floating_runs, integer_runs = [], []
    flips = Counter()
    examples_by_decision = {False: [], True: []}
    max_error = 0.0
    total = 0
    for run, examples in runs:
        floating_scores, integer_scores = [], []
        for example in examples:
            raw_score = profile.score(profile_features(example, run.capacity))
            encoded = encode_features(example, run.capacity)
            score = fixed.score(encoded)
            was_alert = raw_score >= profile.threshold
            is_alert = score >= fixed.threshold
            if was_alert != is_alert:
                flips[is_alert] += 1
            max_error = max(max_error, abs(score / SCORE_SCALE - raw_score))
            examples_by_decision[is_alert].append({
                "run_seed": run.seed,
                "time_ms": example.time_ms,
                "features": encoded,
                "expected_score": score,
                "expected_alert": is_alert,
                "float_alert": was_alert,
            })
            floating_scores.append((example.time_ms, raw_score))
            integer_scores.append((example.time_ms, score))
        floating_runs.append((run, floating_scores))
        integer_runs.append((run, integer_scores))
        total += len(examples)

    hours = sum(run.duration_ms for run, _ in runs) / 3_600_000
    print(f"Replayed {total:,} held-out windows. Maximum absolute score error: {max_error:.6f}")
    print(f"Prediction flips: {sum(flips.values())} "
          f"({flips[True]} no-wait to wait, {flips[False]} wait to no-wait)")
    show_results("Floating point", evaluate_scores(floating_runs, profile.threshold), hours)
    show_results("Fixed point", evaluate_scores(integer_runs, fixed.threshold), hours)

    vectors = []
    for decision, rows in examples_by_decision.items():
        rows.sort(key=lambda row: abs(row["expected_score"] - fixed.threshold))
        unique = {}
        for row in rows:
            unique.setdefault((row["features"], row["float_alert"]), row)
        nearest = list(unique.values())[:3]
        changed = [row for row in unique.values() if row["float_alert"] != decision][:3]
        vectors.extend(nearest + [row for row in changed if row not in nearest])

    corners = sorted(product(*FEATURE_BOUNDS), key=fixed.score)
    for case, encoded in (("accumulator_min", corners[0]), ("accumulator_max", corners[-1])):
        score = fixed.score(encoded)
        vectors.append({
            "case": case,
            "features": encoded,
            "expected_score": score,
            "expected_alert": score >= fixed.threshold,
        })
    return vectors


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=Path("profiles/service-b-astra-v1.json"))
    parser.add_argument("--vectors", type=Path, help="Where to write the FPGA test vectors")
    args = parser.parse_args()
    if args.vectors and args.vectors.resolve() == args.profile.resolve():
        parser.error("--vectors must be a different file from --profile.")

    profile = Profile.load(args.profile)
    fixed = FixedProfile.quantize(profile)
    limits = fixed.numeric_limits()
    print(f"Integer weights {fixed.weights}, intercept {fixed.intercept}, threshold {fixed.threshold}")
    print(f"Accumulator bounds {limits['accumulator_bounds']} "
          f"require {limits['accumulator_signed_bits']} signed bits.")
    vectors = replay(profile, fixed)
    if args.vectors:
        artifact = {
            "schema_version": 1,
            "feature_order": FEATURES,
            "rounding": "nearest_ties_to_even",
            "feature_scale": FEATURE_SCALE,
            "weight_scale": FEATURE_SCALE,
            "score_scale": SCORE_SCALE,
            "weights": fixed.weights,
            "intercept": fixed.intercept,
            "threshold": fixed.threshold,
            "comparison": "score >= threshold",
            "numeric_limits": limits,
            "vectors": vectors,
        }
        args.vectors.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {len(vectors)} FPGA test vectors to {args.vectors}.")


if __name__ == "__main__":
    main()
