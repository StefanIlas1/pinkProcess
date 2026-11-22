"""Reproduce the 360-step queue-history diagnostic adapter from its floating-point parent."""

import argparse
from functools import partial
from pathlib import Path

from kit.reference_forecast import make_forecaster
from lm_reference import adapt_flan
from lm_reference.flan_prompt import measured_history, telemetry_prompt
from pinkprocess.kit_dataset import dataset_snapshot
from pinkprocess.kit_evidence import analyst_evidence


def development_evidence(rng, forecast):
    """Draw one synthetic pool trace. Draw order is part of the published recipe."""
    sample_count = rng.randint(50, 190)
    capacity = rng.randint(1, 40)
    height = rng.randint(2, 32)
    floor = rng.randint(1, 8)
    onset = rng.randint(0, sample_count // 3)
    peak_at = rng.randint(sample_count // 2, 2 * sample_count // 3)
    mode = rng.choice(("queue_free", "pulse", "rising", "steady", "falling"))
    start_ms = rng.choice((0, 0, 10000))
    rows = []
    for i in range(sample_count):
        if mode == "queue_free":
            waiting = 0
        elif mode == "pulse":
            waiting = max(0, height - abs(i - peak_at))
        elif mode == "rising":
            waiting = max(0, (i - onset) * height // (sample_count - onset))
        elif mode == "steady":
            waiting = floor
        else:
            waiting = max(floor, height - i * height // sample_count)
        busy = capacity if waiting else rng.randrange(capacity + 1)
        rows.append(f"{start_ms + i * 100},{busy},{waiting}\n")

    csv = "time_ms,busy,waiting\n" + "".join(rows)
    evidence = analyst_evidence(dataset_snapshot("dev", "dev", csv, capacity, forecast))
    evidence["origin"] = rng.choice(("dataset", "dataset", "live", "demo"))
    return evidence


def target_answer(task, evidence):
    observations = evidence["observations"]
    peak = observations["peak_waiting"]
    latest = evidence["latest"]
    forecast = evidence["forecast_interpretation"]
    if task == 0:
        primary_forecast = forecast.split(". ")[0].rstrip(".")
        return (f"{measured_history(evidence)} The peak was {peak['waiting']}. "
                f"{primary_forecast}. Cause unknown. Measure arrivals and connection hold times.")
    if task == 1:
        return f"{latest['waiting']} requests are waiting."
    if task == 2:
        seconds_ago = (latest["time_ms"] - peak["time_ms"]) / 1000
        return f"The queue peaked at {peak['waiting']} requests {seconds_ago:g} seconds ago."
    if task == 3:
        return f"The mean queue was {observations['waiting']['mean']:.4g} requests."
    if task == 4:
        return f"{latest['busy']} of {evidence['capacity']} connections are busy."
    if task == 5:
        return "The measurements do not identify endpoints, latency or root cause."
    if task == 6:
        return forecast
    if task == 7:
        return f"The pool capacity is {evidence['capacity']} connections."
    if task == 8:
        return f"Waiting went from {observations['first']['waiting']} to {latest['waiting']} requests."
    if task == 9:
        return f"{observations['covered_ms'] / 1000:g} seconds were recorded."
    return "The requested metric is not in the telemetry."


def example(rng, task, *, forecast, detailed, diagnostic, balanced_history):
    generate_legacy = partial(adapt_flan.example, detailed=detailed, diagnostic=diagnostic,
                              balanced_history=balanced_history, history_prompt=True)
    if rng.random() < 0.18:
        return generate_legacy(rng, task)

    evidence = development_evidence(rng, forecast)
    # The legacy generator supplied the question vocabulary in the original run.
    # Drawing it after the trace also preserves the seed's order across batches.
    legacy_prompt, _ = generate_legacy(rng, task)
    question = legacy_prompt.rsplit("\nQuestion: ", 1)[1].split("\nAnswer:", 1)[0]
    return telemetry_prompt(question, evidence, history=True), target_answer(task, evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Directory with diagnostic parent model.safetensors and config.json")
    parser.add_argument("tokenizer", type=Path, help="Pinned upstream spiece.model")
    parser.add_argument("out", type=Path, help="Destination for merged floating-point weights")
    args = parser.parse_args()
    profiles = Path(__file__).resolve().parents[1] / "profiles"
    generate = partial(example, forecast=make_forecaster(profiles))
    adapt_flan.train(args.source, args.tokenizer, args.out, steps=360, seed=4227,
                     detailed=True, diagnostic=True, balanced_history=True, example_factory=generate)


if __name__ == "__main__":
    main()
