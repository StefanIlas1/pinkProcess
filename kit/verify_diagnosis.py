"""Evaluate a frozen W8A8 diagnosis model on seeded, independently generated incidents."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from random import Random
import re
from time import perf_counter

from lm_reference.flan_prompt import telemetry_prompt
from lm_reference.model_assets import file_digest, verify_model
from pinkprocess.kit_evidence import analyst_evidence


QUESTIONS = (
    "Should I keep this page open, or has the trouble passed? Give the readings that led you there, "
    "say what the predictor expects separately, and explain what you would measure next to narrow down the cause.",
    "For the handoff, tell me whether callers are still stuck. What did the pool actually show, "
    "what is only expected over the next ten seconds, and what new measurement would help explain the incident?",
    "Is this still affecting requests? State what in the measurements supports your answer, "
    "distinguish what could happen next, and say what you'd collect before naming a cause.",
    "Can the on-call engineer close this incident based on the queue? Include current and worst readings, "
    "what the kit predicts, and what evidence you still need to say why it happened.",
    "Give me a short incident update. Explain whether the queue recovered using the observed numbers, "
    "identify the prediction as such, and tell me what to measure to investigate the cause.",
    "What would you report to the next shift about whether the backlog is still present? "
    "Use the actual readings, distinguish the forecast, and suggest a measurement that could narrow the cause.",
)


def generate_cases(seed):
    rng = Random(seed)
    profiles = [(cleared, variant) for cleared in (True, False) for variant in range(len(QUESTIONS))]
    rng.shuffle(profiles)
    for cleared, variant in profiles:
        capacity = rng.randint(12, 43)
        busy = capacity - rng.randint(3, 9) if cleared else capacity
        waiting = 0 if cleared else rng.randint(3, 15)
        peak_waiting = waiting + rng.randint(16, 49)
        current_ms = rng.randrange(80_000, 240_000, 100)
        peak_age = rng.randrange(5_500, 26_000, 100)
        first = {"time_ms": current_ms - 30_000, "busy": busy, "waiting": rng.randint(1, 4)}
        latest = {"time_ms": current_ms, "busy": busy, "waiting": waiting}
        peak = {"time_ms": current_ms - peak_age, "busy": capacity, "waiting": peak_waiting}
        if cleared:
            warn = variant % 2 == 0
            forecast = {"model": "pool_wait_logistic", "version": "synthetic-onset", "alert": warn,
                        "time_ms": current_ms, "horizon_ms": 10_000}
            forecast_key = "onset_warning" if warn else "no_onset_warning"
        else:
            persists = variant % 3 != 0
            grows = variant % 3 == 1
            forecast = {"model": "queue_outlook", "version": "synthetic-outlook",
                        "scores": [7 if persists else -9, 11 if grows else -13],
                        "time_ms": current_ms, "horizon_ms": 10_000}
            forecast_key = "persist" if persists else "drain"
        snapshot = {
            "origin": "demo", "state": "ready", "source": "Independent synthetic pool incident",
            "fresh": True, "capacity": capacity, "latest": latest, "forecast": forecast,
            "forecast_reading": latest.copy(),
            "observations": {
                "first": first, "latest": latest.copy(), "peak_waiting": peak,
                "start_ms": first["time_ms"], "end_ms": current_ms,
                "sample_count": 301, "covered_ms": 30_000, "complete": True,
            },
        }
        yield {
            "id": sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()[:12],
            "question": QUESTIONS[variant],
            "snapshot": snapshot,
            "expected": {
                "now": "cleared" if cleared else "backlog persists", "current_waiting": waiting,
                "peak_waiting": peak_waiting, "forecast": forecast_key,
            },
        }


def search(pattern, text):
    return re.search(pattern, text, re.IGNORECASE) is not None


def grade(answer, expected):
    """Conservative lexical checks. Raw answers remain available for adjudication."""
    waiting = expected["current_waiting"]
    peak = expected["peak_waiting"]
    empty = r"\b(?:0|zero|no)\s+(?:(?:requests?|callers?)\s+(?:are\s+)?)?waiting\b|\b(?:queue|backlog)\s+(?:is\s+)?empty\b"
    cleared = r"\b(?:clear(?:ed)?|recover(?:ed|y)?|resolved|drained|subsided)\b"
    present_count = rf"\b{waiting}\s+(?:requests?\s+)?(?:are\s+)?waiting\b|\b(?:queue|backlog)\s+(?:is\s+)?(?:at\s+)?{waiting}\b"
    still = r"\b(?:still|remains?|ongoing|persist(?:s|ing|ent)?|currently|now)\b"
    peak_fact = rf"\b(?:peak(?:ed)?|worst|maximum)\b[^.!?]{{0,34}}\b{peak}\b|\b{peak}\b[^.!?]{{0,34}}\b(?:peak|worst|maximum)\b"
    future = r"\b(?:forecast\w*|predict\w*|expect\w*|warning)\b"
    forecasts = {
        "onset_warning": r"\b(?:warn\w*|predict\w*)\b[^.!?]{0,70}\b(?:start\w*|return\w*|new|onset)\b",
        "no_onset_warning": r"\b(?:no|without)\b[^.!?]{0,32}\bwarn\w*\b",
        "persist": r"\b(?:remain\w*|stay\w*|persist\w*|continue\w*)\b[^.!?]{0,55}\b(?:non-?empty|backlog|queue|requests?|waiting|positive)\b|\bqueue\b[^.!?]{0,28}\b(?:persist\w*|continue\w*)\b",
        "drain": r"\b(?:drain\w*|empt(?:y|ies)|clear\w*)\b",
    }
    observed = search(empty, answer) if waiting == 0 else search(present_count, answer)
    current = (search(cleared, answer) and observed if waiting == 0 else
               observed and search(still, answer))
    # Require the forecast signal and its direction within the same sentence
    # so a statement about the current queue cannot count as a prediction.
    forecast = any(search(future, part) and search(forecasts[expected["forecast"]], part)
                   for part in re.split(r"[.!?]+", answer))
    uncertain = search(
        r"(?:\b(?:cause|endpoint|latency|reason)\b[^.!?]{0,42}\b(?:unknown|unclear|unavailable|"
        r"cannot|can't|not known|not identify|not enough)\b|\b(?:cannot|can't|does not|do not|"
        r"doesn't|insufficient|not enough)\b[^.!?]{0,58}\b(?:cause|endpoint|latency|reason)\b)", answer
    )
    next_measurement = search(
        r"\b(?:measure|collect|check|track|instrument|record|capture|monitor|sample|look at)\b"
        r"[^.!?]{0,60}\b(?:hold times?|acquisition (?:time|latency|wait)|endpoints?|request (?:latency|"
        r"rates?|arrivals?|timings?)|query (?:latency|times?|duration)|wait times?|traces?)\b", answer
    )
    return {
        "current_state": current,
        "cites_current_and_peak": observed and search(peak_fact, answer),
        "forecast_separated_and_correct": forecast,
        "uncertainty_about_cause": uncertain,
        "concrete_next_measurement": next_measurement,
    }


def evaluate(model_dir, manifest_sha256, seed):
    verify_model(model_dir, manifest_sha256)
    from lm_reference import flan, flan_prompt
    from pinkprocess import kit_evidence

    manifest = json.loads((model_dir / "model.json").read_bytes())
    frozen = {
        "manifest_sha256": manifest_sha256, "weights_sha256": manifest["weights_sha256"],
        "tokenizer_sha256": manifest["tokenizer_sha256"],
        "runtime_sha256": file_digest(Path(flan.__file__)),
        "prompt_sha256": file_digest(Path(flan_prompt.__file__)),
        "evidence_adapter_sha256": file_digest(Path(kit_evidence.__file__)),
    }
    runtime = flan.Flan(model_dir, None, reference=True)
    checks_passed = Counter()
    cases = []
    for case in generate_cases(seed):
        prompt = telemetry_prompt(case["question"], analyst_evidence(case["snapshot"]))
        start = perf_counter()
        answer = runtime.generate(prompt)
        checks = grade(answer, case["expected"])
        checks_passed.update({name: int(passed) for name, passed in checks.items()})
        cases.append({**case, "prompt": prompt, "answer": answer, "checks": checks,
                      "seconds": round(perf_counter() - start, 4)})
    return {
        "execution": "W8A8 software reference",
        "frozen_candidate": frozen,
        "holdout_seed": seed,
        "policy": "All five checks must pass on each incident. Lexical scoring requires raw-answer review.",
        "total_cases": len(cases),
        "cases_passing_every_check": sum(all(case["checks"].values()) for case in cases),
        "checks_passing_by_criterion": dict(checks_passed),
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--seed", type=int, required=True, help="Draw after freezing the candidate.")
    args = parser.parse_args()
    if args.results.exists():
        parser.error("Choose a new evidence path to preserve the original result.")
    result = evaluate(args.model, args.manifest_sha256, args.seed)
    with args.results.open("x") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))
    return 0 if result["cases_passing_every_check"] == result["total_cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
