"""Run the frozen telemetry questions on the W8A8 CPU reference."""
import argparse
import json
from pathlib import Path
from time import perf_counter

from lm_reference.export_flan import file_sha
from lm_reference.flan import Flan
from lm_reference.flan_prompt import telemetry_prompt


# These inputs are fixed independently of the training example generator.
INCIDENTS = {
    "partial": {"origin": "live", "state": "ready", "capacity": 29,
                "latest": {"time_ms": 203900, "busy": 13, "waiting": 2},
                "observations": {"peak_waiting": {"time_ms": 195300, "waiting": 37},
                                 "first": {"waiting": 11}, "waiting": {"mean": 8.75},
                                 "covered_ms": 17300, "complete": False}},
    "recovered": {"origin": "demo", "state": "ready", "capacity": 9,
                  "latest": {"time_ms": 80300, "busy": 3, "waiting": 0},
                  "observations": {"peak_waiting": {"time_ms": 70900, "waiting": 63},
                                   "first": {"waiting": 15}, "waiting": {"mean": 11.625},
                                   "covered_ms": 30000, "complete": True},
                  "forecast_interpretation": "The FPGA issued no queue-onset warning for the next ten seconds."},
    "rising": {"origin": "live", "state": "ready", "capacity": 19,
               "latest": {"time_ms": 31000, "busy": 18, "waiting": 27},
               "observations": {"peak_waiting": {"time_ms": 31000, "waiting": 27},
                                "first": {"waiting": 3}, "waiting": {"mean": 13.6},
                                "covered_ms": 18000, "complete": True},
               "forecast_interpretation": "The FPGA predicts the queue will remain nonempty over the next ten seconds. It expects more than the current 27 waiting at the end of that period."},
}

CASES = [
    ("partial", "How many requests were waiting on average across the available samples?", ["8.75"]),
    ("partial", "How crowded did the queue get at its worst, and how far in the past?", ["37", "8.6"]),
    ("partial", "Can you tell me the pool size?", ["29"]),
    ("partial", "What share of those connections are occupied at this instant?", ["13", "29"]),
    ("partial", "How much continuous data do we actually have?", ["17.3", "partial"]),
    ("partial", "Do you see a prediction on this snapshot?", ["unavailable"]),
    ("recovered", "What can you tell the on-call engineer about the incident?", ["demo", "63", "3 of 9", "no queue-onset warning", "hold times"]),
    ("rising", "How does the backlog now compare to its starting level?", ["3", "27"]),
    ("rising", "What does the predictor say about the end of its horizon?", ["more than", "27", "ten seconds"]),
    ("rising", "Was this caused by the orders endpoint holding connections too long?", ["do not identify"]),
]


def run(model_dir):
    model = Flan(model_dir, None, reference=True)
    results = []
    for incident, question, expected in CASES:
        evidence = INCIDENTS[incident]
        start = perf_counter()
        answer = model.generate(telemetry_prompt(question, evidence))
        missing = [phrase for phrase in expected if phrase.lower() not in answer.lower()]
        results.append(dict(incident=incident, question=question, answer=answer,
                            expected_phrases=expected, missing=missing, elapsed_s=perf_counter() - start))
    return {"execution": "Mac NumPy W8A8 reference", "manifest_sha256": file_sha(model_dir / "model.json"),
            "incidents": INCIDENTS, "cases": results}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("results", type=Path)
    args = p.parse_args()
    args.results.write_text(json.dumps(run(args.model), indent=2) + "\n")
