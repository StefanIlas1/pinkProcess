"""Count current-overlay score submissions for a recorded physical diagnosis."""

import argparse
import json
from pathlib import Path

import numpy as np

from lm_reference.fabric import ROWS, WIDTH
from lm_reference.flan import Flan
from lm_reference.flan_prompt import telemetry_prompt
from pinkprocess.kit_evidence import analyst_evidence


class CountingFabric:
    def __init__(self):
        self.submissions = 0

    def matmat_array(self, weights, activations, rows):
        batch, columns = activations.shape
        if activations.dtype != np.int8 or columns % WIDTH:
            raise ValueError("Expected full-width W8A8 activation blocks.")
        # The physical driver skips all-zero 64-byte activation vectors.
        blocks = activations.reshape(batch, -1, WIDTH)
        active_blocks = int(np.any(blocks, axis=2).sum())
        tile_rows = ROWS - 3  # Three rows reserved for live telemetry.
        self.submissions += active_blocks * ((rows + tile_rows - 1) // tile_rows)
        return activations.astype(np.float32) @ np.asarray(weights).astype(np.float32).T


class CountedFlan(Flan):
    def linear(self, name, x):
        output = super().linear(name, x)
        if name == "lm_head":
            self.steps.append(self.fabric.submissions)
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--case", default="ambiguous-cause")
    args = parser.parse_args()
    report = json.loads(args.report.read_bytes())
    case = next(c for c in report["cases"] if c["case"] == args.case)
    prompt = telemetry_prompt(report["question"], analyst_evidence(case["started"]["snapshot"]))
    runtime = CountedFlan(args.model, CountingFabric())
    runtime.steps = []
    answer = runtime.generate(prompt)
    if answer != case["result"]["answer"] or runtime.fabric_projections != case["result"]["fabric_projections"]:
        raise ValueError("Replay diverged from the recorded physical answer or projection count.")
    # classifier.v performs FETCH and ACCUMULATE for every four weights of all 512 rows.
    seconds_per_submission = ROWS * (WIDTH // 4) * 2 / 50_000_000
    print(json.dumps({
        "case": args.case, "capture_id": case["result"]["capture_id"],
        "input_tokens": len(runtime.tokenizer.encode(prompt)), "decoder_steps_including_eos": len(runtime.steps),
        "first_token_submissions": runtime.steps[0], "total_submissions": runtime.steps[-1],
        "first_token_compute_floor_s": round(runtime.steps[0] * seconds_per_submission, 3),
        "full_answer_compute_floor_s": round(runtime.steps[-1] * seconds_per_submission, 3),
        "physical_first_observed_text_s": case.get("first_partial_seconds"),
        "physical_complete_answer_s": case.get("question_to_answer_seconds"),
    }, indent=2))


if __name__ == "__main__":
    main()
