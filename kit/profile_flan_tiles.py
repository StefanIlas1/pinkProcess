"""Count classifier submissions and projection-array MACs for a recorded physical diagnosis."""

import argparse
import json
from pathlib import Path

import numpy as np

from lm_reference.fabric import ROWS, WIDTH
from lm_reference.flan import Flan
from lm_reference.flan_prompt import telemetry_prompt
from lm_reference.projection_overlay import MAX_SEGMENT_BUFFER_BYTES
from pinkprocess.kit_evidence import analyst_evidence


class CountingFabric:
    def __init__(self):
        self.submissions = 0
        self.array_macs = 0
        self.weight_reads = 0
        self.activation_reads = 0
        self.cached_activation_reads = 0

    def matmat_array(self, weights, activations, rows):
        batch, columns = activations.shape
        if activations.dtype != np.int8 or columns % WIDTH:
            raise ValueError("Expected full-width W8A8 activation blocks.")
        # Each tile uses all 64 lanes, including padding in the final tile.
        position_lanes, output_lanes = (1, 64) if batch == 1 else (8, 8)
        position_tiles = (batch + position_lanes - 1) // position_lanes
        output_tiles = (rows + output_lanes - 1) // output_lanes
        self.array_macs += position_tiles * output_tiles * 64 * columns
        # Each output tile rereads its weights for each group of positions.
        self.weight_reads += position_tiles * output_tiles * columns * output_lanes
        tile_activation_bytes = position_tiles * columns * 8
        self.activation_reads += output_tiles * tile_activation_bytes
        # The new fabric retains one activation tile until the next descriptor.
        tiles_per_segment = min(MAX_SEGMENT_BUFFER_BYTES // (columns * output_lanes),
                                MAX_SEGMENT_BUFFER_BYTES // (position_tiles * 64 * 4), 65535)
        segments = (output_tiles + tiles_per_segment - 1) // tiles_per_segment
        self.cached_activation_reads += tile_activation_bytes * (segments if columns <= 1024 else output_tiles)
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
            self.steps.append((self.fabric.submissions, self.fabric.array_macs))
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
    first_submissions, first_macs = runtime.steps[0]
    total_submissions, total_macs = runtime.steps[-1]
    peak_macs_per_second = 64 * 50_000_000
    print(json.dumps({
        "case": args.case, "capture_id": case["result"]["capture_id"],
        "input_tokens": len(runtime.tokenizer.encode(prompt)), "decoder_steps_including_eos": len(runtime.steps),
        "first_token_submissions": first_submissions, "total_submissions": total_submissions,
        "first_token_compute_floor_s": round(first_submissions * seconds_per_submission, 3),
        "full_answer_compute_floor_s": round(total_submissions * seconds_per_submission, 3),
        "projection_array": {
            "clock_hz": 50_000_000, "mac_lanes": 64,
            "first_token_padded_macs": first_macs, "full_answer_padded_macs": total_macs,
            "first_token_mac_only_floor_s": first_macs / peak_macs_per_second,
            "full_answer_mac_only_floor_s": total_macs / peak_macs_per_second,
        },
        "projection_ddr_reads": {
            "scope": "Projection engine only, without the optional vocabulary reducer. Each AXI read beat carries 8 bytes.",
            "weight_bytes": runtime.fabric.weight_reads,
            "activation_bytes_without_block_ram_reuse": runtime.fabric.activation_reads,
            "activation_bytes_with_block_ram_reuse": runtime.fabric.cached_activation_reads,
            "cached_read_only_floor_s_at_50mhz": (
                runtime.fabric.weight_reads + runtime.fabric.cached_activation_reads) / (8 * 50_000_000),
        },
        "physical_first_observed_text_s": case.get("first_partial_seconds"),
        "physical_complete_answer_s": case.get("question_to_answer_seconds"),
    }, indent=2))


if __name__ == "__main__":
    main()
