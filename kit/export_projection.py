"""Export the first encoder and decoder query projections from a verified diagnosis replay."""

import argparse
import json
from pathlib import Path

import numpy as np

from lm_reference.flan import Flan
from lm_reference.flan_prompt import telemetry_prompt
from lm_reference.model_assets import file_digest, verify_model
from pinkprocess.kit_evidence import analyst_evidence


class CaptureFabric:
    def matmat_array(self, weights, activations, rows):
        matrix = np.asarray(weights)
        if self.capture is not None:
            self.capture.update(activations=activations.copy(), weights=matrix.copy())
        return activations.astype(np.float32) @ matrix.astype(np.float32).T


class CaptureFlan(Flan):
    def linear(self, name, x):
        self.fabric.capture = None
        for side in ("encoder", "decoder"):
            if name == f"{side}.block.0.layer.0.SelfAttention.q" and side not in self.samples:
                self.samples[side] = {}
                self.fabric.capture = self.samples[side]
        return super().linear(name, x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--case", default="ambiguous-cause")
    args = parser.parse_args()
    shipped_manifest = Path(__file__).resolve().parents[1] / "profiles/flan-telemetry-diagnostic/model.json"
    verify_model(args.model, file_digest(shipped_manifest))
    report = json.loads(args.report.read_bytes())
    case = next(c for c in report["cases"] if c["case"] == args.case)
    result = case["result"]
    if result["active"]["checkpoint_sha256"] != file_digest(args.model / "weights.bin"):
        raise ValueError("Capture used different weights.")
    prompt = telemetry_prompt(report["question"], analyst_evidence(case["started"]["snapshot"]))
    runtime = CaptureFlan(args.model, CaptureFabric())
    runtime.samples = {}
    if runtime.generate(prompt) != result["answer"] or runtime.fabric_projections != result["fabric_projections"]:
        raise ValueError("Software replay diverged from the physical run.")
    args.output.mkdir(parents=True, exist_ok=True)
    for side, arrays in runtime.samples.items():
        path = args.output / f"projection-{side}.npz"
        np.savez(path, capture_id=result["capture_id"], **arrays)
        print(path, arrays["activations"].shape, arrays["weights"].shape)


if __name__ == "__main__":
    main()
