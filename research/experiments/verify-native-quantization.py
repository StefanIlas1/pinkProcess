from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter_ns
from types import SimpleNamespace

import numpy as np

from lm_reference.flan import Flan
from lm_reference.flan_prompt import telemetry_prompt
from lm_reference.projection_native import NativeProjection, _load_library
from pinkprocess.kit_evidence import analyst_evidence

proof = json.loads(Path("build/live-kit/language-physical.json").read_bytes())
replay = json.loads(Path("build/flan/three-diagnosis-traces-reproduction.json").read_bytes())
saved = next(c for c in replay["cases"] if c["case"] == "ambiguous-cause")
edit = json.loads(Path("kit/evidence/physical-diagnosis-edit.json").read_bytes())
edit_prompt = telemetry_prompt(proof["question"], analyst_evidence(proof["baseline"]["started"]["snapshot"]))
base = proof["baseline"]["started"]
record = dict(question=proof["question"], captured_at_ms=base["captured_at_ms"],
              snapshot=base["snapshot"], prompt=edit_prompt)
assert sha256(json.dumps(record, allow_nan=False, sort_keys=True,
                         separators=(",", ":")).encode()).hexdigest() == edit["capture_id"]
native = NativeProjection(None, _load_library(Path("build/projection_native.so").resolve()))

class ComparedQuantizer:
    scope = "query"
    argmax_mmio = None

    def __init__(self):
        self.native = self

    @staticmethod
    def bind_model(_): pass

    @staticmethod
    def invalidate_weights(_): pass

    @staticmethod
    def profile(): return {}

    def start_generation(self):
        self.counts = Counter()
        self.sizes = Counter()

    def quantize_activations(self, x):
        start = perf_counter_ns()
        actual = native.quantize_activations(x)
        self.counts["native_ns"] += perf_counter_ns() - start
        start = perf_counter_ns()
        expected = Flan.quantize_activations(x)
        self.counts["numpy_ns"] += perf_counter_ns() - start
        self.counts["calls"] += 1
        if actual is None:
            self.counts["fallback"] += 1
            self.sizes[("fallback",x.shape)] += 1
            return expected
        self.sizes[("native",x.shape)] += 1
        self.counts["float_values"] += x.size
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1].view(np.uint32), expected[1].view(np.uint32))
        return actual

runner = ComparedQuantizer()
model = Flan("build/flan/diagnostic-install", None, reference=True, query_projection=runner)
head = model.t["lm_head.weight"]
original = sha256(head).hexdigest()
assert original == edit["original"]["classifier_sha256"], original
row = edit["selected"]["target_row"]
assert row == proof["astra"]["response"]["target_row"]
payload = bytes(v & 255 for v in proof["astra"]["response"]["row"])
report = {}
for name, prompt, expected in (
    ("saved", saved["prompt"], saved["answer"]),
    ("edited", edit_prompt, edit["edited_answer"]),
    ("rollback", edit_prompt, edit["original_answer"]),
):
    if name == "edited":
        model.install_weight_version(SimpleNamespace(
            patch=SimpleNamespace(row=row, payload=payload),
            classifier_sha256=edit["selected"]["classifier_sha256"],
        ))
    elif name == "rollback":
        model.install_weight_version(SimpleNamespace(patch=None, classifier_sha256=original))
    answer = model.generate(prompt)
    runner.native = None
    timings = runner.counts.copy()
    shapes = runner.sizes.copy()
    numpy_answer = model.generate(prompt)
    runner.native = runner
    assert answer == numpy_answer, (name, answer, numpy_answer)
    if name != "edited":
        assert answer == expected, (name, answer, expected)
    if name == "rollback":
        calibration_matches = model.first_output_activation == proof["astra"]["calibration"]["activation"]
    else:
        calibration_matches = None
    report[name] = dict(answer_sha256=sha256(answer.encode()).hexdigest(),
                        head_sha256=sha256(head).hexdigest(), exact_local_numpy=True,
                        exact_archived_board_answer=answer == expected,
                        rollback_calibration_matches_board=calibration_matches,
                        **timings, observed_shapes={str(k):v for k,v in shapes.items()})
    print(name, report[name], flush=True)
Path("build/native-quantization-replay.json").write_text(json.dumps(report, indent=2) + "\n")
